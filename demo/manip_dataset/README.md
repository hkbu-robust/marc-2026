# Manip-trainer teleop

Two client profiles talk to the same sim (`bash simulation-platform/marc.sh manip-trainer`):

| Profile | Service | What it is for |
|---|---|---|
| **`manip-pick`** | `agent-manip-pick` | Test GraspGen-X (**G**), wrist-YOLO (**H**), or remote OpenVLA (**V**). No HDF5. |
| `tools` | `agent-manip-probe` | Record OpenVLA RLDS source + GraspGenX grasp captures. Manual grasp + **P** place. |

## GraspGen-X pick tester (no dataset)

```bash
# terminal 1
bash simulation-platform/marc.sh manip-trainer

# terminal 2
cd demo
docker compose --profile manip-pick up agent-manip-pick
```

In the trainer GUI, choose the object and **Place near** (or **Place far**, then **A** in the client). Press **G** to snapshot with near-field YOLO and run GraspGen-X pick + overhead basket. Press **H** for a wrist-YOLO pick. Press **V** to stream the base camera to zero-shot OpenVLA at `http://10.0.0.200:8003/v1`; its close action uses the simulator-calibrated `MARC_GRIP_CLOSE=0.68`. **O** cycles the class, **N** recovers to ready (nothing is saved). Press `?` for the key map.

Requires an interactive TTY. `MARC_PICK_POLICY=graspgenx` is pinned on this service.
For fast iteration it reuses the existing `marc-agent:local` image, which already
contains the local GraspGen-X package, generator/discriminator checkpoints, and
Robotiq assets under `/agent/models/runtime/graspgenx`; it does not pull anything.
Live pick/place Python is bind-mounted
(`marc_agent/`, `manip_dataset/`, `nearfield.py`, `arm_pick.py`, `pnp_params.json`);
normal Python edits therefore need no rebuild.

Run the pick tester without collector/probe containers on the same 8 GB GPU.
Isaac Sim and GraspGen-X can coexist only after those extra CUDA residents are
stopped; otherwise sampler loading reports the CUDA OOM instead of silently
falling back to a bogus IK grasp.

The platform publishes base and gripper stereo RGB, `32FC1` depth, and organized
`PointCloud2` topics. The quick path reconstructs an arm-frame object cloud from
the already-aligned base depth crop. **G** now runs this sequence: base YOLO/depth
snapshot → support-plane-reduced object cloud → deterministic camera-entry stage
→ GraspMoE pose selection → class/posture profile → Robotiq contact motion
→ immediate `0.703` close → lift → hold verification. Far TCPs, missing
depth, unknown side-grasp profiles, and unverified `holding=True` all fail closed.
The handoff stays above the observed object top. YOLO supplies the crop, while its
deprojected depth cloud supplies a robust top-down footprint: elongated objects
finalize jaw yaw on the footprint's narrow axis and execute at its centre
cross-section. The GraspGen-X candidate still has to provide a safe top-down pose,
and the measured narrow width must fit the configured aperture. World closing-axis
yaw is converted through the default wrist frame before IK (the default local +X
points toward arm-base -Y). The validated lying motion is a vertical top-down
grasp. The validated upright motion raises quickly, folds at clearance, extends
behind the object, descends only to its guarded transition, folds in place, then
makes a slow level 15 mm advance before closing. It never restores the lower
cloud-contact Z after folding. Upright profiles use a measured 30 mm left
pad-midpoint bias and accept stable centres through 0.68 m; lying profiles retain
the 0.58 m envelope.

Class routing is centralized in `grasp_profiles.py`. Mug, cola can, juice,
disposable cup, and umbrella each have copied `lying_<class>` and
`upright_<class>` profiles. Sunblock always uses `upright_sunblock`, regardless
of observed posture. Its dedicated floor-load variant keeps the gripper fully
open and rotates distal joint 5 outward to `+0.95` rad beside the chassis while
elevated, making the final gripper rod nearly parallel to the floor. It measures
the resulting pad pose, then extends and descends with that orientation locked,
advances to surround the short object, pauses, and only closes. Its auto-approach
target is 0.65 m instead of the shared 0.70 m upright target, bringing this one
profile slightly closer and keeping the rear descent away from full extension.
Tissue and pencilcase always use their lying profile.
An upright cracker box uses `upright_cracker_box`, copied from the lying-tumbler
motion. A flat cracker box resolves to `lying_cracker_box` with `flip_required`
and fails before arm movement until a physical flip-and-reacquire trajectory is
calibrated; it never falls through to a wide generic pinch.
Clearly elongated footprints use their narrow-axis yaw; near-round/upright objects
keep the default jaw axis and avoid an arbitrary wrist rotation. The saved cloud
determines width/yaw, while fresh YOLO RGB-D localization is gated independently
per axis against the saved centre. This keeps a valid lateral correction even when
staged-arm occlusion corrupts forward depth by more than 4 cm. Out-of-reach
post-stage depth remains optional input instead of aborting the saved-cloud grasp.
The GraspGen-X deterministic handoff ends directly above the saved object centre,
including the live-calibrated forward TCP correction, while the chassis actively
holds its forward world position. GraspGen-X then
sends one immediate `0.703` close with ramping and sensor early-stop disabled.
Three consecutive post-lift hold samples verify the grasp without reloading the
base-camera stack; camera-motion verification is only a fallback when hold is not
confirmed.

## Dataset collector

Keyboard-driven demonstration collector against the MARC **manip-trainer** profile.
Produces raw HDF5 episodes for OpenVLA and records the successful/failed EE grasp
pose needed to label later GraspGenX data.

## Run (dataset collector)

```bash
# terminal 1
bash simulation-platform/marc.sh manip-trainer

# terminal 2
cd demo
mkdir -p datasets weights
MARC_TEAM_ID=u1 docker compose up --build agent-manip-probe
```

The probe image is a separate Dockerfile target (`manip-probe`) with torch +
ultralytics. First `--build` is large (CUDA torch). The runtime loads
`models/runtime/stage2-nearfield-detector/model.pt`; use `MARC_YOLO_WEIGHTS` only
to point a development probe at another explicit local checkpoint. Missing or
unsupported weights fail closed; nothing downloads at runtime.

Isaac Sim already uses the GPU. `MARC_YOLO_DEVICE=cpu` is slower but avoids
sharing CUDA when debugging locally.

Requires an interactive TTY (`stdin_open` / `tty` on `agent-manip-probe`). Press `?` for the key map.

## Ego-cam YOLO dataset (approach detector)

CCTV `dataset-gen` is aerial. Auto-approach sees **Base_Camera_Left** (~640×480, near-ground),
so that detector is fine-tuned on a **separate** ego-cam set — do not mix the two image
distributions in one `yolo detect train`.

```bash
# from simulation-platform/  (stop dataset-gen / manip-trainer first; one GPU)
# EGO_SCENES = how many *new* frames this run. Existing images are never overwritten;
# numbering resumes after the highest 00000.jpg/.png already on disk (JPEG+PNG mix is fine).
HEADLESS=true EGO_SCENES=200 docker compose --profile ego-dataset-gen up
```

Writes `trainer_output/ego_yolo/{images,labels,overlays,data.yaml}` (new frames are **PNG**). Then:

```bash
yolo detect train data=trainer_output/ego_yolo/data.yaml model=yolo11x.pt imgsz=640
```

Point the probe at the resulting `best.pt` with `MARC_YOLO_WEIGHTS`. Class names match the
lost-object list except **umbrella / long_umbrella** (skipped). Every remaining class is
forced into the set at least once; extras are shuffled. The chassis is teleported in xy
and yaw each scene so boxes are not glued to one patch of the image.

## Episode flow

1. Arm moves to a **joint-space ready** pose (`MARC_READY_J1…J6`), then teleop uses EE-delta IK.
2. **MANUAL_BASE** — `i/k/j/l` place and aim the chassis; `A` starts auto-approach.
3. **AUTO_APPROACH** — when two resolutions share `base_left`, locks the **taller 480×640 chassis** and **prefers 360×640 chassis depth** when that shape is in the lock window. Then **stop-turn-go** on **YOLO + RGB-D**: ALIGN steers the deprojected object lateral coordinate onto the arm/base centerline (using the YOLO box midpoint only when depth localization is unavailable), DRIVE goes straight for a short burst, and it re-ALIGNS if the target drifts or is lost. Reaching a range or frame boundary blocks further forward motion but still permits final yaw alignment; completion requires the arm-frame centering tolerance. It does **not** crawl without a detection, and a lost box is not treated as “already at our feet.” Stop is **arm-base forward using the calibrated pitched/offset base-camera transform** (`MARC_APPROACH_STOP_M`, default 0.56 m). The final crawl stops at whichever safety boundary arrives first: the object's lower box edge entering the configured bottom-frame margin (`MARC_APPROACH_FRAME_BOTTOM_MARGIN`, default 10%) or arm-base forward reaching the hard range fallback (`MARC_APPROACH_ALREADY_NEAR`, default 0.44 m). Every class routed to a copied upright side profile uses the farther fallback (`MARC_APPROACH_UPRIGHT_ALREADY_NEAR`, default 0.70 m); lying/top-down profiles use the generic braking point. Wall-clock `OVERRUN_M / speed` is only a cap, because the practice sim often closes much slower than `cmd_vel` (`C` cancels).
   Each tick logs `err` (box vs image center), `obj=used/raw`, `name`/`conf`, `fwd` (arm-base), and `d` (bbox depth). `obj=0/0` means YOLO saw nothing; `obj=0/3` means boxes existed but were skipped (e.g. `person`).
   Inference is always restricted to the exact `MARC_TARGET_OBJECT`; an unsupported target fails closed and never falls back to all classes. RGB is passed as a PIL image. Lock dumps PNGs under `datasets/yolo_debug/`.
4. The collector takes the same validated near-field RGB-D snapshot as
   `agent-manip-picker` and calls the shared `stage_for_openvla()` deterministic
   forward/align/view sequence. **ARM_TELEOP** unlocks only after that sequence succeeds;
   deterministic staging is not recorded, so the first demonstration action starts from
   the same base-camera pose where OpenVLA receives control at runtime. If localization
   or deterministic staging fails, the collector automatically returns to ready, settles
   the camera, and retries with a fresh snapshot up to three total attempts before falling
   back to `MANUAL_BASE`.

   Tissue and sunblock have an additional near-range lock because their detector
   confidence can be intermittent. Once an exact detection of the requested class is
   centered within 10 cm beyond the approach stop range, its world coordinate is locked.
   On missed frames, live chassis pose projects that fixed coordinate back into arm-frame
   forward/lateral range, so the robot continues crawling until the normal pick distance
   instead of trusting stale-box floor/background depth. The class-keyed lock is also
   accepted when the fresh handoff snapshot misses, so deterministic camera entry can run.
5. **ARM_TELEOP** — EE nudges + grip; steps are recorded with live images.
   Closing (`v`) **commands `0.68` immediately** and holds that command;
   `gripper/holding` does not stop it early. The chassis is held at its
   captured world pose throughout arm teleop and every blocking policy pick, actively
   correcting backwards slope drift instead of relying on a zero-velocity command alone.
   `MARC_GRIP_CLOSE` is the final commanded position and remains clamped below the
   physical joint limit.

   The `V` pick path uses a narrower learned-policy seam. Before OpenVLA receives its
   first frame, the arm opens and moves deterministically to a pre-grasp pose above and
   slightly behind the localized target, where both gripper and object are visible in the
   base camera. OpenVLA controls only the final approach and close. A `holding=True`
   signal is not sufficient for success: the runtime lifts the gripper 8 cm and requires
   the live RGB-D localization to show that the target rose with it. Finger-on-finger
   contact is opened and restaged instead of triggering the basket motion.
5. **`P`** — scripted **place in the rear basket**. Hard-coded, so its ticks are **not written
   into the dataset** (no images / actions / joints). The training episode is approach + grasp
   only; `P` exists so `basket/occupied` can go true for the `E` keep filter. Hand nudges
   cannot place: the basket is *behind* the arm (`place_fwd` is negative in `pnp_params.json`).

   The motion is **pure joint space** — interpolations between poses, no Cartesian tracking:

   `keep latched grip` → `raise` → **`tuck inside chassis`** → `rotate tucked +π` → `extend into basket` → `release` → `retract` → `rotate tucked back` → `fold to ready`.

   Place copies the live squeeze command and does **not** start another close ramp. It
   also does not fold to the 9 cm ready pose while holding.

   Why it is built this way: a straight Cartesian line to a point behind the shoulder has no
   collision-free interpolation, and IK will cheerfully return a pose that reaches the basket
   with the elbow folded through the forearm. Servoing toward it drives the arm into itself.
   The arm first retracts to a checked central pose, then only joint 1 rotates. Every sampled
   arm-link origin plus collision padding must stay inside `MARC_PLACE_SWEEP_RADIUS_M`; otherwise
   placement is refused before motion. IK solves are seeded from neighbouring poses and rejected
   on branch jumps. Missing or unsettled joint feedback also stops the current trajectory while
   the gripper remains closed.
6. **`E`** — after `P` finishes, sensor-only keep if grasp was seen after close **and** `basket/occupied` is true; else discard to `datasets/marc_pnp_failed/`. During the blocking `P` motion, `E` is an immediate manual approval: it cancels the remaining scripted placement at the next control tick, holds the measured arm pose, keeps the episode without requiring basket occupancy, opens the gripper, returns home, and resets directly to the next collection cycle. No second `E` is needed.

So a keepable episode is: `A` → grasp with `v` → **`P` + `E` at any point during/after placement**. Without `P` the basket is never
filled, so `E` always evaluates `basket=False` and discards. Kept episodes contain only the
teleop pick steps collected before `P`.

## Layout

```
datasets/marc_pnp/Episode 001/
  episode.hdf5
  meta.json
```

Episode folders are named `Episode 001`, `Episode 002`, … (zero-padded, sequential under that
parent). That matches common OpenVLA / OXE-style corpora better than opaque hashes;
`to_rlds.py` still emits `episode_XXXX.npz` from whatever source folder name it finds.

HDF5 `steps/` datasets: `image_base_left`, `q_arm`, `q_grip`, `q_inner`, `ee_pos`,
`ee_quat`, `grasping`, `action` (7,), `t_sim`, `t_wall`. The camera key follows
`MARC_PRIMARY_CAM`, but the supported OpenVLA workflow pins it to `base_left`.

The HDF5 is the single episode source. Per-frame PNG copies, secondary RGB views, and
depth streams are not written: vanilla OpenVLA fine-tuning loads one primary RGB image,
no depth, and no proprioception. The two camera-lock diagnostics under
`datasets/yolo_debug/` remain because they are single startup snapshots rather than one
copy per transition.

**Action convention:** EE delta in arm_base  
Raw HDF5 uses `[dx, dy, dz, droll, dpitch, dyaw, close_intent]`, with
`close_intent=1`. The RLDS converter flips only that last label to OpenVLA/OXE's
canonical `0=closed, 1=open`. The physical `q_grip` remains proprioception and is
never used as an action token, so the arm's `0.68` close ceiling is not rescaled.
One `v` press records exactly one pre-close image paired with close intent; the full
sensor-confirmed finger ramp still executes physically but its internal control ticks
are not duplicated as identical training transitions.

**State convention:** OpenVLA `StateEncoding.JOINT` expects eight values: seven
joint slots followed by gripper state. MARC has six arm joints, so RLDS state is
`[q1, q2, q3, q4, q5, q6, 0, q_grip]`. The zero is a required padding slot, not
an extra measured joint.

### Which camera trains the policy

`MARC_PRIMARY_CAM` is pinned to `base_left` for the supported dataset path and is
recorded in `meta.json`/HDF5 attrs as `primary_camera`. The base view is used because:

- OpenVLA's pretraining mixture is overwhelmingly **third-person** RGB, and its
  single-image ViT backbone has seen far fewer wrist views, so a base view is closer to
  the pretrained distribution and needs less adaptation from a small LoRA run.
- The base is parked during `ARM_TELEOP`, so within a recorded episode the only things
  moving in frame are the arm and the object. Backgrounds stay fixed, which makes the
  image-to-action mapping much easier to learn from few episodes.
- A base view keeps the object and gripper visible together through approach and close.
  A wrist camera can lose the target as it closes in or become fully occluded by the
  grasped object.

Changing to a wrist or multi-image policy now requires recollection by design. Keeping
unused streams in every episode was not justified for the selected single-image OpenVLA
experiment and increased one 192-transition episode from roughly 41 MiB of useful
base-left HDF5 imagery to 424 MiB including unused streams and duplicate PNGs.

### Duplicate publishers on one camera topic

Some camera topics carry **two publishers at different resolutions and viewpoints**.
The SDK caches one message per topic, so a getter returns whichever publisher wrote last. Startup prints
`[probe] camera publishers:` with a publisher count and node name per topic — anything
above 1 there is the cause. The primary view locks the first resolution it sees; foreign
frames are dropped so a step is skipped rather than paired with a stale or mislabeled
training image.

At write time the recorder also drops any step whose primary image differs from the
episode's dominant shape. No training image is fabricated from a neighbouring frame.

VLA steps start at the first arm teleop subframe after auto-approach stop (base motion is not labeled as VLA actions).

## Convert for OpenVLA

```bash
cd demo
python3 -m manip_dataset.to_rlds \
  --in datasets/marc_pnp \
  --out datasets/marc_pnp_rlds \
  --image-size 224 \
  --camera base_left
```

`base_left` is pinned for OpenVLA training and export fails if that view is absent; it
never silently substitutes a gripper-camera image. `--camera auto` remains available for
deliberately mixed/legacy exports and uses each episode's recorded `primary_camera`.

Produces `episodes/episode_XXXX.npz`, `manifest.json`,
`dataset_statistics.json`, and a self-contained `marc_pnp_dataset_builder.py`.
Conversion validates the complete source export before reporting success. It rejects
episodes with empty object names, non-pick tasks, or language mentioning a place/drop/
basket phase: scripted `P` actions were intentionally not recorded, so labeling those
frames as pick-and-place would be incorrect. Older episodes with those labels must not
be used for fine-tuning; collect fresh episodes after clearing or changing
`MARC_DATASET_ROOT`.

You can rerun the same validation independently:

```bash
python3 -m manip_dataset.validate_openvla_dataset \
  --export datasets/marc_pnp_rlds \
  --camera base_left
```

Materialize the actual TFDS/RLDS dataset from an OpenVLA environment:

```bash
python datasets/marc_pnp_rlds/marc_pnp_dataset_builder.py \
  --manual-dir datasets/marc_pnp_rlds \
  --data-dir datasets/tfds
```

Then validate both the materialized TFDS dataset and one sample passed through
OpenVLA's official `RLDSDataset` loader (run this inside that OpenVLA environment):

```bash
python3 -m manip_dataset.validate_openvla_dataset \
  --export datasets/marc_pnp_rlds \
  --tfds-root datasets/tfds \
  --openvla-dir /path/to/openvla
```

The TFDS check validates the 8D state directly. Current upstream OpenVLA fine-tuning
loads image, action, and language but sets `load_proprio=False`; the official-loader
check therefore verifies the exact fields that training consumes rather than claiming
that training reads state.

The builder emits deterministic episode-level `train`/`val` splits (90/10), so frames
from one episode cannot leak into both. It registers as `marc_pnp:1.2.0`; image key is
`image`, state key is `state`, language key is `language_instruction`, and
`unnorm_key=marc_pnp`.

## Fine-tune OpenVLA with augmentation and early stopping

Use an official OpenVLA checkout. The launcher makes a patched training copy under the
work directory; it does not edit that checkout. Training uses upstream color/crop image
augmentation, evaluates the unaugmented `val` episodes every 500 optimizer steps, keeps
the best LoRA adapter, stops after five non-improving checks, then merges that adapter
into a directly loadable Hugging Face model.

```bash
OPENVLA_DIR=/path/to/openvla \
OPENVLA_WORK_DIR=/path/to/a-large-writable-volume/marc-openvla \
training/train_openvla_lora.sh /path/to/tfds-root
```

Defaults use micro-batch 2 plus 8-way gradient accumulation and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to leave VRAM headroom. Override with
`OPENVLA_BATCH_SIZE`, `OPENVLA_GRAD_ACCUM`, `OPENVLA_VAL_STEPS`, or
`OPENVLA_EARLY_STOP_PATIENCE`. The default work directory is
`/tmp/marc-openvla`, and the launcher refuses a read-only target or one with less
than 20 GiB free. On this machine `/mnt/data` is currently mounted read-only, so
do not use it unless it is remounted writable. The merged model is written to
`$OPENVLA_WORK_DIR/final-model`.

To run it in the agent:

```bash
MARC_PICK_POLICY=openvla \
MARC_OPENVLA_MODEL_DIR=/path/to/marc-openvla/final-model \
docker compose -f demo/docker-compose.yml up --build agent
```

## GraspGenX capture boundary

Each success and failure now stores `grasp_capture` in `meta.json`: the first
latched-grasp (or final attempted) EE pose in `arm_base`, outcome, and gripper.
This preserves the demonstrated grasp without deriving yaw from depth.

It is intentionally **not** labeled as a train-ready GraspGenX dataset yet.
The upstream GraspGenX loader additionally requires the exact object mesh and
an object-relative 4x4 grasp transform. The manipulation API exposes neither
the spawned object's mesh path nor its exact pose, so manufacturing that
transform from YOLO/depth would train on the same unreliable estimate this
dataset is meant to replace. Join the recorded arm-base grasp pose to an exact
trainer-side object pose/mesh export before running GraspGenX training.

`language_instruction` is pick-only (`"pick up the tumbler"`). Place (`P`) is still used as a keep-filter but is not in the action labels, so the text does not mention the basket.

## Code changes need a rebuild

The Dockerfile `COPY`s `demo/` into the image, so an edit is invisible until the
image is rebuilt — except bind-mounted paths. `agent-manip-probe` mounts
`manip_dataset/`. `agent-manip-pick` also mounts `marc_agent/`, `nearfield.py`,
`arm_kin.py`, `arm_pick.py`, and `pnp_params.json`, and reuses the full local `marc-agent:local`
image so its Python changes are immediate. The collector's YOLO/torch live in the
`manip-probe` image target (`marc-agent-manip-probe:local`); dependency or
Dockerfile changes there still need `--build`.

```bash
cd demo
MARC_TEAM_ID=u1 docker compose run --rm -it --build agent-manip-probe
```

## Env tunables

Defaults for ready/teleop numerics are defined once in
[`arm_control.py`](arm_control.py) as `DEFAULT_*` / `DEFAULT_READY_*` (from
`arm_kin.DEFAULT_ARM_Q` for J5/J6). Override them in **`demo/.env`** (next to
`docker-compose.yml`) — that file is loaded by `agent-manip-probe` and
`agent-manip-pick` via `env_file`.
A `.env` under `manip_dataset/` is **not** read by Compose.

| Variable | Default (code) | Meaning |
|---|---|---|
| `MARC_TEAM_ID` | `u1` | Any value, as long as it matches the manip-trainer namespace |
| `MARC_DATASET_ROOT` | `demo/datasets/marc_pnp` | Success output root |
| `MARC_YOLO_WEIGHTS` | `models/runtime/stage2-nearfield-detector/model.pt` | Explicit local Ultralytics `.pt`; runtime downloads are disabled |
| `MARC_YOLO_CONF` | `0.15` | Detection confidence floor |
| `MARC_YOLO_IMGSZ` | `1920` | Selected-contract inference size |
| `MARC_YOLO_DEVICE` | `0` | CUDA index, or `cpu` |
| `MARC_YOLO_HALF` | `0` | FP16 predict. Off by default (avoids Ultralytics `half` deprecation spam) |
| `MARC_YOLO_CAM` | `base_left` | RGB camera for centering (`base_left`/`base_right`) |
| `MARC_TARGET_OBJECT` | `tumbler` | Exact required class; unsupported names fail closed |
| `MARC_APPROACH_STOP_M` | `0.56` | Stop when calibrated arm-base forward is below this (m) |
| `MARC_APPROACH_ALREADY_NEAR` | `0.44` | Hard calibrated arm-base fwd floor (m); the base never crawls closer even if box-edge geometry is noisy |
| `MARC_APPROACH_FRAME_BOTTOM_MARGIN` | `0.10` | Stop forward motion when the object's lower YOLO-box edge enters this bottom fraction of the RGB frame |
| `MARC_APPROACH_CRAWL` | `0.30` (`.env`: `0.80`) | Requested approach speed (m/s); clamped by `MAX_STEP` |
| `MARC_APPROACH_MAX_STEP` | `0.04` (`.env`: `0.08`) | Max distance travelled per camera frame (m). Caps effective speed to the sensor rate |
| `MARC_APPROACH_STALE_STOP` | `0.5` | Hold still if no usable RGB/depth for this long (s), instead of driving blind |
| `MARC_APPROACH_K` | `3` | Persistence frames, counted only on **new** RGB frames |
| `MARC_APPROACH_PERCENTILE` | `30` | Percentile of depth pixels inside the YOLO box |
| `MARC_APPROACH_MIN_VALID` | `4` | Min valid depth pixels in the box; below this range is unknown |
| `MARC_APPROACH_OVERRUN_M` | `0.68` | Commanded extra crawl after detect (m). Live stop is arm-base fwd ≤ `ALREADY_NEAR`; this only sizes the wall-clock cap when the sim closes slower than `cmd_vel` |
| `MARC_APPROACH_YAW_GAIN` | `1.2` | P-gain on image-right error → `angular_z` (negated). Flip the sign if it turns the wrong way |
| `MARC_APPROACH_YAW_MAX` | `0.35` | Clamp on yaw rate (rad/s) |
| `MARC_APPROACH_YAW_MIN` | `0.10` | Floor on yaw rate (rad/s) while aligning, so a small error still turns the chassis |
| `MARC_APPROACH_CENTER_TOL` | `0.05` | \|image error\| counted as centered (half-width units); ends `ALIGN` |
| `MARC_APPROACH_REALIGN` | `0.12` | \|image error\| that stops `DRIVE` and returns to `ALIGN`. Also the max off-axis error allowed to stop |
| `MARC_APPROACH_ALIGN_HOLD` | `2` | New RGB frames within `CENTER_TOL` required before driving |
| `MARC_APPROACH_ALIGN_SETTLE` | `0.15` | Stand-still time (s) on entering `ALIGN` so cameras catch up with the pose |
| `MARC_APPROACH_ALIGN_TIMEOUT` | `0` | If `>0`, log when ALIGN has taken this long — **does not drive anyway** |
| `MARC_APPROACH_BURST_M` | `0.15` (`.env`: `0.25`) | Straight-line travel (m) per `DRIVE` burst before re-checking centering; `0` disables re-checks |
| `MARC_APPROACH_LAT_MIN_M` | `0.05` | Ignore depths nearer than this inside the YOLO box |
| `MARC_APPROACH_LAT_MAX_M` | `20.0` | Ignore depths beyond this inside the YOLO box |
| `MARC_APPROACH_LOG_EVERY` | `1` | Print each N ticks (`0` = off) |
| `MARC_BASE_LIN` | `0.08` | Manual base forward (`i`) speed (m/s) |
| `MARC_BASE_LIN_BACK` | same as `MARC_BASE_LIN` | Manual base reverse (`k`) speed (m/s) |
| `MARC_BASE_ANG` | `0.25` | Manual base yaw (`j`/`l`) rate (rad/s) |
| `MARC_TELEOP_DX` etc. | see `DEFAULT_TELEOP_*` | Arm click step sizes (IK teleop after ready) |
| `MARC_PRIMARY_CAM` | `base_left` | Sole RGB view recorded and exported for OpenVLA; keep at `base_left` for train/runtime consistency |
| `MARC_GRIP_DWELL` | `pnp_params.grip_dwell` (`1.0`) | Seconds a `v`/`b` click holds pose while the fingers travel, so `gripper/holding` can be observed |
| `MARC_GRIP_RAMP` | `0` | `0` jumps to `MARC_GRIP_CLOSE` on the first control tick; a positive value restores a software ramp over that many seconds |
| `MARC_GRIP_EXTRA` | `0.04` | Used only when `MARC_GRIP_STOP_ON_GRASP=1`; ignored by the default full-close mode |
| `MARC_GRIP_CONFIRM_SAMPLES` | `3` | Used only when `MARC_GRIP_STOP_ON_GRASP=1` to reject transient hold signals |
| `MARC_GRIP_STOP_ON_GRASP` | `0` | `0` always finishes and holds the full close; `1` enables the older sensor-stopped close |
| `MARC_GRIP_CLOSE` | `pnp_params.grip_close` (`0.58`), `.env` sets `0.68` | Final close command, clamped to the `0.703` joint limit; manual `v` reaches and holds `0.68` with the provided configuration |
| `MARC_OPENVLA_STAGE_BACK_M` | `0.01` | Deterministic handoff distance behind the target; the 1 cm stand-off keeps the handoff just short of the tumbler |
| `MARC_OPENVLA_STAGE_MIN_FORWARD_M` | `0.16` | Minimum forward-only clearance from the current ready EE before lateral alignment or descent can begin |
| `MARC_OPENVLA_STAGE_UP_M` | `0.225` | Handoff height above the target; shortens the final descent by 7 cm from the previous 0.155 m handoff |
| `MARC_IK_JOINT_STEP` | `0.04` | Maximum DLS joint change per control tick (rad); bounds Cartesian arm speed and prevents IK branch-jump commands |
| `MARC_OPENVLA_VERIFY_LIFT_M` | `0.08` | Lift applied before visually accepting a candidate OpenVLA grasp |
| `MARC_OPENVLA_VERIFY_MIN_RISE_M` | `0.035` | Minimum measured target rise required after the verification lift |
| `MARC_OPENVLA_VERIFY_MAX_XY_DRIFT_M` | `0.12` | Maximum target-localization XY drift allowed during verification |
| `MARC_BASE_HOLD_POS_KP` | `2.0` | Longitudinal world-pose hold gain used to counter slope drift during manipulation |
| `MARC_BASE_HOLD_MAX_MPS` | `0.08` | Maximum forward/backward correction speed while holding the manipulation pose |
| `MARC_BASE_HOLD_DEADBAND_M` | `0.002` | Longitudinal drift tolerated before correction starts |
| `MARC_BASE_HOLD_YAW_KP` | `1.5` | Heading hold gain during manipulation |
| `MARC_BASE_HOLD_MAX_RADPS` | `0.15` | Maximum heading correction rate during manipulation |
| `MARC_BASE_HOLD_YAW_DEADBAND_RAD` | `0.01` | Heading error tolerated before correction starts |
| `MARC_PLACE_YAW` | `3.1416` | Joint-1 rotation from front to basket. Trim if the drop lands off-centre |
| `MARC_PLACE_SPEED` | `pnp_params.carry_speed`, `.env` sets `0.5` | Maximum joint rate (rad/s) for feedback-paced, cosine-eased `P` motion. Joint 1's URDF limit is `1.396` |
| `MARC_PLACE_IK_GUARD` | `1.2` | Max rad any joint may move from its IK seed before the solution is rejected as a branch jump (which is what folds the arm into itself) |
| `MARC_PLACE_APEX_FWD` / `UP` | `0.14` / `0.40` | Central tucked EE waypoint used before either 180° rotation |
| `MARC_PLACE_SWEEP_RADIUS_M` | `0.22` | Maximum conservative arm-link sweep radius during basket rotation; larger paths fail closed |
| `MARC_PLACE_REC_EVERY` | `3` | Unused for the dataset (`P` is not recorded); kept only if you pass a custom `on_step` to `place_in_basket` |
| `MARC_READY_J1`…`J6` | `DEFAULT_ARM_Q` = `0, 0.262, -2.094, 0, -0.785, 1.571` | **Joint-space ready start** (rad). Unset = default pose; set any subset to override those joints only. Interpolated once; no Cartesian IK during ready |
| `MARC_READY_MOVE_S` | `2.0` | Seconds to interpolate into the ready joint pose |
| `MARC_READY_OPEN_GRIP` | `0` | `1` releases the gripper when moving to ready. Default `0` keeps a grasped object held through `E`/`N`; press `b` in MANUAL_BASE to release |
| `MARC_READY_FWD` / `UP` / `PITCH` | (unused for ready) | Kept for env compat only — tip pose is FK of the ready joints |

Ready = smooth joint move to `J1…J6` (default `DEFAULT_ARM_Q` unless overridden), then
teleop uses normal EE-delta IK from that FK pose. Startup logs the resolved joints and
whether they came from defaults or env.
The gripper command is *not* touched by the ready move, so ending an attempt with `E` while
holding an object keeps holding it. Use `b` (MANUAL_BASE or ARM_TELEOP) to release.
