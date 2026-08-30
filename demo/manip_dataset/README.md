# Manip-trainer teleop

Two client profiles talk to the same sim (`bash simulation-platform/marc.sh manip-trainer`):

| Profile | Service | What it is for |
|---|---|---|
| **`manip-pick`** | `agent-manip-pick` | Test GraspGen-X (**G**) or wrist-YOLO (**H**) grip + basket drop. No HDF5. |
| `tools` | `agent-manip-probe` | Record OpenVLA RLDS source + GraspGenX grasp captures. Manual grasp + **P** place. |

## GraspGen-X pick tester (no dataset)

```bash
# terminal 1
bash simulation-platform/marc.sh manip-trainer

# terminal 2
cd demo
docker compose --profile manip-pick up --build agent-manip-pick
```

In the trainer GUI, choose the object and **Place near** (or **Place far**, then **A** in the client). Press **G** to snapshot with near-field YOLO and run GraspGen-X pick + overhead basket. Press **H** for a wrist-YOLO pick (Stage 2 YOLO stays loaded; no GraspGen-X). **O** cycles the class, **N** recovers to ready (nothing is saved). Press `?` for the key map.

Requires an interactive TTY. `MARC_PICK_POLICY=graspgenx` is pinned on this service.
The image is `marc-agent-manip-pick:local` and builds with `MARC_QWEN_BACKEND=remote`
so `--build` does **not** Hub-pull the 17.5 GB Qwen snapshot (pick never loads it).
It still bakes GraspGen-X + near-field YOLO. Live pick/place Python is bind-mounted
(`marc_agent/`, `manip_dataset/`, `nearfield.py`, `arm_pick.py`, `pnp_params.json`);
Dockerfile/image changes still need `--build`. If a previous build is stuck on
`Fetching 16 files` for Qwen, stop it and rebuild.

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
3. **AUTO_APPROACH** — when two resolutions share `base_left`, locks the **taller 480×640 chassis** and **prefers 360×640 chassis depth** when that shape is in the lock window. Then **stop-turn-go** on **YOLO**: ALIGN yaws in place until the box sits in the base-camera center, DRIVE goes straight for a short burst, and it re-ALIGNS if the box drifts or is lost. It does **not** crawl without a detection, and a lost box is not treated as “already at our feet.” Stop is **arm-base forward** from the same localize as pick (`MARC_APPROACH_STOP_M`, default 0.26 m). After that it keeps crawling until that forward range is at pick range (`MARC_APPROACH_ALREADY_NEAR`, default 0.16 m); wall-clock `OVERRUN_M / speed` is only a cap, because the practice sim often closes much slower than `cmd_vel`. No extra crawl if already closer than `ALREADY_NEAR` (`C` cancels).
   Each tick logs `err` (box vs image center), `obj=used/raw`, `name`/`conf`, `fwd` (arm-base), and `d` (bbox depth). `obj=0/0` means YOLO saw nothing; `obj=0/3` means boxes existed but were skipped (e.g. `person`).
   Inference is always restricted to the exact `MARC_TARGET_OBJECT`; an unsupported target fails closed and never falls back to all classes. RGB is passed as a PIL image. Lock dumps PNGs under `datasets/yolo_debug/`.
4. **ARM_TELEOP** — EE nudges + grip; steps are recorded with live images.
   Closing (`v`) **ramps** over `MARC_GRIP_RAMP` and stops the instant `gripper/holding`
   goes true, plus `MARC_GRIP_EXTRA` rad of bite. `MARC_GRIP_CLOSE` is only the ceiling:
   commanding it and holding it there keeps squeezing after contact, and the sim resolves
   that penetration by ejecting the object — which reads as the grasp slipping out. The
   log prints the value it actually settled at. Because that value can land *below* half
   of `grip_close`, "is the gripper closed" tracks the commanded intent (`grip_target`),
   not the published number, or an early stop would silently unlatch the grasp.
5. **`P`** — scripted **place in the rear basket**. Hard-coded, so its ticks are **not written
   into the dataset** (no images / actions / joints). The training episode is approach + grasp
   only; `P` exists so `basket/occupied` can go true for the `E` keep filter. Hand nudges
   cannot place: the basket is *behind* the arm (`place_fwd` is negative in `pnp_params.json`).

   The motion is **pure joint space** — interpolations between poses, no Cartesian tracking:

   `keep latched grip` (no re-squeeze) → `raise to carry height` → **`rotate over basket` (joint 1 +π)** → `lower into basket` → `release` → `raise clear` → `rotate back to front` → `fold to ready`.

   Pick already stopped the fingers at contact. Place copies that live squeeze and does **not** command json/env close again — squeezing past contact ejects the object. It also does not fold to the 9 cm ready pose while holding.

   Why it is built this way: a straight Cartesian line to a point behind the shoulder has no
   collision-free interpolation, and IK will cheerfully return a pose that reaches the basket
   with the elbow folded through the forearm. Servoing toward it drives the arm into itself.
   Rotating a single joint cannot do that — every other joint holds the shape it had at carry
   height, which is the same height and the same idea as the arc in `arm_pick.py`. IK is used
   only twice (the vertical raise in front, and the drop into the basket), always seeded at the
   neighbouring pose, and **refused if it moves any joint more than `MARC_PLACE_IK_GUARD` rad
   from that seed** — a branch jump shows up as a deviation of radians, a real move as tenths.
   If either solve is refused, `P` aborts *before moving* and keeps holding, so you can `N`.
6. **`E`** — sensor-only keep if grasp was seen after close **and** `basket/occupied` is true; else discard to `datasets/marc_pnp_failed/`.

So a keepable episode is: `A` → grasp with `v` → **`P`** → `E`. Without `P` the basket is never
filled, so `E` always evaluates `basket=False` and discards. Kept episodes contain only the
teleop pick steps collected before `P`.

## Layout

```
datasets/marc_pnp/Episode 001/
  episode.hdf5
  meta.json
  images/
    base_left/000000.png …
    base_right/…
    gripper_left/…
    gripper_right/…
```

Episode folders are named `Episode 001`, `Episode 002`, … (zero-padded, sequential under that
parent). That matches common OpenVLA / OXE-style corpora better than opaque hashes;
`to_rlds.py` still emits `episode_XXXX.npz` from whatever source folder name it finds.

Every kept RGB frame is also dumped as a PNG under `images/<view>/`, with the per-view
totals recorded in `meta.json` as `png_counts`.

HDF5 `steps/` datasets: `image_base_left`, `image_base_right`, `image_gripper_left`,
`image_gripper_right`, `depth_base`, `depth_gripper`, `q_arm`, `q_grip`, `q_inner`,
`ee_pos`, `ee_quat`, `grasping`, `action` (7,), `t_sim`, `t_wall`.

Per the API reference each rig is a stereo pair **plus** a depth camera
(`base_camera/{left,right}/image` + `base_camera/depth/image`, same for `gripper_camera/`),
so all **six** robot views are written every step. `steps.attrs["rgb_views"]` lists the
views that made it into the file. Episodes recorded before this change used
`image_base`/`image_gripper` for the left eyes; `to_rlds.py` still reads those.

**Action convention:** EE delta in arm_base  
Raw HDF5 uses `[dx, dy, dz, droll, dpitch, dyaw, close_intent]`, with
`close_intent=1`. The RLDS converter flips only that last label to OpenVLA/OXE's
canonical `0=closed, 1=open`. The physical `q_grip` remains proprioception and is
never used as an action token, so the arm's `0.68` close ceiling is not rescaled.

### Which camera trains the policy

`MARC_PRIMARY_CAM` (default `base_left`) selects the **training** view out of
`base_left`, `base_right`, `gripper_left`, `gripper_right`; it is recorded in
`meta.json`/HDF5 attrs as `primary_camera`. The base rig is the default because:

- OpenVLA's pretraining mixture is overwhelmingly **third-person** RGB, and its
  single-image ViT backbone has seen far fewer wrist views, so a base view is closer to
  the pretrained distribution and needs less adaptation from a small LoRA run.
- The base is parked during `ARM_TELEOP`, so within a recorded episode the only things
  moving in frame are the arm and the object. Backgrounds stay fixed, which makes the
  image-to-action mapping much easier to learn from few episodes.
- A base view keeps the object, gripper, and basket visible at once, so approach and
  place phases are not ambiguous — a wrist camera loses the target as it closes in and
  can be fully occluded by the grasped object.

The other five views are still written to every episode, so switching later is a re-export
(`to_rlds.py --camera gripper_left`) rather than a re-collection. A wrist view is mainly
worth trying as a *second* input if you move to a policy that accepts multiple cameras;
`base_right` gives you stereo for depth-aware variants.

### Duplicate publishers on one camera topic

Some camera topics carry **two publishers at different resolutions and viewpoints**
(e.g. `base_camera/depth/image` alternating 480×640 and 360×640). The SDK caches one
message per topic, so a getter returns whichever publisher wrote last. Startup prints
`[probe] camera publishers:` with a publisher count and node name per topic — anything
above 1 there is the cause. Each view therefore locks the first resolution it sees:
foreign frames are dropped for the primary view (the step is skipped rather than paired
with a stale training image) and fall back to the last valid frame for the others, so a
foreign camera's pixels are never stored under this camera's name.

**This cannot fail a write.** At write time the recorder drops any step whose *primary*
image is off-resolution (the training image is the one thing that may not be faked), then
puts each secondary view on its dominant resolution by borrowing the nearest good frame —
capped at `MAX_FILL_FRAC` (10%) of the episode, beyond which the view is dropped rather
than mostly fabricated. Fill counts are printed and stored in `steps.attrs["filled_frames"]`
and `meta.json`, so a view that needed patching is visible rather than silent. Before this,
a single missing gripper frame at episode start made that whole view unstackable and it
vanished from the file without an error.

VLA steps start at the first arm teleop subframe after auto-approach stop (base motion is not labeled as VLA actions).

## Convert for OpenVLA

```bash
cd demo
python3 -m manip_dataset.to_rlds \
  --in datasets/marc_pnp \
  --out datasets/marc_pnp_rlds \
  --image-size 224 \
  --camera auto
```

`--camera auto` exports each episode's recorded `primary_camera`; pass an explicit
view such as `base_left` or `gripper_left` to force one (with a recorded-view fallback).

Produces `episodes/episode_XXXX.npz`, `manifest.json`,
`dataset_statistics.json`, and a self-contained `marc_pnp_dataset_builder.py`.
Materialize the actual TFDS/RLDS dataset from an OpenVLA environment:

```bash
python datasets/marc_pnp_rlds/marc_pnp_dataset_builder.py \
  --manual-dir datasets/marc_pnp_rlds \
  --data-dir datasets/tfds
```

The builder emits deterministic episode-level `train`/`val` splits (90/10), so frames
from one episode cannot leak into both. It registers as `marc_pnp:1.1.0`; image key is
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
`arm_pick.py`, and `pnp_params.json`. YOLO/torch live in the `manip-probe` image
target (`marc-agent-manip-probe:local`); adding ultralytics or changing the
Dockerfile needs `--build`. If the key map printed at startup is missing keys
you expect, you are running a stale image.

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
| `MARC_APPROACH_STOP_M` | `0.26` | Stop when arm-base forward (pick localize) is below this (m) |
| `MARC_APPROACH_ALREADY_NEAR` | `0.16` | Arm-base fwd (m) counted as pick range: no yaw, no overrun |
| `MARC_APPROACH_CRAWL` | `0.30` | Requested approach speed (m/s); clamped by `MAX_STEP` |
| `MARC_APPROACH_MAX_STEP` | `0.04` | Max distance travelled per camera frame (m). Caps effective speed to the sensor rate |
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
| `MARC_APPROACH_BURST_M` | `0.15` | Straight-line travel (m) per `DRIVE` burst before re-checking centering; `0` disables re-checks |
| `MARC_APPROACH_LAT_MIN_M` | `0.05` | Ignore depths nearer than this inside the YOLO box |
| `MARC_APPROACH_LAT_MAX_M` | `20.0` | Ignore depths beyond this inside the YOLO box |
| `MARC_APPROACH_LOG_EVERY` | `1` | Print each N ticks (`0` = off) |
| `MARC_BASE_LIN` | `0.08` | Manual base forward (`i`) speed (m/s) |
| `MARC_BASE_LIN_BACK` | same as `MARC_BASE_LIN` | Manual base reverse (`k`) speed (m/s) |
| `MARC_BASE_ANG` | `0.25` | Manual base yaw (`j`/`l`) rate (rad/s) |
| `MARC_TELEOP_DX` etc. | see `DEFAULT_TELEOP_*` | Arm click step sizes (IK teleop after ready) |
| `MARC_PRIMARY_CAM` | `base_left` | Training view (`base_left`/`base_right`/`gripper_left`/`gripper_right`; `base`/`gripper` accepted as the left eye). All six views are always recorded |
| `MARC_GRIP_DWELL` | `pnp_params.grip_dwell` (`1.0`) | Seconds a `v`/`b` click holds pose while the fingers travel, so `gripper/holding` can be observed |
| `MARC_GRIP_RAMP` | `1.2` | Seconds for a close to ramp from open to `MARC_GRIP_CLOSE` |
| `MARC_GRIP_EXTRA` | `0.02` | Extra rad of bite taken after contact is detected. `0` stops dead on contact |
| `MARC_GRIP_STOP_ON_GRASP` | `1` | Stop closing as soon as `gripper/holding` goes true. `0` restores commanding the full close and holding it |
| `MARC_GRIP_CLOSE` | `pnp_params.grip_close` (`0.58`), `.env` sets `0.68` | **Ceiling** for a close, clamped to the `0.703` joint limit — not the value that ends up held, since the ramp stops at contact. The effective value is printed at startup and after each `v` |
| `MARC_PLACE_YAW` | `3.1416` | Joint-1 rotation from front to basket. Trim if the drop lands off-centre |
| `MARC_PLACE_SPEED` | `pnp_params.carry_speed`, `.env` sets `0.5` | Joint rate (rad/s) for `P`. Joint 1's URDF limit is `1.396` |
| `MARC_PLACE_IK_GUARD` | `1.2` | Max rad any joint may move from its IK seed before the solution is rejected as a branch jump (which is what folds the arm into itself) |
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
