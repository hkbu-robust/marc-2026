# Manip-trainer teleop dataset collector

Keyboard-driven demonstration collector against the MARC **manip-trainer** profile.
Produces raw HDF5 episodes for later OpenVLA LoRA fine-tuning.

## Run

```bash
# terminal 1
bash simulation-platform/marc.sh manip-trainer

# terminal 2
cd demo
mkdir -p datasets
MARC_TEAM_ID=u1 docker compose up --build agent-manip-probe
```

Requires an interactive TTY (`stdin_open` / `tty` on `agent-manip-probe`). Press `?` for the key map.

## Episode flow

1. Arm moves to a **joint-space ready** pose (`MARC_READY_J1…J6`), then teleop uses EE-delta IK.
2. **MANUAL_BASE** — `i/k/j/l` place and aim the chassis; `A` starts auto-approach.
3. **AUTO_APPROACH** — baselines base depth, then **stop-turn-go**: it turns in place with zero linear velocity until the nearest object sits in the base-camera center (`ALIGN`), drives straight for a short burst (`DRIVE`), and re-aligns whenever the object drifts. **ALIGN is skipped** when the object is already centered, the inner patch is already nearer than `MARC_APPROACH_ALREADY_NEAR`, the lower image band has dropped vs baseline (object at the robot's feet), or a previously seen object has left the centering band. It stops on **center−surround contrast**, on inner-patch depth (`MARC_APPROACH_INNER_STOP`) once aimed, or immediately if already too near (`C` cancels).
   Each tick logs the phase plus `err` (image error), `obj` (object columns found) and `flr` (measured floor depth). `obj=0` means there is nothing to center on, so it will not yaw — see `MARC_APPROACH_LAT_*`.
   The sim publishes **two streams at different resolutions on one depth topic** (e.g. 480×640 and 360×640, with different geometry). The baseline counts the resolutions it sees, locks the majority one, and ignores the rest; without that the two streams interleave and `c0` lands between them, leaving a ±0.014 false swing on empty floor that swamps `MARC_APPROACH_CONTRAST`.
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

   The motion is **pure joint space** — eight interpolations between poses, no Cartesian tracking:

   `squeeze` (re-close, dwell) → `fold to ready` (`DEFAULT_ARM_Q`, still holding) → `raise to carry height` → **`rotate over basket` (joint 1 only, 180°)** → `lower into basket` → `release` → `raise clear` → `rotate back to front` → `fold to ready`.

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
`[dx, dy, dz, droll, dpitch, dyaw, grip]` with `grip` in `{0,1}`.

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

`--camera auto` exports each episode's recorded `primary_camera`; pass `base` or `gripper`
to force a view (it falls back to the other one if the requested view is absent).

Produces `episodes/episode_XXXX.npz`, `manifest.json`, and `dataset_statistics.json` (action mean/std). Wire those into your OpenVLA dataset transform / OXE RLDS builder; this repo does not run training.

## Code changes need a rebuild

The Dockerfile `COPY`s `demo/` into `marc-agent:local`, so an edit is invisible until the
image is rebuilt:

```bash
cd demo
MARC_TEAM_ID=u1 docker compose run --rm -it --build agent-manip-probe
```

`agent-manip-probe` also bind-mounts `./manip_dataset`, so edits inside that package are
live without a rebuild; changes to other `demo/*.py` files still need `--build`. If the key
map printed at startup is missing keys you expect, you are running a stale image.

## Env tunables

Defaults for ready/teleop numerics are defined once in
[`arm_control.py`](arm_control.py) as `DEFAULT_*` / `DEFAULT_READY_*` (from
`arm_kin.DEFAULT_ARM_Q` for J5/J6). Override them in **`demo/.env`** (next to
`docker-compose.yml`) — that file is loaded by `agent-manip-probe` via `env_file`.
A `.env` under `manip_dataset/` is **not** read by Compose.

| Variable | Default (code) | Meaning |
|---|---|---|
| `MARC_TEAM_ID` | `u1` | Any value, as long as it matches the manip-trainer namespace |
| `MARC_DATASET_ROOT` | `demo/datasets/marc_pnp` | Success output root |
| `MARC_APPROACH_CONTRAST` | `0.015` | Primary stop: rise in (ring−center) vs baseline (m). Pitch cancels; centered objects raise this |
| `MARC_APPROACH_TAU` | `0.08` | Secondary whole-ROI drop (`d0−d`); kept high so empty bumps rarely trip |
| `MARC_APPROACH_D_NEAR` | `0` | Absolute near stop (m); `≤0` disables |
| `MARC_APPROACH_CRAWL` | `0.30` | Requested approach speed (m/s); clamped by `MAX_STEP` |
| `MARC_APPROACH_MAX_STEP` | `0.04` | Max distance travelled per depth frame (m). Caps effective speed to the sensor rate |
| `MARC_APPROACH_STALE_STOP` | `0.5` | Hold still if no usable depth for this long (s), instead of driving blind |
| `MARC_APPROACH_K` | `3` | Persistence frames, counted only on **new** depth frames |
| `MARC_APPROACH_INNER_FRAC` | `0.18` | Center patch size for contrast |
| `MARC_APPROACH_OUTER_FRAC` | `0.40` | Outer patch; ring = outer − inner |
| `MARC_APPROACH_PERCENTILE` | `30` | Percentile inside center/ring |
| `MARC_APPROACH_MIN_VALID` | `4` | Min valid pixels per region; below this the frame is unusable |
| `MARC_APPROACH_EMA` | `0.50` | EMA on contrast |
| `MARC_APPROACH_OVERRUN_M` | `0.285` | Extra forward creep after detect (m) |
| `MARC_APPROACH_YAW_GAIN` | `1.2` | P-gain on image-right error → `angular_z` (negated). Flip the sign if it turns the wrong way |
| `MARC_APPROACH_YAW_MAX` | `0.35` | Clamp on yaw rate (rad/s) |
| `MARC_APPROACH_YAW_MIN` | `0.10` | Floor on yaw rate (rad/s) while aligning, so a small error still turns the chassis |
| `MARC_APPROACH_CENTER_TOL` | `0.05` | \|image error\| counted as centered (half-width units); ends `ALIGN`. Lower = tighter centering, but it can hunt or time out |
| `MARC_APPROACH_REALIGN` | `0.12` | \|image error\| that stops `DRIVE` and returns to `ALIGN` |
| `MARC_APPROACH_ALIGN_HOLD` | `2` | New depth frames within `CENTER_TOL` required before driving |
| `MARC_APPROACH_ALIGN_SETTLE` | `0.15` | Stand-still time (s) on entering `ALIGN` so depth catches up with the pose |
| `MARC_APPROACH_ALIGN_TIMEOUT` | `4.0` | Give up aligning after this long (s) and drive anyway |
| `MARC_APPROACH_BURST_M` | `0.15` | Straight-line travel (m) per `DRIVE` burst before re-checking centering; `0` disables re-checks |
| `MARC_APPROACH_LAT_MIN_M` | `0.05` | Ignore depths nearer than this when centering. Raise above the arm distance if the log shows `obj=0` with a close `flr` |
| `MARC_APPROACH_LAT_MAX_M` | `20.0` | Ignore depths beyond this when centering |
| `MARC_APPROACH_LAT_CLOSER_M` | `0.04` | How much closer than the floor a column must be to count as object |
| `MARC_APPROACH_INNER_STOP` | `0.65` | When already centered, stop driving once inner-patch depth is below this (m). Stops the ALIGN loop if `A` is pressed with the object already in the inner patch |
| `MARC_APPROACH_ALREADY_NEAR` | `0.50` | Inner-patch depth (m) counted as pick range: no yaw, no overrun, stop immediately |
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
