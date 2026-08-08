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
3. **AUTO_APPROACH** — baselines base depth, crawls forward, stops on **center−surround contrast** (object in view) then overrun (`C` cancels).
4. **ARM_TELEOP** — EE nudges + grip; steps are recorded with live images.
5. **`E`** — sensor-only keep if grasp was seen after close **and** `basket/occupied` is true; else discard to `datasets/marc_pnp_failed/`.

## Layout

```
datasets/marc_pnp/<episode_id>/
  episode.hdf5
  meta.json
```

HDF5 `steps/` datasets: `image_gripper`, `image_base` (optional), `depth_*`, `q_arm`, `q_grip`, `q_inner`, `ee_pos`, `ee_quat`, `grasping`, `action` (7,), `t_sim`, `t_wall`.

**Action convention:** EE delta in arm_base  
`[dx, dy, dz, droll, dpitch, dyaw, grip]` with `grip` in `{0,1}`.

VLA steps start at the first arm teleop subframe after auto-approach stop (base motion is not labeled as VLA actions).

## Convert for OpenVLA

```bash
cd demo
python3 -m manip_dataset.to_rlds \
  --in datasets/marc_pnp \
  --out datasets/marc_pnp_rlds \
  --image-size 224
```

Produces `episodes/episode_XXXX.npz`, `manifest.json`, and `dataset_statistics.json` (action mean/std). Wire those into your OpenVLA dataset transform / OXE RLDS builder; this repo does not run training.

## Env tunables

Defaults for ready/teleop numerics are defined once in
[`arm_control.py`](arm_control.py) as `DEFAULT_*` / `DEFAULT_READY_*` (from
`arm_kin.DEFAULT_ARM_Q` for J5/J6). Override them in **`demo/.env`** (next to
`docker-compose.yml`) — that file is loaded by `agent-manip-probe` via `env_file`.
A `.env` under `manip_dataset/` is **not** read by Compose.

| Variable | Default (code) | Meaning |
|---|---|---|
| `MARC_TEAM_ID` | `u1` (required) | Must match manip-trainer namespace |
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
| `MARC_APPROACH_OVERRUN_M` | `0.15` | Extra forward creep after detect (m) |
| `MARC_APPROACH_LOG_EVERY` | `1` | Print each N ticks (`0` = off) |
| `MARC_BASE_LIN` | `0.08` | Manual base forward (`i`) speed (m/s) |
| `MARC_BASE_LIN_BACK` | same as `MARC_BASE_LIN` | Manual base reverse (`k`) speed (m/s) |
| `MARC_BASE_ANG` | `0.25` | Manual base yaw (`j`/`l`) rate (rad/s) |
| `MARC_TELEOP_DX` etc. | see `DEFAULT_TELEOP_*` | Arm click step sizes (IK teleop after ready) |
| `MARC_READY_J1`…`J6` | from `DEFAULT_ARM_Q` (J1=`0`) | **Joint-space ready start** (rad). Interpolated once; no Cartesian IK during ready |
| `MARC_READY_MOVE_S` | `2.0` | Seconds to interpolate into the ready joint pose |
| `MARC_READY_OPEN_GRIP` | `0` | `1` releases the gripper when moving to ready. Default `0` keeps a grasped object held through `E`/`N`; press `b` in MANUAL_BASE to release |
| `MARC_READY_FWD` / `UP` / `PITCH` | (unused for ready) | Kept for env compat only — tip pose is FK of the ready joints |

Ready = smooth joint move to `J1…J6`, then teleop uses normal EE-delta IK from that FK pose.
The gripper command is *not* touched by the ready move, so ending an attempt with `E` while
holding an object keeps holding it. Use `b` (MANUAL_BASE or ARM_TELEOP) to release.
