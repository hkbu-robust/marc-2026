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

1. Arm moves to a **down-tilted ready** pose.
2. **MANUAL_BASE** — `i/k/j/l` place and aim the chassis; `A` starts auto-approach.
3. **AUTO_APPROACH** — baselines **base** depth ROI, crawls forward, stops on residual / near depth (`C` cancels).
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

| Variable | Default | Meaning |
|---|---|---|
| `MARC_TEAM_ID` | `u1` (required) | Must match manip-trainer namespace |
| `MARC_DATASET_ROOT` | `demo/datasets/marc_pnp` | Success output root |
| `MARC_APPROACH_TAU` | `0.15` | Depth residual stop (m) |
| `MARC_APPROACH_D_NEAR` | `0.55` | Absolute near stop (m) |
| `MARC_APPROACH_CRAWL` | `0.05` | Auto-approach speed (m/s) |
| `MARC_APPROACH_K` | `5` | Persistence frames |
| `MARC_BASE_LIN` / `MARC_BASE_ANG` | `0.08` / `0.25` | Manual base teleop |
| `MARC_TELEOP_DX` etc. | `0.01` / `0.05` | Arm click step sizes |
| `MARC_READY_PITCH` | `0.70` | Ready pose pitch-down (rad) |
