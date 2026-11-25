# Stage 1 Evaluation, Replay & Ablation Scripts (`scripts/`)

These scripts reproduce the Stage 1 repeated-run measurements, projection replays, held-out selector ablations, and paper tables reported in *Grounding Vision–Language Models for Target Localization* (IEEE MetaCom 2026).

Run commands from the **repository root** (e.g., `python -m scripts.stage1_scores ...`).

## Script Overview

| Script | Purpose | Hardware |
|---|---|---|
| `run_stage1_repeats.py` | Runs the containerized agent `N` times against a fresh local Isaac Sim practice platform and collects score files, experiment events, and round dumps | GPU |
| `stage1_scores.py` | Computes mean, std, median, and per-component score summaries across repeated runs | CPU |
| `stage1_decisions.py` | Tallies Stage 1 decision paths (Qwen choice, fallback, geometry override, OOM skips) | CPU |
| `stage1_replay.py` | Replays projection and footprint-prior ablations (Table VI) directly on logged runs and held-out sets | CPU |
| `stage1_regress.py` | Compares a candidate run against baseline runs and checks round-level invariants | CPU |
| `heldout_selection.py` | Evaluates full runtime rounds or frozen selector arms (`A`: geometry, `B`: base Qwen, `C`: fine-tuned Qwen) on held-out cases (Table V) | GPU |
| `selector_arms_report.py` | Scores selector arms with exact-match rates, paired cluster bootstrap CIs, and McNemar tests | CPU |
| `small_target_classes.py` | Derives the small-object class list (`< 5×10⁻⁴` frame area) from detector training labels | CPU |
| `stage1_sensitivity.py` | Tests target-credit sensitivity across scoring-kernel radius grids | CPU |
| `paper_numbers.py` | Aggregates experiment outputs into LaTeX macros (`data/paper_numbers.yaml`) | CPU |

## Common Usage

### Repeated Public Practice Runs (GPU)

Build the submission images once with `docker compose build`, then run repeated cold starts:

```bash
python -m scripts.run_stage1_repeats \
  --runs 10 \
  --agent-image robust-agent:finals \
  --qwen-image robust-qwen:finals \
  --out ./experiments/final-repeats
```

Use `--extra-compose scripts/compose/e2e-config-2-base-qwen.yml` (with `MARC_BASE_QWEN_DIR` pointing to a local `Qwen/Qwen3-VL-8B-Instruct` snapshot) or `--no-qwen` to reproduce the Table III selector configurations.

### Projection & Footprint-Prior Ablations (CPU)

```bash
python -m scripts.stage1_replay --runs ./experiments/final-repeats/run-* --out ./experiments/replay-on
python -m scripts.stage1_replay --runs ./experiments/final-repeats/run-* --env MARC_GROUND_CALIBRATION=0 --out ./experiments/replay-off
```

### Camera Geometry Capture

`scripts/data/sdk_cameras.json` stores the 6 CCTV camera intrinsics (`K`), extrinsics (`R`, `t`), and ground heights captured from the live practice platform (`scripts/frozen_eval/capture_sdk_geometry.py`). Pass `--geometry` to replay scripts to use a different capture.
