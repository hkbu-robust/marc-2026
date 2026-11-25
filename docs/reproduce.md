# Reproduction & Configuration Guide

This guide covers everything needed to verify the CPU test suite, build and launch the containerized Stage 1 system (`hkbu-robust/marc-2026`), connect to the local Isaac Sim practice platform, and reproduce the evaluation tables in *Grounding Vision–Language Models for Target Localization* (IEEE MetaCom 2026).

---

## 1. CPU Unit & Replay Tests (No GPU or ROS 2 Required)

Requires Python 3.10+. From the repository root:

```bash
pip install -e ".[test]"
pytest
```

This runs the unit and replay test suite across `marc_agent/`, `scripts/`, and CPU-safe `demo/` modules (`tests/ros/` skips automatically when `rclpy` is absent).

---

## 2. Running the Offline Submission Stack (`docker compose up`)

### Prerequisites
- Native **Ubuntu 22.04** host with **Docker**, **Docker Compose v2**, **NVIDIA Driver**, and **NVIDIA Container Toolkit**
- GPU with **≥ 48 GB VRAM** recommended to co-host the baked `Qwen3-VL-8B` vLLM server + YOLO26x agent (tested on NVIDIA RTX PRO 5000 48 GB and RTX PRO 6000 Blackwell Max-Q)
- Internet access **at build time only** (to download the pinned public weights from Hugging Face); **runtime is strictly offline** (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `YOLO_OFFLINE=1`).

### Environment Configuration (Sanitized Secrets & Keys)

All competition credentials were removed from this public release. Before launching Docker Compose, provide a session token via your shell or a local `.env` file (gitignored):

```bash
cp demo/.env.example .env
# Edit .env and set MARC_TOKEN=<your-session-token>
```

Or pass it inline:

```bash
MARC_TEAM_ID=robust MARC_TOKEN=<your-session-token> docker compose up
```

| Variable | Default | Notes |
|---|---|---|
| `MARC_TOKEN` | *(empty — required)* | Session token issued by your local practice platform (`simulation-platform/marc.sh`) or competition organizers |
| `MARC_TEAM_ID` | `robust` | Participant team ID matching the platform configuration |
| `ROS_DOMAIN_ID` | `0` | Must match the simulation platform's `ROS_DOMAIN_ID` |
| `VLLM_API_KEY` / `MARC_QWEN_API_KEY` | `local-vllm-key` | Loopback-only bearer key shared between `qwen` and `agent` containers |
| `MARC_QWEN_BASE_URL` | `http://127.0.0.1:8004/v1` | Override only when hosting vLLM on a separate machine |

### Baked Weights (Downloaded Automatically at Build Time)

`docker compose build` / `docker compose up` automatically fetches the pinned public checkpoints via `models/runtime/fetch_runtime_weights.py`:
- **Stage 1 CCTV Detector (34-class YOLO26x):** [`roBUst-marc-2026/stage1-cctv-detector`](https://huggingface.co/roBUst-marc-2026/stage1-cctv-detector) (rev `cb5a918f5bcd2c6561ef367fd275cbba49cbcf9c`, AGPL-3.0)
- **Stage 1 VLM Selector (LoRA-merged Qwen3-VL-8B):** [`roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned`](https://huggingface.co/roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned) (rev `b2360c86972018edd3df1c7ac21e28c129317928`, Apache-2.0)

---

## 3. Running Locally Against the Isaac Sim Practice Platform

The simulation platform is **not** started by the root `docker-compose.yml` (during official evaluation it runs on separate hardware). To run locally against the organizers' practice environment (`marc2026_chungmu`, 33 rounds):

1. Pull `nvcr.io/nvidia/isaac-sim:5.1.0` directly from NVIDIA NGC (accepting the NVIDIA Isaac Sim EULA) and build the local platform image via `simulation-platform/`:
   ```bash
   cp simulation-platform/.env.example simulation-platform/.env
   cp simulation-platform/problems.yaml.example simulation-platform/problems.yaml
   bash simulation-platform/marc.sh up
   ```
2. Note the session token configured for your practice run and start the participant stack in a second terminal:
   ```bash
   MARC_TEAM_ID=robust MARC_TOKEN=<practice-token> docker compose up
   ```
3. **Fast live-code development override** (skips rebuilding images and bind-mounts `marc_agent/` and `demo/`):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml up --no-build
   ```
4. **Remote vLLM workstation override** (if serving Qwen3-VL-8B on another host instead of baking 17.5 GB locally):
   ```bash
   export MARC_QWEN_BASE_URL="http://<workstation-host>:8004/v1"
   export MARC_QWEN_API_KEY="local-vllm-key"
   docker compose -f demo/docker-compose.yml -f demo/docker-compose.qwen-remote.yml up --build agent
   ```

---

## 4. Reproducing Paper Evaluations & Training

### Repeated Runs & Ablations (Paper Tables III & VI)

Build the submission images once (`docker compose build`), then run repeated cold starts of the 33 public practice rounds (`marc2026_chungmu`):

```bash
# 10 cold starts of the deployed final-system-v1 system (expected: ~87.25 ± 0.71)
python -m scripts.run_stage1_repeats \
  --runs 10 \
  --agent-image robust-agent:finals \
  --qwen-image robust-qwen:finals \
  --out ./experiments/final-repeats

# Summarize scores and decision paths (CPU)
python -m scripts.stage1_scores ./experiments/final-repeats/run-*
python -m scripts.stage1_decisions ./experiments/final-repeats/run-*
```

- **Geometry + Rule Reader arm (Table III):** pass `--no-qwen` to `scripts.run_stage1_repeats`.
- **Base `Qwen3-VL-8B-Instruct` arm (Table III):** download `Qwen/Qwen3-VL-8B-Instruct` locally, set `export MARC_BASE_QWEN_DIR=/path/to/Qwen3-VL-8B-Instruct`, and pass `--extra-compose scripts/compose/e2e-config-2-base-qwen.yml`.
- **Projection & footprint-prior ablation (Table VI, CPU):**
  ```bash
  python -m scripts.stage1_replay --runs ./experiments/final-repeats/run-* --out ./experiments/replay-on
  python -m scripts.stage1_replay --runs ./experiments/final-repeats/run-* --env MARC_GROUND_CALIBRATION=0 --env MARC_LANDMARK_FOOTPRINT=0 --out ./experiments/replay-plane
  ```

See [scripts/README.md](../scripts/README.md) for the full list of evaluation scripts.

### Fine-Tuning Pipelines (`training/` & `demo/vlm_dataset/`)

All workstation-specific paths are parameterized with environment variables:

- **Qwen3-VL-8B LoRA fine-tuning:** set `MARC_PRETRAINED_ROOT` (or `MARC_QWEN_MODEL_DIR`) to your local `Qwen/Qwen3-VL-8B-Instruct` directory (default `${MARC_PRETRAINED_ROOT:-./pretrained}/Qwen3-VL-8B-Instruct`) and `QWEN_LORA_OUTPUT_DIR` for adapter output before running `training/train_qwen_lora.sh <dataset_dir>` (see [demo/vlm_dataset/README.md](../demo/vlm_dataset/README.md) for dataset generation).
- **Stage 1 YOLO26x detector:** see `training/train_cctv_detector.py` and `training/configs/stage1-cctv-yolo26x.yaml`.
