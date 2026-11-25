# Participant Application & Docker Runtime (`demo/`)

This directory contains the containerized ROS 2 Humble participant application (`participant_app.py`), Dockerfiles for both the `agent` and `qwen` (vLLM) services, DDS configuration, and dataset conversion helpers.

## Standard Launch (Root Compose)

From the **repository root**, set your session token and run:

```bash
export MARC_TOKEN=<your-session-token>   # or copy demo/.env.example -> .env
docker compose up
```

See [docs/reproduce.md](../docs/reproduce.md) for prerequisites, environment variables, local Isaac Sim practice setup, and development overrides (`docker-compose.dev.yml`, `demo/docker-compose.qwen-remote.yml`, `demo/docker-compose.qwen-local.yml`).

## Key Files

| File / Directory | Purpose |
|---|---|
| `participant_app.py` | ROS 2 participant entry point connecting `marc_sdk` callbacks to `marc_agent` Stage 1 perception (and optional Stage 2 navigation/manipulation) |
| `Dockerfile` | Builds the offline-at-runtime ROS 2 Humble agent image (bakes YOLO26x checkpoints and Stage 2 manipulation dependencies) |
| `Dockerfile.vllm` | Builds the offline-at-runtime `vllm/vllm-openai:v0.22.0` image with the fine-tuned `Qwen3-VL-8B` weights baked in |
| `vllm-entrypoint.sh` | Hardware-aware vLLM launcher (auto-configures tensor parallelism, dtype, and memory fraction on loopback `:8004`) |
| `fastdds.xml` | Fast DDS network profile for platform ↔ participant communication |
| `vlm_dataset/` | CCTV ground-truth converters for Qwen3-VL LoRA JSONL and YOLO26x detection datasets ([vlm_dataset/README.md](vlm_dataset/README.md)) |
| `manip_dataset/`, `nearfield.py`, `mock_agent_navigation.py`, `arm_kin.py`, `arm_pick.py`, `urdf/` | Stage 2 mobile navigation, near-field RGB-D detection, and GraspGen-X manipulation runtime (outside Stage 1 paper scope) |
| `.env.example` | Safe environment variable template |
