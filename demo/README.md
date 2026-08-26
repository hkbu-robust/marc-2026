# SDK Demo Participant (reference implementation)

A runnable baseline built on **`marc_sdk`**. The protocol and ROS wiring (registration,
subscribe/publish, msg 301/311/302, `cmd_vel`, `world_pose`, occupancy) are all handled by
`MARCClient`, so you only fill in the **perception (VLA grounding) + navigation + manipulation**
logic through callbacks.

> This demo does not actually "solve" anything: it looks up pre-extracted answers and drives to
> pre-computed positions to exercise the full flow. **Replace the mock parts with your real
> perception/decision code before submission.**

## File layout

| File | Description |
|---|---|
| `participant_app.py` | Entry point -- registers `MARCClient` callbacks and wires the full Stage 1 / Stage 2 flow |
| `mock_agent_vla.py` | Legacy Stage 2-only demo lookup. The Stage 1 path does not construct or call it |
| `mock_agent_navigation.py` | Stage 2 navigation: goal-seek control + occupancy-grid D* Lite + lidar overlay |
| `mock_agent_manipulation.py` | Robot-arm pick motion (joint-angle keyframes) |
| `mock_demo_data.yaml` | Demo-only data extracted from the scenario (per-problem answer coordinates + obstacles). Real participants do not have this |
| `Dockerfile`, `docker-compose.yml` | Containerized run (the standard submission form) |
| `launch.sh` | Host run script (sets up ROS 2 Humble + PYTHONPATH) |

The three `mock_agent_*` files stand in for the agent code you will develop
(perception / navigation / manipulation). `participant_app.py` just calls them and connects the flow.

## Running

### A. Docker (standard -- submission method)

The participant application must be **developed/submitted as a Docker image** (required). The standard
entry point is **`docker compose up`**, and the team ID/token are entered in `docker-compose.yml`.

```bash
# Precondition: the simulation platform must already be running (a separate machine for scoring).

# 1) Fill MARC_TEAM_ID / MARC_TOKEN in docker-compose.yml with your assigned values
# 2) Standard run:
cd demo
docker compose up        # builds, then starts the agent (the organizer scores with the same command)
```

- **Separate machines**: the participant app runs on **separate, dedicated hardware** from the platform. The two machines share the same LAN and the **same `ROS_DOMAIN_ID`**, and DDS goes over the network (UDP). `network_mode: host` makes discovery happen on the machine NIC.
- GPU: the participant machine's GPU is used for VLA inference (`deploy.resources` -- requires nvidia-container-toolkit). Remove that block if not needed.
- Because CCTV video is transmitted over the network, a gigabit+ LAN is recommended.
- Submission: push to your private repository's `master` branch, tag the submission commit, and add the `marc-challenge-office` account as a collaborator -> the organizers clone it and run `docker compose up` (teams are scored one at a time, in sequence). Keep the token in the compose file of your private repository only (never in a public repository). See the Submission Guide for the full procedure.

### B. Host run (for development)

```bash
export MARC_TEAM_ID=u1
export MARC_TOKEN=<token>
cd demo
./launch.sh u1
# or set the env vars directly:
#   MARC_TEAM_ID=u1 MARC_TOKEN=<token> python3 participant_app.py
```

Even without `pip install`ing `marc_sdk`, `launch.sh` adds the SDK path (the starter-kit root) to
`PYTHONPATH`, so it runs directly from the kit.

## Behavior (non-interactive / automatic)

1. **Registration** -> SESSION_ACK
2. **Stage 1 candidate stage**: parse `voice_command` -> fetch every discovered CCTV once -> YOLO objects/people -> query-conditioned SAR selection. It records a pixel-space candidate handoff; #8/#9 still own object-relation choice, world projection, and submission.
3. **Stage 2**: `task_description` -> mock VLA -> `submit_stage2_grounding()` -> on reveal, occupancy D* Lite + lidar to the
   target -> arm pick -> D* Lite to the owner zone -> `task_complete()`
4. **End**: log the final score

Stage 1 candidate settings:

| Variable | Default | Meaning |
|---|---|---|
| `MARC_STAGE1_DETECTOR` | bundled YOLO when present | `yolo` for live candidates; `stub` is test-only |
| `MARC_SAR_DETECTOR` | `fallback` | `qwen` enables Qwen verification; fallback uses deterministic pose/anchor evidence |
| `MARC_QWEN_BACKEND` | `local` | `local` loads Transformers in the agent; `remote` calls a workstation vLLM server |
| `MARC_QWEN_MODEL_DIR` | `/agent/models/runtime/stage1-sar-qwen/model` | Complete pinned local Transformers snapshot |
| `MARC_QWEN_BASE_URL` | -- | Remote vLLM URL including `/v1`, e.g. `http://192.168.1.20:8000/v1` |
| `MARC_QWEN_API_KEY` | -- | Bearer key configured on the vLLM server; never written to metrics |
| `MARC_QWEN_MODEL` | `qwen3-vl-8b` | Name supplied through vLLM `--served-model-name` |
| `MARC_QWEN_TIMEOUT_S` | `30` | Remote request deadline before deterministic fallback |
| `MARC_QWEN_MAX_CROPS` | `12` | Per-person crops; annotated camera views still cover every candidate |
| `MARC_STAGE1_METRICS_PATH` | `artifacts/stage1_rounds.jsonl` | Per-round candidate counts, latency, selection method, and failures |

### Qwen on the participant/platform machine

For practice, mount the downloaded snapshot instead of copying 17.5 GB into every image:

```bash
export MARC_QWEN_HOST_MODEL_DIR="$HOME/marc-2026-work/pretrained/Qwen3-VL-8B-Instruct"
docker compose -f demo/docker-compose.yml \
  -f demo/docker-compose.qwen-local.yml up --build agent
```

The root submission image must eventually bundle this same pinned snapshot because judging
runs offline and does not apply the development override.

### Qwen on a separate workstation

Start vLLM on the workstation:

```bash
conda activate marc
vllm serve "$HOME/marc-2026-work/pretrained/Qwen3-VL-8B-Instruct" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen3-vl-8b \
  --dtype bfloat16 --gpu-memory-utilization 0.90 \
  --max-model-len 8192 --limit-mm-per-prompt '{"image":24}' \
  --api-key marc-qwen-secret
```

Then start the participant on the MARC PC:

```bash
export MARC_QWEN_BASE_URL="http://WORKSTATION_IP:8000/v1"
export MARC_QWEN_API_KEY="marc-qwen-secret"
docker compose -f demo/docker-compose.yml \
  -f demo/docker-compose.qwen-remote.yml up --build agent
```

Remote mode sends JPEG data URLs plus the finite candidate prompt. It does not stream raw
CCTV video, and any connection, authentication, timeout, response, or grammar failure uses
the same deterministic candidate fallback. This mode is for practice; submission cannot
depend on a separately started workstation process.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MARC_TEAM_ID` | `u1` | Team ID |
| `MARC_TOKEN` | -- | Session token (required) |
| `ROS_DOMAIN_ID` | `0` | ROS domain |
| `NAV_USE_MOCK_OBSTACLES` | unset | If `1`, inject yaml discs from `mock_demo_data.yaml` (debug only; scoring has no yaml) |
