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
- GPU: the agent uses this machine's `nvidia` container runtime (requires nvidia-container-toolkit).
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
2. **Stage 1 candidate stage**: parse `voice_command` -> fetch every discovered CCTV once -> GPU YOLO objects/people -> cached all-camera Qwen scene description -> Qwen validates YOLO classes and selects the command-aligned target/anchor -> GPU YOLO refresh of the selected camera -> world projection and submission.
3. **Stage 2**: `task_description` -> mock VLA -> `submit_stage2_grounding()` -> on reveal, occupancy D* Lite + lidar to the
   target -> arm pick -> D* Lite to the owner zone -> `task_complete()`
4. **End**: log the final score

Stage 1 candidate settings:

| Variable | Default | Meaning |
|---|---|---|
| `MARC_STAGE1_DETECTOR` | bundled YOLO when present | `yolo` for live candidates; `stub` is test-only |
| `MARC_SAR_DETECTOR` | `qwen` | Base and submission Compose pin Qwen verification on |
| `MARC_QWEN_BACKEND` | `remote` in demo | The demo uses the OpenAI-compatible Qwen vLLM on this host; root submission Compose remains local Qwen |
| `MARC_QWEN_MODEL_DIR` | `/agent/models/runtime/stage1-sar-qwen/model` | Complete pinned local Transformers snapshot |
| `MARC_QWEN_BASE_URL` | `http://127.0.0.1:8000/v1` in demo | Remote vLLM URL including `/v1` |
| `STREAM` | `false` | `true` selects `http://10.0.0.200:8000/v1`; otherwise Qwen uses localhost |
| `MARC_QWEN_API_KEY` | local vLLM key in demo | Bearer key configured on the vLLM server; never written to metrics |
| `MARC_QWEN_MODEL` | `qwen3-vl-8b` in demo | Name supplied through vLLM `--served-model-name` |
| `MARC_QWEN_TIMEOUT_S` | `90` | Remote request budget before deterministic fallback; wall wait is also capped |
| `MARC_QWEN_WALL_TIMEOUT_S` | `12` | Per-attempt HTTP wait cap; retries once when remaining budget covers two attempts |
| `MARC_QWEN_MAX_NEW_TOKENS` | `48` | Local and remote generation budget; 8 tokens truncates `TARGET_ID\|ANCHOR_ID` |
| `MARC_QWEN_MAX_CROPS` | `12` | Per-person crops; annotated camera views still cover every candidate |
| `MARC_STAGE1_METRICS_PATH` | `artifacts/stage1_rounds.jsonl` | Per-round candidate counts, latency, selection method, and failures |

### Qwen on the demo host

The standard `demo/docker-compose.yml` uses the host-network vLLM endpoint at
`http://127.0.0.1:8000/v1`, serving `qwen3-vl-8b`. Start that service before
`docker compose up`; the existing remote adapter handles the OpenAI-compatible API.
Use `STREAM=true docker compose up --build agent` when vLLM is on the machine at
`10.0.0.200`; the flag overrides `MARC_QWEN_BASE_URL` for that run.

### Local Qwen override

The root submission Compose file still forces local Qwen.
The root submission build bakes the pinned model into the image and needs no Qwen
environment variables at runtime.

For practice, mount the downloaded snapshot instead of copying 17.5 GB into every image:

```bash
export MARC_QWEN_HOST_MODEL_DIR="$HOME/marc-2026-work/pretrained/Qwen3-VL-8B-Instruct"
docker compose -f demo/docker-compose.yml \
  -f demo/docker-compose.qwen-local.yml up --build agent
```

The root submission image downloads this pinned snapshot during `docker compose`
build. Judging runtime is offline and does not apply a host bind-mount.

For a faster local rebuild while iterating, you can still mount a host snapshot
with `demo/docker-compose.qwen-local.yml` as above.
That override skips the image copy because the read-only host mount supplies the
same complete local snapshot; inference still runs inside the participant container.

### Qwen on a separate workstation

Start vLLM on the workstation:

```bash
conda activate marc
VLLM_BATCH_INVARIANT=1 vllm serve "$HOME/marc-2026-work/pretrained/Qwen3-VL-8B-Instruct" \
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

Remote mode first caches a compact, command-independent description of all camera views, then
includes that reference with every finite-ID target/anchor selection prompt. The winning camera
is re-run through GPU YOLO before either box is projected. It does not stream raw CCTV video,
and any connection, authentication, timeout, response, or grammar failure uses the same
deterministic candidate fallback. The remote compose overlay also sets the image
build arg so `--build` does not Hub-pull the 17.5 GB local snapshot. This mode is for
practice; submission cannot depend on a separately started workstation process.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MARC_TEAM_ID` | `u1` | Team ID |
| `MARC_TOKEN` | -- | Session token (required) |
| `ROS_DOMAIN_ID` | `0` | ROS domain |
| `NAV_USE_MOCK_OBSTACLES` | unset | If `1`, inject yaml discs from `mock_demo_data.yaml` (debug only; scoring has no yaml) |
