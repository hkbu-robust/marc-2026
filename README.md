# MARC 2026 — roBUst participant submission

This private repository contains the complete MARC 2026 participant application. The
submission entry point is the root [`docker-compose.yml`](docker-compose.yml); the
organizer-provided simulation/scoring platform runs separately on the same LAN and ROS
domain.

## Submission launch

On a native Ubuntu 22.04 host with Docker, Docker Compose, the NVIDIA driver, and NVIDIA
Container Toolkit installed, clone this repository and run exactly:

```bash
docker compose up
```

No setup script or manually copied model file is required. On a fresh machine Compose
builds both images, downloads the pinned dependencies and weights during the build, then
starts the participant. The initial build requires internet access and substantial disk
space; startup can take several minutes while vLLM initializes the model.

The root Compose project starts these services:

| Service | Purpose |
|---|---|
| `qwen` | Baked `vllm/vllm-openai:v0.22.0` server for the fine-tuned Qwen3-VL-8B model, bound to participant-host loopback on port 8004. GPU count, tensor parallelism, dtype, memory utilization, context length, and CPU offload are selected at startup from the visible NVIDIA hardware. |
| `agent` | ROS 2 Humble participant containing the SDK, Stage 1/2 code, both YOLO checkpoints, GraspGenX, and gripper assets. It starts only after the Qwen health check passes and runs `demo/participant_app.py`. |

The simulation platform is deliberately not part of this root Compose project. During
evaluation, the organizers run it on separate hardware. `network_mode: host`, Fast DDS,
and the matching `ROS_DOMAIN_ID` allow the participant to communicate with it.

Useful runtime commands:

```bash
docker compose logs -f qwen agent
docker compose ps
docker compose down
```

## Offline runtime and baked assets

The scoring runtime does not need internet access. Model repositories are pinned and
downloaded only in Docker build layers; runtime Hub/Transformers/YOLO clients are forced
offline. Large weights are intentionally excluded from Git.

| Runtime asset | Build source |
|---|---|
| Fine-tuned Qwen3-VL-8B | [roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned](https://huggingface.co/roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned) |
| Stage 1 CCTV YOLO | [roBUst-marc-2026/stage1-cctv-detector](https://huggingface.co/roBUst-marc-2026/stage1-cctv-detector) |
| Stage 2 near-field YOLO | [roBUst-marc-2026/stage2-nearfield-detector](https://huggingface.co/roBUst-marc-2026/stage2-nearfield-detector) |
| GraspGenX weights | [adithyamurali/GraspGenXModel](https://huggingface.co/adithyamurali/GraspGenXModel) |
| Gripper descriptions | [adithyamurali/gripper_descriptions](https://huggingface.co/datasets/adithyamurali/gripper_descriptions) |
| GraspGenX code | [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX) |

Exact revisions and YOLO SHA-256 checksums live in
[`models/runtime/fetch_runtime_weights.py`](models/runtime/fetch_runtime_weights.py).
Docker builds fail rather than silently accepting incomplete Qwen/GraspGenX snapshots or
changed YOLO checkpoints.

## Required configuration

The root Compose file contains submission defaults because the organizers launch it without
exporting variables. Confirm these values before freezing the submission:

- `MARC_TEAM_ID` and `MARC_TOKEN` are the assigned private team credentials.
- `ROS_DOMAIN_ID` matches the platform; its default is `0`.
- `RMW_IMPLEMENTATION` remains `rmw_fastrtps_cpp`.
- The repository and any image containing credentials remain private.

Optional environment overrides include `MARC_EXPERIMENTS_HOST_DIR`,
`MARC_QWEN_WALL_TIMEOUT_S`, `MARC_QWEN_MAX_CROPS`, and the `VLLM_*` tuning variables in
[`demo/vllm-entrypoint.sh`](demo/vllm-entrypoint.sh). Defaults are intended to work on the
organizer reference host (Ubuntu 22.04, RTX PRO 5000 48 GB) while adapting to other visible
NVIDIA GPU layouts.

## What the participant does

- Stage 1 parses each command, gathers all discovered CCTV frames, proposes objects and
  people with the Stage 1 detector, resolves spatial/SAR ambiguity with local Qwen plus
  deterministic geometry, projects the selected target into world coordinates, and submits
  the grounding result.
- Stage 2 grounds the requested item, navigates toward the revealed search area using the
  occupancy map and lidar, reacquires it with robot RGB-D and the near-field detector,
  approaches and retries the GraspGenX pick while time permits, verifies grasp/basket state,
  and returns to the runtime-provided owner position before completing.
- Large inference components are unloaded or handed off between stages where possible to
  limit GPU pressure.

The latest full Chungmu Stage 1 rehearsal completed all 33 rounds with a **95.9578** average.

## Fast local Python iteration

After the baked images have been built once, reuse their dependencies and weights while
mounting the current Python source read-only:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --no-build
```

This override is for development only. It also raises the navigation speed limit for faster
simulation tests. It does not create a standalone development image or download host-side
weights. Stop it with the same file set and `down`:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml down
```

## Frozen systems for experiments

`docker compose up` rebuilds `robust-agent:finals` and `robust-qwen:finals` from the current
checkout (`pull_policy: build`). Those tags are shared by everyone on the machine, and a
`docker compose up` from any clone replaces them, so they are not a stable experiment baseline.
Experiments use these fixed images instead:

| System | Compose override | Agent image | Qwen image |
|---|---|---|---|
| **Final Stage 1 system** (tag `final-system-v1`, ADR 0002) | `docker-compose.frozen.yml` | `marc-agent:final-system-v1` | `marc-qwen-vllm:qualifier-b2360c86` |
| Qualifier submission (`qualifier` tag, code `1450aa9`) | `docker-compose.qualifier.yml` | `marc-agent:qualifier-1450aa9` | `marc-qwen-vllm:qualifier-b2360c86` |

The Qwen model is unchanged between the two (`roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned@b2360c86`),
and so is the Stage 1 detector (SHA-256 `64296e4b…`). Check that the images exist, then start one:

```bash
docker images | grep -E 'final-system-v1|qualifier'
docker compose -f docker-compose.yml -f docker-compose.frozen.yml up
docker compose -f docker-compose.yml -f docker-compose.frozen.yml down
```

The overrides change only the image names. They remove the build sections and disable pulls,
so the frozen images are never rebuilt or overwritten, even with `--build`. If an image is
missing, Compose stops with `No such image`. Environment variables are unchanged, so this is
the submitted runtime configuration. Repeated runs with round dumps use
`scripts/run_stage1_repeats.py`, which defaults to `marc-agent:final-system-v1`
(`scripts/README.md`).

On a machine without the frozen images, build them once from a fresh clone of the tag. The
build needs internet; the Qwen image downloads about 17.5 GB.

```bash
git clone --branch final-system-v1 https://github.com/ahmad-rev0/marc-2026-robust.git
cd marc-2026-robust
docker buildx build --load --allow network.host --network host -f demo/Dockerfile \
  --target agent --build-arg MARC_BUILD_MODE=baked --build-arg MARC_QWEN_BACKEND=remote \
  -t marc-agent:final-system-v1 .
docker buildx build --load --allow network.host --network host -f demo/Dockerfile.vllm \
  --build-arg MARC_BUILD_MODE=baked -t marc-qwen-vllm:qualifier-b2360c86 .
```

For the qualifier agent, check out `1450aa9` instead and tag the agent `marc-agent:qualifier-1450aa9`.

## Local simulation platform

The bundled practice platform is separate from the submission entry point and requires the
organizer/NVIDIA setup described in the developer guide:

```bash
docker login nvcr.io
bash simulation-platform/marc.sh setup
cd simulation-platform
docker compose --profile platform up
```

Do not run this platform stack on the organizer participant machine; their scoring platform
is supplied independently.

## Submission checklist

Before the qualifier deadline:

1. Keep the GitHub repository private and push the frozen code to `master`.
2. From a fresh clone, confirm the root `docker compose up` path and offline runtime.
3. Create and push the annotated `qualifier` tag at the submitted `master` commit.
4. Add `marc-challenge-office` as a collaborator.
5. Do not add local model snapshots, generated results, caches, or experiment artifacts to Git.

The official requirements are in the
[MARC submission guide](https://marc-challenge.github.io/marc-dev-guide/en/submit-guide.html).

## Repository layout

| Path | Contents |
|---|---|
| `docker-compose.yml` | Organizer-facing one-command submission entry point |
| `docker-compose.dev.yml` | Rebuild-free Python development mounts |
| `docker-compose.frozen.yml` | Runs the frozen final system (`final-system-v1`) for experiments |
| `docker-compose.qualifier.yml` | Runs the qualifier submission images (for comparisons) |
| `demo/` | Participant entry point, vLLM image, navigation, near-field, and manipulation runtime |
| `marc_agent/` | Perception, reasoning, navigation, pick, delivery, and runtime policies |
| `marc_sdk/` | Organizer ROS 2 protocol client |
| `models/runtime/` | Small contracts/provenance plus the build-time model fetcher; large weights are ignored |
| `simulation-platform/` | Local-only organizer practice platform recipes |
| `tests/` | Unit and integration tests |
| `NOTICES.md` | Third-party software and asset notices |
