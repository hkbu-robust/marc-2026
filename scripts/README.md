# Stage 1 measurement scripts

All of them read #85 harness run folders (`run-NN/` with `robust_*.json` and
`experiments/<run>/events.jsonl` plus round dumps), as written by
`run_stage1_repeats.py`. Run them from the repo root in the `marc` env, e.g.
`conda run -n marc python -m scripts.stage1_replay ...`.

| script | answers |
|---|---|
| `run_stage1_repeats.py` | runs an agent image N times on a fresh platform and collects each run (GPU) |
| `stage1_scores.py` | score statistics over repeated runs (#88) |
| `stage1_decisions.py` | which decision paths the rounds took (#85) |
| `stage1_replay.py` | what a projection change does to logged rounds, on CPU (#110) |
| `stage1_regress.py` | whether a fresh run regressed against baseline runs (#110) |
| `heldout_selection.py` | what the whole runtime round (detector, refinement, Qwen, guards, projection) does on #87 held-out cases (#115, #117) |
| `small_target_classes.py` | which detector classes count as small (from the training labels; #115) |
| `stage1_sensitivity.py` | which credited targets fail attribution under other scorer constants (#135) |
| `paper_numbers.py` | the paper's numbers as LaTeX macros (`numbers.tex`) from `final-experiments/` (#130; mapping in `data/paper_numbers.yaml`, input formats in `docs/paper/paper_numbers_inputs.md`) |

`stage1_runs.py` holds what the replay and the gate share: the loader for logged
rounds and the reverse-engineered scoring kernel.

## What every Stage 1 PR attaches

1. **Regression gate.** Run one 33-round smoke run of the branch (~12 GPU-min) and
   compare it against the rolling baseline: the #88 cache-off runs at first, then
   the last merged PR's run. Build the branch image under its own tag (never
   `:local`; tags are machine-wide). The runner refuses to start if the agent
   container does not run that image.

   ```bash
   docker build -f scripts/frozen_logging.Dockerfile -t marc-agent:fs-<ticket> \
       --label org.opencontainers.image.revision=$(git rev-parse HEAD) .
   python -m scripts.run_stage1_repeats --runs 1 --agent-image marc-agent:fs-<ticket> \
       --out ~/marc-2026-work/experiments/ablations/final-system/<ticket>-<slug>/smoke
   python -m scripts.stage1_regress compare \
       --baseline ~/marc-2026-work/experiments/ablations/04-repeated-runs/cache-off/run-* \
       --candidate <smoke-run-dir> --allow <rounds this ticket should change> \
       --out <smoke-run-dir>/gate.md
   ```

   **Rolling baseline:** the #88 cache-off runs plus every smoke run merged into
   `final-system` since (`ablations/final-system/*/smoke/run-*`, `111-118-smoke`).
   A single previous run would give zero-width bands, and lf06, hf08 and sar03 flip
   between unchanged runs; the union keeps those flips and the values earlier
   tickets changed on purpose.

   Attach `gate.md`. It fails when a round outside `--allow` leaves its baseline
   band of totals by more than 0.5 points, or an invariant breaks (all rounds
   submitted, landmark/relation filled on object rounds, no sentinel coordinate,
   all cameras loaded, p95 simulated round time within the limit, every command
   parses to its target with this checkout, all `tests/test_*.py` pass). Other
   differences (camera, target/anchor point, submitted fields, decision paths) are
   listed per round for the reviewer. Bands of rounds that flip between unchanged
   runs (lf06, hf08, sar03) are wide; read their rows, not just the verdict.

2. **CPU replay delta (geometry PRs).** Replay the baseline runs with the new
   mechanism on (its default) and off, and attach both summaries. The example uses
   #112's planned flag; use your ticket's.

   ```bash
   RUNS=~/marc-2026-work/experiments/ablations/04-repeated-runs/cache-off/run-*
   SETS=~/marc-2026-work/experiments/ablations/05-frozen-eval-sets
   python -m scripts.stage1_replay --runs $RUNS --cases $SETS/heldout \
       --out <dir>/on
   python -m scripts.stage1_replay --runs $RUNS --cases $SETS/heldout \
       --env MARC_GROUND_CALIBRATION=0 --out <dir>/off
   ```

   With every new flag off, the replay must reproduce the logged runs: the
   summary's largest coordinate change stays below 1 mm and `test_stage1_replay`
   passes. The "on" summary gives the point delta (Stage 1 average, anchor and
   target credit, target-credit flips) and the held-out anchor error.

## Held-out selection

Detection changes (#115) and selection changes (#117) act before projection, so the
replay cannot show them, and the 33 public rounds are what the old rules were fitted
on. `heldout_selection.py run` replays each #87 held-out object case through
`run_stage1_round` inside the agent image, with this checkout's sources mounted and a
capped vLLM; `summarise` compares two runs (target credit within 1 m x 0.40 + anchor
credit x 0.15, gold-target rate, paired bootstrap CI; all, detector-miss, other).

```bash
docker run -d --name harness-qwen --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    --network host --ipc host -e VLLM_PORT=8014 -e VLLM_GPU_MEMORY_UTILIZATION=0.37 \
    -e VLLM_API_KEY=marc-qwen-secret -e VLLM_MODEL_NAME=qwen3-vl-8b \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 marc-qwen-vllm:qualifier-b2360c86
# then, per arm (an example wrapper: ablations/final-system/115-tiled-refinement/selection/run.sh)
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --network host --entrypoint '' \
    -e MARC_SAR_DETECTOR=qwen-remote -e MARC_QWEN_BACKEND=remote \
    -e MARC_QWEN_BASE_URL=http://127.0.0.1:8014/v1 -e MARC_QWEN_API_KEY=marc-qwen-secret \
    -e MARC_QWEN_MODEL=qwen3-vl-8b -e HOME=/root \
    -v <worktree>/marc_agent:/agent/marc_agent:ro -v <worktree>/scripts:/agent/scripts:ro \
    -v <sets>:/root/marc-2026-work/experiments/ablations/05-frozen-eval-sets:ro -v <out>:/out \
    -w /agent marc-agent:final-system-v1 python3 -m scripts.heldout_selection run \
    --out /out/<arm>.jsonl --env MARC_TILED_REFINE=1
python -m scripts.heldout_selection summarise --off <off>.jsonl --on <arm>.jsonl --cases odd
```

Selector arms on the frozen #87 sets (#89): `heldout_selection.py frozen --root <87-eval-sets>
--set heldout|public --arm A|B|C` feeds each case's `candidates_final` and `parsed_final` to
`run_stage1_round` (no detector, no reader, no refinement), and records the raw selector answer and
the final answer after guards. Run it in `marc-agent:final-system-v1` with only `scripts/` mounted.
`selector_arms_report.py RUN_DIR --root <87-eval-sets>` scores the arms: gold-ID exact match,
exact-kernel credit, cluster bootstrap over scenes and McNemar.

Mount a git worktree pinned at the commit under test, not the working clone, so the
code cannot change under a running arm. Send one warm-up request first: the first
vLLM request compiles kernels for about a minute. Held-out cases have one camera, so
camera choice is not measured; report anchor-geometry results on odd scenes (#112).

## Sharing the GPU

Two experiment runs (yours and a teammate's) can share the 96 GB card if each has
its own vLLM port, its own ROS domain and its own clone:

```bash
python -m scripts.run_stage1_repeats ... --project fs-a --ros-domain-id 88 --vllm-port 8004
python -m scripts.run_stage1_repeats ... --project fs-b --ros-domain-id 89 --vllm-port 8014
```

- **Memory:** `--vllm-gpu-fraction` (default 0.37) is the share of the physical card
  vLLM reserves: ~36 GB, as on the 48 GB judging card; 16.8 GB are weights, the rest
  is KV cache the rounds barely use. The entrypoint's default is 0.75 (~72 GB), and
  then nothing else fits. (`VLLM_GPU_MEMORY_MIBS` does not cap memory; the entrypoint
  only uses it to pick a fraction.) A run needs about 36 + 7 (Isaac Sim) + 2 (agent)
  GB; the runner refuses to start below `--min-free-gpu-mib` (45000).
- **Separate clones:** the platform writes score files to `<clone>/results/`, and the
  runner picks the new file there, so two runs from one clone would mix them up.
- **Timing:** concurrent runs slow each other down. Fine for scores and decisions;
  run timing-sensitive tests (warm-up, round time, latency) alone.
- Outside the runner (`docker compose up` by hand), set the same variables:
  `MARC_VLLM_PORT`, `MARC_VLLM_GPU_FRACTION` and `ROS_DOMAIN_ID`, with
  `scripts/frozen-logging.compose.yml` as the last `-f` file. The root
  `docker-compose.yml` (the submission entry point) is unchanged.

## Rules the replay relies on

- **Read feature flags at call time** (`os.environ.get` inside the function), not
  at import, so `--env` reaches them.
- **Gate every new mechanism behind its flag**, so the "off" replay still
  reproduces the frozen system.
- **Runs need their round dumps** (`MARC_ROUND_DUMP=1`, which the harness sets):
  candidate image sizes come from the dumped frames. Rounds the agent did not parse
  or submit are carried with their logged score.
- **Anchor projection goes through `stage1_pipeline.project_anchor`**: the held-out
  anchor replay calls it directly.
- **Selection is held fixed.** The replay recomposes the logged target/anchor pair
  (and the logged fallthrough); a change that would make the ranking or Qwen pick
  another pair only shows in the GPU smoke run.
- **Target credit off the logged point uses the organisers' kernel.** The point must be
  strictly nearer the GT target than every other object or person in that camera, then
  1 within `R_ACQUIRE` falling linearly to 0 at `R_MISS` (`stage1_runs.target_credit`).
  The constants and the distractor margin are `MARC_SCORING_*` and private at competition
  time, so never tune to the defaults; `stage1_sensitivity.py` replays a grid of them.
  A replayed target given credit further out than anything the platform credited for that
  round is counted as uncertain in the summary.

## Camera geometry

`data/sdk_cameras.json` is K, the world-from-camera rotation and translation, and
`ground_height` of the 6 rigs, captured from the live runtime `Tf2ClientLookup` on
2026-10-07 (practice platform image `sha256:a8dbb5ad…`). It equals the generator
camera blocks to 4 decimals (#87). Pass `--geometry` to replay another capture.
