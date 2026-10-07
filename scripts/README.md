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
- **Target credit off the logged point is a model.** The scorer is binary at about
  1.9 m in the #88 rounds but may match the nearest object (#91). A replayed target
  credited further out than anything the platform credited for that round is
  counted as uncertain in the summary.

## Camera geometry

`data/sdk_cameras.json` is K, the world-from-camera rotation and translation, and
`ground_height` of the 6 rigs, captured from the live runtime `Tf2ClientLookup` on
2026-10-07 (practice platform image `sha256:a8dbb5ad…`). It equals the generator
camera blocks to 4 decimals (#87). Pass `--geometry` to replay another capture.
