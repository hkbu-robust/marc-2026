# Paper visualization shortlist

Use these as the paper-ready shortlist; the larger source galleries remain in `docs/luka/`.

## Recommended figures

1. `pipeline-overview.svg` — compact Stage 1/Stage 2 method overview. The dashed return-to-owner block is intentionally marked pending.
2. `stage1-cctv-detections.jpg` — full-frame Stage 1 CCTV detections across four cameras.
3. `stage2-nearfield-detections.jpg` — target-conditioned robot-view detections for tumbler, cracker box, mug, and umbrella.
4. `stage1-detector-comparison.jpg` and `stage2-detector-comparison.jpg` — exact same-frame model comparisons with processed ground truth.
5. `stage1-pr-curve.png`, `stage2-pr-curve.png`, and the two training-curve files — quantitative detector figures.

## Safe caption language

- **Stage 1:** “Qualitative detections from the simulation-derived CCTV validation set. Boxes show full-frame YOLO candidates across four fixed-camera views.”
- **Stage 2:** “Target-conditioned detections from the simulation-derived robot RGB validation set. Each panel requests one object class.”
- **Comparisons:** Say “validation frame,” not “competition run” or “end-to-end practice run.”

## Provenance and current gap

The images were curated from the readiness-report assets in `docs/luka/`, which in turn select experiment outputs from `../marc-2026-work/`. The successful Chungmu practice result in `results/marc2026_chungmu/u1_20260829_181730.json` retained scores and coordinates but no camera frames. The next successful rehearsal should supply the missing paper figure: one Stage 1 selected-target/anchor overlay with world XYZ and a Stage 2 sequence showing approach, RGB-D detection, grasp, basket confirmation, and delivery.

Round-image dumping is already implemented and enabled by default in `marc_agent/perception/round_dump.py`; preserve `artifacts/rounds/stage1/` and `artifacts/rounds/stage2/` after that run.
