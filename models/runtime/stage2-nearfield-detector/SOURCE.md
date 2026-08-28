# Stage 2 near-field detector

This bundle is the selected provisional robot-view exact-target detector for
the MARC 2026 near-field acquisition path.

## Artifact

- Format: inference-only Ultralytics YOLO26x PyTorch checkpoint (`model.pt`)
- Deployment artifact SHA-256:
  `3092b0c64f5d543f20a8dd6fa0bc9ce645298be9b36722dc39f7878e4ee58232`
- Selected training checkpoint: `epoch40.pt` from run
  `stage2-roboview-yolo26x-1920-provisional-20260828`
- Selected training checkpoint SHA-256:
  `931e1824a4cf08cb5865f112d9405e83b936d14616d067aace5c98b78488d146`
- The deployment copy has optimizer, scaler, EMA, and resume state stripped.
  The external run retains the full selected checkpoint.
- No adapter or external base checkpoint is required at inference time.

## Training provenance

- Stage 1 seed: `stage1-cctv-yolo26x-1920-20260824/weights/last.pt`
- Dataset revision: `6723cca004d05de4cde72827923941605502fdcc`
- Source descriptor SHA-256:
  `28827a048bbe1adc80f790dbd3411f1baacb7a44ebc59dbe9328dfae1edfbfa5`
- Derived descriptor SHA-256:
  `1b1c9cabc73648c22b939a26f2b47a73f5a4e0d03f617978706763046e017992`
- Training code commit: `d3f0be5261a8d33cd0fd3b6df4a4cb60ff319c9b`
- Training: 100 epochs, batch 4, 1920 px, seed 0, AMP, `rect=True`
- Peak recorded training memory: 37.6230 GiB

## Selection and validation

All 12 saved checkpoints were screened at 1920 px through exact-target
inference. `epoch40.pt` won the declared macro-F1, worst-class-recall,
tiny-recall, small-recall, overall-recall hierarchy. Detailed validation then
selected the one-to-many/NMS head, confidence 0.3, NMS IoU 0.5, top 10, and
1920 px.

The operational validation result is macro F1 0.9558, overall recall 0.9733,
tiny recall 0.9552, and worst-class recall 0.9231. The selected-head generic
validation result is mAP50 0.9812 and mAP50-95 0.8295.

The real near-field adapter was warmed for five calls and timed for 30 batch-1
calls on an NVIDIA RTX PRO 6000 Blackwell Max-Q. Mean / median / p95 wall time
was 19.50 / 19.49 / 19.71 ms, with 0.5913 GiB peak allocated and 1.125 GiB
peak reserved CUDA memory.

## Runtime

Use `detect_candidates(rgb, target_type)`. It returns confidence-descending
candidates for only the requested class; the legacy `detect()` wrapper returns
the best candidate. The competition runtime is offline, so load the explicit
local `model.pt` path and include the pinned runtime dependencies in the image.

## Required limitations

- Ordinary umbrella is absent; this provisional dataset covers nine of ten
  scored classes.
- No independent test split exists.
- No reliable scenario, range, or bearing manifest exists.
- The same validation split is used for selection and readiness reporting.
- This is a provisional nine-class integration model, not final
  challenge-readiness evidence.
- Empty-label intent metadata is absent; empty-label frames are not known
  intentional negatives.

## Licensing note

Ultralytics documents its framework and model weights as AGPL-3.0. Confirm that
the team can comply with AGPL-3.0 or has an appropriate Ultralytics enterprise
license before submission.
