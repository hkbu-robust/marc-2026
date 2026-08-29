# Stage 2 near-field detector

This bundle is the selected provisional ten-class robot-view exact-target
detector for the MARC 2026 near-field acquisition path. Human review remains
pending; this bundle does not close issue #16 or establish final challenge
readiness.

## Artifact

- Format: inference-only Ultralytics YOLO26x PyTorch checkpoint (`model.pt`)
- Deployment artifact SHA-256:
  `ff8e156b65ee556e2eae31a5fee94f8a3056d9dc92c771c1be48762cea7f6cc7`
- Selected training checkpoint: `best.pt` from run
  `stage2-roboview-yolo26x-1920-umbrella-e52c3156-20260829`
- Selected training checkpoint SHA-256:
  `1f8fe71bd25b1132cd1285e27998411741e725764502ed50b63adef0c3120d78`
- The deployment copy has optimizer, scaler, EMA, and resume state stripped.
  The external run retains the selected training checkpoint.
- No adapter or external base checkpoint is required at inference time.

## Training provenance

- Stage 1 seed: `stage1-cctv-yolo26x-1920-umbrella-ceccc62/weights/last.pt`
- Stage 1 seed SHA-256:
  `12712d12e3321dffeb49296285a7d99f8fab12b46aa99a808b5d3cd1da80da22`
- Dataset revision: `e52c3156f004f6b07bdec044465c9d1be59ad4f3`
- Source descriptor SHA-256:
  `ee810912b28da56806cdd3b768d2f8cb76373950e7d1784310bdc7d9f456dd6a`
- Derived descriptor SHA-256:
  `cdf6bbd2e0292f5352e084749711244454f782c7db23234b8c422d1674aaca78`
- Training code commit: `7446b48f5dd2a7b6cd242a2fef8440a1b17a8033`
- Training: 100 epochs, batch 4, 1920 px, seed 0, AMP, `rect=True`
- Peak recorded training memory: 37.6309 GiB

## Selection and validation

All 12 saved checkpoints were screened at 1920 px through exact-target
inference. `best.pt` won the declared macro-F1, worst-class-recall,
tiny-recall, small-recall, overall-recall hierarchy. Detailed validation then
selected the one-to-many/NMS head, confidence 0.20, NMS IoU 0.5, top 10, and
1920 px.

The operational validation result is macro F1 0.9529, overall recall 0.9567,
tiny recall 0.9137, and worst-class recall 0.7692 (`sunblock`). Ordinary
umbrella recall is 0.9423 (49/52), F1 is 0.9245, AP50 is 0.9809, and AP50-95
is 0.8639. The selected-head generic validation result is mAP50 0.9773 and
mAP50-95 0.8609.

Against the previous nine-class Stage 2 checkpoint on the same revised
validation frames and selected settings, `cracker_box` recall decreased by
8.00 percentage points and `sunblock` recall decreased by 19.23 points. These
are reported as material regressions. Ordinary umbrella is unsupported by the
baseline and is not counted as a miss.

The real near-field adapter was warmed for five calls and timed for 30 batch-1
calls on an NVIDIA RTX PRO 6000 Blackwell Max-Q. Mean / median / p95 wall time
was 19.17 / 19.16 / 19.24 ms, with 0.5913 GiB peak allocated and 1.125 GiB
peak reserved CUDA memory.

## Runtime

Use `detect_candidates(rgb, target_type)`. It returns confidence-descending
candidates for only the requested class; the legacy `detect()` wrapper returns
the best candidate. The competition runtime is offline, so load the explicit
local `model.pt` path and include the pinned runtime dependencies in the image.

## Required limitations

- No independent test split exists.
- Train and validation share scenario/background families, so the split is weak.
- The same validation split is used for model selection and readiness reporting.
- Empty-label intent metadata is absent; intentional negatives cannot be separated from unknown empties.
- Human blind review is pending.
- This ten-class replacement candidate does not close issue #16 or establish final challenge readiness.

## Licensing note

Ultralytics documents its framework and model weights as AGPL-3.0. Confirm that
the team can comply with AGPL-3.0 or has an appropriate Ultralytics enterprise
license before submission.
