# Stage 1 CCTV detector

This bundle is the selected full-frame detector for MARC 2026 Stage 1 CCTV
object, landmark, and person/pose candidates.

## Artifact

- Format: inference-only Ultralytics YOLO26x PyTorch checkpoint (`model.pt`)
- Deployment artifact SHA-256:
  `06a0abac38a897c0d3c72392154699305a0855bbc12779c6a7bdfff7fe803e2f`
- Selected training checkpoint: `last.pt` from run
  `stage1-cctv-yolo26x-1920-20260824`
- Selected training checkpoint SHA-256:
  `cf1d3b7975af147b9406c12baf7dbb64333ef1d5af5620e114325af99a90b55f`
- Ultralytics had already stripped optimizer, scaler, EMA, and resume state from
  the run's final `last.pt`. The external checkpoint archive retains its exact
  selected bytes and also keeps the unselected full-state `epoch70.pt` for a
  possible training resume.
- No adapter or external base checkpoint is required at inference time.

## Permanent archive

- Private Hugging Face model repository:
  <https://huggingface.co/roBUst-marc-2026/stage1-cctv-detector>
- Immutable release revision:
  `9227ecd11069d103c7dabdd287c6fc20a83eb6c1`

## Upstream

- Model: COCO-pretrained Ultralytics `yolo26x.pt`
- Immutable release: `ultralytics/assets` tag `v8.4.0`
- Source asset:
  <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt>
- Source checkpoint SHA-256:
  `9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92`
- Source checkpoint metadata: Ultralytics `8.3.222`, dated 2025-12-30
- License: AGPL-3.0; the bundled `LICENSE` is copied from the exact installed
  Ultralytics distribution used for selection.

## Training provenance

- Dataset revision: `075767315653d3286fb636e9f278397ae73d84c9`
- Source dataset descriptor SHA-256:
  `88dee1597d7a25d7b1dc23bc7814da3669ee28e750ccf22a42c15499eca1d66c`
- Training code commit: `b2afe15524c6ff42f7b9db1eb049d83e4424bf43`
- Classes: 33 ordered classes recorded in `runtime_contract.json`
- Ignored organizer-excluded classes: `master_chef_can`, `wood_block`,
  `umbrella`, and `long_umbrella`
- Input geometry: native 1280x720 frames, rectangular letterbox to
  `imgsz=1920`
- Training: 100-epoch maximum, batch 4, seed 0, AMP, `rect=True`,
  `scale=0.2`, no mosaic/mixup/cutmix
- Resolved optimizer: AdamW, initial learning rate 0.00027
- Framework: Python 3.11.15, PyTorch 2.13.0+cu130,
  `ultralytics-opencv-headless` 8.4.118
- Peak recorded training memory: 28.3066 GiB

## Selection and validation

All 12 saved checkpoints were screened on the 415-image, six-camera validation
split with one one-to-many/NMS IoU 0.7 top-50 inference pass each. Confidence
and candidate limits were scored from those cached predictions; `last.pt` had
the highest mean per-class F1. Only that winner received the complete two-head
and runtime-setting evaluation.

At equal confidence 0.001 and 20 candidates per image, the one-to-many head had
0.7679 macro recall versus 0.7489 for the end-to-end head. The selected runtime
configuration is one-to-many/NMS with confidence 0.3, NMS IoU 0.5, and at most
50 candidates per image, using FP16 inference on CUDA.

Selected-head validation:

- mAP50-95: 0.9232
- AP validation recall: 0.9721
- Operational recall at selected runtime settings: 0.9764
- Tiny/small operational recall: 0.9695 / 0.9870
- Worst-class operational recall: 0.9354
- Worst-camera operational recall: 0.9682
- Mean measured inference time: 10.70 ms/image on an NVIDIA RTX PRO 6000
  Blackwell Max-Q
- Peak measured inference memory: 2.7637 GiB

The complete per-class, per-camera, size-bucket, checkpoint-screening, latency,
and candidate-setting results remain in the external run's
`evaluation_report.json`.

## Runtime

Load `model.pt` only through the local path recorded by this bundle. Apply the
settings and class order in `runtime_contract.json`. The example output records
the candidate interface expected by the downstream Stage 1 runtime.

The competition runtime is offline. Set `YOLO_OFFLINE=1`, include the exact
runtime dependencies in the participant image, and do not resolve model names
through a network-backed cache.

## Licensing note

Ultralytics documents its framework and model weights as AGPL-3.0. A private
competition repository may not by itself satisfy those terms. Confirm that the
team can comply with AGPL-3.0 or has an appropriate Ultralytics enterprise
license before submission.
