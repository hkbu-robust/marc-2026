# Stage 1 CCTV detector

This bundle is the selected full-frame detector for MARC 2026 Stage 1 CCTV
object, landmark, and person/pose candidates.

## Artifact

- Format: inference-only Ultralytics YOLO26x PyTorch checkpoint (`model.pt`)
- Deployment artifact SHA-256:
  `8c6e69d3f8b9df6e06cba59c904f62489497037e930e9e94727d01bcde96a15a`
- Selected training checkpoint: `best.pt` from run
  `stage1-cctv-yolo26x-1920-umbrella-ceccc62`
- Selected training checkpoint SHA-256:
  `8c6e69d3f8b9df6e06cba59c904f62489497037e930e9e94727d01bcde96a15a`
- Ultralytics stripped optimizer, scaler, EMA, and resume state from the
  selected checkpoint. `model.pt` preserves the selected checkpoint bytes.
- No adapter or external base checkpoint is required at inference time.

## External archive

This replacement has not yet been mirrored to the private Hugging Face model
repository. Its previous immutable revision contains the superseded 33-class
checkpoint and must not be used for this bundle.

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

- Dataset revision: `ceccc62d27e7fc186fd186e66632da38aeac6668`
- Source dataset descriptor SHA-256:
  `a9277d45dfaeefa3aaec34f6cf82218c4b362dd7b19c06271f8dc6ff417b1f9a`
- Training code commit: `4010e06cce663c0c8c9ff3617731511ffd89b5a4`
- Classes: 34 ordered classes recorded in `runtime_contract.json`, including
  `umbrella` at class ID 4
- Excluded classes: `long_umbrella`, `master_chef_can`, and `wood_block`
- Dataset: 3,085 train images / 122,189 boxes and 343 validation images /
  13,485 boxes across six CCTV cameras
- Input geometry: native 1280x720 frames, rectangular letterbox to
  `imgsz=1920`
- Training: 100 epochs (18,720.1 seconds), batch 4, seed 0, AMP, `rect=True`,
  `scale=0.2`, no mosaic/mixup/cutmix
- Resolved optimizer: AdamW, initial learning rate 0.000263
- Framework: Python 3.11.15, PyTorch 2.13.0+cu130,
  `ultralytics-opencv-headless` 8.4.118
- Peak recorded training memory: 28.4629 GiB
- Determinism note: Ultralytics warns that `cache=ram` can make otherwise
  deterministic training non-deterministic.

## Selection and validation

Ultralytics selected `best.pt` at epoch 92 on the 343-image, six-camera
validation split. Metrics embedded in the selected checkpoint are:

- Precision: 0.96806
- Recall: 0.95277
- mAP50: 0.97695
- mAP50-95: 0.90205

These are training-time validation metrics, not an independent challenge
holdout. The runtime retains the previously selected one-to-many/NMS settings:
confidence 0.3, NMS IoU 0.5, at most 50 candidates, and FP16 inference on CUDA.

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
