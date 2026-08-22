# CCTV → Qwen2-VL instruction set

Host-side converter. **Does not** start Isaac. `dataset-gen` writes detector GT
(`scene_XXXXX/{camera}.png` + `.json`); this turns that into chat turns for
Qwen2-VL-Instruct / LLaMA-Factory.

Generate with a Qwen-oriented mix (object copies 4-6, ~35% of non-vehicle
landmark types per scene, vehicles rare). From `simulation-platform/`:

```bash
TRAINER_CLASS_INCLUSION_PROBABILITY=0.35 TRAINER_AUTO_SCENES=100 HEADLESS=true \
  docker compose --profile dataset-gen up --force-recreate
```

Then from the repo root:

```bash
python3 demo/vlm_dataset/cctv_gt_to_qwen.py \
  --in trainer_output/marc2026_chungmu \
  --out trainer_output/qwen2vl_cctv
```

The converter prints `hard=` / `hard_frac=` (same-class object clones),
SAR situation counts, and per-class landmark/object histograms.

SAR rows use `target_type: person` and `situation` (`person_sitting` is always
`normal`). Object rows still use `relation`. Same dump also converts to YOLO:

```bash
python3 demo/vlm_dataset/cctv_gt_to_yolo.py \
  --in trainer_output/marc2026_chungmu \
  --out trainer_output/yolo_cctv
```

Use the same `--val-frac` on both so train/val scenes match. YOLO classes are
poses (`person_crouching`), not identity meshes. Do **not** mix ego-cam frames.

Assistant text is a `GroundingResult` JSON (`camera_id`, `target_type`,
`landmark`, `relation`, `anchor_coord`, `target_coord`) so it matches
`client.submit_grounding`. Same-class object clones in one frame are preferred
(hard / disambiguation rows). Landmark copies are not required for that signal.

New `dataset-gen` runs also stamp `world_xyz` on each detection (boot patch
`patch_gt_world_xyz.py`). Older dumps without that field use a ground-plane
back-projection from the box centre and camera extrinsics.

Do **not** mix this JSONL with ego-cam YOLO images or OpenVLA pick episodes.
