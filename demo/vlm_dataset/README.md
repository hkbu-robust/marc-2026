# CCTV → Qwen2-VL instruction set

Host-side converter. **Does not** start Isaac. `dataset-gen` writes detector GT
(`scene_XXXXX/{camera}.png` + `.json`); this turns that into chat turns for
Qwen2-VL-Instruct / LLaMA-Factory.

```bash
# after dataset-gen (with TRAINER_MIN/MAX_INSTANCES 4-6) has finished:
python3 demo/vlm_dataset/cctv_gt_to_qwen.py \
  --in trainer_output/marc2026_chungmu \
  --out trainer_output/qwen2vl_cctv
```

Assistant text is a `GroundingResult` JSON (`camera_id`, `target_type`,
`landmark`, `relation`, `anchor_coord`, `target_coord`) so it matches
`client.submit_grounding`. Same-class clones in one frame are preferred
(hard / disambiguation rows).

New `dataset-gen` runs also stamp `world_xyz` on each detection (boot patch
`patch_gt_world_xyz.py`). Older dumps without that field use a ground-plane
back-projection from the box centre and camera extrinsics.

Do **not** mix this JSONL with ego-cam YOLO images or OpenVLA pick episodes.
