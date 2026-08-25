# CCTV → Qwen2-VL instruction set

Host-side converter. **Does not** start Isaac. `dataset-gen` writes detector GT
(`scene_XXXXX/{camera}.png` + `.json`); this turns that into chat turns for
Qwen2-VL-Instruct / LLaMA-Factory.

Generate with recipe-driven SAR scenes plus object copies (4–6, umbrellas
skipped to match the 33-class runtime contract, ~35% of non-vehicle landmark
types per scene). From `simulation-platform/`:

```bash
TRAINER_SAR_RECIPES=1 TRAINER_PEOPLE_ALL_CHARS=1 \
  TRAINER_CLASS_INCLUSION_PROBABILITY=0.35 TRAINER_AUTO_SCENES=100 HEADLESS=true \
  docker compose --profile dataset-gen up --force-recreate
```

`TRAINER_SAR_RECIPES=1` cycles the 14 practice archetypes in
`simulation-platform/patches/sar_recipes.yaml` (one target + ordinary
distractors per scene, including no-landmark parking prone/collapse and
fallen-bicycle proximity). Seat pose pools include `person_prone` so
bench/table lie recipes can spawn. Parking vs grass is a recipe field stamped
onto person detections as `support_class` / `surface_class` / `recipe_id`.

Then from the repo root:

```bash
python3 demo/vlm_dataset/cctv_gt_to_qwen.py \
  --in trainer_output/marc2026_chungmu \
  --out trainer_output/qwen2vl_cctv
```

The converter prints `hard=` / `hard_frac=` (same-class object clones that
still uniquely identify), SAR situation counts, and per-class histograms.

SAR rows use the R03 pose×support matrix (`marc_agent.perception.situation_rules`,
version 2026.1). Assistant text is **only** `{camera_id}:person:{n}|{situation}`
(no coordinates). The user turn is the spoken command plus the finite candidate
ID list. Picnic-table lying/prone is `abnormal`; crouch is `abnormal`
everywhere; unlabeled ground `person_lying_down` (no `surface_class`) is
dropped; `person_prone` is always `accident`. Frames with no landmark
(`sar06` / `sar07` / `sarH6`) still emit a SAR row.

Object rows keep eight-relation referring expressions without grounding xyz,
and only when the chosen landmark/relation uniquely identifies that instance
(ambiguous nearest-2D `near` / `beside` among same-class clones is skipped).
Umbrellas stay skipped.

Same dump also converts to YOLO (object-detection training export; **no**
situation labels; same `--val-frac` / scene grouping):

```bash
python3 demo/vlm_dataset/cctv_gt_to_yolo.py \
  --in trainer_output/marc2026_chungmu \
  --out trainer_output/yolo_cctv
```

YOLO classes are poses (`person_crouching`) and the canonical 33 object /
landmark names, not identity meshes. Do **not** mix ego-cam frames.

New `dataset-gen` runs stamp `world_xyz`, `support_class`, `surface_class`,
and `recipe_id` on person detections (boot patch `patch_gt_world_xyz.py`).
Older dumps without those fields fall back to geometry (`on` / `above` /
fallen-vehicle `beside`) then the pose-only table.

Do **not** mix this JSONL with ego-cam YOLO images or OpenVLA pick episodes.
This converter does not train Qwen.
