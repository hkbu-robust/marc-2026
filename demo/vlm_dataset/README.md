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

`TRAINER_SAR_RECIPES=1` is interpolated in
`simulation-platform/docker-compose.yml`, so a prefixed `up` matches
`docker compose run -e`. It cycles the 14 practice archetypes in
`simulation-platform/patches/sar_recipes.yaml` (one target + ordinary
distractors per scene). A restarted run continues the cycle from
`max(scene_*)+1` (or `TRAINER_SAR_RECIPE_CURSOR` if set). Every recipe
packs the target person with its support (and parking extras such as a
taxi) into camera midfield, then skip-saves unless that pair is in frame
and within 2.8 m XY. Climb recipes (`above: true`: trash/vending sit)
also freeze the stack on the roof; do **not** set `TRAINER_PEOPLE_ABOVE`
for a mixed recipe run. Seat-lie recipes keep `on_landmark_type` even when
above-mode is on. Beside recipes put the person on the ground 1.2–2.0 m from
the named support. Parking vs grass is a **placement proxy**: parking recipes
anchor beside a taxi (and tip the bicycle when that is the support); park
recipes stay beside trash. The converter stamps `support_class` /
`surface_class` / `recipe_id` only after that place.

Then from the repo root:

```bash
python3 demo/vlm_dataset/cctv_gt_to_qwen.py \
  --in trainer_output/marc2026_chungmu \
  --out trainer_output/qwen2vl_cctv
```

The converter writes **separate** JSONL files so a Qwen LoRA is not trained on
two grammars:

- `train_sar.jsonl` / `val_sar.jsonl` — SAR verifier prompt (same string as
  runtime `format_sar_verifier_prompt`) and `{id}|{situation}`
- `train_obj.jsonl` / `val_obj.jsonl` — unique-relation object JSON without xyz

Fine-tune the SAR LoRA on the SAR files only. A camera with a recipe scene but
no recipe target in frame emits **no** SAR row (walkers-only frames are not
trained as “find the climber”). If GT `support_class` names a landmark that is
not within 2.5 m XY, those GT fields are ignored and geometry is used.

The converter prints `hard=` / `hard_frac=` (same-class object clones that
still uniquely identify), SAR situation counts, and per-class histograms.

SAR rows use the R03 pose×support matrix (`marc_agent.perception.situation_rules`,
version 2026.1). Assistant text is **only** `{camera_id}:person:{n}|{situation}`
(no coordinates). Picnic-table lying/prone is `abnormal`; crouch is `abnormal`
everywhere; unlabeled ground `person_lying_down` (no `surface_class`) is
dropped; `person_prone` is always `accident`. Parking-proxy frames
(`sar06` / `sar07` / `sarH6`) still emit a SAR row when the target is in view.

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
