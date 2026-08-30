# CCTV → Qwen3-VL instruction set

Host-side converter. **Does not** start Isaac. `dataset-gen` writes detector GT
(`scene_XXXXX/{camera}.png` + `.json`); this turns that into chat turns for
Qwen3-VL-8B-Instruct / LLaMA-Factory.

Recommended one-shot profile (420 isolated scenes by default, 30 balanced
cycles of every SAR recipe, then strict export):

```bash
bash simulation-platform/marc.sh qwen-dataset-gen
```

Ready JSONL and its coverage report are written to
`trainer_output/qwen3vl_lora/`; raw GT stays separately in
`trainer_output/qwen3vl_gt/`. Override the size with
`QWEN_DATASET_SCENES=840`. The export exits nonzero and lists missing object,
landmark, relation, SAR-recipe, situation, or camera coverage instead of
silently producing an incomplete fine-tuning set. Re-run the same command to
append fresh scenes until the report is complete.

Generate with recipe-driven SAR scenes plus object copies (4–6,
`long_umbrella` / `master_chef_can` / `wood_block` skipped, ~35% of
non-vehicle landmark types per scene).
From `simulation-platform/`:

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

The converter writes separate JSONL files so SAR and object validation remain
inspectable, but both use the production finite-board format and belong in the
same Qwen LoRA run:

- `train.jsonl` / `val.jsonl` — shuffled combined corpus (use these for LoRA)
- `train_sar.jsonl` / `val_sar.jsonl` — SAR verifier prompt (same string as
  runtime `format_sar_verifier_prompt`) and the runtime `Pxx` answer token
- `train_obj.jsonl` / `val_obj.jsonl` — packed annotated object board and exact
  `TARGET_ID`
- `dataset_info.json` — LLaMA-Factory registrations for `marc_qwen_train` and
  `marc_qwen_val`; `coverage.json` retains the generator coverage report

Start LoRA training from the repo root (the optional argument is the dataset folder):

```bash
training/train_qwen_lora.sh trainer_output/qwen3vl_lora
```

The script defaults to the local model used by `useful-commands/vllm launch`.
Override it with `MARC_QWEN_MODEL_DIR` and the output with
`QWEN_LORA_OUTPUT_DIR`.

Fine-tune on both training files. A camera with a recipe scene but
no recipe target in frame emits **no** SAR row (walkers-only frames are not
trained as “find the climber”). If GT `support_class` names a landmark that is
not within 2.5 m XY, those GT fields are ignored and geometry is used.

The converter prints `hard=` / `hard_frac=` (same-class object clones that
still uniquely identify), SAR situation counts, and per-class histograms.

SAR rows use the R03 pose×support matrix (`marc_agent.perception.situation_rules`,
version 2026.1). Assistant text is **only** the numbered runtime board alias
`Pxx` (no coordinates or explanation). Picnic-table lying/prone is `abnormal`; crouch is `abnormal`
everywhere; unlabeled ground `person_lying_down` (no `surface_class`) is
dropped; `person_prone` is always `accident`. Parking-proxy frames
(`sar06` / `sar07` / `sarH6`) still emit a SAR row when the target is in view.

Object rows keep eight-relation referring expressions without grounding xyz,
and only when the chosen landmark/relation uniquely identifies that instance
(ambiguous nearest-2D `near` / `beside` among same-class clones is skipped).
Ordinary `umbrella` object rows are kept; `long_umbrella`, `master_chef_can`,
and `wood_block` are skipped.

Same dump also converts to YOLO (object-detection training export; **no**
situation labels; same `--val-frac` / scene grouping). **Do not convert a
SAR-recipe dump as detector gold.** Generate ordinary CCTV with
`TRAINER_YOLO_MODE=1` (bench/table seats only; injects `umbrella` if the GUI
catalog omitted it; does not spawn chef/wood; wins over recipes and
`TRAINER_PEOPLE_ABOVE`):

```bash
ENV_MARC_SCENARIO=marc2026_chungmu TRAINER_YOLO_MODE=1 TRAINER_PEOPLE_ALL_CHARS=1 \
  TRAINER_SAR_RECIPES=0 TRAINER_PEOPLE_ABOVE=0 \
  TRAINER_AUTO_SCENES=4803 HEADLESS=true \
  docker compose --profile dataset-gen up --force-recreate
```

Then from the repo root:

```bash
python3 demo/vlm_dataset/cctv_gt_to_yolo.py \
  --in trainer_output/marc2026_chungmu \
  --out trainer_output/yolo_cctv
```

The converter writes `umbrella` boxes, drops `long_umbrella`, `master_chef_can`,
and `wood_block` (`wooden_block` aliases to `wood_block` then drops), and drops
detections with `occlusion >= 0.85`. Train a new corpus without rewriting the
archived 33-class pins:

`python3 training/train_cctv_detector.py --no-verify-hash ...`

YOLO classes are poses (`person_crouching`) and the canonical object /
landmark names, including `umbrella`. The training job drops `long_umbrella`,
`master_chef_can`, and `wood_block`. Do **not** mix ego-cam frames.

New `dataset-gen` runs stamp `world_xyz`, `support_class`, `surface_class`,
and `recipe_id` on person detections (boot patch `patch_gt_world_xyz.py`).
Older dumps without those fields fall back to geometry (`on` / `above` /
fallen-vehicle `beside`) then the pose-only table.

Do **not** mix this JSONL with ego-cam YOLO images or OpenVLA pick episodes.
This converter does not train Qwen.
