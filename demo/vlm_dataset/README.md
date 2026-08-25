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

SAR rows use `target_type: person` and `situation` from **pose + support**,
not pose alone. Sitting on a bench is `normal`; sitting, reaching, or standing
on a vending machine / trash can / hydrant / postbox is `abnormal` (prompts
include “climbed on top of the drink machine” and pose-specific variants).
Hanging-leg and full-frame sitters are labelled from `world_xyz` when the 2D
box shares the landmark’s ground line. Object rows use `relation`
(`beside` / `near` / `on` / `above` / `left` / `right` / `front` / `behind`).

At runtime: train YOLO on the same dump (`cctv_gt_to_yolo.py`, pose/object
class names). Qwen returns a `GroundingResult`; match `target_type` /
`landmark` to YOLO boxes of those classes and use the coords.

dataset-gen must actually put a climber **on** the landmark (`TRAINER_PEOPLE_ABOVE=1`)
or the converter cannot invent the visual:

```bash
TRAINER_PEOPLE_ABOVE=1 TRAINER_PEOPLE_ABOVE_TYPES=vending_machine,trash_can,hydrant,postbox \
  TRAINER_CLASS_INCLUSION_PROBABILITY=0.35 TRAINER_PEOPLE_ALL_CHARS=1 \
  TRAINER_AUTO_SCENES=100 HEADLESS=true \
  docker compose --profile dataset-gen up --force-recreate
```

Same dump also converts to YOLO:

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
