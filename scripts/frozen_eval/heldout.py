"""Held-out evaluation set from fresh generator scenes (#87).

    python -m scripts.frozen_eval.heldout specs --gt GT_DIR --training-record JSON --out SETS/heldout-build
    # detect_pass.py inside the frozen agent image -> SETS/heldout-build/detections.jsonl
    python -m scripts.frozen_eval.heldout assemble --build SETS/heldout-build --root SETS
    python -m scripts.frozen_eval.heldout audit --root SETS --out SETS/label-audit      # before freeze
    python -m scripts.frozen_eval.heldout audit-score --sheet SETS/label-audit/audit.csv

``specs`` turns generator GT (``qwen-dataset-gen`` into a new empty folder; one
populated camera per scene) into one spec per target that a command describes
uniquely:

- **SAR**: the recipe target with the recipe command, kept only if it passes
  ``relations.sar_check`` (pose, support relation, no same-pose rival).
- **Objects**: at most one case per scene (scenes are the clusters, and every
  object in a scene would give ~19 cases). Options are the (target, relation,
  landmark class) triples that pick out one instance under ``relations`` (our
  rule); the relation used least so far wins, then the least-used target
  class, then a same-class distractor, so relations and classes stay balanced. The command must
  parse back to the same target, relation and landmark.

Strata come from the deployed adapter's training record (``training_record``):
``unseen`` = 0 training rows for the recipe (SAR) or relation (objects),
``rare`` = fewer than ``RARE_MAX``, else ``seen``. ``absent_from_original_sft``
keeps the issue's first definition (no rows in the original 800-scene SFT).

Leakage: training rows keep camera and target box but no world position, so the
target box bottom-centre is back-projected with the fixed rig geometry on both
sides. A held-out case is a near-duplicate, and is dropped, when a training row
of the same camera asks the same question within ``LEAK_M`` of the target:

- SAR: same recipe (a recipe fixes command, pose and support);
- objects: same target class and same landmark class. Matching on class alone
  would drop 60%: thousands of training rows cover each fixed camera's ground,
  so a fresh random placement is almost always within 0.5 m of an old one of
  the same class, while the scene around it is new. ``nearest_same_class_m``
  keeps that stricter distance for sensitivity checks.

``assemble`` joins the detector pass: gold IDs are detections of the target
class (any pose for people) with IoU >= ``GOLD_IOU`` against the GT box.
``detector_miss`` cases stay in the set and are reported apart.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import yaml

from marc_agent.command_parser import MISSION_OBJECT_TYPES, CommandParseError, parse_command
from scripts.frozen_eval import case_store
from scripts.frozen_eval.geometry import geometry_from_gt, ground_point
from scripts.frozen_eval.relations import (
    RELATIONS,
    RULE_VERSION,
    is_person,
    object_choices,
    object_command,
    sar_check,
)

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "simulation-platform" / "patches" / "sar_recipes.yaml"
SKIPPED_OBJECTS = frozenset({"long_umbrella", "master_chef_can", "wood_block"})  # detector ignores
RARE_MAX = 25
LEAK_M = 0.5
GOLD_IOU = 0.5


def _scene_payloads(gt_dir: Path) -> list[tuple[int, Path, dict]]:
    out = []
    for scene_dir in sorted(gt_dir.glob("scene_*")):
        for json_path in sorted(scene_dir.glob("rig_*.json")):
            if (json_path.with_suffix(".png")).is_file():
                out.append((int(scene_dir.name.split("_")[1]), json_path,
                            json.loads(json_path.read_text(encoding="utf-8"))))
    return out


def rig_geometries(payloads: list[tuple[int, Path, dict]]) -> dict[str, dict]:
    """One camera block per rig; raise if a rig moved between scenes."""
    rigs: dict[str, dict] = {}
    for _, path, payload in payloads:
        block = {"camera": payload["camera"], "image": payload["image"]}
        known = rigs.setdefault(payload["frame_id"], block)
        if known != block:
            raise ValueError(f"camera {payload['frame_id']} differs between scenes ({path})")
    return rigs


def stratum(count: int) -> str:
    if count == 0:
        return "unseen"
    return "rare" if count < RARE_MAX else "seen"


class LeakIndex:
    """Training targets as ground points, keyed by (camera, kind, recipe-or-class, landmark)."""

    def __init__(self, rows: list[dict], rigs: dict[str, dict]):
        self.points: dict[tuple, list[tuple[float, float]]] = collections.defaultdict(list)
        self.skipped = collections.Counter()
        geoms = {cam: geometry_from_gt(block) for cam, block in rigs.items()}
        for row in rows:
            label = row["recipe"] if row["kind"] == "sar" else row["target_type"]
            if label is None or row["camera"] not in geoms:
                self.skipped["no_label" if label is None else "unknown_camera"] += 1
                continue
            point = ground_point(geoms[row["camera"]], row["bbox"])
            if point is None:
                self.skipped["no_ground_point"] += 1
                continue
            self.points[(row["camera"], row["kind"], label, row.get("landmark"))].append(point[:2])
            if row["kind"] == "obj":
                self.points[(row["camera"], row["kind"], label, "*")].append(point[:2])

    def nearest_m(self, camera: str, kind: str, label: str, landmark, point) -> float | None:
        """Planar distance to the nearest training target; ``landmark="*"`` ignores the landmark."""
        pts = self.points.get((camera, kind, label, landmark))
        if not pts or point is None:
            return None
        return min(math.hypot(point[0] - x, point[1] - y) for x, y in pts)


def _det_summary(det: dict) -> dict:
    return {k: det.get(k) for k in ("class", "instance_id", "bbox", "world_xyz", "occlusion")}


def _parsed_matches(command: str, target: str, relation: str | None, landmark: str | None) -> dict | None:
    try:
        parsed = parse_command(command)
    except CommandParseError:
        return None
    rel = parsed.relation.value if parsed.relation else None
    if parsed.target_type != target or (relation and rel != relation) or (
            landmark and parsed.landmark != landmark):
        return None
    return json.loads(json.dumps(asdict(parsed), default=str))


def build_specs(gt_dir: Path, record: dict) -> tuple[list[dict], list[dict], dict]:
    payloads = _scene_payloads(gt_dir)
    rigs = rig_geometries(payloads)
    recipes = {r["id"]: r for r in yaml.safe_load(RECIPES.read_text(encoding="utf-8"))["recipes"]}
    train = record["train_counts"]
    original = record["original_train_counts"]
    leaks = LeakIndex(record["rows"], rigs)
    used_relations = collections.Counter()
    used_classes = collections.Counter()
    stats = collections.Counter()
    specs, dropped = [], []

    for scene, json_path, payload in payloads:
        camera = payload["frame_id"]
        geom = geometry_from_gt(payload)
        dets = payload["detections"]
        landmarks = [d for d in dets if d.get("kind") == "landmark" and not is_person(d)
                     and d.get("world_xyz")]
        people = [d for d in dets if is_person(d) and d.get("world_xyz")]
        objects = [d for d in dets if d.get("kind") == "object" and not is_person(d)
                   and d["class"] in MISSION_OBJECT_TYPES and d["class"] not in SKIPPED_OBJECTS
                   and d.get("world_xyz")]
        base = {"scene": scene, "camera_id": camera,
                "frames": {camera: str(json_path.with_suffix(".png"))}, "gt_json": str(json_path)}

        def emit(case_id, kind, label, command, parsed, target, anchors, extra, tags):
            point = ground_point(geom, target["bbox"])
            landmark = None if kind == "sar" else extra["landmark_class"]
            near = leaks.nearest_m(camera, kind, label, landmark, point)
            if kind == "obj":
                tags = {**tags, "nearest_same_class_m": leaks.nearest_m(camera, kind, label, "*", point)}
            spec = {
                **base, "case_id": case_id, "kind": "sar" if kind == "sar" else "object",
                "command": command, "parsed": parsed,
                "target": _det_summary(target), "anchors": [_det_summary(a) for a in anchors],
                **extra,
                "tags": {**tags, "nearest_training_target_m": near},
            }
            if near is not None and near <= LEAK_M:
                dropped.append({**spec, "drop_reason": "near_duplicate"})
                return False
            specs.append(spec)
            return True

        for person in people:
            recipe_id = person.get("recipe_id")
            if not recipe_id:
                continue
            recipe = recipes[recipe_id]
            check = sar_check(person, recipe, people, landmarks)
            case_id = f"ho-s{scene:05d}-{person['instance_id']}"
            if not check.ok:
                dropped.append({**base, "case_id": case_id, "kind": "sar", "recipe_id": recipe_id,
                                "drop_reason": check.reason})
                continue
            parsed = _parsed_matches(person["command"], "person", None, None)
            if parsed is None:
                dropped.append({**base, "case_id": case_id, "kind": "sar", "recipe_id": recipe_id,
                                "drop_reason": "command_parse_mismatch"})
                continue
            n = int(train["recipe"].get(recipe_id, 0))
            emit(case_id, "sar", recipe_id, person["command"], parsed, person, check.anchors,
                 {"recipe_id": recipe_id, "relation": None,
                  "landmark_class": check.anchors[0]["class"] if check.anchors else None,
                  "anchor_rule": "support" if recipe.get("support_class") else "nearest_landmark"},
                 {"stratum": stratum(n), "train_count": n,
                  "absent_from_original_sft": int(original["recipe"].get(recipe_id, 0)) == 0,
                  "distractor_present": check.distractor_present, "novel_recipe": False})

        # One object case per scene: the least-used relation, then the least-used target
        # class, then a same-class distractor. A near-duplicate falls through to the next.
        choices = object_choices(payload, objects, landmarks)
        options = sorted(
            (c for obj_choices in choices.values() for c in obj_choices),
            key=lambda c: (used_relations[c.relation], used_classes[c.target["class"]],
                           not c.distractor_present, RELATIONS.index(c.relation),
                           c.target["instance_id"], c.landmark_class))
        stats["objects_without_unique_description"] += sum(not c for c in choices.values())
        for option in options:
            obj = option.target
            command = object_command(obj["class"], option.relation, option.landmark_class)
            parsed = _parsed_matches(command, obj["class"], option.relation, option.landmark_class)
            if parsed is None:
                stats["object_options_parse_mismatch"] += 1
                continue
            n = int(train["relation"].get(option.relation, 0))
            if emit(f"ho-s{scene:05d}-{obj['instance_id']}", "obj", obj["class"], command, parsed,
                    obj, option.anchors,
                    {"recipe_id": None, "relation": option.relation,
                     "landmark_class": option.landmark_class, "anchor_rule": "relation"},
                    {"stratum": stratum(n), "train_count": n,
                     "absent_from_original_sft": int(original["relation"].get(option.relation, 0)) == 0,
                     "distractor_present": option.distractor_present, "novel_recipe": False}):
                used_relations[option.relation] += 1
                used_classes[obj["class"]] += 1
                break
        else:
            stats["scenes_without_object_case"] += 1

    meta = {
        "relation_rule": RULE_VERSION,
        "relation_rule_note": "Our rule (scripts/frozen_eval/relations.py), not the organisers'.",
        "rare_max_train_rows": RARE_MAX,
        "leak_m": LEAK_M,
        "leak_index_skipped_rows": dict(leaks.skipped),
        "object_selection": dict(stats),
        "scenes": len({s for s, _, _ in payloads}),
        "rigs": rigs,
    }
    return specs, dropped, meta


def is_person_candidate(cand: dict) -> bool:
    return str(cand.get("class_name") or "").startswith("person_")


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gold_ids(candidates: list[dict], gt: dict, *, person: bool) -> list[str]:
    """Candidate IDs that match one GT instance: same class (any pose for people), IoU >= GOLD_IOU."""
    out = []
    for cand in candidates:
        if person != is_person_candidate(cand):
            continue
        if not person and cand["class_name"] != gt["class"]:
            continue
        if _iou(cand["bbox_xyxy"], gt["bbox"]) >= GOLD_IOU:
            out.append(cand["candidate_id"])
    return out


def assemble_case(spec: dict, detection: dict) -> dict:
    person = spec["kind"] == "sar"
    cands = detection["candidates"]
    target_ids = gold_ids(cands, spec["target"], person=person)
    anchor_ids = sorted({cid for a in spec["anchors"] for cid in gold_ids(cands, a, person=False)})
    pose_match = None
    if person and target_ids:
        poses = {c["candidate_id"]: c["class_name"] for c in cands}
        pose_match = any(poses[cid] == spec["target"]["class"] for cid in target_ids)
    anchor = spec["anchors"][0] if spec["anchors"] else None
    return {
        "case_id": spec["case_id"],
        "kind": spec["kind"],
        "command": spec["command"],
        "parsed": detection["parsed"],
        "candidates": cands,
        "gold": {
            "camera_ids": [spec["camera_id"]],
            "target_ids": target_ids,
            "anchor_ids": anchor_ids,
            "target_xyz": spec["target"]["world_xyz"],
            "anchor_xyz": anchor["world_xyz"] if anchor else None,
            "target_class": spec["target"]["class"],
            "landmark_class": spec["landmark_class"],
            "relation": spec["relation"],
            "recipe_id": spec["recipe_id"],
            "anchor_rule": spec["anchor_rule"],
            "target_gt": spec["target"],
            "anchors_gt": spec["anchors"],
        },
        "tags": {
            **spec["tags"],
            "detector_miss": not target_ids,
            "anchor_miss": anchor is not None and not anchor_ids,
            "target_pose_detected": pose_match,
            "proxy_count": sum(c["source"] == "proxy" for c in cands),
            "refine_count": sum(c["source"] == "refine" for c in cands),
        },
        "source": {"scene": spec["scene"], "gt_json": spec["gt_json"],
                   "detector": detection["detector"], "decisions": detection["decisions"]},
    }


AUDIT_N = 50
AUDIT_SEED = 87


def write_audit(root: Path, out_dir: Path) -> Path:
    """50 random held-out cases (fixed seed) as images + a sheet for one person to fill."""
    from PIL import Image, ImageDraw

    cases = case_store.read_cases(root, "heldout")
    sample = sorted(random.Random(AUDIT_SEED).sample(cases, min(AUDIT_N, len(cases))),
                    key=lambda c: c.case_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in sample:
        cam = case.gold["camera_ids"][0]
        im = Image.open(case.frames[cam]).convert("RGB")
        draw = ImageDraw.Draw(im)
        for gt, colour in ([(a, (0, 140, 255)) for a in case.gold["anchors_gt"]]
                           + [(case.gold["target_gt"], (0, 255, 0))]):
            draw.rectangle(gt["bbox"], outline=colour, width=3)
        draw.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
        draw.text((6, 6), f"{case.case_id} | {case.command}", fill=(255, 255, 0))
        im.save(out_dir / f"{case.case_id}.png")
    path = out_dir / "audit.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "command", "kind", "relation_or_recipe",
                         "label_correct", "note", "auditor"])
        for case in sample:
            writer.writerow([case.case_id, case.command, case.kind,
                             case.gold["relation"] or case.gold["recipe_id"], "", "", ""])
    return path


def wilson(errors: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = errors / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def score_audit(sheet: Path) -> dict:
    """``label_correct`` y/n per row -> label error rate with a Wilson 95% CI."""
    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    answered = [r for r in rows if r["label_correct"].strip().lower() in ("y", "n")]
    errors = sum(r["label_correct"].strip().lower() == "n" for r in answered)
    low, high = wilson(errors, len(answered))
    return {"audited": len(answered), "of": len(rows), "label_errors": errors,
            "error_rate": errors / len(answered) if answered else None,
            "wilson95": [low, high],
            "auditors": sorted({r["auditor"] for r in answered if r["auditor"]})}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the held-out evaluation set (#87).")
    sub = parser.add_subparsers(dest="action", required=True)
    p_specs = sub.add_parser("specs")
    p_specs.add_argument("--gt", type=Path, required=True, help="generator <scenario> folder")
    p_specs.add_argument("--training-record", type=Path, required=True)
    p_specs.add_argument("--out", type=Path, required=True)
    p_asm = sub.add_parser("assemble")
    p_asm.add_argument("--build", type=Path, required=True)
    p_asm.add_argument("--root", type=Path, required=True)
    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--root", type=Path, required=True)
    p_audit.add_argument("--out", type=Path, required=True)
    p_score = sub.add_parser("audit-score")
    p_score.add_argument("--sheet", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.action == "audit":
        print(write_audit(args.root, args.out))
        return 0
    if args.action == "audit-score":
        print(json.dumps(score_audit(args.sheet), indent=1))
        return 0
    if args.action == "specs":
        record = json.loads(args.training_record.read_text(encoding="utf-8"))
        specs, dropped, meta = build_specs(args.gt, record)
        args.out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("specs.jsonl", specs), ("dropped.jsonl", dropped)):
            (args.out / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        meta["spec_counts"] = {
            "specs": len(specs),
            "kind": collections.Counter(s["kind"] for s in specs),
            "relation": collections.Counter(s["relation"] for s in specs),
            "target_class": collections.Counter(s["target"]["class"] for s in specs),
            "distractor_present": collections.Counter(
                s["tags"]["distractor_present"] for s in specs),
            "recipe": collections.Counter(s["recipe_id"] for s in specs),
            "stratum": collections.Counter(s["tags"]["stratum"] for s in specs),
            "dropped": collections.Counter(d["drop_reason"] for d in dropped),
            "dropped_sar_by_recipe": collections.Counter(
                f"{d['recipe_id']}:{d['drop_reason']}" for d in dropped if d["kind"] == "sar"),
        }
        (args.out / "specs_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        print(json.dumps(meta["spec_counts"], indent=1, default=str))
        return 0

    specs = {s["case_id"]: s for s in _jsonl(args.build / "specs.jsonl")}
    detections = {d["case_id"]: d for d in _jsonl(args.build / "detections.jsonl")}
    missing = sorted(set(specs) - set(detections))
    if missing:
        raise SystemExit(f"{len(missing)} specs have no detector pass, e.g. {missing[:3]}")
    for case_id, spec in specs.items():
        case = assemble_case(spec, detections[case_id])
        case_store.write_case(args.root, "heldout", case, {k: Path(v) for k, v in spec["frames"].items()})
    print(json.dumps(case_store.counts(case_store.read_cases(args.root, "heldout")), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
