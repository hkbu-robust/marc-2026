"""Run the runtime Stage 1 round offline on #87 frozen cases (#115, #117).

Inside the agent image, with this checkout's sources mounted and a vLLM serving the
deployed Qwen (see scripts/README.md, "Held-out selection"):

    python3 -m scripts.heldout_selection --set heldout --out <dir>/off.jsonl \
        --env MARC_TILED_REFINE=0

Each case goes through ``run_stage1_round`` exactly as a live round does: the live
detector on the case's stored frames, the query-conditioned refinement, selection
(geometry ranking, Qwen, guards) and projection with the shipped ground model. Only
the platform is replaced (frames come from disk; no clock budget). One JSON line per
case: the submission, the selected target box and the distances to the GT target and
anchor. ``summarise`` turns two such files into a paired comparison.

``frozen`` (#89) runs one selector arm on the frozen #87 cases with no detection and
no command reading: each case's ``candidates_final`` are the round's candidates and its
``parsed_final`` is the command reading, so arms differ only in the selector. Per case
it records the raw selector answer (Qwen's pick before the runtime guards; geometry
when no Qwen call is made; nothing when the reply is unusable), the final answer
after guards and fallbacks, gold-ID exact match, and exact-kernel credit
(``stage1_runs.target_credit``; anchor ``1 - d/5``)::

    python3 -m scripts.heldout_selection frozen --root <87-eval-sets> --set heldout \
        --arm C --out /out/C.jsonl --env MARC_SAR_DETECTOR=qwen-remote ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

SETS = Path.home() / "marc-2026-work/experiments/ablations/05-frozen-eval-sets"
TARGET_CREDIT_M = 1.0  # well inside the ~1.85 m the scorer credited in #88 (#91 caveat)
MATCH_IOU = 0.5


def _iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def run(set_name: str, sets: Path, out: Path, limit: Optional[int],
        only: Optional[set[str]] = None) -> None:
    from PIL import Image

    from marc_agent.perception import cctv_yolo
    from marc_agent.perception.cctv_project import CctvProjector, static_lookup
    from marc_agent.perception.ground_model import default_ground_model
    from marc_agent.perception.stage1_pipeline import run_stage1_round
    from marc_agent.types import CctvDetectionRound, CctvFrame
    from scripts.stage1_replay import load_geometry

    detector = cctv_yolo.CctvYoloDetector()
    detector.load()
    cctv_yolo._detector = detector
    projector = CctvProjector(static_lookup(load_geometry()), default_ground_model())
    paths = sorted((sets / set_name / "cases").glob("*/case.json"))
    done = set()
    if out.exists():
        done = {json.loads(line)["case_id"] for line in out.read_text().splitlines() if line}
    with open(out, "a", encoding="utf-8") as sink:
        for path in paths[:limit]:
            case = json.loads(path.read_text(encoding="utf-8"))
            if case["kind"] != "object" or case["case_id"] in done:
                continue
            if only is not None and case["case_id"] not in only:
                continue
            frames = {}
            for camera, rel in sorted(case["frames"].items()):
                rgb = np.asarray(Image.open(path.parent / rel).convert("RGB"))
                frames[camera] = CctvFrame(camera, rgb.shape[1], rgb.shape[0], rgb)
            objects, people = detector.detect(frames)
            round_ = CctvDetectionRound(objects=objects, people=people, frames=frames,
                                        camera_ids=sorted(frames))
            started = time.perf_counter()
            result = run_stage1_round(None, case["command"], detector=lambda *a, **k: round_,
                                      projector=projector)
            sink.write(json.dumps(_record(case, result, time.perf_counter() - started)) + "\n")
            sink.flush()


def _record(case: dict, result, seconds: float) -> dict:
    gold = case["gold"]
    row = {"case_id": case["case_id"], "seconds": round(seconds, 2),
           "failure": result.failure_reason, "method": result.method}
    grounding = result.grounding
    if grounding is None:
        return row
    # The submitted target: the selected pair, or a lower pair the projection fell through to.
    target = next((item for item in result.candidates.detections.objects
                   if item.candidate_id == result.target_id), None)
    row.update({
        "camera_id": grounding.camera_id,
        "landmark": grounding.landmark,
        "relation": grounding.relation,
        "target_xyz": list(grounding.target_coord),
        "anchor_xyz": list(grounding.anchor_coord),
        "target_id": target.candidate_id if target else None,
        "target_source": getattr(target, "source", None) if target else None,
        "target_conf": target.confidence if target else None,
        "target_box": list(target.bbox_xyxy) if target else None,
        "target_m": float(np.linalg.norm(np.subtract(grounding.target_coord, gold["target_xyz"]))),
        "anchor_m": float(np.linalg.norm(np.subtract(grounding.anchor_coord, gold["anchor_xyz"]))),
        # Public cases have the expected point but no GT box.
        "gold_target": bool(target and "target_gt" in gold
                            and _iou(target.bbox_xyxy, gold["target_gt"]["bbox"]) >= MATCH_IOU),
    })
    return row


def credit(row: dict) -> float:
    """The part of a round score a one-camera case can measure: target and anchor."""
    if "target_m" not in row:
        return 0.0
    return 0.40 * (row["target_m"] <= TARGET_CREDIT_M) + 0.15 * max(0.0, 1.0 - row["anchor_m"] / 5.0)


def summarise(off: Path, on: Path, sets: Path = SETS, cases: str = "all") -> str:
    """Paired comparison of two runs: credit, gold-target rate, anchor error, bootstrap CI."""
    rows = {}
    for label, path in (("off", off), ("on", on)):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows.setdefault(row["case_id"], {})[label] = row
    tags = {}
    for case_id in rows:
        case = json.loads((sets / "heldout/cases" / case_id / "case.json").read_text())
        scene = int(Path(case["source"]["gt_json"]).parent.name.split("_")[-1])
        tags[case_id] = {"miss": case["tags"]["detector_miss"], "odd": scene % 2 == 1}
    keep = [c for c, r in rows.items() if {"off", "on"} <= set(r)
            and (cases == "all" or (cases == "odd") == tags[c]["odd"])]
    lines = [f"cases: {len(keep)} ({cases} scenes)", "",
             "| subset | n | credit off | credit on | delta [95% CI] | gold target off | on | median anchor m off | on |",
             "|---|---|---|---|---|---|---|---|---|"]
    rng = random.Random(0)
    for label, subset in (("all", keep), ("detector_miss", [c for c in keep if tags[c]["miss"]]),
                          ("not miss", [c for c in keep if not tags[c]["miss"]])):
        if not subset:
            continue
        off_c = np.array([credit(rows[c]["off"]) for c in subset])
        on_c = np.array([credit(rows[c]["on"]) for c in subset])
        diff = on_c - off_c
        boots = [diff[[rng.randrange(len(diff)) for _ in diff]].mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        gold = lambda arm: np.mean([rows[c][arm].get("gold_target", False) for c in subset])
        anchor = lambda arm: np.median([rows[c][arm].get("anchor_m", np.inf) for c in subset])
        lines.append(f"| {label} | {len(subset)} | {off_c.mean():.4f} | {on_c.mean():.4f} | "
                     f"{diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] | {gold('off'):.3f} | {gold('on'):.3f} | "
                     f"{anchor('off'):.2f} | {anchor('on'):.2f} |")
    changed = [c for c in keep if rows[c]["off"].get("target_id") != rows[c]["on"].get("target_id")
               or abs(credit(rows[c]["off"]) - credit(rows[c]["on"])) > 1e-9]
    lines += ["", f"cases whose selection or credit changed: {len(changed)}"]
    return "\n".join(lines)


ARMS = ("A", "B", "C")
PERSON_PREFIX = "person_"


def frozen_round(case, frames: dict):
    """The case's frozen final candidates as the round the selector sees (no detector)."""
    from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate

    objects, people = [], []
    for item in case.candidates:
        frame = frames[item["camera_id"]]
        size = (frame.width, frame.height)
        box = tuple(float(v) for v in item["bbox_xyxy"])
        if item["class_name"].startswith(PERSON_PREFIX):
            people.append(PersonCandidate(item["candidate_id"], box, item["class_name"],
                                          float(item["confidence"]), item["camera_id"], size))
        else:
            source = item.get("source") or "detector"
            objects.append(ObjectCandidate(item["candidate_id"], item["class_name"], box,
                                           float(item["confidence"]), item["camera_id"], size,
                                           "detector" if source == "refine" else source))
    return CctvDetectionRound(objects=objects, people=people, frames=frames,
                              camera_ids=sorted(frames))


def to_parsed(raw: dict):
    from marc_agent.command_parser import Intent, ParsedCommand, Relation, Situation

    return ParsedCommand(
        intent=Intent(raw["intent"]), target_type=raw["target_type"],
        landmark=raw.get("landmark"),
        relation=Relation(raw["relation"]) if raw.get("relation") else None,
        situation=Situation(raw["situation"]) if raw.get("situation") else None,
        scene_hint=raw.get("scene_hint"), place_cues=tuple(raw.get("place_cues") or ()),
    )


def raw_answer(notes: list[dict], pre_anchor: Optional[tuple]) -> dict:
    """The selector's own answer before the runtime guards, from the decision notes.

    ``qwen_choice``: Qwen's pair as returned by the selector; ``geometry_override``:
    the pair Qwen gave before a guard replaced it; ``fallback``: none (unusable reply
    or call failure); ``geometry_only`` / ``occlusion_hypothesis``: the geometry pick
    (no Qwen call on that route)."""
    selector = next((n for n in reversed(notes) if n.get("decision") == "selector"), None)
    override = next((n for n in reversed(notes) if n.get("decision") == "geometry_override"), None)
    outcome = selector.get("outcome") if selector else None
    row = {"outcome": outcome, "reason": selector.get("reason") if selector else None,
           "target_id": None, "anchor_id": None}
    if outcome == "geometry_override" and override is not None:
        row.update(target_id=override.get("qwen_target"), anchor_id=override.get("qwen_anchor"))
    elif outcome == "fallback":
        pass
    elif pre_anchor is not None:
        row.update(target_id=pre_anchor[0], anchor_id=pre_anchor[1])
    elif selector is not None:
        row["target_id"] = selector.get("selected")
    if outcome is None and pre_anchor is not None:  # occlusion hypothesis: no selector note
        row.update(outcome="occlusion_hypothesis", target_id=pre_anchor[0], anchor_id=pre_anchor[1])
    return row


def run_frozen(root: Path, set_name: str, arm: str, out: Path, kinds: tuple[str, ...],
               limit: Optional[int] = None, only: Optional[set[str]] = None) -> None:
    """Run one selector arm on the frozen cases; append one JSON line per case to ``out``."""
    import contextvars

    from marc_agent.command_llm import Reading
    from marc_agent.experiment_log import decision_scope
    from marc_agent.perception import object_selection, stage1_pipeline
    from marc_agent.perception.cctv_project import CctvProjector, static_lookup
    from marc_agent.perception.ground_model import default_ground_model
    from marc_agent.perception.object_selection import ObjectSelection, rank_object_candidates
    from marc_agent.types import CctvFrame
    from scripts.frozen_eval.case_store import load_set
    from scripts.stage1_replay import load_geometry

    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    if arm == "A":
        os.environ["MARC_SAR_DETECTOR"] = "fallback"  # no verifier: the geometry selection
    elif os.environ.get("MARC_SAR_DETECTOR") != "qwen-remote":
        raise ValueError("arms B and C need MARC_SAR_DETECTOR=qwen-remote and a vLLM URL")
    current: dict = {}
    # The frozen reading and candidates replace the reader and the refinement passes.
    stage1_pipeline.read_command = lambda *a, **k: Reading(
        current["parsed"], "frozen")
    stage1_pipeline.refine_detection_round = lambda parsed, detections, **k: detections
    inner = object_selection._select_object_candidate

    def capture(*args, **kwargs):
        picked = inner(*args, **kwargs)
        current["pre_anchor"] = None if picked is None else (
            picked.target.candidate_id, picked.anchor.candidate_id if picked.anchor else None)
        return picked

    object_selection._select_object_candidate = capture
    projector = CctvProjector(static_lookup(load_geometry()), default_ground_model())
    done = set()
    if out.exists():
        done = {json.loads(line)["case_id"] for line in out.read_text().splitlines() if line}
    cases = [c for c in load_set(root, set_name, version="final")
             if c.kind in kinds and c.case_id not in done and (only is None or c.case_id in only)]
    with open(out, "a", encoding="utf-8") as sink:
        for case in cases[:limit]:
            frames = {}
            for camera in sorted(case.frames):
                rgb = case.frame(camera)
                frames[camera] = CctvFrame(camera, rgb.shape[1], rgb.shape[0], rgb)
            round_ = frozen_round(case, frames)
            parsed = to_parsed(case.parsed_final)
            current.clear()
            current["parsed"] = parsed
            started = time.perf_counter()
            with decision_scope("frozen") as notes:
                result = contextvars.copy_context().run(
                    stage1_pipeline.run_stage1_round, None, case.command,
                    detector=lambda *a, **k: round_, projector=projector)
            seconds = time.perf_counter() - started
            row = _frozen_record(case, result, notes, current.get("pre_anchor"), seconds)
            row["arm"] = arm
            if case.kind == "object":
                board = rank_object_candidates(parsed, round_, projector=projector,
                                               command_text=case.command, visual_proposals=True)
                row["board_targets"] = sorted({item.target.candidate_id for item in board})
                raw = row["raw"]
                by_id = {item.candidate_id: item for item in round_.objects}
                if raw["target_id"] in by_id:
                    projected = stage1_pipeline._project_selection(projector, parsed, ObjectSelection(
                        by_id[raw["target_id"]], by_id.get(raw["anchor_id"]), 0.0))
                    if projected is not None:
                        raw["target_xyz"], raw["anchor_xyz"] = (list(map(float, p)) for p in projected)
            sink.write(json.dumps(row) + "\n")
            sink.flush()


def _frozen_record(case, result, notes: list[dict], pre_anchor, seconds: float) -> dict:
    gold = case.gold
    selection = result.candidates.object_selection
    sar = result.candidates.sar_selection
    raw = raw_answer(notes, pre_anchor if case.kind == "object" else None)
    if sar is not None and raw["target_id"] is None and raw["outcome"] == "qwen_choice":
        raw["target_id"] = sar.selected.candidate_id if sar.selected else None
    validated = (sar.raw_output if sar is not None
                 else selection.raw_output if selection is not None else None)
    final = {"target_id": result.target_id, "anchor_id": result.anchor_id,
             "method": result.method, "failure": result.failure_reason}
    if result.grounding is not None:
        final.update(target_xyz=list(map(float, result.grounding.target_coord)),
                     anchor_xyz=list(map(float, result.grounding.anchor_coord)),
                     camera_id=result.grounding.camera_id,
                     relation=result.grounding.relation, landmark=result.grounding.landmark)
    return {
        "case_id": case.case_id, "kind": case.kind, "seconds": round(seconds, 3),
        "raw": raw, "final": final, "reply": validated,
        "decisions": [n for n in notes if n.get("decision") in (
            "selector", "geometry_override", "class_screen_failed", "projection_fallthrough",
            "qwen_skipped", "scene_description_failed")],
        "gold_target_ids": gold.get("target_ids") or [],
        "gold_anchor_ids": gold.get("anchor_ids") or [],
        "detector_miss": bool(case.tags.get("detector_miss")),
    }


def exact_match(row: dict, answer: str = "final") -> dict:
    """Gold-ID match of one answer: target, anchor (objects) and joint."""
    pick = row[answer]
    target = pick.get("target_id") in set(row["gold_target_ids"])
    if row["kind"] != "object":
        return {"target": target, "anchor": None, "joint": target}
    anchors = set(row["gold_anchor_ids"])
    anchor = pick.get("anchor_id") in anchors if anchors else None
    return {"target": target, "anchor": anchor, "joint": target and anchor is not False}


LANDMARK_CLASSES = frozenset({
    "bench", "bicycle", "hydrant", "kick_scooter", "picnic_table", "postbox",
    "vending_machine", "trash_can", "police_car", "normal_car_1", "normal_car_2",
    "sports_car", "sports_car_2", "suv", "taxi"})


def gt_distractors(gold: dict, gt_detections: list[dict]) -> list:
    """World points of every other object or person in the camera (landmarks excluded)."""
    target_id = (gold.get("target_gt") or {}).get("instance_id")
    return [d["world_xyz"] for d in gt_detections
            if d.get("world_xyz") and d["class"] not in LANDMARK_CLASSES
            and d.get("instance_id") != target_id and d["world_xyz"] != gold["target_xyz"]]


def kernel_credit(pick: dict, gold: dict, distractors: list) -> tuple[float, Optional[float]]:
    """Exact-kernel target credit and anchor credit ``1 - d/5`` of one answer."""
    from scripts.stage1_runs import target_credit

    if "target_xyz" not in pick:
        return 0.0, 0.0
    target = target_credit(pick["target_xyz"], gold["target_xyz"], distractors)
    anchor = max(0.0, 1.0 - float(np.linalg.norm(
        np.subtract(pick["anchor_xyz"], gold["anchor_xyz"]))) / 5.0)
    return target, anchor


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", default="heldout")
    r.add_argument("--sets", type=Path, default=SETS)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    r.add_argument("--limit", type=int)
    r.add_argument("--cases", type=Path, help="file of case IDs, one per line: run only these")
    s = sub.add_parser("summarise")
    s.add_argument("--off", type=Path, required=True)
    s.add_argument("--on", type=Path, required=True)
    s.add_argument("--sets", type=Path, default=SETS)
    s.add_argument("--cases", choices=["all", "odd", "even"], default="all")
    f = sub.add_parser("frozen", help="one selector arm on the frozen #87 cases (#89)")
    f.add_argument("--root", type=Path, required=True, help="the frozen 87-eval-sets folder")
    f.add_argument("--set", default="heldout", choices=["heldout", "public"])
    f.add_argument("--arm", required=True, choices=ARMS)
    f.add_argument("--kinds", nargs="+", default=["object"], choices=["object", "sar"])
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    f.add_argument("--limit", type=int)
    f.add_argument("--cases", type=Path, help="file of case IDs, one per line: run only these")
    args = parser.parse_args(argv)
    if args.cmd == "frozen":
        for pair in args.env:
            key, _, value = pair.partition("=")
            os.environ[key] = value
        only = set(args.cases.read_text().split()) if args.cases else None
        run_frozen(args.root, args.set, args.arm, args.out, tuple(args.kinds), args.limit, only)
    elif args.cmd == "run":
        for pair in args.env:
            key, _, value = pair.partition("=")
            os.environ[key] = value
        only = set(args.cases.read_text().split()) if args.cases else None
        run(args.set, args.sets, args.out, args.limit, only)
    else:
        print(summarise(args.off, args.on, args.sets, args.cases))


if __name__ == "__main__":
    sys.exit(main())
