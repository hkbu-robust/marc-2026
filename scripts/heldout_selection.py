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


def run(set_name: str, sets: Path, out: Path, limit: Optional[int]) -> None:
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
        "gold_target": bool(target and _iou(target.bbox_xyxy, gold["target_gt"]["bbox"]) >= MATCH_IOU),
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", default="heldout")
    r.add_argument("--sets", type=Path, default=SETS)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    r.add_argument("--limit", type=int)
    s = sub.add_parser("summarise")
    s.add_argument("--off", type=Path, required=True)
    s.add_argument("--on", type=Path, required=True)
    s.add_argument("--sets", type=Path, default=SETS)
    s.add_argument("--cases", choices=["all", "odd", "even"], default="all")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        for pair in args.env:
            key, _, value = pair.partition("=")
            os.environ[key] = value
        run(args.set, args.sets, args.out, args.limit)
    else:
        print(summarise(args.off, args.on, args.sets, args.cases))


if __name__ == "__main__":
    sys.exit(main())
