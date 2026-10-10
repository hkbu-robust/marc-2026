"""Public evaluation set from a declared capture run (#87): #128 run-01 for the final system.

    python -m scripts.frozen_eval.public build --run RUN_DIR --geometry sdk_cameras.json \\
        --training-record JSON --root SETS [--review-out DIR]
    python -m scripts.frozen_eval.public confirm --root SETS --review review.csv

``build`` writes one case per round (33) from what the frozen agent saw:
the command, the parsed command and the deployed detector's full candidate list
(``stage1.pipeline.completed``), all camera frames of the round dump, and the
platform's expected coordinates (``score.received``).

The platform gives coordinates, not candidate IDs. A gold **proposal** is every
candidate of the expected class whose projection lies within ``GOLD_MAX_M`` of
the expected point, the nearest per camera; the anchor likewise. If none does
and the live run's pick scored target 1.0, that pick is proposed instead
(hidden or proxy-projected targets project metres off but still score 1.0). Projection uses
the SDK geometry: the ray through the box bottom-centre and centre meets the
plane at the expected height. Each proposal stores its distance, the runner-up
distance, and whether the live run picked it and scored target 1.0. It is
**unconfirmed** until ``auto-confirm`` accepts an unambiguous proposal by rule, or
a reviewer's verdict is applied with ``confirm``; ``case_store.freeze`` refuses
unconfirmed public cases. The organisers' expected coordinates stay the ground
truth; gold IDs only link them to candidates for exact-match metrics.

The capture run is the final system itself, so each case also stores its
candidates and parse as the ``final`` version (``case_store.load_set(...,
version="final")``), the same keys the refreshed held-out cases carry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from marc_agent.perception.cctv_project import CameraGeometry, intersect_ray_plane
from scripts.frozen_eval import case_store
from scripts.frozen_eval.geometry import project_point
from scripts.frozen_eval.heldout import is_person_candidate, stratum

GOLD_MAX_M = 1.0


def _events(run_dir: Path) -> list[dict]:
    paths = sorted(run_dir.glob("experiments/*/events.jsonl"))
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one events.jsonl under {run_dir}, got {paths}")
    return [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines() if line]


def load_geometry(path: Path) -> dict[str, CameraGeometry]:
    raw = json.loads(path.read_text(encoding="utf-8"))["geometry"]
    return {
        cam: CameraGeometry(
            fx=g["fx"], fy=g["fy"], cx=g["cx"], cy=g["cy"],
            rotation=tuple(tuple(row) for row in g["rotation"]),
            translation=tuple(g["translation"]), ground_height=g["ground_height"],
        )
        for cam, g in raw.items()
    }


def projection_error_m(geometry: CameraGeometry, bbox, expected_xyz) -> float | None:
    """Planar distance from the expected point to the box's ray at the expected height."""
    best = None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    for u, v in ((0.5 * (x0 + x1), y1), (0.5 * (x0 + x1), 0.5 * (y0 + y1))):
        hit = intersect_ray_plane(geometry, u, v, float(expected_xyz[2]))
        if hit is None:
            continue
        d = math.hypot(hit[0] - expected_xyz[0], hit[1] - expected_xyz[1])
        best = d if best is None else min(best, d)
    return best


def propose(candidates: list[dict], geometry: dict[str, CameraGeometry], expected_xyz,
            match) -> tuple[list[str], dict]:
    """Nearest matching candidate per camera within GOLD_MAX_M, plus evidence."""
    scored = []
    for cand in candidates:
        if not match(cand) or cand["camera_id"] not in geometry:
            continue
        d = projection_error_m(geometry[cand["camera_id"]], cand["bbox_xyxy"], expected_xyz)
        if d is not None:
            scored.append((d, cand["candidate_id"], cand["camera_id"]))
    scored.sort()
    gold, seen_cameras = [], set()
    for d, cid, cam in scored:
        if d <= GOLD_MAX_M and cam not in seen_cameras:
            gold.append(cid)
            seen_cameras.add(cam)
    evidence = {
        "nearest": [{"candidate_id": cid, "camera_id": cam, "error_m": round(d, 3)}
                    for d, cid, cam in scored[:4]],
        "gold_max_m": GOLD_MAX_M,
    }
    return gold, evidence


def _sources(candidates: list[dict], decisions: list[dict]) -> list[dict]:
    refined = {c for d in decisions if d.get("decision") == "refine_added"
               for c in (d.get("candidates") or []) if isinstance(c, str)}
    zoomed = {c for d in decisions if d.get("decision") == "tiled_refine"
              for c in (d.get("candidates") or []) if isinstance(c, str)}
    proxies = {d.get("candidate") for d in decisions if d.get("decision") == "proxy_created"}
    return [{**c, "source": "proxy" if c["candidate_id"] in proxies
             else "zoom" if c["candidate_id"] in zoomed
             else "refine" if c["candidate_id"] in refined else "detector"} for c in candidates]


def build_cases(run_dir: Path, geometry: dict[str, CameraGeometry], record: dict) -> list[tuple]:
    events = _events(run_dir)
    by_round: dict[int, dict] = {}
    for e in events:
        if e.get("stage") in ("stage1", 1) and e.get("round"):
            by_round.setdefault(int(e["round"]), {})[e["event"]] = e
    experiment_dir = sorted(run_dir.glob("experiments/*"))[0]
    out = []
    for rnd in sorted(by_round):
        ev = by_round[rnd]
        pipe, score = ev["stage1.pipeline.completed"], ev["score.received"]["scores"]
        sub = ev.get("stage1.submission.completed") or {}
        expected = score["expected"]
        kind = "sar" if expected["target_kind"] == "person" else "object"
        candidates = _sources(pipe["candidates"], pipe.get("decisions") or [])
        if kind == "sar":
            is_target = is_person_candidate
        else:
            is_target = lambda c: c.get("class_name") == expected["target_type"]  # noqa: E731
        target_ids, target_ev = propose(candidates, geometry, expected["target_coord"], is_target)
        # Target scoring checks the instance, not a distance kernel: a hidden or
        # proxy-projected target lands metres away yet scores 1.0. A live pick
        # that scored 1.0 is therefore gold evidence too; disagreement is flagged.
        # A cache hit (cache-on capture) can submit another round's candidate; only
        # this round's own candidates are evidence.
        live_id = sub.get("target_id")
        live_full = (score.get("target") == 1.0
                     and live_id in {c["candidate_id"] for c in candidates})
        target_ev["basis"] = "projection" if target_ids else None
        target_ev["conflict"] = bool(live_full and target_ids and live_id not in target_ids)
        if live_full and not target_ids:
            target_ids = [live_id]
            target_ev["basis"] = "live_target_score_1"
        anchor_ids, anchor_ev = [], {}
        if expected.get("anchor_coord") and expected.get("landmark_prefix"):
            anchor_ids, anchor_ev = propose(
                candidates, geometry, expected["anchor_coord"],
                lambda c: c.get("class_name") == expected["landmark_prefix"])
        problem_id = score["problem_id"]
        relation = expected.get("relation")
        if kind == "sar":
            n = int(record["train_counts"]["recipe"].get(problem_id, 0))
            n_original = int(record["original_train_counts"]["recipe"].get(problem_id, 0))
        else:
            n = int(record["train_counts"]["relation"].get(relation, 0))
            n_original = int(record["original_train_counts"]["relation"].get(relation, 0))
        same_class = [c for c in candidates if is_target(c)]
        frames = {
            cam: experiment_dir / pipe["artifact_dir"] / "cameras" / f"{cam}.png"
            for cam in pipe["camera_ids"]
        }
        cluster = json.dumps({k: expected.get(k) for k in sorted(expected)}, sort_keys=True)
        case = {
            "case_id": f"pub-r{rnd:02d}-{problem_id}",
            "kind": kind,
            "command": pipe["command"],
            "parsed": pipe["parsed"],
            "candidates": candidates,
            "gold": {
                "camera_ids": [expected["camera_id"]],
                "target_ids": target_ids,
                "anchor_ids": anchor_ids,
                "target_xyz": expected["target_coord"],
                "anchor_xyz": expected.get("anchor_coord"),
                "target_class": expected["target_type"],
                "landmark_class": expected.get("landmark_prefix"),
                "relation": relation,
                "recipe_id": problem_id if kind == "sar" else None,
                "anchor_rule": "platform_expected",
                "confirmed_by": None,
                "proposal": {"target": target_ev, "anchor": anchor_ev},
            },
            "tags": {
                "problem_id": problem_id,
                "round": rnd,
                "cluster": cluster,
                "stratum": stratum(n),
                "train_count": n,
                "absent_from_original_sft": n_original == 0,
                "distractor_present": len(same_class) > len(target_ids),
                "detector_miss": not target_ids,
                "anchor_miss": bool(expected.get("anchor_coord")) and not anchor_ids,
                "proxy_count": sum(c["source"] == "proxy" for c in candidates),
                "refine_count": sum(c["source"] == "refine" for c in candidates),
                "live_selected": {"target_id": sub.get("target_id"), "anchor_id": sub.get("anchor_id"),
                                  "camera_id": sub.get("camera_id"),
                                  "target_score": score.get("target"), "anchor_score": score.get("anchor")},
                "live_selected_is_gold": sub.get("target_id") in target_ids,
            },
            "source": {"run_dir": str(run_dir), "round": rnd, "artifact_dir": pipe["artifact_dir"]},
            # The capture run is the final system: its candidates and parse are the final version.
            "candidates_final": candidates,
            "parsed_final": pipe["parsed"],
            "gold_final": {},
            "tags_final": {},
        }
        out.append((case, frames))
    return out


def review_image(case: dict, frames: dict[str, Path], geometry: dict[str, CameraGeometry],
                 out_path: Path) -> None:
    """Every camera with gold (green), runner-ups (orange) and anchors (blue) boxed, and the
    platform's expected target (magenta X) and anchor (cyan X) projected into it."""
    from PIL import Image, ImageDraw

    boxes = {c["candidate_id"]: c for c in case["candidates"]}
    gold = set(case["gold"]["target_ids"])
    anchors = set(case["gold"]["anchor_ids"])
    near = {n["candidate_id"] for n in case["gold"]["proposal"]["target"].get("nearest", [])} - gold
    tiles = []
    for cam, path in sorted(frames.items()):
        im = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for cid, colour in ([(c, (255, 140, 0)) for c in near] + [(c, (0, 140, 255)) for c in anchors]
                            + [(c, (0, 255, 0)) for c in gold]):
            cand = boxes[cid]
            if cand["camera_id"] == cam:
                draw.rectangle(cand["bbox_xyxy"], outline=colour, width=4)
                draw.text((cand["bbox_xyxy"][0], max(0, cand["bbox_xyxy"][1] - 12)), cid, fill=colour)
        for key, colour in (("anchor_xyz", (0, 255, 255)), ("target_xyz", (255, 0, 255))):
            point = case["gold"].get(key)
            pixel = project_point(geometry[cam], point) if point and cam in geometry else None
            if pixel and 0 <= pixel[0] < im.width and 0 <= pixel[1] < im.height:
                x, y = pixel
                draw.line([x - 14, y - 14, x + 14, y + 14], fill=colour, width=3)
                draw.line([x - 14, y + 14, x + 14, y - 14], fill=colour, width=3)
        draw.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
        mark = " (expected camera)" if cam in case["gold"]["camera_ids"] else ""
        draw.text((6, 6), f"{case['case_id']} | {cam}{mark} | {case['command']}", fill=(255, 255, 0))
        tiles.append(im.resize((im.width // 2, im.height // 2)))
    width, height = tiles[0].size
    sheet = Image.new("RGB", (2 * width, height * ((len(tiles) + 1) // 2)))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * width, (i // 2) * height))
    sheet.save(out_path)


def write_review_sheet(cases: list[tuple], geometry: dict[str, CameraGeometry], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, frames in cases:
        review_image(case, frames, geometry, out_dir / f"{case['case_id']}.png")
        t = case["gold"]["proposal"]["target"].get("nearest", [])
        rows.append({
            "case_id": case["case_id"], "command": case["command"],
            "proposed_target_ids": " ".join(case["gold"]["target_ids"]),
            "proposed_anchor_ids": " ".join(case["gold"]["anchor_ids"]),
            "basis": case["gold"]["proposal"]["target"].get("basis") or "",
            "conflict_with_live_pick": case["gold"]["proposal"]["target"].get("conflict"),
            "nearest_error_m": t[0]["error_m"] if t else "",
            "runner_up_error_m": t[1]["error_m"] if len(t) > 1 else "",
            "live_selected_is_gold": case["tags"]["live_selected_is_gold"],
            "live_target_score": case["tags"]["live_selected"]["target_score"],
            "verdict": "", "corrected_target_ids": "", "corrected_anchor_ids": "", "note": "",
            "reviewer": "",
        })
    path = out_dir / "review.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# Rounds whose target no camera shows (#136 visibility check; earlier verdicts in #87).
HIDDEN_TARGET_PROBLEMS = {"lf03", "lf06", "hf06", "lf07", "hf02"}


def ambiguity(case: dict) -> list[str]:
    """Why the rule's gold proposal needs a visual check; empty means accept by rule."""
    reasons = []
    target = case["gold"]["proposal"]["target"]
    problem = case["tags"]["problem_id"].split("_")[0]
    if problem in HIDDEN_TARGET_PROBLEMS:
        reasons.append("hidden-target round")
    if not case["gold"]["target_ids"]:
        reasons.append("no proposal within 1 m")
    if target.get("basis") == "live_target_score_1":
        reasons.append("live-pick fallback")
    if target.get("conflict"):
        reasons.append("live pick disagrees")
    near = target.get("nearest") or []
    gold = set(case["gold"]["target_ids"])
    for item in near:
        same_cam = [n for n in near if n["camera_id"] == item["camera_id"]]
        if (item["candidate_id"] in gold and len(same_cam) > 1
                and same_cam[1]["error_m"] <= case["gold"]["proposal"]["target"]["gold_max_m"]):
            reasons.append("runner-up within 1 m in the same camera")
            break
    if case["gold"].get("anchor_xyz") and case["gold"].get("landmark_class") and not case["gold"]["anchor_ids"]:
        reasons.append("no anchor proposal within 1 m")
    return reasons


def auto_confirm(root: Path, review_csv: Path) -> int:
    """Confirm every unambiguous proposal by rule; list the ambiguous ones in the review
    sheet (``needs_check``) for a visual check applied later with ``confirm``."""
    rows = list(csv.DictReader(review_csv.open(encoding="utf-8")))
    by_id = {r["case_id"]: r for r in rows}
    confirmed = 0
    for case_dir in sorted((root / "public" / "cases").iterdir()):
        path = case_dir / "case.json"
        case = json.loads(path.read_text(encoding="utf-8"))
        reasons = ambiguity(case)
        row = by_id.get(case["case_id"])
        if row is not None:
            row["needs_check"] = "; ".join(reasons)
        if reasons or case["gold"].get("confirmed_by"):
            continue
        case["gold"]["confirmed_by"] = {
            "reviewer": "rule", "verdict": "ok",
            "note": f"unambiguous proposal ({case['gold']['proposal']['target'].get('basis')})",
        }
        path.write_text(json.dumps(case, indent=1, default=str), encoding="utf-8")
        confirmed += 1
    fields = list(rows[0]) + ([] if "needs_check" in rows[0] else ["needs_check"])
    with review_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pending = [r["case_id"] for r in rows if r.get("needs_check")]
    print(f"{confirmed} confirmed by rule; {len(pending)} need a visual check: {pending}")
    return 0


def confirm(root: Path, review_csv: Path) -> int:
    """Apply a filled review sheet: ``verdict`` ok|corrected, corrected IDs (empty = detector
    miss), optional ``note``, and ``reviewer``. Rows without a verdict and reviewer are skipped."""
    rows = {r["case_id"]: r for r in csv.DictReader(review_csv.open(encoding="utf-8"))}
    updated = 0
    for case_dir in sorted((root / "public" / "cases").iterdir()):
        path = case_dir / "case.json"
        case = json.loads(path.read_text(encoding="utf-8"))
        row = rows.get(case["case_id"])
        if not row or row["verdict"] not in ("ok", "corrected") or not row["reviewer"]:
            continue
        confirmed = {"reviewer": row["reviewer"], "verdict": row["verdict"],
                     "note": (row.get("note") or "").strip()}
        if row["verdict"] == "corrected":
            # Keep what was proposed: a proxy the platform rewards is still worth reporting.
            confirmed["proposed_target_ids"] = case["gold"]["target_ids"]
            confirmed["proposed_anchor_ids"] = case["gold"]["anchor_ids"]
            case["gold"]["target_ids"] = row["corrected_target_ids"].split()
            case["gold"]["anchor_ids"] = row["corrected_anchor_ids"].split()
            case["tags"]["detector_miss"] = not case["gold"]["target_ids"]
            case["tags"]["anchor_miss"] = (bool(case["gold"].get("anchor_xyz"))
                                           and not case["gold"]["anchor_ids"])
            case["tags"]["live_selected_is_gold"] = (
                case["tags"]["live_selected"]["target_id"] in case["gold"]["target_ids"])
        case["gold"]["confirmed_by"] = confirmed
        path.write_text(json.dumps(case, indent=1, default=str), encoding="utf-8")
        updated += 1
    print(f"{updated} public cases confirmed")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the public evaluation set (#87).")
    sub = parser.add_subparsers(dest="action", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--run", type=Path, required=True, help="the declared capture run (#128 run-01)")
    p_build.add_argument("--geometry", type=Path, required=True, help="capture_sdk_geometry output")
    p_build.add_argument("--training-record", type=Path, required=True)
    p_build.add_argument("--root", type=Path, required=True)
    p_build.add_argument("--review-out", type=Path)
    p_auto = sub.add_parser("auto-confirm")
    p_auto.add_argument("--root", type=Path, required=True)
    p_auto.add_argument("--review", type=Path, required=True)
    p_confirm = sub.add_parser("confirm")
    p_confirm.add_argument("--root", type=Path, required=True)
    p_confirm.add_argument("--review", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.action == "confirm":
        return confirm(args.root, args.review)
    if args.action == "auto-confirm":
        return auto_confirm(args.root, args.review)
    record = json.loads(args.training_record.read_text(encoding="utf-8"))
    geometry = load_geometry(args.geometry)
    cases = build_cases(args.run, geometry, record)
    for case, frames in cases:
        case_store.write_case(args.root, "public", case, frames)
    if args.review_out:
        print(write_review_sheet(cases, geometry, args.review_out))
    print(json.dumps(case_store.counts(case_store.read_cases(args.root, "public")), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
