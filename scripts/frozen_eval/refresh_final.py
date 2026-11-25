"""Merge the final-system parse and detector pass into the held-out cases (#87 refresh).

    python -m scripts.frozen_eval.refresh_final --root SETS_ROOT \\
        --parses SETS_ROOT/heldout-build-final/parses.jsonl \\
        --detections SETS_ROOT/heldout-build-final/detections.jsonl --image-id sha256:...

Adds to each ``heldout/cases/<id>/case.json``, next to version 1 (which is left as is:
``parsed``, ``candidates``, ``gold.target_ids``/``anchor_ids`` and ``tags``):

- ``parsed_final`` and ``parse_final`` (source qwen|rules|partial, fallback, fixes, ...);
- ``candidates_final``: the final detector pass (``detect_final.py``);
- ``gold_final``: ``target_ids`` / ``anchor_ids`` recomputed on ``candidates_final`` with
  the builder's rule (``heldout.gold_ids``: GT class, any pose for people, IoU >= 0.5);
- ``tags_final``: ``detector_miss``, ``anchor_miss``, ``target_pose_detected`` and
  per-source counts; ``source.detector_final``: weights, contract, image, decisions.

Then writes ``heldout_detector_recall_final.json`` (same layout as version 1's
``heldout_detector_recall.json``: [found, n, recall, Wilson 95% CI of the miss rate],
plus the CI of the recall) and ``heldout_detector_recall_v1_vs_final.md``.
Idempotent: re-running replaces only the ``*_final`` keys.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from scripts.frozen_eval.heldout import gold_ids, wilson

V1_KEYS = ("parsed", "candidates", "gold", "tags")


def _jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


def merge_case(case: dict, parse: dict, detection: dict, image: dict) -> dict:
    person = case["kind"] == "sar"
    cands = detection["candidates"]
    gold = case["gold"]
    target_ids = gold_ids(cands, gold["target_gt"], person=person)
    anchor_ids = sorted({cid for a in gold["anchors_gt"] for cid in gold_ids(cands, a, person=False)})
    pose_match = None
    if person and target_ids:
        poses = {c["candidate_id"]: c["class_name"] for c in cands}
        pose_match = any(poses[cid] == gold["target_gt"]["class"] for cid in target_ids)
    sources = collections.Counter(c["source"] for c in cands)
    has_anchor = bool(gold["anchors_gt"])
    case["parsed_final"] = parse["parsed_final"]
    case["parse_final"] = {
        "reader": "marc_agent.command_llm.read_command (MARC_COMMAND_PARSER=qwen, expected_intent=find)",
        "source": parse["source"],
        "fallback_reason": parse["fallback_reason"],
        "fixes": parse["fixes"],
        "pose_hint": parse["pose_hint"],
        "situation_hint": parse["situation_hint"],
        "latency_s": parse["latency_s"],
        "rules_parsed": parse["rules_parsed"],
        "same_as_v1_parsed": parse["parsed_final"] == case["parsed"],
    }
    case["candidates_final"] = cands
    case["gold_final"] = {"target_ids": target_ids, "anchor_ids": anchor_ids}
    case["tags_final"] = {
        "detector_miss": not target_ids,
        "anchor_miss": has_anchor and not anchor_ids,
        "target_pose_detected": pose_match,
        "candidate_count": len(cands),
        "source_counts": dict(sorted(sources.items())),
        "target_from": sorted({c["source"] for c in cands if c["candidate_id"] in target_ids}),
        "oom": detection["oom"],
    }
    case.setdefault("source", {})["detector_final"] = {
        **detection["detector"],
        "image": image,
        "parsed_used_for_refine": detection["parsed"],
        "decisions": detection["decisions"],
        "ms": detection["ms"],
    }
    return case


def recall_table(cases: list[dict], tags_key: str) -> dict:
    def summarise(groups: dict) -> dict:
        out = {}
        for key in sorted(groups):
            found, n = groups[key]
            low, high = wilson(n - found, n)
            out[key] = [found, n, round(found / n, 3), [round(low, 3), round(high, 3)],
                        [round(1 - high, 3), round(1 - low, 3)]]
        return out

    by_class, by_relation, overall = {}, {}, {}
    for case in cases:
        hit = 0 if case[tags_key]["detector_miss"] else 1
        for table, key in ((by_class, case["gold"]["target_class"]),
                           (by_relation, case["gold"]["relation"] or "sar"),
                           (overall, "all")):
            found, n = table.get(key, (0, 0))
            table[key] = (found + hit, n + 1)
    anchors = [c for c in cases if c["gold"]["anchors_gt"]]
    anchor_found = sum(not c[tags_key]["anchor_miss"] for c in anchors)
    return {
        "layout": "[found, n, recall, wilson95 of the miss rate, wilson95 of the recall]",
        "by_class": summarise(by_class),
        "by_relation": summarise(by_relation),
        "overall": summarise(overall),
        "anchor": summarise({"with_anchor_gt": (anchor_found, len(anchors))}),
        "sar_pose_detected": dict(collections.Counter(
            str(c[tags_key]["target_pose_detected"]).lower() for c in cases if c["kind"] == "sar")),
    }


def comparison_md(v1: dict, final: dict, cases: list[dict]) -> str:
    lines = ["# Held-out detector recall: version 1 (qualifier pass) vs final (`final-system-v1`)", "",
             "Recall = share of cases with at least one candidate matching the GT target "
             "(GT class, any pose for people, IoU >= 0.5). Wilson 95% CI on the recall.", ""]
    for section in ("overall", "by_class", "by_relation", "anchor"):
        lines += [f"## {section}", "", "| key | v1 found/n | v1 recall [95% CI] | final found/n | "
                  "final recall [95% CI] | delta |", "|---|---|---|---|---|---|"]
        for key in final[section]:
            f = final[section][key]
            a = v1[section].get(key)
            v1_cell = (f"{a[0]}/{a[1]} | {a[2]:.3f} [{a[4][0]:.3f}, {a[4][1]:.3f}]" if a else "- | -")
            delta = f"{f[2] - a[2]:+.3f}" if a else "-"
            lines.append(f"| {key} | {v1_cell} | {f[0]}/{f[1]} | {f[2]:.3f} [{f[4][0]:.3f}, {f[4][1]:.3f}] | {delta} |")
        lines.append("")
    flips = collections.Counter((c["tags"]["detector_miss"], c["tags_final"]["detector_miss"]) for c in cases)
    aflips = collections.Counter((c["tags"]["anchor_miss"], c["tags_final"]["anchor_miss"]) for c in cases)
    lines += ["## Case transitions (v1 -> final)", "",
              f"- detector_miss: hit->hit {flips[(False, False)]}, miss->hit {flips[(True, False)]}, "
              f"hit->miss {flips[(False, True)]}, miss->miss {flips[(True, True)]}",
              f"- anchor_miss: hit->hit {aflips[(False, False)]}, miss->hit {aflips[(True, False)]}, "
              f"hit->miss {aflips[(False, True)]}, miss->miss {aflips[(True, True)]}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parses", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image", default="robust-agent:finals")
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args(argv)

    parses, detections = _jsonl(args.parses), _jsonl(args.detections)
    paths = sorted((args.root / "heldout" / "cases").glob("*/case.json"))
    missing = [p.parent.name for p in paths if p.parent.name not in parses or p.parent.name not in detections]
    if missing:
        raise SystemExit(f"{len(missing)} cases lack a parse or detector pass, e.g. {missing[:3]}")
    ooms = [cid for cid, d in detections.items() if d["oom"]["messages"] or d["oom"]["zoom_skipped"]]
    if ooms:
        raise SystemExit(f"{len(ooms)} cases hit CUDA OOM; re-run them first: {ooms[:5]}")
    image = {"tag": args.image, "id": args.image_id, "revision": "60085d9fd4e0005281042341224d4cbae7c97f59"}
    cases = []
    for path in paths:
        case = json.loads(path.read_text(encoding="utf-8"))
        before = {k: json.dumps(case[k], sort_keys=True) for k in V1_KEYS}
        case = merge_case(case, parses[case["case_id"]], detections[case["case_id"]], image)
        assert all(json.dumps(case[k], sort_keys=True) == before[k] for k in V1_KEYS), path
        path.write_text(json.dumps(case, indent=1, default=str), encoding="utf-8")
        cases.append(case)
    final = recall_table(cases, "tags_final")
    v1 = recall_table(cases, "tags")
    (args.root / "heldout_detector_recall_final.json").write_text(json.dumps(final, indent=1))
    (args.root / "heldout_detector_recall_v1_vs_final.md").write_text(comparison_md(v1, final, cases))
    summary = {
        "cases": len(cases),
        "kind": dict(collections.Counter(c["kind"] for c in cases)),
        "detector_miss": {"v1": sum(c["tags"]["detector_miss"] for c in cases),
                          "final": sum(c["tags_final"]["detector_miss"] for c in cases)},
        "anchor_miss": {"v1": sum(c["tags"]["anchor_miss"] for c in cases),
                        "final": sum(c["tags_final"]["anchor_miss"] for c in cases)},
        "parse_source": dict(collections.Counter(c["parse_final"]["source"] for c in cases)),
        "parse_fallback": dict(collections.Counter(
            c["parse_final"]["fallback_reason"] for c in cases if c["parse_final"]["fallback_reason"])),
        "parse_same_as_v1": sum(c["parse_final"]["same_as_v1_parsed"] for c in cases),
        "target_from_final": dict(collections.Counter(
            ",".join(c["tags_final"]["target_from"]) or "miss" for c in cases)),
    }
    (args.root / "heldout-build-final" / "refresh_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
