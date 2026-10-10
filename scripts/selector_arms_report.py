"""Score the #89 selector arms and write ``numbers.json`` (#130 format) and ``report.md``.

Reads ``heldout-<arm>.jsonl`` from ``heldout_selection frozen`` runs in RUN_DIR, the
frozen #87 cases (tags, gold, GT camera files) and the manifest's per-case tags::

    python -m scripts.selector_arms_report RUN_DIR --root <87-eval-sets> [--arms A B C]

Exact match: gold-ID joint match (``heldout_selection.exact_match``), detector misses
excluded. Credit: exact kernel (``heldout_selection.kernel_credit``); anchor credit on
odd scenes only. ``--sar`` scores the held-out person cases instead (``sar_report``).
Paired differences: cluster bootstrap over generator scenes and an
exact McNemar test on discordant pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from scripts.frozen_eval.case_store import load_set
from scripts.heldout_selection import exact_match, gt_distractors, kernel_credit

RESAMPLES = 10_000
DEGENERATE_PX2 = 400  # person gold boxes below this are hidden people (#87 label audit)
SEED = 89


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_ci(values: np.ndarray, clusters: list[str], resamples: int = RESAMPLES,
               seed: int = SEED) -> list[float]:
    """95% percentile CI of the mean, resampling clusters with replacement."""
    names = sorted(set(clusters))
    index = {name: i for i, name in enumerate(names)}
    ids = np.array([index[c] for c in clusters])
    sums = np.bincount(ids, weights=values, minlength=len(names))
    counts = np.bincount(ids, minlength=len(names)).astype(float)
    draws = np.random.default_rng(seed).integers(0, len(names), (resamples, len(names)))
    means = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def load_rows(run_dir: Path, arms: list[str], prefixes: list[str]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for prefix in prefixes:
        for arm in arms:
            for line in (run_dir / f"{prefix}-{arm}.jsonl").read_text().splitlines():
                row = json.loads(line)
                rows.setdefault(row["case_id"], {})[arm] = row
    return {cid: by_arm for cid, by_arm in rows.items() if set(arms) <= set(by_arm)}


def case_table(root: Path, case_ids: set[str]) -> dict[str, dict]:
    """Per case: scene, parity, on/above, stratum, relation, gold points and distractors."""
    manifest = json.loads((root / "manifest.json").read_text())
    tags = manifest["sets"]["heldout"]["case_tags"]
    table = {}
    for case in load_set(root, "heldout", version="final"):
        if case.case_id not in case_ids:
            continue
        gt_json = case.source["gt_json"]
        gt = json.loads(Path(gt_json).read_text())
        table[case.case_id] = {
            "scene": re.search(r"scene_(\d+)", gt_json).group(1),
            "odd": tags[case.case_id]["scene_parity"] == "odd",
            "on_above": tags[case.case_id]["on_above"],
            "stratum": case.tags.get("stratum"),
            "relation": case.gold.get("relation"),
            "miss": bool(case.tags.get("detector_miss")),
            "gold": case.gold,
            "distractors": gt_distractors(case.gold, gt["detections"]),
            "command": case.command,
            "kind": case.kind,
            "recipe": case.gold.get("recipe_id"),
            "degenerate": case.kind == "sar" and _box_area(case.gold) < DEGENERATE_PX2,
        }
    return table


def _box_area(gold: dict) -> float:
    box = (gold.get("target_gt") or {}).get("bbox") or [0, 0, 0, 0]
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def per_case_metrics(row: dict, info: dict) -> dict:
    out = {}
    for answer in ("raw", "final"):
        match = exact_match(row, answer)
        target, anchor = kernel_credit(row[answer], info["gold"], info["distractors"])
        out[answer] = {"joint": match["joint"], "target": match["target"],
                       "target_credit": target, "anchor_credit": anchor,
                       "credit": 0.40 * target + 0.15 * anchor}
    reason = (row["raw"].get("reason") or "")
    out["fallback"] = row["raw"].get("outcome") == "fallback"
    out["invalid"] = out["fallback"] and "Timeout" not in reason and "Connection" not in reason
    out["seconds"] = row["seconds"]
    out["on_board"] = bool(set(row.get("board_targets") or []) & set(row["gold_target_ids"]))
    return out


def arm_summary(metrics: dict[str, dict], table: dict[str, dict], arm: str) -> dict:
    ids = sorted(metrics)
    hit = [c for c in ids if not table[c]["miss"]]
    odd = [c for c in ids if table[c]["odd"]]
    m = lambda c: metrics[c][arm]
    secs = np.array([m(c)["seconds"] for c in ids])
    return {
        "exact_raw": float(np.mean([m(c)["raw"]["joint"] for c in hit])),
        "exact_guarded": float(np.mean([m(c)["final"]["joint"] for c in hit])),
        "target_exact_guarded": float(np.mean([m(c)["final"]["target"] for c in hit])),
        "target_credit": float(np.mean([m(c)["final"]["target_credit"] for c in ids])),
        "target_credit_raw": float(np.mean([m(c)["raw"]["target_credit"] for c in ids])),
        "anchor_credit": float(np.mean([m(c)["final"]["anchor_credit"] for c in odd])),
        "anchor_credit_raw": float(np.mean([m(c)["raw"]["anchor_credit"] for c in odd])),
        "credit_odd": float(np.mean([m(c)["final"]["credit"] for c in odd])),
        "credit_odd_raw": float(np.mean([m(c)["raw"]["credit"] for c in odd])),
        "invalid_rate": float(np.mean([m(c)["invalid"] for c in ids])),
        "fallback_rate": float(np.mean([m(c)["fallback"] for c in ids])),
        "latency_p50_s": float(np.percentile(secs, 50)),
        "latency_p95_s": float(np.percentile(secs, 95)),
    }


def paired(metrics, table, ids, left, right, key) -> dict:
    """``left - right`` on one per-case metric: value, cluster CI, McNemar for binaries."""
    answer, field = key
    a = np.array([float(metrics[c][left][answer][field]) for c in ids])
    b = np.array([float(metrics[c][right][answer][field]) for c in ids])
    diff = a - b
    out = {"value": float(diff.mean()), "ci": cluster_ci(diff, [table[c]["scene"] for c in ids]),
           "n": len(ids)}
    if field in ("joint", "target"):
        out["mcnemar_p"] = mcnemar_p(int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()))
        out["discordant"] = [int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())]
    return out


def report(run_dir: Path, root: Path, arms: list[str], prefixes: list[str]) -> dict:
    rows = load_rows(run_dir, arms, prefixes)
    table = case_table(root, set(rows))
    metrics = {c: {arm: per_case_metrics(rows[c][arm], table[c]) for arm in arms} for c in rows}
    degenerate = [c for c in metrics if table[c]["degenerate"]]
    ids = sorted(c for c in metrics if not table[c]["degenerate"])
    metrics = {c: metrics[c] for c in ids}
    hit = [c for c in ids if not table[c]["miss"]]
    odd = [c for c in ids if table[c]["odd"]]
    objects = [c for c in hit if table[c]["kind"] == "object"]
    shortlist = float(np.mean([metrics[c][arms[-1]]["on_board"] for c in objects]))
    numbers = {"heldout": {
        "cases": len(ids), "non_miss_cases": len(hit), "detector_miss": len(ids) - len(hit),
        "odd_cases": len(odd),
        "excluded_degenerate_person_boxes": len(degenerate), "shortlist_recall": shortlist,
        "arms": {arm: arm_summary(metrics, table, arm) for arm in arms},
        "diff": {}, "strata": {}}}
    held = numbers["heldout"]
    others = [a for a in arms if a != "C"] if "C" in arms else []
    for other in others:
        held["diff"][f"C_minus_{other}"] = {
            "exact_raw": paired(metrics, table, hit, "C", other, ("raw", "joint")),
            "exact_guarded": paired(metrics, table, hit, "C", other, ("final", "joint")),
            "target_credit": paired(metrics, table, ids, "C", other, ("final", "target_credit")),
            "anchor_credit_odd": paired(metrics, table, odd, "C", other, ("final", "anchor_credit")),
            "credit_odd": paired(metrics, table, odd, "C", other, ("final", "credit")),
        }
    for stratum in ("rare", "seen"):
        sub = [c for c in hit if table[c]["stratum"] == stratum]
        held["strata"][stratum] = {f"C_minus_{o}": paired(metrics, table, sub, "C", o, ("final", "joint"))
                                   for o in others}
        held["strata"][stratum]["n"] = len(sub)
    subsets = {"objects": objects, "sar": [c for c in hit if table[c]["kind"] == "sar"],
               "without on/above": [c for c in hit if not table[c]["on_above"]],
               "on/above only": [c for c in hit if table[c]["on_above"]]}
    for relation in sorted({table[c]["relation"] for c in hit}, key=str):
        subsets[f"relation {relation}"] = [c for c in hit if table[c]["relation"] == relation]
    for stratum in ("rare", "seen"):
        subsets[f"stratum {stratum}"] = [c for c in hit if table[c]["stratum"] == stratum]
    held["breakdowns"] = {}
    for name, sub in subsets.items():
        if not sub:
            continue
        entry = {"n": len(sub)}
        for arm in arms:
            entry[arm] = {"exact_raw": float(np.mean([metrics[c][arm]["raw"]["joint"] for c in sub])),
                          "exact_guarded": float(np.mean([metrics[c][arm]["final"]["joint"] for c in sub]))}
        for other in others:
            entry[f"C_minus_{other}"] = paired(metrics, table, sub, "C", other, ("final", "joint"))
        held["breakdowns"][name] = entry
    discordant = []
    for c in hit:
        flags = {arm: metrics[c][arm]["final"]["joint"] for arm in arms}
        if len(set(flags.values())) > 1:
            discordant.append({"case_id": c, "command": table[c]["command"],
                               "relation": table[c]["relation"], "stratum": table[c]["stratum"],
                               "correct": flags,
                               "picks": {arm: [rows[c][arm]["final"].get("target_id"),
                                               rows[c][arm]["final"].get("anchor_id"),
                                               rows[c][arm]["raw"].get("outcome")] for arm in arms},
                               "gold": [rows[c][arms[0]]["gold_target_ids"],
                                        rows[c][arms[0]]["gold_anchor_ids"]]})
    (run_dir / "discordant.json").write_text(json.dumps(discordant, indent=1))
    (run_dir / "per_case_metrics.json").write_text(json.dumps(metrics, indent=0))
    return numbers


def guard_counts(row: dict) -> dict[str, int]:
    """SAR guards that replaced the selector's person in one row, by guard name."""
    counts: dict[str, int] = {}
    for note in row.get("decisions") or []:
        if note.get("decision") == "geometry_override":
            for guard in note.get("guards") or [note.get("reason")]:
                counts[guard] = counts.get(guard, 0) + 1
    return counts


def audited_ids(audit_csv: Path, kind: str = "sar") -> set[str]:
    """Case IDs of ``kind`` that passed every check of the #87 human label audit."""
    with open(audit_csv, newline="", encoding="utf-8") as handle:
        return {r["case_id"] for r in csv.DictReader(handle)
                if r["kind"] == kind and r["label_correct"].strip().lower() == "y"}


def sar_report(run_dir: Path, root: Path, arms: list[str], prefix: str = "heldout-sar",
               audit_csv: Optional[Path] = None) -> dict:
    """The ``heldout_sar`` block: person exact match per arm, paired differences against C,
    guard overrides, unusable replies, per-recipe counts and the audited subset.

    Degenerate gold person boxes (< ``DEGENERATE_PX2``) are dropped from every number;
    detector misses are dropped from exact match only."""
    rows = load_rows(run_dir, arms, [prefix])
    table = case_table(root, set(rows))
    sar = sorted(c for c in rows if table[c]["kind"] == "sar")
    degenerate = [c for c in sar if table[c]["degenerate"]]
    ids = [c for c in sar if not table[c]["degenerate"]]
    metrics = {c: {arm: per_case_metrics(rows[c][arm], table[c]) for arm in arms} for c in ids}
    hit = [c for c in ids if not table[c]["miss"]]
    odd = [c for c in ids if table[c]["odd"]]
    block = {"cases": len(ids), "non_miss_cases": len(hit), "detector_miss": len(ids) - len(hit),
             "detector_miss_ids": [c for c in ids if table[c]["miss"]], "odd_cases": len(odd),
             "scenes": len({table[c]["scene"] for c in ids}),
             "excluded_degenerate_person_boxes": len(degenerate),
             "excluded_degenerate_ids": degenerate, "arms": {}, "diff": {}}
    for arm in arms:
        summary = arm_summary(metrics, table, arm)
        for key in ("target_credit_raw", "anchor_credit_raw", "credit_odd_raw"):
            summary[key] = None  # the harness projects raw answers for object cases only
        guards: dict[str, int] = {}
        effect = {"fixed": 0, "broke": 0, "neutral": 0}
        for c in ids:
            fired = guard_counts(rows[c][arm])
            for guard, n in fired.items():
                guards[guard] = guards.get(guard, 0) + n
            if fired and c in hit:
                raw, final = metrics[c][arm]["raw"]["joint"], metrics[c][arm]["final"]["joint"]
                effect["fixed" if final > raw else "broke" if raw > final else "neutral"] += 1
        summary.update(
            correct_raw=int(sum(metrics[c][arm]["raw"]["joint"] for c in hit)),
            correct_guarded=int(sum(metrics[c][arm]["final"]["joint"] for c in hit)),
            guard_overrides=guards, guard_effect_non_miss=effect,
            unusable_replies=int(sum(metrics[c][arm]["fallback"] for c in ids)))
        block["arms"][arm] = summary
    for other in [a for a in arms if a != "C"] if "C" in arms else []:
        block["diff"][f"C_minus_{other}"] = {
            "exact_guarded": paired(metrics, table, hit, "C", other, ("final", "joint")),
            "exact_raw": paired(metrics, table, hit, "C", other, ("raw", "joint")),
            "target_credit": paired(metrics, table, ids, "C", other, ("final", "target_credit")),
            "anchor_credit_odd": paired(metrics, table, odd, "C", other, ("final", "anchor_credit")),
        }
    block["recipes"] = {}
    for recipe in sorted({table[c]["recipe"] for c in hit}, key=str):
        sub = [c for c in hit if table[c]["recipe"] == recipe]
        block["recipes"][recipe] = {"n": len(sub), **{
            arm: int(sum(metrics[c][arm]["final"]["joint"] for c in sub)) for arm in arms}}
    if audit_csv is not None:
        sub = [c for c in hit if c in audited_ids(audit_csv)]
        block["audited_subset"] = {"n": len(sub), "ids": sub, **{
            arm: int(sum(metrics[c][arm]["final"]["joint"] for c in sub)) for arm in arms}}
    discordant = [{"case_id": c, "command": table[c]["command"], "recipe": table[c]["recipe"],
                   "correct": {arm: metrics[c][arm]["final"]["joint"] for arm in arms},
                   "picks": {arm: [rows[c][arm]["raw"].get("target_id"),
                                   rows[c][arm]["final"].get("target_id"),
                                   rows[c][arm]["raw"].get("outcome")] for arm in arms},
                   "gold": rows[c][arms[0]]["gold_target_ids"]}
                  for c in hit if len({metrics[c][arm]["final"]["joint"] for arm in arms}) > 1]
    (run_dir / "discordant.json").write_text(json.dumps(discordant, indent=1))
    (run_dir / "per_case_metrics.json").write_text(json.dumps(metrics, indent=0))
    return block


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--prefixes", nargs="+", default=["heldout"],
                        help="run files RUN_DIR/<prefix>-<arm>.jsonl, e.g. heldout heldout-sar")
    parser.add_argument("--out", type=Path, help="default RUN_DIR/scored.json")
    parser.add_argument("--sar", action="store_true",
                        help="score RUN_DIR/heldout-sar-<arm>.jsonl as the heldout_sar block")
    parser.add_argument("--audit", type=Path, help="#87 audit_filled.csv (SAR sanity subset)")
    parser.add_argument("--numbers", type=Path,
                        help="with --sar: add the block to this numbers.json under heldout_sar")
    args = parser.parse_args(argv)
    if args.sar:
        block = sar_report(args.run_dir, args.root, args.arms, audit_csv=args.audit)
        (args.out or args.run_dir / "scored.json").write_text(json.dumps({"heldout_sar": block}, indent=1))
        if args.numbers is not None:
            numbers = json.loads(args.numbers.read_text())
            numbers["heldout_sar"] = block
            args.numbers.write_text(json.dumps(numbers, indent=1))
        print(json.dumps({k: block[k] for k in ("cases", "non_miss_cases", "arms", "diff")}, indent=1))
        return
    numbers = report(args.run_dir, args.root, args.arms, args.prefixes)
    (args.out or args.run_dir / "scored.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers["heldout"]["arms"], indent=1))
    print(json.dumps(numbers["heldout"]["diff"], indent=1))


if __name__ == "__main__":
    main()
