"""Rescore logged Stage 1 rounds under the exact kernel on a constant grid.

    python -m scripts.stage1_sensitivity \
        --run ~/marc-2026-work/experiments/ablations/final-system/review-improvements/baseline-repeats/run-01 \
        --scenario ~/marc-2026-work/experiments/ablations/final-system/review-improvements/scenario/marc2026_chungmu.yaml \
        --out ~/marc-2026-work/experiments/ablations/final-system/135-good-faith/sensitivity

The grid is margin 0–0.3 m, acquire radius 1–2 m, miss radius 3–5 m. It does not
tune those constants. A round is listed when its target credit falls to 0 at a
grid point where the logged submission currently passes.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import yaml

from scripts.stage1_runs import read_run, target_credit


def distractors_by_problem(scenario: Path) -> dict[str, dict]:
    """problem id -> target, anchor, and the other instances in that camera."""
    doc = yaml.safe_load(Path(scenario).read_text(encoding="utf-8"))
    missions = {mission["id"]: mission for mission in doc["missions"]}
    out = {}
    for problem in doc["challenge_problems"]:
        expected = problem["expected"]
        mission = missions[expected["camera_id"]]
        _camera, rest = expected["target_object_id"].split("/", 1)
        type_name, index_text = rest.rsplit("_", 1)
        index = int(index_text)
        target = None
        others = []
        for obj in mission.get("objects") or []:
            for inst_index, inst in enumerate(obj["resolved"]["instances"]):
                point = inst["position"][:3]
                if obj["type"] == type_name and inst_index == index:
                    target = point
                else:
                    others.append(point)
        anchor = next(
            landmark["resolved"]["position"][:3]
            for landmark in mission["landmarks"]
            if landmark["name"] == expected["landmark_name"]
        )
        out[problem["id"]] = {
            "target_coord": target,
            "anchor_coord": anchor,
            "distractor_coords": others,
        }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    problems = distractors_by_problem(args.scenario)
    logged = read_run(args.run)
    margins = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
    acquires = (1.0, 1.5, 2.0)
    misses = (3.0, 4.0, 5.0)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for round_ in logged.rounds:
        problem = problems.get(round_.problem_id)
        submission = round_.detail.get("submission") or {}
        point = submission.get("target_coord")
        if problem is None or point is None or problem["target_coord"] is None:
            continue
        base = target_credit(
            point, problem["target_coord"], problem["distractor_coords"],
            r_acquire=2.0, r_miss=5.0, margin=0.0,
        )
        for margin, acquire, miss in itertools.product(margins, acquires, misses):
            credit = target_credit(
                point, problem["target_coord"], problem["distractor_coords"],
                r_acquire=acquire, r_miss=miss, margin=margin,
            )
            if credit == 0.0 and base > 0.0:
                rows.append({
                    "problem_id": round_.problem_id,
                    "margin_m": margin,
                    "r_acquire_m": acquire,
                    "r_miss_m": miss,
                    "base_credit": round(base, 4),
                    "credit": 0.0,
                    "logged_target": round_.detail.get("target"),
                })
    path = args.out / "attribution_failures.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "problem_id", "margin_m", "r_acquire_m", "r_miss_m",
            "base_credit", "credit", "logged_target",
        ])
        writer.writeheader()
        writer.writerows(rows)
    summary = args.out / "summary.md"
    by_problem: dict[str, list] = {}
    for row in rows:
        by_problem.setdefault(row["problem_id"], []).append(row)
    lines = [
        "# Scorer-constant sensitivity",
        "",
        f"Run `{args.run}`. Defaults (margin 0, acquire 2 m, miss 5 m) are the baseline.",
        f"Grid points where a currently credited target falls to 0: {len(rows)}.",
        "",
    ]
    for problem_id in sorted(by_problem):
        sample = by_problem[problem_id][0]
        lines.append(
            f"- {problem_id}: {len(by_problem[problem_id])} failing grid points "
            f"(logged target credit {sample['logged_target']})"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
