"""Score statistics over repeated Stage 1 runs (#88, reused by #90).

    python -m scripts.stage1_scores RUN_DIR [RUN_DIR ...] --out OUT_DIR [--label NAME]

Each RUN_DIR is a harness ``run-NN/`` folder: the platform score file
(``robust_*.json``) plus ``experiments/<run>/events.jsonl``. Writes to OUT_DIR:

- ``runs.csv``: Stage 1 average and the mean of each component per run;
- ``rounds.csv``: per round over runs: score range, answer changes, lost points;
- ``summary.md``: headline statistics, a cluster-bootstrap CI, round times;
- ``strip.png``: every run's Stage 1 average.

A component is averaged only over rounds where its weight is non-zero; SAR
rounds score landmark and relation with weight 0. Two runs give the same
answer to a round when they submit the same camera and target and anchor
coordinates within ``SAME_ANSWER_M``; candidate IDs alone are not enough,
because one ID can project to different points. Configuration clusters group rounds
whose ``expected`` fields match, so repeated public configurations count once
in the bootstrap.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

COMPONENTS = ("camera", "object_type", "landmark", "relation", "anchor", "target")
LOST_TOTAL = 95.0
SAME_ANSWER_M = 0.5


@dataclass
class RoundScore:
    problem_id: str
    total: float
    components: dict[str, Optional[float]]
    cluster: tuple
    answer: Optional[tuple] = None  # (camera, target xyz, anchor xyz)
    sim_s: Optional[float] = None
    time_limit_s: Optional[float] = None


@dataclass
class RunScores:
    name: str
    average: float
    rounds: list[RoundScore] = field(default_factory=list)


def _cluster(expected: dict) -> tuple:
    coord = tuple(round(value, 1) for value in expected.get("target_coord") or ())
    return (
        expected.get("camera_id"), expected.get("target_type"), expected.get("relation"),
        expected.get("landmark_name"), expected.get("situation"), coord,
    )


def _round_times(events_path: Path) -> dict[str, dict]:
    """Simulation round time and time limit, keyed by problem ID."""
    received, submitted, problems = {}, {}, {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        name, number = event.get("event"), event.get("round")
        if number is None:
            continue
        if name == "stage1.round.received":
            received[number] = event
        elif name == "stage1.submission.completed":
            submitted[number] = event
        elif name == "score.received" and not event.get("is_final"):
            problems[number] = (event.get("scores") or {}).get("problem_id")
    out = {}
    for number, event in submitted.items():
        start = received.get(number, {})
        sim_s = None
        if "simulation_timestamp" in event and "simulation_timestamp" in start:
            sim_s = event["simulation_timestamp"] - start["simulation_timestamp"]
        out[problems.get(number) or str(number)] = {
            "sim_s": sim_s,
            "time_limit_s": start.get("time_limit_s"),
        }
    return out


def read_run(run_dir: Path) -> RunScores:
    score_files = sorted(run_dir.glob("robust_*.json"))
    if len(score_files) != 1:
        raise SystemExit(f"{run_dir}: expected one score file, found {len(score_files)}")
    data = json.loads(score_files[0].read_text(encoding="utf-8"))
    events = sorted(run_dir.glob("experiments/*/events.jsonl"))
    submitted = _round_times(events[-1]) if events else {}
    run = RunScores(run_dir.name, data["scores"]["stage1_average"])
    for detail in data["round_details"]:
        weights = detail.get("weights") or {}
        extra = submitted.get(detail["problem_id"], {})
        submission = detail.get("submission") or {}
        run.rounds.append(RoundScore(
            problem_id=detail["problem_id"],
            total=detail["total"],
            components={
                name: detail.get(name) if weights.get(name, 1.0) > 0 else None
                for name in COMPONENTS
            },
            cluster=_cluster(detail.get("expected") or {}),
            answer=(
                submission.get("camera_id"),
                tuple(submission.get("target_coord") or ()),
                tuple(submission.get("anchor_coord") or ()),
            ),
            sim_s=extra.get("sim_s"),
            time_limit_s=extra.get("time_limit_s"),
        ))
    return run


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "q1": _quantile(values, 0.25),
        "q3": _quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def component_means(run: RunScores) -> dict[str, float]:
    means = {}
    for name in COMPONENTS:
        values = [item.components[name] for item in run.rounds if item.components[name] is not None]
        means[name] = statistics.fmean(values) if values else float("nan")
    return means


def cluster_bootstrap(runs: list[RunScores], draws: int = 10000, seed: int = 88) -> dict:
    """95% CI of the Stage 1 average, resampling configuration clusters.

    Each round's score is first averaged over runs, so the interval reflects
    problem-level uncertainty, not run-to-run noise.
    """
    by_problem: dict[str, list[float]] = {}
    clusters: dict[tuple, list[str]] = {}
    for run in runs:
        for item in run.rounds:
            by_problem.setdefault(item.problem_id, []).append(item.total)
            members = clusters.setdefault(item.cluster, [])
            if item.problem_id not in members:
                members.append(item.problem_id)
    round_mean = {problem: statistics.fmean(scores) for problem, scores in by_problem.items()}
    groups = [[round_mean[problem] for problem in members] for members in clusters.values()]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = [score for group in rng.choices(groups, k=len(groups)) for score in group]
        estimates.append(statistics.fmean(sample))
    return {
        "clusters": len(groups),
        "point": statistics.fmean(round_mean.values()),
        "low": _quantile(estimates, 0.025),
        "high": _quantile(estimates, 0.975),
    }


def answer_groups(answers: list[tuple]) -> list[int]:
    """Sizes of the groups of answers that agree with each group's first answer."""
    firsts: list[tuple] = []
    sizes: list[int] = []
    for camera, target, anchor in answers:
        for index, (first_camera, first_target, first_anchor) in enumerate(firsts):
            if (
                camera == first_camera
                and math.dist(target, first_target) <= SAME_ANSWER_M
                and math.dist(anchor, first_anchor) <= SAME_ANSWER_M
            ):
                sizes[index] += 1
                break
        else:
            firsts.append((camera, target, anchor))
            sizes.append(1)
    return sizes


def round_table(runs: list[RunScores]) -> list[dict]:
    order = [item.problem_id for item in runs[0].rounds]
    rows = []
    for problem in order:
        items = [item for run in runs for item in run.rounds if item.problem_id == problem]
        totals = [item.total for item in items]
        groups = answer_groups([item.answer for item in items])
        modal = max(groups) if groups else 0
        lost = sorted({
            name for item in items for name in COMPONENTS
            if name != "anchor" and item.components[name] is not None and item.components[name] < 1.0
        })
        rows.append({
            "problem_id": problem,
            "runs": len(items),
            "mean": statistics.fmean(totals),
            "min": min(totals),
            "max": max(totals),
            "distinct_answers": len(groups),
            "answer_changes": len(items) - modal,
            "mean_anchor": statistics.fmean(
                item.components["anchor"] for item in items
                if item.components["anchor"] is not None
            ),
            "lost_components": " ".join(lost),
            "runs_below_95": sum(total < LOST_TOTAL for total in totals),
        })
    return rows


def _p95(values: list[float]) -> Optional[float]:
    return _quantile(values, 0.95) if values else None


def _f(value, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(runs: list[RunScores], label: str) -> str:
    out = io.StringIO()
    averages = [run.average for run in runs]
    stats = describe(averages)
    out.write(f"# Stage 1 scores: {label}, {len(runs)} run(s)\n\n")
    out.write("## Stage 1 average\n\n")
    out.write("| n | mean | sd | median | IQR | min | max | best-of-N (development) |\n")
    out.write("|---:|---:|---:|---:|---|---:|---:|---:|\n")
    out.write(
        f"| {stats['n']} | {stats['mean']:.3f} | {stats['sd']:.3f} | {stats['median']:.3f} | "
        f"{stats['q1']:.3f}–{stats['q3']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | "
        f"{stats['max']:.3f} (N={stats['n']}) |\n\n"
    )
    out.write("Per run: " + ", ".join(f"{run.name} {run.average:.3f}" for run in runs) + "\n\n")
    boot = cluster_bootstrap(runs)
    out.write(
        f"Problem-level 95% CI (bootstrap over {boot['clusters']} configuration clusters, "
        f"round scores averaged over runs): {boot['point']:.3f} "
        f"[{boot['low']:.3f}, {boot['high']:.3f}].\n\n"
    )
    out.write("## Components (run means; each over rounds where it has weight)\n\n")
    out.write("| component | rounds/run | mean | sd | min | max |\n|---|---:|---:|---:|---:|---:|\n")
    per_run = [component_means(run) for run in runs]
    for name in COMPONENTS:
        values = [means[name] for means in per_run]
        counted = sum(item.components[name] is not None for item in runs[0].rounds)
        part = describe(values)
        out.write(
            f"| {name} | {counted} | {part['mean']:.4f} | {part['sd']:.4f} | "
            f"{part['min']:.4f} | {part['max']:.4f} |\n"
        )
    sim = [item.sim_s for run in runs for item in run.rounds if item.sim_s is not None]
    limits = {item.time_limit_s for run in runs for item in run.rounds if item.time_limit_s}
    out.write("\n## Round time (simulation time, pooled over runs)\n\n")
    out.write(
        f"p95 {_f(_p95(sim), 2)} s, max {_f(max(sim) if sim else None, 2)} s, "
        f"over {len(sim)} rounds; mission time limit(s): "
        f"{', '.join(f'{limit:g} s' for limit in sorted(limits)) or 'not recorded'}; "
        f"rounds over the limit: "
        f"{sum(1 for run in runs for item in run.rounds if item.sim_s and item.time_limit_s and item.sim_s > item.time_limit_s)}.\n\n"
    )
    rows = round_table(runs)
    changed = [row for row in rows if row["answer_changes"]]
    out.write("## Per-round stability\n\n")
    out.write(
        f"{len(changed)} of {len(rows)} rounds changed their submitted answer "
        f"(camera, or target or anchor moved more than {SAME_ANSWER_M:g} m) in at least one run; "
        f"{sum(row['answer_changes'] for row in rows)} answer changes in total "
        f"against each round's most common answer.\n\n"
    )
    keys = ["problem_id", "mean", "min", "max", "distinct_answers", "answer_changes",
            "mean_anchor", "lost_components", "runs_below_95"]
    out.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
    for row in rows:
        out.write("| " + " | ".join(_f(row[key], 2) for key in keys) + " |\n")
    out.write(
        f"\n`lost_components` lists any non-anchor component below 1 in any run; anchor "
        f"credit is partial in most rounds and is summarised by `mean_anchor`. "
        f"`runs_below_95` counts runs where the round total was below {LOST_TOTAL:g}.\n"
    )
    return out.getvalue()


def strip_plot(groups: dict[str, list[float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    rng = random.Random(0)
    for index, (label, values) in enumerate(groups.items()):
        xs = [index + rng.uniform(-0.12, 0.12) for _ in values]
        ax.scatter(xs, values, s=22, color="#2f6db5", alpha=0.8, zorder=3)
        ax.hlines(statistics.fmean(values), index - 0.25, index + 0.25, color="#222222", lw=1.5)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("Stage 1 average (/100)")
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write(runs: list[RunScores], out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "stage1_average", *COMPONENTS])
        for run in runs:
            means = component_means(run)
            writer.writerow([run.name, run.average, *(means[name] for name in COMPONENTS)])
    rows = round_table(runs)
    with (out_dir / "rounds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.md").write_text(markdown(runs, label), encoding="utf-8")
    strip_plot({label: [run.average for run in runs]}, out_dir / "strip.png")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--label", default="runs")
    args = parser.parse_args(argv)
    runs = [read_run(path) for path in args.run_dirs]
    if args.out:
        write(runs, args.out, args.label)
    else:
        print(markdown(runs, args.label))


if __name__ == "__main__":
    main()
