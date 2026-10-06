"""Count how every Stage 1 round was decided, from experiment run folders (#85).

    python -m scripts.stage1_decisions RUN_DIR [RUN_DIR ...] --out OUT_DIR

Reads each run's ``events.jsonl`` and writes ``rounds.csv`` (one row per round),
``runs.csv`` (one row per run) and ``summary.md`` to OUT_DIR. Without --out it
prints the Markdown to stdout.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class Round:
    run: str
    round: int
    problem_id: Optional[str] = None
    command: Optional[str] = None
    total: Optional[float] = None
    camera: Optional[float] = None
    target: Optional[float] = None
    anchor: Optional[float] = None
    round_time_s: Optional[float] = None
    round_sim_s: Optional[float] = None
    cameras: int = 0
    frames: int = 0
    coverage_complete: Optional[bool] = None
    grounding_method: Optional[str] = None
    selector_path: Optional[str] = None
    selector_outcome: Optional[str] = None
    selector_reason: Optional[str] = None
    override_guards: Optional[str] = None
    qwen_skipped: Optional[str] = None
    scene_description_failed: bool = False
    class_screen_failures: int = 0
    refine_added: int = 0
    landmark_proxy: bool = False
    projection_proxy: Optional[str] = None
    fallthrough_rank: Optional[int] = None
    anchor_collapsed: bool = False
    sentinel: Optional[str] = None
    cache_source_round: Optional[int] = None
    cache_same_pair: Optional[bool] = None
    cache_target_shift_m: Optional[float] = None

    @property
    def path(self) -> str:
        """The single label used for the paper's decision-path table."""
        if self.cache_source_round is not None:
            return "cache reuse"
        if self.sentinel:
            return "sentinel coordinate"
        if self.selector_outcome == "fallback":
            return "Qwen fallback"
        if self.selector_outcome == "geometry_override":
            return "geometry override"
        if self.selector_outcome == "geometry_only":
            return "geometry only (no Qwen)"
        if self.selector_outcome == "qwen_choice":
            return "Qwen choice accepted"
        return "unknown"


@dataclass
class Run:
    name: str
    rounds: list[Round] = field(default_factory=list)


def _events(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def read_run(run_dir: Path) -> Run:
    """One run folder (with events.jsonl) as a list of decided rounds."""
    run = Run(run_dir.name)
    rounds: dict[int, Round] = {}
    received: dict[int, dict] = {}

    def row(number: int) -> Round:
        return rounds.setdefault(number, Round(run.name, number))

    for event in _events(run_dir / "events.jsonl"):
        name = event.get("event")
        number = event.get("round")
        if name == "stage1.round.received" and number is not None:
            received[number] = event
        elif name == "stage1.pipeline.completed" and number is not None:
            _apply_pipeline(row(number), event)
        elif name == "stage1.submission.completed" and number is not None:
            start = received.get(number)
            if start is not None:
                item = row(number)
                item.round_time_s = event["monotonic_offset_s"] - start["monotonic_offset_s"]
                if "simulation_timestamp" in event and "simulation_timestamp" in start:
                    item.round_sim_s = (
                        event["simulation_timestamp"] - start["simulation_timestamp"]
                    )
        elif name == "score.received" and number is not None and not event.get("is_final"):
            scores = event.get("scores") or {}
            item = row(number)
            item.problem_id = scores.get("problem_id")
            item.total = scores.get("total", event.get("total"))
            item.camera = scores.get("camera")
            item.target = scores.get("target")
            item.anchor = scores.get("anchor")
    run.rounds = [rounds[key] for key in sorted(rounds)]
    return run


def _apply_pipeline(item: Round, event: dict) -> None:
    item.command = event.get("command")
    cameras = set(event.get("camera_ids") or [])
    frames = set(event.get("frames_loaded") or [])
    item.cameras = len(cameras)
    item.frames = len(frames)
    item.coverage_complete = event.get(
        "camera_coverage_complete", bool(cameras) and cameras <= frames
    )
    item.grounding_method = event.get("grounding_method")
    for note in event.get("decisions") or []:
        kind = note.get("decision")
        if kind == "selector":
            item.selector_path = note.get("path")
            item.selector_outcome = note.get("outcome")
            item.selector_reason = note.get("reason")
        elif kind == "geometry_override":
            guards = note.get("guards") or [note.get("reason")]
            item.override_guards = " > ".join(str(guard) for guard in guards)
        elif kind == "qwen_skipped":
            item.qwen_skipped = note.get("reason")
        elif kind == "scene_description_failed":
            item.scene_description_failed = True
        elif kind == "class_screen_failed":
            item.class_screen_failures += 1
        elif kind == "refine_added":
            item.refine_added += len(note.get("candidates") or [])
        elif kind == "proxy_created":
            item.landmark_proxy = True
        elif kind == "projection_proxy":
            item.projection_proxy = note.get("variant")
        elif kind == "projection_fallthrough":
            item.fallthrough_rank = note.get("rank")
        elif kind == "anchor_collapsed":
            item.anchor_collapsed = True
        elif kind == "sentinel":
            item.sentinel = note.get("reason")
        elif kind == "cache_hit":
            item.cache_source_round = note.get("source_round")
            item.cache_same_pair = note.get("same_pair")
            item.cache_target_shift_m = note.get("target_shift_m")


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _p95(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def run_totals(run: Run) -> dict:
    rounds = run.rounds
    times = [item.round_time_s for item in rounds if item.round_time_s is not None]
    sim_times = [item.round_sim_s for item in rounds if item.round_sim_s is not None]
    scores = [item.total for item in rounds if item.total is not None]
    return {
        "run": run.name,
        "rounds": len(rounds),
        "mean_score": _mean(scores),
        "qwen_choice": sum(item.selector_outcome == "qwen_choice" for item in rounds),
        "qwen_fallback": sum(item.selector_outcome == "fallback" for item in rounds),
        "geometry_only": sum(item.selector_outcome == "geometry_only" for item in rounds),
        "geometry_override": sum(item.selector_outcome == "geometry_override" for item in rounds),
        "qwen_skipped_time": sum(item.qwen_skipped == "time_budget" for item in rounds),
        "landmark_proxy": sum(item.landmark_proxy for item in rounds),
        "projection_proxy": sum(item.projection_proxy is not None for item in rounds),
        "fallthrough": sum(item.fallthrough_rank is not None for item in rounds),
        "anchor_collapsed": sum(item.anchor_collapsed for item in rounds),
        "sentinel": sum(item.sentinel is not None for item in rounds),
        "cache_hit": sum(item.cache_source_round is not None for item in rounds),
        "cache_disagree": sum(item.cache_same_pair is False for item in rounds),
        "incomplete_coverage": sum(item.coverage_complete is False for item in rounds),
        "p95_round_wall_s": _p95(times),
        "max_round_wall_s": max(times) if times else None,
        "p95_round_sim_s": _p95(sim_times),
        "max_round_sim_s": max(sim_times) if sim_times else None,
    }


def path_table(runs: list[Run]) -> list[dict]:
    """Rounds, share and mean score for each decision path, over all runs."""
    rounds = [item for run in runs for item in run.rounds]
    groups: dict[str, list[Round]] = {}
    for item in rounds:
        groups.setdefault(item.path, []).append(item)
    detail = Counter()
    for item in rounds:
        if item.path == "cache reuse":
            continue  # the selector note describes the discarded fresh answer
        if item.selector_outcome in {"fallback", "geometry_override", "geometry_only"}:
            detail[(item.path, item.selector_reason)] += 1
        if item.sentinel:
            detail[("sentinel coordinate", item.sentinel)] += 1
    table = []
    for name, items in sorted(groups.items(), key=lambda pair: -len(pair[1])):
        scores = [item.total for item in items if item.total is not None]
        reasons = ", ".join(
            f"{reason} {count}" for (path, reason), count in sorted(detail.items())
            if path == name and reason
        )
        table.append({
            "path": name,
            "rounds": len(items),
            "share": len(items) / len(rounds) if rounds else 0.0,
            "mean_score": _mean(scores),
            "reasons": reasons,
        })
    return table


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown(runs: list[Run]) -> str:
    out = io.StringIO()
    out.write(f"# Stage 1 decision paths: {len(runs)} run(s)\n\n")
    out.write("## Decision paths over all rounds\n\n")
    out.write("| Path | Rounds | Share | Mean round score | Reasons |\n|---|---:|---:|---:|---|\n")
    for row in path_table(runs):
        out.write(
            f"| {row['path']} | {row['rounds']} | {row['share']:.1%} | "
            f"{_fmt(row['mean_score'])} | {row['reasons']} |\n"
        )
    out.write("\nA round counts once, under its first matching path: cache reuse, "
              "sentinel, Qwen fallback, geometry override, geometry only, Qwen choice.\n")
    out.write("Proxies and fall-throughs can happen on any path; they are counted "
              "separately below.\n\n## Per run\n\n")
    totals = [run_totals(run) for run in runs]
    columns = list(totals[0]) if totals else []
    out.write("| " + " | ".join(columns) + " |\n")
    out.write("|" + "---|" * len(columns) + "\n")
    for row in totals:
        out.write("| " + " | ".join(_fmt(row[key]) for key in columns) + " |\n")
    out.write("\n## Per round\n\n")
    keys = ["run", "round", "problem_id", "total", "path", "selector_reason",
            "override_guards",
            "landmark_proxy", "projection_proxy", "fallthrough_rank",
            "anchor_collapsed", "sentinel", "cache_source_round", "cache_same_pair",
            "coverage_complete", "round_time_s", "round_sim_s"]
    out.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
    for run in runs:
        for item in run.rounds:
            values = {**asdict(item), "path": item.path}
            out.write("| " + " | ".join(_fmt(values[key]) for key in keys) + " |\n")
    return out.getvalue()


def write(runs: list[Run], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(item), "path": item.path} for run in runs for item in run.rounds]
    if rows:
        with (out_dir / "rounds.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    totals = [run_totals(run) for run in runs]
    if totals:
        with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(totals[0]))
            writer.writeheader()
            writer.writerows(totals)
    (out_dir / "summary.md").write_text(markdown(runs), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    runs = [read_run(path) for path in args.run_dirs]
    if args.out:
        write(runs, args.out)
    else:
        print(markdown(runs))


if __name__ == "__main__":
    main()
