"""Replay Stage 1 projection offline on logged rounds and frozen cases (#110).

    python -m scripts.stage1_replay --runs RUN_DIR [RUN_DIR ...] \
        [--cases SET_DIR ...] [--env KEY=VALUE ...] [--geometry JSON] --out OUT_DIR

Projection runs after selection, so a logged round can be recomposed without a GPU:
each round's logged candidates and selected target/anchor IDs are rebuilt into a
``Stage1CandidateResult`` and passed to the runtime ``compose_stage1_grounding``
with a ``CctvProjector`` over the captured camera geometry. The submission it
returns is rescored with ``scripts.stage1_runs.score``. ``--env`` sets runtime
flags for the replay only, e.g. ``MARC_GROUND_CALIBRATION=0``.

``--cases`` takes a frozen-case set folder (``<set>/cases/<id>/case.json``, #87
schema 1) and scores the runtime anchor projection (``project_anchor``) of each
case's gold anchor box against its GT ``anchor_xyz``.

Selection is held fixed: a change that would make the ranking or Qwen pick another
pair shows up only in a GPU run (``scripts.stage1_regress``).

Writes ``rounds.csv``, ``runs.csv``, ``cases.csv`` (with ``--cases``) and
``summary.md`` to OUT_DIR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest import mock

from marc_agent.command_parser import Intent, ParsedCommand, Relation, Situation
from marc_agent.experiment_log import decision_scope
from marc_agent.perception.cctv_project import CameraGeometry, CctvProjector, static_lookup
from marc_agent.perception.object_selection import ObjectSelection
from marc_agent.perception.sar_selection import SarSelectionResult, SelectionStatus
from marc_agent.perception.stage1_pipeline import (
    Stage1CandidateResult,
    compose_stage1_grounding,
    project_anchor,
)
from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate
from scripts.stage1_runs import (
    ANCHOR_ZERO_M,
    COMPONENTS,
    LoggedRound,
    LoggedRun,
    RoundScore,
    read_run,
    score,
)

DEFAULT_GEOMETRY = Path(__file__).resolve().parent / "data" / "sdk_cameras.json"
MOVED_M = 1e-3


def load_geometry(path: Path = DEFAULT_GEOMETRY) -> dict[str, CameraGeometry]:
    """Camera geometry captured from the live ``Tf2ClientLookup`` (see scripts/README.md)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))["geometry"]
    return {
        camera: CameraGeometry(
            fx=item["fx"], fy=item["fy"], cx=item["cx"], cy=item["cy"],
            rotation=tuple(tuple(float(v) for v in row) for row in item["rotation"]),
            translation=tuple(float(v) for v in item["translation"]),
            ground_height=float(item["ground_height"]),
        )
        for camera, item in payload.items()
    }


def projector_for(path: Path = DEFAULT_GEOMETRY) -> CctvProjector:
    return CctvProjector(static_lookup(load_geometry(path)))


def _parsed(record: dict) -> ParsedCommand:
    return ParsedCommand(
        intent=Intent(record["intent"]),
        target_type=record["target_type"],
        landmark=record.get("landmark"),
        relation=Relation(record["relation"]) if record.get("relation") else None,
        situation=Situation(record["situation"]) if record.get("situation") else None,
        scene_hint=record.get("scene_hint"),
        place_cues=tuple(record.get("place_cues") or ()),
    )


def _detections(round_: LoggedRound) -> CctvDetectionRound:
    """Logged candidates in their logged order (objects first, then people)."""
    objects, people = [], []
    for item in round_.pipeline["candidates"]:
        camera = item["camera_id"]
        if camera not in round_.image_sizes:
            raise SystemExit(f"round {round_.number}: no round dump frame for {camera}")
        common = dict(
            candidate_id=item["candidate_id"],
            bbox_xyxy=tuple(float(v) for v in item["bbox_xyxy"]),
            confidence=float(item["confidence"]),
            camera_id=camera,
            image_size=round_.image_sizes[camera],
        )
        # The record stores a person's pose class as its class name.
        if item["class_name"].startswith("person_"):
            people.append(PersonCandidate(pose_class=item["class_name"], **common))
        else:
            objects.append(ObjectCandidate(class_name=item["class_name"], **common))
    return CctvDetectionRound(
        objects=objects, people=people, camera_ids=list(round_.pipeline["camera_ids"])
    )


def rebuild(round_: LoggedRound) -> Stage1CandidateResult:
    """The candidate-stage result the runtime handed to ``compose_stage1_grounding``."""
    pipeline, submitted = round_.pipeline, round_.submission
    parsed = _parsed(pipeline["parsed"])
    detections = _detections(round_)
    by_id = {item.candidate_id: item for item in (*detections.objects, *detections.people)}
    target_id, anchor_id = submitted.get("target_id"), submitted.get("anchor_id")
    for note in round_.decisions:
        # The submitted pair is the fallback; composition started from the failed one.
        if note.get("decision") == "projection_fallthrough":
            target_id, anchor_id = note["failed_target"], note.get("failed_anchor")
    object_selection = sar_selection = None
    if target_id is not None and parsed.target_type == "person":
        sar_selection = SarSelectionResult(
            status=SelectionStatus.SELECTED,
            selected=by_id[target_id],
            requested_situation=None,
            evidence_situation=None,
            method=submitted.get("method") or "",
            reason="replay",
        )
    elif target_id is not None:
        object_selection = ObjectSelection(
            target=by_id[target_id],
            anchor=by_id[anchor_id] if anchor_id else None,
            score=float(pipeline.get("selection_score") or 0.0),
            method=submitted.get("method") or "",
        )
    return Stage1CandidateResult(
        command_text=pipeline["command"],
        parsed=parsed,
        detections=detections,
        sar_selection=sar_selection,
        failure_reason=pipeline.get("failure_reason"),
        total_ms=0.0,
        object_selection=object_selection,
    )


@dataclass(frozen=True)
class ReplayedRound:
    run: str
    round: LoggedRound
    submission: dict
    target_id: Optional[str]
    anchor_id: Optional[str]
    decisions: list[dict]
    logged: RoundScore
    replayed: RoundScore

    @property
    def target_moved_m(self) -> float:
        return math.dist(self.submission["target_coord"], self.round.submission["target_coord"])

    @property
    def anchor_moved_m(self) -> float:
        return math.dist(self.submission["anchor_coord"], self.round.submission["anchor_coord"])


def replay_round(run: str, round_: LoggedRound, projector: CctvProjector) -> ReplayedRound:
    with decision_scope("stage1", round_.number) as notes:
        result = compose_stage1_grounding(rebuild(round_), projector)
    if result.grounding is None:
        raise SystemExit(f"{run} round {round_.number}: replay produced no submission")
    grounding = result.grounding
    submission = {
        "camera_id": grounding.camera_id,
        "target_type": grounding.target_type,
        "landmark": grounding.landmark,
        "relation": grounding.relation,
        "situation": grounding.situation,
        "target_coord": list(grounding.target_coord),
        "anchor_coord": list(grounding.anchor_coord),
    }
    return ReplayedRound(
        run=run,
        round=round_,
        submission=submission,
        target_id=result.target_id,
        anchor_id=result.anchor_id,
        decisions=list(notes),
        logged=score(round_, round_.detail["submission"]),
        replayed=score(round_, submission),
    )


def replay_run(run: LoggedRun, projector: CctvProjector) -> list[ReplayedRound]:
    missing = [r.problem_id for r in run.rounds if r.pipeline is None or r.submission is None]
    if missing:
        raise SystemExit(f"{run.name}: rounds without logged pipeline/submission: {missing}")
    return [replay_round(run.name, round_, projector) for round_ in run.rounds]


@dataclass(frozen=True)
class AnchorCase:
    case_id: str
    kind: str
    anchor_class: str
    camera_id: str
    error_m: Optional[float]  # None when the anchor box does not project

    @property
    def credit(self) -> float:
        return 0.0 if self.error_m is None else max(0.0, 1.0 - self.error_m / ANCHOR_ZERO_M)


def replay_cases(set_dir: Path, projector: CctvProjector) -> tuple[list[AnchorCase], int]:
    """Anchor error of each case whose gold anchor was detected; also the count skipped."""
    rows, skipped = [], 0
    for path in sorted(Path(set_dir).glob("cases/*/case.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        gold = case["gold"]
        candidates = {item["candidate_id"]: item for item in case["candidates"]}
        anchors = [candidates[cid] for cid in gold.get("anchor_ids") or [] if cid in candidates]
        if not anchors or not gold.get("anchor_xyz"):
            skipped += 1
            continue
        # Several detections can match one GT box; take the one the runtime trusts most.
        anchor = max(anchors, key=lambda item: (item["confidence"], item["candidate_id"]))
        xyz = project_anchor(projector, anchor["camera_id"], tuple(anchor["bbox_xyxy"]))
        rows.append(AnchorCase(
            case_id=case["case_id"],
            kind=case["kind"],
            anchor_class=anchor["class_name"],
            camera_id=anchor["camera_id"],
            error_m=None if xyz is None else math.dist(xyz, gold["anchor_xyz"]),
        ))
    return rows, skipped


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.fmean(values):.3f} ± {sd:.3f}"


def _xyz(values) -> str:
    return " ".join(f"{v:.3f}" for v in values)


def round_rows(replayed: list[ReplayedRound]) -> list[dict]:
    rows = []
    for item in replayed:
        logged, new = item.logged, item.replayed
        row = {
            "run": item.run,
            "round": item.round.number,
            "problem_id": item.round.problem_id,
            "logged_total": round(item.round.detail["total"], 4),
            "replay_total": round(new.total, 4),
            "delta": round(new.total - item.round.detail["total"], 4),
            "target_moved_m": round(item.target_moved_m, 4),
            "anchor_moved_m": round(item.anchor_moved_m, 4),
            "logged_target_m": round(logged.target_m, 3),
            "replay_target_m": round(new.target_m, 3),
            "logged_anchor_m": round(logged.anchor_m, 3),
            "replay_anchor_m": round(new.anchor_m, 3),
            "target_uncertain": new.target_uncertain,
            "replay_target_xyz": _xyz(item.submission["target_coord"]),
            "replay_anchor_xyz": _xyz(item.submission["anchor_coord"]),
            "target_id": item.target_id,
            "anchor_id": item.anchor_id,
            "landmark": item.submission["landmark"],
            "relation": item.submission["relation"],
            "decisions": " ".join(sorted({note["decision"] for note in item.decisions})),
        }
        for name in COMPONENTS:
            row[f"logged_{name}"] = round(item.round.detail[name], 4)
            row[f"replay_{name}"] = round(new.components[name], 4)
        rows.append(row)
    return rows


def run_rows(runs: list[LoggedRun], replayed: list[ReplayedRound]) -> list[dict]:
    rows = []
    for run in runs:
        items = [item for item in replayed if item.run == run.name]

        def mean(value) -> float:
            return round(statistics.fmean(value(item) for item in items), 4)

        replay_avg = mean(lambda i: i.replayed.total)
        rows.append({
            "run": run.name,
            "logged_average": round(run.average, 4),
            "recomputed_average": mean(lambda i: i.logged.total),
            "replay_average": replay_avg,
            "delta": round(replay_avg - run.average, 4),
            "logged_anchor": mean(lambda i: i.logged.components["anchor"]),
            "replay_anchor": mean(lambda i: i.replayed.components["anchor"]),
            "logged_target": mean(lambda i: i.logged.components["target"]),
            "replay_target": mean(lambda i: i.replayed.components["target"]),
        })
    return rows


def summary(
    runs: list[LoggedRun],
    replayed: list[ReplayedRound],
    cases: dict[str, tuple[list[AnchorCase], int]],
    env: dict[str, str],
    geometry: Path,
    seconds: float,
) -> str:
    lines = ["# Stage 1 projection replay", ""]
    lines.append(f"- Geometry: `{geometry}`")
    lines.append("- Env: " + (", ".join(f"`{k}={v}`" for k, v in env.items()) or "none"))
    lines.append(f"- Replay time: {seconds:.2f} s")
    if replayed:
        def per_run(value) -> list[float]:
            return [
                statistics.fmean(value(item) for item in replayed if item.run == run.name)
                for run in runs
            ]

        moved = max(max(i.target_moved_m, i.anchor_moved_m) for i in replayed)
        lines += [
            f"- Runs: {len(runs)}; rounds: {len(replayed)}",
            f"- Largest coordinate change vs the log: {moved * 1000:.3f} mm",
            "",
            "| per run (mean ± sd) | logged | replay |",
            "|---|---|---|",
            f"| Stage 1 average | {_mean_sd([run.average for run in runs])} "
            f"| {_mean_sd(per_run(lambda i: i.replayed.total))} |",
            f"| anchor credit | {_mean_sd(per_run(lambda i: i.logged.components['anchor']))} "
            f"| {_mean_sd(per_run(lambda i: i.replayed.components['anchor']))} |",
            f"| target credit | {_mean_sd(per_run(lambda i: i.logged.components['target']))} "
            f"| {_mean_sd(per_run(lambda i: i.replayed.components['target']))} |",
            "",
        ]
        flips = [i for i in replayed if i.replayed.components["target"] != i.round.detail["target"]]
        uncertain = [i for i in replayed if i.replayed.target_uncertain]
        lines.append(f"Target credit flips: {len(flips)}")
        for item in flips:
            lines.append(
                f"- {item.run} {item.round.problem_id}: {item.round.detail['target']:.0f} → "
                f"{item.replayed.components['target']:.0f} "
                f"({item.logged.target_m:.2f} m → {item.replayed.target_m:.2f} m)"
            )
        lines.append("")
        lines.append(
            "Moved targets credited beyond any distance the platform has credited "
            f"for that round (uncertain credit): {len(uncertain)}"
        )
        lines.append("")
        by_problem: dict[str, list[ReplayedRound]] = {}
        for item in replayed:
            by_problem.setdefault(item.round.problem_id, []).append(item)
        lines += [
            "| round | logged total | replay total | Δ | anchor m (logged → replay) |",
            "|---|---|---|---|---|",
        ]
        for problem, items in by_problem.items():
            logged = statistics.fmean(i.round.detail["total"] for i in items)
            new = statistics.fmean(i.replayed.total for i in items)
            lines.append(
                f"| {problem} | {logged:.2f} | {new:.2f} | {new - logged:+.2f} | "
                f"{statistics.fmean(i.logged.anchor_m for i in items):.2f} → "
                f"{statistics.fmean(i.replayed.anchor_m for i in items):.2f} |"
            )
        lines.append("")
    for set_dir, (rows, skipped) in cases.items():
        errors = [row.error_m for row in rows if row.error_m is not None]
        lines += [
            f"## Anchor replay on `{set_dir}`",
            "",
            f"- Cases with a detected gold anchor: {len(rows)} "
            f"(skipped without one: {skipped}; not projected: {len(rows) - len(errors)})",
            f"- Mean 3-D anchor error: {statistics.fmean(errors):.3f} m"
            f" (median {statistics.median(errors):.3f} m)" if errors else "- No projected anchors",
            f"- Mean anchor credit: {statistics.fmean(r.credit for r in rows):.4f}" if rows else "",
            "",
            "| anchor class | n | mean error m | mean credit |",
            "|---|---|---|---|",
        ]
        by_class: dict[str, list[AnchorCase]] = {}
        for row in rows:
            by_class.setdefault(row.anchor_class, []).append(row)
        for name, items in sorted(by_class.items()):
            errs = [r.error_m for r in items if r.error_m is not None]
            lines.append(
                f"| {name} | {len(items)} | "
                f"{statistics.fmean(errs) if errs else float('nan'):.3f} | "
                f"{statistics.fmean(r.credit for r in items):.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _env(pairs: list[str]) -> dict[str, str]:
    env = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--env expects KEY=VALUE, got {pair!r}")
        env[key] = value
    return env


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", type=Path, default=[])
    parser.add_argument("--cases", nargs="*", type=Path, default=[])
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.runs and not args.cases:
        parser.error("give --runs and/or --cases")
    env = _env(args.env)
    projector = projector_for(args.geometry)
    runs = [read_run(path) for path in args.runs]
    started = time.perf_counter()
    with mock.patch.dict(os.environ, env):
        replayed = [item for run in runs for item in replay_run(run, projector)]
        cases = {str(path): replay_cases(path, projector) for path in args.cases}
    seconds = time.perf_counter() - started
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "rounds.csv", round_rows(replayed))
    _write_csv(args.out / "runs.csv", run_rows(runs, replayed) if runs else [])
    case_rows = [
        {"set": name, "case_id": r.case_id, "kind": r.kind, "anchor_class": r.anchor_class,
         "camera_id": r.camera_id,
         "error_m": None if r.error_m is None else round(r.error_m, 4),
         "credit": round(r.credit, 4)}
        for name, (rows, _) in cases.items() for r in rows
    ]
    _write_csv(args.out / "cases.csv", case_rows)
    text = summary(runs, replayed, cases, env, args.geometry, seconds)
    (args.out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
