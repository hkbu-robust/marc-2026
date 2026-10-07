"""Logged Stage 1 rounds of a #85 harness run folder, and the platform's scoring kernel.

A run folder (``run-NN/``) holds the platform score file ``robust_*.json`` and one
``experiments/<run>/events.jsonl``; round dumps sit under that experiment folder in
``artifacts/stage1/round-NNN/cameras/<camera>.png``. ``read_run`` joins, per round,
what the platform scored with what the agent logged.

``score`` recomputes a round from a submission. No organiser document publishes the
scorer; these rules reproduce all 330 #88 cache-off rounds: categorical components
are exact matches (a SAR round's "type" is its situation), anchor credit is
``max(0, 1 - d/5 m)`` on the 3-D distance, and the total is ``100 * sum(w*c) / sum(w)``
with the round's logged weights (SAR rounds weight landmark and relation 0).
Target credit was binary there: 1 up to 1.854 m, 0 from 1.976 m. #91 found August
rounds scoring 0 at 0.23-0.69 m, so the scorer may match the nearest object instead;
a recomputed target credited further from the expected point than both
``TARGET_CERTAIN_M`` and the distance the platform credited for that round is flagged
uncertain.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

COMPONENTS = ("camera", "object_type", "landmark", "relation", "anchor", "target")
ANCHOR_ZERO_M = 5.0
TARGET_CREDITED_M = 1.854  # largest distance credited in the #88 rounds
TARGET_UNCREDITED_M = 1.976  # smallest distance not credited
TARGET_RADIUS_M = (TARGET_CREDITED_M + TARGET_UNCREDITED_M) / 2.0
TARGET_CERTAIN_M = 0.5
SAME_POINT_M = 1e-3


@dataclass(frozen=True)
class LoggedRound:
    """One round: the platform's score detail plus the agent's logged events."""

    number: int
    problem_id: str
    detail: dict  # robust_*.json round_details[i]: expected, submission, weights, components
    pipeline: Optional[dict]  # stage1.pipeline.completed: command, parsed, candidates, decisions
    submission: Optional[dict]  # stage1.submission.completed: target_id, anchor_id, coordinates
    sim_s: Optional[float]  # simulation seconds from round received to submission
    time_limit_s: Optional[float]
    image_sizes: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def expected(self) -> dict:
        return self.detail.get("expected") or {}

    @property
    def is_person(self) -> bool:
        return self.expected.get("target_kind") == "person"

    @property
    def decisions(self) -> list[dict]:
        return list((self.pipeline or {}).get("decisions") or [])


@dataclass(frozen=True)
class LoggedRun:
    name: str
    average: float
    total_rounds: Optional[int]
    rounds: list[LoggedRound]


@dataclass(frozen=True)
class RoundScore:
    components: dict[str, float]
    total: float
    target_m: float
    anchor_m: float
    target_uncertain: bool


def _events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _image_sizes(round_dir: Path) -> dict[str, tuple[int, int]]:
    sizes = {}
    for png in sorted((round_dir / "cameras").glob("*.png")):
        if png.stem.endswith("_boxes"):
            continue
        with Image.open(png) as image:
            sizes[png.stem] = image.size
    return sizes


def read_run(run_dir: Path) -> LoggedRun:
    run_dir = Path(run_dir)
    score_files = sorted(run_dir.glob("robust_*.json"))
    event_files = sorted(run_dir.glob("experiments/*/events.jsonl"))
    if len(score_files) != 1 or len(event_files) != 1:
        raise SystemExit(
            f"{run_dir}: expected one robust_*.json and one events.jsonl, "
            f"found {len(score_files)} and {len(event_files)}"
        )
    scores = json.loads(score_files[0].read_text(encoding="utf-8"))
    experiment_dir = event_files[0].parent
    by_kind: dict[str, dict[int, dict]] = {}
    total_rounds = None
    for event in _events(event_files[0]):
        name, number = event.get("event"), event.get("round")
        if number is None or (name == "score.received" and event.get("is_final")):
            continue
        by_kind.setdefault(name, {})[number] = event
        if name == "stage1.round.received":
            total_rounds = event.get("total_rounds", total_rounds)
    numbers = {
        (event.get("scores") or {}).get("problem_id"): number
        for number, event in by_kind.get("score.received", {}).items()
    }
    rounds = []
    for index, detail in enumerate(scores["round_details"], start=1):
        number = numbers.get(detail["problem_id"], index)
        received = by_kind.get("stage1.round.received", {}).get(number) or {}
        submitted = by_kind.get("stage1.submission.completed", {}).get(number)
        pipeline = by_kind.get("stage1.pipeline.completed", {}).get(number)
        sim_s = None
        if submitted and "simulation_timestamp" in submitted and "simulation_timestamp" in received:
            sim_s = submitted["simulation_timestamp"] - received["simulation_timestamp"]
        sizes = {}
        if pipeline and pipeline.get("artifact_dir"):
            sizes = _image_sizes(experiment_dir / pipeline["artifact_dir"])
        rounds.append(LoggedRound(
            number=number,
            problem_id=detail["problem_id"],
            detail=detail,
            pipeline=pipeline,
            submission=submitted,
            sim_s=sim_s,
            time_limit_s=received.get("time_limit_s"),
            image_sizes=sizes,
        ))
    return LoggedRun(run_dir.name, scores["scores"]["stage1_average"], total_rounds, rounds)


def _matches(submitted, expected) -> float:
    """Exact match; a null never matches (SAR relation scores 0 at weight 0)."""
    return 1.0 if expected is not None and submitted == expected else 0.0


def score(round_: LoggedRound, submission: dict) -> RoundScore:
    """Score a (replayed) submission for this round with the reverse-engineered kernel.

    A target at the logged point keeps the platform's logged credit, so rounds the
    replay does not move score exactly as they did.
    """
    expected = round_.expected
    logged = round_.detail.get("submission") or {}
    anchor_m = math.dist(submission["anchor_coord"], expected["anchor_coord"])
    target_m = math.dist(submission["target_coord"], expected["target_coord"])
    unmoved = (
        logged.get("target_coord") is not None
        and math.dist(submission["target_coord"], logged["target_coord"]) < SAME_POINT_M
    )
    target = round_.detail["target"] if unmoved else float(target_m <= TARGET_RADIUS_M)
    credited_m = TARGET_CERTAIN_M
    if round_.detail["target"] and logged.get("target_coord") is not None:
        credited_m = max(credited_m, math.dist(logged["target_coord"], expected["target_coord"]))
    components = {
        "camera": _matches(submission.get("camera_id"), expected.get("camera_id")),
        "object_type": (
            _matches(submission.get("situation"), expected.get("situation"))
            if round_.is_person
            else _matches(submission.get("target_type"), expected.get("target_type"))
        ),
        "landmark": _matches(submission.get("landmark"), expected.get("landmark_prefix")),
        "relation": _matches(submission.get("relation"), expected.get("relation")),
        "anchor": max(0.0, 1.0 - anchor_m / ANCHOR_ZERO_M),
        "target": target,
    }
    weights = round_.detail["weights"]
    total = 100.0 * sum(weights[name] * components[name] for name in COMPONENTS) / sum(
        weights[name] for name in COMPONENTS
    )
    return RoundScore(
        components=components,
        total=total,
        target_m=target_m,
        anchor_m=anchor_m,
        target_uncertain=not unmoved and credited_m < target_m < TARGET_UNCREDITED_M,
    )
