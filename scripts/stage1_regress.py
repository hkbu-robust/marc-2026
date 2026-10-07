"""Per-PR regression gate: compare a Stage 1 run against baseline runs (#110).

    python -m scripts.stage1_regress compare --baseline RUN_DIR [RUN_DIR ...] \
        --candidate RUN_DIR [--allow lf06,hf08 ...] [--skip-tests] --out REPORT.md

Every folder is a #85 harness run (``scripts/run_stage1_repeats.py``). Each round of
the candidate is compared with that round's band over the baseline runs, because
some rounds (lf06, hf08, sar03) flip between unchanged runs.

FAIL when
- a round not in ``--allow`` scores outside its band of totals by more than
  ``BAND_TOLERANCE`` points; or
- an invariant breaks: every round submitted; landmark and relation filled on
  object rounds; no sentinel or missing coordinate; every baseline camera loaded;
  p95 round time within the time limit (simulation time); every command parses to its
  expected target type with this checkout's parser; every ``tests/test_*.py`` passes.

Other differences from the baseline (score components, camera, target or anchor
point, submitted fields, decision paths) are listed per round for review but do
not fail on their own: a geometry fix moves points on every round, and a changed
point that does not change the score is not a regression the gate can call.
Targets and anchors are compared by position, because candidate IDs are renumbered
between runs (#88).

Writes a markdown report and exits 1 on FAIL.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from marc_agent.command_parser import CommandParseError, Intent, ParseErrorCode, parse_command
from marc_agent.perception.stage1_pipeline import IGNORED_CLASS_DUMMY_XYZ
from scripts.stage1_runs import COMPONENTS, LoggedRound, LoggedRun, read_run
from scripts.stage1_scores import SAME_ANSWER_M

REPO = Path(__file__).resolve().parent.parent
BAND_TOLERANCE = 0.5  # points of one round's total
COMPONENT_TOLERANCE = 0.01  # credit; 0.01 anchor credit = 5 cm


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class RoundVerdict:
    problem_id: str
    total: Optional[float]
    band: tuple[float, float]
    status: str  # "in band", "out of band", "allowed change", "missing"
    changes: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status in {"out of band", "missing"}


def _decision_kinds(round_: LoggedRound) -> set[str]:
    kinds = set()
    for note in round_.decisions:
        kind = note.get("decision")
        if note.get("variant"):
            kind = f"{kind}:{note['variant']}"
        elif kind == "selector" and note.get("outcome"):
            kind = f"selector:{note['outcome']}"
        kinds.add(kind)
    return kinds


def _nearest_m(point, others) -> float:
    return min((math.dist(point, other) for other in others if other), default=math.inf)


def compare_round(
    problem_id: str,
    candidate: Optional[LoggedRound],
    baseline: list[LoggedRound],
    allowed: bool,
) -> RoundVerdict:
    totals = [item.detail["total"] for item in baseline]
    band = (min(totals), max(totals))
    if candidate is None:
        return RoundVerdict(problem_id, None, band, "missing")
    total = candidate.detail["total"]
    out = total < band[0] - BAND_TOLERANCE or total > band[1] + BAND_TOLERANCE
    status = ("allowed change" if allowed else "out of band") if out else "in band"
    changes = []
    for name in COMPONENTS:
        if not candidate.detail["weights"].get(name):
            continue
        low = min(item.detail[name] for item in baseline)
        high = max(item.detail[name] for item in baseline)
        value = candidate.detail[name]
        if value < low - COMPONENT_TOLERANCE or value > high + COMPONENT_TOLERANCE:
            changes.append(f"{name} {value:.3f} (band {low:.3f}–{high:.3f})")
    submitted = candidate.detail.get("submission") or {}
    base_subs = [item.detail.get("submission") or {} for item in baseline]
    for name in ("camera_id", "landmark", "relation", "situation", "target_type"):
        seen = {sub.get(name) for sub in base_subs}
        if submitted.get(name) not in seen:
            shown = ", ".join(sorted(str(value) for value in seen))
            changes.append(f"{name} {submitted.get(name)} (baseline {shown})")
    for name in ("target_coord", "anchor_coord"):
        if submitted.get(name):
            moved = _nearest_m(submitted[name], [sub.get(name) for sub in base_subs])
            if moved > SAME_ANSWER_M:
                kind = name.split("_")[0]
                chosen = (candidate.submission or {}).get(f"{kind}_id")
                changes.append(
                    f"{kind} point {moved:.2f} m from every baseline answer ({chosen})"
                )
    kinds = _decision_kinds(candidate)
    base_kinds = [_decision_kinds(item) for item in baseline]
    new = kinds - set().union(*base_kinds)
    gone = set.intersection(*base_kinds) - kinds if base_kinds else set()
    if new:
        changes.append("new decisions: " + ", ".join(sorted(new)))
    if gone:
        changes.append("missing decisions: " + ", ".join(sorted(gone)))
    return RoundVerdict(problem_id, total, band, status, changes)


def _parse_check(round_: LoggedRound) -> Optional[str]:
    """None when this checkout's parser reads the logged command as the expected target."""
    command = (round_.pipeline or {}).get("command")
    if not command:
        return "no logged command"
    try:
        try:
            parsed = parse_command(command, expected_intent=Intent.FIND)
        except CommandParseError as exc:
            if exc.code is not ParseErrorCode.INTENT_MISMATCH:
                raise
            parsed = parse_command(command)  # as run_stage1_candidate_pipeline does
    except CommandParseError as exc:
        return f"parse error {exc.code.value}"
    want = "person" if round_.is_person else round_.expected.get("target_type")
    if parsed.target_type != want:
        return f"target {parsed.target_type}, expected {want}"
    return None


def _run_test(path: Path) -> tuple[Path, int, str]:
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, str(path)], cwd=REPO, env=env, capture_output=True, text=True
    )
    return path, proc.returncode, (proc.stdout + proc.stderr)[-2000:]


def unit_tests() -> Check:
    """Each test file runs as its own script (``unittest discover`` trips on demo/)."""
    paths = sorted((REPO / "tests").glob("test_*.py"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        results = list(pool.map(_run_test, paths))
    failed = [path.name for path, code, _ in results if code != 0]
    detail = f"{len(paths) - len(failed)}/{len(paths)} test files pass"
    if failed:
        detail += ": failing " + ", ".join(failed)
    return Check("unit tests", not failed, detail)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def invariants(candidate: LoggedRun, baselines: list[LoggedRun]) -> list[Check]:
    rounds = candidate.rounds
    problems = {r.problem_id for run in baselines for r in run.rounds}
    submitted = [r for r in rounds if r.submission is not None]
    expected_rounds = candidate.total_rounds or len(problems)
    checks = [Check(
        "every round submitted",
        len(submitted) == expected_rounds and {r.problem_id for r in submitted} >= problems,
        f"{len(submitted)}/{expected_rounds}",
    )]

    empty = [
        r.problem_id for r in rounds
        if not r.is_person and not all(
            (r.detail.get("submission") or {}).get(name) for name in ("landmark", "relation")
        )
    ]
    checks.append(Check(
        "landmark and relation on object rounds", not empty, ", ".join(empty)
    ))

    def placeholder_point(r: LoggedRound) -> bool:
        submitted = r.detail.get("submission") or {}
        points = [submitted.get(name) for name in ("target_coord", "anchor_coord")]
        return (
            any(note.get("decision") == "sentinel" for note in r.decisions)
            or any(not point or math.dist(point, IGNORED_CLASS_DUMMY_XYZ) < 1e-6
                   for point in points)
        )

    placeholders = [r.problem_id for r in rounds if placeholder_point(r)]
    checks.append(Check(
        "no sentinel or missing coordinates", not placeholders, ", ".join(placeholders)
    ))

    cameras = {
        camera for run in baselines for r in run.rounds
        for camera in (r.pipeline or {}).get("frames_loaded") or ()
    }
    short = [
        r.problem_id for r in rounds
        if not cameras <= set((r.pipeline or {}).get("frames_loaded") or ())
    ]
    checks.append(Check(
        f"all {len(cameras)} cameras loaded every round", not short, ", ".join(short)
    ))

    times = [r.sim_s for r in rounds if r.sim_s is not None]
    limits = {r.time_limit_s for r in rounds if r.time_limit_s is not None}
    limit = min(limits) if limits else None
    if times and limit is not None:
        p95 = _p95(times)
        checks.append(Check(
            "p95 round time within the limit (simulation s)",
            p95 <= limit, f"p95 {p95:.1f} s, max {max(times):.1f} s, limit {limit:.0f} s",
        ))
    else:
        checks.append(Check("p95 round time within the limit (simulation s)", False,
                            "no round times logged"))

    parse_errors = [
        f"{r.problem_id} ({error})" for r in rounds
        if (error := _parse_check(r)) is not None
    ]
    checks.append(Check(
        "commands parse to the expected target", not parse_errors, "; ".join(parse_errors)
    ))
    return checks


def report(
    candidate: LoggedRun,
    baselines: list[LoggedRun],
    verdicts: list[RoundVerdict],
    checks: list[Check],
    allow: set[str],
) -> tuple[str, bool]:
    failed = any(v.failed for v in verdicts) or not all(c.ok for c in checks)
    base_avgs = [run.average for run in baselines]
    lines = [
        f"# Stage 1 regression gate: {'FAIL' if failed else 'PASS'}",
        "",
        f"- Candidate: `{candidate.name}`, Stage 1 average {candidate.average:.3f}",
        f"- Baseline: {len(baselines)} runs ({', '.join(run.name for run in baselines)}), "
        f"average {min(base_avgs):.3f}–{max(base_avgs):.3f}",
        f"- Allowed to change: {', '.join(sorted(allow)) or 'none'}",
        f"- A round fails when its total leaves the baseline band by more than "
        f"{BAND_TOLERANCE} points.",
        "",
        "## Invariants",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    lines += [f"| {c.name} | {'ok' if c.ok else 'FAIL'} | {c.detail} |" for c in checks]
    lines += [
        "",
        "## Rounds",
        "",
        "| round | total | baseline band | status | changes vs baseline |",
        "|---|---|---|---|---|",
    ]
    for v in verdicts:
        total = "—" if v.total is None else f"{v.total:.2f}"
        status = v.status.upper() if v.failed else v.status
        lines.append(
            f"| {v.problem_id} | {total} | {v.band[0]:.2f}–{v.band[1]:.2f} | {status} "
            f"| {'; '.join(v.changes)} |"
        )
    failing = [v.problem_id for v in verdicts if v.failed]
    if failing:
        lines += ["", f"Failing rounds: {', '.join(failing)}"]
    return "\n".join(lines) + "\n", failed


def compare(
    baseline_dirs: list[Path],
    candidate_dir: Path,
    allow: set[str],
    *,
    run_tests: bool = True,
) -> tuple[str, bool]:
    baselines = [read_run(path) for path in baseline_dirs]
    candidate = read_run(candidate_dir)
    by_problem: dict[str, list[LoggedRound]] = {}
    for run in baselines:
        for round_ in run.rounds:
            by_problem.setdefault(round_.problem_id, []).append(round_)
    unknown = allow - set(by_problem)
    if unknown:
        raise SystemExit(f"--allow names rounds not in the baseline: {', '.join(sorted(unknown))}")
    rounds = {round_.problem_id: round_ for round_ in candidate.rounds}
    verdicts = [
        compare_round(problem, rounds.get(problem), base, problem in allow)
        for problem, base in by_problem.items()
    ]
    verdicts += [
        RoundVerdict(problem, rounds[problem].detail["total"], (math.nan, math.nan),
                     "out of band", ["no baseline for this round"])
        for problem in rounds if problem not in by_problem
    ]
    checks = invariants(candidate, baselines)
    checks.append(unit_tests() if run_tests else Check("unit tests", True, "skipped"))
    return report(candidate, baselines, verdicts, checks, allow)


def _allow(values: list[str]) -> set[str]:
    return {item.strip() for value in values for item in value.split(",") if item.strip()}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cmp = sub.add_parser("compare", help="compare one run against baseline runs")
    cmp.add_argument("--baseline", nargs="+", type=Path, required=True)
    cmp.add_argument("--candidate", type=Path, required=True)
    cmp.add_argument("--allow", nargs="*", default=[], help="round IDs expected to change")
    cmp.add_argument("--skip-tests", action="store_true")
    cmp.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    text, failed = compare(
        args.baseline, args.candidate, _allow(args.allow), run_tests=not args.skip_tests
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
