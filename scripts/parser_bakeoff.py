"""Command-reader bake-off (#114): old rules vs lenient rules vs Qwen extraction.

    python -m scripts.parser_bakeoff --out <dir> [--qwen-url http://127.0.0.1:8014/v1]

Sets:
- **test**: ``tests/fixtures/parser_test_set.json``, written blind and checksummed before any
  prompt work; fields the two independent labellers disagree on, and items the writer marked
  ambiguous, are excluded (``parser_test_adjudication.json``). Its relation labels used
  "next to"/"by"/"leaning against" -> beside; the team then adopted the organisers' labelling
  (-> near, #114), so those labels are converted to near before scoring.
- **dev**: the non-public cases of ``tests/fixtures/command_cases.yaml`` (the old parser's own
  fixtures, so old-rule accuracy there is inflated by construction).
- **public**: the 33 public commands with the platform's expected fields (#88 run-01).

Per reader: per-field accuracy, and the no-submission rate (a reader that raises makes the
round submit nothing; the old rules raise on parse errors).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEST = REPO / "tests/fixtures/parser_test_set.json"
ADJ = REPO / "tests/fixtures/parser_test_adjudication.json"
DEV = REPO / "tests/fixtures/command_cases.yaml"
PUBLIC_RUN = Path.home() / "marc-2026-work/experiments/ablations/04-repeated-runs/cache-off/run-01"
FIELDS = ("target_type", "landmark", "relation", "scene_hint", "situation")  # Stage 1 is always find
CONVENTION_RE = re.compile(r"\bnext\s+to\b|\bleaning\s+against\b|\bby\b", re.I)


def old_parser():
    """The rule parser as it was on final-system before #114."""
    source = subprocess.run(["git", "show", "final-system:marc_agent/command_parser.py"],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
    path = Path(tempfile.mkdtemp()) / "old_command_parser.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("old_command_parser", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["old_command_parser"] = module
    spec.loader.exec_module(module)
    return module


def test_items():
    payload = json.loads(TEST.read_text())
    items = payload["items"]
    canon = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(canon).hexdigest() == payload["sha256"], "test set changed"
    adj = json.loads(ADJ.read_text())
    out = []
    for item in items:
        label = dict(item["label"])
        label["situation"] = label.pop("situation_hint", None)
        if label.get("relation") == "beside" and CONVENTION_RE.search(item["command"]):
            label["relation"] = "near"
        skip = set(adj["excluded_fields"].get(item["id"], ()))
        if item["id"] in adj["ambiguous_items"]:
            skip = set(FIELDS) | {"pose_hint"}
        out.append((item["id"], item["command"], label, skip))
    return out


def dev_items():
    out = []
    for case in json.loads(DEV.read_text())["cases"]:
        if case["tier"] == "practice":
            continue
        label = case.get("expected")
        out.append((case["id"], case["text"], label, set() if label else set(FIELDS)))
    return out


def public_items():
    from scripts.stage1_runs import read_run

    out = []
    for r in read_run(PUBLIC_RUN).rounds:
        e = r.expected
        label = {"intent": "find", "target_type": "person" if r.is_person else e.get("target_type"),
                 "relation": None if r.is_person else e.get("relation"),
                 "situation": e.get("situation") if r.is_person else None}
        # Landmark/scene are not stated in every public command; the platform's landmark is
        # scored separately (#113 fills it), so only type, relation and situation count here.
        out.append((r.problem_id, r.pipeline["command"], label, {"landmark", "scene_hint"}))
    return out


def run_reader(name, read, items, default_relation):
    rows, fails = [], 0
    for item_id, text, label, skip in items:
        started = time.perf_counter()
        try:
            got = read(text)
            error = None
        except Exception as exc:  # old rules raise: the round would submit nothing
            got, error = None, type(exc).__name__
            fails += 1
        latency = time.perf_counter() - started
        row = {"id": item_id, "error": error, "latency_s": round(latency, 3), "fields": {}}
        for field in FIELDS:
            if field in skip or label is None:
                continue
            want = label.get(field)
            have = None if got is None else got.get(field)
            if field == "relation" and default_relation and have is None and got is not None \
                    and got.get("target_type") != "person":
                have = "near"  # the submission layer's default (#113)
            row["fields"][field] = (have == want, have, want)
        rows.append(row)
    return {"reader": name, "rows": rows, "no_submission": fails, "n": len(items)}


def as_dict(parsed) -> dict:
    return {
        "intent": parsed.intent.value, "target_type": parsed.target_type,
        "landmark": parsed.landmark, "relation": parsed.relation.value if parsed.relation else None,
        "scene_hint": parsed.scene_hint,
        "situation": parsed.situation.value if parsed.situation else None,
    }


def _old_runtime(old, text):
    """The old rules as the Stage 1 pipeline called them: FIND, else without intent."""
    try:
        return old.parse_command(text, expected_intent=old.Intent.FIND)
    except old.CommandParseError as error:
        if error.code is not old.ParseErrorCode.INTENT_MISMATCH:
            raise
        return old.parse_command(text)


def readers(qwen_url: str | None):
    old = old_parser()
    from marc_agent import command_llm
    from marc_agent.command_parser import Intent, parse_command_lenient

    out = {
        "old rules": lambda t: as_dict(_old_runtime(old, t)),
        "lenient rules": lambda t: as_dict(parse_command_lenient(t, expected_intent=Intent.FIND)[0]),
    }
    if qwen_url:
        os.environ.update({"MARC_QWEN_BACKEND": "remote", "MARC_QWEN_BASE_URL": qwen_url,
                           "MARC_QWEN_API_KEY": os.environ.get("MARC_QWEN_API_KEY", "marc-qwen-secret"),
                           "MARC_QWEN_MODEL": "qwen3-vl-8b", "MARC_COMMAND_PARSER": "qwen"})
        complete = command_llm.qwen_complete()
        out["qwen"] = lambda t: as_dict(command_llm.read_command(
            t, expected_intent=Intent.FIND, complete=complete).parsed)
    return out


def table(results, title) -> list[str]:
    lines = [f"### {title}", "", "| reader | no submission | " + " | ".join(FIELDS) + " | p95 latency s |",
             "|---|---|" + "---|" * len(FIELDS) + "---|"]
    for res in results:
        cells = []
        for field in FIELDS:
            marks = [row["fields"][field][0] for row in res["rows"] if field in row["fields"]]
            cells.append(f"{sum(marks)}/{len(marks)}" if marks else "-")
        lat = sorted(row["latency_s"] for row in res["rows"])
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0
        lines.append(f"| {res['reader']} | {res['no_submission']}/{res['n']} | " + " | ".join(cells)
                     + f" | {p95:.2f} |")
    return lines + [""]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qwen-url")
    parser.add_argument("--sets", nargs="*", default=["dev", "public", "test"])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    loaders = {"dev": dev_items, "public": public_items, "test": test_items}
    report = ["# Command-reader bake-off (#114)", "",
              "Public relation accuracy counts the submission default (near) when a reader states",
              "none. The situation column is the text hint; the submitted SAR situation comes from",
              "R03 on the detected pose (#114), so it does not decide the round.", ""]
    for name in args.sets:
        items = loaders[name]()
        # Dev/test labels record what the text states; the public labels are the scorer's
        # expectations, so only there does a missing relation count as the default near.
        results = [run_reader(rname, read, items, default_relation=(name == "public"))
                   for rname, read in readers(args.qwen_url).items()]
        report += table(results, f"{name} ({len(items)} commands)")
        (args.out / f"{name}.json").write_text(json.dumps(results, indent=1))
    (args.out / "bakeoff.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
