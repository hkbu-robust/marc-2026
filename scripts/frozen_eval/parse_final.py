"""Read every held-out command once with the final system's reader (#87 refresh).

Run inside ``marc-agent:final-system-v1`` (its ``/agent/marc_agent`` is the tag's code)
with ``MARC_COMMAND_PARSER=qwen`` and the deployed Qwen served by vLLM::

    docker run --rm --network host --entrypoint python3 \\
      -e MARC_COMMAND_PARSER=qwen -e MARC_QWEN_BACKEND=remote \\
      -e MARC_QWEN_BASE_URL=http://127.0.0.1:8024/v1 -e MARC_QWEN_API_KEY=local-vllm-key \\
      -e MARC_QWEN_MODEL=qwen3-vl-8b \\
      -v $REPO/scripts/frozen_eval:/work/frozen_eval:ro -v $ROOT:$ROOT \\
      marc-agent:final-system-v1 /work/frozen_eval/parse_final.py $ROOT heldout $OUT.jsonl

One line per case: the runtime reading (``command_llm.read_command`` with
``expected_intent=FIND``, as ``run_stage1_candidate_pipeline`` calls it) with its
source (qwen | rules | partial), fallback reason and fixes, plus the lenient rule
reading for reference. One warm-up request (not a case) is sent first, because the
first vLLM request compiles kernels and would hit the reader's 6 s timeout.
Resumable: cases already in OUT are skipped.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path

sys.path.insert(0, "/agent")

from marc_agent import command_llm  # noqa: E402
from marc_agent.command_parser import Intent, parse_command_lenient  # noqa: E402
from marc_agent.experiment_log import decision_scope  # noqa: E402


def as_dict(parsed) -> dict:
    out = {}
    for key, value in asdict(parsed).items():
        if isinstance(value, Enum):
            value = value.value
        elif isinstance(value, tuple):
            value = list(value)
        out[key] = value
    return out


def warm_up() -> dict:
    for attempt in range(1, 31):
        reading = command_llm.read_command("Find the mug near the bench.", expected_intent=Intent.FIND,
                                           timeout_s=120.0)
        if reading.source == "qwen":
            return {"attempts": attempt, "latency_s": reading.latency_s}
        time.sleep(10)
    raise SystemExit("warm-up: Qwen never answered")


def run(root: Path, set_name: str, out_path: Path) -> int:
    assert command_llm.parser_mode() == "qwen", "set MARC_COMMAND_PARSER=qwen"
    cases = sorted((root / set_name / "cases").glob("*/case.json"))
    done = set()
    if out_path.exists():
        done = {json.loads(line)["case_id"] for line in out_path.read_text().splitlines() if line}
    print(f"[parse-final] warm-up {warm_up()}", flush=True)
    env = {k: v for k, v in os.environ.items() if k.startswith("MARC_") and "KEY" not in k}
    with out_path.open("a", encoding="utf-8") as out:
        for index, path in enumerate(cases):
            case = json.loads(path.read_text(encoding="utf-8"))
            if case["case_id"] in done:
                continue
            with decision_scope("stage1") as notes:
                reading = command_llm.read_command(case["command"], expected_intent=Intent.FIND)
            rules, rules_error = parse_command_lenient(case["command"], expected_intent=Intent.FIND)
            out.write(json.dumps({
                "case_id": case["case_id"],
                "command": case["command"],
                "parsed_final": as_dict(reading.parsed),
                "source": reading.source,
                "fallback_reason": reading.fallback_reason,
                "fixes": list(reading.fixes),
                "pose_hint": reading.pose_hint,
                "situation_hint": reading.situation_hint,
                "latency_s": reading.latency_s,
                "rules_parsed": as_dict(rules),
                "rules_error": rules_error,
                "notes": notes,
                "env": env,
            }, default=str) + "\n")
            out.flush()
            if index % 50 == 0:
                print(f"[parse-final] {index + 1}/{len(cases)} {case['case_id']} {reading.source}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])))
