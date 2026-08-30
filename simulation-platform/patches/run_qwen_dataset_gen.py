#!/usr/bin/env python3
"""Generate isolated CCTV GT, then export a coverage-checked Qwen3-VL corpus."""

from __future__ import annotations

import os
import subprocess


PYTHON = "/isaac-sim/python.sh"
TRAINER = "/metacom2026/simulation_app/marc2026/trainer.py"


def main() -> None:
    subprocess.run(
        [PYTHON, "/marc-patches/boot.py", TRAINER],
        check=True,
        env=os.environ.copy(),
    )
    scenario = os.environ.get("ENV_MARC_SCENARIO", "marc2026_chungmu")
    subprocess.run(
        [
            PYTHON,
            "/marc-agent-repo/demo/vlm_dataset/cctv_gt_to_qwen.py",
            "--in", f"/metacom2026/trainer_output/{scenario}",
            "--out", "/qwen-output",
            "--format", "llama-factory",
            "--require-complete",
        ],
        check=True,
        env=os.environ.copy(),
    )


if __name__ == "__main__":
    main()
