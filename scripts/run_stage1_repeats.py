"""Run the agent N times on a fresh platform and collect each run (#85).

    python -m scripts.run_stage1_repeats --out ~/marc-2026-work/experiments/ablations/NN-name \
        --runs 2 [--scene-cache on] [--agent-image marc-agent:final-system-v1]

For every run: tear down, start the platform headless, start vLLM (kept between
runs) and a fresh agent, wait for the platform's FINISHED state, then copy the
score file, the agent's experiment folder and both container logs into
OUT/run-NN/ and record provenance and the decision summary. Stage selection
comes from ./problems.yaml, as for any platform run. vLLM is restarted with
the agent unless --warm-qwen is given, so every run starts like a fresh
`docker compose up`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.stage1_decisions import read_run, run_totals, write

REPO = Path(__file__).resolve().parent.parent
PLATFORM_COMPOSE = ["-f", "simulation-platform/docker-compose.yml", "--profile", "platform"]
AGENT_COMPOSE = [
    "-f", "docker-compose.yml",
    "-f", "docker-compose.frozen.yml",
    "-f", "scripts/frozen-logging.compose.yml",
]
FINISHED = re.compile(r"State: \S+ → FINISHED")
SCENARIO = "marc2026_chungmu"


def log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def sh(args: list[str], *, env=None, check=True, capture=False) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=REPO, env=env, check=check, text=True,
        capture_output=capture,
    )


def compose(project: str, files: list[str], *args: str, env=None, **kwargs):
    return sh(["docker", "compose", "-p", project, *files, *args], env=env, **kwargs)


def gpu_free_mib() -> int:
    out = sh(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture=True,
    ).stdout
    return int(out.split()[0])


def platform_log(project: str) -> str:
    return compose(project, PLATFORM_COMPOSE, "logs", "--no-color", "platform",
                   capture=True, check=False).stdout


def wait_finished(project: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if FINISHED.search(platform_log(project)):
            return True
        time.sleep(15)
    return False


def provenance(args, env: dict, agent_project: str) -> dict:
    def out(cmd):
        return sh(cmd, capture=True, check=False).stdout.strip()

    images = {}
    for image in (args.agent_image, args.qwen_image, "marc-platform:practice"):
        images[image] = out(["docker", "image", "inspect", image, "--format", "{{.Id}}"])
    return {
        "commit": out(["git", "rev-parse", "HEAD"]),
        "dirty": bool(out(["git", "status", "--porcelain", "--", "marc_agent", "demo"])),
        "images": images,
        "served_models": out([
            "docker", "compose", "-p", agent_project, *AGENT_COMPOSE, "exec", "-T", "qwen",
            "python3", "-c",
            "import os, urllib.request; print(urllib.request.urlopen(urllib.request.Request("
            "'http://127.0.0.1:' + os.environ['VLLM_PORT'] + '/v1/models', "
            "headers={'Authorization': 'Bearer ' "
            "+ os.environ['VLLM_API_KEY']}), timeout=5).read().decode())",
        ]),
        "env": {
            key: env[key] for key in (
                "ROS_DOMAIN_ID", "MARC_TEAM_ID", "HEADLESS", "ENV_MARC_SCENARIO",
                "MARC_STAGE1_SCENE_CACHE", "MARC_ROUND_DUMP", "MARC_AGENT_IMAGE",
                "MARC_QWEN_IMAGE", "MARC_VLLM_PORT", "MARC_VLLM_GPU_FRACTION",
            )
        },
        # Set by `docker build --label org.opencontainers.image.revision=<sha>`.
        "agent_revision": out([
            "docker", "image", "inspect", args.agent_image, "--format",
            '{{index .Config.Labels "org.opencontainers.image.revision"}}',
        ]),
        "problems_yaml": (REPO / "problems.yaml").read_text(encoding="utf-8"),
        "gpu": out(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader"]),
    }


def _problems(record: dict) -> list[str]:
    """Why a run is incomplete; empty when it is usable."""
    problems = []
    if not record["finished"]:
        problems.append("no FINISHED state")
    if not record["score_files"]:
        problems.append("no platform score file")
    rounds = record.get("rounds_logged", 0)
    if not rounds:
        problems.append("no logged rounds")
    elif record.get("round_dumps") != rounds:
        problems.append(f"{record.get('round_dumps')} round dumps for {rounds} rounds")
    return problems


def one_run(index: int, args, env: dict) -> dict:
    run_dir = args.out / f"run-{index:02d}"
    experiments = run_dir / "experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    env = {**env, "MARC_EXPERIMENTS_HOST_DIR": str(experiments)}
    platform, agent = f"{args.project}-platform", f"{args.project}-agent"

    log(f"run {index:02d}: resetting platform and agent")
    # A fresh `docker compose up` starts vLLM cold, and round 1 pays for it.
    # Restart it every run unless --warm-qwen asks to keep it loaded.
    services = ["agent"] if args.warm_qwen else ["agent", "qwen"]
    compose(agent, AGENT_COMPOSE, "rm", "-sf", *services, env=env, check=False)
    compose(platform, PLATFORM_COMPOSE, "down", env=env, check=False)

    free = gpu_free_mib()
    if free < args.min_free_gpu_mib:
        raise SystemExit(f"GPU has {free} MiB free; need {args.min_free_gpu_mib}. Stopping.")

    results = REPO / "results" / SCENARIO
    before = set(results.glob("*.json")) if results.is_dir() else set()
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    compose(platform, PLATFORM_COMPOSE, "up", "-d", "--no-build", "--force-recreate",
            "platform", env=env)
    time.sleep(args.agent_delay_s)
    compose(agent, AGENT_COMPOSE, "up", "-d", "--no-build", "--wait", "qwen", env=env)
    compose(agent, AGENT_COMPOSE, "up", "-d", "--no-build", "--no-deps",
            "--force-recreate", "agent", env=env)
    running = compose(agent, AGENT_COMPOSE, "ps", "-q", "agent",
                      capture=True, check=False, env=env).stdout.strip()
    running_id = sh(["docker", "inspect", running, "--format", "{{.Image}}"],
                    capture=True, check=False).stdout.strip()
    wanted_id = sh(["docker", "image", "inspect", args.agent_image, "--format", "{{.Id}}"],
                   capture=True, check=False).stdout.strip()
    if not running_id or running_id != wanted_id:
        raise SystemExit(f"agent container runs {running_id or 'nothing'}, "
                         f"not {args.agent_image} ({wanted_id})")
    log(f"run {index:02d}: started; waiting for FINISHED")
    finished = wait_finished(platform, args.timeout_s)
    wall_s = time.monotonic() - t0

    (run_dir / "platform.log").write_text(platform_log(platform), encoding="utf-8")
    (run_dir / "agent.container.log").write_text(
        compose(agent, AGENT_COMPOSE, "logs", "--no-color", "agent",
                capture=True, check=False, env=env).stdout,
        encoding="utf-8",
    )
    new_scores = sorted(set(results.glob("*.json")) - before - {results / "_summary.json"})
    for path in new_scores:
        shutil.copy2(path, run_dir / path.name)
    record = {
        "run": index,
        "finished": finished,
        "warm_qwen": args.warm_qwen,
        "started_utc": started.isoformat(),
        "wall_s": wall_s,
        "score_files": [path.name for path in new_scores],
        **provenance(args, env, agent),
    }
    compose(agent, AGENT_COMPOSE, "rm", "-sf", "agent", env=env, check=False)
    compose(platform, PLATFORM_COMPOSE, "down", env=env, check=False)

    folders = sorted(path for path in experiments.iterdir() if path.is_dir())
    record["experiment_dirs"] = [path.name for path in folders]
    if folders:
        summary = read_run(folders[-1])
        dumps = list((folders[-1] / "artifacts" / "stage1").glob("round-*"))
        record["round_dumps"] = len(dumps)
        record["rounds_logged"] = len(summary.rounds)
        record["totals"] = run_totals(summary)
        write([summary], run_dir / "decisions")
    record["problems"] = _problems(record)
    (run_dir / "run.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log(
        f"run {index:02d}: finished={finished} wall={wall_s:.0f}s "
        f"scores={record['score_files']} rounds={record.get('rounds_logged')} "
        f"dumps={record.get('round_dumps')} problems={record['problems'] or 'none'}"
    )
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--first", type=int, default=1, help="number of the first run folder")
    parser.add_argument("--scene-cache", choices=["on", "off"], default="off",
                        help="on reproduces the qualifier submission; off is the final system")
    parser.add_argument("--agent-image", default="marc-agent:final-system-v1")
    parser.add_argument("--qwen-image", default="marc-qwen-vllm:qualifier-b2360c86")
    parser.add_argument("--project", default="marc85")
    parser.add_argument("--ros-domain-id", default="85")
    parser.add_argument("--agent-delay-s", type=float, default=20.0)
    parser.add_argument("--timeout-s", type=float, default=2400.0)
    parser.add_argument("--min-free-gpu-mib", type=int, default=45000,
                        help="refuse to start with less free GPU memory than this")
    parser.add_argument("--vllm-port", type=int, default=8004,
                        help="host port for this run's vLLM; give each concurrent run its own")
    parser.add_argument("--vllm-gpu-fraction", default="0.37",
                        help="share of the physical GPU vLLM reserves (0.37 of 96 GB = ~36 GB, "
                             "as on the 48 GB judging card); empty = the entrypoint's 0.75")
    parser.add_argument("--warm-qwen", action="store_true",
                        help="keep vLLM loaded between runs instead of a cold start each run")
    args = parser.parse_args(argv)
    args.out = args.out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "ROS_DOMAIN_ID": args.ros_domain_id,
        "MARC_TEAM_ID": "robust",
        "HEADLESS": "true",
        "ENV_MARC_SCENARIO": SCENARIO,
        "MARC_STAGE1_SCENE_CACHE": "1" if args.scene_cache == "on" else "0",
        "MARC_ROUND_DUMP": "1",
        "MARC_AGENT_IMAGE": args.agent_image,
        "MARC_QWEN_IMAGE": args.qwen_image,
        "MARC_VLLM_PORT": str(args.vllm_port),
        "MARC_VLLM_GPU_FRACTION": args.vllm_gpu_fraction,
    }
    records = []
    try:
        for index in range(args.first, args.first + args.runs):
            records.append(one_run(index, args, env))
    finally:
        compose(f"{args.project}-agent", AGENT_COMPOSE, "down", env=env, check=False)
        compose(f"{args.project}-platform", PLATFORM_COMPOSE, "down", env=env, check=False)

    finished = [path for path in sorted(args.out.glob("run-*/experiments/*")) if path.is_dir()]
    if finished:
        write([read_run(path) for path in finished], args.out / "decisions")
    failed = {record["run"]: record["problems"] for record in records if record["problems"]}
    if failed:
        sys.exit(f"incomplete runs: {failed}")


if __name__ == "__main__":
    main()
