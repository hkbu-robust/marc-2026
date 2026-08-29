#!/usr/bin/env python3
"""Download pinned runtime weights at Docker build (or on a developer host).

Skip any artifact that is already complete. Hub 429/502/503 sleeps and retries
inside this process so a rate limit does not discard the rest of the layer.
Do not call this at scoring runtime; the image is offline after this script finishes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from marc_agent.graspgenx_policy import has_checkpoint
from marc_agent.perception.qwen_runtime import QwenRuntimeConfig, bundle_status

QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_REVISION = "60595ebc30ec8e3b1d3b9e65d4943ca011c0006a"
GRASPGENX_MODEL_REPO = "adithyamurali/GraspGenXModel"
GRASPGENX_MODEL_REVISION = "7c834043c11a11417e31d6d5ea9355801e40a2c1"
GRIPPER_DATASET_REPO = "adithyamurali/gripper_descriptions"
GRIPPER_DATASET_REVISION = "19a03c00d19aeaf052d0f6801f0041982d676e8a"
YOLO11X_REPO = "Ultralytics/YOLO11"
YOLO11X_FILENAME = "yolo11x.pt"
YOLO11X_SHA256 = "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"
# Initial try plus these sleeps: 429/502/503 inside one Docker RUN so a Hub
# rate limit does not discard Qwen/GraspGen-X already written in this layer.
RETRY_BACKOFF_S = (30, 60, 120, 240, 300)
RETRY_STATUSES = {429, 502, 503}
RETRY_AFTER_CAP_S = 600.0

SnapshotFn = Callable[..., str]


def _default_paths() -> dict[str, Path]:
    return {
        "qwen": _HERE / "stage1-sar-qwen" / "model",
        "graspgenx": _HERE / "graspgenx",
        "grippers": _HERE / "graspgenx" / "grippers",
        "yolo11x": _REPO_ROOT / "demo" / "weights" / "yolo11x.pt",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allow_hub() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("YOLO_OFFLINE", None)


def _http_status(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    if response is not None:
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None and cause is not exc:
        return _http_status(cause)
    return None


def _is_retryable(exc: BaseException) -> bool:
    status = _http_status(exc)
    if status in RETRY_STATUSES:
        return True
    text = str(exc).lower()
    return "429" in text and "too many requests" in text


def _retry_sleep_s(exc: BaseException, fallback: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = None
    getter = getattr(headers, "get", None)
    if callable(getter):
        raw = getter("Retry-After") or getter("retry-after")
    if raw is None:
        return float(fallback)
    try:
        return min(RETRY_AFTER_CAP_S, max(1.0, float(raw)))
    except (TypeError, ValueError):
        return float(fallback)


def _hub_call(label: str, fn: Callable, *args, sleeper=None, delays=None, **kwargs):
    """Call a Hub downloader, sleeping on 429/502/503 (honors Retry-After)."""
    wait = tuple(RETRY_BACKOFF_S if delays is None else delays)
    pause = time.sleep if sleeper is None else sleeper
    attempts = len(wait) + 1
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_retryable(exc) or attempt >= attempts:
                raise
            delay = _retry_sleep_s(exc, wait[attempt - 1])
            status = _http_status(exc)
            print(
                "[fetch-runtime] %s HTTP %s; sleeping %.0fs (attempt %d/%d)"
                % (label, status if status is not None else "?", delay, attempt, attempts),
                flush=True,
            )
            pause(delay)
    raise RuntimeError(f"{label} Hub retry loop exhausted")


def _snapshot() -> SnapshotFn:
    from huggingface_hub import snapshot_download

    return snapshot_download


def qwen_ready(model_dir: Path) -> bool:
    config = QwenRuntimeConfig(
        model_dir=model_dir,
        dtype="bfloat16",
        device_map="auto",
        attention_implementation="sdpa",
        max_new_tokens=48,
    )
    return bundle_status(config).ready


def fetch_qwen(
    dest: Path,
    *,
    snapshot_download: Optional[SnapshotFn] = None,
    skip_if_complete: bool = True,
) -> str:
    if skip_if_complete and qwen_ready(dest):
        return f"skip qwen ({dest})"
    dest.mkdir(parents=True, exist_ok=True)
    downloader = snapshot_download or _snapshot()
    _hub_call(
        "qwen",
        downloader,
        repo_id=QWEN_REPO,
        revision=QWEN_REVISION,
        local_dir=str(dest),
    )
    if not qwen_ready(dest):
        raise RuntimeError(f"Qwen snapshot at {dest} is still incomplete")
    return f"fetched qwen ({dest})"


def fetch_graspgenx(
    dest: Path,
    *,
    snapshot_download: Optional[SnapshotFn] = None,
    skip_if_complete: bool = True,
) -> str:
    if skip_if_complete and has_checkpoint(str(dest)):
        return f"skip graspgenx ({dest})"
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest / "_hf_model"
    if staging.exists():
        shutil.rmtree(staging)
    downloader = snapshot_download or _snapshot()
    _hub_call(
        "graspgenx",
        downloader,
        repo_id=GRASPGENX_MODEL_REPO,
        revision=GRASPGENX_MODEL_REVISION,
        local_dir=str(staging),
    )
    release = staging / "release"
    gen_src = release / "gen" if (release / "gen").is_dir() else staging / "gen"
    dis_src = release / "dis" if (release / "dis").is_dir() else staging / "dis"
    if not gen_src.is_dir() or not dis_src.is_dir():
        raise RuntimeError(f"GraspGen-X snapshot missing gen/dis under {staging}")
    for name, src in (("gen", gen_src), ("dis", dis_src)):
        target = dest / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
    shutil.rmtree(staging, ignore_errors=True)
    if not has_checkpoint(str(dest)):
        raise RuntimeError(f"GraspGen-X weights at {dest} are still incomplete")
    return f"fetched graspgenx ({dest})"


def grippers_ready(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    return any(dest.rglob("x_grippers"))


def fetch_grippers(
    dest: Path,
    *,
    snapshot_download: Optional[SnapshotFn] = None,
    skip_if_complete: bool = True,
) -> str:
    if skip_if_complete and grippers_ready(dest):
        return f"skip grippers ({dest})"
    dest.mkdir(parents=True, exist_ok=True)
    downloader = snapshot_download or _snapshot()
    _hub_call(
        "grippers",
        downloader,
        repo_id=GRIPPER_DATASET_REPO,
        repo_type="dataset",
        revision=GRIPPER_DATASET_REVISION,
        local_dir=str(dest),
    )
    if not grippers_ready(dest):
        raise RuntimeError(f"gripper descriptions at {dest} are still incomplete")
    return f"fetched grippers ({dest})"


def _hub_file_download(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )


def yolo11x_ready(dest: Path) -> bool:
    if not dest.is_file() or dest.stat().st_size <= 1_000_000:
        return False
    return _sha256(dest) == YOLO11X_SHA256


def fetch_yolo11x(
    dest: Path,
    *,
    hub_download=None,
    skip_if_complete: bool = True,
) -> str:
    if skip_if_complete and yolo11x_ready(dest):
        return f"skip yolo11x ({dest})"
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloader = hub_download or _hub_file_download
    fetched = Path(
        _hub_call(
            "yolo11x",
            downloader,
            repo_id=YOLO11X_REPO,
            filename=YOLO11X_FILENAME,
            local_dir=str(dest.parent),
        )
    )
    if fetched.resolve() != dest.resolve():
        shutil.copy2(fetched, dest)
    digest = _sha256(dest)
    if digest != YOLO11X_SHA256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"yolo11x.pt checksum mismatch: got {digest}, expected {YOLO11X_SHA256}"
        )
    return f"fetched yolo11x ({dest})"


def include_qwen(environ: Optional[dict[str, str]] = None) -> bool:
    """True when this process should Hub-pull the local Qwen snapshot.

    Submission and ``MARC_QWEN_BACKEND=local`` bake the weights. Remote/vLLM
    practice builds skip the 17.5 GB pull so ``--build`` after a code change
    does not wait on Hugging Face.
    """
    env = os.environ if environ is None else environ
    return str(env.get("MARC_QWEN_BACKEND") or "local").strip().lower() == "local"


def fetch_all(
    *,
    paths: Optional[dict[str, Path]] = None,
    snapshot_download: Optional[SnapshotFn] = None,
    hub_download=None,
    skip_if_complete: bool = True,
    bake_qwen: Optional[bool] = None,
) -> list[str]:
    _allow_hub()
    dests = paths or _default_paths()
    should_qwen = include_qwen() if bake_qwen is None else bake_qwen
    lines: list[str] = []
    if should_qwen:
        lines.append(
            fetch_qwen(
                dests["qwen"],
                snapshot_download=snapshot_download,
                skip_if_complete=skip_if_complete,
            )
        )
    else:
        lines.append("skip qwen (MARC_QWEN_BACKEND is not local)")
    lines.extend(
        [
            # Fast-iteration branch: do not fetch GraspGenX checkpoints or grippers.
            # fetch_graspgenx(
            #     dests["graspgenx"],
            #     snapshot_download=snapshot_download,
            #     skip_if_complete=skip_if_complete,
            # ),
            # fetch_grippers(
            #     dests["grippers"],
            #     snapshot_download=snapshot_download,
            #     skip_if_complete=skip_if_complete,
            # ),
            # fetch_yolo11x(
            #     dests["yolo11x"],
            #     hub_download=hub_download,
            #     skip_if_complete=skip_if_complete,
            # ),
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when a complete snapshot is already present",
    )
    args = parser.parse_args()
    contract = json.loads((_HERE / "stage1-sar-qwen" / "runtime_contract.json").read_text())
    revision = (contract.get("model") or {}).get("revision")
    if revision and revision != QWEN_REVISION:
        raise SystemExit(
            f"fetch pin {QWEN_REVISION} does not match runtime_contract.json {revision}"
        )
    for line in fetch_all(skip_if_complete=not args.force):
        print(f"[fetch-runtime] {line}", flush=True)


if __name__ == "__main__":
    main()
