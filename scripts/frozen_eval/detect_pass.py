"""Run the deployed Stage 1 detector on held-out case specs (#87).

Run inside the frozen agent image, so the detector code and weights are the ones
the public capture ran with (``marc-agent:frozen-logging``, detector SHA-256
``64296e4b...``)::

    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \\
      -v $REPO/scripts/frozen_eval:/work/frozen_eval:ro -v $SETS:$SETS \\
      --entrypoint python3 marc-agent:frozen-logging \\
      /work/frozen_eval/detect_pass.py $SETS/heldout/specs.jsonl $SETS/heldout/detections.jsonl

Per case, exactly as the runtime does it: full-frame YOLO at the runtime contract
settings (``detect_frames_with_yolo``), then the query-conditioned refinement for
the parsed command (``refine_detection_round``). Full-frame output is computed
once per frame and reused by every case on that frame. The proxies and refinement
additions are captured from the runtime's decision notes.

Standalone on purpose: only the image's ``marc_agent`` (``/agent``), numpy and PIL.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from marc_agent.command_parser import parse_command
from marc_agent.experiment_log import decision_scope
from marc_agent.perception import cctv_yolo
from marc_agent.perception.cctv_yolo import detect_frames_with_yolo, refine_detection_round
from marc_agent.types import CctvDetectionRound, CctvFrame

WEIGHTS = Path("/agent/models/runtime/stage1-cctv-detector/model.pt")


def _frames(paths: dict[str, str]) -> dict[str, CctvFrame]:
    frames = {}
    for camera_id, path in sorted(paths.items()):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        frames[camera_id] = CctvFrame(camera_id, int(rgb.shape[1]), int(rgb.shape[0]), rgb)
    return frames


def _record(candidate, source: str) -> dict:
    """Runtime record format (``Stage1CandidateResult.record``): people keep their pose in class_name."""
    return {
        "candidate_id": candidate.candidate_id,
        "camera_id": candidate.camera_id,
        "class_name": getattr(candidate, "class_name", None) or getattr(candidate, "pose_class", None),
        "confidence": float(candidate.confidence),
        "bbox_xyxy": [float(v) for v in candidate.bbox_xyxy],
        "source": source,
    }


def run(specs_path: Path, out_path: Path) -> int:
    specs = [json.loads(line) for line in specs_path.read_text(encoding="utf-8").splitlines() if line]
    done = set()
    if out_path.exists():
        done = {json.loads(line)["case_id"] for line in out_path.read_text().splitlines() if line}
    weights_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    base_cache: dict[tuple, tuple] = {}
    with out_path.open("a", encoding="utf-8") as out:
        for index, spec in enumerate(specs):
            if spec["case_id"] in done:
                continue
            frame_key = tuple(sorted(spec["frames"].items()))
            frames = _frames(spec["frames"])
            started = time.perf_counter()
            if frame_key not in base_cache:
                base_cache.clear()  # cases are grouped by scene; keep one scene in memory
                objects, people, _, _ = detect_frames_with_yolo(frames)
                base_cache[frame_key] = (objects, people)
            objects, people = base_cache[frame_key]
            parsed = parse_command(spec["command"])
            round_ = CctvDetectionRound(objects=list(objects), people=list(people),
                                        frames=frames, camera_ids=sorted(frames))
            with decision_scope("stage1") as notes:
                refined = refine_detection_round(parsed, round_)
            base_ids = {c.candidate_id for c in list(objects) + list(people)}
            proxy_ids = {n["candidate"] for n in notes if n["decision"] == "proxy_created"}
            candidates = [
                _record(c, "proxy" if c.candidate_id in proxy_ids
                        else "detector" if c.candidate_id in base_ids else "refine")
                for c in list(refined.objects) + list(refined.people)
            ]
            out.write(json.dumps({
                "case_id": spec["case_id"],
                "parsed": asdict(parsed),
                "candidates": candidates,
                "decisions": notes,
                "detector": {
                    "weights_sha256": weights_sha,
                    "contract": asdict(cctv_yolo._detector.contract) if cctv_yolo._detector else None,
                },
                "ms": (time.perf_counter() - started) * 1000.0,
            }, default=str) + "\n")
            out.flush()
            if index % 50 == 0:
                print(f"[detect-pass] {index + 1}/{len(specs)} {spec['case_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), Path(sys.argv[2])))
