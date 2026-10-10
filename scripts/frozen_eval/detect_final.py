"""Final-system detector pass on the stored held-out frames (#87 refresh).

Run inside ``marc-agent:final-system-v1`` with the image's own ``marc_agent`` (the tag's
code; nothing from a checkout is mounted over it) and its baked weights::

    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --entrypoint python3 \\
      -v $REPO/scripts/frozen_eval:/work/frozen_eval:ro -v $ROOT:$ROOT \\
      marc-agent:final-system-v1 /work/frozen_eval/detect_final.py \\
      $ROOT heldout $PARSES.jsonl $OUT.jsonl

Per case, as ``run_stage1_candidate_pipeline`` does it: the full-frame pass
(``detect_frames_with_yolo``, runtime contract: 1920 px, conf 0.30, IoU 0.50, top 50),
then ``refine_detection_round`` driven by the case's **final parse** (``parsed_final``
from ``parse_final.py``, i.e. what the Qwen reader gives the runtime): the zoomed
small-target pass (``MARC_TILED_REFINE`` default on, 2x imgsz, conf >= 0.4) and the
landmark-crop pass (960 px, conf 0.05, every mission class), or the person crops for
SAR. No clock budget (offline). Selection-time hypotheses (``add_hypotheses``) are
not detections and are not stored.

The full-frame output is computed once per frame set and reused by the cases on the
same frames. CUDA OOM in any pass is logged by the runtime as a warning; every such
warning and the zoom pass's ``oom_skipped`` are stored per case (``oom``), so those
cases can be deleted from OUT and re-run (the script resumes, skipping done cases).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "/agent")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from marc_agent.command_parser import Intent, ParsedCommand, Relation, Situation  # noqa: E402
from marc_agent.experiment_log import decision_scope  # noqa: E402
from marc_agent.perception import cctv_yolo  # noqa: E402
from marc_agent.perception.cctv_yolo import detect_frames_with_yolo, refine_detection_round  # noqa: E402
from marc_agent.types import CctvDetectionRound, CctvFrame  # noqa: E402

WEIGHTS = Path("/agent/models/runtime/stage1-cctv-detector/model.pt")


class _OomLog(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        message = record.getMessage()
        if "OOM" in message or "out of memory" in message.lower():
            self.messages.append(message)


def to_parsed(raw: dict) -> ParsedCommand:
    return ParsedCommand(
        intent=Intent(raw["intent"]),
        target_type=raw["target_type"],
        landmark=raw.get("landmark"),
        relation=Relation(raw["relation"]) if raw.get("relation") else None,
        situation=Situation(raw["situation"]) if raw.get("situation") else None,
        scene_hint=raw.get("scene_hint"),
        place_cues=tuple(raw.get("place_cues") or ()),
    )


def _record(candidate, source: str) -> dict:
    return {
        "candidate_id": candidate.candidate_id,
        "camera_id": candidate.camera_id,
        "class_name": getattr(candidate, "class_name", None) or getattr(candidate, "pose_class", None),
        "confidence": float(candidate.confidence),
        "bbox_xyxy": [float(v) for v in candidate.bbox_xyxy],
        "source": source,
    }


def run(root: Path, set_name: str, parses_path: Path, out_path: Path) -> int:
    parses = {}
    for line in parses_path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            parses[row["case_id"]] = row["parsed_final"]
    paths = sorted((root / set_name / "cases").glob("*/case.json"))
    done = set()
    if out_path.exists():
        done = {json.loads(line)["case_id"] for line in out_path.read_text().splitlines() if line}
    oom_log = _OomLog()
    logging.getLogger("marc_agent").addHandler(oom_log)
    logging.getLogger("marc_agent").setLevel(logging.INFO)
    weights_sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    env = {k: v for k, v in os.environ.items() if k.startswith("MARC_")}
    base_cache: dict[tuple, tuple] = {}
    with out_path.open("a", encoding="utf-8") as out:
        for index, path in enumerate(paths):
            case = json.loads(path.read_text(encoding="utf-8"))
            case_id = case["case_id"]
            if case_id in done:
                continue
            parsed = to_parsed(parses[case_id])
            frames = {}
            for camera_id, rel in sorted(case["frames"].items()):
                rgb = np.asarray(Image.open(path.parent / rel).convert("RGB"))
                frames[camera_id] = CctvFrame(camera_id, int(rgb.shape[1]), int(rgb.shape[0]), rgb)
            frame_key = tuple(sorted((cam, hashlib.sha256((path.parent / rel).read_bytes()).hexdigest())
                                     for cam, rel in case["frames"].items()))
            oom_log.messages.clear()
            started = time.perf_counter()
            if frame_key not in base_cache:
                base_cache.clear()
                objects, people, _, _ = detect_frames_with_yolo(frames)
                base_cache[frame_key] = (objects, people, list(oom_log.messages))
            objects, people, base_oom = base_cache[frame_key]
            round_ = CctvDetectionRound(objects=list(objects), people=list(people),
                                        frames=frames, camera_ids=sorted(frames))
            with decision_scope("stage1") as notes:
                refined = refine_detection_round(parsed, round_)
            base_ids = {c.candidate_id for c in list(objects) + list(people)}
            candidates = []
            for c in list(refined.objects) + list(refined.people):
                if c.candidate_id in base_ids:
                    source = "detector"
                else:
                    source = getattr(c, "source", None) or "refine"
                    if source == "detector":  # people from support crops carry no source
                        source = "refine"
                candidates.append(_record(c, source))
            zoom_skipped = [s for n in notes if n.get("decision") == "tiled_refine"
                            for s in n.get("oom_skipped") or []]
            out.write(json.dumps({
                "case_id": case_id,
                "parsed": parses[case_id],
                "candidates": candidates,
                "decisions": notes,
                "oom": {"messages": base_oom + list(oom_log.messages), "zoom_skipped": zoom_skipped},
                "detector": {
                    "weights_sha256": weights_sha,
                    "contract": asdict(cctv_yolo._detector.contract) if cctv_yolo._detector else None,
                    "small_target_classes": list(cctv_yolo._detector.contract.small_target_classes)
                    if cctv_yolo._detector else None,
                    "tiled_refine": cctv_yolo.tiled_refine_enabled(),
                    "env": env,
                },
                "ms": (time.perf_counter() - started) * 1000.0,
            }, default=str) + "\n")
            out.flush()
            if index % 50 == 0:
                print(f"[detect-final] {index + 1}/{len(paths)} {case_id} n={len(candidates)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])))
