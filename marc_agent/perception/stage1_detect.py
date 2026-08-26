"""Orchestrate one Stage 1 CCTV detection round."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional

from marc_agent.perception.bundle_paths import STAGE1_CCTV_WEIGHTS
from marc_agent.perception.cctv_round import fetch_cctv_round
from marc_agent.perception.cctv_yolo import (
    detect_frames_with_yolo,
    person_candidates_to_sar,
)
from marc_agent.perception.stub_detections import load_stub_detections, load_stub_sar
from marc_agent.types import CctvDetectionRound, CctvRoundMetrics

if TYPE_CHECKING:
    from marc_sdk import MARCClient

log = logging.getLogger("marc_agent.stage1_detect")

_STUB_MODES = frozenset({"stub", "json", "fixture"})
_YOLO_MODES = frozenset({"yolo", "live"})
_SAR_YOLO = frozenset({"yolo", "pose"})


def _sar_detector_mode() -> str:
    return (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()


def _default_detector_mode() -> str:
    explicit = (os.environ.get("MARC_STAGE1_DETECTOR") or "").strip().lower()
    if explicit:
        return explicit
    if STAGE1_CCTV_WEIGHTS.is_file():
        return "yolo"
    return "stub"


def detect_stage1_round(
    client: MARCClient,
    *,
    stub_path: Optional[str] = None,
    discovery_timeout_s: Optional[float] = None,
) -> CctvDetectionRound:
    """Fetch CCTV frames once, then produce candidates for this mission round.

    Mode selection:

    - ``MARC_STAGE1_DETECTOR=stub`` — JSON fixture (unit tests)
    - ``MARC_STAGE1_DETECTOR=yolo`` — bundled ``model.pt`` + ``runtime_contract.json``
    - unset — ``yolo`` when the bundled weights exist, otherwise ``stub``

    Legacy SAR adapters (``MARC_SAR_DETECTOR``, default ``fallback``):

    - ``fallback`` — retain raw people; the query-conditioned selector ranks them
    - ``stub`` — explicit unit-test fixture only
    - ``yolo`` / ``pose`` — provisional YOLO person-pose mapping (legacy)
    - ``qwen`` — raw people are verified after parsing in ``stage1_pipeline``
    """
    t0 = time.perf_counter()
    frames, camera_ids = fetch_cctv_round(
        client, discovery_timeout_s=discovery_timeout_s
    )
    fetch_ms = (time.perf_counter() - t0) * 1000.0

    mode = _default_detector_mode()

    if mode in _STUB_MODES:
        round_ = load_stub_detections(
            stub_path,
            camera_ids=camera_ids,
            frames=frames,
        )
        round_.metrics = CctvRoundMetrics(
            fetch_ms=fetch_ms,
            cameras_discovered=len(camera_ids),
            frames_loaded=len(frames),
            failures=tuple(
                f"missing_frame:{camera_id}"
                for camera_id in camera_ids
                if camera_id not in frames
            ),
        )
        log.debug("[stage1-detect] stub fetch=%.0fms", fetch_ms)
        return round_

    if mode in _YOLO_MODES:
        t1 = time.perf_counter()
        sar_mode = _sar_detector_mode()
        objects, people = detect_frames_with_yolo(frames)
        if sar_mode == "stub":
            sar = load_stub_sar(stub_path, camera_ids=camera_ids)
        elif sar_mode in _SAR_YOLO:
            sar = person_candidates_to_sar(people)
        else:
            # Raw people are always retained. Query-conditioned interpretation happens
            # after command parsing in stage1_pipeline.
            sar = []
        infer_ms = (time.perf_counter() - t1) * 1000.0
        log.info(
            "[stage1-detect] yolo fetch=%.0fms infer=%.0fms objects=%d people=%d sar=%d "
            "cameras=%d sar_mode=%s",
            fetch_ms,
            infer_ms,
            len(objects),
            len(people),
            len(sar),
            len(frames),
            sar_mode,
        )
        return CctvDetectionRound(
            objects=objects,
            people=people,
            sar=sar,
            frames=frames,
            camera_ids=camera_ids,
            metrics=CctvRoundMetrics(
                fetch_ms=fetch_ms,
                inference_ms=infer_ms,
                cameras_discovered=len(camera_ids),
                frames_loaded=len(frames),
                failures=tuple(
                    f"missing_frame:{camera_id}"
                    for camera_id in camera_ids
                    if camera_id not in frames
                ),
            ),
        )
    log.warning("[stage1-detect] unknown mode %r — empty candidates", mode)
    return CctvDetectionRound(
        objects=[],
        sar=[],
        frames=frames,
        camera_ids=camera_ids,
        metrics=CctvRoundMetrics(
            fetch_ms=fetch_ms,
            cameras_discovered=len(camera_ids),
            frames_loaded=len(frames),
            failures=(f"unknown_detector_mode:{mode}",),
        ),
    )
