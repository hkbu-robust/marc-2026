"""Orchestrate one Stage 1 CCTV detection round."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Optional

from marc_agent.perception.bundle_paths import STAGE1_CCTV_WEIGHTS
from marc_agent.perception.cctv_round import fetch_cctv_round
from marc_agent.perception.cctv_yolo import detect_frames_with_yolo
from marc_agent.perception.stub_detections import load_stub_detections, load_stub_sar
from marc_agent.types import CctvDetectionRound

if TYPE_CHECKING:
    from marc_sdk import MARCClient

log = logging.getLogger("marc_agent.stage1_detect")

_STUB_MODES = frozenset({"stub", "json", "fixture"})
_YOLO_MODES = frozenset({"yolo", "live"})
_SAR_YOLO = frozenset({"yolo", "pose"})


def _sar_detector_mode() -> str:
    return (os.environ.get("MARC_SAR_DETECTOR") or "stub").strip().lower()


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
) -> CctvDetectionRound:
    """Fetch CCTV frames once, then produce candidates for this mission round.

    Mode selection:

    - ``MARC_STAGE1_DETECTOR=stub`` — JSON fixture (unit tests)
    - ``MARC_STAGE1_DETECTOR=yolo`` — bundled ``model.pt`` + ``runtime_contract.json``
    - unset — ``yolo`` when the bundled weights exist, otherwise ``stub``

    SAR candidates (``MARC_SAR_DETECTOR``, default ``stub``):

    - ``stub`` — from ``MARC_STUB_DETECTIONS`` JSON (empty on live runs unless set)
    - ``yolo`` / ``pose`` — provisional YOLO person-pose mapping (legacy)
    - ``qwen`` — reserved for the future Qwen recognizer
    """
    t0 = time.perf_counter()
    frames, camera_ids = fetch_cctv_round(client)
    fetch_ms = (time.perf_counter() - t0) * 1000.0

    mode = _default_detector_mode()

    if mode in _STUB_MODES:
        round_ = load_stub_detections(
            stub_path,
            camera_ids=camera_ids,
            frames=frames,
        )
        log.debug("[stage1-detect] stub fetch=%.0fms", fetch_ms)
        return round_

    if mode in _YOLO_MODES:
        t1 = time.perf_counter()
        sar_mode = _sar_detector_mode()
        include_sar_yolo = sar_mode in _SAR_YOLO
        objects, sar_yolo = detect_frames_with_yolo(frames, include_sar=include_sar_yolo)
        if sar_mode == "stub":
            sar = load_stub_sar(stub_path, camera_ids=camera_ids)
        elif include_sar_yolo:
            sar = sar_yolo
        else:
            # qwen and other future recognizers attach here
            if sar_mode == "qwen":
                log.debug("[stage1-detect] sar_mode=qwen not implemented yet")
            sar = []
        infer_ms = (time.perf_counter() - t1) * 1000.0
        log.info(
            "[stage1-detect] yolo fetch=%.0fms infer=%.0fms objects=%d sar=%d "
            "cameras=%d sar_mode=%s",
            fetch_ms,
            infer_ms,
            len(objects),
            len(sar),
            len(frames),
            sar_mode,
        )
        return CctvDetectionRound(
            objects=objects,
            sar=sar,
            frames=frames,
            camera_ids=camera_ids,
        )

    log.warning("[stage1-detect] unknown mode %r — empty candidates", mode)
    return CctvDetectionRound(objects=[], sar=[], frames=frames, camera_ids=camera_ids)
