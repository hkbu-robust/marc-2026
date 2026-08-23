"""Orchestrate one Stage 1 CCTV detection round."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

from marc_agent.perception.cctv_round import fetch_cctv_round
from marc_agent.perception.stub_detections import load_stub_detections
from marc_agent.types import CctvDetectionRound

if TYPE_CHECKING:
    from marc_sdk import MARCClient

_STUB_MODES = frozenset({"stub", "json", "fixture"})


def detect_stage1_round(
    client: MARCClient,
    *,
    stub_path: Optional[str] = None,
) -> CctvDetectionRound:
    """Fetch CCTV frames once, then produce candidates for this mission round.

    Detector mode is selected with ``MARC_STAGE1_DETECTOR`` (default ``stub``):
      - ``stub`` / ``json`` / ``fixture``: load candidates from JSON
      - future modes (``yolo``, etc.) will run a real model on ``frames``
    """
    frames, camera_ids = fetch_cctv_round(client)
    mode = (os.environ.get("MARC_STAGE1_DETECTOR") or "stub").strip().lower()

    if mode in _STUB_MODES:
        return load_stub_detections(
            stub_path,
            camera_ids=camera_ids,
            frames=frames,
        )

    # Real detector adapters (#7 follow-up) attach here.
    return CctvDetectionRound(objects=[], sar=[], frames=frames, camera_ids=camera_ids)
