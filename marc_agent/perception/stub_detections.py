"""Load Stage 1 detection candidates from a JSON fixture (provisional detector)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from marc_agent.types import CctvDetectionRound, CctvFrame, ObjectCandidate, SarCandidate


def default_stub_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "tests" / "fixtures" / "stub_detections.json")


def _bbox(raw) -> tuple[float, float, float, float]:
    if not raw or len(raw) < 4:
        raise ValueError(f"bbox_xyxy must have four numbers, got {raw!r}")
    return (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))


def _image_size(raw) -> tuple[int, int]:
    if not raw or len(raw) < 2:
        raise ValueError(f"image_size must be [width, height], got {raw!r}")
    return (int(raw[0]), int(raw[1]))


def _object_candidate(raw: dict) -> ObjectCandidate:
    return ObjectCandidate(
        candidate_id=str(raw["candidate_id"]),
        class_name=str(raw["class_name"]),
        bbox_xyxy=_bbox(raw["bbox_xyxy"]),
        confidence=float(raw["confidence"]),
        camera_id=str(raw["camera_id"]),
        image_size=_image_size(raw["image_size"]),
    )


def _sar_candidate(raw: dict) -> SarCandidate:
    return SarCandidate(
        candidate_id=str(raw["candidate_id"]),
        bbox_xyxy=_bbox(raw["bbox_xyxy"]),
        situation=str(raw["situation"]),
        confidence=float(raw["confidence"]),
        camera_id=str(raw["camera_id"]),
        image_size=_image_size(raw["image_size"]),
    )


def _filter_camera_ids(items: Iterable, camera_ids: Optional[list[str]]):
    if not camera_ids:
        return list(items)
    allowed = set(camera_ids)
    return [item for item in items if item.camera_id in allowed]


def load_stub_sar(
    path: Optional[str] = None,
    *,
    camera_ids: Optional[list[str]] = None,
) -> list[SarCandidate]:
    """Load SAR candidates only from the stub JSON fixture."""
    stub_path = path or os.environ.get("MARC_STUB_DETECTIONS") or default_stub_path()
    with open(stub_path, encoding="utf-8") as stream:
        payload = json.load(stream)
    sar = [_sar_candidate(raw) for raw in payload.get("sar") or []]
    return _filter_camera_ids(sar, camera_ids)


def load_stub_detections(
    path: Optional[str] = None,
    *,
    camera_ids: Optional[list[str]] = None,
    frames: Optional[dict[str, CctvFrame]] = None,
) -> CctvDetectionRound:
    """Load object and SAR candidates from a JSON fixture.

    When ``camera_ids`` is non-empty, candidates for unknown cameras are dropped.
    ``frames`` and ``camera_ids`` are copied into the returned round for downstream use.
    """
    stub_path = path or os.environ.get("MARC_STUB_DETECTIONS") or default_stub_path()
    with open(stub_path, encoding="utf-8") as stream:
        payload = json.load(stream)

    objects = [_object_candidate(raw) for raw in payload.get("objects") or []]
    sar = [_sar_candidate(raw) for raw in payload.get("sar") or []]
    objects = _filter_camera_ids(objects, camera_ids)
    sar = _filter_camera_ids(sar, camera_ids)

    return CctvDetectionRound(
        objects=objects,
        sar=sar,
        frames=dict(frames or {}),
        camera_ids=list(camera_ids or []),
    )
