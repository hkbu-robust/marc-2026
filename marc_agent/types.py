"""Stage 1 perception data types shared across detection, selection, and projection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectCandidate:
    """One detected object or landmark instance in a CCTV frame."""

    candidate_id: str
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    camera_id: str
    image_size: tuple[int, int]


@dataclass(frozen=True)
class SarCandidate:
    """One person detection with a predicted SAR situation in a CCTV frame."""

    candidate_id: str
    bbox_xyxy: tuple[float, float, float, float]
    situation: str
    confidence: float
    camera_id: str
    image_size: tuple[int, int]


@dataclass
class CctvFrame:
    """One decoded CCTV RGB frame fetched for a mission round."""

    camera_id: str
    width: int
    height: int
    rgb: object = field(repr=False)


@dataclass
class CctvDetectionRound:
    """All object and SAR candidates produced for one Stage 1 mission round."""

    objects: list[ObjectCandidate] = field(default_factory=list)
    sar: list[SarCandidate] = field(default_factory=list)
    frames: dict[str, CctvFrame] = field(default_factory=dict, repr=False)
    camera_ids: list[str] = field(default_factory=list)
