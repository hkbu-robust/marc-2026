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
    source: str = "detector"  # or "zoom", "landmark_crop"


@dataclass(frozen=True)
class PersonCandidate:
    """One raw YOLO person proposal retained before SAR interpretation."""

    candidate_id: str
    bbox_xyxy: tuple[float, float, float, float]
    pose_class: str
    confidence: float
    camera_id: str
    image_size: tuple[int, int]


@dataclass(frozen=True)
class SarCandidate:
    """Legacy interpreted person candidate used by the pose-only fallback."""

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


@dataclass(frozen=True)
class CctvRoundMetrics:
    """Structured measurements and failures for one candidate-collection round."""

    fetch_ms: float = 0.0
    load_ms: float = 0.0
    inference_ms: float = 0.0
    cameras_discovered: int = 0
    frames_loaded: int = 0
    failures: tuple[str, ...] = ()


@dataclass
class CctvDetectionRound:
    """All raw and legacy candidates produced for one Stage 1 mission round."""

    objects: list[ObjectCandidate] = field(default_factory=list)
    people: list[PersonCandidate] = field(default_factory=list)
    sar: list[SarCandidate] = field(default_factory=list)
    frames: dict[str, CctvFrame] = field(default_factory=dict, repr=False)
    camera_ids: list[str] = field(default_factory=list)
    metrics: CctvRoundMetrics = field(default_factory=CctvRoundMetrics)
