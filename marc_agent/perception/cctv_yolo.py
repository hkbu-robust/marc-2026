"""Run the bundled Stage 1 CCTV YOLO detector on fetched frames."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from marc_agent.perception.bundle_paths import (
    STAGE1_CCTV_CONTRACT,
    STAGE1_CCTV_WEIGHTS,
)
from marc_agent.perception.candidate_ids import (
    object_candidate_id,
    person_candidate_id,
)
from marc_agent.perception.pose_to_situation import is_person_pose, situation_for_pose
from marc_agent.types import (
    CctvDetectionRound,
    CctvFrame,
    ObjectCandidate,
    PersonCandidate,
    SarCandidate,
)

log = logging.getLogger("marc_agent.cctv_yolo")

REFINE_CONF = 0.12
CROP_PAD_FRAC = 0.15


def _pad_box(
    box: tuple[float, float, float, float],
    image_size: tuple[float, float],
    pad_frac: float = CROP_PAD_FRAC,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = float(image_size[0]), float(image_size[1])
    pad = pad_frac * max(width, height)
    return (
        max(0.0, float(x1) - pad),
        max(0.0, float(y1) - pad),
        min(width, float(x2) + pad),
        min(height, float(y2) + pad),
    )


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _resolve_device(requested: str) -> str:
    req = (requested or "0").strip()
    if req.lower() in ("cpu", "cpu:0"):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return req
        log.warning("[cctv-yolo] CUDA unavailable — using CPU")
    except ImportError:
        log.warning("[cctv-yolo] torch not installed — using CPU")
    return "cpu"


@dataclass(frozen=True)
class RuntimeContract:
    weights: Path
    imgsz: int
    confidence: float
    nms_iou: float
    candidate_limit: int
    use_fp16: bool
    ordered_classes: tuple[str, ...]
    ignored_classes: tuple[str, ...]

    @classmethod
    def load(cls, contract_path: Path = STAGE1_CCTV_CONTRACT) -> "RuntimeContract":
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        inference = payload.get("inference") or {}
        return cls(
            weights=STAGE1_CCTV_WEIGHTS,
            imgsz=int(inference.get("imgsz", 1920)),
            confidence=_env_float(
                "MARC_CCTV_YOLO_CONF",
                float(inference.get("confidence", 0.3)),
            ),
            nms_iou=float(inference.get("nms_iou", 0.5)),
            candidate_limit=int(inference.get("candidate_limit", 50)),
            use_fp16=str(inference.get("quantize", "")).lower() == "fp16",
            ordered_classes=tuple(payload.get("ordered_classes") or ()),
            ignored_classes=tuple(payload.get("ignored_classes") or ()),
        )


def detections_to_candidates(
    camera_id: str,
    image_size: tuple[int, int],
    detections: list[tuple[str, float, tuple[float, float, float, float]]],
) -> tuple[list[ObjectCandidate], list[PersonCandidate]]:
    objects: list[ObjectCandidate] = []
    people: list[PersonCandidate] = []
    class_counts: dict[str, int] = defaultdict(int)

    ordered = sorted(detections, key=lambda item: (item[2], item[0], -item[1]))
    for class_name, confidence, bbox in ordered:
        if is_person_pose(class_name):
            idx = class_counts[class_name]
            class_counts[class_name] += 1
            people.append(
                PersonCandidate(
                    candidate_id=person_candidate_id(camera_id, class_name, idx),
                    bbox_xyxy=bbox,
                    pose_class=class_name,
                    confidence=confidence,
                    camera_id=camera_id,
                    image_size=image_size,
                )
            )
            continue

        idx = class_counts[class_name]
        class_counts[class_name] += 1
        candidate_id = object_candidate_id(camera_id, class_name, idx)
        objects.append(
            ObjectCandidate(
                candidate_id=candidate_id,
                class_name=class_name,
                bbox_xyxy=bbox,
                confidence=confidence,
                camera_id=camera_id,
                image_size=image_size,
            )
        )

    return objects, people


def person_candidates_to_sar(people: list[PersonCandidate]) -> list[SarCandidate]:
    """Adapt raw person proposals to the provisional pose-only SAR contract."""
    return [
        SarCandidate(
            candidate_id=person.candidate_id,
            bbox_xyxy=person.bbox_xyxy,
            situation=situation_for_pose(person.pose_class),
            confidence=person.confidence,
            camera_id=person.camera_id,
            image_size=person.image_size,
        )
        for person in people
    ]


class CctvYoloDetector:
    """Ultralytics wrapper for the bundled Stage 1 checkpoint."""

    def __init__(self, contract: Optional[RuntimeContract] = None):
        self.contract = contract or RuntimeContract.load()
        self.model = None
        self.device = "cpu"
        self.names: dict[int, str] = {}

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.contract.weights.is_file():
            raise FileNotFoundError(
                f"Bundled CCTV weights missing: {self.contract.weights}"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for Stage 1 YOLO inference. "
                "Install ultralytics-opencv-headless in the agent environment."
            ) from exc

        self.device = _resolve_device(_env_str("MARC_CCTV_YOLO_DEVICE", "0"))
        log.info(
            "[cctv-yolo] loading %s device=%s imgsz=%d conf=%.2f iou=%.2f max_det=%d",
            self.contract.weights,
            self.device,
            self.contract.imgsz,
            self.contract.confidence,
            self.contract.nms_iou,
            self.contract.candidate_limit,
        )
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        self.model = YOLO(str(self.contract.weights))
        raw_names = getattr(self.model, "names", None) or {}
        self.names = {int(k): str(v) for k, v in dict(raw_names).items()}

    def _predict_frame(
        self, rgb: np.ndarray, *, confidence: Optional[float] = None
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        self.load()
        assert self.model is not None

        h, w = rgb.shape[:2]
        if h < 2 or w < 2:
            return []

        src: object = np.ascontiguousarray(rgb)
        try:
            from PIL import Image

            src = Image.fromarray(rgb, mode="RGB")
        except ImportError:
            pass

        use_fp16 = self.contract.use_fp16 and self.device != "cpu"
        predict_kwargs = {
            "verbose": False,
            "conf": float(self.contract.confidence if confidence is None else confidence),
            "iou": self.contract.nms_iou,
            "imgsz": self.contract.imgsz,
            "max_det": self.contract.candidate_limit,
            "device": self.device,
        }
        if use_fp16:
            predict_kwargs["quantize"] = "fp16"
        results = self.model.predict(src, **predict_kwargs)
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy()

        out: list[tuple[str, float, tuple[float, float, float, float]]] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            cls_id = int(clss[i])
            class_name = self.names.get(cls_id, str(cls_id))
            out.append((class_name, float(confs[i]), (x1, y1, x2, y2)))
        return out

    def detect(
        self,
        frames: dict[str, CctvFrame],
    ) -> tuple[list[ObjectCandidate], list[PersonCandidate]]:
        all_objects: list[ObjectCandidate] = []
        all_people: list[PersonCandidate] = []

        for camera_id in sorted(frames):
            frame = frames[camera_id]
            rgb = frame.rgb
            if rgb is None:
                continue
            detections = self._predict_frame(np.asarray(rgb))
            image_size = (int(frame.width), int(frame.height))
            objects, people = detections_to_candidates(camera_id, image_size, detections)
            all_objects.extend(objects)
            all_people.extend(people)

        return all_objects, all_people

    def refine_around_landmarks(
        self,
        frames: dict[str, CctvFrame],
        objects: list[ObjectCandidate],
        target_type: str,
        landmark_classes: frozenset[str],
    ) -> list[ObjectCandidate]:
        """Re-run YOLO on padded landmark crops when the requested class is missing."""
        if not landmark_classes or not target_type:
            return []
        if target_type in self.contract.ignored_classes:
            return []
        if target_type not in self.contract.ordered_classes:
            return []
        extra: list[ObjectCandidate] = []
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for item in objects:
            counts[(item.camera_id, item.class_name)] += 1
        for landmark in objects:
            if landmark.class_name not in landmark_classes:
                continue
            frame = frames.get(landmark.camera_id)
            rgb = None if frame is None else getattr(frame, "rgb", None)
            if rgb is None:
                continue
            array = np.asarray(rgb)
            height, width = array.shape[:2]
            x1, y1, x2, y2 = _pad_box(
                landmark.bbox_xyxy, (width, height)
            )
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(max(x1 + 1, x2)), int(max(y1 + 1, y2))
            crop = array[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            detections = self._predict_frame(crop, confidence=REFINE_CONF)
            image_size = (int(frame.width), int(frame.height))
            for class_name, confidence, bbox in detections:
                if class_name != target_type:
                    continue
                mapped = (
                    bbox[0] + ix1,
                    bbox[1] + iy1,
                    bbox[2] + ix1,
                    bbox[3] + iy1,
                )
                key = (landmark.camera_id, class_name)
                idx = counts[key]
                counts[key] += 1
                extra.append(
                    ObjectCandidate(
                        candidate_id=object_candidate_id(
                            landmark.camera_id, class_name, idx
                        ),
                        class_name=class_name,
                        bbox_xyxy=mapped,
                        confidence=confidence,
                        camera_id=landmark.camera_id,
                        image_size=image_size,
                    )
                )
        return extra

    def unload(self) -> None:
        model = self.model
        self.model = None
        self.names = {}
        if model is not None:
            del model


_detector: Optional[CctvYoloDetector] = None


def unload_detector() -> None:
    """Drop the process-wide CCTV YOLO singleton."""
    global _detector
    if _detector is None:
        return
    _detector.unload()
    _detector = None


def detect_frames_with_yolo(
    frames: dict[str, CctvFrame],
) -> tuple[list[ObjectCandidate], list[PersonCandidate]]:
    global _detector
    if _detector is None:
        _detector = CctvYoloDetector()
    return _detector.detect(frames)


def detector_can_emit(class_name: str) -> bool:
    """True when the bundled CCTV head is trained to output this class."""
    name = (class_name or "").strip().lower()
    if not name:
        return False
    contract = RuntimeContract.load()
    if name in contract.ignored_classes:
        return False
    return name in contract.ordered_classes


def refine_detection_round(parsed, round_: CctvDetectionRound) -> CctvDetectionRound:
    """Add missing trained-class boxes from landmark crops. No-op without a live detector."""
    if parsed is None or parsed.target_type == "person":
        return round_
    if any(item.class_name == parsed.target_type for item in round_.objects):
        return round_
    global _detector
    if _detector is None:
        return round_
    names = set()
    if parsed.landmark:
        names.add(parsed.landmark)
    elif parsed.scene_hint == "park":
        names.update({"picnic_table", "bench"})
    extra = _detector.refine_around_landmarks(
        round_.frames,
        round_.objects,
        parsed.target_type,
        frozenset(names),
    )
    if not extra:
        return round_
    objects = list(round_.objects) + extra
    return CctvDetectionRound(
        objects=objects,
        people=list(round_.people),
        sar=list(round_.sar),
        frames=round_.frames,
        camera_ids=list(round_.camera_ids),
        metrics=round_.metrics,
    )
