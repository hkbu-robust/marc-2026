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
from marc_agent.perception.pose_to_situation import is_person_pose, situation_for_pose
from marc_agent.types import CctvFrame, ObjectCandidate, SarCandidate

log = logging.getLogger("marc_agent.cctv_yolo")


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


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

    @classmethod
    def load(cls, contract_path: Path = STAGE1_CCTV_CONTRACT) -> "RuntimeContract":
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        inference = payload.get("inference") or {}
        return cls(
            weights=STAGE1_CCTV_WEIGHTS,
            imgsz=int(inference.get("imgsz", 1920)),
            confidence=float(inference.get("confidence", 0.3)),
            nms_iou=float(inference.get("nms_iou", 0.5)),
            candidate_limit=int(inference.get("candidate_limit", 50)),
            use_fp16=str(inference.get("quantize", "")).lower() == "fp16",
            ordered_classes=tuple(payload.get("ordered_classes") or ()),
        )


def detections_to_candidates(
    camera_id: str,
    image_size: tuple[int, int],
    detections: list[tuple[str, float, tuple[float, float, float, float]]],
    *,
    include_sar: bool = False,
) -> tuple[list[ObjectCandidate], list[SarCandidate]]:
    objects: list[ObjectCandidate] = []
    sar: list[SarCandidate] = []
    class_counts: dict[str, int] = defaultdict(int)

    for class_name, confidence, bbox in detections:
        if is_person_pose(class_name):
            if not include_sar:
                continue
            idx = class_counts[class_name]
            class_counts[class_name] += 1
            candidate_id = f"{camera_id}:{class_name}:{idx}"
            sar.append(
                SarCandidate(
                    candidate_id=candidate_id,
                    bbox_xyxy=bbox,
                    situation=situation_for_pose(class_name),
                    confidence=confidence,
                    camera_id=camera_id,
                    image_size=image_size,
                )
            )
            continue

        idx = class_counts[class_name]
        class_counts[class_name] += 1
        candidate_id = f"{camera_id}:{class_name}:{idx}"
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

    return objects, sar


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
        self.model = YOLO(str(self.contract.weights))
        raw_names = getattr(self.model, "names", None) or {}
        self.names = {int(k): str(v) for k, v in dict(raw_names).items()}

    def _predict_frame(
        self, rgb: np.ndarray
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

        use_half = self.contract.use_fp16 and self.device != "cpu"
        results = self.model.predict(
            src,
            verbose=False,
            conf=self.contract.confidence,
            iou=self.contract.nms_iou,
            imgsz=self.contract.imgsz,
            max_det=self.contract.candidate_limit,
            device=self.device,
            half=use_half,
        )
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
        *,
        include_sar: bool = False,
    ) -> tuple[list[ObjectCandidate], list[SarCandidate]]:
        all_objects: list[ObjectCandidate] = []
        all_sar: list[SarCandidate] = []

        for camera_id in sorted(frames):
            frame = frames[camera_id]
            rgb = frame.rgb
            if rgb is None:
                continue
            detections = self._predict_frame(np.asarray(rgb))
            image_size = (int(frame.width), int(frame.height))
            objects, sar = detections_to_candidates(
                camera_id, image_size, detections, include_sar=include_sar
            )
            all_objects.extend(objects)
            all_sar.extend(sar)

        return all_objects, all_sar


_detector: Optional[CctvYoloDetector] = None


def detect_frames_with_yolo(
    frames: dict[str, CctvFrame],
    *,
    include_sar: bool = False,
) -> tuple[list[ObjectCandidate], list[SarCandidate]]:
    global _detector
    if _detector is None:
        _detector = CctvYoloDetector()
    return _detector.detect(frames, include_sar=include_sar)
