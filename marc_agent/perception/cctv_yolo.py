"""Run the bundled Stage 1 CCTV YOLO detector on fetched frames."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np

from marc_agent.experiment_log import note_decision
from marc_agent.command_parser import (
    MISSION_OBJECT_TYPES,
    PARKED_VEHICLE_LANDMARKS,
    PARKING_SCENES,
)
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

REFINE_CONF = 0.05
OBJECT_REFINE_CONF = 0.005
CROP_PAD_FRAC = 0.15
OBJECT_CROP_PAD_FRAC = 1.0
REFINE_DEDUP_IOU = 0.45
REFINE_IMGSZ = 320
OBJECT_REFINE_IMGSZ = 1600
REFINE_CROP_CAP = 8


def _pad_box(
    box: tuple[float, float, float, float],
    image_size: tuple[float, float],
    pad_frac: float = CROP_PAD_FRAC,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width, height = float(image_size[0]), float(image_size[1])
    pad = pad_frac * max(float(x2) - float(x1), float(y2) - float(y1), 1.0)
    return (
        max(0.0, float(x1) - pad),
        max(0.0, float(y1) - pad),
        min(width, float(x2) + pad),
        min(height, float(y2) + pad),
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "outofmemory" in name:
        return True
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def _release_cuda() -> None:
    from marc_agent.gpu_handoff import empty_cuda_cache

    empty_cuda_cache()


def _box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


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
    except ImportError as exc:
        raise RuntimeError(
            f"CCTV YOLO requested CUDA device {req}, but torch is not installed"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CCTV YOLO requested CUDA device {req}, but CUDA is unavailable"
        )
    index = int(req.removeprefix("cuda:"))
    log.info(
        "[cctv-yolo] selected CUDA device=%s name=%s",
        req,
        torch.cuda.get_device_name(index),
    )
    return req


@dataclass(frozen=True)
class RuntimeContract:
    weights: Path
    imgsz: int
    confidence: float
    nms_iou: Optional[float]
    candidate_limit: int
    end2end: bool
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
            nms_iou=(
                None if inference.get("nms_iou") is None
                else float(inference.get("nms_iou", 0.5))
            ),
            candidate_limit=int(inference.get("candidate_limit", 50)),
            end2end=bool(inference.get("end2end", False)),
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
            self.contract.nms_iou or 0.0,
            self.contract.candidate_limit,
        )
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        self.model = YOLO(str(self.contract.weights))
        raw_names = getattr(self.model, "names", None) or {}
        self.names = {int(k): str(v) for k, v in dict(raw_names).items()}
        loaded_order = tuple(self.names[index] for index in sorted(self.names))
        if loaded_order != self.contract.ordered_classes:
            self.unload()
            raise ValueError("CCTV checkpoint class order differs from runtime contract")

    def _predict_frame(
        self,
        rgb: np.ndarray,
        *,
        confidence: Optional[float] = None,
        imgsz: Optional[int] = None,
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
        letterbox = self.contract.imgsz if imgsz is None else int(imgsz)
        predict_kwargs = {
            "verbose": False,
            "conf": float(self.contract.confidence if confidence is None else confidence),
            "imgsz": letterbox,
            "max_det": self.contract.candidate_limit,
            "device": self.device,
            "end2end": self.contract.end2end,
        }
        if self.contract.nms_iou is not None:
            predict_kwargs["iou"] = self.contract.nms_iou
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
        out.sort(key=lambda item: item[1], reverse=True)
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

        _release_cuda()
        return all_objects, all_people

    def refine_around_landmarks(
        self,
        frames: dict[str, CctvFrame],
        objects: list[ObjectCandidate],
        target_type: str,
        landmark_classes: frozenset[str],
    ) -> list[ObjectCandidate]:
        """Re-run YOLO on landmark-context crops to recover missed tiny objects."""
        if not landmark_classes or not target_type:
            return []
        if target_type in self.contract.ignored_classes:
            return []
        if target_type not in self.contract.ordered_classes:
            return []
        extra: list[ObjectCandidate] = []
        counts: dict[tuple[str, str], int] = defaultdict(int)
        occupied: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for item in objects:
            counts[(item.camera_id, item.class_name)] += 1
            if item.class_name == target_type:
                occupied[item.camera_id].append(item.bbox_xyxy)
        refine_imgsz = {"umbrella": 1600, "pencilcase": 1280}.get(
            target_type, 960
        )
        refine_confidence = (
            OBJECT_REFINE_CONF
            if target_type in {"umbrella", "pencilcase"}
            else REFINE_CONF
        )
        if target_type in {"umbrella", "pencilcase"}:
            detections = []
            for camera_id in sorted(frames):
                frame = frames[camera_id]
                if frame.rgb is None:
                    continue
                try:
                    predicted = self._predict_frame(
                        np.asarray(frame.rgb),
                        imgsz=refine_imgsz,
                        confidence=refine_confidence,
                    )
                except Exception as exc:
                    if not _is_cuda_oom(exc):
                        raise
                    log.warning(
                        "[cctv-yolo] full-frame refine OOM camera=%s; skipping",
                        camera_id,
                    )
                    predicted = []
                _release_cuda()
                detections.extend(
                    (camera_id, class_name, confidence, bbox, (frame.width, frame.height))
                    for class_name, confidence, bbox in predicted
                    if class_name == target_type
                )
            detections.extend(
                item
                for item in self._predict_around_landmarks(
                    frames,
                    objects,
                    landmark_classes,
                    landmarks=self._all_landmarks_for_refine(
                        objects, landmark_classes
                    ),
                    pad_frac=OBJECT_CROP_PAD_FRAC,
                    imgsz=refine_imgsz,
                    confidence=refine_confidence,
                )
                if item[1] == target_type
            )
        else:
            landmarks = self._all_landmarks_for_refine(objects, landmark_classes)
            detections = self._predict_around_landmarks(
                frames,
                objects,
                landmark_classes,
                landmarks=landmarks,
                pad_frac=OBJECT_CROP_PAD_FRAC,
                imgsz=refine_imgsz,
                confidence=refine_confidence,
            )
        for camera_id, class_name, confidence, mapped, image_size in detections:
            # Crop inference is deliberately proposal-oriented: tiny mission items
            # are often classified as a visually similar mission class. Qwen sees
            # the crop and resolves the final identity.
            if class_name not in MISSION_OBJECT_TYPES:
                continue
            if any(
                _box_iou(mapped, existing) >= REFINE_DEDUP_IOU
                for existing in occupied[camera_id]
            ):
                continue
            key = (camera_id, class_name)
            idx = counts[key]
            counts[key] += 1
            extra.append(
                ObjectCandidate(
                    candidate_id=object_candidate_id(camera_id, class_name, idx),
                    class_name=class_name,
                    bbox_xyxy=mapped,
                    confidence=confidence,
                    camera_id=camera_id,
                    image_size=image_size,
                )
            )
            occupied[camera_id].append(mapped)
        if target_type in {"umbrella", "pencilcase"}:
            landmarks = self._landmarks_for_refine(objects, landmark_classes)
            missing = [
                item for item in landmarks if item.camera_id not in occupied
            ]
            if len(missing) == 1:
                # ponytail: a uniquely occluded tiny target uses its landmark as
                # a projection proxy; replace if invisible-target data becomes available.
                landmark = missing[0]
                key = (landmark.camera_id, target_type)
                proxy = ObjectCandidate(
                    candidate_id=object_candidate_id(
                        landmark.camera_id, target_type, counts[key]
                    ),
                    class_name=target_type,
                    bbox_xyxy=landmark.bbox_xyxy,
                    confidence=OBJECT_REFINE_CONF / 2.0,
                    camera_id=landmark.camera_id,
                    image_size=landmark.image_size,
                )
                extra.append(proxy)
                log.info(
                    "[cctv-yolo] unique-occlusion proxy target=%s landmark=%s",
                    proxy.candidate_id,
                    landmark.candidate_id,
                )
                note_decision(
                    "proxy_created", variant="landmark_context",
                    candidate=proxy.candidate_id, landmark=landmark.candidate_id,
                )
        return extra

    def _landmarks_for_refine(
        self,
        objects: list[ObjectCandidate],
        landmark_classes: frozenset[str],
    ) -> list[ObjectCandidate]:
        """One highest-confidence landmark per camera to bound person refinement."""
        best: dict[str, ObjectCandidate] = {}
        for item in objects:
            if item.class_name not in landmark_classes:
                continue
            current = best.get(item.camera_id)
            if current is None or (-item.confidence, item.candidate_id) < (
                -current.confidence, current.candidate_id
            ):
                best[item.camera_id] = item
        return [best[camera_id] for camera_id in sorted(best)]

    def _all_landmarks_for_refine(
        self,
        objects: list[ObjectCandidate],
        landmark_classes: frozenset[str],
    ) -> list[ObjectCandidate]:
        """Matching object landmarks across all cameras, bounded by confidence."""
        picked = [
            item for item in objects
            if item.class_name in landmark_classes
        ]
        picked.sort(key=lambda item: (-float(item.confidence), item.candidate_id))
        return picked[:REFINE_CROP_CAP]

    def _predict_around_landmarks(
        self,
        frames: dict[str, CctvFrame],
        objects: list[ObjectCandidate],
        landmark_classes: frozenset[str],
        landmarks: Optional[list[ObjectCandidate]] = None,
        pad_frac: float = CROP_PAD_FRAC,
        imgsz: int = REFINE_IMGSZ,
        confidence: float = REFINE_CONF,
    ) -> list[
        tuple[
            str,
            str,
            float,
            tuple[float, float, float, float],
            tuple[int, int],
        ]
    ]:
        """Map low-threshold crop detections back into their full camera frames."""
        mapped_detections = []
        crops = (
            landmarks
            if landmarks is not None
            else self._landmarks_for_refine(objects, landmark_classes)
        )
        for landmark in crops:
            frame = frames.get(landmark.camera_id)
            rgb = None if frame is None else getattr(frame, "rgb", None)
            if rgb is None:
                continue
            array = np.asarray(rgb)
            height, width = array.shape[:2]
            x1, y1, x2, y2 = _pad_box(
                landmark.bbox_xyxy, (width, height), pad_frac=pad_frac
            )
            ix1, iy1 = int(x1), int(y1)
            ix2 = int(max(x1 + 1, x2))
            iy2 = int(max(y1 + 1, y2))
            crop = array[iy1:iy2, ix1:ix2]
            if crop.size == 0:
                continue
            image_size = (int(frame.width), int(frame.height))
            try:
                detections = self._predict_frame(
                    crop,
                    confidence=confidence,
                    imgsz=min(imgsz, self.contract.imgsz),
                )
            except Exception as exc:
                if not _is_cuda_oom(exc):
                    raise
                log.warning(
                    "[cctv-yolo] refine crop OOM camera=%s landmark=%s; skipping",
                    landmark.camera_id,
                    landmark.candidate_id,
                )
                _release_cuda()
                continue
            _release_cuda()
            for class_name, confidence, bbox in sorted(
                detections,
                key=lambda item: (item[2], item[0], -item[1]),
            ):
                mapped = (
                    bbox[0] + ix1,
                    bbox[1] + iy1,
                    bbox[2] + ix1,
                    bbox[3] + iy1,
                )
                mapped_detections.append(
                    (landmark.camera_id, class_name, confidence, mapped, image_size)
                )
        return mapped_detections

    def refine_people_around_landmarks(
        self,
        frames: dict[str, CctvFrame],
        objects: list[ObjectCandidate],
        people: list[PersonCandidate],
        landmark_classes: frozenset[str],
    ) -> list[PersonCandidate]:
        """Recover missed people from explicit support crops without duplicates."""
        if not landmark_classes:
            return []
        extra: list[PersonCandidate] = []
        counts: dict[tuple[str, str], int] = defaultdict(int)
        occupied: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for person in people:
            counts[(person.camera_id, person.pose_class)] += 1
            occupied[person.camera_id].append(person.bbox_xyxy)

        for camera_id, pose_class, confidence, mapped, image_size in (
            self._predict_around_landmarks(frames, objects, landmark_classes)
        ):
            if not is_person_pose(pose_class):
                continue
            if any(
                _box_iou(mapped, existing) >= REFINE_DEDUP_IOU
                for existing in occupied[camera_id]
            ):
                continue
            key = (camera_id, pose_class)
            idx = counts[key]
            counts[key] += 1
            candidate = PersonCandidate(
                candidate_id=person_candidate_id(
                    camera_id,
                    pose_class,
                    idx,
                ),
                bbox_xyxy=mapped,
                pose_class=pose_class,
                confidence=confidence,
                camera_id=camera_id,
                image_size=image_size,
            )
            extra.append(candidate)
            occupied[camera_id].append(mapped)
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
) -> tuple[list[ObjectCandidate], list[PersonCandidate], float, float]:
    global _detector
    if _detector is None:
        _detector = CctvYoloDetector()
    load_started = time.perf_counter()
    _detector.load()
    load_ms = (time.perf_counter() - load_started) * 1000.0
    inference_started = time.perf_counter()
    objects, people = _detector.detect(frames)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    return objects, people, load_ms, inference_ms


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
    if parsed is None:
        return round_
    global _detector
    if _detector is None:
        return round_
    try:
        return _refine_detection_round(parsed, round_)
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        log.warning(
            "[cctv-yolo] refine OOM; keeping full-frame detections error=%s",
            exc,
        )
        _release_cuda()
        return round_


def _refine_detection_round(parsed, round_: CctvDetectionRound) -> CctvDetectionRound:
    if parsed.target_type == "person":
        names = frozenset({parsed.landmark}) if parsed.landmark else frozenset()
        extra_people = _detector.refine_people_around_landmarks(
            round_.frames,
            round_.objects,
            round_.people,
            names,
        )
        log.info(
            "[cctv-yolo] person-refine landmark=%s added=%d ids=%s",
            parsed.landmark,
            len(extra_people),
            ",".join(item.candidate_id for item in extra_people) or "-",
        )
        if extra_people:
            note_decision(
                "refine_added", path="sar",
                candidates=[item.candidate_id for item in extra_people],
            )
        if not extra_people:
            return round_
        return CctvDetectionRound(
            objects=list(round_.objects),
            people=list(round_.people) + extra_people,
            sar=list(round_.sar),
            frames=round_.frames,
            camera_ids=list(round_.camera_ids),
            metrics=round_.metrics,
        )
    names = set()
    if parsed.landmark:
        names.add(parsed.landmark)
    elif parsed.scene_hint in PARKING_SCENES:
        names.update(PARKED_VEHICLE_LANDMARKS)
    elif parsed.scene_hint == "park":
        names.update({"picnic_table", "bench"})
    if not names:
        return round_
    extra = _detector.refine_around_landmarks(
        round_.frames,
        round_.objects,
        parsed.target_type,
        frozenset(names),
    )
    log.info(
        "[cctv-yolo] object-refine target=%s landmark=%s added=%d ids=%s",
        parsed.target_type,
        parsed.landmark,
        len(extra),
        ",".join(item.candidate_id for item in extra) or "-",
    )
    if extra:
        note_decision(
            "refine_added", path="object",
            candidates=[item.candidate_id for item in extra],
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
