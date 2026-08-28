"""Exact-target Stage 2 YOLO detector with a legacy single-box wrapper."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = str(
    (_PKG_DIR.parents[1] / "models/runtime/stage2-nearfield-detector/model.pt").resolve()
)
DEFAULT_CONF = 0.15
DEFAULT_IMGSZ = 1920
DEFAULT_DEVICE = "0"
DEFAULT_HALF = False
DEFAULT_CAM = "base_left"
MAX_CANDIDATES = 20


@dataclass
class YoloConfig:
    weights: str = DEFAULT_WEIGHTS
    conf: float = DEFAULT_CONF
    imgsz: int = DEFAULT_IMGSZ
    device: str = DEFAULT_DEVICE
    half: bool = DEFAULT_HALF
    camera: str = DEFAULT_CAM
    nms_iou: float = 0.7
    end2end: bool = False
    candidate_limit: int = MAX_CANDIDATES
    quantize: Optional[str] = None
    ordered_classes: tuple[str, ...] = ()
    status: str = "unknown"
    known_limitations: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "YoloConfig":
        weights = _env_str("MARC_YOLO_WEIGHTS", DEFAULT_WEIGHTS)
        contract_path = Path(weights).with_name("runtime_contract.json")
        contract = json.loads(contract_path.read_text()) if contract_path.is_file() else {}
        inference = contract.get("inference", {})
        quantize = inference.get("quantize")
        quantize = "fp16" if _env_bool("MARC_YOLO_HALF", quantize == "fp16") else None
        return cls(
            weights=weights,
            conf=_env_float("MARC_YOLO_CONF", inference.get("confidence", DEFAULT_CONF)),
            imgsz=_env_int("MARC_YOLO_IMGSZ", inference.get("imgsz", DEFAULT_IMGSZ)),
            device=_env_str("MARC_YOLO_DEVICE", DEFAULT_DEVICE),
            half=_env_bool("MARC_YOLO_HALF", DEFAULT_HALF),
            camera=_env_str("MARC_YOLO_CAM", DEFAULT_CAM),
            nms_iou=float(inference.get("nms_iou") or 0.7),
            end2end=bool(inference.get("end2end", False)),
            candidate_limit=max(
                1, min(MAX_CANDIDATES, int(inference.get("candidate_limit", MAX_CANDIDATES)))
            ),
            quantize=quantize,
            ordered_classes=tuple(str(name) for name in contract.get("ordered_classes", ())),
            status=str(contract.get("status", "unknown")),
            known_limitations=tuple(str(item) for item in contract.get("known_limitations", ())),
        )


def _resolve_device(requested: str) -> str:
    req = (requested or "0").strip()
    if req.lower() in ("cpu", "cpu:0"):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return req
        print("[yolo] CUDA not available — falling back to CPU (YOLO will be slow)")
    except Exception:
        print("[yolo] torch not importable — falling back to CPU")
    return "cpu"


def _ensure_weights(path: Path) -> str:
    """Return an explicit local checkpoint path; runtime downloads are forbidden."""
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"local YOLO weights not found: {path}")
    return str(path)


class YoloTracker:
    """Run YOLO on RGB frames; pick one box and a signed image-center error."""

    def __init__(self, cfg: Optional[YoloConfig] = None):
        self.cfg = cfg or YoloConfig.from_env()
        self.model = None
        self.names: dict = {}
        self.device = "cpu"
        self._target: Optional[str] = None
        self._last_fail = ""

    def set_target(self, name: Optional[str]):
        self._target = None if not name else str(name).strip()

    def load(self) -> bool:
        if self.model is not None:
            return True
        try:
            from ultralytics import YOLO
        except ImportError as e:
            self._last_fail = (
                f"ultralytics not installed ({e}). Rebuild agent-manip-probe "
                f"(Dockerfile target manip-probe)."
            )
            print(f"[yolo] {self._last_fail}")
            return False
        try:
            weights = _ensure_weights(Path(self.cfg.weights))
        except Exception as e:
            self._last_fail = f"weights: {e}"
            print(f"[yolo] {self._last_fail}")
            return False
        self.device = _resolve_device(self.cfg.device)
        print(f"[yolo] loading {weights}  device={self.device}  imgsz={self.cfg.imgsz}")
        try:
            self.model = YOLO(weights)
            raw_names = getattr(self.model, "names", None) or {}
            self.names = {int(k): str(v) for k, v in dict(raw_names).items()}
            loaded_order = tuple(self.names[index] for index in sorted(self.names))
            if self.cfg.ordered_classes and loaded_order != self.cfg.ordered_classes:
                raise ValueError(
                    f"checkpoint class order {loaded_order} does not match contract "
                    f"{self.cfg.ordered_classes}"
                )
            print(
                f"[yolo] ready status={self.cfg.status} classes={len(self.names)} "
                f"conf={self.cfg.conf} max={self.cfg.candidate_limit}"
            )
            if self.cfg.status == "provisional":
                print("[yolo] PROVISIONAL_PIPELINE_ONLY: ordinary umbrella is unsupported")
            return True
        except Exception as e:
            self.model = None
            self._last_fail = f"load failed: {e}"
            print(f"[yolo] {self._last_fail}")
            return False

    def unload(self) -> None:
        model = self.model
        self.model = None
        self.names = {}
        if model is not None:
            del model

    def detect_candidates(
        self, rgb: np.ndarray, target_type: str, conf: Optional[float] = None
    ) -> dict:
        """Return up to 20 confidence-sorted candidates for one exact target."""
        target = str(target_type or "").strip()
        class_ids = {
            name: class_id for class_id, name in (
                self.names.items() if self.names else enumerate(self.cfg.ordered_classes)
            )
        }
        if target not in class_ids:
            return {"status": "unsupported_target", "target_type": target, "rgb_shape": None, "candidates": []}
        if self.model is None:
            return {"status": "detector_unavailable", "target_type": target, "rgb_shape": None, "candidates": []}
        if rgb is None or not isinstance(rgb, np.ndarray) or rgb.size == 0:
            return {"status": "invalid_image", "target_type": target, "rgb_shape": None, "candidates": []}
        h, w = rgb.shape[:2]
        if h < 2 or w < 2:
            return {"status": "invalid_image", "target_type": target, "rgb_shape": (int(h), int(w)), "candidates": []}
        src = np.ascontiguousarray(rgb)
        try:
            from PIL import Image

            src = Image.fromarray(src, mode="RGB")
        except Exception:
            pass
        kw = dict(
            verbose=False,
            conf=float(self.cfg.conf if conf is None else conf),
            imgsz=int(self.cfg.imgsz),
            device=self.device,
            classes=[class_ids[target]],
            max_det=min(MAX_CANDIDATES, int(self.cfg.candidate_limit)),
            end2end=bool(self.cfg.end2end),
            iou=float(self.cfg.nms_iou),
            quantize=self.cfg.quantize if self.device != "cpu" else None,
        )
        try:
            results = self.model.predict(src, **kw)
        except Exception as e:
            self._last_fail = f"predict: {e}"
            return {"status": "inference_error", "target_type": target, "rgb_shape": (int(h), int(w)), "candidates": []}
        if not results:
            return {"status": "not_detected", "target_type": target, "rgb_shape": (int(h), int(w)), "candidates": []}
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"status": "not_detected", "target_type": target, "rgb_shape": (int(h), int(w)), "candidates": []}

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy()
        candidates = []
        for box, confidence, class_id in zip(xyxy, confs, clss):
            cls_id = int(class_id)
            if self.names.get(cls_id) != target:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            x1, x2 = sorted((float(np.clip(x1, 0, w)), float(np.clip(x2, 0, w))))
            y1, y2 = sorted((float(np.clip(y1, 0, h)), float(np.clip(y2, 0, h))))
            candidates.append({
                "xyxy": (x1, y1, x2, y2),
                "confidence": float(confidence),
                "class_id": cls_id,
                "class_name": target,
            })
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        candidates = candidates[:min(MAX_CANDIDATES, int(self.cfg.candidate_limit))]
        for rank, candidate in enumerate(candidates, 1):
            candidate["rank"] = rank
        return {
            "status": "ok" if candidates else "not_detected",
            "target_type": target,
            "rgb_shape": (int(h), int(w)),
            "candidates": candidates,
        }

    def detect(self, rgb: np.ndarray, conf: Optional[float] = None) -> dict:
        """Return the highest-confidence exact-target box for legacy callers."""
        result = self.detect_candidates(rgb, self._target or "", conf=conf)
        shape = result.get("rgb_shape")
        empty = {
            "status": result["status"], "error": None, "n_obj": 0,
            "n_raw": len(result["candidates"]), "bbox": None, "conf": None,
            "cls": None, "name": None, "raw": [], "rgb_shape": shape,
        }
        if not result["candidates"] or shape is None:
            return empty
        candidate = result["candidates"][0]
        x1, y1, x2, y2 = candidate["xyxy"]
        h, w = shape
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        area = max(1.0, x2 - x1) * max(1.0, y2 - y1) / float(w * h)
        return {
            **empty,
            "status": "ok",
            "error": float(np.clip((cx - 0.5 * (w - 1)) / (0.5 * (w - 1)), -1.0, 1.0)),
            "n_obj": 1,
            "bbox": candidate["xyxy"],
            "conf": candidate["confidence"],
            "cls": candidate["class_id"],
            "name": candidate["class_name"],
            "raw": [(item["class_name"], item["confidence"]) for item in result["candidates"]],
            "cx": cx,
            "cy": cy,
            "area": float(area),
        }
