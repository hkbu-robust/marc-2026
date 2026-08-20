"""YOLO detector for auto-approach centering (pretrained now, fine-tuned later)."""

from __future__ import annotations

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


# Names skipped when picking a box (bystander / robot-adjacent false positives).
# Class *filter at predict time* is only applied once the weights actually contain
# the MARC target name (a fine-tune). Pretrained COCO uses classes=None.
SKIP_NAMES = frozenset({"person"})

_PKG_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = str((_PKG_DIR.parent / "weights" / "yolo11x.pt").resolve())
DEFAULT_CONF = 0.15
DEFAULT_IMGSZ = 640
DEFAULT_DEVICE = "0"
DEFAULT_HALF = False
DEFAULT_CAM = "base_left"


@dataclass
class YoloConfig:
    weights: str = DEFAULT_WEIGHTS
    conf: float = DEFAULT_CONF
    imgsz: int = DEFAULT_IMGSZ
    device: str = DEFAULT_DEVICE
    half: bool = DEFAULT_HALF
    camera: str = DEFAULT_CAM

    @classmethod
    def from_env(cls) -> "YoloConfig":
        return cls(
            weights=_env_str("MARC_YOLO_WEIGHTS", DEFAULT_WEIGHTS),
            conf=_env_float("MARC_YOLO_CONF", DEFAULT_CONF),
            imgsz=_env_int("MARC_YOLO_IMGSZ", DEFAULT_IMGSZ),
            device=_env_str("MARC_YOLO_DEVICE", DEFAULT_DEVICE),
            half=_env_bool("MARC_YOLO_HALF", DEFAULT_HALF),
            camera=_env_str("MARC_YOLO_CAM", DEFAULT_CAM),
        )


def _resolve_device(requested: str) -> str:
    req = (requested or "0").strip()
    if req.lower() in ("cpu", "cpu:0"):
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return req
        print("[yolo] CUDA not available — falling back to CPU (YOLO11x will be slow)")
    except Exception:
        print("[yolo] torch not importable — falling back to CPU")
    return "cpu"


def _ensure_weights(path: Path) -> str:
    """Return a weights path Ultralytics can open, downloading yolo11x if needed."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(path.parent))
    try:
        from ultralytics import settings

        settings.update(weights_dir=str(path.parent))
    except Exception:
        pass
    if path.exists():
        return str(path)
    cwd = Path.cwd()
    try:
        os.chdir(path.parent)
        from ultralytics import YOLO

        print(f"[yolo] {path.name} missing — downloading into {path.parent} ...")
        YOLO(path.name if path.suffix == ".pt" else "yolo11x.pt")
    finally:
        os.chdir(cwd)
    downloaded = path if path.exists() else (path.parent / "yolo11x.pt")
    if not downloaded.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {path} and download did not produce yolo11x.pt"
        )
    if downloaded.resolve() != path.resolve() and not path.exists():
        downloaded.replace(path)
    return str(path if path.exists() else downloaded)


class YoloTracker:
    """Run YOLO on RGB frames; pick one box and a signed image-center error."""

    def __init__(self, cfg: Optional[YoloConfig] = None):
        self.cfg = cfg or YoloConfig.from_env()
        self.model = None
        self.names: dict = {}
        self.device = "cpu"
        self._target: Optional[str] = None
        self._class_ids: Optional[list] = None
        self._last_fail = ""

    def set_target(self, name: Optional[str]):
        self._target = None if not name else str(name).strip()
        self._refresh_classes()

    def _refresh_classes(self):
        """Restrict predict() only when the checkpoint has the MARC class name."""
        if not self.names:
            self._class_ids = None
            return
        inv = {str(n): int(i) for i, n in self.names.items()}
        if self._target and self._target in inv:
            self._class_ids = [inv[self._target]]
            return
        self._class_ids = None

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
            self._refresh_classes()
            if self._class_ids is not None:
                n_cls = ",".join(self.names[i] for i in self._class_ids)
            else:
                n_cls = "all (no class filter)"
            print(
                f"[yolo] ready  classes={n_cls}  "
                f"target={self._target or '-'}  "
                f"conf={self.cfg.conf}"
            )
            return True
        except Exception as e:
            self.model = None
            self._last_fail = f"load failed: {e}"
            print(f"[yolo] {self._last_fail}")
            return False

    def detect(self, rgb: np.ndarray, conf: Optional[float] = None) -> Optional[dict]:
        """Return the best box vs image center (always a dict if infer ran).

        Pass a PIL RGB image so Ultralytics does not treat the array as BGR.
        Do not pass ``half=`` (deprecated; also spams every tick).
        """
        if self.model is None or rgb is None or rgb.size == 0:
            return None
        h, w = rgb.shape[:2]
        if h < 2 or w < 2:
            return None
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
        )
        if self._class_ids is not None:
            kw["classes"] = self._class_ids
        try:
            results = self.model.predict(src, **kw)
        except Exception as e:
            self._last_fail = f"predict: {e}"
            return None
        empty = {
            "error": None,
            "n_obj": 0,
            "n_raw": 0,
            "bbox": None,
            "conf": None,
            "cls": None,
            "name": None,
            "raw": [],
            "rgb_shape": (int(h), int(w)),
        }
        if not results:
            return empty
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return empty

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy()
        n_raw = int(len(xyxy))
        raw = []
        for i in range(n_raw):
            cls_id = int(clss[i])
            raw.append((self.names.get(cls_id, str(cls_id)), float(confs[i])))
        empty["n_raw"] = n_raw
        empty["raw"] = raw

        candidates = []
        skipped = []
        for i in range(n_raw):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            conf = float(confs[i])
            cls_id = int(clss[i])
            name = self.names.get(cls_id, str(cls_id))
            rec = (x1, y1, x2, y2, conf, cls_id, name)
            if name in SKIP_NAMES:
                skipped.append(rec)
            else:
                candidates.append(rec)
        if not candidates:
            candidates = skipped
        if not candidates:
            return empty

        best = None
        best_score = -1.0
        for x1, y1, x2, y2, conf, cls_id, name in candidates:
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area = (bw * bh) / float(w * h)
            vert = float(np.clip(cy / float(h), 0.0, 1.0))
            name_boost = 1.5 if self._target and name == self._target else 1.0
            score = conf * (0.35 + 0.65 * vert) * (0.4 + area) * name_boost
            if score > best_score:
                best_score = score
                err = (cx - 0.5 * (w - 1)) / (0.5 * (w - 1))
                best = {
                    "error": float(np.clip(err, -1.0, 1.0)),
                    "n_obj": 1,
                    "n_raw": n_raw,
                    "raw": raw,
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls": cls_id,
                    "name": name,
                    "cx": cx,
                    "cy": cy,
                    "area": float(area),
                    "rgb_shape": (int(h), int(w)),
                }
        return best if best is not None else empty
