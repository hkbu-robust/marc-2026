"""Calibrated ground surface per CCTV camera, used only where the camera is known (#112).

Each camera's published ``ground_height`` is one flat plane, but in chungmu the
ground the scorer measures against sits up to 0.6 m below it (rig_3_c), which
pulls every projected point toward the camera. The organisers ask participants to
"account for the relevant ground height when back-projecting"
(``api-reference.md``, mission-area ground-plane assumption).

The surface is a set of ground points per camera: the world positions of small
objects lying in generator scenes (``scripts/build_ground_model.py``; training-side
GT, never the evaluation sets). Its height at (x, y) is a k-nearest-neighbour
lower envelope: points well above the local median (objects resting on tables or
benches) are dropped, the rest weighted by inverse distance.

This is per-site calibration. Each camera carries a fingerprint of the pose it
was calibrated for; a live camera whose pose or ground height differs gets no
surface, and projection falls back to the flat plane.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("marc_agent.ground_model")

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "ground_model.npz"
NEIGHBOURS = 12
DISTANCE_SOFTENING_M = 0.3
ABOVE_GROUND_M = 0.25  # a neighbour this far above the local median is on furniture
MAX_EXTRAPOLATION_M = 2.0  # farther than this from every ground point: no estimate
FINGERPRINT_TRANSLATION_M = 0.01
FINGERPRINT_ROTATION_DEG = 0.1
FINGERPRINT_GROUND_M = 0.01


@dataclass(frozen=True)
class GroundSurface:
    """Ground points of one camera's view; ``z(x, y)`` is the ground height there."""

    points: np.ndarray  # (N, 3) world xyz

    def z(self, x: float, y: float) -> Optional[float]:
        """Ground height at (x, y); None outside the calibrated area."""
        dist = np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)
        if float(dist.min()) > MAX_EXTRAPOLATION_M:
            return None
        k = min(NEIGHBOURS, len(dist))
        nearest = np.argpartition(dist, k - 1)[:k]
        heights = self.points[nearest, 2]
        weights = 1.0 / (dist[nearest] + DISTANCE_SOFTENING_M)
        keep = heights < np.median(heights) + ABOVE_GROUND_M
        return float(np.sum(heights[keep] * weights[keep]) / np.sum(weights[keep]))


@dataclass(frozen=True)
class CameraFingerprint:
    """The pose and published ground height a surface was calibrated for."""

    rotation: np.ndarray  # (3, 3) world-from-camera
    translation: np.ndarray  # (3,)
    ground_height: float

    def mismatch(self, geometry) -> Optional[str]:
        """None when ``geometry`` (a ``CameraGeometry``) is this camera, else why not."""
        translation = np.asarray(geometry.translation, dtype=float)
        if np.linalg.norm(translation - self.translation) > FINGERPRINT_TRANSLATION_M:
            return "translation"
        rotation = np.asarray(geometry.rotation, dtype=float)
        cos_angle = (np.trace(self.rotation.T @ rotation) - 1.0) / 2.0
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
        if angle > FINGERPRINT_ROTATION_DEG:
            return "rotation"
        if abs(float(geometry.ground_height) - self.ground_height) > FINGERPRINT_GROUND_M:
            return "ground_height"
        return None


class GroundModel:
    """Calibrated surfaces by camera ID, each gated by its pose fingerprint."""

    def __init__(self, cameras: dict[str, tuple[CameraFingerprint, GroundSurface]], source: dict):
        self.cameras = cameras
        self.source = source

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "GroundModel":
        with np.load(path, allow_pickle=False) as data:
            source = json.loads(str(data["source"]))
            cameras = {
                camera: (
                    CameraFingerprint(
                        rotation=data[f"{camera}/rotation"],
                        translation=data[f"{camera}/translation"],
                        ground_height=float(data[f"{camera}/ground_height"]),
                    ),
                    GroundSurface(points=data[f"{camera}/points"].astype(float)),
                )
                for camera in source["cameras"]
            }
        return cls(cameras, source)

    def for_camera(self, camera_id: str, geometry) -> tuple[Optional[GroundSurface], str]:
        """The camera's surface and ``"applied"``, or None and the reason it is not."""
        entry = self.cameras.get(camera_id)
        if entry is None:
            return None, "uncalibrated_camera"
        fingerprint, surface = entry
        mismatch = fingerprint.mismatch(geometry)
        if mismatch is not None:
            return None, f"fingerprint_mismatch:{mismatch}"
        return surface, "applied"


_DEFAULT: Optional[GroundModel] = None


def default_ground_model() -> Optional[GroundModel]:
    """The shipped model, loaded once; None if the artefact is missing or unreadable
    (projection then keeps the flat plane rather than failing at start-up)."""
    global _DEFAULT
    if _DEFAULT is None and DEFAULT_PATH.exists():
        try:
            _DEFAULT = GroundModel.load(DEFAULT_PATH)
        except Exception as exc:  # e.g. a truncated file or a git-LFS pointer
            log.warning("ground model %s unreadable (%s); using the flat plane", DEFAULT_PATH, exc)
    return _DEFAULT
