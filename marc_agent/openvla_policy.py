"""OpenVLA-oriented grasp policy: protocol, geometric stub, optional weights.

The graded ``agent`` image is torch-free. Real OpenVLA inference is a lazy optional
load (manip-probe / later ``models/runtime/openvla`` weights). Missing weights or a
failed import returns ``None`` so the caller falls back to IK.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

log = logging.getLogger("participant_app")

ACTION_DIM = 7
DEFAULT_STEP_M = 0.02
DEFAULT_CLOSE_M = 0.04
DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "runtime", "openvla",
)


@dataclass
class PolicyObservation:
    """One policy tick: third-person RGB, arm proprio, optional EE XYZ in arm_base."""

    rgb: Optional[np.ndarray] = None
    proprio: Optional[np.ndarray] = None  # q(6) + grip
    ee_xyz: Optional[np.ndarray] = None


class GraspPolicy(Protocol):
    def act(self, obs: PolicyObservation) -> np.ndarray:
        """Return OpenVLA-style action ``(dx, dy, dz, droll, dpitch, dyaw, grip)``."""


class GeometricGraspStub:
    """No-weights stand-in: EE deltas toward an arm_base target, then close.

    Used with ``MARC_PICK_POLICY=openvla_geom`` to exercise the action executor
    without a model. Not the scoring default.
    """

    def __init__(self, target_xyz, step_m: float = DEFAULT_STEP_M,
                 close_m: float = DEFAULT_CLOSE_M):
        self.target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        self.step_m = float(step_m)
        self.close_m = float(close_m)

    def act(self, obs: PolicyObservation) -> np.ndarray:
        ee = np.zeros(3, dtype=np.float64)
        if obs is not None and obs.ee_xyz is not None:
            ee = np.asarray(obs.ee_xyz, dtype=np.float64).reshape(3)
        delta = self.target - ee
        dist = float(np.linalg.norm(delta))
        if dist > self.step_m and dist > 1e-9:
            delta = delta * (self.step_m / dist)
        grip = 1.0 if dist <= self.close_m else 0.0
        return np.array(
            [delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip],
            dtype=np.float32,
        )


def resolve_weights_path(path: Optional[str] = None) -> Optional[str]:
    raw = path or os.environ.get("MARC_OPENVLA_WEIGHTS") or DEFAULT_WEIGHTS_DIR
    raw = os.path.expanduser(str(raw))
    if not raw:
        return None
    if os.path.isfile(raw):
        return raw
    if os.path.isdir(raw):
        for name in ("model.pt", "model.safetensors", "openvla.pt", "pytorch_model.bin"):
            cand = os.path.join(raw, name)
            if os.path.isfile(cand):
                return cand
        if any(os.path.isfile(os.path.join(raw, n)) for n in os.listdir(raw) or []):
            return raw
    return None


def load_openvla(path: Optional[str] = None) -> Optional[GraspPolicy]:
    """Load an OpenVLA policy if weights and torch are available; else ``None``."""
    weights = resolve_weights_path(path)
    if weights is None:
        log.info("[openvla] no weights at %s -- IK fallback",
                 path or os.environ.get("MARC_OPENVLA_WEIGHTS") or DEFAULT_WEIGHTS_DIR)
        return None
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("[openvla] weights at %s but torch is unavailable -- IK fallback", weights)
        return None
    log.warning(
        "[openvla] weights present at %s but inference is not wired yet -- IK fallback",
        weights,
    )
    return None
