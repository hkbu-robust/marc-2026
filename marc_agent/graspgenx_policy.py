"""GraspGen-X pick seam: object cloud -> 6-DoF arm_base grasp, then IK.

A missing package or checkpoint logs once and returns ``None`` so the caller
falls back to analytic IK. Do not download weights at runtime.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("participant_app")

DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "runtime",
    "graspgenx",
)
_WEIGHT_SUFFIXES = (".pth", ".pt", ".ckpt", ".safetensors")
GenerateFn = Callable[[np.ndarray], object]
_generator: Optional[GenerateFn] = None


@dataclass(frozen=True)
class Grasp6DoF:
    """One grasp in arm_base: translation used by IK, optional 4x4 pose."""

    xyz: tuple[float, float, float]
    matrix: Optional[np.ndarray] = None
    score: float = 0.0


def checkpoint_dir(path: Optional[str] = None) -> str:
    if path:
        return path
    return os.environ.get("MARC_GRASPGENX_WEIGHTS") or DEFAULT_WEIGHTS_DIR


def gripper_name() -> str:
    return os.environ.get("MARC_GRASPGENX_GRIPPER") or "robotiq_2f_140"


def gripper_assets_dir(root: Optional[str] = None) -> Optional[str]:
    """Directory that contains ``x_grippers/``, or None if assets are missing."""
    base = root or os.environ.get("GRASPGENX_GRIPPER_CFG_DIR") or os.path.join(
        DEFAULT_WEIGHTS_DIR, "grippers"
    )
    if not os.path.isdir(base):
        return None
    for candidate in (
        os.path.join(base, "gripper_descriptions", "assets"),
        os.path.join(base, "assets"),
        base,
    ):
        if os.path.isdir(os.path.join(candidate, "x_grippers")):
            return candidate
    return None


def has_checkpoint(path: Optional[str] = None) -> bool:
    """True when both generator and discriminator snapshots are present."""
    directory = checkpoint_dir(path)
    if not os.path.isdir(directory):
        return False
    gen = os.path.join(directory, "gen")
    dis = os.path.join(directory, "dis")
    return _dir_has_weights(gen) and _dir_has_weights(dis)


def _dir_has_weights(directory: str) -> bool:
    try:
        names = os.listdir(directory)
    except OSError:
        return False
    return any(
        name.endswith(_WEIGHT_SUFFIXES)
        and os.path.isfile(os.path.join(directory, name))
        for name in names
    )


def crop_depth_cloud(
    depth,
    bbox_xyxy,
    k,
    rgb_shape=None,
    *,
    stride: int = 2,
) -> np.ndarray:
    """Unproject depth pixels inside ``bbox_xyxy`` to camera-optical metres."""
    if depth is None or bbox_xyxy is None or k is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if rgb_shape is not None:
        sy = h / max(1.0, float(rgb_shape[0]))
        sx = w / max(1.0, float(rgb_shape[1]))
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
    ix1 = max(0, int(math.floor(x1)))
    iy1 = max(0, int(math.floor(y1)))
    ix2 = min(w, int(math.ceil(x2)))
    iy2 = min(h, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return np.zeros((0, 3), dtype=np.float32)
    fx, fy, cx, cy = _pinhole(k)
    ys, xs = np.mgrid[iy1:iy2:stride, ix1:ix2:stride]
    z = arr[ys, xs].astype(np.float32)
    valid = np.isfinite(z) & (z > 0.05) & (z < 5.0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = z[valid]
    x = (xs - cx) * z / max(fx, 1e-6)
    y = (ys - cy) * z / max(fy, 1e-6)
    return np.stack([x, y, z], axis=1)


def _pinhole(k) -> tuple[float, float, float, float]:
    km = np.asarray(k, dtype=np.float64).reshape(-1)
    if km.size >= 9:
        return float(km[0]), float(km[4]), float(km[2]), float(km[5])
    if km.size >= 4:
        return float(km[0]), float(km[1]), float(km[2]), float(km[3])
    return 500.0, 500.0, 320.0, 240.0


def points_from_target(target, client=None) -> np.ndarray:
    """Object cloud in arm_base, falling back to the localized contact point."""
    depth = getattr(target, "depth", None)
    bbox = getattr(target, "bbox_xyxy", None)
    k = getattr(target, "k", None)
    rgb_shape = getattr(target, "rgb_shape", None)
    if depth is None and client is not None and bbox is not None:
        try:
            depth = _client_depth(client)
        except Exception:
            depth = None
    optical = crop_depth_cloud(depth, bbox, k, rgb_shape)
    rotation = getattr(target, "R", None)
    translation = getattr(target, "t", None)
    if optical.size and rotation is not None:
        try:
            from nearfield import car_to_arm_base, optical_to_car_base

            converted = []
            for ox, oy, oz in optical:
                fwd_c, lat_c, up_c = optical_to_car_base(
                    float(ox), float(oy), float(oz), R=rotation, t=translation
                )
                converted.append(car_to_arm_base(fwd_c, lat_c, up_c))
            if converted:
                return np.asarray(converted, dtype=np.float32)
        except Exception:
            pass
    fwd = float(getattr(target, "fwd", 0.0) or 0.0)
    lat = float(getattr(target, "lat", 0.0) or 0.0)
    z = float(getattr(target, "z", 0.0) or 0.0)
    return np.array([[fwd, lat, z]], dtype=np.float32)


def _client_depth(client):
    getter = getattr(client, "get_robot_depth", None)
    if not callable(getter):
        return None
    return getter("base")


def as_grasp(pose) -> Optional[Grasp6DoF]:
    if pose is None:
        return None
    if isinstance(pose, Grasp6DoF):
        return pose
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape == (4, 4):
        xyz = arr[:3, 3]
        return Grasp6DoF(
            xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            matrix=arr,
        )
    if arr.size >= 3:
        xyz = arr.reshape(-1)[:3]
        return Grasp6DoF(xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return None


def grasp_yaw_from_matrix(matrix) -> Optional[float]:
    """Tool-Z yaw in arm_base from a 4x4 grasp, or None if the XY axis is degenerate.

    Projects the gripper closing axis (column 0) onto the arm_base floor. Joint-1
    at zero points +fwd, so this yaw is the delta ``arm_pick`` applies as ``grasp_yaw``.
    """
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        return None
    axis = arr[:2, 0]
    if float(np.linalg.norm(axis)) < 1e-6:
        axis = arr[:2, 1]
    if float(np.linalg.norm(axis)) < 1e-6:
        return None
    return float(math.atan2(axis[1], axis[0]))


def best_grasp(
    points: np.ndarray,
    *,
    generate_fn: Optional[GenerateFn] = None,
) -> Optional[Grasp6DoF]:
    """Return the best 6-DoF grasp in the same frame as ``points``, or None."""
    generator = generate_fn if generate_fn is not None else load_graspgenx()
    if generator is None:
        return None
    try:
        pose = generator(np.asarray(points, dtype=np.float32))
    except Exception as exc:
        log.warning("[graspgenx] generate failed (%s) -- IK fallback", exc)
        return None
    return as_grasp(pose)


def load_graspgenx(path: Optional[str] = None) -> Optional[GenerateFn]:
    """Load a local GraspGen-X sampler. Returns None without downloading."""
    global _generator
    if _generator is not None:
        return _generator
    directory = checkpoint_dir(path)
    if not has_checkpoint(directory):
        log.warning("[graspgenx] no weights at %s -- IK fallback", directory)
        return None
    os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", directory)
    try:
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.samplers import run_planner_on_object
        from graspgenx.utils.checkpoint_io import load_model_cfg
    except Exception as exc:
        log.warning("[graspgenx] import failed (%s) -- IK fallback", exc)
        return None
    try:
        cfg = load_model_cfg(
            os.path.join(directory, "gen"),
            os.path.join(directory, "dis"),
            None,
            None,
        )
        gripper = gripper_name()
        assets = gripper_assets_dir()
        if not assets:
            raise FileNotFoundError("GraspGen-X gripper assets are missing")
        sampler = GraspGenXSampler(cfg, gripper, assets_dir=assets)
    except Exception as exc:
        log.warning("[graspgenx] load failed (%s) -- IK fallback", exc)
        return None

    def generate(points: np.ndarray):
        grasps, conf, _tags, _obb = run_planner_on_object(
            np.asarray(points, dtype=np.float32),
            sampler,
            planner="diffusion",
            num_grasps=32,
            grasp_threshold=-1.0,
            topk_num_grasps=1,
        )
        if grasps is None or len(grasps) == 0:
            return None
        idx = int(np.argmax(conf)) if conf is not None and len(conf) else 0
        return grasps[idx]

    _generator = generate
    return generate


def unload_graspgenx() -> None:
    """Drop cached GraspGen-X tensors. Callers must also drop their generator."""
    global _generator
    _generator = None
    log.info("[graspgenx] unload")
