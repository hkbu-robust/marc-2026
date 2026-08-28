"""Local lidar corridor: slow or stop in the chassis envelope, never paint the map.

``get_lidar()`` is already robot/base frame (REP-103, +X forward). Occupancy D* stays
the only global layer. This module classifies live hits: pad/ground/self are ignored,
side objects do not stop, and only a body-height box ahead of the bumper forces reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Drop pad floor, curb lip, and canopy. Body-height hits (people, cars, benches) remain.
Z_MIN = 0.20
Z_MAX = 1.80
# Chassis plus the ~0.30 m spawn-pad ring. Hits inside this radius are self / pad.
BODY_R = 0.45
STOP_X_MIN = 0.40
STOP_X_MAX = 0.80
CORRIDOR_Y = 0.30
CLEAR_FAR = 2.50
NUDGE_Y = 0.80


@dataclass(frozen=True)
class LidarAdvice:
    """Corridor readout for one cloud. ``clearance_m`` / ``ahead_m`` are inf when empty."""

    clearance_m: float
    stop: bool
    speed_scale: float
    yaw_bias: float
    ahead_m: float


def pointcloud2_xyz(msg) -> Optional[np.ndarray]:
    """Unpack ``sensor_msgs/PointCloud2`` XYZ without ROS. None if the layout is unusable."""
    if msg is None:
        return None
    fields = {f.name: f for f in (msg.fields or [])}
    if not all(n in fields for n in ("x", "y", "z")):
        return None
    fx, fy, fz = fields["x"], fields["y"], fields["z"]
    # FLOAT32 = 7
    if int(fx.datatype) != 7 or int(fy.datatype) != 7 or int(fz.datatype) != 7:
        return None
    h = int(msg.height or 0)
    w = int(msg.width or 0)
    n = w * h if h > 0 else w
    step = int(msg.point_step or 0)
    if n <= 0 or step < 12:
        return None
    raw = msg.data
    if isinstance(raw, np.ndarray):
        raw = raw.tobytes()
    elif not isinstance(raw, (bytes, bytearray, memoryview)):
        raw = bytes(raw)
    nbytes = n * step
    if len(raw) < nbytes:
        n = len(raw) // step
        nbytes = n * step
        if n <= 0:
            return None
    be = bool(getattr(msg, "is_bigendian", False))
    fmt = ">" if be else "<"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [f"{fmt}f4", f"{fmt}f4", f"{fmt}f4"],
        "offsets": [int(fx.offset), int(fy.offset), int(fz.offset)],
        "itemsize": step,
    })
    arr = np.frombuffer(raw, dtype=dtype, count=n)
    return np.stack((arr["x"], arr["y"], arr["z"]), axis=1).astype(np.float32, copy=False)


def _usable(xyz) -> Optional[np.ndarray]:
    """Finite body-height hits outside the chassis, within look-ahead."""
    if xyz is None:
        return None
    pts = np.asarray(xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
        return None
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.hypot(x, y)
    mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & (z >= Z_MIN) & (z <= Z_MAX)
        & (rng >= BODY_R) & (rng <= CLEAR_FAR)
    )
    pts = pts[mask]
    if pts.shape[0] == 0:
        return None
    return pts


def _speed_scale(clearance: float) -> float:
    if not np.isfinite(clearance):
        return 1.0
    if clearance <= STOP_X_MAX:
        return 0.0
    span = CLEAR_FAR - STOP_X_MAX
    if span <= 1e-9:
        return 1.0
    return float(np.clip((clearance - STOP_X_MAX) / span, 0.0, 1.0))


def _yaw_bias(pts: np.ndarray) -> float:
    """Steer away from the nearer side cluster. Corridor-only / empty -> 0."""
    x, y = pts[:, 0], pts[:, 1]
    side = (x >= BODY_R) & (np.abs(y) > CORRIDOR_Y) & (np.abs(y) <= NUDGE_Y)
    if not np.any(side):
        return 0.0
    sx, sy = x[side], y[side]
    left = sy > 0.0
    right = sy < 0.0
    near_l = float(np.min(sx[left])) if np.any(left) else None
    near_r = float(np.min(sx[right])) if np.any(right) else None
    if near_l is None and near_r is None:
        return 0.0
    if near_l is None:
        nearer, sign = near_r, 1.0
    elif near_r is None:
        nearer, sign = near_l, -1.0
    elif near_l <= near_r:
        nearer, sign = near_l, -1.0
    else:
        nearer, sign = near_r, 1.0
    strength = float(np.clip(1.0 - nearer / CLEAR_FAR, 0.0, 1.0))
    return sign * strength


def advise(xyz) -> LidarAdvice:
    """Robot-frame Nx3 -> corridor stop / slow / yaw. Empty or unusable -> cruise."""
    pts = _usable(xyz)
    if pts is None:
        return LidarAdvice(
            clearance_m=float("inf"), stop=False, speed_scale=1.0,
            yaw_bias=0.0, ahead_m=float("inf"),
        )
    x, y = pts[:, 0], pts[:, 1]
    corridor = (x >= STOP_X_MIN) & (x <= CLEAR_FAR) & (np.abs(y) <= CORRIDOR_Y)
    if np.any(corridor):
        clearance = float(np.min(x[corridor]))
    else:
        clearance = float("inf")
    wide = (x >= STOP_X_MIN) & (x <= CLEAR_FAR) & (np.abs(y) <= NUDGE_Y)
    ahead = float(np.min(x[wide])) if np.any(wide) else float("inf")
    stop = bool(np.any(
        (x >= STOP_X_MIN) & (x <= STOP_X_MAX) & (np.abs(y) <= CORRIDOR_Y)
    ))
    return LidarAdvice(
        clearance_m=clearance,
        stop=stop,
        speed_scale=_speed_scale(clearance),
        yaw_bias=_yaw_bias(pts),
        ahead_m=ahead,
    )
