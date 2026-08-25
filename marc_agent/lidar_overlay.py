"""Project live lidar hits into world discs for the occupancy overlay.

Occupancy is roads-only. PointCloud2 hits in the robot/base frame are height-gated,
range-capped, clustered, and transformed into world discs the planner stamps as
blocked cells. Frontal bumper-box tests force a hard local stop even when D* still
has a path.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

# Drop ground / canopy, cap range, cluster hits to cell-sized discs.
LIDAR_Z_MIN = 0.15
LIDAR_Z_MAX = 1.5
LIDAR_RANGE_M = 20.0
LIDAR_MIN_RANGE_M = 0.20
LIDAR_HIT_RADIUS = 0.35
LIDAR_CLUSTER_M = 0.25


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


def lidar_xyz_to_obstacles(
    xyz,
    pose,
    radius_m: float = LIDAR_HIT_RADIUS,
    cell_m: float = LIDAR_CLUSTER_M,
):
    """Robot-frame XYZ hits -> world discs ``[(x, y, radius_m), ...]``.

    ``pose`` is chassis ``(x, y, yaw)``. Lidar +X is forward, +Y left (REP-103).
    Hits are height-gated, range-capped, and clustered to ``cell_m`` centres.
    """
    if xyz is None:
        return []
    pts = np.asarray(xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
        return []
    z = pts[:, 2]
    rng = np.hypot(pts[:, 0], pts[:, 1])
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(z)
    mask = (
        finite
        & (z >= LIDAR_Z_MIN) & (z <= LIDAR_Z_MAX)
        & (rng <= LIDAR_RANGE_M) & (rng >= LIDAR_MIN_RANGE_M)
    )
    pts = pts[mask]
    if pts.shape[0] == 0:
        return []
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    cc, ss = math.cos(yaw), math.sin(yaw)
    wx = x + pts[:, 0] * cc - pts[:, 1] * ss
    wy = y + pts[:, 0] * ss + pts[:, 1] * cc
    qi = np.round(wx / cell_m).astype(np.int32)
    qj = np.round(wy / cell_m).astype(np.int32)
    uniq = np.unique(np.stack((qi, qj), axis=1), axis=0)
    r = float(radius_m)
    return [(float(i) * cell_m, float(j) * cell_m, r) for i, j in uniq]


def lidar_cloud_to_obstacles(msg, pose, radius_m=LIDAR_HIT_RADIUS, cell_m=LIDAR_CLUSTER_M):
    """``get_lidar()`` PointCloud2 + world pose -> world discs."""
    xyz = pointcloud2_xyz(msg)
    return lidar_xyz_to_obstacles(xyz, pose, radius_m=radius_m, cell_m=cell_m)


def frontal_hit(obstacles, pose, x_max=1.2, y_half=0.4, x_min=0.15):
    """True if any disc centre (expanded by its radius) sits in a box ahead of the bumper."""
    if not obstacles:
        return False
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    cc, ss = math.cos(yaw), math.sin(yaw)
    for ox, oy, rad in obstacles:
        dx, dy = float(ox) - x, float(oy) - y
        fwd = dx * cc + dy * ss
        lat = -dx * ss + dy * cc
        r = float(rad)
        if (x_min - r) <= fwd <= (x_max + r) and abs(lat) <= (y_half + r):
            return True
    return False


def discs_to_cells(
    discs: Iterable[Sequence[float]],
    res: float,
    ox: float,
    oy: float,
) -> list:
    """World discs ``[(wx, wy, r), ...]`` -> unique occupancy ``(row, col)`` cells."""
    cells = []
    seen = set()
    inv = 1.0 / float(res)
    for item in discs or []:
        wx, wy = float(item[0]), float(item[1])
        col = int((wx - float(ox)) * inv)
        row = int((wy - float(oy)) * inv)
        key = (row, col)
        if key not in seen:
            seen.add(key)
            cells.append([row, col])
    return cells
