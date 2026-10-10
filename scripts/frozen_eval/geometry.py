"""Camera geometry from generator GT camera blocks, in the runtime projector's convention.

The GT ``camera`` block is the USD camera: ``position``, ``orientation_euler_deg``
(roll, pitch, yaw), focal length and apertures in mm, ``ground_height``. The
world-from-camera rotation is ``Rx(roll) @ Ry(pitch) @ Rz(yaw)`` in USD camera axes
(+X right, +Y up, -Z forward). This was fitted on GT ``world_xyz``: 96% of object
origins project inside their GT box, and every other Euler order fits under 8%.
(``cctv_gt_to_qwen._ground_xyz`` uses ``Rz @ Ry @ Rx``, which does not fit.)

``CameraGeometry`` (runtime ``cctv_project``) uses optical axes (+Z forward, +Y
down), so the rotation is right-multiplied by ``diag(1, -1, -1)``.
"""

from __future__ import annotations

import math

import numpy as np

from marc_agent.perception.cctv_project import CameraGeometry, intersect_ray_plane

USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def _rot(axis: str, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def geometry_from_gt(payload: dict) -> CameraGeometry:
    """Runtime ``CameraGeometry`` for one generator GT payload."""
    cam = payload["camera"]
    width = float(payload["image"]["width"])
    height = float(payload["image"]["height"])
    euler = cam["orientation_euler_deg"]
    rot_usd = _rot("x", euler["roll"]) @ _rot("y", euler["pitch"]) @ _rot("z", euler["yaw"])
    rotation = rot_usd @ USD_TO_OPTICAL
    return CameraGeometry(
        fx=width * cam["focal_length_mm"] / cam["horizontal_aperture_mm"],
        fy=height * cam["focal_length_mm"] / cam["vertical_aperture_mm"],
        cx=width / 2.0,
        cy=height / 2.0,
        rotation=tuple(tuple(float(v) for v in row) for row in rotation),
        translation=tuple(float(v) for v in cam["position"]),
        ground_height=float(cam["ground_height"]),
    )


def project_point(geometry: CameraGeometry, xyz) -> tuple[float, float] | None:
    """World point -> pixel, or None when behind the camera."""
    rotation = np.asarray(geometry.rotation, dtype=float)
    p_cam = rotation.T @ (np.asarray(xyz, dtype=float) - np.asarray(geometry.translation))
    if p_cam[2] <= 1e-6:
        return None
    return (geometry.cx + geometry.fx * p_cam[0] / p_cam[2],
            geometry.cy + geometry.fy * p_cam[1] / p_cam[2])


def ground_point(geometry: CameraGeometry, bbox) -> tuple[float, float, float] | None:
    """Bottom-centre of a box intersected with the camera's ground plane."""
    u = 0.5 * (float(bbox[0]) + float(bbox[2]))
    v = float(bbox[3])
    return intersect_ray_plane(geometry, u, v, geometry.ground_height)


def compare_geometry(a: CameraGeometry, b: CameraGeometry) -> dict:
    """Differences between two geometries of one camera (generator vs SDK)."""
    rot_a = np.asarray(a.rotation, dtype=float)
    rot_b = np.asarray(b.rotation, dtype=float)
    cos_angle = (np.trace(rot_a.T @ rot_b) - 1.0) / 2.0
    return {
        "translation_m": float(np.linalg.norm(np.asarray(a.translation) - np.asarray(b.translation))),
        "rotation_deg": math.degrees(math.acos(max(-1.0, min(1.0, cos_angle)))),
        "fx": float(a.fx - b.fx),
        "fy": float(a.fy - b.fy),
        "cx": float(a.cx - b.cx),
        "cy": float(a.cy - b.cy),
        "ground_height_m": float(a.ground_height - b.ground_height),
    }
