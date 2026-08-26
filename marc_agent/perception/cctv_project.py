"""Project CCTV contact pixels onto a per-camera ground or raised plane.

Geometry is a lookup of ``(K, T_world_from_cam, ground_height)``. Unit tests inject
that lookup; the runtime adapter reads CameraInfo, ground height, and tf2 from a
connected MARC client. The camera frame is optical (no extra axis flip).
"""

from __future__ import annotations

import logging
import math
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Optional

from marc_agent.command_parser import Relation


log = logging.getLogger("marc_agent.cctv_project")

GeometryLookup = Callable[[str], Optional["CameraGeometry"]]
_CLIENT_PROJECTORS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Approximate support heights (metres) for raised-plane ``on`` / ``above``.
SUPPORT_HEIGHT_M = {
    "bench": 0.45,
    "picnic_table": 0.74,
    "trash_can": 0.90,
    "vending_machine": 1.85,
    "postbox": 1.15,
    "hydrant": 0.70,
    "bicycle": 0.40,
    "kick_scooter": 0.35,
    "police_car": 1.50,
    "normal_car_1": 1.50,
    "normal_car_2": 1.50,
    "sports_car": 1.40,
    "sports_car_2": 1.40,
    "suv": 1.70,
    "taxi": 1.50,
}


@dataclass(frozen=True)
class CameraGeometry:
    """Pinhole K, world-from-camera rotation/translation, and ground plane z."""

    fx: float
    fy: float
    cx: float
    cy: float
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation: tuple[float, float, float]
    ground_height: float


def k_from_camera_info(info) -> Optional[tuple[float, float, float, float]]:
    """Read pinhole fx, fy, cx, cy from CameraInfo.k without testing array truthiness."""
    if info is None:
        return None
    k = getattr(info, "k", None)
    if k is None:
        return None
    try:
        values = [float(v) for v in k]
    except (TypeError, ValueError):
        return None
    if len(values) < 6:
        return None
    fx, fy, cx, cy = values[0], values[4], values[2], values[5]
    if fx <= 1.0 or fy <= 1.0:
        return None
    return fx, fy, cx, cy


def contact_pixel(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    """Bottom-center of a detector box (feet / ground contact), not the box center."""
    x1, _y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, float(y2))


def raised_plane_z(
    ground_height: float,
    support_class: Optional[str],
    relation: Optional[Relation | str],
) -> float:
    """Ground plane, or ground + class height for ``on`` / ``above`` when known."""
    rel = relation.value if isinstance(relation, Relation) else relation
    if rel not in {"on", "above"} or not support_class:
        return float(ground_height)
    extra = SUPPORT_HEIGHT_M.get(support_class)
    if extra is None:
        return float(ground_height)
    return float(ground_height) + extra


def _quat_to_rotation(
    x: float, y: float, z: float, w: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0.0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _rotate(rotation, vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        rotation[0][0] * vec[0] + rotation[0][1] * vec[1] + rotation[0][2] * vec[2],
        rotation[1][0] * vec[0] + rotation[1][1] * vec[1] + rotation[1][2] * vec[2],
        rotation[2][0] * vec[0] + rotation[2][1] * vec[1] + rotation[2][2] * vec[2],
    )


def _column(rotation, index: int) -> tuple[float, float, float]:
    return (rotation[0][index], rotation[1][index], rotation[2][index])


def intersect_ray_plane(
    geometry: CameraGeometry,
    u: float,
    v: float,
    z_plane: float,
) -> Optional[tuple[float, float, float]]:
    """Back-project ``(u, v)`` and intersect the ray with ``z = z_plane`` (t > 0)."""
    if geometry.fx == 0.0 or geometry.fy == 0.0:
        return None
    direction_cam = (
        (float(u) - geometry.cx) / geometry.fx,
        (float(v) - geometry.cy) / geometry.fy,
        1.0,
    )
    direction_world = _rotate(geometry.rotation, direction_cam)
    origin = geometry.translation
    dz = direction_world[2]
    if abs(dz) < 1e-12:
        return None
    t_ray = (float(z_plane) - origin[2]) / dz
    if t_ray <= 0.0:
        return None
    return (
        origin[0] + t_ray * direction_world[0],
        origin[1] + t_ray * direction_world[1],
        origin[2] + t_ray * direction_world[2],
    )


def project_pixel(
    camera_id: str,
    u: float,
    v: float,
    *,
    z_plane: Optional[float] = None,
    lookup: GeometryLookup,
) -> Optional[tuple[float, float, float]]:
    """Project one pixel through ``lookup(camera_id)`` onto a horizontal plane."""
    geometry = lookup(camera_id)
    if geometry is None:
        return None
    plane = geometry.ground_height if z_plane is None else z_plane
    return intersect_ray_plane(geometry, u, v, plane)


def world_relation_score(
    target_xyz: tuple[float, float, float],
    anchor_xyz: tuple[float, float, float],
    geometry: CameraGeometry,
    relation: Relation,
) -> float:
    """Score left/right/front/behind using camera axes in world XY."""
    delta = (
        target_xyz[0] - anchor_xyz[0],
        target_xyz[1] - anchor_xyz[1],
        target_xyz[2] - anchor_xyz[2],
    )
    right = _column(geometry.rotation, 0)
    along_right = delta[0] * right[0] + delta[1] * right[1] + delta[2] * right[2]
    origin = geometry.translation
    t_dist = math.sqrt(
        (target_xyz[0] - origin[0]) ** 2
        + (target_xyz[1] - origin[1]) ** 2
        + (target_xyz[2] - origin[2]) ** 2
    )
    a_dist = math.sqrt(
        (anchor_xyz[0] - origin[0]) ** 2
        + (anchor_xyz[1] - origin[1]) ** 2
        + (anchor_xyz[2] - origin[2]) ** 2
    )
    if relation is Relation.LEFT:
        return -along_right
    if relation is Relation.RIGHT:
        return along_right
    if relation is Relation.FRONT:
        return a_dist - t_dist
    if relation is Relation.BEHIND:
        return t_dist - a_dist
    return 0.0


def _xy_dist(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _point_to_aabb_dist(
    px: float, py: float, xmin: float, ymin: float, xmax: float, ymax: float
) -> float:
    dx = 0.0 if xmin <= px <= xmax else min(abs(px - xmin), abs(px - xmax))
    dy = 0.0 if ymin <= py <= ymax else min(abs(py - ymin), abs(py - ymax))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    if xmin <= px <= xmax:
        return dy
    if ymin <= py <= ymax:
        return dx
    return math.hypot(dx, dy)


def pair_alignment_score(
    projector: "CctvProjector",
    camera_id: str,
    target_box: tuple[float, float, float, float],
    anchor_box: tuple[float, float, float, float],
    relation: Optional[Relation | str],
    support_class: Optional[str],
) -> Optional[float]:
    """Score one target/landmark pair in world metres. Higher is better.

    ``on`` / ``above`` prefer a contact pixel inside the support footprint.
    ``near`` / ``beside`` prefer small ground XY distance. Directional
    relations reuse camera-axis scores. Returns None when either end fails.
    """
    geometry = projector.lookup(camera_id)
    if geometry is None:
        return None
    target_xyz = projector.project_contact(
        camera_id,
        target_box,
        relation=relation,
        support_class=support_class,
    )
    if target_xyz is None:
        return None
    rel = relation.value if isinstance(relation, Relation) else relation
    if rel in {"on", "above"}:
        z_plane = raised_plane_z(geometry.ground_height, support_class, relation)
        x1, y1, x2, y2 = anchor_box
        corners = []
        for u, v in (
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
            ((x1 + x2) / 2.0, y2),
        ):
            point = projector.project_pixel(camera_id, u, v, z_plane=z_plane)
            if point is not None:
                corners.append(point)
        if len(corners) < 2:
            anchor_xyz = projector.project_contact(camera_id, anchor_box)
            if anchor_xyz is None:
                return None
            dist = _xy_dist(target_xyz, anchor_xyz)
        else:
            xs = [point[0] for point in corners]
            ys = [point[1] for point in corners]
            dist = _point_to_aabb_dist(
                target_xyz[0], target_xyz[1], min(xs), min(ys), max(xs), max(ys)
            )
        return 1.0 / (1.0 + dist)
    anchor_xyz = projector.project_contact(camera_id, anchor_box)
    if anchor_xyz is None:
        return None
    if rel in {"left", "right", "front", "behind"} and isinstance(relation, Relation):
        return world_relation_score(target_xyz, anchor_xyz, geometry, relation)
    return 1.0 / (1.0 + _xy_dist(target_xyz, anchor_xyz))


def static_lookup(geometry_by_camera: dict[str, CameraGeometry]) -> GeometryLookup:
    """Injected lookup for tests: camera id -> frozen geometry."""

    def lookup(camera_id: str) -> Optional[CameraGeometry]:
        return geometry_by_camera.get(camera_id)

    return lookup


class CctvProjector:
    """Pixel-to-world adapter over a geometry lookup."""

    def __init__(self, lookup: GeometryLookup):
        self.lookup = lookup

    @classmethod
    def from_client(cls, client) -> "CctvProjector":
        """Return the process-wide projector for this client (one tf2 listener)."""
        projector = _CLIENT_PROJECTORS.get(client)
        if projector is None:
            projector = cls(Tf2ClientLookup(client))
            try:
                _CLIENT_PROJECTORS[client] = projector
            except TypeError:
                pass
        return projector

    def project_pixel(
        self,
        camera_id: str,
        u: float,
        v: float,
        *,
        z_plane: Optional[float] = None,
    ) -> Optional[tuple[float, float, float]]:
        return project_pixel(
            camera_id, u, v, z_plane=z_plane, lookup=self.lookup
        )

    def project_contact(
        self,
        camera_id: str,
        bbox_xyxy: tuple[float, float, float, float],
        *,
        relation: Optional[Relation | str] = None,
        support_class: Optional[str] = None,
        raised: bool = False,
    ) -> Optional[tuple[float, float, float]]:
        geometry = self.lookup(camera_id)
        if geometry is None:
            return None
        u, v = contact_pixel(bbox_xyxy)
        rel = relation
        if raised and rel is None:
            rel = Relation.ON
        z_plane = raised_plane_z(geometry.ground_height, support_class, rel)
        return intersect_ray_plane(geometry, u, v, z_plane)


def _keep_cctv_transform(transform) -> bool:
    """Drop robot odom (car_base_link) so tf2 never logs TF_OLD_DATA on CCTV lookup."""
    parent = (getattr(getattr(transform, "header", None), "frame_id", None) or "").lstrip("/")
    child = (getattr(transform, "child_frame_id", None) or "").lstrip("/")
    return parent.startswith("rig_") or child.startswith("rig_")


def _start_tf_listener(client):
    """Latch CCTV frames from /tf_static and /tf without a second executor or robot odom."""
    from tf2_msgs.msg import TFMessage
    from tf2_ros.buffer import Buffer

    from marc_sdk import protocol as P

    buffer = Buffer()

    def ingest(msg, *, static: bool):
        setter = buffer.set_transform_static if static else buffer.set_transform
        for transform in getattr(msg, "transforms", ()) or ():
            if not _keep_cctv_transform(transform):
                continue
            try:
                setter(transform, "cctv")
            except Exception:
                continue

    subscribe = getattr(client, "subscribe", None)
    if subscribe is None:
        raise RuntimeError("MARC client has no subscribe(); connect before projecting")
    static_sub = subscribe(
        "/tf_static", TFMessage, lambda msg: ingest(msg, static=True), qos=P.QOS_TRANSIENT
    )
    live_sub = subscribe(
        "/tf", TFMessage, lambda msg: ingest(msg, static=False), qos=P.QOS_RELIABLE
    )
    return buffer, (static_sub, live_sub)


class Tf2ClientLookup:
    """Runtime adapter: CameraInfo + ground height + tf2 ``world`` <- ``camera_id``."""

    def __init__(self, client, *, timeout_s: float = 2.0):
        self._client = client
        self._timeout_s = timeout_s
        self._buffer = None
        self._listener = None
        self._geometry: dict[str, CameraGeometry] = {}

    def _ensure_tf(self):
        if self._buffer is not None:
            return self._buffer
        if getattr(self._client, "subscribe", None) is None:
            raise RuntimeError("MARC client has no subscribe(); connect before projecting")
        self._buffer, self._listener = _start_tf_listener(self._client)
        return self._buffer

    def __call__(self, camera_id: str) -> Optional[CameraGeometry]:
        cached = self._geometry.get(camera_id)
        if cached is not None:
            return cached
        try:
            self._ensure_tf()
            info = self._client.get_cctv_info(camera_id)
            ground_height = self._client.get_cctv_ground_height(camera_id)
            if info is None or ground_height is None:
                log.warning(
                    "[cctv-project] missing CameraInfo or ground height camera=%s",
                    camera_id,
                )
                return None
            intrinsics = k_from_camera_info(info)
            if intrinsics is None:
                log.warning("[cctv-project] CameraInfo.k unusable camera=%s", camera_id)
                return None
            fx, fy, cx, cy = intrinsics
            translation, quaternion = self._lookup_tf(camera_id)
            if translation is None or quaternion is None:
                return None
            geometry = CameraGeometry(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rotation=_quat_to_rotation(*quaternion),
                translation=translation,
                ground_height=float(ground_height),
            )
            self._geometry[camera_id] = geometry
            return geometry
        except Exception as exc:
            log.warning(
                "[cctv-project] geometry lookup failed camera=%s error=%s detail=%s",
                camera_id,
                type(exc).__name__,
                exc,
            )
            return None

    def _lookup_tf(
        self, camera_id: str
    ) -> tuple[Optional[tuple[float, float, float]], Optional[tuple[float, float, float, float]]]:
        from rclpy.duration import Duration
        from rclpy.time import Time

        buffer = self._ensure_tf()
        deadline = time.monotonic() + self._timeout_s
        last_error: Optional[Exception] = None
        while time.monotonic() <= deadline:
            try:
                trans = buffer.lookup_transform(
                    "world",
                    camera_id,
                    Time(),
                    timeout=Duration(
                        seconds=max(0.05, min(0.2, deadline - time.monotonic()))
                    ),
                )
                t = trans.transform.translation
                q = trans.transform.rotation
                return (float(t.x), float(t.y), float(t.z)), (
                    float(q.x),
                    float(q.y),
                    float(q.z),
                    float(q.w),
                )
            except Exception as exc:
                last_error = exc
                time.sleep(0.02)
        log.warning(
            "[cctv-project] tf2 lookup failed camera=%s error=%s detail=%s",
            camera_id,
            type(last_error).__name__ if last_error else "Timeout",
            last_error,
        )
        return None, None
