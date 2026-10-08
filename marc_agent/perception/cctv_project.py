"""Project CCTV contact pixels onto a per-camera ground or raised plane.

Geometry is a lookup of ``(K, T_world_from_cam, ground_height)``. Unit tests inject
that lookup; the runtime adapter reads CameraInfo, ground height, and tf2 from a
connected MARC client. The camera frame is optical (no extra axis flip).
"""

from __future__ import annotations

import logging
import math
import os
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Optional

from marc_agent.command_parser import Relation
from marc_agent.perception.ground_model import GroundModel, GroundSurface, default_ground_model
from marc_agent.perception.situation_rules import POSE_GROUPS


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


# Landmark footprint prior (#112): for each class, where the scorer's landmark
# point (the prim origin) falls in a full detection box, as fractions (fu, fv) of
# the box width and height, and its height dz above the calibrated ground.
# Per-class medians from scripts/fit_landmark_footprint.py on the #87 held-out
# generator GT, even-numbered scenes only (219 frames, 2026-10-08); odd scenes
# are the test. Tipped-over bicycles, scooters and trash cans widen fv; medians.
LANDMARK_FOOTPRINT: dict[str, tuple[float, float, float]] = {
    "bench": (0.548, 0.778, -0.034),
    "bicycle": (0.502, 0.841, 0.136),
    "hydrant": (0.634, 0.925, -0.037),
    "kick_scooter": (0.519, 0.898, 0.139),
    "normal_car_1": (0.500, 0.631, -0.040),
    "normal_car_2": (0.495, 0.753, -0.010),
    "picnic_table": (0.532, 0.797, -0.036),
    "police_car": (0.505, 0.719, -0.021),
    "postbox": (0.534, 0.843, -0.687),  # the prim origin sits below the ground
    "sports_car": (0.503, 0.655, -0.022),
    "sports_car_2": (0.512, 0.719, -0.022),
    "suv": (0.487, 0.747, -0.012),
    "taxi": (0.497, 0.690, -0.023),
    "trash_can": (0.640, 0.994, 0.334),
    "vending_machine": (0.367, 0.885, -0.037),
}

LOWER_HALF_CENTER = 0.75
PROSTRATE_HALF_CENTER = 0.60
BENDING_CONTACT_CENTER = 1.05
BENDING_ANCHOR_PULL = 0.45
BENDING_CENTER_HEIGHT_M = 0.18
GROUND_LYING_CONTACT_CENTER = 0.95


def contact_pixel(
    bbox_xyxy: tuple[float, float, float, float],
    pose_class: Optional[str] = None,
) -> tuple[float, float]:
    """Lower-half centroid of a detector box (ground contact), not the box center.

    Upright boxes use ``y1 + 0.75·height``. Prostrate poses sample higher in the
    image (closer to the torso) so a long lying box does not project from padded
    feet. Camera pitch is applied later by the calibrated ray, not here.
    """
    x1, y1, x2, y2 = bbox_xyxy
    height = max(0.0, float(y2) - float(y1))
    pose = (pose_class or "").strip().lower()
    if pose == "person_bending":
        frac = BENDING_CONTACT_CENTER
    elif pose in POSE_GROUPS["prostrate"]:
        frac = PROSTRATE_HALF_CENTER
    else:
        frac = LOWER_HALF_CENTER
    return ((float(x1) + float(x2)) / 2.0, float(y1) + frac * height)


def support_lift(
    support_class: Optional[str],
    relation: Optional[Relation | str],
) -> float:
    """Height above the ground for ``on`` / ``above`` a known support, else 0."""
    rel = relation.value if isinstance(relation, Relation) else relation
    if rel not in {"on", "above"} or not support_class:
        return 0.0
    return SUPPORT_HEIGHT_M.get(support_class, 0.0)


def raised_plane_z(
    ground_height: float,
    support_class: Optional[str],
    relation: Optional[Relation | str],
) -> float:
    """Ground plane, or ground + class height for ``on`` / ``above`` when known."""
    lift = support_lift(support_class, relation)
    if lift == 0.0:
        return float(ground_height)
    return float(ground_height) + lift


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


SURFACE_ITERATIONS = 8
SURFACE_TOLERANCE_M = 1e-3


def _flag_on(name: str) -> bool:
    """Feature flags default on and are read at call time (scripts/README.md)."""
    return os.environ.get(name, "1") != "0"


def pair_likelihood_enabled() -> bool:
    """Bounded pair score. Off until a parsed-relation replay shows a gain."""
    return os.environ.get("MARC_PAIR_LIKELIHOOD", "0").strip().lower() not in {
        "", "0", "false", "no", "off",
    }


_NEAR_DECAY_M = 2.0
_DIRECTION_DECAY_M = 3.0


def directional_likelihood(
    target_xyz: tuple[float, float, float],
    anchor_xyz: tuple[float, float, float],
    geometry: CameraGeometry,
    relation: Relation,
) -> Optional[float]:
    """cos(offset, camera axis) − d/3, and None when the offset faces the other way."""
    delta = (
        target_xyz[0] - anchor_xyz[0],
        target_xyz[1] - anchor_xyz[1],
        target_xyz[2] - anchor_xyz[2],
    )
    length = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
    if length < 1e-6:
        return None
    right = _column(geometry.rotation, 0)
    forward = _column(geometry.rotation, 2)
    if relation is Relation.LEFT:
        axis = tuple(-value for value in right)
    elif relation is Relation.RIGHT:
        axis = right
    elif relation is Relation.FRONT:
        axis = tuple(-value for value in forward)
    elif relation is Relation.BEHIND:
        axis = forward
    else:
        return None
    axis_len = math.sqrt(sum(value * value for value in axis))
    if axis_len < 1e-6:
        return None
    cosine = sum(delta[i] * axis[i] for i in range(3)) / (length * axis_len)
    if cosine <= 0.0:
        return None
    return cosine - length / _DIRECTION_DECAY_M


def intersect_ray_surface(
    geometry: CameraGeometry,
    u: float,
    v: float,
    surface: GroundSurface,
    lift: float = 0.0,
) -> Optional[tuple[float, float, float]]:
    """Intersect the pixel's ray with ``surface`` raised by ``lift`` metres.

    Fixed point: hit the plane at the current height, look the ground up at that
    XY, repeat until the height settles (at most ``SURFACE_ITERATIONS``).
    """
    plane_z = float(geometry.ground_height) + lift
    z = plane_z
    for _ in range(SURFACE_ITERATIONS):
        hit = intersect_ray_plane(geometry, u, v, z)
        ground = None if hit is None else surface.z(hit[0], hit[1])
        if ground is None:
            # Outside the calibrated area (or behind the camera): the flat plane.
            return intersect_ray_plane(geometry, u, v, plane_z)
        z_next = ground + lift
        settled = abs(z_next - z) < SURFACE_TOLERANCE_M
        z = z_next
        if settled:
            break
    return intersect_ray_plane(geometry, u, v, z)


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


def _support_footprint_xy(
    projector: "CctvProjector",
    camera_id: str,
    anchor_box: tuple[float, float, float, float],
    z_plane: float,
) -> Optional[tuple[float, float, float, float]]:
    """Axis-aligned world XY of the support box projected onto ``z_plane``."""
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
        return None
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def snap_xy_to_footprint(
    xyz: tuple[float, float, float],
    footprint: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    xmin, ymin, xmax, ymax = footprint
    return (
        min(max(xyz[0], xmin), xmax),
        min(max(xyz[1], ymin), ymax),
        xyz[2],
    )


def pair_alignment_score(
    projector: "CctvProjector",
    camera_id: str,
    target_box: tuple[float, float, float, float],
    anchor_box: tuple[float, float, float, float],
    relation: Optional[Relation | str],
    support_class: Optional[str],
    anchor_image_size: Optional[tuple[int, int]] = None,
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
        footprint = _support_footprint_xy(projector, camera_id, anchor_box, z_plane)
        if footprint is None:
            anchor_xyz = projector.project_landmark(
                camera_id, anchor_box, support_class, anchor_image_size
            )
            if anchor_xyz is None:
                return None
            dist = _xy_dist(target_xyz, anchor_xyz)
        else:
            xmin, ymin, xmax, ymax = footprint
            dist = _point_to_aabb_dist(
                target_xyz[0], target_xyz[1], xmin, ymin, xmax, ymax
            )
        return 1.0 / (1.0 + dist)
    anchor_xyz = projector.project_landmark(camera_id, anchor_box, support_class, anchor_image_size)
    if anchor_xyz is None:
        return None
    if rel in {"left", "right", "front", "behind"} and isinstance(relation, Relation):
        if pair_likelihood_enabled():
            return directional_likelihood(target_xyz, anchor_xyz, geometry, relation)
        return world_relation_score(target_xyz, anchor_xyz, geometry, relation)
    dist = _xy_dist(target_xyz, anchor_xyz)
    if pair_likelihood_enabled() and rel in {"near", "beside", None}:
        return -dist / _NEAR_DECAY_M
    return 1.0 / (1.0 + dist)


def static_lookup(geometry_by_camera: dict[str, CameraGeometry]) -> GeometryLookup:
    """Injected lookup for tests: camera id -> frozen geometry."""

    def lookup(camera_id: str) -> Optional[CameraGeometry]:
        return geometry_by_camera.get(camera_id)

    return lookup


class CctvProjector:
    """Pixel-to-world adapter over a geometry lookup.

    With a ``GroundModel`` and ``MARC_GROUND_CALIBRATION`` on (default), every
    projection lands on the camera's calibrated ground surface when its pose
    fingerprint matches (#112); otherwise on the flat ``ground_height`` plane,
    exactly as without a model.
    """

    def __init__(self, lookup: GeometryLookup, ground: Optional[GroundModel] = None):
        self.lookup = lookup
        self.ground = ground

    @classmethod
    def from_client(cls, client) -> "CctvProjector":
        """Return the process-wide projector for this client (one tf2 listener)."""
        projector = _CLIENT_PROJECTORS.get(client)
        if projector is None:
            ground = default_ground_model()
            if ground is None:
                log.warning("[cctv-project] no ground model shipped; flat planes only")
            projector = cls(Tf2ClientLookup(client), ground)
            try:
                _CLIENT_PROJECTORS[client] = projector
            except TypeError:
                pass
        return projector

    def ground_status(self, camera_id: str) -> str:
        """Whether this camera's projections use calibrated ground, and if not why."""
        if self.ground is None:
            return "no_model"
        if not _flag_on("MARC_GROUND_CALIBRATION"):
            return "disabled"
        geometry = self.lookup(camera_id)
        if geometry is None:
            return "no_geometry"
        return self.ground.for_camera(camera_id, geometry)[1]

    def _surface(self, camera_id: str, geometry: CameraGeometry) -> Optional[GroundSurface]:
        if self.ground is None or not _flag_on("MARC_GROUND_CALIBRATION"):
            return None
        return self.ground.for_camera(camera_id, geometry)[0]

    def _hit(
        self,
        camera_id: str,
        geometry: CameraGeometry,
        u: float,
        v: float,
        z_plane: float,
        *,
        ground_contact: bool,
    ) -> Optional[tuple[float, float, float]]:
        """The pixel on the calibrated ground, else on the plane ``z = z_plane``.

        Only ground contact uses the surface. A raised plane (on/above a support)
        stays flat: SUPPORT_HEIGHT_M was set against the flat plane (#112).
        """
        surface = self._surface(camera_id, geometry) if ground_contact else None
        if surface is None:
            return intersect_ray_plane(geometry, u, v, z_plane)
        return intersect_ray_surface(geometry, u, v, surface)

    def project_pixel(
        self,
        camera_id: str,
        u: float,
        v: float,
        *,
        z_plane: Optional[float] = None,
    ) -> Optional[tuple[float, float, float]]:
        geometry = self.lookup(camera_id)
        if geometry is None:
            return None
        plane = geometry.ground_height if z_plane is None else z_plane
        return self._hit(camera_id, geometry, u, v, plane, ground_contact=z_plane is None)

    def project_landmark(
        self,
        camera_id: str,
        bbox_xyxy: tuple[float, float, float, float],
        class_name: Optional[str] = None,
        image_size: Optional[tuple[int, int]] = None,
    ) -> Optional[tuple[float, float, float]]:
        """World point of a landmark's prim origin, which the scorer measures (#112).

        With ``MARC_LANDMARK_FOOTPRINT`` on (default), a known class's box maps
        through its footprint prior: the origin's pixel as a fraction of the box,
        and its height above the ground. A box cut by the frame edge, or an
        unknown class, keeps the generic contact rule (the prior assumes a full box).
        """
        prior = LANDMARK_FOOTPRINT.get(class_name or "")
        x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
        clipped = image_size is None or (
            x1 <= 1.0 or y1 <= 1.0 or x2 >= image_size[0] - 1 or y2 >= image_size[1] - 1
        )
        if prior is None or clipped or not _flag_on("MARC_LANDMARK_FOOTPRINT"):
            return self.project_contact(camera_id, bbox_xyxy)
        geometry = self.lookup(camera_id)
        if geometry is None:
            return None
        fu, fv, dz = prior
        hit = self._hit(
            camera_id, geometry, x1 + fu * (x2 - x1), y1 + fv * (y2 - y1),
            geometry.ground_height, ground_contact=True,
        )
        if hit is None:
            return None
        return (hit[0], hit[1], hit[2] + dz)

    def project_contact(
        self,
        camera_id: str,
        bbox_xyxy: tuple[float, float, float, float],
        *,
        relation: Optional[Relation | str] = None,
        support_class: Optional[str] = None,
        raised: bool = False,
        pose_class: Optional[str] = None,
        support_box: Optional[tuple[float, float, float, float]] = None,
        calibrated: bool = True,
    ) -> Optional[tuple[float, float, float]]:
        """Contact point of a box. ``calibrated=False`` keeps the flat plane, for
        geometry that must share a frame with people (they stay on the plane)."""
        geometry = self.lookup(camera_id)
        if geometry is None:
            return None
        rel = relation
        if raised and rel is None:
            rel = Relation.ON
        rel_name = rel.value if isinstance(rel, Relation) else rel
        if pose_class == "person_lying_down" and rel_name not in {"on", "above"}:
            x1, y1, x2, y2 = bbox_xyxy
            u = (float(x1) + float(x2)) / 2.0
            v = float(y1) + GROUND_LYING_CONTACT_CENTER * (float(y2) - float(y1))
        else:
            u, v = contact_pixel(bbox_xyxy, pose_class=pose_class)
        if (
            pose_class == "person_bending"
            and support_box is not None
            and rel_name not in {"on", "above"}
        ):
            support_u = (float(support_box[0]) + float(support_box[2])) / 2.0
            u += BENDING_ANCHOR_PULL * (support_u - u)
        z_plane = raised_plane_z(geometry.ground_height, support_class, rel)
        ground_contact = calibrated and support_lift(support_class, rel) == 0.0
        if pose_class:
            # People keep the flat plane (#112): in the scored scenario a person's
            # point sits 0.1-0.5 m above the calibrated terrain (public sar03,
            # sar07-09), unlike objects, which sit on it, and unlike the held-out
            # generator scenes. The pose constants below were fitted to this plane;
            # re-fitting people is #122.
            xyz = intersect_ray_plane(geometry, u, v, z_plane)
        else:
            xyz = self._hit(camera_id, geometry, u, v, z_plane, ground_contact=ground_contact)
        if xyz is None:
            return None
        if pose_class == "person_bending" and rel_name not in {"on", "above"}:
            xyz = (xyz[0], xyz[1], xyz[2] + BENDING_CENTER_HEIGHT_M)
        if rel_name in {"on", "above"} and support_box is not None:
            footprint = _support_footprint_xy(self, camera_id, support_box, z_plane)
            if footprint is not None:
                xyz = snap_xy_to_footprint(xyz, footprint)
        return xyz


_ROBOT_FRAME_TOKENS = ("car_base_link", "odom", "base_link")


def _keep_cctv_transform(transform, static: bool = True) -> bool:
    """Keep camera extrinsics and drop robot frames, which make tf2 log TF_OLD_DATA.

    Camera extrinsics are latched on /tf_static, so any non-robot frame is kept there
    and a camera name other than ``rig_*`` still projects. Live /tf is high-rate robot
    traffic: only ``rig_*`` frames are kept, as before, so the listener adds no load.
    """
    parent = (getattr(getattr(transform, "header", None), "frame_id", None) or "").lstrip("/")
    child = (getattr(transform, "child_frame_id", None) or "").lstrip("/")
    if not static:
        return parent.startswith("rig_") or child.startswith("rig_")
    blob = f"{parent} {child}"
    if any(token in blob for token in _ROBOT_FRAME_TOKENS):
        return False
    return bool(parent or child)


def _start_tf_listener(client):
    """Latch CCTV frames from /tf_static and /tf without a second executor or robot odom."""
    from tf2_msgs.msg import TFMessage
    from tf2_ros.buffer import Buffer

    from marc_sdk import protocol as P

    buffer = Buffer()

    def ingest(msg, *, static: bool):
        setter = buffer.set_transform_static if static else buffer.set_transform
        for transform in getattr(msg, "transforms", ()) or ():
            if not _keep_cctv_transform(transform, static):
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
