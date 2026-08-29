"""Unattended-foreground priors for same-class lost-item distractors.

A unique target instance is never demoted for sitting next to a person.
Place commands rank by empty-bay occupancy, not camera proximity. Ownership
is personal space (~1.2 m); a person a few metres away occupies the stall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from marc_agent.command_parser import PARKED_VEHICLE_LANDMARKS, VEHICLE_LANDMARKS
from marc_agent.perception.situation_rules import POSE_GROUPS
from marc_agent.types import ObjectCandidate, PersonCandidate

PERSONAL_SPACE_M = 1.2
PERSONAL_SPACE_DIAG_FRAC = 0.12
BAY_RADIUS_M = 3.0
BAY_DIAG_FRAC = 0.16
MIN_CAMERA_DIST_M = 0.5
MIN_DEPTH_M = 0.15
NORMAL_POSES = POSE_GROUPS["upright"] | POSE_GROUPS["seated"]


@dataclass(frozen=True)
class PriorEvidence:
    """Ownership, camera proximity, and person-gap for one instance."""

    owned: bool
    camera_prox: float
    person_gap: float


def prior_key(evidence: PriorEvidence, *, demote_owned: bool) -> tuple:
    """Sort key: unowned first (when an alternative exists), then closer, then farther."""
    owned = 1 if demote_owned and evidence.owned else 0
    return (owned, -float(evidence.camera_prox), -float(evidence.person_gap))


def distinct_target_count(ranked: Sequence) -> int:
    """Number of distinct target candidate ids in a ranked pair list."""
    return len({item.target.candidate_id for item in ranked})


def box_gap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Pixel gap between boxes; 0 when they overlap."""
    ax1, ay1, ax2, ay2 = (float(v) for v in left)
    bx1, by1, bx2, by2 = (float(v) for v in right)
    dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0.0, max(by1 - ay2, ay1 - by2))
    if dx == 0.0 and dy == 0.0:
        return 0.0
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def bbox_area_frac(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> float:
    width = max(1.0, float(box[2]) - float(box[0]))
    height = max(1.0, float(box[3]) - float(box[1]))
    area = max(1.0, float(image_size[0]) * float(image_size[1]))
    return (width * height) / area


def _image_diag(image_size: tuple[int, int]) -> float:
    return math.hypot(float(image_size[0]), float(image_size[1]))


def _pixel_space(image_size: tuple[int, int]) -> float:
    return PERSONAL_SPACE_DIAG_FRAC * max(1.0, _image_diag(image_size))


def _xy_dist(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _pose_name(person) -> str:
    return str(getattr(person, "pose_class", "") or "").strip().lower()


def _same_camera_people(people: Iterable, camera_id: str) -> list:
    return [
        item for item in people
        if getattr(item, "camera_id", None) == camera_id
    ]


def _person_pixel_gap(target: ObjectCandidate, people: Iterable) -> tuple[float, float]:
    """(gap to any person, gap to nearest normal-pose person). Inf if none."""
    any_gap = math.inf
    normal_gap = math.inf
    for person in _same_camera_people(people, target.camera_id):
        gap = box_gap(target.bbox_xyxy, person.bbox_xyxy)
        if gap < any_gap:
            any_gap = gap
        if _pose_name(person) in NORMAL_POSES and gap < normal_gap:
            normal_gap = gap
    return any_gap, normal_gap


def _project_xy(projector, camera_id: str, box, pose_class: Optional[str] = None):
    if projector is None:
        return None
    project = getattr(projector, "project_contact", None)
    if not callable(project):
        return None
    try:
        xyz = project(camera_id, box, pose_class=pose_class)
    except TypeError:
        xyz = project(camera_id, box)
    if xyz is None or len(xyz) < 2:
        return None
    return (float(xyz[0]), float(xyz[1]))


def _camera_origin(projector, camera_id: str):
    if projector is None:
        return None
    lookup = getattr(projector, "lookup", None)
    if not callable(lookup):
        return None
    geometry = lookup(camera_id)
    if geometry is None:
        return None
    translation = getattr(geometry, "translation", None)
    if translation is None or len(translation) < 3:
        return None
    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )


def people_world_xy(people: Iterable, projector) -> tuple[tuple[float, float], ...]:
    """Project CCTV person contact pixels to world XY for near-field reuse."""
    points = []
    for person in people:
        xy = _project_xy(
            projector,
            getattr(person, "camera_id", ""),
            person.bbox_xyxy,
            pose_class=_pose_name(person) or None,
        )
        if xy is not None:
            points.append(xy)
    return tuple(points)


def evidence_for_candidate(
    target: ObjectCandidate,
    people: Iterable[PersonCandidate] = (),
    projector=None,
) -> PriorEvidence:
    """Image-space priors, with world metres when the projector can see the target."""
    any_gap, normal_gap = _person_pixel_gap(target, people)
    pixel_owned = any_gap < _pixel_space(target.image_size)
    person_gap = normal_gap if math.isfinite(normal_gap) else any_gap
    camera_prox = bbox_area_frac(target.bbox_xyxy, target.image_size)
    owned = pixel_owned

    world = _project_xy(projector, target.camera_id, target.bbox_xyxy)
    origin = _camera_origin(projector, target.camera_id)
    if world is not None and origin is not None:
        dist = math.sqrt(
            (world[0] - origin[0]) ** 2
            + (world[1] - origin[1]) ** 2
            + (0.0 - origin[2]) ** 2
        )
        camera_prox = 1.0 / max(MIN_CAMERA_DIST_M, dist)
        world_gaps = []
        normal_world = []
        for person in _same_camera_people(people, target.camera_id):
            pxy = _project_xy(
                projector,
                person.camera_id,
                person.bbox_xyxy,
                pose_class=_pose_name(person) or None,
            )
            if pxy is None:
                continue
            gap = _xy_dist(world, pxy)
            world_gaps.append(gap)
            if _pose_name(person) in NORMAL_POSES:
                normal_world.append(gap)
        if world_gaps:
            owned = min(world_gaps) < PERSONAL_SPACE_M
            person_gap = min(normal_world) if normal_world else min(world_gaps)
    return PriorEvidence(owned=owned, camera_prox=camera_prox, person_gap=person_gap)


def evidence_nearfield(
    world_xy: Optional[tuple],
    depth_m: Optional[float],
    people_xy: Iterable[tuple] = (),
) -> PriorEvidence:
    """World/depth priors for robot-view boxes inside the reveal disc."""
    depth = float(depth_m) if depth_m is not None and math.isfinite(depth_m) else 10.0
    camera_prox = 1.0 / max(MIN_DEPTH_M, depth)
    points = [item for item in people_xy if item is not None and len(item) >= 2]
    if world_xy is None or not points:
        return PriorEvidence(owned=False, camera_prox=camera_prox, person_gap=math.inf)
    gap = min(_xy_dist(world_xy, (float(item[0]), float(item[1]))) for item in points)
    return PriorEvidence(
        owned=gap < PERSONAL_SPACE_M,
        camera_prox=camera_prox,
        person_gap=gap,
    )


def parked_vehicle_count(
    camera_id: str,
    objects: Iterable[ObjectCandidate] = (),
) -> int:
    """Parked cars visible on one camera. Taxis count as road, not lot evidence."""
    return sum(
        1
        for item in objects
        if getattr(item, "camera_id", None) == camera_id
        and getattr(item, "class_name", None) in PARKED_VEHICLE_LANDMARKS
    )


def stall_is_empty(
    target: ObjectCandidate,
    people: Iterable[PersonCandidate] = (),
    objects: Iterable[ObjectCandidate] = (),
    projector=None,
) -> bool:
    """True when no person or parked vehicle occupies the target's parking bay.

    Adjacent vehicles (pixel gap, no overlap) do not fill the stall. Persons of
    any pose within a bay-scale radius do. Camera proximity is not used.
    """
    world = _project_xy(projector, target.camera_id, target.bbox_xyxy)
    bay_px = BAY_DIAG_FRAC * max(1.0, _image_diag(target.image_size))
    for person in _same_camera_people(people, target.camera_id):
        pxy = _project_xy(
            projector,
            person.camera_id,
            person.bbox_xyxy,
            pose_class=_pose_name(person) or None,
        )
        if world is not None and pxy is not None:
            if _xy_dist(world, pxy) < BAY_RADIUS_M:
                return False
            continue
        if box_gap(target.bbox_xyxy, person.bbox_xyxy) < bay_px:
            return False
    for item in objects:
        if getattr(item, "camera_id", None) != target.camera_id:
            continue
        if getattr(item, "class_name", None) not in VEHICLE_LANDMARKS:
            continue
        if item.candidate_id == target.candidate_id:
            continue
        if box_gap(target.bbox_xyxy, item.bbox_xyxy) <= 0.0:
            return False
    return True
