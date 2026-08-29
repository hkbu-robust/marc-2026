"""Stage 2 retrieval state: mission, reveal search area, and approach-goal helper.

Nav reads ``search_xy`` each tick so a late msg 411 can retarget the drive without
treating the hint as the object itself. ``search_area_approach_goal`` returns a
road-side point on the hint disk edge (or the center when there is no radius).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


DEFAULT_GOAL_TOLERANCE = 0.7


def _xy(seq) -> Optional[tuple]:
    if seq is None or len(seq) < 2:
        return None
    return (float(seq[0]), float(seq[1]))


def search_area_approach_goal(hint_center_xy, hint_radius_m, pose_xy) -> tuple:
    """A road-reachable point near the hint disk edge or center — not the object centroid.

    Outside the disk: the near intersection of the pose→center ray with the circle.
    Inside the disk: the current pose (already in the search area).
    Missing/zero radius: the hint center (or grounding fallback the caller passed).
    """
    if hint_center_xy is None or len(hint_center_xy) < 2:
        raise ValueError("hint_center_xy is required")
    if pose_xy is None or len(pose_xy) < 2:
        raise ValueError("pose_xy is required")
    cx, cy = float(hint_center_xy[0]), float(hint_center_xy[1])
    px, py = float(pose_xy[0]), float(pose_xy[1])
    radius = float(hint_radius_m or 0.0)
    dx, dy = cx - px, cy - py
    dist = math.hypot(dx, dy)
    if radius <= 0.0 or dist <= 1e-9:
        return (cx, cy)
    if dist <= radius:
        return (px, py)
    scale = (dist - radius) / dist
    return (px + dx * scale, py + dy * scale)


HINT_JUMP_M = 0.50


def snapshot_approach_goal(
    prev,
    hint_center_xy,
    hint_radius_m,
    pose_xy,
    jump_m: float = HINT_JUMP_M,
):
    """Keep the first approach XY until the reveal hint jumps, not when the robot moves.

    ``prev`` is ``(center_xy, radius, goal_xy)`` or None. Returns the same triple,
    or None when the hint is missing.
    """
    center = _xy(hint_center_xy)
    if center is None:
        return None
    radius = float(hint_radius_m or 0.0)
    if prev is not None:
        prev_center, prev_radius, _prev_goal = prev
        jumped = math.hypot(
            center[0] - float(prev_center[0]),
            center[1] - float(prev_center[1]),
        ) > float(jump_m)
        radius_changed = abs(radius - float(prev_radius)) > 0.05
        if not jumped and not radius_changed:
            return prev
    goal = search_area_approach_goal(center, radius, pose_xy)
    return (center, radius, goal)


@dataclass
class RetrievalState:
    owner_xy: Optional[tuple] = None
    grounding_xy: Optional[tuple] = None
    hint_center: Optional[tuple] = None
    hint_radius: float = 0.0
    target_type: Optional[str] = None
    task_description: str = ""
    time_limit: Optional[float] = None
    grounding_score: Optional[float] = None
    picked: bool = False
    basket_occupied: bool = False
    pick_attempts: int = 0
    pick_reason: str = ""
    people_xy: tuple = ()

    @property
    def hint_center_xy(self) -> Optional[tuple]:
        return self.hint_center

    @hint_center_xy.setter
    def hint_center_xy(self, value) -> None:
        self.hint_center = value

    @property
    def hint_radius_m(self) -> Optional[float]:
        r = float(self.hint_radius or 0.0)
        return r if r > 0.0 else None

    @hint_radius_m.setter
    def hint_radius_m(self, value) -> None:
        self.hint_radius = float(value or 0.0)

    def adopt_owner(self, owner_position) -> None:
        xy = _xy(owner_position)
        if xy is not None:
            self.owner_xy = xy

    def adopt_mission(self, mission) -> None:
        if mission is None:
            return
        self.task_description = str(getattr(mission, "task_description", "") or "")
        tl = getattr(mission, "time_limit", None)
        self.time_limit = float(tl) if tl is not None else None
        self.adopt_owner(getattr(mission, "owner_position", None))

    def adopt_grounding(self, target_coord) -> None:
        xy = _xy(target_coord)
        if xy is not None:
            self.grounding_xy = xy

    def adopt_people_xy(self, points) -> None:
        if not points:
            self.people_xy = ()
            return
        cleaned = []
        for item in points:
            xy = _xy(item)
            if xy is not None:
                cleaned.append(xy)
        self.people_xy = tuple(cleaned)

    def adopt_reveal(self, reveal) -> None:
        if reveal is None:
            return
        xy = _xy(getattr(reveal, "hint_center", None))
        if xy is not None:
            self.hint_center = xy
        self.hint_radius = float(getattr(reveal, "hint_radius", 0.0) or 0.0)
        self.target_type = str(getattr(reveal, "target_type", "") or "") or None
        score = getattr(reveal, "grounding_score", None)
        if score is not None:
            self.grounding_score = float(score)

    def adopt_pick(self, result) -> None:
        """Store #13 pick/basket outcome for the delivery leg (#14)."""
        if result is None:
            return
        self.picked = bool(getattr(result, "ok", False))
        self.basket_occupied = bool(getattr(result, "basket_occupied", False))
        self.pick_attempts = int(getattr(result, "attempts", 0) or 0)
        self.pick_reason = str(getattr(result, "reason", "") or "")

    def search_xy(self) -> Optional[tuple]:
        """Reveal hint if present, else Stage 2 grounding target_coord."""
        if self.hint_center is not None:
            return self.hint_center
        return self.grounding_xy

    def search_tol(self, default: float = DEFAULT_GOAL_TOLERANCE) -> float:
        """Reached when inside the hint disc (or ``default`` if no radius)."""
        r = float(self.hint_radius or 0.0)
        if r > 0.0:
            return max(float(default), r)
        return float(default)

    def approach_goal(self, pose_xy) -> Optional[tuple]:
        center = self.search_xy()
        if center is None:
            return None
        if pose_xy is None:
            return center
        return search_area_approach_goal(center, self.hint_radius, pose_xy)


def close_enough_to_search(
    pose_xy,
    hint_center,
    hint_radius,
    goal_tol: float = DEFAULT_GOAL_TOLERANCE,
) -> bool:
    """True when the chassis is inside the disc or within ``goal_tol`` of its rim."""
    if pose_xy is None or hint_center is None:
        return False
    if len(pose_xy) < 2 or len(hint_center) < 2:
        return False
    dist = math.hypot(
        float(pose_xy[0]) - float(hint_center[0]),
        float(pose_xy[1]) - float(hint_center[1]),
    )
    radius = float(hint_radius or 0.0)
    return dist <= radius + float(goal_tol)


def nearfield_start_reason(
    reached: bool,
    pose_xy,
    hint_center,
    hint_radius,
    goal_tol: float = DEFAULT_GOAL_TOLERANCE,
) -> str:
    """Why near-field YOLO should start (or `skip_*` if it should not)."""
    if reached:
        return "nav_reached"
    if pose_xy is None or hint_center is None:
        return "skip_no_pose"
    if len(pose_xy) < 2 or len(hint_center) < 2:
        return "skip_no_pose"
    dist = math.hypot(
        float(pose_xy[0]) - float(hint_center[0]),
        float(pose_xy[1]) - float(hint_center[1]),
    )
    radius = float(hint_radius or 0.0)
    if dist <= radius:
        return "inside_disc"
    if dist <= radius + float(goal_tol):
        return "close_enough"
    return "skip_too_far"
