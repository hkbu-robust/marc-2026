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
