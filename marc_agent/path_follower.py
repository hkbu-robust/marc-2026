"""Pure path-following controllers lifted from the organizer demo.

No SDK client or I/O: callers publish ``(linear_x, angular_z)`` themselves.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class GoalSeekController:
    """Simple proportional (P) goal-seeking controller.

    ``compute(pose, goal, cruise)`` returns one step of ``(linear_x, angular_z, reached)``.
    When the heading error is large it rotates in place; when it is small it moves forward
    with heading correction, decelerating near the goal (unless ``cruise``).
    """

    GOAL_TOLERANCE = 0.7
    HEADING_TOLERANCE = math.pi / 4
    MAX_LINEAR = 0.8
    MAX_ANGULAR = 1.2
    K_LINEAR = 0.5
    K_ANGULAR = 0.8
    STEERING_TRACK = 0.38

    def compute(self, pose: tuple, goal: tuple, cruise: bool = False) -> tuple:
        """Compute one step of commands from the current pose and goal.

        Args:
            pose: (x, y, yaw_radians) world coordinates.
            goal: (gx, gy) target XY coordinates (m).
            cruise: if True, skip near-goal deceleration (pass-through waypoints).

        Returns:
            (linear_x, angular_z, reached: bool)
        """
        x, y, yaw = pose
        gx, gy = goal
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        if dist < self.GOAL_TOLERANCE:
            return 0.0, 0.0, True

        target_yaw = math.atan2(dy, dx)
        heading_err = _wrap_pi(target_yaw - yaw)

        angular = _clamp(self.K_ANGULAR * heading_err, -self.MAX_ANGULAR, self.MAX_ANGULAR)
        if abs(heading_err) > self.HEADING_TOLERANCE:
            linear = 0.0
        elif cruise:
            linear = self.MAX_LINEAR
        else:
            linear = _clamp(self.K_LINEAR * dist, 0.0, self.MAX_LINEAR)

        if linear > 1e-3:
            w_max = 1.8 * linear / self.STEERING_TRACK
            angular = _clamp(angular, -w_max, w_max)

        return linear, angular, False


class PathFollower:
    """Follow a world waypoint path with GoalSeekController."""

    WAYPOINT_TOLERANCE = 1.0

    def __init__(
        self,
        path: Sequence[Tuple[float, float]],
        controller: Optional[GoalSeekController] = None,
        passthrough: bool = False,
    ):
        self.path: List[Tuple[float, float]] = [(float(x), float(y)) for (x, y) in path]
        self.idx = 0
        self.ctrl = controller or GoalSeekController()
        self._passthrough = bool(passthrough)

    def compute(self, pose: tuple):
        """Compute one step of commands from the current pose.

        Returns:
            (linear_x, angular_z, done: bool, info: dict)
            info = {"idx", "total", "waypoint": (x, y)}
        """
        if not self.path:
            info: Dict = {"idx": 0, "total": 0, "waypoint": (0.0, 0.0)}
            return 0.0, 0.0, True, info

        x, y, _ = pose
        while self.idx < len(self.path) - 1:
            wx, wy = self.path[self.idx]
            if math.hypot(wx - x, wy - y) < self.WAYPOINT_TOLERANCE:
                self.idx += 1
            else:
                break

        wx, wy = self.path[self.idx]
        final = self.idx >= len(self.path) - 1
        cruise = (not final) or self._passthrough
        linear, angular, reached = self.ctrl.compute(pose, (wx, wy), cruise=cruise)
        done = final and reached
        info = {"idx": self.idx, "total": len(self.path), "waypoint": (wx, wy)}
        return linear, angular, done, info
