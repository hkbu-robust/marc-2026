"""Mock navigation agent -- stands in for the participant's Stage 2 navigation.

This module bundles the demo's navigation pieces (participants replace/extend them):
  - GoalSeekController / PathFollower -- P-control that returns ``(linear_x, angular_z)``
    scalars to publish with ``client.send_cmd_vel(...)``.
  - SpeedGovernor -- IMU + Ackermann cap on those scalars (rollover). Lidar speed
    lives in ``marc_agent.lidar_safety.advise``.
  - OccupancyGridPlanner -- D* Lite over the runtime occupancy grid
    (``MARCClient.get_occupancy_map()``), with A* as a fallback.
  - lidar corridor helpers re-exported from ``marc_agent.lidar_safety``.
  - run_nav_leg -- thin wrapper around ``marc_agent.navigation_runtime.Stage2Navigator``.
  - load_obstacles / obstacles_excluding_goal -- optional yaml discs for debugging
    only; the production navigator in ``participant_app.py`` never reads them.
"""

from __future__ import annotations

import heapq
import logging
import math
import os
import sys

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marc_agent.lidar_safety import (  # noqa: E402
    advise,
    pointcloud2_xyz,
)

log = logging.getLogger("participant_app")

SQRT2 = math.sqrt(2.0)
_INF = 1e18

# Overlay cells not restamped for this long become free (moving people/ghosts).
OVERLAY_TTL_S = 30.0


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

    # Reached test -- XY distance (m)
    GOAL_TOLERANCE = 0.7
    # Above this heading error, rotate in place (linear=0 -> pure wz -> chassis X-pattern
    # in-place spin); below it, move forward with heading correction. The chassis can spin in
    # place (X-pattern steering + opposite left/right wheels), so large turns (e.g. a goal
    # behind the robot) spin in place first, then drive forward -- avoids the Ackermann-only
    # "turn around and stall" problem.
    HEADING_TOLERANCE = math.pi / 4   # ~45 deg: beyond this, rotate in place then advance

    # Speed / turn caps. Large turns are handled in place (see HEADING_TOLERANCE), so forward
    # motion rarely needs a hard swerve; the Ackermann w-limit below guarantees a tight arc.
    MAX_LINEAR = 1.0           # m/s; SpeedGovernor cuts this on corners / tilt
    MAX_ANGULAR = 1.2          # rad/s
    K_LINEAR = 0.5
    K_ANGULAR = 0.8
    # 4-wheel-steer track (m). The chassis stops (0) when 2*v <= w*T (a turn too tight for the
    # speed), so the controller caps w within 2v/T to always emit a valid (forward + steer)
    # command. Same value as the platform's chassis steering track.
    STEERING_TRACK = 0.38

    def compute(self, pose: tuple, goal: tuple, cruise: bool = False) -> tuple:
        """Compute one step of commands from the current pose and goal.

        Args:
            pose: (x, y, yaw_radians) -- world coordinates (client.get_world_pose()).
            goal: (gx, gy)           -- target XY coordinates (m).
            cruise: if True, skip the near-goal stop and proportional deceleration
                (K_LINEAR*dist) and cruise at MAX_LINEAR -- for intermediate
                pass-through waypoints; only the final goal (cruise=False) stops
                and decelerates on approach.

        Returns:
            (linear_x, angular_z, reached: bool)
        """
        x, y, yaw = pose
        gx, gy = goal
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)

        # Intermediate (cruise) waypoints must keep moving through the 0.40–0.70 m
        # band so PathFollower can skip at WAYPOINT_TOLERANCE. Stopping here deadlocks
        # at D* corners. The final goal still stops at GOAL_TOLERANCE.
        if dist < self.GOAL_TOLERANCE and not cruise:
            return 0.0, 0.0, True

        target_yaw = math.atan2(dy, dx)
        heading_err = _wrap_pi(target_yaw - yaw)

        angular = _clamp(self.K_ANGULAR * heading_err,
                         -self.MAX_ANGULAR, self.MAX_ANGULAR)
        if abs(heading_err) > self.HEADING_TOLERANCE:
            linear = 0.0                       # rotate in place
        elif cruise:
            linear = self.MAX_LINEAR           # pass-through waypoint: no deceleration
        else:
            linear = _clamp(self.K_LINEAR * dist, 0.0, self.MAX_LINEAR)  # decelerate near goal

        # Ackermann validity: a turn too tight for the speed (2v <= w*T) makes the chassis
        # stop. Cap w within 2v/T (safety 0.9) so even a goal behind the robot keeps moving
        # in the tightest valid arc (this chassis must move forward to steer).
        if linear > 1e-3:
            w_max = 1.8 * linear / self.STEERING_TRACK
            angular = _clamp(angular, -w_max, w_max)

        return linear, angular, False


class PathFollower:
    """Controller that follows a waypoint path in sequence.

    Takes the world waypoint list produced by ``OccupancyGridPlanner.plan()`` and,
    every step, computes ``GoalSeekController`` commands from the current pose toward
    the active waypoint. When it gets close enough to the active waypoint it advances
    to the next, and it is done once it reaches the last waypoint (= goal).
    """

    # Distance to consider an intermediate waypoint passed (m). Tighter than the
    # 0.7 m goal stop so D* corners are driven, not skipped onto the curb.
    WAYPOINT_TOLERANCE = 0.40

    def __init__(self, path, controller: GoalSeekController = None,
                 passthrough: bool = False):
        self.path = [(float(x), float(y)) for (x, y) in path]
        self.idx = 0
        self.ctrl = controller or GoalSeekController()
        # passthrough=True: the final point is also a pass-through (no deceleration) --
        # used when this goal is itself a waypoint, not a stop point.
        self._passthrough = bool(passthrough)

    def compute(self, pose: tuple):
        """Compute one step of commands from the current pose.

        Returns:
            (linear_x, angular_z, done: bool, info: dict)
            info = {"idx", "total", "waypoint": (x, y)}
        """
        x, y, _ = pose
        # Skip intermediate waypoints that have been reached (keep the last one).
        while self.idx < len(self.path) - 1:
            wx, wy = self.path[self.idx]
            if math.hypot(wx - x, wy - y) < self.WAYPOINT_TOLERANCE:
                self.idx += 1
            else:
                break

        wx, wy = self.path[self.idx]
        # Intermediate waypoints cruise (full speed); the final point decelerates -- unless
        # passthrough, where the final point also cruises.
        final = self.idx >= len(self.path) - 1
        cruise = (not final) or self._passthrough
        linear, angular, reached = self.ctrl.compute(pose, (wx, wy), cruise=cruise)
        done = final and reached
        info = {"idx": self.idx, "total": len(self.path), "waypoint": (wx, wy)}
        return linear, angular, done, info


# ---------------------------------------------------------------------------
# Speed governor: IMU dynamics. Lidar slow/stop is lidar_safety.advise.
# ---------------------------------------------------------------------------

_G = 9.81


def _imu_xyz(part):
    if part is None:
        return 0.0, 0.0, 0.0
    return (
        float(getattr(part, "x", 0.0) or 0.0),
        float(getattr(part, "y", 0.0) or 0.0),
        float(getattr(part, "z", 0.0) or 0.0),
    )


def _quat_roll_pitch(q):
    """Body roll, pitch (rad) from a geometry_msgs-style quaternion. Zeros -> level."""
    if q is None:
        return 0.0, 0.0
    x, y, z = _imu_xyz(q)
    w = float(getattr(q, "w", 1.0) or 0.0)
    if (x * x + y * y + z * z + w * w) < 1e-12:
        return 0.0, 0.0
    sinp = _clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(sinp)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return roll, pitch


def _az_jolt(az):
    """Vertical accel away from rest (0 or ±g), so gravity-on and gravity-off IMUs both work."""
    az = float(az)
    return min(abs(az), abs(az - _G), abs(az + _G))


def _lerp_cap(x, x0, x1, y0, y1):
    """x<=x0 -> y0, x>=x1 -> y1, linear in between."""
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


class SpeedGovernor:
    """Cap PathFollower cmd_vel so straights can cruise and corners do not flip the chassis.

    ``limit(linear, angular, imu)`` never raises ``linear`` above the request. Missing
    IMU still caps from commanded yaw rate. After cutting v, recaps w with the same
    Ackermann rule as ``GoalSeekController`` so the chassis does not stall on ``2v <= w*T``.
    Lidar slow/stop is ``lidar_safety.advise``, not this governor.
    """

    STEERING_TRACK = GoalSeekController.STEERING_TRACK
    A_LAT_MAX = 0.65          # m/s^2; v <= A_LAT / |w|
    ROLL_START = 0.08         # rad (~4.5 deg) -- begin cutting
    ROLL_HARD = 0.30          # rad (~17 deg)
    ROLL_RATE_START = 0.50    # rad/s
    ROLL_RATE_HARD = 2.00
    AY_START = 1.00           # m/s^2 lateral
    AY_HARD = 3.50
    AY_SPIKE = 4.50
    AZ_SPIKE = 4.00           # m/s^2 jolt from rest
    V_FLOOR = 0.20            # m/s -- creep over a lip / after a bump
    V_SPIKE = 0.22
    SPIKE_HOLD_S = 0.80
    TAU_DOWN_S = 0.08
    TAU_UP_S = 0.12

    def __init__(self):
        self._v_filt = None
        self._spike_t = 0.0
        self._spike_v = self.V_SPIKE

    def limit(self, linear, angular, imu=None, xyz=None, dt=None):
        """Return ``(linear_x, angular_z)`` capped for IMU stability. ``xyz`` is ignored."""
        requested = float(linear)
        ang = float(angular)
        dt = 0.10 if dt is None else max(1e-3, float(dt))

        if requested <= 1e-6:
            self._spike_t = max(0.0, self._spike_t - dt)
            return 0.0, ang

        v_cap = abs(requested)
        w_used = abs(ang)
        roll = pitch = wx = ay = az = 0.0
        if imu is not None:
            wx, _, wz = _imu_xyz(getattr(imu, "angular_velocity", None))
            _, ay, az = _imu_xyz(getattr(imu, "linear_acceleration", None))
            roll, pitch = _quat_roll_pitch(getattr(imu, "orientation", None))
            w_used = max(w_used, abs(wz))

        v_cap = min(v_cap, self.A_LAT_MAX / (w_used + 1e-3))
        tilt = max(abs(roll), abs(pitch))
        v_cap = min(
            v_cap,
            _lerp_cap(tilt, self.ROLL_START, self.ROLL_HARD, abs(requested), self.V_FLOOR),
            _lerp_cap(abs(wx), self.ROLL_RATE_START, self.ROLL_RATE_HARD,
                      abs(requested), self.V_FLOOR),
            _lerp_cap(abs(ay), self.AY_START, self.AY_HARD, abs(requested), self.V_FLOOR),
        )

        if imu is not None and (_az_jolt(az) >= self.AZ_SPIKE or abs(ay) >= self.AY_SPIKE):
            self._spike_v = self.V_SPIKE
            self._spike_t = self.SPIKE_HOLD_S

        if self._spike_t > 0.0:
            v_cap = min(v_cap, self._spike_v)
            self._spike_t = max(0.0, self._spike_t - dt)

        if self._v_filt is None:
            self._v_filt = v_cap
        else:
            tau = self.TAU_DOWN_S if v_cap < self._v_filt else self.TAU_UP_S
            alpha = dt / (tau + dt)
            self._v_filt += alpha * (v_cap - self._v_filt)

        out_lin = min(requested, self._v_filt)
        out_ang = ang
        if out_lin > 1e-3:
            w_max = 1.8 * out_lin / self.STEERING_TRACK
            out_ang = _clamp(ang, -w_max, w_max)
        return out_lin, out_ang


# Lidar corridor lives in marc_agent.lidar_safety; re-exported above.
# Frontal recovery: command zero, then replan (no reverse/open-loop).
RECOVERY_LINEAR_X = 0.0
RECOVERY_HOLD_S = 0.50
GOAL_RETARGET_M = 0.50


def recovery_cmd(frontal: bool) -> tuple:
    """cmd_vel on a frontal lidar hit: zero linear and yaw, then replan."""
    return 0.0, 0.0


def resolve_xy(goal):
    """``(x, y)`` from a tuple or zero-arg callable; None if missing."""
    if goal is None:
        return None
    if callable(goal):
        goal = goal()
    if goal is None or len(goal) < 2:
        return None
    return (float(goal[0]), float(goal[1]))


def resolve_tol(tol, default=0.7) -> float:
    if callable(tol):
        tol = tol()
    if tol is None:
        return float(default)
    return float(tol)


def run_nav_leg(
    client,
    goal,
    deadline,
    tol=0.7,
    inflation_m=0.55,
    exclude_m=0.6,
    cruise=False,
    extra_obstacles=None,
    dt=0.10,
    stuck_time_s=8.0,
    stuck_move_m=0.05,
    replan_cap=50,
):
    """Follow occupancy D* Lite to ``goal`` until inside ``tol``. True if reached.

    Thin wrapper around ``Stage2Navigator`` so older demo call sites keep working.
    """
    from marc_agent.navigation_runtime import Stage2Navigator
    nav = Stage2Navigator(
        client,
        dt=dt,
        inflation_m=inflation_m,
        exclude_m=exclude_m,
        stuck_time_s=stuck_time_s,
        stuck_move_m=stuck_move_m,
        replan_cap=replan_cap,
        cruise=cruise,
    )
    result = nav.navigate_to(
        goal, deadline, leg="search", tol=tol, extra_obstacles=extra_obstacles)
    return result.reached


def inflate_cells(inflation_m, res) -> int:
    """Cells to erode so a positive inflation never rounds to zero pad."""
    if float(inflation_m) <= 0.0 or not res:
        return 0
    return max(1, int(math.ceil(float(inflation_m) / float(res) - 1e-9)))


def path_is_blocked(
    planner, waypoints, extra_obstacles=None, inflation_m=0.25, ignore_overlay=False,
):
    """True if any remaining polyline segment fails Bresenham LOS on the inflated grid.

    ``ignore_overlay=True`` checks occupancy only (100 / unknown). Use that to reject
    off-road cuts while still keeping a D* detour whose start chord hits lidar overlay.
    """
    if planner is None or not planner.ready or waypoints is None or len(waypoints) < 2:
        return False
    cells = []
    for wp in waypoints:
        r, c = planner._w2c(float(wp[0]), float(wp[1]))
        if not planner._in_bounds(r, c):
            return True
        cells.append((r, c))
    infl_c = inflate_cells(inflation_m, planner._res)
    # Crop must outrun erosion: _erode treats the crop border as blocked, so
    # pad < infl_c false-blocks every chord on a 5 cm grid (infl_c=11 at 0.55 m).
    pad = max(4, infl_c + 2)
    rmin = max(0, min(r for r, _ in cells) - pad)
    rmax = min(planner._h, max(r for r, _ in cells) + pad + 1)
    cmin = max(0, min(c for _, c in cells) - pad)
    cmax = min(planner._w, max(c for _, c in cells) + pad + 1)
    sub = planner._grid[rmin:rmax, cmin:cmax]
    trav = OccupancyGridPlanner._traversable(sub)
    if not ignore_overlay and planner._overlay is not None:
        trav = np.logical_and(
            trav, np.logical_not(planner._overlay[rmin:rmax, cmin:cmax]))
    if extra_obstacles:
        planner._block_obstacles(trav, extra_obstacles, rmin, cmin)
    if infl_c > 0:
        trav = OccupancyGridPlanner._erode(trav, infl_c)
    H, W = trav.shape
    local = [(r - rmin, c - cmin) for r, c in cells]
    for i in range(len(local) - 1):
        a, b = local[i], local[i + 1]
        if not (0 <= a[0] < H and 0 <= a[1] < W and 0 <= b[0] < H and 0 <= b[1] < W):
            return True
        if not OccupancyGridPlanner._los(trav, a, b):
            return True
    return False


# ---------------------------------------------------------------------------
# Occupancy-grid D* Lite (A* fallback).
# ---------------------------------------------------------------------------

class _DStarLite:
    """Koenig/Likhachev D* Lite on an 8-connected boolean grid (search from goal)."""

    def __init__(self, trav, goal):
        self.trav = np.asarray(trav, dtype=bool)
        self.H, self.W = self.trav.shape
        self.goal = (int(goal[0]), int(goal[1]))
        self.start = self.goal
        self.km = 0.0
        self.g = {}
        self.rhs = {}
        self.U = []
        self._stamp = {}
        self._n = 0
        self.rhs[self.goal] = 0.0
        self._push(self.goal)

    def _g(self, s):
        return self.g.get(s, _INF)

    def _rhs(self, s):
        return self.rhs.get(s, _INF)

    @staticmethod
    def _h(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _key(self, s):
        val = min(self._g(s), self._rhs(s))
        return (val + self._h(self.start, s) + self.km, val)

    def _in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def _succ(self, s):
        r, c = s
        if not (self._in_bounds(r, c) and self.trav[r, c]):
            return
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not self._in_bounds(nr, nc) or not self.trav[nr, nc]:
                    continue
                if dr != 0 and dc != 0 and not (self.trav[r + dr, c] and self.trav[r, c + dc]):
                    continue
                step = SQRT2 if (dr and dc) else 1.0
                yield (nr, nc), step

    def _adjacent(self, s):
        r, c = s
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc):
                    yield (nr, nc)

    def _push(self, s):
        self._n += 1
        k1, k2 = self._key(s)
        self._stamp[s] = self._n
        heapq.heappush(self.U, (k1, k2, self._n, s))

    def _invalidate(self, s):
        self._stamp[s] = 0

    def _pop_valid(self):
        while self.U:
            k1, k2, n, s = heapq.heappop(self.U)
            if self._stamp.get(s) == n:
                return k1, k2, s
        return None

    def _top_key(self):
        while self.U:
            k1, k2, n, s = self.U[0]
            if self._stamp.get(s) == n:
                return (k1, k2)
            heapq.heappop(self.U)
        return (_INF, _INF)

    def _update_vertex(self, u):
        if not self._in_bounds(*u):
            return
        if u != self.goal:
            if not self.trav[u]:
                self.rhs[u] = _INF
            else:
                best = _INF
                for n, cost in self._succ(u):
                    val = cost + self._g(n)
                    if val < best:
                        best = val
                self.rhs[u] = best
        else:
            self.rhs[u] = 0.0
        self._invalidate(u)
        if abs(self._g(u) - self._rhs(u)) > 1e-6:
            self._push(u)

    def move_start(self, start):
        start = (int(start[0]), int(start[1]))
        if start == self.start:
            return
        self.km += self._h(self.start, start)
        self.start = start

    def update_cells(self, trav, changed_rc):
        self.trav = np.asarray(trav, dtype=bool)
        affected = set()
        for rc in changed_rc:
            s = (int(rc[0]), int(rc[1]))
            affected.add(s)
            affected.update(self._adjacent(s))
        for s in affected:
            self._update_vertex(s)

    def compute(self, start):
        start = (int(start[0]), int(start[1]))
        self.move_start(start)
        limit = max(64, self.H * self.W * 8)
        for _ in range(limit):
            tk = self._top_key()
            sk = self._key(self.start)
            if not (tk < sk or abs(self._rhs(self.start) - self._g(self.start)) > 1e-6):
                return True
            popped = self._pop_valid()
            if popped is None:
                break
            k_old1, k_old2, u = popped
            k_new = self._key(u)
            if (k_old1, k_old2) < k_new:
                self._push(u)
            elif self._g(u) - self._rhs(u) > 1e-6:
                self.g[u] = self._rhs(u)
                for n in self._adjacent(u):
                    self._update_vertex(n)
            else:
                self.g[u] = _INF
                self._update_vertex(u)
                for n in self._adjacent(u):
                    self._update_vertex(n)
        return abs(self._rhs(self.start) - self._g(self.start)) <= 1e-6

    def extract_path(self, start):
        start = (int(start[0]), int(start[1]))
        if self._rhs(start) >= _INF * 0.5:
            return None
        if start == self.goal:
            return [start]
        path = [start]
        cur = start
        seen = {cur}
        limit = self.H * self.W + 2
        while cur != self.goal:
            best, best_v = None, _INF
            for n, cost in self._succ(cur):
                val = cost + self._g(n)
                if val < best_v:
                    best_v = val
                    best = n
            if best is None or best in seen:
                return None
            cur = best
            seen.add(cur)
            path.append(cur)
            if len(path) > limit:
                return None
        return path


class OccupancyGridPlanner:
    """D* Lite planner over an occupancy grid (A* fallback).

    Update the grid with ``update_from_msg()`` / ``update_from_array()`` and get a
    list of world-coordinate waypoints with ``plan()``. If there is no grid,
    ``ready == False``.
    """

    def __init__(self):
        self._grid = None        # int8 (H, W), row0 = bottom
        self._res = None         # m/cell
        self._ox = None          # origin x (world)
        self._oy = None          # origin y (world)
        self._h = 0
        self._w = 0
        self._dstar = None
        self._crop = None        # (rmin, rmax, cmin, cmax)
        self._last_trav = None
        self._plan_goal_xy = None
        self._infl_c = None
        self._dstar_goal = None
        self._overlay = None     # bool (H, W), True = lidar-blocked
        self._overlay_t = None   # float64 last restamp time (sim s)
        self.last_method = None  # "dstar" | "astar"
        self.last_status = None  # "ok" | "no_map" | "no_path"
        self.last_start_trav = None  # bool | None after last plan
        self.last_goal_trav = None

    def update_from_msg(self, msg) -> bool:
        """Update the grid from a nav_msgs/OccupancyGrid message."""
        info = msg.info
        w = int(info.width)
        h = int(info.height)
        if w == 0 or h == 0:
            return False
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        return self.update_from_array(
            grid, float(info.resolution),
            float(info.origin.position.x), float(info.origin.position.y))

    def update_from_array(self, grid, res, ox, oy) -> bool:
        """Update the grid from a numpy occupancy array (0=free, 100=occ, -1=unknown)."""
        arr = np.asarray(grid, dtype=np.int8)
        if arr.ndim != 2 or arr.size == 0:
            return False
        map_changed = (
            self._grid is None
            or arr.shape != getattr(self._grid, "shape", ())
            or self._res != float(res)
            or self._ox != float(ox)
            or self._oy != float(oy)
        )
        self._grid = arr
        self._res = float(res)
        self._ox = float(ox)
        self._oy = float(oy)
        self._h, self._w = arr.shape
        if map_changed or self._overlay is None or self._overlay.shape != arr.shape:
            self._overlay = np.zeros(arr.shape, dtype=bool)
            self._overlay_t = np.zeros(arr.shape, dtype=np.float64)
        return True

    @property
    def ready(self) -> bool:
        return self._grid is not None

    @property
    def resolution(self):
        return self._res

    def _w2c(self, wx, wy):
        """world (x, y) -> cell (row, col)."""
        col = int((wx - self._ox) / self._res)
        row = int((wy - self._oy) / self._res)
        return row, col

    def _c2w(self, row, col):
        """cell (row, col) -> world (x, y) (cell center)."""
        wx = self._ox + (col + 0.5) * self._res
        wy = self._oy + (row + 0.5) * self._res
        return wx, wy

    def cell_free(self, wx, wy, inflation_m=0.0) -> bool:
        """True when occupancy is 0 at (wx, wy) with a square inflation radius.

        Lidar overlay cells are blocked too so a tumbler behind a car is not a
        legal sit-on-the-bumper goal. Occupancy 100 is never punched.
        """
        if not self.ready:
            return False
        row, col = self._w2c(wx, wy)
        if not self._in_bounds(row, col):
            return False
        rad = inflate_cells(inflation_m, self._res)
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc):
                    return False
                if int(self._grid[nr, nc]) != 0:
                    return False
                if self._overlay is not None and bool(self._overlay[nr, nc]):
                    return False
        return True

    def nearest_free_xy(self, xy, inflation_m=0.25, max_radius_m=25.0):
        """World XY of the nearest inflated-free cell, or None if none exist."""
        if not self.ready or xy is None or len(xy) < 2:
            return None
        wx, wy = float(xy[0]), float(xy[1])
        if self.cell_free(wx, wy, inflation_m):
            return (wx, wy)
        r0, c0 = self._w2c(wx, wy)
        cell_m = float(self._res) if self._res else 0.05
        max_rad = int(max(1, round(float(max_radius_m) / max(cell_m, 1e-6))))
        for rad in range(1, max_rad + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    nr, nc = r0 + dr, c0 + dc
                    if not self._in_bounds(nr, nc):
                        continue
                    gx, gy = self._c2w(nr, nc)
                    if self.cell_free(gx, gy, inflation_m):
                        return (gx, gy)
        return None

    def free_approach_xy(self, start_xy, goal_xy, inflation_m=0.25):
        """Last free cell on the start→goal ray, else nearest free to the goal.

        Stops before an occupied or unknown blob so the chassis does not stand
        on the far side of a parked car. Does not punch occupancy holes.
        """
        if not self.ready or start_xy is None or goal_xy is None:
            return None
        if len(start_xy) < 2 or len(goal_xy) < 2:
            return None
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        dx, dy = gx - sx, gy - sy
        dist = math.hypot(dx, dy)
        step = max(float(self._res) if self._res else 0.05, 0.05)
        last_free = None
        n = max(1, int(math.ceil(dist / step)))
        for i in range(n + 1):
            t = i / n
            x = sx + t * dx
            y = sy + t * dy
            if self.cell_free(x, y, inflation_m):
                last_free = (x, y)
                continue
            if last_free is not None:
                return last_free
            break
        if last_free is not None:
            return last_free
        return self.nearest_free_xy(goal_xy, inflation_m=inflation_m)

    def plan(self, start_xy, goal_xy, inflation_m=0.25, margin_m=12.0,
             snap_m=1.5, extra_obstacles=None, snap_goal_m=None):
        """Return a detour path from start_xy -> goal_xy as a list of world waypoints.

        Args:
            start_xy: (x, y) current robot position (world).
            goal_xy:  (x, y) target position (world).
            inflation_m: erode the free area by the robot radius to build a path that
                         stays away from the road boundary (avoids step stalls).
            margin_m: search margin outside the start/goal bounding box (crop for planning speed).
            snap_m:  maximum radius to correct a start cell to the nearest free
                     cell when it is not traversable.
            extra_obstacles: [(x, y, radius_m), ...] extra discs carved for this
                     call only (tests / yaml). Live lidar does not stamp the grid.
            snap_goal_m: maximum radius to snap a blocked goal onto free road.
                     Defaults to ``snap_m``. Off-road objects (parking) need a
                     larger value so the path ends at the street, not inside the lot.

        Returns:
            [(x, y), ...] path waypoints. If the goal cell is traversable the last one is
            exactly goal_xy; if it is blocked (a vehicle etc.), up to the free cell nearest
            the goal. None on failure.
        """
        self.last_method = None
        if not self.ready:
            self.last_status = "no_map"
            return None
        self.last_status = "no_path"

        sr, sc = self._w2c(*start_xy)
        gr, gc = self._w2c(*goal_xy)
        if not (self._in_bounds(sr, sc) and self._in_bounds(gr, gc)):
            return None

        margin_c = int(math.ceil(margin_m / self._res))
        rmin = max(0, min(sr, gr) - margin_c)
        rmax = min(self._h, max(sr, gr) + margin_c + 1)
        cmin = max(0, min(sc, gc) - margin_c)
        cmax = min(self._w, max(sc, gc) + margin_c + 1)
        infl_c = inflate_cells(inflation_m, self._res)

        reuse = (
            self._dstar is not None
            and self._crop is not None
            and self._infl_c == infl_c
            and self._plan_goal_xy is not None
            and math.hypot(goal_xy[0] - self._plan_goal_xy[0],
                           goal_xy[1] - self._plan_goal_xy[1]) < 0.05
        )
        if reuse:
            ormin, ormax, ocmin, ocmax = self._crop
            if ormin <= sr < ormax and ocmin <= sc < ocmax:
                rmin, rmax, cmin, cmax = self._crop
            else:
                rmin, rmax = min(rmin, ormin), max(rmax, ormax)
                cmin, cmax = min(cmin, ocmin), max(cmax, ocmax)
                reuse = False

        sub = self._grid[rmin:rmax, cmin:cmax]
        trav = self._traversable(sub)
        if self._overlay is not None:
            trav = np.logical_and(
                trav, np.logical_not(self._overlay[rmin:rmax, cmin:cmax]))
        if extra_obstacles:
            self._block_obstacles(trav, extra_obstacles, rmin, cmin)
        if infl_c > 0:
            trav = self._erode(trav, infl_c)

        ls, lg = (sr - rmin, sc - cmin), (gr - rmin, gc - cmin)
        H, W = trav.shape
        goal_free = (0 <= lg[0] < H and 0 <= lg[1] < W and bool(trav[lg[0], lg[1]]))
        self.last_start_trav = (
            0 <= ls[0] < H and 0 <= ls[1] < W and bool(trav[ls[0], ls[1]])
        )
        self.last_goal_trav = goal_free
        cell_m = float(self._res) if self._res else 0.05
        goal_snap = float(snap_m if snap_goal_m is None else snap_goal_m)
        ls = self._snap(trav, ls, snap_m, cell_m)
        lg = self._snap(trav, lg, goal_snap, cell_m)
        if ls is None or lg is None:
            self._dstar = None
            return None

        if reuse and self._dstar_goal != lg:
            reuse = False
        if reuse and self._last_trav is not None and self._last_trav.shape != trav.shape:
            reuse = False

        cells = None
        if reuse:
            changed = np.argwhere(trav != self._last_trav)
            self._dstar.update_cells(trav, changed)
            if self._dstar.compute(ls):
                cells = self._dstar.extract_path(ls)
            if cells:
                self.last_method = "dstar"
        else:
            self._dstar = _DStarLite(trav, lg)
            if self._dstar.compute(ls):
                cells = self._dstar.extract_path(ls)
            if cells:
                self.last_method = "dstar"

        if not cells:
            cells = self._astar(trav, ls, lg)
            if cells:
                self.last_method = "astar"
                self._dstar = None

        self._last_trav = trav.copy()
        self._crop = (rmin, rmax, cmin, cmax)
        self._plan_goal_xy = (float(goal_xy[0]), float(goal_xy[1]))
        self._infl_c = infl_c
        self._dstar_goal = lg

        if not cells:
            self.last_status = "no_path"
            return None

        self.last_status = "ok"
        cells = self._simplify(trav, cells)
        path = [self._c2w(r + rmin, c + cmin) for (r, c) in cells]

        if path:
            if goal_free:
                path[-1] = (float(goal_xy[0]), float(goal_xy[1]))
            join = path[0]
            near_start = math.hypot(
                join[0] - float(start_xy[0]), join[1] - float(start_xy[1])
            ) < max(1.5 * cell_m, 0.15)
            if near_start and len(path) > 1:
                path = path[1:]
        return path

    def _disc_mask(self, wx, wy, rad):
        """Slice + boolean disc on the full grid, or None if off-map."""
        if not self.ready or self._res is None:
            return None
        r, c = self._w2c(wx, wy)
        rad_c = int(round(float(rad) / self._res))
        if rad_c <= 0:
            return None
        H, W = self._h, self._w
        if r + rad_c < 0 or r - rad_c >= H or c + rad_c < 0 or c - rad_c >= W:
            return None
        r0, r1 = max(0, r - rad_c), min(H, r + rad_c + 1)
        c0, c1 = max(0, c - rad_c), min(W, c + rad_c + 1)
        yy, xx = np.ogrid[r0:r1, c0:c1]
        disc = (yy - r) ** 2 + (xx - c) ** 2 <= rad_c ** 2
        return r0, r1, c0, c1, disc

    def stamp_overlay(self, obstacles, now_s=0.0):
        """Union-stamp world discs onto the persistent lidar overlay."""
        if not self.ready or self._overlay is None:
            return
        now = float(now_s)
        for item in obstacles or []:
            wx, wy, rad = float(item[0]), float(item[1]), float(item[2])
            sl = self._disc_mask(wx, wy, rad)
            if sl is None:
                continue
            r0, r1, c0, c1, disc = sl
            self._overlay[r0:r1, c0:c1][disc] = True
            self._overlay_t[r0:r1, c0:c1][disc] = now

    def replace_overlay(self, obstacles, now_s=0.0):
        """Redraw the lidar layer from this tick's discs. Does not punch occupancy."""
        if not self.ready or self._overlay is None:
            return
        self._overlay[:] = False
        self._overlay_t[:] = 0.0
        self.stamp_overlay(obstacles, now_s=now_s)

    def expire_overlay(self, now_s=0.0):
        """Free overlay cells not restamped within OVERLAY_TTL_S."""
        if self._overlay is None or self._overlay_t is None:
            return
        stale = self._overlay & ((float(now_s) - self._overlay_t) > OVERLAY_TTL_S)
        self._overlay[stale] = False

    def clear_goal_hole(self, goal_xy, exclude_m=0.6):
        """Drop lidar overlay near the goal. Never edits occupancy cars or unknown."""
        if not self.ready or self._overlay is None or exclude_m <= 0:
            return
        sl = self._disc_mask(float(goal_xy[0]), float(goal_xy[1]), float(exclude_m))
        if sl is None:
            return
        r0, r1, c0, c1, disc = sl
        self._overlay[r0:r1, c0:c1][disc] = False

    def clear_start_hole(self, start_xy, exclude_m=0.20):
        """Free only the chassis cell so D* can start; do not punch a bumper-sized hole."""
        cell = float(self._res) if self._res else 0.05
        self.clear_goal_hole(start_xy, exclude_m=max(float(exclude_m), cell * 0.51))

    def _block_obstacles(self, trav, obstacles, rmin, cmin):
        """Mark each (x, y, radius) disc from extra_obstacles as not traversable in trav."""
        H, W = trav.shape
        for (wx, wy, rad) in obstacles:
            r, c = self._w2c(wx, wy)
            lr, lc = r - rmin, c - cmin
            rad_c = int(round(float(rad) / self._res))
            if rad_c <= 0:
                continue
            if lr + rad_c < 0 or lr - rad_c >= H or lc + rad_c < 0 or lc - rad_c >= W:
                continue  # outside the crop
            r0, r1 = max(0, lr - rad_c), min(H, lr + rad_c + 1)
            c0, c1 = max(0, lc - rad_c), min(W, lc + rad_c + 1)
            yy, xx = np.ogrid[r0:r1, c0:c1]
            disc = (yy - lr) ** 2 + (xx - lc) ** 2 <= rad_c ** 2
            trav[r0:r1, c0:c1][disc] = False

    def _in_bounds(self, r, c):
        return 0 <= r < self._h and 0 <= c < self._w

    @staticmethod
    def _traversable(sub):
        """Only free road (0) is open. Occupied (100) and unknown (-1) are blocked."""
        return np.asarray(sub) == 0

    @staticmethod
    def _erode(mask, iters):
        """8-neighbor erosion, iters times. Border cells are treated as blocked (protected by the crop margin)."""
        g = mask
        for _ in range(iters):
            e = np.zeros_like(g)
            e[1:-1, 1:-1] = (
                g[1:-1, 1:-1] & g[:-2, 1:-1] & g[2:, 1:-1]
                & g[1:-1, :-2] & g[1:-1, 2:]
                & g[:-2, :-2] & g[:-2, 2:] & g[2:, :-2] & g[2:, 2:]
            )
            g = e
        return g

    @staticmethod
    def _snap(trav, cell, snap_m, res_cells=None):
        """If the cell is not traversable, return the nearest free cell via a spiral search."""
        r, c = cell
        H, W = trav.shape
        if not (0 <= r < H and 0 <= c < W):
            return None
        if trav[r, c]:
            return cell
        # snap_m is in meters; use the occupancy resolution, not a hardcoded 5 cm cell.
        cell_m = float(res_cells) if res_cells else 0.05
        max_rad = int(max(1, round(snap_m / max(cell_m, 1e-6))))
        for rad in range(1, max_rad + 1):
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and trav[nr, nc]:
                        return (nr, nc)
        return None

    @staticmethod
    def _astar(grid, start, goal):
        """8-connected A*. A cell list (start..goal) or None."""
        H, W = grid.shape

        def h(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        openh = [(h(start, goal), 0.0, start)]
        gscore = {start: 0.0}
        came = {}
        while openh:
            _, gc, cur = heapq.heappop(openh)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                return path[::-1]
            if gc > gscore.get(cur, 1e18):
                continue
            r, c = cur
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < H and 0 <= nc < W) or not grid[nr, nc]:
                        continue
                    # Diagonal moves only when both orthogonal cells are free (prevents corner cutting)
                    if dr != 0 and dc != 0 and not (grid[r + dr, c] and grid[r, c + dc]):
                        continue
                    step = SQRT2 if (dr and dc) else 1.0
                    ng = gc + step
                    nb = (nr, nc)
                    if ng < gscore.get(nb, 1e18):
                        gscore[nb] = ng
                        came[nb] = cur
                        heapq.heappush(openh, (ng + h(nb, goal), ng, nb))
        return None

    @classmethod
    def _simplify(cls, grid, path):
        """Merge straight segments via line-of-sight to reduce the number of waypoints."""
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        n = len(path)
        while i < n - 1:
            j = n - 1
            while j > i + 1 and not cls._los(grid, path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    @staticmethod
    def _los(grid, a, b):
        """Check whether the straight line a->b is entirely free, using Bresenham."""
        r0, c0 = a
        r1, c1 = b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if not grid[r, c]:
                return False
            if (r, c) == (r1, c1):
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc


# ---------------------------------------------------------------------------
# Demo obstacles + Stage 2 answer, read from mock_demo_data.yaml
# (data pre-extracted by tools/mock_data_builder instead of the scenario).
# ---------------------------------------------------------------------------

def _default_data_path() -> str:
    """demo/mock_demo_data.yaml next to this module."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "mock_demo_data.yaml")


def load_mock_data(path: str | None = None) -> dict:
    """Load the pre-extracted demo data (obstacles + Stage 2 answer)."""
    path = path or os.environ.get("MOCKUP_DEMO_DATA") or _default_data_path()
    if not os.path.exists(path):
        log.warning("[mock-data] not found: %s -- run tools/mock_data_builder/gen_mock_demo_data.py", path)
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_obstacles(path: str | None = None) -> list:
    """Return a list of obstacles [(x, y, radius_m), ...] from mock_demo_data.yaml.

    The runtime occupancy map omits 3D obstructions (vehicles/bicycles/people/objects).
    Scoring does not ship this yaml -- use lidar discs instead. Opt in with
    NAV_USE_MOCK_OBSTACLES=1 when debugging without a live cloud.
    """
    data = load_mock_data(path)
    obstacles = [(float(o["x"]), float(o["y"]), float(o["radius"]))
                 for o in (data.get("obstacles") or [])]
    log.info("[mock-data] loaded %d obstacles", len(obstacles))
    return obstacles


def obstacles_excluding_goal(obstacles, goal_xy, exclude_m=0.6):
    """Return the list with obstacles within exclude_m of the goal removed.

    Keeps the goal itself (target object / owner zone) reachable, while still routing
    around nearby obstructions.
    """
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    ex2 = float(exclude_m) ** 2
    out = []
    for (x, y, r) in obstacles:
        if (x - gx) ** 2 + (y - gy) ** 2 <= ex2:
            continue
        out.append((x, y, r))
    return out
