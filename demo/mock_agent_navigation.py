"""Mock navigation agent -- stands in for the participant's Stage 2 navigation.

This module bundles the demo's navigation pieces (participants replace/extend them):
  - GoalSeekController / PathFollower -- P-control that returns ``(linear_x, angular_z)``
    scalars to publish with ``client.send_cmd_vel(...)``.
  - SpeedGovernor -- IMU + low-Z lidar cap on those scalars (rollover / curb bounce).
  - run_nav_leg -- live occupancy/D* execution with lidar overlay and backup recovery.
  - OccupancyGridPlanner -- D* Lite over the runtime occupancy grid
    (``MARCClient.get_occupancy_map()``), with A* as a fallback.
  - lidar_cloud_to_obstacles -- live PointCloud2 hits as extra discs (roads-only map
    omits vehicles/people/objects).
  - load_obstacles / obstacles_excluding_goal -- optional yaml discs for debugging
    (``NAV_USE_MOCK_OBSTACLES=1``); scoring does not have ``mock_demo_data.yaml``.
"""

from __future__ import annotations

import heapq
import logging
import math
import os

import numpy as np
import yaml

log = logging.getLogger("participant_app")

SQRT2 = math.sqrt(2.0)
_INF = 1e18

# Lidar overlay: drop ground / canopy, cap range, cluster hits to cell-sized discs.
LIDAR_Z_MIN = 0.15
LIDAR_Z_MAX = 1.5
LIDAR_RANGE_M = 20.0
LIDAR_MIN_RANGE_M = 0.20
LIDAR_HIT_RADIUS = 0.35
LIDAR_CLUSTER_M = 0.25
# Overlay cells not restamped for this long become free (moving people/ghosts).
OVERLAY_TTL_S = 2.0


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
    MAX_LINEAR = 1.0           # m/s; SpeedGovernor cuts this on corners / curbs / tilt
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
            cruise: if True, skip the near-goal proportional deceleration (K_LINEAR*dist)
                and cruise at MAX_LINEAR -- for intermediate pass-through waypoints; only
                the final goal (cruise=False) decelerates on approach.

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

    # Distance to consider an intermediate waypoint passed (m). Looser than the goal
    # tolerance to smooth out corners.
    WAYPOINT_TOLERANCE = 1.0

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
# Speed governor: IMU dynamics + curb-height lidar look-ahead.
# ---------------------------------------------------------------------------

# Curb band sits below LIDAR_Z_MIN so these hits never enter occupancy overlay.
CURB_Z_MIN = 0.03
CURB_Z_MAX = 0.18
CURB_X_MIN = 0.30
CURB_X_MAX = 2.00
CURB_Y_HALF = 0.25
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


def curb_hits_ahead(xyz):
    """Count curb-height lidar hits in a short bumper box.

    ``xyz`` is robot-frame Nx3 (lidar +X forward). Returns ``(n_hits, min_forward_m)``.
    No hits -> ``(0, inf)``. Not an occupancy overlay -- curbs stay drivable, just slow.
    """
    if xyz is None:
        return 0, float("inf")
    pts = np.asarray(xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
        return 0, float("inf")
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    mask = (
        finite
        & (z >= CURB_Z_MIN) & (z <= CURB_Z_MAX)
        & (x >= CURB_X_MIN) & (x <= CURB_X_MAX)
        & (np.abs(y) <= CURB_Y_HALF)
    )
    hits = pts[mask]
    if hits.shape[0] == 0:
        return 0, float("inf")
    return int(hits.shape[0]), float(np.min(hits[:, 0]))


class SpeedGovernor:
    """Cap PathFollower cmd_vel so straights can cruise and corners / curbs do not flip the chassis.

    ``limit(linear, angular, imu, xyz)`` never raises ``linear`` above the request. Missing
    IMU/lidar still caps from commanded yaw rate. After cutting v, recaps w with the same
    Ackermann rule as ``GoalSeekController`` so the chassis does not stall on ``2v <= w*T``.
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
    CURB_NEAR_M = 0.45        # at this forward range, v -> V_FLOOR
    CURB_FAR_M = CURB_X_MAX
    TAU_DOWN_S = 0.08
    TAU_UP_S = 0.40

    def __init__(self):
        self._v_filt = None
        self._spike_t = 0.0
        self._spike_v = self.V_SPIKE

    def limit(self, linear, angular, imu=None, xyz=None, dt=None):
        """Return ``(linear_x, angular_z)`` capped for stability and curb bounce."""
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

        n_curb, x_curb = curb_hits_ahead(xyz)
        if n_curb > 0:
            v_curb = _lerp_cap(x_curb, self.CURB_NEAR_M, self.CURB_FAR_M,
                               self.V_FLOOR, abs(requested))
            v_cap = min(v_cap, v_curb)

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


# ---------------------------------------------------------------------------
# Lidar overlay: PointCloud2 (Base_LiDAR) -> world discs.
# ---------------------------------------------------------------------------

def pointcloud2_xyz(msg):
    """Nx3 float32 XYZ from a sensor_msgs/PointCloud2, or None."""
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


def lidar_xyz_to_obstacles(xyz, pose, radius_m=LIDAR_HIT_RADIUS, cell_m=LIDAR_CLUSTER_M):
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


RECOVERY_LINEAR_X = -0.12   # m/s reverse while a frontal lidar disc is in the bumper box
RECOVERY_HOLD_S = 0.50
GOAL_RETARGET_M = 0.50


def recovery_cmd(frontal: bool) -> tuple:
    """cmd_vel while recovering from a frontal lidar hit: reverse, no yaw."""
    if frontal:
        return RECOVERY_LINEAR_X, 0.0
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
    inflation_m=0.25,
    exclude_m=0.6,
    cruise=False,
    extra_obstacles=None,
    dt=0.10,
    stuck_time_s=8.0,
    stuck_move_m=0.05,
    replan_cap=50,
):
    """Follow occupancy D* Lite to ``goal`` until inside ``tol``. True if reached.

    ``goal`` and ``tol`` may be callables so a late Stage 2 reveal can retarget.
    Live lidar discs overlay the grid; a frontal hit reverses briefly then replans.
    Fail closed: no open-loop drive when the map or path is missing.
    """
    extra = list(extra_obstacles or [])
    planner = OccupancyGridPlanner()
    follower = None
    governor = SpeedGovernor()

    last_xy = None
    last_progress_t = client.now_s()
    last_log_t = 0.0
    last_goal = None
    reached_goal = False
    last_lidar_n = 0
    n_replan = 0
    logged_no_map = False
    prev_frontal = False
    recovery_t0 = None
    occ_logged = False

    while client.is_running and client.now_s() < deadline:
        pose = client.get_world_pose()
        if pose is None:
            client.sleep(dt)
            continue
        x, y, yaw = pose
        cur_goal = resolve_xy(goal)
        cur_tol = resolve_tol(tol)
        if cur_goal is None:
            client.send_cmd_vel(0.0, 0.0)
            client.sleep(dt)
            continue
        dist = math.hypot(cur_goal[0] - x, cur_goal[1] - y)
        if dist < cur_tol:
            reached_goal = True
            log.info("[STAGE2] reached search/goal at (%.2f, %.2f) d=%.2f tol=%.2f",
                     x, y, dist, cur_tol)
            client.send_cmd_vel(0.0, 0.0)
            break

        occ = client.get_occupancy_map()
        if occ is not None:
            planner.update_from_msg(occ)
            if not occ_logged:
                occ_logged = True
                log.info("[Stage2] occupancy received: %dx%d res=%.3f origin=(%.1f, %.1f)",
                         occ.info.width, occ.info.height, occ.info.resolution,
                         occ.info.origin.position.x, occ.info.origin.position.y)

        cloud = client.get_lidar()
        lidar_obs = lidar_cloud_to_obstacles(cloud, pose) if cloud is not None else []
        last_lidar_n = len(lidar_obs)
        now_s = client.now_s()
        planner.expire_overlay(now_s)
        hole = max(float(exclude_m), cur_tol)
        planner.stamp_overlay(extra + lidar_obs, now_s)
        planner.clear_goal_hole(cur_goal, exclude_m=hole)

        hold_linear = frontal_hit(extra + lidar_obs, pose)
        need_replan = follower is None
        if last_goal is None or math.hypot(
                cur_goal[0] - last_goal[0], cur_goal[1] - last_goal[1]) > GOAL_RETARGET_M:
            need_replan = True
        last_goal = cur_goal

        recovering = False
        if hold_linear:
            if recovery_t0 is None:
                recovery_t0 = now_s
            if now_s - recovery_t0 < RECOVERY_HOLD_S:
                recovering = True
            else:
                need_replan = True
                recovery_t0 = None
        else:
            recovery_t0 = None
        if hold_linear and not prev_frontal:
            need_replan = True
        prev_frontal = hold_linear

        if last_xy is not None:
            if math.hypot(x - last_xy[0], y - last_xy[1]) > stuck_move_m:
                last_progress_t = client.now_s()
            elif client.now_s() - last_progress_t > stuck_time_s:
                log.info("[Stage2] stall detected -> replanning")
                need_replan = True
                last_progress_t = client.now_s()
        last_xy = (x, y)

        if follower is not None and not need_replan:
            remain = [(x, y)] + follower.path[follower.idx:]
            if path_is_blocked(planner, remain, inflation_m=inflation_m):
                log.info("[Stage2] path blocked -> replanning")
                need_replan = True

        if recovering:
            lin, ang = recovery_cmd(True)
            client.send_cmd_vel(linear_x=lin, angular_z=ang)
            client.sleep(dt)
            continue

        if need_replan:
            if not planner.ready:
                if not logged_no_map:
                    log.warning("[Stage2] NO_MAP -- occupancy not received, holding")
                    logged_no_map = True
                follower = None
                client.send_cmd_vel(0.0, 0.0)
                client.sleep(dt)
                continue
            n_replan += 1
            if n_replan > replan_cap:
                log.error("[Stage2] ABORTED after %d replans", replan_cap)
                client.send_cmd_vel(0.0, 0.0)
                return False
            path = planner.plan((x, y), cur_goal, inflation_m=inflation_m, snap_m=8.0)
            if path:
                follower = PathFollower(path, passthrough=cruise)
                log.info("[Stage2] occupancy %s path %d waypoints (%d lidar discs)",
                         planner.last_method or "dstar", len(path), last_lidar_n)
            else:
                follower = None
                sr, sc = planner._w2c(x, y)
                gr, gc = planner._w2c(*cur_goal)
                sv = int(planner._grid[sr, sc]) if planner._in_bounds(sr, sc) else 99
                gv = int(planner._grid[gr, gc]) if planner._in_bounds(gr, gc) else 99
                if now_s - last_log_t >= 1.0:
                    log.warning("[Stage2] NO_PATH (%s) -- holding "
                                "(start cell=(%d,%d) val=%d, goal cell=(%d,%d) val=%d) "
                                "[0=free 100=occ -1=unknown]",
                                planner.last_status, sr, sc, sv, gr, gc, gv)
                    last_log_t = now_s
                client.send_cmd_vel(0.0, 0.0)
                client.sleep(dt)
                continue

        wp_str = ""
        if follower is not None:
            linear, angular, done, info = follower.compute(pose)
            wp_str = " wp %d/%d=(%.1f, %.1f)" % (
                info["idx"], info["total"], info["waypoint"][0], info["waypoint"][1])
            if done:
                reached_goal = True
                log.info("[STAGE2] reached goal at (%.2f, %.2f)", x, y)
                client.send_cmd_vel(0.0, 0.0)
                break
        else:
            linear, angular = 0.0, 0.0

        if hold_linear:
            linear, angular = recovery_cmd(True)
        elif linear > 1e-6:
            xyz = pointcloud2_xyz(cloud) if cloud is not None else None
            imu = client.get_imu() if hasattr(client, "get_imu") else None
            linear, angular = governor.limit(
                linear, angular, imu=imu, xyz=xyz, dt=dt)

        client.send_cmd_vel(linear_x=linear, angular_z=angular)

        now = client.now_s()
        if now - last_log_t >= 1.0:
            extra_s = " lidar=%d replan=%d" % (last_lidar_n, n_replan)
            if hold_linear:
                extra_s += " FRONTAL"
            log.info("[STAGE2] pose=(%.2f, %.2f, yaw=%.0fdeg) dist=%.2fm%s%s v=%.2f w=%+.2f",
                     x, y, math.degrees(yaw), dist, wp_str, extra_s, linear, angular)
            last_log_t = now

        client.sleep(dt)

    return reached_goal


def path_is_blocked(planner, waypoints, extra_obstacles=None, inflation_m=0.25):
    """True if any remaining polyline segment fails Bresenham LOS on the inflated grid."""
    if planner is None or not planner.ready or waypoints is None or len(waypoints) < 2:
        return False
    cells = []
    for wp in waypoints:
        r, c = planner._w2c(float(wp[0]), float(wp[1]))
        if not planner._in_bounds(r, c):
            return True
        cells.append((r, c))
    pad = 4
    rmin = max(0, min(r for r, _ in cells) - pad)
    rmax = min(planner._h, max(r for r, _ in cells) + pad + 1)
    cmin = max(0, min(c for _, c in cells) - pad)
    cmax = min(planner._w, max(c for _, c in cells) + pad + 1)
    sub = planner._grid[rmin:rmax, cmin:cmax]
    trav = sub == 0
    if planner._overlay is not None:
        trav = np.logical_and(
            trav, np.logical_not(planner._overlay[rmin:rmax, cmin:cmax]))
    if extra_obstacles:
        planner._block_obstacles(trav, extra_obstacles, rmin, cmin)
    infl_c = int(round(float(inflation_m) / planner._res)) if planner._res else 0
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

    def plan(self, start_xy, goal_xy, inflation_m=0.25, margin_m=12.0,
             snap_m=1.5, extra_obstacles=None):
        """Return a detour path from start_xy -> goal_xy as a list of world waypoints.

        Args:
            start_xy: (x, y) current robot position (world).
            goal_xy:  (x, y) target position (world).
            inflation_m: erode the free area by the robot radius to build a path that
                         stays away from the road boundary (avoids step stalls).
            margin_m: search margin outside the start/goal bounding box (crop for planning speed).
            snap_m:  maximum radius to correct a start/goal cell to the nearest free
                     cell when it is not traversable.
            extra_obstacles: [(x, y, radius_m), ...] extra discs carved for this
                     call only (tests / yaml). Live lidar should use stamp_overlay.

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
        infl_c = int(round(inflation_m / self._res))

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
        trav = sub == 0
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
        ls = self._snap(trav, ls, snap_m)
        lg = self._snap(trav, lg, snap_m)
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
            if len(path) > 1:
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

    def expire_overlay(self, now_s=0.0):
        """Free overlay cells not restamped within OVERLAY_TTL_S."""
        if self._overlay is None or self._overlay_t is None:
            return
        stale = self._overlay & ((float(now_s) - self._overlay_t) > OVERLAY_TTL_S)
        self._overlay[stale] = False

    def clear_goal_hole(self, goal_xy, exclude_m=0.6):
        """Keep the goal reachable: drop overlay inside exclude_m of goal_xy."""
        if not self.ready or self._overlay is None or exclude_m <= 0:
            return
        sl = self._disc_mask(float(goal_xy[0]), float(goal_xy[1]), float(exclude_m))
        if sl is None:
            return
        r0, r1, c0, c1, disc = sl
        self._overlay[r0:r1, c0:c1][disc] = False

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
        # snap_m is in meters, but here it is approximated as a cell-grid radius
        max_rad = int(max(1, round(snap_m / 0.05)))
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
