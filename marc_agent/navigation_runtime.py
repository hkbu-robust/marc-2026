"""Live Stage 2 search-area execution: occupancy + lidar overlay + D* Lite.

Wraps the #24 planner (``OccupancyGridPlanner`` / ``PathFollower``) in a typed
``Stage2Navigator``. Production never reads ``mock_demo_data.yaml``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from marc_agent.lidar_overlay import (
    frontal_hit,
    lidar_cloud_to_obstacles,
    pointcloud2_xyz,
)
from marc_agent.retrieval_state import DEFAULT_GOAL_TOLERANCE

_DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from mock_agent_navigation import (  # noqa: E402
    GOAL_RETARGET_M,
    OccupancyGridPlanner,
    PathFollower,
    SpeedGovernor,
    path_is_blocked,
    resolve_tol,
    resolve_xy,
)

log = logging.getLogger("participant_app")

# Stop and paint overlay before GoalSeeking through a live lidar blob.
# Bumper box matches the Stage 2 straight-approach stop in participant_app.
START_HOLE_M = 0.20
BLOCK_HOLD_S = 0.50
PAINT_DWELL_S = 1.0
BACKUP_V = 0.20
ESCAPE_ANGULAR = 1.2
BUMPER_X_MAX = 0.8
BUMPER_Y_HALF = 0.35
SPAWN_EGRESS_M = 1.0
SPAWN_EGRESS_S = 4.0
CREEP_V = 0.20


class NavStatus(str, Enum):
    REACHED = "reached"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    NO_MAP = "no_map"
    NO_PATH = "no_path"


class ReplanReason:
    STALL = "stall"
    PATH_BLOCKED = "path_blocked"
    GOAL_RETARGET = "goal_retarget"
    FRONTAL = "frontal"
    NO_MAP = "no_map"
    NO_PATH = "no_path"
    REPLAN_CAP = "replan_cap"
    FIRST_PLAN = "first_plan"


def _occupancy_cell_free(planner, x, y) -> bool:
    """True when the chassis cell is occupancy-free (0), ignoring lidar overlay."""
    if planner is None or not planner.ready:
        return False
    row, col = planner._w2c(x, y)
    if not planner._in_bounds(row, col):
        return False
    return int(planner._grid[row, col]) == 0


def _frontal_escape(planner, pose, follower, bumper_hot, escape_turn):
    """Reverse on free road. If a detour exists, latch a turn until the bumper clears.

    Returns ``(cmd, escape_turn)`` where ``cmd`` is ``(linear, angular)`` or None.
    """
    if not bumper_hot:
        return None, 0.0
    x, y, yaw = pose
    if not _occupancy_cell_free(planner, x, y):
        return None, 0.0
    turn = float(escape_turn)
    if follower is not None and follower.path:
        if abs(turn) < 1e-9:
            idx = min(int(follower.idx), len(follower.path) - 1)
            wx, wy = follower.path[idx]
            herr = math.atan2(
                math.sin(math.atan2(wy - y, wx - x) - yaw),
                math.cos(math.atan2(wy - y, wx - x) - yaw),
            )
            turn = 1.0 if herr >= 0.0 else -1.0
        angular = turn * ESCAPE_ANGULAR
    else:
        turn = 0.0
        angular = 0.0
    behind = BACKUP_V * 0.25
    bx = x - behind * math.cos(float(yaw))
    by = y - behind * math.sin(float(yaw))
    linear = -BACKUP_V if _occupancy_cell_free(planner, bx, by) else 0.0
    return (linear, angular), turn


def _overlay_busy(planner, lidar_n: int) -> bool:
    """True when live lidar discs or a stamped overlay must keep full robot inflation."""
    if int(lidar_n) > 0:
        return True
    overlay = getattr(planner, "_overlay", None)
    if overlay is None:
        return False
    return bool(overlay.any())


def _plan_attempts(inflation_m: float, allow_thin: bool = True):
    """Configured inflation first, then optional thinner inflation, then a wider crop."""
    attempts = [(float(inflation_m), 12.0)]
    if allow_thin:
        for infl in (0.10, 0.0):
            if infl < float(inflation_m) - 1e-9:
                attempts.append((infl, 12.0))
    for margin in (40.0, 80.0):
        attempts.append((float(inflation_m), margin))
    return attempts


@dataclass
class NavigationResult:
    status: NavStatus
    reason: str = ""
    replan_count: int = 0
    goal_xy: Optional[tuple] = None
    pose_xy: Optional[tuple] = None
    leg: str = ""
    path: Optional[tuple] = None
    last_method: Optional[str] = None

    @property
    def reached(self) -> bool:
        return self.status == NavStatus.REACHED


class Stage2Navigator:
    """Drive to a (possibly live) goal on the runtime occupancy grid."""

    def __init__(
        self,
        client,
        dt: float = 0.10,
        inflation_m: float = 0.25,
        exclude_m: float = 0.6,
        stuck_time_s: float = 8.0,
        stuck_move_m: float = 0.05,
        replan_cap: int = 50,
        cruise: bool = False,
    ):
        self.client = client
        self.dt = float(dt)
        self.inflation_m = float(inflation_m)
        self.exclude_m = float(exclude_m)
        self.stuck_time_s = float(stuck_time_s)
        self.stuck_move_m = float(stuck_move_m)
        self.replan_cap = int(replan_cap)
        self.cruise = bool(cruise)
        self._path_inflation = float(inflation_m)
        self.last_path = None
        self.last_method = None
        self._escape_turn = 0.0

    def _warm_lidar(self, deadline_s, ticks: int = 8):
        """Subscribe and wait briefly so the first plan is not empty of discs."""
        client = self.client
        if client.get_lidar() is not None:
            return
        until = min(client.now_s() + self.dt * max(0, int(ticks)), float(deadline_s) - self.dt)
        while client.now_s() < until:
            client.sleep(self.dt)
            if client.get_lidar() is not None:
                return

    def _plan_path(self, planner, start_xy, goal_xy, allow_thin: bool = True):
        """Plan with inflation then crop-margin fallbacks.

        Reject a path whose occupancy-only start chord cuts 100 / unknown (no
        inflation — D* already squeezed past overlay). Keep a detour whose start
        chord only hits lidar overlay; reverse until the bumper box is clear.
        """
        for inflation, margin in _plan_attempts(self.inflation_m, allow_thin=allow_thin):
            path = planner.plan(
                start_xy,
                goal_xy,
                inflation_m=inflation,
                snap_m=8.0,
                snap_goal_m=25.0,
                margin_m=margin,
            )
            if not path:
                continue
            chord = [start_xy] + list(path)
            if path_is_blocked(
                planner, chord, inflation_m=0.0, ignore_overlay=True,
            ):
                continue
            self._path_inflation = float(inflation)
            if abs(inflation - self.inflation_m) > 1e-9 or abs(margin - 12.0) > 1e-9:
                log.info(
                    "[Stage2] path fallback inflation=%.2f margin=%.1f method=%s",
                    inflation,
                    margin,
                    planner.last_method or "dstar",
                )
            return path
        return None

    def navigate_to(self, goal_xy, deadline_s, *, leg: str = "search",
                    tol=DEFAULT_GOAL_TOLERANCE, extra_obstacles=None) -> NavigationResult:
        """Follow occupancy D* Lite until inside ``tol`` of ``goal_xy``.

        ``goal_xy`` and ``tol`` may be callables so a late Stage 2 reveal can retarget.
        For the first metre (or 4 s) lidar overlay is ignored so a sunken spawn pad
        cannot trap the chassis; occupancy-only D* or a 0.20 m/s creep on free road.
        After egress, live lidar overlay and bumper reverse apply. Off-road / unknown
        start stays fail-closed (zero, no_path).
        """
        client = self.client
        extra = list(extra_obstacles or [])
        self._warm_lidar(deadline_s)
        planner = OccupancyGridPlanner()
        follower = None
        governor = SpeedGovernor()

        last_xy = None
        last_progress_t = client.now_s()
        last_log_t = 0.0
        last_goal = None
        last_lidar_n = 0
        n_replan = 0
        logged_no_map = False
        prev_frontal = False
        occ_logged = False
        last_reason = ""
        last_pose = None
        last_goal_xy = None
        had_map = False
        had_path = False
        hold_until = 0.0
        pending_after_hold = ""
        did_initial_paint = False
        did_block_paint = False
        egress_origin = None
        egress_t0 = None
        logged_egress = False
        prev_egress = False

        while client.is_running and client.now_s() < deadline_s:
            pose = client.get_world_pose()
            if pose is None:
                client.sleep(self.dt)
                continue
            x, y, yaw = pose
            last_pose = (x, y)
            cur_goal = resolve_xy(goal_xy)
            cur_tol = resolve_tol(tol, default=DEFAULT_GOAL_TOLERANCE)
            last_goal_xy = cur_goal
            if cur_goal is None:
                client.send_cmd_vel(0.0, 0.0)
                client.sleep(self.dt)
                continue
            dist = math.hypot(cur_goal[0] - x, cur_goal[1] - y)
            if dist < cur_tol:
                client.send_cmd_vel(0.0, 0.0)
                log.info("[STAGE2] %s reached at (%.2f, %.2f) d=%.2f tol=%.2f",
                         leg, x, y, dist, cur_tol)
                return NavigationResult(
                    NavStatus.REACHED, reason="reached", replan_count=n_replan,
                    goal_xy=cur_goal, pose_xy=(x, y), leg=leg,
                    path=self.last_path, last_method=self.last_method,
                )

            occ = client.get_occupancy_map()
            if occ is not None:
                planner.update_from_msg(occ)
                had_map = True
                if not occ_logged:
                    occ_logged = True
                    log.info("[Stage2] occupancy received: %dx%d res=%.3f origin=(%.1f, %.1f)",
                             occ.info.width, occ.info.height, occ.info.resolution,
                             occ.info.origin.position.x, occ.info.origin.position.y)

            cloud = client.get_lidar()
            lidar_obs = lidar_cloud_to_obstacles(cloud, pose) if cloud is not None else []
            last_lidar_n = len(lidar_obs)
            now_s = client.now_s()
            if egress_origin is None:
                egress_origin = (x, y)
                egress_t0 = now_s
            in_egress = (
                math.hypot(x - egress_origin[0], y - egress_origin[1]) < SPAWN_EGRESS_M
                and (now_s - egress_t0) < SPAWN_EGRESS_S
            )
            if in_egress and not logged_egress:
                logged_egress = True
                log.info(
                    "[Stage2] %s spawn egress %.1fm/%.1fs (ignore lidar overlay)",
                    leg, SPAWN_EGRESS_M, SPAWN_EGRESS_S,
                )
            planner.expire_overlay(now_s)
            hole = max(float(self.exclude_m), cur_tol)
            overlay_discs = list(extra)
            if not in_egress:
                overlay_discs.extend(lidar_obs)
            planner.stamp_overlay(overlay_discs, now_s)
            planner.clear_goal_hole(cur_goal, exclude_m=hole)
            planner.clear_start_hole((x, y), exclude_m=START_HOLE_M)

            if (not in_egress) and lidar_obs and not did_initial_paint:
                did_initial_paint = True
                hold_until = max(hold_until, now_s + PAINT_DWELL_S)
                log.info("[Stage2] %s lidar map dwell %.1fs (%d discs)",
                         leg, PAINT_DWELL_S, last_lidar_n)

            frontal_discs = list(extra)
            if not in_egress:
                frontal_discs.extend(lidar_obs)
            hold_linear = frontal_hit(
                frontal_discs, pose, x_max=BUMPER_X_MAX, y_half=BUMPER_Y_HALF,
            )
            bumper_hot = hold_linear
            need_replan = follower is None
            replan_reason = ReplanReason.FIRST_PLAN if need_replan else ""
            if last_goal is None or math.hypot(
                    cur_goal[0] - last_goal[0], cur_goal[1] - last_goal[1]) > GOAL_RETARGET_M:
                need_replan = True
                replan_reason = ReplanReason.GOAL_RETARGET
            last_goal = cur_goal
            if prev_egress and not in_egress:
                follower = None
                need_replan = True
                replan_reason = ReplanReason.FIRST_PLAN
            prev_egress = in_egress

            if bumper_hot:
                last_reason = ReplanReason.FRONTAL
                if not prev_frontal:
                    log.info("[Stage2] %s FRONTAL lidar stop -> reverse on free road", leg)
            prev_frontal = bumper_hot

            if last_xy is not None:
                if math.hypot(x - last_xy[0], y - last_xy[1]) > self.stuck_move_m:
                    last_progress_t = client.now_s()
                elif client.now_s() - last_progress_t > self.stuck_time_s:
                    log.info("[Stage2] %s stall -> replan", leg)
                    need_replan = True
                    replan_reason = ReplanReason.STALL
                    last_reason = ReplanReason.STALL
                    last_progress_t = client.now_s()
            last_xy = (x, y)

            if follower is not None and not need_replan and now_s >= hold_until:
                remain = list(follower.path[follower.idx:])
                overlay_blocked = (
                    len(remain) >= 2
                    and path_is_blocked(
                        planner, remain, inflation_m=self._path_inflation,
                    )
                )
                join_blocked = False
                if remain and _occupancy_cell_free(planner, x, y):
                    join = [(x, y), remain[0]]
                    join_blocked = path_is_blocked(
                        planner, join, inflation_m=0.0,
                        ignore_overlay=True,
                    )
                if overlay_blocked or join_blocked:
                    log.info("[Stage2] %s path blocked -> replan", leg)
                    last_reason = ReplanReason.PATH_BLOCKED
                    if not did_block_paint:
                        did_block_paint = True
                        hold_until = max(hold_until, now_s + PAINT_DWELL_S)
                    else:
                        hold_until = max(hold_until, now_s + BLOCK_HOLD_S)
                    pending_after_hold = ReplanReason.PATH_BLOCKED

            in_hold = now_s < hold_until
            if in_hold:
                hold_linear = True
                need_replan = False
            elif pending_after_hold:
                need_replan = True
                replan_reason = pending_after_hold
                pending_after_hold = ""

            if need_replan:
                if not planner.ready:
                    last_reason = ReplanReason.NO_MAP
                    if not logged_no_map:
                        log.warning("[Stage2] %s NO_MAP -- occupancy not received, holding", leg)
                        logged_no_map = True
                    follower = None
                    client.send_cmd_vel(0.0, 0.0)
                    client.sleep(self.dt)
                    continue
                counts_cap = replan_reason == ReplanReason.PATH_BLOCKED or (
                    had_path
                    and replan_reason in (
                        ReplanReason.STALL,
                        ReplanReason.FRONTAL,
                        ReplanReason.GOAL_RETARGET,
                    )
                )
                if counts_cap:
                    n_replan += 1
                    if n_replan > self.replan_cap:
                        log.error("[Stage2] %s ABORTED after %d replans", leg, self.replan_cap)
                        client.send_cmd_vel(0.0, 0.0)
                        return NavigationResult(
                            NavStatus.ABORTED, reason=ReplanReason.REPLAN_CAP,
                            replan_count=n_replan, goal_xy=cur_goal, pose_xy=(x, y), leg=leg,
                            path=self.last_path, last_method=self.last_method,
                        )
                    log.info("[Stage2] %s replan reason=%s count=%d", leg, replan_reason, n_replan)
                allow_thin = not _overlay_busy(
                    planner, 0 if in_egress else last_lidar_n,
                )
                path = self._plan_path(planner, (x, y), cur_goal, allow_thin=allow_thin)
                if path:
                    follower = PathFollower(path, passthrough=self.cruise)
                    had_path = True
                    last_reason = replan_reason or "ok"
                    self.last_path = tuple(path)
                    self.last_method = planner.last_method
                    log.info("[Stage2] occupancy %s path %d waypoints (%d lidar discs)",
                             planner.last_method or "dstar", len(path), last_lidar_n)
                else:
                    follower = None
                    last_reason = ReplanReason.NO_PATH
                    sr, sc = planner._w2c(x, y)
                    gr, gc = planner._w2c(*cur_goal)
                    sv = int(planner._grid[sr, sc]) if planner._in_bounds(sr, sc) else 99
                    gv = int(planner._grid[gr, gc]) if planner._in_bounds(gr, gc) else 99
                    start_trav = planner.last_start_trav
                    goal_trav = planner.last_goal_trav
                    if now_s - last_log_t >= 1.0:
                        log.warning("[Stage2] %s NO_PATH (%s) -- holding "
                                    "(start cell=(%d,%d) val=%d trav=%s, "
                                    "goal cell=(%d,%d) val=%d trav=%s) "
                                    "[0=free 100=occ -1=unknown]",
                                    leg, planner.last_status, sr, sc, sv, start_trav,
                                    gr, gc, gv, goal_trav)
                        last_log_t = now_s
                    cmd, self._escape_turn = _frontal_escape(
                        planner, (x, y, yaw), None, bumper_hot, self._escape_turn,
                    )
                    if cmd is not None:
                        client.send_cmd_vel(cmd[0], cmd[1])
                    elif in_egress and _occupancy_cell_free(planner, x, y):
                        client.send_cmd_vel(CREEP_V, 0.0)
                    else:
                        client.send_cmd_vel(0.0, 0.0)
                    client.sleep(self.dt)
                    continue

            wp_str = ""
            if follower is not None:
                linear, angular, done, info = follower.compute(pose)
                wp_str = " wp %d/%d=(%.1f, %.1f)" % (
                    info["idx"], info["total"], info["waypoint"][0], info["waypoint"][1])
                if done:
                    client.send_cmd_vel(0.0, 0.0)
                    log.info("[STAGE2] %s reached goal at (%.2f, %.2f)", leg, x, y)
                    return NavigationResult(
                        NavStatus.REACHED, reason="reached", replan_count=n_replan,
                        goal_xy=cur_goal, pose_xy=(x, y), leg=leg,
                        path=self.last_path, last_method=self.last_method,
                    )
            else:
                linear, angular = 0.0, 0.0

            if hold_linear:
                linear, angular = 0.0, 0.0
            cmd, self._escape_turn = _frontal_escape(
                planner, (x, y, yaw), follower, bumper_hot, self._escape_turn,
            )
            if cmd is not None:
                linear, angular = cmd
            elif linear > 1e-6:
                xyz = pointcloud2_xyz(cloud) if cloud is not None else None
                imu = client.get_imu() if hasattr(client, "get_imu") else None
                linear, angular = governor.limit(
                    linear, angular, imu=imu, xyz=xyz, dt=self.dt)

            client.send_cmd_vel(linear_x=linear, angular_z=angular)

            now = client.now_s()
            if now - last_log_t >= 1.0:
                extra_s = " lidar=%d replan=%d" % (last_lidar_n, n_replan)
                if in_egress:
                    extra_s += " EGRESS"
                if bumper_hot:
                    extra_s += " FRONTAL"
                if cmd is not None and linear < -1e-6:
                    extra_s += " REVERSE"
                log.info("[STAGE2] %s pose=(%.2f, %.2f, yaw=%.0fdeg) dist=%.2fm%s%s v=%.2f w=%+.2f",
                         leg, x, y, math.degrees(yaw), dist, wp_str, extra_s, linear, angular)
                last_log_t = now

            client.sleep(self.dt)

        status = NavStatus.ABORTED
        reason = last_reason or "aborted"
        if not getattr(client, "is_running", True):
            status = NavStatus.ABORTED
            reason = last_reason or "aborted"
        elif last_reason == ReplanReason.NO_MAP or not had_map:
            status = NavStatus.NO_MAP
            reason = ReplanReason.NO_MAP
        elif last_reason == ReplanReason.NO_PATH or (had_map and not had_path):
            status = NavStatus.NO_PATH
            reason = ReplanReason.NO_PATH
        elif client.now_s() >= deadline_s:
            status = NavStatus.TIMEOUT
            reason = "timeout"
        client.send_cmd_vel(0.0, 0.0)
        return NavigationResult(
            status, reason=reason, replan_count=n_replan,
            goal_xy=last_goal_xy, pose_xy=last_pose, leg=leg,
            path=self.last_path, last_method=self.last_method,
        )
