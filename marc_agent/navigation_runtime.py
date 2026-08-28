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


def _pose_on_free_cell(planner, x, y) -> bool:
    """True when the chassis cell is occupancy-free (already on the road)."""
    if planner is None or not planner.ready:
        return False
    row, col = planner._w2c(x, y)
    if not planner._in_bounds(row, col):
        return False
    val = int(planner._grid[row, col])
    if not ((val == 0) or (val < 0)):
        return False
    overlay = getattr(planner, "_overlay", None)
    if overlay is not None and bool(overlay[row, col]):
        return False
    return True


def _plan_attempts(inflation_m: float):
    """Configured inflation first, then thinner inflation, then a wider crop."""
    attempts = [(float(inflation_m), 12.0)]
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

    def _plan_path(self, planner, start_xy, goal_xy):
        """Plan with inflation then crop-margin fallbacks. First unblocked success wins."""
        for inflation, margin in _plan_attempts(self.inflation_m):
            path = planner.plan(
                start_xy,
                goal_xy,
                inflation_m=inflation,
                snap_m=8.0,
                margin_m=margin,
            )
            if not path:
                continue
            chord = [start_xy] + list(path)
            blocked = path_is_blocked(planner, chord, inflation_m=inflation)
            if blocked:
                join_dist = math.hypot(
                    path[0][0] - float(start_xy[0]),
                    path[0][1] - float(start_xy[1]),
                )
                cell_m = float(getattr(planner, "_res", None) or 0.05)
                if len(path) <= 1 or join_dist < max(1.5 * cell_m, 0.15):
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
        Live lidar discs overlay the grid. A frontal hazard commands zero and replans.
        Fail closed: no open-loop drive when the map or path is missing.
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
            planner.expire_overlay(now_s)
            hole = max(float(self.exclude_m), cur_tol)
            planner.stamp_overlay(extra + lidar_obs, now_s)
            planner.clear_goal_hole(cur_goal, exclude_m=hole)
            planner.clear_start_hole((x, y), exclude_m=hole)

            hold_linear = frontal_hit(extra + lidar_obs, pose)
            need_replan = follower is None
            replan_reason = ReplanReason.FIRST_PLAN if need_replan else ""
            if last_goal is None or math.hypot(
                    cur_goal[0] - last_goal[0], cur_goal[1] - last_goal[1]) > GOAL_RETARGET_M:
                need_replan = True
                replan_reason = ReplanReason.GOAL_RETARGET
            last_goal = cur_goal

            if hold_linear:
                last_reason = ReplanReason.FRONTAL
                if not prev_frontal:
                    need_replan = True
                    replan_reason = ReplanReason.FRONTAL
                    log.info("[Stage2] %s FRONTAL lidar stop -> cmd zero, replan", leg)
            prev_frontal = hold_linear

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

            if follower is not None and not need_replan:
                remain = list(follower.path[follower.idx:])
                if remain and _pose_on_free_cell(planner, x, y):
                    remain = [(x, y)] + remain
                if len(remain) >= 2 and path_is_blocked(
                    planner, remain, inflation_m=self._path_inflation
                ):
                    log.info("[Stage2] %s path blocked -> replan", leg)
                    need_replan = True
                    replan_reason = ReplanReason.PATH_BLOCKED
                    last_reason = ReplanReason.PATH_BLOCKED

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
                        )
                    log.info("[Stage2] %s replan reason=%s count=%d", leg, replan_reason, n_replan)
                path = self._plan_path(planner, (x, y), cur_goal)
                if path:
                    follower = PathFollower(path, passthrough=self.cruise)
                    had_path = True
                    last_reason = replan_reason or "ok"
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
                    )
            else:
                linear, angular = 0.0, 0.0

            if hold_linear:
                linear, angular = 0.0, 0.0
            elif linear > 1e-6:
                xyz = pointcloud2_xyz(cloud) if cloud is not None else None
                imu = client.get_imu() if hasattr(client, "get_imu") else None
                linear, angular = governor.limit(
                    linear, angular, imu=imu, xyz=xyz, dt=self.dt)

            client.send_cmd_vel(linear_x=linear, angular_z=angular)

            now = client.now_s()
            if now - last_log_t >= 1.0:
                extra_s = " lidar=%d replan=%d" % (last_lidar_n, n_replan)
                if hold_linear:
                    extra_s += " FRONTAL"
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
        )
