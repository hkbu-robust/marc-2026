"""Live Stage 2 search-area execution: occupancy D* Lite + local lidar corridor.

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

import numpy as np

from marc_agent.lidar_safety import (
    CLEAR_FAR,
    DISC_R,
    STOP_X_MAX,
    advise,
    overlay_discs,
    pointcloud2_xyz,
)
from marc_agent.retrieval_state import DEFAULT_GOAL_TOLERANCE, close_enough_to_search

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

BACKUP_V = 0.20
YAW_BIAS_GAIN = 0.5
RESUME_M = 1.20
BACKUP_LOOK_M = (0.5, 1.0, 1.5)
RAM_STUCK_S = 2.0
RAM_LINEAR_MIN = 0.05


class NavStatus(str, Enum):
    REACHED = "reached"
    TIMEOUT = "timeout"
    ABORTED = "aborted"
    NO_MAP = "no_map"
    NO_PATH = "no_path"


class ReplanReason:
    STALL = "stall"
    RAM = "ram"
    PATH_BLOCKED = "path_blocked"
    GOAL_RETARGET = "goal_retarget"
    FRONTAL = "frontal"
    NO_MAP = "no_map"
    NO_PATH = "no_path"
    REPLAN_CAP = "replan_cap"
    FIRST_PLAN = "first_plan"


def _ram_bumper_disc(pose):
    """World disc at the bumper so overlay survives the 0.20 m start hole."""
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    cc, ss = math.cos(yaw), math.sin(yaw)
    return (x + STOP_X_MAX * cc, y + STOP_X_MAX * ss, DISC_R)


def _occupancy_cell_free(planner, x, y) -> bool:
    """True when the chassis cell is occupancy-free (0)."""
    if planner is None or not planner.ready:
        return False
    row, col = planner._w2c(x, y)
    if not planner._in_bounds(row, col):
        return False
    return int(planner._grid[row, col]) == 0


def _last_free_behind(planner, pose):
    """Last occupancy-free cell along ``-heading``, or None if reverse is sealed."""
    x, y, yaw = pose
    if not _occupancy_cell_free(planner, x, y):
        return None
    cc, ss = math.cos(float(yaw)), math.sin(float(yaw))
    last = None
    for dist in (BACKUP_V * 0.25,) + BACKUP_LOOK_M:
        bx = x - dist * cc
        by = y - dist * ss
        if not _occupancy_cell_free(planner, bx, by):
            break
        last = (bx, by)
    return last


def _reverse_on_road(planner, pose):
    """Straight reverse along ``-heading`` while the backing cells stay occupancy-free."""
    if _last_free_behind(planner, pose) is None:
        return None
    return (-BACKUP_V, 0.0)


def _frontal_escape(planner, pose, bumper_hot):
    """Straight reverse on free road while the bumper box is hot. No yaw into the car."""
    if not bumper_hot:
        return None
    return _reverse_on_road(planner, pose)


def _overlay_ahead(planner, pose) -> bool:
    """True when lidar overlay sits along +heading within a short look-ahead."""
    if planner is None or not planner.ready or planner._overlay is None:
        return False
    x, y, yaw = pose
    cc, ss = math.cos(float(yaw)), math.sin(float(yaw))
    for dist in BACKUP_LOOK_M:
        r, c = planner._w2c(x + dist * cc, y + dist * ss)
        if planner._in_bounds(r, c) and bool(planner._overlay[r, c]):
            return True
    return False


def _poly_after_start_hole(origin, waypoints, hole_m=0.20):
    """Remaining polyline starting just outside the chassis overlay hole."""
    if not waypoints:
        return None
    ox, oy = float(origin[0]), float(origin[1])
    fx, fy = float(waypoints[0][0]), float(waypoints[0][1])
    dx, dy = fx - ox, fy - oy
    dist = math.hypot(dx, dy)
    if dist <= hole_m:
        return list(waypoints) if len(waypoints) >= 2 else None
    t = hole_m / dist
    return [(ox + t * dx, oy + t * dy)] + list(waypoints)


def _remaining_blocked(planner, pose_xy, remain, inflation_m) -> bool:
    """True when the followed path hugs occupancy or cuts a lidar disc.

    Occupancy inflation uses ``[pose] + remain`` so a PathFollower chord along
    the curb is rejected. Overlay uses the followed waypoints (or the chord
    after the 0.20 m start hole) so the chassis hole next to a car disc does
    not replan-loop.
    """
    if not remain:
        return False
    poly = [pose_xy] + list(remain)
    if path_is_blocked(
        planner, poly, inflation_m=inflation_m, ignore_overlay=True,
    ):
        return True
    if len(remain) >= 2:
        return path_is_blocked(
            planner, remain, inflation_m=inflation_m, ignore_overlay=False,
        )
    skipped = _poly_after_start_hole(pose_xy, remain)
    if skipped is None:
        return False
    return path_is_blocked(
        planner, skipped, inflation_m=inflation_m, ignore_overlay=False,
    )


def _should_backup(planner, pose, advice) -> bool:
    """True when NO_PATH should reverse instead of holding: lidar or overlay ahead."""
    if advice is not None:
        if advice.stop:
            return True
        if math.isfinite(advice.ahead_m) and advice.ahead_m < CLEAR_FAR:
            return True
    return _overlay_ahead(planner, pose)


def _plan_attempts(inflation_m: float, allow_thin: bool = True):
    """Configured inflation first, then a wider crop. Never thin the chassis radius."""
    del allow_thin
    attempts = [(float(inflation_m), 12.0)]
    for margin in (40.0, 80.0):
        attempts.append((float(inflation_m), margin))
    return attempts


def _world_discs_to_base(discs, pose):
    """World ``(x, y, r)`` disc centres -> robot-frame XYZ at body height."""
    if not discs:
        return None
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    cc, ss = math.cos(yaw), math.sin(yaw)
    rows = []
    for item in discs:
        dx = float(item[0]) - x
        dy = float(item[1]) - y
        rows.append([dx * cc + dy * ss, -dx * ss + dy * cc, 0.4])
    return np.asarray(rows, dtype=np.float32)


def _merge_xyz(cloud_xyz, extra_xyz):
    parts = [a for a in (cloud_xyz, extra_xyz) if a is not None and getattr(a, "size", 0)]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.vstack(parts)


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
        inflation_m: float = 0.55,
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

    def _warm_lidar(self, deadline_s, ticks: int = 8):
        """Wait briefly so the first tick can see a cloud if one is already publishing."""
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

        Snap an occupied/unknown goal onto the nearest free inflated cell.
        A free goal behind a wall stays put so D* can fail closed or go around.
        Never thin inflation below the configured chassis radius.
        Occupancy 100/unknown still fail-closed. Live lidar discs sit on overlay.
        """
        plan_goal = goal_xy
        free = getattr(planner, "cell_free", None)
        snap = getattr(planner, "nearest_free_xy", None)
        if (
            callable(free)
            and callable(snap)
            and not free(float(goal_xy[0]), float(goal_xy[1]), self.inflation_m)
        ):
            snapped = snap(goal_xy, inflation_m=self.inflation_m)
            if snapped is not None:
                plan_goal = snapped
        for inflation, margin in _plan_attempts(self.inflation_m, allow_thin=allow_thin):
            path = planner.plan(
                start_xy,
                plan_goal,
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
            # Overlay + curb pad on the followed polyline. Skip a 1-point path
            # so the chassis start hole is not treated as a blocked join.
            poly = list(path)
            if len(poly) >= 2 and path_is_blocked(
                planner, poly, inflation_m=inflation, ignore_overlay=False,
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
                    tol=DEFAULT_GOAL_TOLERANCE, extra_obstacles=None,
                    search_xy=None, search_radius=0.0,
                    handoff_fn=None) -> NavigationResult:
        """Follow occupancy D* Lite until inside ``tol`` of ``goal_xy``.

        ``goal_xy`` and ``tol`` may be callables so a late Stage 2 reveal can retarget.
        Abort early when the chassis is inside the reveal disc or ``handoff_fn``
        sees the tumbler, so near-field YOLO can start without driving to the
        approach edge. Live lidar stamps overlay discs so D* detours unmarked
        cars, and still scales or reverses in the chassis corridor. Off-road /
        unknown start stays fail-closed (zero, no_path). ``extra_obstacles`` is
        a test hook of world discs.
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
        n_replan = 0
        logged_no_map = False
        prev_frontal = False
        occ_logged = False
        last_reason = ""
        last_pose = None
        last_goal_xy = None
        had_map = False
        had_path = False
        stop_latched = False
        last_cmd_linear = 0.0
        handoff_warned = False

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
            disc_xy = resolve_xy(search_xy) if search_xy is not None else None
            disc_r = float(search_radius or 0.0)
            if close_enough_to_search((x, y), disc_xy, disc_r, cur_tol):
                client.send_cmd_vel(0.0, 0.0)
                log.info("[STAGE2] %s inside_disc at (%.2f, %.2f)", leg, x, y)
                return NavigationResult(
                    NavStatus.REACHED, reason="inside_disc", replan_count=n_replan,
                    goal_xy=cur_goal, pose_xy=(x, y), leg=leg,
                    path=self.last_path, last_method=self.last_method,
                )
            if cur_goal is None:
                client.send_cmd_vel(0.0, 0.0)
                last_cmd_linear = 0.0
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
            xyz = pointcloud2_xyz(cloud) if cloud is not None else None
            advice = advise(_merge_xyz(xyz, _world_discs_to_base(extra, pose)))
            now_s = client.now_s()
            discs = overlay_discs(xyz, pose)
            discs.extend(extra)
            if planner.ready:
                planner.expire_overlay(now_s)
                planner.stamp_overlay(discs, now_s=now_s)
                planner.clear_start_hole((x, y), exclude_m=0.20)
            if advice.stop:
                stop_latched = True
            elif not math.isfinite(advice.ahead_m) or advice.ahead_m >= RESUME_M:
                stop_latched = False
            bumper_hot = bool(advice.stop or stop_latched)
            ram_hot = False
            need_replan = follower is None
            replan_reason = ReplanReason.FIRST_PLAN if need_replan else ""
            if last_goal is None or math.hypot(
                    cur_goal[0] - last_goal[0], cur_goal[1] - last_goal[1]) > GOAL_RETARGET_M:
                need_replan = True
                replan_reason = ReplanReason.GOAL_RETARGET
            last_goal = cur_goal

            falling_edge = prev_frontal and not bumper_hot
            if bumper_hot:
                last_reason = ReplanReason.FRONTAL
                if not prev_frontal:
                    log.info("[Stage2] %s FRONTAL lidar stop -> reverse, stamp, replan", leg)
                    need_replan = True
                    replan_reason = ReplanReason.FRONTAL
            elif falling_edge:
                log.info("[Stage2] %s bumper clear -> replan around overlay", leg)
                need_replan = True
                replan_reason = ReplanReason.FRONTAL
            prev_frontal = bumper_hot

            if callable(handoff_fn):
                try:
                    if handoff_fn((x, y, yaw)):
                        client.send_cmd_vel(0.0, 0.0)
                        log.info("[STAGE2] %s yolo_seen at (%.2f, %.2f)", leg, x, y)
                        return NavigationResult(
                            NavStatus.REACHED, reason="yolo_seen",
                            replan_count=n_replan, goal_xy=cur_goal, pose_xy=(x, y),
                            leg=leg, path=self.last_path, last_method=self.last_method,
                        )
                except Exception:
                    if not handoff_warned:
                        log.warning("[Stage2] %s handoff_fn failed", leg, exc_info=True)
                        handoff_warned = True

            if last_xy is not None:
                moved = math.hypot(x - last_xy[0], y - last_xy[1]) > self.stuck_move_m
                if moved:
                    last_progress_t = client.now_s()
                elif (
                    last_cmd_linear > RAM_LINEAR_MIN
                    and not bumper_hot
                    and client.now_s() - last_progress_t >= RAM_STUCK_S
                ):
                    log.info("[Stage2] %s ram unstick -> stamp overlay, reverse, replan", leg)
                    last_reason = ReplanReason.RAM
                    ram_hot = True
                    need_replan = True
                    replan_reason = ReplanReason.RAM
                    last_progress_t = client.now_s()
                    if planner.ready:
                        planner.stamp_overlay(
                            [_ram_bumper_disc((x, y, yaw))], now_s=now_s,
                        )
                        planner.clear_start_hole((x, y), exclude_m=0.20)
                elif client.now_s() - last_progress_t > self.stuck_time_s:
                    log.info("[Stage2] %s stall -> replan", leg)
                    need_replan = True
                    replan_reason = ReplanReason.STALL
                    last_reason = ReplanReason.STALL
                    last_progress_t = client.now_s()
            last_xy = (x, y)

            if follower is not None and not need_replan and not bumper_hot:
                remain = list(follower.path[follower.idx:])
                if remain and _occupancy_cell_free(planner, x, y):
                    if _remaining_blocked(planner, (x, y), remain, self.inflation_m):
                        log.info("[Stage2] %s path blocked -> replan", leg)
                        last_reason = ReplanReason.PATH_BLOCKED
                        need_replan = True
                        replan_reason = ReplanReason.PATH_BLOCKED

            if need_replan:
                if not planner.ready:
                    last_reason = ReplanReason.NO_MAP
                    if not logged_no_map:
                        log.warning("[Stage2] %s NO_MAP -- occupancy not received, holding", leg)
                        logged_no_map = True
                    follower = None
                    client.send_cmd_vel(0.0, 0.0)
                    last_cmd_linear = 0.0
                    client.sleep(self.dt)
                    continue
                counts_cap = replan_reason == ReplanReason.PATH_BLOCKED or (
                    had_path
                    and replan_reason in (
                        ReplanReason.STALL,
                        ReplanReason.RAM,
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
                planner.clear_start_hole((x, y), exclude_m=0.20)
                path = self._plan_path(planner, (x, y), cur_goal, allow_thin=True)
                if path:
                    follower = PathFollower(path, passthrough=self.cruise)
                    had_path = True
                    last_reason = replan_reason or "ok"
                    self.last_path = tuple(path)
                    self.last_method = planner.last_method
                    log.info("[Stage2] occupancy %s path %d waypoints (clr=%.2f)",
                             planner.last_method or "dstar", len(path), advice.clearance_m)
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
                    cmd = _frontal_escape(planner, (x, y, yaw), bumper_hot)
                    if cmd is None and ram_hot:
                        cmd = _reverse_on_road(planner, (x, y, yaw))
                    if cmd is None and _should_backup(planner, (x, y, yaw), advice):
                        cmd = _reverse_on_road(planner, (x, y, yaw))
                    if cmd is not None:
                        client.send_cmd_vel(cmd[0], cmd[1])
                        last_cmd_linear = cmd[0]
                    else:
                        client.send_cmd_vel(0.0, 0.0)
                        last_cmd_linear = 0.0
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

            cmd = _frontal_escape(planner, (x, y, yaw), bumper_hot)
            if cmd is None and ram_hot:
                cmd = _reverse_on_road(planner, (x, y, yaw))
            if cmd is not None:
                linear, angular = cmd
            else:
                if bumper_hot:
                    linear, angular = 0.0, 0.0
                else:
                    linear *= advice.speed_scale
                    if (
                        abs(advice.yaw_bias) > 1e-9
                        and _occupancy_cell_free(planner, x, y)
                    ):
                        angular += YAW_BIAS_GAIN * advice.yaw_bias
                    if linear > 1e-6:
                        imu = client.get_imu() if hasattr(client, "get_imu") else None
                        linear, angular = governor.limit(
                            linear, angular, imu=imu, dt=self.dt)

            client.send_cmd_vel(linear_x=linear, angular_z=angular)
            last_cmd_linear = linear

            now = client.now_s()
            if now - last_log_t >= 1.0:
                extra_s = " clr=%.1f replan=%d" % (advice.clearance_m, n_replan)
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
