#!/usr/bin/env python3
"""Numpy-only checks for D* Lite, lidar overlay, SpeedGovernor, and search-area nav.

Run from repo root or demo/:
  python3 demo/test_nav_planner.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from mock_agent_navigation import (
    OccupancyGridPlanner,
    SpeedGovernor,
    LIDAR_CLUSTER_M,
    OVERLAY_TTL_S,
    frontal_hit,
    lidar_xyz_to_obstacles,
    path_is_blocked,
    pointcloud2_xyz,
    recovery_cmd,
    resolve_xy,
)

from retrieval_state import RetrievalState


class _Fail(Exception):
    pass


def _check(cond, msg):
    if not cond:
        raise _Fail(msg)


def _planner(h=40, w=80, res=0.05):
    grid = np.zeros((h, w), dtype=np.int8)
    p = OccupancyGridPlanner()
    _check(p.update_from_array(grid, res, 0.0, 0.0), "update_from_array failed")
    return p, grid


def _cell_xy(p, row, col):
    return p._c2w(row, col)


def test_empty_corridor():
    p, _ = _planner()
    start = _cell_xy(p, 20, 5)
    goal = _cell_xy(p, 20, 70)
    path = p.plan(start, goal, inflation_m=0.0, margin_m=20.0, snap_m=1.0)
    _check(path, "empty corridor: no path")
    _check(p.last_method == "dstar", "empty corridor: expected D* Lite, got %s" % p.last_method)
    _check(math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) < 0.2,
           "empty corridor: last waypoint not at goal")


def test_wall_repair_goes_around():
    p, grid = _planner()
    start = _cell_xy(p, 8, 5)
    goal = _cell_xy(p, 8, 70)
    path1 = p.plan(start, goal, inflation_m=0.0, margin_m=20.0, snap_m=1.0)
    _check(path1, "pre-wall: no path")
    _check(p.last_method == "dstar", "pre-wall: expected dstar")

    # Wall across most of the corridor with a gap at high row (around).
    grid[0:28, 40] = 100
    path2 = p.plan(start, goal, inflation_m=0.0, margin_m=20.0, snap_m=1.0)
    _check(path2, "post-wall: no path")
    _check(p.last_method == "dstar", "post-wall: expected dstar repair, got %s" % p.last_method)

    wall_hits = 0
    for wx, wy in path2:
        r, c = p._w2c(wx, wy)
        if c == 40 and r < 28:
            wall_hits += 1
    _check(wall_hits == 0, "post-wall: path still crosses the wall (%d cells)" % wall_hits)

    rows = [p._w2c(wx, wy)[0] for wx, wy in path2]
    _check(max(rows) >= 28, "post-wall: path did not use the gap (max row=%s)" % max(rows))


def test_lidar_xyz_to_world():
    xyz = np.array([[2.0, 0.0, 0.5]], dtype=np.float32)
    discs = lidar_xyz_to_obstacles(xyz, (0.0, 0.0, 0.0))
    _check(len(discs) == 1, "yaw0: expected 1 disc, got %s" % discs)
    wx, wy, rad = discs[0]
    _check(abs(wx - 2.0) < LIDAR_CLUSTER_M and abs(wy) < LIDAR_CLUSTER_M,
           "yaw0: disc at (%s, %s), expected ~(2, 0)" % (wx, wy))
    _check(rad > 0, "yaw0: radius")

    discs = lidar_xyz_to_obstacles(xyz, (0.0, 0.0, math.pi / 2))
    _check(len(discs) == 1, "yaw90: expected 1 disc")
    wx, wy, _ = discs[0]
    _check(abs(wx) < LIDAR_CLUSTER_M and abs(wy - 2.0) < LIDAR_CLUSTER_M,
           "yaw90: disc at (%s, %s), expected ~(0, 2)" % (wx, wy))

    ground = np.array([[1.0, 0.0, -0.5]], dtype=np.float32)
    _check(lidar_xyz_to_obstacles(ground, (0.0, 0.0, 0.0)) == [], "ground hit should drop")
    equator = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    _check(len(lidar_xyz_to_obstacles(equator, (0.0, 0.0, 0.0))) == 1, "z=0 car body should keep")

    far = np.array([[25.0, 0.0, 0.5]], dtype=np.float32)
    _check(lidar_xyz_to_obstacles(far, (0.0, 0.0, 0.0)) == [], "far hit should drop")


def test_path_blocked_and_frontal():
    p, _ = _planner()
    a = _cell_xy(p, 20, 10)
    b = _cell_xy(p, 20, 50)
    _check(not path_is_blocked(p, [a, b], extra_obstacles=None, inflation_m=0.0),
           "clear segment should not be blocked")
    mid = _cell_xy(p, 20, 30)
    discs = [(mid[0], mid[1], 0.4)]
    _check(path_is_blocked(p, [a, b], extra_obstacles=discs, inflation_m=0.0),
           "disc on segment should block")

    pose = (0.0, 0.0, 0.0)
    _check(frontal_hit([(0.8, 0.0, 0.2)], pose), "disc ahead should be a frontal hit")
    _check(not frontal_hit([(-2.0, 0.0, 0.2)], pose), "disc behind should not be frontal")
    _check(not frontal_hit([(0.8, 2.0, 0.1)], pose), "disc to the side should not be frontal")


def test_parking_lot_path_goes_around_unknown():
    p, grid = _planner(h=40, w=50, res=0.25)
    grid[:, :] = -1
    grid[:, 2] = 0
    grid[2, :] = 0
    grid[:, 47] = 0
    _check(p.update_from_array(grid, 0.25, 0.0, 0.0), "reload parking map")
    start = _cell_xy(p, 30, 2)
    goal = _cell_xy(p, 30, 30)
    path = p.plan(start, goal, inflation_m=0.0, margin_m=20.0, snap_m=8.0, snap_goal_m=25.0)
    _check(path, "parking lot: no path")
    _check(len(path) >= 2, "parking lot: expected an L, got %d waypoints" % len(path))
    chord = math.hypot(path[-1][0] - start[0], path[-1][1] - start[1])
    _check(chord > 1.0, "parking lot: snapped goal should be away from start")
    for wx, wy in path:
        r, c = p._w2c(wx, wy)
        _check(grid[r, c] == 0, "parking lot: waypoint on non-road cell %s" % ((r, c),))
        _check(
            not (8 <= r <= 28 and 8 <= c <= 40),
            "parking lot: path cut through the lot at %s" % ((r, c),),
        )
    _check(
        not path_is_blocked(p, [start] + list(path), inflation_m=0.0),
        "parking lot: on-road polyline should be clear",
    )


def test_off_road_snap_keeps_join_waypoint():
    p, grid = _planner(h=40, w=80, res=0.25)
    grid[:, :] = 100
    grid[20, 16:70] = 0
    _check(p.update_from_array(grid, 0.25, 0.0, 0.0), "reload occupied map")
    start = _cell_xy(p, 8, 20)
    goal = _cell_xy(p, 20, 60)
    path = p.plan(start, goal, inflation_m=0.0, margin_m=20.0, snap_m=8.0)
    _check(path, "off-road snap: no path")
    _check(len(path) >= 2, "off-road snap: expected join waypoint, got %d" % len(path))
    jr, jc = p._w2c(path[0][0], path[0][1])
    _check(grid[jr, jc] == 0, "join should sit on free road, cell=%s" % ((jr, jc),))
    _check(
        math.hypot(path[0][0] - start[0], path[0][1] - start[1]) > 0.3,
        "join should not be the off-road start",
    )
    _check(
        not path_is_blocked(p, path, inflation_m=0.0),
        "snapped road path should be clear",
    )


def test_pointcloud2_xyz():
    class _F:
        def __init__(self, name, offset):
            self.name = name
            self.offset = offset
            self.datatype = 7
            self.count = 1

    class _Msg:
        fields = [_F("x", 0), _F("y", 4), _F("z", 8)]
        height = 1
        width = 2
        point_step = 16
        is_bigendian = False
        data = np.zeros(32, dtype=np.uint8)
        data[0:12] = np.frombuffer(np.array([1.0, 0.0, 0.5], dtype=np.float32).tobytes(), dtype=np.uint8)
        data[16:28] = np.frombuffer(np.array([2.0, 0.5, 0.4], dtype=np.float32).tobytes(), dtype=np.uint8)

    xyz = pointcloud2_xyz(_Msg())
    _check(xyz is not None and xyz.shape == (2, 3), "pointcloud shape %s" % (None if xyz is None else str(xyz.shape)))
    _check(abs(float(xyz[0, 0]) - 1.0) < 1e-5 and abs(float(xyz[1, 1]) - 0.5) < 1e-5, "pointcloud values")


def test_overlay_persists_across_occupancy_reload():
    p, grid = _planner()
    hit = _cell_xy(p, 20, 40)
    p.stamp_overlay([(hit[0], hit[1], 0.4)], now_s=1.0)
    r, c = p._w2c(hit[0], hit[1])
    _check(bool(p._overlay[r, c]), "stamp should block overlay cell")
    grid2 = grid.copy()
    _check(p.update_from_array(grid2, 0.05, 0.0, 0.0), "reload same origin")
    _check(bool(p._overlay[r, c]), "overlay should survive same-origin occupancy reload")
    _check(p.update_from_array(grid2, 0.05, 10.0, 10.0), "reload new origin")
    _check(not bool(p._overlay[r, c]), "overlay should reset when origin changes")


def test_overlay_goal_hole():
    p, _ = _planner()
    goal = _cell_xy(p, 20, 40)
    p.stamp_overlay([(goal[0], goal[1], 0.4)], now_s=1.0)
    gr, gc = p._w2c(goal[0], goal[1])
    _check(bool(p._overlay[gr, gc]), "goal cell stamped")
    p.clear_goal_hole(goal, exclude_m=0.6)
    _check(not bool(p._overlay[gr, gc]), "goal hole should free the goal cell")


def test_overlay_expiry():
    p, _ = _planner()
    hit = _cell_xy(p, 12, 12)
    p.stamp_overlay([(hit[0], hit[1], 0.4)], now_s=0.0)
    r, c = p._w2c(hit[0], hit[1])
    p.expire_overlay(now_s=OVERLAY_TTL_S)
    _check(bool(p._overlay[r, c]), "should still be blocked at TTL boundary")
    p.expire_overlay(now_s=OVERLAY_TTL_S + 0.1)
    _check(not bool(p._overlay[r, c]), "stale overlay cell should expire")


def test_no_path_status():
    p, grid = _planner()
    grid[:, 40] = 100
    start = _cell_xy(p, 20, 5)
    goal = _cell_xy(p, 20, 70)
    path = p.plan(start, goal, inflation_m=0.0, margin_m=20.0, snap_m=1.0)
    _check(path is None, "walled corridor should have no path")
    _check(p.last_status == "no_path", "last_status=%s" % p.last_status)
    empty = OccupancyGridPlanner()
    _check(empty.plan((0.0, 0.0), (1.0, 0.0)) is None, "unready plan should be None")
    _check(empty.last_status == "no_map", "unready last_status=%s" % empty.last_status)


def test_overlay_blocks_plan():
    p, _ = _planner()
    start = _cell_xy(p, 20, 5)
    goal = _cell_xy(p, 20, 70)
    wall = _cell_xy(p, 20, 40)
    p.stamp_overlay([(wall[0], wall[1], 0.6)], now_s=1.0)
    path = p.plan(start, goal, inflation_m=0.0, margin_m=20.0, snap_m=1.0)
    _check(path, "overlay wall should still leave a path around")
    hits = 0
    for wx, wy in path:
        r, c = p._w2c(wx, wy)
        if bool(p._overlay[r, c]):
            hits += 1
    _check(hits == 0, "path crossed overlay (%d cells)" % hits)


class _V:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        if w is not None:
            self.w = float(w)


class _Imu:
    """Duck-typed sensor_msgs/Imu. roll is about +X; az defaults to +g (level, gravity-on)."""

    def __init__(self, roll=0.0, ay=0.0, az=9.81, wx=0.0, wz=0.0):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        self.orientation = _V(x=sr, y=0.0, z=0.0, w=cr)
        self.angular_velocity = _V(x=wx, y=0.0, z=wz)
        self.linear_acceleration = _V(x=0.0, y=ay, z=az)


def _ackermann_ok(linear, angular):
    if linear <= 1e-3:
        return True
    return abs(angular) <= 1.8 * linear / SpeedGovernor.STEERING_TRACK + 1e-9


def test_governor_straight_cruise():
    g = SpeedGovernor()
    lin, ang = g.limit(1.0, 0.0, imu=_Imu(), xyz=np.zeros((0, 3)), dt=0.1)
    _check(abs(lin - 1.0) < 1e-9, "straight quiet IMU should keep cruise, got %s" % lin)
    _check(abs(ang) < 1e-12, "straight should not invent angular, got %s" % ang)


def test_governor_high_yaw_caps_and_ackermann():
    g = SpeedGovernor()
    lin, ang = g.limit(1.0, 1.2, imu=None, xyz=None, dt=0.1)
    _check(lin < 0.70, "high |angular| should cut cruise, got v=%s" % lin)
    _check(lin > 0.30, "high |angular| should still move, got v=%s" % lin)
    _check(_ackermann_ok(lin, ang), "recap must stay Ackermann-valid v=%s w=%s" % (lin, ang))


def test_governor_roll_and_ay_slower_than_level():
    cruise = 1.0
    w = 0.15
    level, _ = SpeedGovernor().limit(cruise, w, imu=_Imu(), dt=0.1)
    rolled, _ = SpeedGovernor().limit(cruise, w, imu=_Imu(roll=0.28), dt=0.1)
    lat, _ = SpeedGovernor().limit(cruise, w, imu=_Imu(ay=3.2), dt=0.1)
    _check(rolled < level - 0.15, "roll should cut vs level (%s vs %s)" % (rolled, level))
    _check(lat < level - 0.15, "high a_y should cut vs level (%s vs %s)" % (lat, level))


def test_governor_curb_lidar_slower_than_high_z():
    cruise = 1.0
    curb = np.array([[0.6, 0.0, 0.08], [0.7, 0.05, 0.10], [0.55, -0.04, 0.06]], dtype=np.float32)
    high = np.array([[0.6, 0.0, 0.80], [0.7, 0.05, 0.90], [0.55, -0.04, 1.00]], dtype=np.float32)
    v_curb, _ = SpeedGovernor().limit(cruise, 0.0, imu=_Imu(), xyz=curb, dt=0.1)
    v_high, _ = SpeedGovernor().limit(cruise, 0.0, imu=_Imu(), xyz=high, dt=0.1)
    _check(abs(v_high - cruise) < 1e-9, "high-Z hits are not curbs, got %s" % v_high)
    _check(v_curb < v_high - 0.3, "curb-Z look-ahead should slow (%s vs %s)" % (v_curb, v_high))
    _check(v_curb >= SpeedGovernor.V_FLOOR - 1e-9, "curb floor should still creep, got %s" % v_curb)


def test_governor_az_spike_latches():
    g = SpeedGovernor()
    spiked, _ = g.limit(1.0, 0.0, imu=_Imu(az=20.0), dt=0.1)
    _check(spiked <= SpeedGovernor.V_SPIKE + 1e-9, "spike should drop v, got %s" % spiked)
    held, _ = g.limit(1.0, 0.0, imu=_Imu(), dt=0.1)
    _check(held <= SpeedGovernor.V_SPIKE + 0.05, "latch should keep v low after quiet IMU, got %s" % held)
    fresh, _ = SpeedGovernor().limit(1.0, 0.0, imu=_Imu(), dt=0.1)
    _check(abs(fresh - 1.0) < 1e-9, "fresh quiet governor should cruise, got %s" % fresh)
    _check(held < fresh - 0.5, "latched call must be slower than a fresh quiet sample")


def test_governor_missing_sensors_never_raises():
    g = SpeedGovernor()
    lin, ang = g.limit(1.0, 1.0, imu=None, xyz=None, dt=0.1)
    _check(lin < 1.0, "missing sensors should still cap from commanded yaw, got %s" % lin)
    _check(_ackermann_ok(lin, ang), "missing-sensor recap must stay Ackermann-valid")
    held, _ = g.limit(0.30, 0.0, imu=None, xyz=None, dt=0.1)
    _check(held <= 0.30 + 1e-12, "governor must never raise linear (%s)" % held)


def test_search_tol_uses_radius():
    s = RetrievalState(hint_center=(1.0, 2.0), hint_radius=3.0)
    _check(abs(s.search_tol(0.7) - 3.0) < 1e-12, "radius 3 should be the stop distance, got %s" % s.search_tol(0.7))
    s2 = RetrievalState(grounding_xy=(0.0, 0.0))
    _check(abs(s2.search_tol(0.7) - 0.7) < 1e-12, "no radius -> default tol")
    s3 = RetrievalState(hint_center=(1.0, 2.0), hint_radius=0.4)
    _check(abs(s3.search_tol(0.7) - 0.7) < 1e-12, "small radius still at least default, got %s" % s3.search_tol(0.7))
    _check(s.search_xy() == (1.0, 2.0), "hint wins over grounding")
    s4 = RetrievalState(grounding_xy=(8.0, 9.0))
    _check(s4.search_xy() == (8.0, 9.0), "grounding fallback")
    s4.hint_center = (3.0, 4.0)
    _check(s4.search_xy() == (3.0, 4.0), "late reveal replaces grounding")


def test_callable_goal_retarget():
    box = {"xy": (1.0, 2.0)}
    def goal():
        return box["xy"]
    _check(resolve_xy(goal) == (1.0, 2.0), "first goal")
    box["xy"] = (9.0, 8.0)
    _check(resolve_xy(goal) == (9.0, 8.0), "updated callable goal")
    st = RetrievalState(grounding_xy=(1.0, 1.0))
    _check(resolve_xy(st.search_xy) == (1.0, 1.0), "bound search_xy")
    st.hint_center = (5.0, 6.0)
    _check(resolve_xy(st.search_xy) == (5.0, 6.0), "reveal retarget via bound method")
    _check(resolve_xy(None) is None, "missing goal")
    _check(resolve_xy((2.5, 3.5)) == (2.5, 3.5), "plain tuple")


def test_recovery_cmd_zeros_on_frontal():
    pose = (0.0, 0.0, 0.0)
    hit = [(0.6, 0.0, 0.2)]
    _check(frontal_hit(hit, pose), "bumper disc should be a frontal hit")
    lin, ang = recovery_cmd(True)
    _check(lin == 0.0, "recovery must command zero linear, got lin=%s" % lin)
    _check(ang == 0.0, "recovery has no yaw")
    lin2, ang2 = recovery_cmd(False)
    _check(lin2 == 0.0 and ang2 == 0.0, "no hit -> zero cmd")


def main():
    tests = [
        test_empty_corridor,
        test_wall_repair_goes_around,
        test_lidar_xyz_to_world,
        test_path_blocked_and_frontal,
        test_parking_lot_path_goes_around_unknown,
        test_off_road_snap_keeps_join_waypoint,
        test_pointcloud2_xyz,
        test_overlay_persists_across_occupancy_reload,
        test_overlay_goal_hole,
        test_overlay_expiry,
        test_no_path_status,
        test_overlay_blocks_plan,
        test_governor_straight_cruise,
        test_governor_high_yaw_caps_and_ackermann,
        test_governor_roll_and_ay_slower_than_level,
        test_governor_curb_lidar_slower_than_high_z,
        test_governor_az_spike_latches,
        test_governor_missing_sensors_never_raises,
        test_search_tol_uses_radius,
        test_callable_goal_retarget,
        test_recovery_cmd_zeros_on_frontal,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("ok  %s" % fn.__name__)
        except _Fail as e:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, e))
        except Exception as e:
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    if failed:
        print("%d failed" % failed)
        return 1
    print("all %d passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
