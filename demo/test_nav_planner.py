#!/usr/bin/env python3
"""Numpy-only checks for D* Lite repair, lidar XY transform, and path-blocked.

Run from repo root or demo/:
  python3 demo/test_nav_planner.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from mock_agent_navigation import (
    OccupancyGridPlanner,
    LIDAR_CLUSTER_M,
    OVERLAY_TTL_S,
    frontal_hit,
    lidar_xyz_to_obstacles,
    path_is_blocked,
    pointcloud2_xyz,
)


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

    ground = np.array([[1.0, 0.0, 0.02]], dtype=np.float32)
    _check(lidar_xyz_to_obstacles(ground, (0.0, 0.0, 0.0)) == [], "ground hit should drop")

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


def main():
    tests = [
        test_empty_corridor,
        test_wall_repair_goes_around,
        test_lidar_xyz_to_world,
        test_path_blocked_and_frontal,
        test_pointcloud2_xyz,
        test_overlay_persists_across_occupancy_reload,
        test_overlay_goal_hole,
        test_overlay_expiry,
        test_no_path_status,
        test_overlay_blocks_plan,
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
