"""Synthetic pose/map/lidar drive loop for Stage2Navigator (no live ROS)."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from marc_agent.navigation_runtime import NavStatus, Stage2Navigator
from marc_agent.retrieval_state import search_area_approach_goal


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "nav_runtime_cases.yaml")


def _load():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _xyz_cloud(points):
    """Minimal PointCloud2 with FLOAT32 xyz for lidar_cloud_to_obstacles."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2:
        pts = pts.reshape(1, 3)
    n = int(pts.shape[0])
    step = 16
    payload = np.zeros(n * step, dtype=np.uint8)
    for i, row in enumerate(pts):
        payload[i * step:i * step + 12] = np.frombuffer(row[:3].tobytes(), dtype=np.uint8)

    class _F:
        def __init__(self, name, offset):
            self.name = name
            self.offset = offset
            self.datatype = 7
            self.count = 1

    class _Msg:
        fields = [_F("x", 0), _F("y", 4), _F("z", 8)]
        height = 1
        width = n
        point_step = step
        is_bigendian = False
        data = payload

    return _Msg()


class _Pos:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.z = 0.0


class _Origin:
    def __init__(self, x, y):
        self.position = _Pos(x, y)


class _Info:
    def __init__(self, width, height, res, ox, oy):
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(res)
        self.origin = _Origin(ox, oy)


class _Occ:
    def __init__(self, grid, res, ox, oy):
        arr = np.asarray(grid, dtype=np.int8)
        h, w = arr.shape
        self.info = _Info(w, h, res, ox, oy)
        self.data = arr.reshape(-1).tolist()


class FakeClient:
    """Unicycle client paced by ``sleep``; occupancy is optional."""

    def __init__(self, start, occ=None, stop_after_s=None, lidar_xyz=None, lidar_after_s=None):
        self._pose = [float(start[0]), float(start[1]), float(start[2])]
        self._t = 0.0
        self._v = 0.0
        self._w = 0.0
        self.is_running = True
        self.cmds = []
        self._occ = occ
        self.stop_after_s = stop_after_s
        self._lidar_xyz = lidar_xyz
        self._lidar_after_s = lidar_after_s

    def now_s(self):
        return self._t

    def sleep(self, dt):
        dt = float(dt)
        yaw = self._pose[2]
        self._pose[0] += self._v * math.cos(yaw) * dt
        self._pose[1] += self._v * math.sin(yaw) * dt
        self._pose[2] += self._w * dt
        self._t += dt
        if self.stop_after_s is not None and self._t >= float(self.stop_after_s):
            self.is_running = False

    def get_world_pose(self):
        return (self._pose[0], self._pose[1], self._pose[2])

    def send_cmd_vel(self, linear_x=0.0, angular_z=0.0):
        self._v = float(linear_x)
        self._w = float(angular_z)
        self.cmds.append((self._v, self._w))

    def get_occupancy_map(self):
        return self._occ

    def get_lidar(self):
        if self._lidar_xyz is None:
            return None
        if self._lidar_after_s is not None and self._t < float(self._lidar_after_s):
            return None
        return _xyz_cloud(self._lidar_xyz)

    def get_imu(self):
        return None


def _grid_for(case):
    if case.get("kind") == "no_map":
        return None
    h, w = case["grid_hw"]
    grid = np.zeros((int(h), int(w)), dtype=np.int8)
    if "wall_col" in case:
        grid[:, int(case["wall_col"])] = 100
    ox, oy = case["origin"]
    return _Occ(grid, case["res"], ox, oy)


class NavigationRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _load()

    def test_runtime_step_budget(self):
        steps = self.data["runtime_steps"]
        self.assertGreaterEqual(len(steps), 25)

    def test_drive_cases(self):
        for case in self.data["drive_cases"]:
            with self.subTest(case=case["id"]):
                occ = _grid_for(case)
                client = FakeClient(
                    case["start"], occ=occ, stop_after_s=case.get("stop_after_s"))
                nav = Stage2Navigator(
                    client,
                    dt=float(case["dt"]),
                    inflation_m=0.0,
                    exclude_m=0.4,
                    stuck_time_s=30.0,
                    replan_cap=20,
                )
                extra = [tuple(item) for item in case.get("extra_obstacles") or []]
                result = nav.navigate_to(
                    tuple(case["goal"]),
                    float(case["deadline_s"]),
                    leg="search",
                    tol=0.7,
                    extra_obstacles=extra,
                )
                allowed = case.get("expect_status_in")
                if allowed:
                    self.assertIn(result.status.value, allowed)
                else:
                    self.assertEqual(case["expect_status"], result.status.value)
                if "expect_replan_max" in case:
                    self.assertLessEqual(result.replan_count, int(case["expect_replan_max"]))
                if case.get("expect_frontal_zero"):
                    self.assertTrue(client.cmds, "expected cmd_vel samples")
                    self.assertTrue(
                        all(abs(v) < 1e-9 for v, _w in client.cmds),
                        "frontal hazard must command zero linear, got %s" % client.cmds[:8],
                    )

    def test_live_approach_goal_not_centroid(self):
        pose = (0.0, 0.0)
        center = (12.0, 0.0)
        goal = search_area_approach_goal(center, 3.0, pose)
        self.assertLess(abs(goal[0] - 9.0), 1e-6)
        self.assertNotEqual(goal, center)

    def test_production_adapter_skips_yaml_nav_inputs(self):
        root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root, "demo", "participant_app.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertNotIn("load_obstacles", src)
        self.assertNotIn("load_mock_data", src)
        self.assertNotIn("mock_demo_data.yaml", src)
        self.assertNotIn("_DELIVERY_COORD", src)
        self.assertIn("Stage2Navigator", src)
        self.assertIn("search_area_approach_goal", src)
        self.assertIn("snapshot_approach_goal", src)
        self.assertIn("expected_intent=Intent.BRING", src)
        self.assertIn("MARC_STAGE2_MOCK_VLA", src)

    def test_unknown_cells_are_not_traversable(self):
        grid = np.full((20, 40), -1, dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient([1.0, 5.0, 0.0], occ=occ)
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.0,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=5,
        )
        result = nav.navigate_to((8.0, 5.0), 2.0, leg="search", tol=0.7)
        self.assertEqual("no_path", result.status.value)

    def test_no_path_holds_do_not_burn_replan_cap(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        grid[:, 20] = 100
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient([2.0, 5.0, 0.0], occ=occ)
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.0,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=3,
        )
        result = nav.navigate_to((16.0, 5.0), 2.0, leg="search", tol=0.7)
        self.assertEqual("no_path", result.status.value)
        self.assertLessEqual(result.replan_count, 1)

    def test_inflation_fallback_opens_narrow_corridor(self):
        res = 0.1
        h, w = 30, 80
        grid = np.full((h, w), 100, dtype=np.int8)
        grid[15, :] = 0
        occ = _Occ(grid, res, 0.0, 0.0)
        client = FakeClient([0.5, 1.55, 0.0], occ=occ)
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=5,
        )
        result = nav.navigate_to((7.0, 1.55), 40.0, leg="search", tol=0.7)
        self.assertEqual("reached", result.status.value)

    def test_start_hole_clears_lidar_overlay(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.zeros((20, 40), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        start = (2.0, 5.0)
        goal = (16.0, 5.0)
        planner.stamp_overlay([(start[0], start[1], 2.0)])
        blocked = planner.plan(start, goal, inflation_m=0.0, snap_m=0.5, margin_m=12.0)
        self.assertIsNone(blocked)
        planner.clear_start_hole(start, exclude_m=2.5)
        opened = planner.plan(start, goal, inflation_m=0.0, snap_m=0.5, margin_m=12.0)
        self.assertIsNotNone(opened)

    def test_snap_uses_grid_resolution(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.full((20, 20), 100, dtype=np.int8)
        grid[10, 2] = 0
        grid[10, 6] = 0
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        start = planner._c2w(10, 2)
        goal = planner._c2w(10, 6)
        grid[10, 2] = 100
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        missed = planner.plan(start, goal, inflation_m=0.0, snap_m=1.0, margin_m=12.0)
        self.assertIsNone(missed)
        found = planner.plan(start, goal, inflation_m=0.0, snap_m=2.5, margin_m=12.0)
        self.assertIsNotNone(found)

    def test_off_road_start_does_not_cut_across_blocked_cells(self):
        from mock_agent_navigation import OccupancyGridPlanner

        res = 0.25
        grid = np.full((40, 80), 100, dtype=np.int8)
        grid[20, 16:70] = 0
        occ = _Occ(grid, res, 0.0, 0.0)
        helper = OccupancyGridPlanner()
        helper.update_from_array(grid, res, 0.0, 0.0)
        pose = helper._c2w(8, 20)
        goal = helper._c2w(20, 60)
        client = FakeClient([pose[0], pose[1], 0.0], occ=occ)
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.0,
            exclude_m=0.6,
            stuck_time_s=30.0,
            replan_cap=8,
        )
        result = nav.navigate_to(goal, 2.0, leg="search", tol=0.7)
        self.assertEqual("no_path", result.status.value)

    def test_parking_lot_plan_goes_around_on_road(self):
        from mock_agent_navigation import OccupancyGridPlanner, path_is_blocked

        res = 0.25
        grid = np.full((40, 50), -1, dtype=np.int8)
        grid[:, 2] = 0
        grid[2, :] = 0
        grid[:, 47] = 0
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        start = planner._c2w(30, 2)
        goal = planner._c2w(30, 30)
        nav = Stage2Navigator(
            FakeClient([start[0], start[1], 0.0]),
            dt=0.10,
            inflation_m=0.0,
            exclude_m=0.6,
        )
        path = nav._plan_path(planner, start, goal)
        self.assertIsNotNone(path)
        self.assertGreaterEqual(len(path), 2)
        for wx, wy in path:
            row, col = planner._w2c(wx, wy)
            self.assertEqual(0, int(grid[row, col]))
            if 8 <= row <= 28 and 8 <= col <= 40:
                self.fail("path entered the parking interior at %s" % ((row, col),))
        self.assertFalse(path_is_blocked(planner, [start] + list(path), inflation_m=0.0))

    def test_off_road_start_joins_road(self):
        from mock_agent_navigation import OccupancyGridPlanner

        res = 0.25
        grid = np.full((40, 80), 100, dtype=np.int8)
        grid[20, 16:70] = 0
        occ = _Occ(grid, res, 0.0, 0.0)
        helper = OccupancyGridPlanner()
        helper.update_from_array(grid, res, 0.0, 0.0)
        pose = helper._c2w(20, 20)
        goal = helper._c2w(20, 60)
        client = FakeClient([pose[0], pose[1], 0.0], occ=occ)
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.0,
            exclude_m=0.6,
            stuck_time_s=30.0,
            replan_cap=8,
        )
        result = nav.navigate_to(goal, 25.0, leg="search", tol=0.7)
        self.assertEqual("reached", result.status.value)

    def test_path_blocked_zeros_linear_after_motion(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient(
            [1.0, 5.0, 0.0],
            occ=occ,
            lidar_xyz=[[2.0, 0.0, 0.4]],
            lidar_after_s=0.3,
        )
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=8,
        )
        nav.navigate_to((16.0, 5.0), 5.0, leg="search", tol=0.7)
        moving = [i for i, (v, _w) in enumerate(client.cmds) if abs(v) > 1e-6]
        stopped = [i for i, (v, _w) in enumerate(client.cmds) if abs(v) < 1e-9]
        self.assertTrue(moving, "expected forward motion before the lidar wall")
        self.assertTrue(stopped, "path_blocked / dwell must command zero linear")
        self.assertGreater(max(stopped), min(moving))

    def test_overlay_live_skips_thin_inflation(self):
        res = 0.1
        h, w = 30, 80
        grid = np.full((h, w), 100, dtype=np.int8)
        grid[15, :] = 0
        occ = _Occ(grid, res, 0.0, 0.0)
        client = FakeClient(
            [0.5, 1.55, 0.0],
            occ=occ,
            lidar_xyz=[[0.4, 1.2, 0.4]],
        )
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=5,
        )
        result = nav.navigate_to((7.0, 1.55), 5.0, leg="search", tol=0.7)
        self.assertEqual("no_path", result.status.value)

    def test_start_hole_leaves_disc_ahead_blocked(self):
        from mock_agent_navigation import OccupancyGridPlanner

        res = 0.05
        grid = np.zeros((80, 160), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        start = (2.0, 2.0)
        ahead = (2.4, 2.0)
        planner.stamp_overlay([(ahead[0], ahead[1], 0.15)], now_s=1.0)
        planner.clear_start_hole(start, exclude_m=0.20)
        row, col = planner._w2c(ahead[0], ahead[1])
        self.assertTrue(bool(planner._overlay[row, col]))


if __name__ == "__main__":
    unittest.main()
