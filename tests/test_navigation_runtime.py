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

from marc_agent.navigation_runtime import (
    NavStatus,
    Stage2Navigator,
    _frontal_escape,
    _plan_attempts,
    _ram_bumper_disc,
)
from marc_agent.retrieval_state import search_area_approach_goal


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "nav_runtime_cases.yaml")


def _load():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _xyz_cloud(points):
    """Minimal PointCloud2 with FLOAT32 xyz for lidar_safety.advise."""
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

    def __init__(
        self,
        start,
        occ=None,
        stop_after_s=None,
        lidar_xyz=None,
        lidar_after_s=None,
        lidar_origin_max_m=None,
        lidar_until_s=None,
        lidar_world_xyz=None,
        immobile=False,
    ):
        self._pose = [float(start[0]), float(start[1]), float(start[2])]
        self._t = 0.0
        self._v = 0.0
        self._w = 0.0
        self.is_running = True
        self.cmds = []
        self._occ = occ
        self.stop_after_s = stop_after_s
        self._lidar_xyz = lidar_xyz
        self._lidar_world = lidar_world_xyz
        self._lidar_after_s = lidar_after_s
        self._lidar_origin = (float(start[0]), float(start[1]))
        self._lidar_origin_max_m = lidar_origin_max_m
        self._lidar_until_s = lidar_until_s
        self._lidar_left_origin = False
        self._immobile = bool(immobile)

    def now_s(self):
        return self._t

    def sleep(self, dt):
        dt = float(dt)
        if not self._immobile:
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
        if self._lidar_after_s is not None and self._t < float(self._lidar_after_s):
            return None
        if self._lidar_until_s is not None and self._t >= float(self._lidar_until_s):
            return None
        if self._lidar_origin_max_m is not None:
            away = math.hypot(
                self._pose[0] - self._lidar_origin[0],
                self._pose[1] - self._lidar_origin[1],
            )
            if away >= float(self._lidar_origin_max_m):
                self._lidar_left_origin = True
            if self._lidar_left_origin:
                return None
        if self._lidar_world is not None:
            x, y, yaw = self._pose
            cc, ss = math.cos(yaw), math.sin(yaw)
            rows = []
            for item in self._lidar_world:
                dx = float(item[0]) - x
                dy = float(item[1]) - y
                rows.append([dx * cc + dy * ss, -dx * ss + dy * cc, float(item[2])])
            return _xyz_cloud(rows)
        if self._lidar_xyz is None:
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
                        all(v <= 1e-9 for v, _w in client.cmds),
                        "frontal hazard must not command forward, got %s" % client.cmds[:8],
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
        self.assertIn("STAGE2_INFLATION_M = 0.55", src)
        self.assertIn("peek_yolo_handoff", src)
        self.assertIn("handoff_fn", src)

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

    def test_plan_attempts_keep_configured_inflation(self):
        attempts = _plan_attempts(0.25)
        self.assertTrue(all(abs(infl - 0.25) < 1e-9 for infl, _margin in attempts))
        self.assertTrue(all(infl > 0.0 for infl, _margin in attempts))
        zero = _plan_attempts(0.0)
        self.assertTrue(all(abs(infl) < 1e-9 for infl, _margin in zero))

    def test_free_approach_does_not_sit_in_occupied_cell(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.zeros((20, 40), dtype=np.int8)
        grid[:, 20:25] = 100
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        start = (2.0, 5.0)
        goal = (12.0, 5.0)
        xy = planner.free_approach_xy(start, goal, inflation_m=0.25)
        self.assertIsNotNone(xy)
        self.assertTrue(planner.cell_free(xy[0], xy[1], 0.25))
        self.assertLess(xy[0], 10.0)
        occupied = planner.nearest_free_xy(goal, inflation_m=0.25)
        self.assertIsNotNone(occupied)
        self.assertTrue(planner.cell_free(occupied[0], occupied[1], 0.25))

    def test_plan_path_keeps_inflation_floor(self):
        from mock_agent_navigation import OccupancyGridPlanner

        res = 0.1
        h, w = 30, 80
        grid = np.full((h, w), 100, dtype=np.int8)
        grid[15, :] = 0
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        start, goal = (0.5, 1.55), (7.0, 1.55)
        nav = Stage2Navigator(FakeClient([0.5, 1.55, 0.0]), inflation_m=0.25)
        self.assertIsNone(nav._plan_path(planner, start, goal, allow_thin=True))
        self.assertIsNotNone(
            planner.plan(start, goal, inflation_m=0.0, snap_m=8.0, margin_m=12.0)
        )

    def test_body_inflation_plans_wide_corridor(self):
        from mock_agent_navigation import OccupancyGridPlanner

        res = 0.1
        h, w = 30, 80
        grid = np.full((h, w), 100, dtype=np.int8)
        grid[10:21, :] = 0
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        start, goal = (0.5, 1.55), (7.0, 1.55)
        nav = Stage2Navigator(FakeClient([0.5, 1.55, 0.0]), inflation_m=0.25)
        path = nav._plan_path(planner, start, goal, allow_thin=True)
        self.assertIsNotNone(path)

    def test_inflate_cells_never_rounds_positive_pad_to_zero(self):
        from mock_agent_navigation import inflate_cells

        self.assertEqual(0, inflate_cells(0.0, 0.5))
        self.assertEqual(1, inflate_cells(0.25, 0.5))
        self.assertEqual(1, inflate_cells(0.45, 0.5))
        self.assertEqual(2, inflate_cells(0.55, 0.5))
        self.assertEqual(9, inflate_cells(0.45, 0.05))
        self.assertEqual(11, inflate_cells(0.55, 0.05))

    def test_fine_grid_free_chord_not_eaten_by_inflation_crop(self):
        from mock_agent_navigation import OccupancyGridPlanner, path_is_blocked

        res = 0.05
        grid = np.zeros((80, 200), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        start = (1.0, 2.0)
        goal = (8.0, 2.0)
        self.assertFalse(
            path_is_blocked(
                planner, [start, goal], inflation_m=0.55, ignore_overlay=True,
            ),
            "0.55 m pad must not false-block a free 5 cm-grid chord",
        )

    def test_coarse_grid_inflation_keeps_path_off_the_curb(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.full((20, 40), 100, dtype=np.int8)
        grid[8:12, :] = 0
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        start = planner._c2w(8, 4)
        goal = planner._c2w(8, 32)
        padded = planner.plan(start, goal, inflation_m=0.25, snap_m=8.0, margin_m=12.0)
        self.assertIsNotNone(padded)
        for wx, wy in padded:
            row, col = planner._w2c(wx, wy)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if planner._in_bounds(nr, nc):
                    self.assertNotEqual(
                        100,
                        int(planner._grid[nr, nc]),
                        "padded path %s sits on the curb" % ((row, col),),
                    )
        raw = planner.plan(start, goal, inflation_m=0.0, snap_m=8.0, margin_m=12.0)
        self.assertIsNotNone(raw)
        hugged = False
        for wx, wy in raw:
            row, col = planner._w2c(wx, wy)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if planner._in_bounds(nr, nc) and int(planner._grid[nr, nc]) == 100:
                    hugged = True
        self.assertTrue(hugged, "0.0 inflation should still use the last free cell")

    def test_path_follower_does_not_skip_a_metre_ahead(self):
        from mock_agent_navigation import PathFollower

        follower = PathFollower([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)])
        _lin, _ang, _done, info = follower.compute((0.9, 0.0, 0.0))
        self.assertEqual(0, info["idx"])

    def test_path_follower_keeps_moving_inside_goal_tolerance(self):
        from mock_agent_navigation import PathFollower

        follower = PathFollower([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)])
        linear, angular, done, info = follower.compute((0.55, 0.0, 0.0))
        self.assertEqual(0, info["idx"])
        self.assertFalse(done)
        self.assertGreater(abs(linear) + abs(angular), 1e-6)

    def test_path_follower_advances_inside_waypoint_tolerance(self):
        from mock_agent_navigation import PathFollower

        follower = PathFollower([(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)])
        _lin, _ang, _done, info = follower.compute((0.35, 0.0, 0.0))
        self.assertEqual(1, info["idx"])
        self.assertEqual((2.0, 0.0), info["waypoint"])

    def test_path_follower_stops_on_final_inside_goal_tolerance(self):
        from mock_agent_navigation import PathFollower

        follower = PathFollower([(2.0, 0.0)])
        linear, angular, done, info = follower.compute((1.45, 0.0, 0.0))
        self.assertEqual(0, info["idx"])
        self.assertTrue(done)
        self.assertEqual(0.0, linear)
        self.assertEqual(0.0, angular)

    def test_goal_hole_does_not_erase_occupancy_cars(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.zeros((20, 40), dtype=np.int8)
        grid[10, 20] = 100
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        before = planner._grid.copy()
        planner.clear_goal_hole(planner._c2w(10, 20), exclude_m=2.5)
        np.testing.assert_array_equal(before, planner._grid)

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

    def test_spawn_pad_lidar_ring_still_moves(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        ring = []
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            ring.append([0.30 * math.cos(rad), 0.30 * math.sin(rad), 0.4])
        client = FakeClient(
            [1.0, 5.0, 0.0],
            occ=occ,
            lidar_xyz=ring,
            lidar_origin_max_m=0.55,
        )
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=20,
        )
        result = nav.navigate_to((16.0, 5.0), 20.0, leg="search", tol=0.7)
        self.assertTrue(any(v > 1e-6 for v, _w in client.cmds))
        x, y, _yaw = client.get_world_pose()
        self.assertGreaterEqual(math.hypot(x - 1.0, y - 5.0), 0.5)
        self.assertIn(result.status.value, ("reached", "timeout"))

    def test_spawn_pad_off_road_no_creep(self):
        grid = np.full((20, 40), 100, dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        ring = []
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            ring.append([0.30 * math.cos(rad), 0.30 * math.sin(rad), 0.4])
        client = FakeClient([1.0, 5.0, 0.0], occ=occ, lidar_xyz=ring)
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=8,
        )
        result = nav.navigate_to((16.0, 5.0), 2.0, leg="search", tol=0.7)
        self.assertEqual("no_path", result.status.value)
        self.assertFalse(
            any(v > 1e-6 for v, _w in client.cmds),
            "off-road spawn must not creep, got %s" % client.cmds[:8],
        )

    def test_far_corridor_hit_slows_not_freeze(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient(
            [1.0, 5.0, 0.0],
            occ=occ,
            lidar_world_xyz=[(3.0, 5.0, 0.4)],
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
        nav.navigate_to((16.0, 5.0), 2.0, leg="search", tol=0.7)
        after = [v for v, _w in client.cmds[4:] if v > 1e-6]
        self.assertTrue(after, "2 m corridor hit must keep forward motion, got %s" % client.cmds[:12])

    def test_close_corridor_hit_reverses(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient([1.0, 5.0, 0.0], occ=occ, lidar_xyz=[[0.5, 0.0, 0.4]])
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=8,
        )
        nav.navigate_to((16.0, 5.0), 3.0, leg="search", tol=0.7)
        reverse = [(v, w) for v, w in client.cmds if v < -1e-6]
        self.assertTrue(reverse, "0.5 m corridor hit must reverse, got %s" % client.cmds[:12])
        self.assertTrue(
            all(abs(w) < 1e-9 for _v, w in reverse),
            "frontal reverse must be straight, got %s" % reverse[:8],
        )

    def test_lidar_ahead_reverses_then_goes_around(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient(
            [3.0, 5.0, 0.0],
            occ=occ,
            lidar_xyz=[[0.5, 0.0, 0.4]],
            lidar_until_s=0.8,
        )
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=20,
        )
        result = nav.navigate_to((16.0, 5.0), 40.0, leg="search", tol=0.7)
        self.assertEqual("reached", result.status.value)
        reverse = [i for i, (v, _w) in enumerate(client.cmds) if v < -1e-6]
        forward = [i for i, (v, _w) in enumerate(client.cmds) if v > 1e-6]
        self.assertTrue(reverse, "expected reverse on free road with a frontal hit")
        self.assertTrue(forward, "expected forward after the bumper box cleared")
        self.assertLess(min(reverse), max(forward))
        self.assertTrue(
            all(abs(w) < 1e-9 for v, w in client.cmds if v < -1e-6),
            "frontal reverse must not yaw into the car",
        )
        path = result.path or ()
        self.assertTrue(
            any(abs(wy - 5.0) > 0.4 for _wx, wy in path),
            "after bumper clear, D* must leave the car line, got %s" % (path,),
        )

    def test_overlay_narrow_strip_keeps_dstar_detour(self):
        from mock_agent_navigation import OccupancyGridPlanner, path_is_blocked

        res = 0.1
        grid = np.full((40, 80), 100, dtype=np.int8)
        grid[2:32, :] = 0
        occ = _Occ(grid, res, 0.0, 0.0)
        helper = OccupancyGridPlanner()
        helper.update_from_array(grid, res, 0.0, 0.0)
        start = helper._c2w(16, 8)
        goal = helper._c2w(16, 70)
        disc = helper._c2w(16, 35) + (0.35,)
        hug = helper._c2w(3, 8)
        hug_goal = helper._c2w(3, 70)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        planner.stamp_overlay([disc], now_s=1.0)
        planner.clear_start_hole(start, exclude_m=0.20)
        curb = planner.plan(hug, hug_goal, inflation_m=0.0, snap_m=8.0, margin_m=12.0)
        self.assertIsNotNone(curb)
        chord = [hug] + list(curb)
        self.assertTrue(
            path_is_blocked(planner, chord, inflation_m=0.25, ignore_overlay=True),
            "0.25 m occupancy inflation should reject the curb-hugging overlay detour",
        )
        self.assertFalse(
            path_is_blocked(planner, chord, inflation_m=0.0, ignore_overlay=True),
            "inflation-0 occupancy must still see the on-road detour",
        )
        live = OccupancyGridPlanner()
        live.update_from_array(grid, res, 0.0, 0.0)
        nav = Stage2Navigator(
            FakeClient([start[0], start[1], 0.0], occ=occ),
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=20,
        )
        self.assertIsNotNone(nav._plan_path(live, start, goal))
        client = FakeClient([start[0], start[1], 0.0], occ=occ)
        result = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=20,
        ).navigate_to(goal, 40.0, leg="search", tol=0.7, extra_obstacles=[disc])
        self.assertEqual("reached", result.status.value)
        self.assertTrue(any(v > 1e-6 for v, _w in client.cmds))

    def test_side_lidar_does_not_block_body_corridor(self):
        res = 0.1
        h, w = 30, 80
        grid = np.full((h, w), 100, dtype=np.int8)
        grid[10:21, :] = 0
        occ = _Occ(grid, res, 0.0, 0.0)
        client = FakeClient(
            [0.5, 1.55, 0.0],
            occ=occ,
            lidar_xyz=[[0.4, 2.0, 0.4]],
        )
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=5,
        )
        result = nav.navigate_to((7.0, 1.55), 8.0, leg="search", tol=0.7)
        self.assertEqual("reached", result.status.value)
        self.assertTrue(any(v > 1e-6 for v, _w in client.cmds))

    def test_start_hole_leaves_disc_ahead_blocked(self):
        from mock_agent_navigation import OccupancyGridPlanner

        res = 0.05
        grid = np.zeros((80, 160), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        start = (2.0, 2.0)
        ahead = (2.4, 2.0)
        planner.stamp_overlay([(ahead[0], ahead[1], 0.45)], now_s=1.0)
        planner.clear_start_hole(start, exclude_m=0.20)
        row, col = planner._w2c(ahead[0], ahead[1])
        self.assertTrue(bool(planner._overlay[row, col]))

    def test_side_lidar_does_not_stamp_overlay_on_the_road(self):
        from mock_agent_navigation import OccupancyGridPlanner, path_is_blocked
        from marc_agent.lidar_safety import overlay_discs

        grid = np.zeros((20, 40), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        start = (1.0, 5.0)
        goal = (16.0, 5.0)
        pose = (start[0], start[1], 0.0)
        side = overlay_discs(np.asarray([[0.4, 2.0, 0.4]], dtype=np.float32), pose)
        self.assertEqual([], side)
        planner.stamp_overlay(side, now_s=1.0)
        self.assertFalse(path_is_blocked(planner, [start, goal], inflation_m=0.0))
        corridor = overlay_discs(np.asarray([[2.0, 0.0, 0.4]], dtype=np.float32), pose)
        self.assertTrue(corridor)
        planner.stamp_overlay(corridor, now_s=1.0)
        self.assertTrue(path_is_blocked(planner, [start, goal], inflation_m=0.0))

    def test_live_lidar_disc_detours_on_free_occupancy(self):
        from mock_agent_navigation import OccupancyGridPlanner, path_is_blocked

        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        start = (1.0, 5.0)
        goal = (16.0, 5.0)
        car = (3.0, 5.0, 0.45)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        planner.stamp_overlay([car], now_s=1.0)
        planner.clear_start_hole(start, exclude_m=0.20)
        chord = [start, goal]
        self.assertTrue(
            path_is_blocked(planner, chord, inflation_m=0.0),
            "straight chord through a live lidar disc must be blocked",
        )
        self.assertFalse(
            path_is_blocked(planner, chord, inflation_m=0.0, ignore_overlay=True),
            "occupancy-only must still see free cells",
        )
        row, col = planner._w2c(car[0], car[1])
        self.assertTrue(bool(planner._overlay[row, col]))
        nav = Stage2Navigator(
            FakeClient([start[0], start[1], 0.0], occ=occ, lidar_world_xyz=[(3.0, 5.0, 0.4)]),
            dt=0.10,
            inflation_m=0.25,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=20,
        )
        planned = nav._plan_path(planner, start, goal)
        self.assertIsNotNone(planned)
        through_plan = [
            (wx, wy) for wx, wy in planned
            if abs(wy - 5.0) < 0.30 and abs(wx - 3.0) < 0.40
        ]
        self.assertEqual(
            [], through_plan, "_plan_path must not cut the chassis disc, got %s" % (planned,),
        )
        result = nav.navigate_to(goal, 40.0, leg="search", tol=0.7)
        self.assertEqual("reached", result.status.value)
        self.assertTrue(any(v > 1e-6 for v, _w in nav.client.cmds))
        path = result.path or ()
        through = [
            (wx, wy) for wx, wy in path
            if abs(wy - 5.0) < 0.30 and abs(wx - 3.0) < 0.40
        ]
        self.assertEqual([], through, "D* must detour around the lidar disc, got %s" % (path,))

    def test_plan_path_does_not_punch_goal_overlay(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.zeros((20, 40), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        start = (1.0, 5.0)
        goal = (16.0, 5.0)
        bumper = (15.6, 5.0, 0.4)
        planner.stamp_overlay([bumper], now_s=1.0)
        planner.clear_start_hole(start, exclude_m=0.20)
        nav = Stage2Navigator(FakeClient([start[0], start[1], 0.0]), inflation_m=0.25)
        nav._plan_path(planner, start, goal)
        row, col = planner._w2c(bumper[0], bumper[1])
        self.assertTrue(bool(planner._overlay[row, col]))
        planner.clear_goal_hole(goal, exclude_m=0.6)
        self.assertFalse(
            bool(planner._overlay[row, col]),
            "0.6 m goal hole would erase a bumper disc; live nav must not call it",
        )

    def test_remaining_path_uses_inflation_pad(self):
        from mock_agent_navigation import OccupancyGridPlanner, path_is_blocked

        res = 0.1
        grid = np.full((30, 80), 100, dtype=np.int8)
        grid[10:21, :] = 0
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, res, 0.0, 0.0)
        start = planner._c2w(10, 8)
        goal = planner._c2w(10, 70)
        self.assertTrue(
            path_is_blocked(planner, [start, goal], inflation_m=0.55),
            "0.55 m pad must reject a curb-hugging remaining chord",
        )
        self.assertFalse(
            path_is_blocked(
                planner, [start, goal], inflation_m=0.0, ignore_overlay=True,
            ),
            "raw occupancy 0 along the curb cell must stay open",
        )

    def test_no_path_frontal_reverses_instead_of_creep(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        grid[:, 20] = 100
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient([2.0, 5.0, 0.0], occ=occ, lidar_xyz=[[0.5, 0.0, 0.4]])
        nav = Stage2Navigator(
            client,
            dt=0.10,
            inflation_m=0.55,
            exclude_m=0.4,
            stuck_time_s=30.0,
            replan_cap=8,
        )
        nav.navigate_to((16.0, 5.0), 2.0, leg="search", tol=0.7)
        reverse = [(v, w) for v, w in client.cmds if v < -1e-6]
        self.assertTrue(reverse, "NO_PATH with a frontal disc must reverse, got %s" % client.cmds[:12])
        self.assertFalse(
            any(v > 1e-6 for v, _w in client.cmds),
            "occupancy-free NO_PATH must not creep forward, got %s" % client.cmds[:12],
        )
        self.assertTrue(all(abs(w) < 1e-9 for _v, w in reverse))

    def test_overlay_union_survives_empty_cloud_until_ttl(self):
        from mock_agent_navigation import OVERLAY_TTL_S, OccupancyGridPlanner

        grid = np.zeros((20, 40), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        disc = (5.0, 5.0, 0.45)
        planner.stamp_overlay([disc], now_s=1.0)
        row, col = planner._w2c(5.0, 5.0)
        planner.expire_overlay(2.0)
        planner.stamp_overlay([], now_s=2.0)
        self.assertTrue(
            bool(planner._overlay[row, col]),
            "union stamp must keep the disc when the next cloud is empty",
        )
        planner.expire_overlay(1.0 + OVERLAY_TTL_S + 0.1)
        self.assertFalse(bool(planner._overlay[row, col]))

    def test_goal_snap_skips_overlay_disc(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.zeros((20, 40), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        tumbler = (8.0, 5.0)
        planner.stamp_overlay([(tumbler[0], tumbler[1], 0.45)], now_s=1.0)
        self.assertFalse(planner.cell_free(tumbler[0], tumbler[1], 0.0))
        snapped = planner.nearest_free_xy(tumbler, inflation_m=0.55)
        self.assertIsNotNone(snapped)
        self.assertGreater(
            math.hypot(snapped[0] - tumbler[0], snapped[1] - tumbler[1]),
            0.40,
        )

    def test_live_loop_unions_overlay_not_replace(self):
        root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root, "marc_agent", "navigation_runtime.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertNotIn("replace_overlay", src)
        self.assertIn("stamp_overlay", src)
        self.assertIn("expire_overlay", src)
        self.assertIn("inflation_m=self.inflation_m", src)

    def test_frontal_escape_is_straight_reverse(self):
        from mock_agent_navigation import OccupancyGridPlanner

        grid = np.zeros((20, 40), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.5, 0.0, 0.0)
        pose = (2.0, 5.0, 0.0)
        self.assertIsNone(_frontal_escape(planner, pose, False))
        cmd = _frontal_escape(planner, pose, True)
        self.assertIsNotNone(cmd)
        self.assertLess(cmd[0], 0.0)
        self.assertEqual(0.0, cmd[1])

    def test_inside_disc_aborts_before_approach_edge(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        start = [5.0, 5.0, 0.0]
        client = FakeClient(start, occ=occ)
        nav = Stage2Navigator(
            client, dt=0.10, inflation_m=0.0, exclude_m=0.4,
            stuck_time_s=30.0, replan_cap=20,
        )
        result = nav.navigate_to(
            (16.0, 5.0), 5.0, leg="search", tol=0.7,
            search_xy=(5.0, 5.0), search_radius=2.0,
        )
        self.assertEqual("reached", result.status.value)
        self.assertEqual("inside_disc", result.reason)
        pose = client.get_world_pose()
        self.assertLess(math.hypot(pose[0] - 5.0, pose[1] - 5.0), 0.05)
        self.assertTrue(all(v <= 1e-9 for v, _w in client.cmds))

    def test_handoff_fn_true_on_tick_two(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient([2.0, 5.0, 0.0], occ=occ)
        nav = Stage2Navigator(
            client, dt=0.10, inflation_m=0.0, exclude_m=0.4,
            stuck_time_s=30.0, replan_cap=20,
        )
        ticks = [0]

        def handoff(_pose):
            ticks[0] += 1
            return ticks[0] >= 2

        result = nav.navigate_to(
            (16.0, 5.0), 5.0, leg="search", tol=0.7, handoff_fn=handoff,
        )
        self.assertEqual("reached", result.status.value)
        self.assertEqual("yolo_seen", result.reason)
        self.assertGreaterEqual(ticks[0], 2)

    def test_ram_unstick_stamps_overlay_reverses_and_detours(self):
        from marc_agent.lidar_safety import DISC_R, STOP_X_MAX
        from mock_agent_navigation import OccupancyGridPlanner

        start = (2.0, 5.0, 0.0)
        disc = _ram_bumper_disc(start)
        self.assertAlmostEqual(disc[0], start[0] + STOP_X_MAX)
        self.assertAlmostEqual(disc[1], start[1])
        self.assertAlmostEqual(disc[2], DISC_R)
        # Live occupancy is 5 cm; a 0.5 m test grid would let the start hole eat the bumper disc.
        grid = np.zeros((200, 400), dtype=np.int8)
        planner = OccupancyGridPlanner()
        planner.update_from_array(grid, 0.05, 0.0, 0.0)
        planner.stamp_overlay([disc], now_s=1.0)
        planner.clear_start_hole((start[0], start[1]), exclude_m=0.20)
        self.assertFalse(planner.cell_free(disc[0], disc[1], 0.0))

        occ = _Occ(grid, 0.05, 0.0, 0.0)
        client = FakeClient([2.0, 5.0, 0.0], occ=occ, immobile=True)
        nav = Stage2Navigator(
            client, dt=0.10, inflation_m=0.0, exclude_m=0.4,
            stuck_time_s=30.0, replan_cap=20,
        )
        result = nav.navigate_to((16.0, 5.0), 8.0, leg="search", tol=0.7)
        self.assertTrue(client.cmds)
        fwd_i = next(
            (i for i, (v, _w) in enumerate(client.cmds) if v > 0.05), None,
        )
        self.assertIsNotNone(fwd_i, "expected a forward command before ram")
        rev_i = next(
            (
                i for i, (v, w) in enumerate(client.cmds)
                if i > fwd_i and v < -0.05 and abs(w) < 0.15
            ),
            None,
        )
        self.assertIsNotNone(rev_i, "expected straight reverse after forward stuck")
        self.assertGreaterEqual(result.replan_count, 1)
        self.assertIsNotNone(result.path)
        self.assertGreaterEqual(len(result.path), 2)

    def test_inplace_turn_does_not_ram_unstick(self):
        grid = np.zeros((20, 40), dtype=np.int8)
        occ = _Occ(grid, 0.5, 0.0, 0.0)
        client = FakeClient([2.0, 5.0, math.pi], occ=occ, immobile=True)
        nav = Stage2Navigator(
            client, dt=0.10, inflation_m=0.0, exclude_m=0.4,
            stuck_time_s=30.0, replan_cap=20,
        )
        nav.navigate_to((16.0, 5.0), 3.0, leg="search", tol=0.7)
        self.assertTrue(client.cmds)
        self.assertTrue(
            all(v > -0.05 for v, _w in client.cmds),
            "in-place rotate must not reverse, got %s" % client.cmds[:12],
        )
        self.assertTrue(
            any(abs(w) > 0.05 and abs(v) < 0.05 for v, w in client.cmds),
            "expected in-place yaw commands",
        )


if __name__ == "__main__":
    unittest.main()
