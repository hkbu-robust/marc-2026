"""Synthetic pose/map/lidar drive loop for Stage2Navigator (no live ROS)."""

from __future__ import annotations

import math
import os
import unittest

import numpy as np
import yaml

from marc_agent.navigation_runtime import NavStatus, Stage2Navigator
from marc_agent.retrieval_state import search_area_approach_goal


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "nav_runtime_cases.yaml")


def _load():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


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

    def __init__(self, start, occ=None, stop_after_s=None):
        self._pose = [float(start[0]), float(start[1]), float(start[2])]
        self._t = 0.0
        self._v = 0.0
        self._w = 0.0
        self.is_running = True
        self.cmds = []
        self._occ = occ
        self.stop_after_s = stop_after_s

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
        return None

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


if __name__ == "__main__":
    unittest.main()
