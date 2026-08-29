"""Wrist-YOLO pick: image errors, EE-delta signs, bbox-depth yaw, fail-closed center."""

from __future__ import annotations

import math
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from marc_agent.pick_runtime import PickRuntime, pick_policy_name  # noqa: E402
from marc_agent.wrist_pick import (  # noqa: E402
    DEFAULT_STEP_M,
    center_over_bbox,
    delta_from_errors,
    image_errors,
    yaw_from_bbox_depth,
    yaw_from_points_xy,
)


def _det(*, error, cy, n_obj=1, h=48, w=64, cx=32.0):
    return {
        "n_obj": n_obj,
        "error": error,
        "cx": cx,
        "cy": cy,
        "rgb_shape": (h, w),
        "bbox": (10.0, 10.0, 40.0, 30.0),
    }


class _Arm:
    def __init__(self):
        self.deltas = []

    def apply_delta(self, dx=0.0, dy=0.0, dz=0.0, **_k):
        self.deltas.append((float(dx), float(dy), float(dz)))


class _Detector:
    def __init__(self, dets):
        self._dets = list(dets)
        self.i = 0
        self.targets = []

    def set_target(self, name):
        self.targets.append(name)

    def detect(self, _rgb):
        d = self._dets[min(self.i, len(self._dets) - 1)]
        self.i += 1
        return d


class WristPickTests(unittest.TestCase):
    def test_image_errors_from_yolo_error_and_cy(self):
        self.assertIsNone(image_errors(None))
        self.assertIsNone(image_errors({"n_obj": 0, "error": 0.4}))
        err_x, err_y = image_errors(_det(error=0.5, cy=47, h=48))
        self.assertAlmostEqual(0.5, err_x)
        self.assertAlmostEqual(1.0, err_y, places=2)
        err_x, err_y = image_errors(_det(error=-0.25, cy=0, h=48))
        self.assertAlmostEqual(-0.25, err_x)
        self.assertAlmostEqual(-1.0, err_y, places=2)

    def test_delta_from_errors_signs_and_clamp(self):
        env = {"MARC_WRIST_SIGN_X": "", "MARC_WRIST_SIGN_Y": ""}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("MARC_WRIST_SIGN_X", None)
            os.environ.pop("MARC_WRIST_SIGN_Y", None)
            dx, dy = delta_from_errors(0.5, 0.5, step_m=0.02)
            self.assertGreater(dx, 0.0)
            self.assertLess(dy, 0.0)
            self.assertAlmostEqual(0.01, dx, places=5)
            self.assertAlmostEqual(-0.01, dy, places=5)
            dx, dy = delta_from_errors(1.0, -1.0, step_m=0.012)
            self.assertAlmostEqual(-DEFAULT_STEP_M, dx, places=5)
            self.assertAlmostEqual(-DEFAULT_STEP_M, dy, places=5)
            dx, dy = delta_from_errors(4.0, 4.0, step_m=0.01)
            self.assertAlmostEqual(0.01, dx, places=5)
            self.assertAlmostEqual(-0.01, dy, places=5)

    def test_yaw_from_bbox_depth_long_axis_y(self):
        pts = np.column_stack([
            np.full(16, 0.30),
            np.linspace(-0.08, 0.08, 16),
            np.full(16, -0.20),
        ])
        yaw = yaw_from_points_xy(pts)
        self.assertIsNotNone(yaw)
        self.assertAlmostEqual(abs(yaw), 0.5 * math.pi, places=1)

        depth = np.full((32, 32), 0.40, dtype=np.float32)
        k = np.array([[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]])
        bbox = (14.0, 2.0, 18.0, 30.0)
        yaw = yaw_from_bbox_depth(depth, bbox, (32, 32), k, np.eye(4))
        self.assertIsNotNone(yaw)
        self.assertAlmostEqual(abs(yaw), 0.5 * math.pi, delta=0.2)
        self.assertIsNone(
            yaw_from_bbox_depth(depth, bbox, (32, 32), k, None)
        )

    def test_center_stops_when_errors_inside_tol(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        client = SimpleNamespace(get_robot_image=lambda _c: rgb)
        centered = _det(error=0.02, cy=24.0, h=48, w=64)
        arm = _Arm()
        ok = center_over_bbox(
            client, _Detector([centered]), arm, "tumbler", tol=0.08, max_steps=5,
        )
        self.assertTrue(ok)
        self.assertEqual([], arm.deltas)

        off = _det(error=0.40, cy=40.0, h=48, w=64)
        arm = _Arm()
        ok = center_over_bbox(
            client, _Detector([off, centered]), arm, "tumbler",
            tol=0.08, max_steps=5, step_m=0.01,
        )
        self.assertTrue(ok)
        self.assertEqual(1, len(arm.deltas))
        self.assertGreater(arm.deltas[0][0], 0.0)
        self.assertLess(arm.deltas[0][1], 0.0)

    def test_center_fail_closed_on_empty_detect(self):
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        client = SimpleNamespace(get_robot_image=lambda _c: rgb)
        arm = _Arm()
        ok = center_over_bbox(
            client,
            _Detector([{"n_obj": 0, "error": None, "rgb_shape": (48, 64)}]),
            arm,
            "tumbler",
        )
        self.assertFalse(ok)
        self.assertEqual([], arm.deltas)

    def test_pick_policy_name_allows_wrist(self):
        with patch.dict(os.environ, {"MARC_PICK_POLICY": "wrist"}):
            self.assertEqual("wrist", pick_policy_name())
        with patch.dict(os.environ, {"MARC_PICK_POLICY": "graspgenx"}):
            self.assertEqual("graspgenx", pick_policy_name())

    def test_wrist_policy_skips_graspgenx(self):
        ggx = []
        ik = []
        client = SimpleNamespace(
            is_running=True,
            is_grasping=lambda: True,
            is_basket_occupied=lambda: True,
            now_s=lambda: 0.0,
            sleep=lambda _dt: None,
        )
        runtime = PickRuntime(
            client, log=lambda _m: None,
            max_attempts=1, policy="wrist",
            place_fn=lambda: None,
            recover_fn=lambda: None,
            poll_dt=0.0, grasp_poll=1, basket_poll=1,
        )
        runtime._apply_graspgenx = lambda t: ggx.append(t) or True
        runtime._ik_grasp = lambda t: ik.append(t)
        target = SimpleNamespace(fwd=0.30, lat=0.0, z=-0.22, valid=True, name="tumbler")
        result = runtime.run(target)
        self.assertTrue(result.ok)
        self.assertEqual("wrist", result.policy)
        self.assertEqual([], ggx)
        self.assertEqual(1, len(ik))


if __name__ == "__main__":
    unittest.main()
