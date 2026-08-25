"""Lidar PointCloud2 / XYZ overlay fixtures (no live ROS)."""

from __future__ import annotations

import math
import os
import unittest

import numpy as np
import yaml

from marc_agent.lidar_overlay import (
    LIDAR_CLUSTER_M,
    discs_to_cells,
    frontal_hit,
    lidar_xyz_to_obstacles,
    pointcloud2_xyz,
)


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "lidar_overlay_cases.yaml")


def _load():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)["cases"]


class _Field:
    def __init__(self, name, offset):
        self.name = name
        self.offset = offset
        self.datatype = 7
        self.count = 1


class _Cloud:
    def __init__(self, xyz):
        pts = np.asarray(xyz, dtype=np.float32)
        self.fields = [_Field("x", 0), _Field("y", 4), _Field("z", 8)]
        self.height = 1
        self.width = int(pts.shape[0])
        self.point_step = 16
        self.is_bigendian = False
        raw = np.zeros(self.width * self.point_step, dtype=np.uint8)
        for i, pt in enumerate(pts):
            raw[i * 16:i * 16 + 12] = np.frombuffer(pt.tobytes(), dtype=np.uint8)
        self.data = raw


class LidarOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load()

    def test_fixture_size(self):
        self.assertGreaterEqual(len(self.cases), 10)
        self.assertEqual(len(self.cases), len({case["id"] for case in self.cases}))

    def test_xyz_overlay_cases(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                xyz = np.asarray(case["xyz"], dtype=np.float32)
                if xyz.size == 0:
                    xyz = xyz.reshape(0, 3)
                discs = lidar_xyz_to_obstacles(xyz, case["pose"])
                self.assertEqual(case["expected_n"], len(discs))
                if case["expected_n"] == 1 and "expected_world" in case:
                    wx, wy, _ = discs[0]
                    exp = case["expected_world"]
                    self.assertLess(
                        math.hypot(wx - exp[0], wy - exp[1]),
                        LIDAR_CLUSTER_M + 1e-6,
                        msg="%s disc (%s,%s) expected ~%s" % (case["id"], wx, wy, exp),
                    )
                if "expected_cells" in case:
                    cells = discs_to_cells(
                        discs, case["res"], case["origin"][0], case["origin"][1])
                    self.assertEqual(case["expected_cells"], cells)

    def test_pointcloud2_roundtrip(self):
        xyz = np.array([[1.0, 0.25, 0.4], [2.0, -0.1, 0.5]], dtype=np.float32)
        got = pointcloud2_xyz(_Cloud(xyz))
        self.assertIsNotNone(got)
        self.assertEqual((2, 3), got.shape)
        np.testing.assert_allclose(got, xyz, atol=1e-5)

    def test_frontal_hit_box(self):
        pose = (0.0, 0.0, 0.0)
        self.assertTrue(frontal_hit([(0.8, 0.0, 0.2)], pose))
        self.assertFalse(frontal_hit([(-2.0, 0.0, 0.2)], pose))
        self.assertFalse(frontal_hit([(0.8, 2.0, 0.1)], pose))


if __name__ == "__main__":
    unittest.main()
