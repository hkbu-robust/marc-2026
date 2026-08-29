"""Robot-frame lidar corridor (no occupancy overlay)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from marc_agent.lidar_safety import (
    CLEAR_FAR,
    STOP_X_MAX,
    advise,
    pointcloud2_xyz,
)


def _xyz(*rows):
    return np.asarray(rows, dtype=np.float32)


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


class LidarSafetyTests(unittest.TestCase):
    def test_empty_cruises(self):
        empty = advise(None)
        self.assertFalse(empty.stop)
        self.assertEqual(empty.speed_scale, 1.0)
        self.assertTrue(math.isinf(empty.clearance_m))
        self.assertEqual(advise(_xyz()).speed_scale, 1.0)

    def test_ground_ignored(self):
        got = advise(_xyz([1.0, 0.0, -0.5]))
        self.assertFalse(got.stop)
        self.assertEqual(got.speed_scale, 1.0)

    def test_spawn_ring_ignored(self):
        ring = []
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            ring.append([0.30 * math.cos(rad), 0.30 * math.sin(rad), 0.4])
        got = advise(_xyz(*ring))
        self.assertFalse(got.stop)
        self.assertEqual(got.speed_scale, 1.0)

    def test_close_body_hit_stops(self):
        got = advise(_xyz([0.5, 0.0, 0.4]))
        self.assertTrue(got.stop)
        self.assertEqual(got.speed_scale, 0.0)
        self.assertLessEqual(got.clearance_m, STOP_X_MAX)

    def test_side_car_does_not_stop(self):
        got = advise(_xyz([1.0, 1.2, 0.4]))
        self.assertFalse(got.stop)
        self.assertEqual(got.speed_scale, 1.0)
        self.assertTrue(math.isinf(got.clearance_m))

    def test_offset_bumper_in_corridor_stops(self):
        got = advise(_xyz([0.6, 0.4, 0.4]))
        self.assertTrue(got.stop)
        self.assertEqual(got.speed_scale, 0.0)

    def test_far_corridor_slows(self):
        got = advise(_xyz([2.0, 0.0, 0.4]))
        self.assertFalse(got.stop)
        self.assertGreater(got.speed_scale, 0.0)
        self.assertLess(got.speed_scale, 1.0)
        self.assertAlmostEqual(got.clearance_m, 2.0, places=3)

    def test_beyond_look_ahead_cruises(self):
        got = advise(_xyz([CLEAR_FAR + 0.5, 0.0, 0.4]))
        self.assertFalse(got.stop)
        self.assertEqual(got.speed_scale, 1.0)

    def test_canopy_ignored(self):
        got = advise(_xyz([0.5, 0.0, 2.5]))
        self.assertFalse(got.stop)
        self.assertEqual(got.speed_scale, 1.0)

    def test_nudge_away_from_nearer_side(self):
        left = advise(_xyz([1.0, 0.7, 0.4]))
        self.assertFalse(left.stop)
        self.assertLess(left.yaw_bias, 0.0)
        right = advise(_xyz([1.0, -0.7, 0.4]))
        self.assertGreater(right.yaw_bias, 0.0)

    def test_pointcloud2_roundtrip(self):
        xyz = np.array([[1.0, 0.25, 0.4], [2.0, -0.1, 0.5]], dtype=np.float32)
        got = pointcloud2_xyz(_Cloud(xyz))
        self.assertIsNotNone(got)
        self.assertEqual((2, 3), got.shape)
        np.testing.assert_allclose(got, xyz, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
