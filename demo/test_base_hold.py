"""Tests for the stationary-base guard used during manipulation."""

from __future__ import annotations

import time
import unittest

from manip_dataset.base_hold import StationaryBase


class _Client:
    def __init__(self):
        self.commands = []

    def send_cmd_vel(self, linear_x=0.0, angular_z=0.0):
        self.commands.append((linear_x, angular_z))


class StationaryBaseTests(unittest.TestCase):
    def test_republishes_zero_for_entire_guarded_pick(self):
        client = _Client()
        hold = StationaryBase(client, interval_s=0.01)

        with hold:
            time.sleep(0.035)

        self.assertFalse(hold.active)
        self.assertGreaterEqual(len(client.commands), 4)
        self.assertTrue(all(command == (0.0, 0.0) for command in client.commands))

    def test_corrects_backward_slope_drift_toward_anchor(self):
        client = _Client()
        hold = StationaryBase(
            client,
            position_kp=2.0,
            max_linear_mps=0.08,
            position_deadband_m=0.002,
        )
        hold._anchor = (1.0, 2.0, 0.0)

        linear, angular = hold._correction((0.98, 2.0, 0.0))

        self.assertAlmostEqual(0.04, linear)
        self.assertEqual(0.0, angular)

    def test_pose_correction_is_bounded_and_yaw_wraps(self):
        client = _Client()
        hold = StationaryBase(
            client,
            position_kp=10.0,
            max_linear_mps=0.08,
            position_deadband_m=0.0,
            yaw_kp=10.0,
            max_angular_rps=0.15,
            yaw_deadband_rad=0.0,
        )
        hold._anchor = (0.0, 0.0, -3.13)

        linear, angular = hold._correction((-1.0, 0.0, 3.13))

        self.assertAlmostEqual(-0.08, linear)
        self.assertAlmostEqual(0.15, angular)


if __name__ == "__main__":
    unittest.main()
