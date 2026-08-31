"""Focused tests for manual arm/gripper control."""

from __future__ import annotations

import unittest
import threading
from types import SimpleNamespace

import numpy as np

from arm_pick import ARM_JOINT_NAMES
from manip_dataset.arm_control import ArmTeleop, _cosine_ease


class _ContactClient:
    def __init__(self, contact_sample: int):
        self.contact_sample = contact_sample
        self.grasp_samples = 0
        self.arm_state_samples = 0

    def get_arm_state(self):
        self.arm_state_samples += 1
        noisy_q = np.full(6, 0.001 * self.arm_state_samples)
        return SimpleNamespace(name=ARM_JOINT_NAMES, position=noisy_q)

    def is_grasping(self):
        self.grasp_samples += 1
        return self.grasp_samples >= self.contact_sample


class _FollowingClient:
    def __init__(self):
        self.q = np.zeros(6)

    def get_arm_state(self):
        return SimpleNamespace(name=ARM_JOINT_NAMES, position=self.q.copy())


class _TransientContactClient(_ContactClient):
    def __init__(self):
        super().__init__(contact_sample=0)
        self.holding = iter((False, True, False, False, True, True, True))

    def is_grasping(self):
        return next(self.holding, True)


class _LinearKin:
    @staticmethod
    def fk(q):
        result = np.eye(4)
        result[0, 3] = q[0]
        return result


class GripperCloseTests(unittest.TestCase):
    def test_close_records_one_trigger_while_physical_ramp_keeps_running(self):
        client = _ContactClient(contact_sample=3)
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=10.0,
            grip_close=0.68,
            grip_ramp_s=1.0,
            grip_dwell_s=0.2,
            grip_stop_on_grasp=True,
            grip_extra=0.02,
            grip_confirm_samples=3,
        )
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm._closed_since_last_open = False
        arm._saw_grasp_after_close = False
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        published = []
        arm._publish = lambda _q: published.append(arm.grip_cmd)
        arm.snapshot_obs = lambda: {"frame": "pre-close"}
        recorded = []

        count = arm._grip_click(
            arm.cfg.grip_close,
            lambda obs, action: recorded.append((obs, action.copy())),
        )

        self.assertEqual(1, count)
        self.assertEqual(1, len(recorded))
        self.assertEqual("pre-close", recorded[0][0]["frame"])
        np.testing.assert_array_equal(
            recorded[0][1],
            np.array([0.0] * 6 + [1.0], dtype=np.float32),
        )
        self.assertGreater(len(published), 1)
        self.assertTrue(arm.saw_grasp_after_close)

    def test_close_commands_extra_bite_after_contact_before_holding(self):
        client = _ContactClient(contact_sample=3)
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=10.0,
            grip_close=0.68,
            grip_ramp_s=1.0,
            grip_dwell_s=0.2,
            grip_stop_on_grasp=True,
            grip_extra=0.02,
            grip_confirm_samples=3,
        )
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm._closed_since_last_open = False
        arm._saw_grasp_after_close = False
        logs = []
        published = []
        published_q = []
        arm._log = logs.append
        arm._sleep = lambda _dt: None
        arm._publish = lambda q: (
            published.append(arm.grip_cmd),
            published_q.append(np.asarray(q).copy()),
        )

        arm._grip_click(arm.cfg.grip_close, None)

        contact_command = published[2]
        self.assertAlmostEqual(0.204, contact_command)
        bite_index = next(
            index for index, value in enumerate(published) if value > contact_command
        )
        self.assertGreaterEqual(bite_index, 5)
        self.assertAlmostEqual(contact_command + 0.02, published[bite_index])
        self.assertTrue(
            all(value == published[bite_index] for value in published[bite_index:])
        )
        self.assertAlmostEqual(0.224, arm.grip_cmd)
        self.assertTrue(arm.saw_grasp_after_close)
        self.assertTrue(all(np.array_equal(q, published_q[0]) for q in published_q))
        self.assertTrue(any("hold confirmed (3 samples)" in line for line in logs))
        self.assertTrue(any("continuing close to 0.224" in line for line in logs))
        self.assertTrue(any("post-contact bite complete" in line for line in logs))

    def test_full_close_mode_ignores_hold_sensor_and_reaches_target(self):
        client = _ContactClient(contact_sample=1)
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=10.0,
            grip_close=0.68,
            grip_ramp_s=1.0,
            grip_dwell_s=0.2,
            grip_stop_on_grasp=False,
            grip_extra=0.04,
            grip_confirm_samples=3,
        )
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm._closed_since_last_open = False
        arm._saw_grasp_after_close = False
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        published = []
        arm._publish = lambda _q: published.append(arm.grip_cmd)

        arm._grip_click(arm.cfg.grip_close, None)

        self.assertAlmostEqual(0.68, arm.grip_cmd)
        self.assertAlmostEqual(0.68, arm.grip_target)
        self.assertAlmostEqual(0.68, published[-1])
        self.assertTrue(any(value < 0.68 for value in published))
        self.assertEqual(0, client.grasp_samples)

    def test_zero_ramp_publishes_full_close_on_first_tick(self):
        client = _ContactClient(contact_sample=1)
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=30.0,
            grip_close=0.68,
            grip_ramp_s=0.0,
            grip_dwell_s=0.2,
            grip_stop_on_grasp=False,
            grip_extra=0.04,
            grip_confirm_samples=3,
        )
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm._closed_since_last_open = False
        arm._saw_grasp_after_close = False
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        published = []
        arm._publish = lambda _q: published.append(arm.grip_cmd)

        arm._grip_click(arm.cfg.grip_close, None)

        self.assertGreater(len(published), 1)
        self.assertTrue(all(value == 0.68 for value in published))

    def test_place_trajectory_eases_and_advances_on_feedback(self):
        client = _FollowingClient()
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=10.0,
            place_rec_every=3,
            settle_tol_joint=0.01,
        )
        arm.kin = _LinearKin()
        arm.grip_cmd = 0.3
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        published = []

        def publish(q):
            client.q = np.asarray(q).copy()
            published.append(client.q.copy())

        arm._publish = publish
        result = arm._joint_move(
            "test", np.array([1.0, 0, 0, 0, 0, 0]), None, 1.0
        )

        self.assertEqual(0, result)
        positions = np.array([q[0] for q in published[:16]])
        increments = np.diff(np.concatenate(([0.0], positions)))
        self.assertAlmostEqual(1.0, positions[-1])
        self.assertLess(increments[0], increments[len(increments) // 2])
        self.assertLess(increments[-1], increments[len(increments) // 2])
        self.assertLessEqual(float(np.max(increments)), 0.1 + 1e-9)

    def test_place_trajectory_stops_at_current_tick_when_cancelled(self):
        client = _FollowingClient()
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=10.0,
            place_rec_every=3,
            settle_tol_joint=0.01,
        )
        arm.kin = _LinearKin()
        arm.grip_cmd = 0.3
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        cancel = threading.Event()
        published = []

        def publish(q):
            client.q = np.asarray(q).copy()
            published.append(client.q.copy())
            if len(published) == 3:
                cancel.set()

        arm._publish = publish
        arm._joint_move(
            "test",
            np.array([1.0, 0, 0, 0, 0, 0]),
            None,
            1.0,
            cancel_event=cancel,
        )

        self.assertEqual(3, len(published))
        self.assertLess(float(published[-1][0]), 1.0)

    def test_place_trajectory_fails_closed_when_feedback_does_not_follow(self):
        client = _FollowingClient()

        def drifting_state():
            client.q += 1e-5
            return SimpleNamespace(name=ARM_JOINT_NAMES, position=client.q.copy())

        client.get_arm_state = drifting_state
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=10.0,
            place_rec_every=3,
            settle_tol_joint=0.01,
        )
        arm.kin = _LinearKin()
        arm.grip_cmd = 0.3
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        arm._publish = lambda _q: None

        result = arm._joint_move(
            "stale", np.array([1.0, 0, 0, 0, 0, 0]), None, 1.0
        )

        self.assertIsNone(result)

    def test_transient_hold_does_not_stop_close(self):
        client = _TransientContactClient()
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=10.0,
            grip_close=0.68,
            grip_ramp_s=1.0,
            grip_dwell_s=0.2,
            grip_stop_on_grasp=True,
            grip_extra=0.02,
            grip_confirm_samples=3,
        )
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm._closed_since_last_open = False
        arm._saw_grasp_after_close = False
        arm._sleep = lambda _dt: None
        arm._publish = lambda _q: None
        logs = []
        arm._log = logs.append

        arm._grip_click(arm.cfg.grip_close, None)

        self.assertTrue(any("ignored transient hold signal" in line for line in logs))
        self.assertTrue(any("hold confirmed (3 samples)" in line for line in logs))
        self.assertGreater(arm.grip_cmd, 0.20)

    def test_cosine_ease_has_stationary_endpoints(self):
        self.assertEqual(0.0, _cosine_ease(0.0))
        self.assertEqual(1.0, _cosine_ease(1.0))
        self.assertAlmostEqual(0.5, _cosine_ease(0.5))


if __name__ == "__main__":
    unittest.main()
