"""Focused tests for manual arm/gripper control."""

from __future__ import annotations

import unittest
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from arm_pick import ARM_JOINT_NAMES
from manip_dataset import arm_control
from manip_dataset.arm_control import ArmTeleop, _cosine_ease
from manip_dataset.grasp_profiles import TUMBLER_UPRIGHT


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


class _CommandFollowingClient(_FollowingClient):
    def __init__(self):
        super().__init__()
        self.commands = []
        self.elapsed = 0.0

    def send_arm_command(self, msg):
        self.q = np.asarray(msg.position[:6], dtype=float)
        self.commands.append(self.q.copy())

    def sleep(self, dt):
        self.elapsed += float(dt)


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


class _XYZKin:
    @staticmethod
    def fk(q):
        result = np.eye(4)
        result[:3, 3] = np.asarray(q[:3], dtype=float)
        return result


class _YawKin:
    @staticmethod
    def fk(q):
        yaw = float(q[1])
        c, s = np.cos(yaw), np.sin(yaw)
        result = np.eye(4)
        result[:3, :3] = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        )
        return result


class GripperCloseTests(unittest.TestCase):
    def test_immediate_grip_temporarily_disables_ramp_and_sensor_stop(self):
        arm = object.__new__(ArmTeleop)
        arm.cfg = SimpleNamespace(grip_ramp_s=1.2, grip_stop_on_grasp=True)
        calls = []

        def set_grip(value):
            calls.append(
                (float(value), arm.cfg.grip_ramp_s, arm.cfg.grip_stop_on_grasp)
            )
            return True

        arm.set_grip = set_grip

        self.assertTrue(arm.set_grip_immediate(0.703))
        self.assertEqual([(0.703, 0.0, False)], calls)
        self.assertEqual(1.2, arm.cfg.grip_ramp_s)
        self.assertTrue(arm.cfg.grip_stop_on_grasp)

    def test_world_closing_yaw_accounts_for_default_wrist_axis(self):
        # The real default EE frame has local +X (jaw closing) along world -Y.
        base_R = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        desired_world_yaw = 0.0

        relative = arm_control.closing_axis_relative_yaw(
            base_R, desired_world_yaw
        )
        c, s = np.cos(relative), np.sin(relative)
        target_R = base_R @ np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        )

        self.assertAlmostEqual(np.pi / 2.0, relative)
        np.testing.assert_allclose(target_R[:2, 0], (1.0, 0.0), atol=1e-9)

    def test_upright_tilt_rotates_about_aligned_jaw_axis(self):
        base_R = np.array(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        yaw = -0.5 * np.pi
        tilt = 0.80

        target_quat, relative_yaw = arm_control.grasp_target_quat(
            base_R, yaw, tilt
        )
        c, s = np.cos(tilt), np.sin(tilt)
        expected_R = base_R @ np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
        )

        self.assertAlmostEqual(0.0, relative_yaw, places=6)
        self.assertLess(expected_R[0, 2], 0.0)
        self.assertGreater(expected_R[2, 2], 0.0)
        np.testing.assert_allclose(expected_R[:, 0], (0.0, -1.0, 0.0))
        self.assertLess(
            np.linalg.norm(
                arm_control.arm_kin.quat_error_axis_angle(
                    target_quat,
                    arm_control.arm_kin.R_to_quat(expected_R),
                )
            ),
            1e-9,
        )

    def test_pad_center_goal_compensates_local_gripper_offset(self):
        target_R = np.array(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

        goal = arm_control.pad_center_to_ee_goal(
            (0.55, -0.01, -0.35), target_R, 0.0205
        )

        np.testing.assert_allclose(goal, (0.5705, -0.01, -0.35), atol=1e-9)

    def test_upright_handoff_clears_folds_and_descends_before_returning(self):
        client = _FollowingClient()
        client.q[:3] = (0.329, 0.001, 0.094)
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.kin = _XYZKin()
        arm.cfg = SimpleNamespace()
        arm._log = lambda line: events.append(("log", line))
        arm.set_grip = lambda value: events.append(("grip", value)) or True
        arm.reset_grasp_flags = lambda: events.append(("reset",))
        def move_ee_to(xyz, label):
            client.q[:3] = xyz
            events.append((label, tuple(xyz), {}))
            return True

        arm.move_ee_to = move_ee_to
        def move_grasp_line(xyz, yaw, label, **kwargs):
            client.q[:3] = xyz
            events.append((label, tuple(xyz), yaw, kwargs))
            return True

        arm.move_grasp_line = move_grasp_line
        arm.move_grasp_pose = lambda xyz, yaw, label, **kwargs: (
            events.append((label, tuple(xyz), yaw, kwargs)) or True
        )
        def move_grasp_pose_joint_path(xyz, yaw, label, **kwargs):
            events.append((label, tuple(xyz), yaw, kwargs))
            folded_q = client.q.copy()
            folded_q[:3] = (0.481, 0.044, -0.2795)
            return folded_q

        arm.move_grasp_pose_joint_path = move_grasp_pose_joint_path
        events = []
        target = SimpleNamespace(
            fwd=0.581,
            lat=0.044,
            z=-0.359,
            fold_z=-0.090,
            grasp_yaw=-0.5 * np.pi,
            valid=True,
        )

        ok = arm.stage_for_graspgenx(target, profile=TUMBLER_UPRIGHT)

        self.assertTrue(ok)
        moves = [event for event in events if event[0].startswith("ggx-")]
        self.assertEqual(
            [
                "ggx-upright-clear-up",
                "ggx-upright-fold-clear",
                "ggx-upright-rear-align-folded",
                "ggx-upright-descend-to-fold",
                "ggx-upright-coordinated-fold",
            ],
            [event[0] for event in moves],
        )
        np.testing.assert_allclose(moves[0][1], (0.329, 0.001, 0.234))
        np.testing.assert_allclose(moves[1][1], (0.329, 0.001, 0.2545))
        np.testing.assert_allclose(moves[2][1], (0.481, 0.044, 0.2545))
        np.testing.assert_allclose(moves[3][1], (0.481, 0.044, -0.259))
        np.testing.assert_allclose(moves[4][1], (0.481, 0.044, -0.259))
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_tilt_rad,
            moves[1][3]["approach_tilt_rad"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_tilt_rad,
            moves[2][3]["approach_tilt_rad"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_tilt_rad,
            moves[3][3]["approach_tilt_rad"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_fold_speed_rad_s,
            moves[4][3]["speed_rad_s"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_joint_step_rad,
            moves[2][3]["joint_step_rad"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_joint_step_rad,
            moves[3][3]["joint_step_rad"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_cart_step_m,
            moves[2][3]["cart_step_m"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_cart_step_m,
            moves[3][3]["cart_step_m"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_rate_hz,
            moves[2][3]["control_rate_hz"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_rate_hz,
            moves[3][3]["control_rate_hz"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_cart_step_m,
            moves[0][3]["cart_step_m"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_joint_step_rad,
            moves[0][3]["joint_step_rad"],
        )
        self.assertEqual(
            TUMBLER_UPRIGHT.side_transit_rate_hz,
            moves[0][3]["control_rate_hz"],
        )
        self.assertTrue(moves[0][3]["preserve_current_orientation"])
        self.assertAlmostEqual(0.481, target.folded_pad_fwd)
        self.assertAlmostEqual(0.044, target.folded_pad_lat)
        self.assertAlmostEqual(-0.259, target.folded_pad_z)

    def test_joint_delta_limit_preserves_dls_direction(self):
        delta = np.array((0.10, -0.05, 0.025))

        limited = arm_control.limit_joint_delta(delta, 0.02)

        self.assertAlmostEqual(0.02, float(np.max(np.abs(limited))))
        np.testing.assert_allclose(limited / limited[0], delta / delta[0])

    def test_coordinated_fold_uses_one_eased_twenty_hz_joint_path(self):
        client = _CommandFollowingClient()
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.kin = _XYZKin()
        arm.cfg = SimpleNamespace(rate_hz=30.0, settle_tol_joint=0.01)
        arm.grip_cmd = 0.0
        arm._log = lambda _line: None

        q_goal = np.array((0.10, 0.0, 0.0, 0.0, -0.20, 0.0))
        with patch.object(arm_control.arm_kin, "dls_solve", return_value=q_goal):
            result = arm.move_grasp_pose_joint_path(
                (0.10, 0.0, 0.0205),
                0.0,
                "test-fold",
                approach_tilt_rad=0.0,
                pad_center_offset_m=0.0205,
                speed_rad_s=0.80,
                control_rate_hz=20.0,
            )

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[:4], (0.10, 0.0, 0.0, 0.0))
        self.assertAlmostEqual(-0.20, result[4])
        self.assertAlmostEqual(0.0, result[5])
        self.assertEqual(9, len(client.commands))
        self.assertAlmostEqual(8.0 / 20.0, client.elapsed)

    def test_cartesian_servo_preserves_dls_direction_near_joint_limit(self):
        client = _FollowingClient()
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=20.0,
            max_ticks_per_click=1,
            pos_step=0.008,
            settle_tol_m=0.015,
            settle_tol_rot_rad=0.05,
            ik_joint_step=0.02,
            grip_close=0.68,
        )
        arm.kin = _XYZKin()
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm.target_pos = None
        arm.target_quat = None
        arm._closed_since_last_open = False
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        published = []
        arm._publish = lambda q: published.append(np.asarray(q).copy())

        def solve(_kin, q, _target_pos, _target_quat):
            return np.asarray(q) + np.array((0.10, 0.05, 0.025, 0.0, 0.0, 0.0))

        with patch.object(arm_control.arm_kin, "dls_solve", side_effect=solve):
            arm.apply_delta(dx=0.10)

        np.testing.assert_allclose(
            published[0], (0.02, 0.01, 0.005, 0.0, 0.0, 0.0), atol=1e-9
        )

    def test_cartesian_servo_waits_for_yaw_after_position_is_settled(self):
        client = _FollowingClient()
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=20.0,
            max_ticks_per_click=20,
            pos_step=0.008,
            settle_tol_m=0.015,
            settle_tol_rot_rad=0.05,
            ik_joint_step=0.10,
            grip_close=0.68,
        )
        arm.kin = _YawKin()
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm.target_pos = None
        arm.target_quat = None
        arm._closed_since_last_open = False
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        published = []

        def publish(q):
            client.q = np.asarray(q).copy()
            published.append(client.q.copy())

        def solve(_kin, q, _target_pos, _target_quat):
            result = np.asarray(q).copy()
            result[1] = 0.60
            return result

        arm._publish = publish
        with patch.object(arm_control.arm_kin, "dls_solve", side_effect=solve):
            arm.apply_delta(dyaw=0.60)

        self.assertGreaterEqual(len(published), 6)
        self.assertAlmostEqual(0.60, client.q[1], places=6)

    def test_grasp_pose_uses_multiple_servo_passes_for_large_wrist_turn(self):
        client = _FollowingClient()
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.cfg = SimpleNamespace(
            rate_hz=20.0,
            max_ticks_per_click=4,
            pos_step=0.008,
            settle_tol_m=0.015,
            settle_tol_rot_rad=0.05,
            ik_joint_step=0.10,
            grip_close=0.68,
        )
        arm.kin = _YawKin()
        arm.grip_cmd = 0.0
        arm.grip_target = 0.0
        arm.target_pos = None
        arm.target_quat = None
        arm._closed_since_last_open = False
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        arm._publish = lambda q: setattr(client, "q", np.asarray(q).copy())

        def solve(_kin, q, _target_pos, _target_quat):
            result = np.asarray(q).copy()
            result[1] = 1.20
            return result

        with patch.object(arm_control.arm_kin, "dls_solve", side_effect=solve):
            ok = arm.move_grasp_pose((0.0, 0.0, 0.0), 1.20)

        self.assertTrue(ok)
        self.assertAlmostEqual(1.20, client.q[1], places=6)

    def test_final_grasp_descent_tracks_one_continuous_centered_line(self):
        client = _FollowingClient()
        client.q[:3] = (0.48, -0.02, -0.10)
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.kin = _XYZKin()
        arm.cfg = SimpleNamespace(
            rate_hz=20.0,
            pos_step=0.008,
            max_ticks_per_click=40,
            settle_tol_rot_rad=0.05,
        )
        arm.grip_cmd = 0.0
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        arm.target_pos = None
        arm.target_quat = None
        published = []

        def publish(q):
            client.q = np.asarray(q, dtype=float).copy()
            published.append(client.q.copy())

        def solve(_kin, q, target_pos, _target_quat):
            result = np.asarray(q, dtype=float).copy()
            result[:3] = np.asarray(target_pos, dtype=float)
            return result

        arm._publish = publish
        with patch.dict(
            "os.environ",
            {
                "MARC_GRASP_LINE_STEP_M": "0.03",
                "MARC_GRASP_LINE_MAX_XY_ERROR_M": "0.02",
                "MARC_GRASP_LINE_JOINT_STEP_RAD": "0.025",
            },
        ), patch.object(arm_control.arm_kin, "dls_solve", side_effect=solve):
            ok = arm.move_grasp_line((0.48, -0.02, -0.22), 0.0, "contact")

        self.assertTrue(ok)
        self.assertGreater(len(published), 10)
        self.assertTrue(all(np.allclose(p[:2], (0.48, -0.02)) for p in published))
        self.assertTrue(np.all(np.diff([p[2] for p in published]) <= 1e-9))
        np.testing.assert_allclose(client.q[:3], (0.48, -0.02, -0.22), atol=0.006)

    def test_contact_pose_is_held_stable_before_close(self):
        client = _FollowingClient()
        client.q[:3] = (0.40, 0.01, -0.31)
        arm = object.__new__(ArmTeleop)
        arm._client = client
        arm.kin = _XYZKin()
        arm.cfg = SimpleNamespace(rate_hz=10.0)
        arm.grip_cmd = 0.0
        arm._log = lambda _line: None
        arm._sleep = lambda _dt: None
        published = []
        arm._publish = lambda q: published.append(np.asarray(q).copy())

        with patch.dict(
            "os.environ",
            {
                "MARC_GRASP_CONTACT_SETTLE_S": "0.5",
                "MARC_GRASP_CONTACT_MAX_DRIFT_M": "0.006",
            },
        ):
            ok = arm.stabilize_grasp_pose()

        self.assertTrue(ok)
        self.assertEqual(5, len(published))
        self.assertTrue(all(np.allclose(q, client.q) for q in published))

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

    def test_place_lift_preserves_live_grasp_xy_and_orientation_seed(self):
        arm = object.__new__(ArmTeleop)
        arm._place_params = {
            "carry_up": 0.246,
            "place_fwd": -0.29,
            "place_lat": 0.0,
            "place_up": 0.05,
        }
        arm.cfg = SimpleNamespace(place_ik_guard=1.2, place_lift_ik_guard=1.4)

        class _StartKin:
            @staticmethod
            def fk(q):
                result = np.eye(4)
                result[:3, 3] = np.asarray(q[:3], dtype=float)
                return result

        arm.kin = _StartKin()
        calls = []

        def guarded(label, target, seed, max_joint_dev=None):
            calls.append(
                (
                    label,
                    np.asarray(target, dtype=float),
                    np.asarray(seed).copy(),
                    max_joint_dev,
                )
            )
            return np.asarray(seed).copy()

        arm._guarded_ik = guarded
        q_grasp = np.array([0.417, 0.012, -0.252, 0.4, -1.1, 2.0])

        pose = arm._grasp_lift_pose(q_grasp)

        self.assertIsNotNone(pose)
        label, target, seed, max_joint_dev = calls[0]
        self.assertEqual("grasp-high", label)
        np.testing.assert_allclose(target, (0.417, 0.012, 0.246))
        np.testing.assert_allclose(seed, q_grasp)
        self.assertAlmostEqual(1.4, max_joint_dev)

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
