"""Focused tests for the deterministic-to-GraspGen-X handoff."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from manip_dataset.graspgenx_pick import (
    StagedGraspGenXPickRuntime,
    filter_object_points,
    select_topdown_contact_pose,
)


def _topdown_pose(contact_xyz):
    pose = np.eye(4, dtype=np.float64)
    # +X closing, +Z approach; point local +Z down into the object.
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    contact = np.asarray(contact_xyz, dtype=np.float64)
    pose[:3, 3] = contact - pose[:3, 2] * 0.195
    return pose


def _cloud(n=80):
    x = np.linspace(0.36, 0.44, n)
    y = np.linspace(-0.035, 0.035, n)
    z = np.linspace(-0.39, -0.26, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


class _Client:
    def __init__(self):
        self.t = 0.0

    def is_grasping(self):
        return True

    def sleep(self, dt):
        self.t += float(dt)

    def now_s(self):
        return self.t


class _Arm:
    def __init__(self, events):
        self.events = events

    def stage_for_openvla(self, _target):
        self.events.append("stage")
        return True

    def move_grasp_pose(self, xyz, yaw, label):
        self.events.append((label, tuple(xyz), yaw))
        return True

    def set_grip(self, value):
        self.events.append(("grip", float(value)))
        return True

    def lift_for_grasp_verification(self):
        self.events.append("lift")
        return True

    def go_ready(self, open_grip=True):
        self.events.append(("ready", open_grip))
        return True


class GraspGenXStagingTests(unittest.TestCase):
    def test_support_plane_is_removed_but_object_is_retained(self):
        rng = np.random.default_rng(7)
        support = np.column_stack(
            [
                rng.uniform(0.34, 0.46, 60),
                rng.uniform(-0.06, 0.06, 60),
                rng.normal(-0.40, 0.002, 60),
            ]
        )
        obj = np.column_stack(
            [
                rng.uniform(0.37, 0.43, 80),
                rng.uniform(-0.03, 0.03, 80),
                rng.uniform(-0.36, -0.25, 80),
            ]
        )
        far = np.column_stack(
            [
                rng.uniform(0.70, 0.80, 40),
                rng.uniform(0.30, 0.40, 40),
                rng.uniform(-0.40, -0.20, 40),
            ]
        )
        target = SimpleNamespace(fwd=0.40, lat=0.0)

        filtered = filter_object_points(np.vstack([support, obj, far]), target)

        self.assertGreaterEqual(filtered.shape[0], 70)
        self.assertLess(filtered.shape[0], 110)
        self.assertGreater(float(np.min(filtered[:, 2])), -0.39)
        self.assertLess(float(np.max(np.abs(filtered[:, 1]))), 0.14)

    def test_selector_converts_gripper_base_to_fingertip_and_rejects_side_pose(self):
        good = _topdown_pose((0.40, 0.01, -0.28))
        side = good.copy()
        side[:3, :3] = np.eye(3)
        far = _topdown_pose((0.70, 0.0, -0.28))

        selected = select_topdown_contact_pose(
            (np.stack([side, far, good]), np.array([0.99, 0.95, 0.80])),
            _cloud(),
            (0.40, 0.0, -0.40),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(2, selected["index"])
        np.testing.assert_allclose(selected["xyz"], (0.40, 0.01, -0.28), atol=1e-6)
        self.assertAlmostEqual(0.0, selected["yaw"])

    def test_runtime_stages_before_inference_then_short_approach_snap_and_lift(self):
        events = []
        client = _Client()
        target = SimpleNamespace(
            fwd=0.40,
            lat=0.0,
            z=-0.40,
            valid=True,
            name="tumbler",
        )

        def generate(_points):
            events.append("infer")
            return np.stack([_topdown_pose((0.40, 0.0, -0.28))]), np.array([0.9])

        def observe():
            events.append("observe")
            return SimpleNamespace(fwd=0.40, lat=0.0, z=-0.34, valid=True)

        runtime = StagedGraspGenXPickRuntime(
            client,
            log=lambda _msg: None,
            policy="graspgenx",
            grasp_generator=generate,
            target_observer=observe,
            points_fn=lambda _target, client=None: _cloud(),
            grasp_poll=1,
        )
        runtime._teleop = _Arm(events)

        runtime._grasp(target)

        self.assertEqual("stage", events[0])
        self.assertEqual("infer", events[1])
        self.assertEqual("ggx-pregrasp", events[2][0])
        self.assertEqual("ggx-contact", events[3][0])
        self.assertEqual(("grip", 0.68), events[4])
        self.assertEqual("lift", events[5])
        self.assertEqual("observe", events[6])
        self.assertTrue(runtime.poll_grasping())

    def test_hold_sensor_is_rejected_when_object_did_not_rise(self):
        events = []
        target = SimpleNamespace(
            fwd=0.40,
            lat=0.0,
            z=-0.40,
            valid=True,
            name="tumbler",
        )
        runtime = StagedGraspGenXPickRuntime(
            _Client(),
            log=lambda _msg: None,
            policy="graspgenx",
            grasp_generator=lambda _points: (
                np.stack([_topdown_pose((0.40, 0.0, -0.28))]),
                np.array([0.9]),
            ),
            target_observer=lambda: SimpleNamespace(
                fwd=0.40, lat=0.0, z=-0.40, valid=True
            ),
            points_fn=lambda _target, client=None: _cloud(),
            grasp_poll=1,
        )
        runtime._teleop = _Arm(events)

        runtime._grasp(target)

        self.assertFalse(runtime.poll_grasping())


if __name__ == "__main__":
    unittest.main()
