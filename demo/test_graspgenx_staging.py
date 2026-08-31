"""Focused tests for the deterministic-to-GraspGen-X handoff."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from manip_dataset.graspgenx_pick import (
    StagedGraspGenXPickRuntime,
    filter_object_points,
    object_contact_z,
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
        self.cfg = SimpleNamespace(grip_close=0.58, grip_extra=0.04)
        self.grip_cmd = 0.0

    def stage_for_openvla(self, _target):
        self.events.append("stage")
        return True

    def move_grasp_pose(self, xyz, yaw, label):
        self.events.append((label, tuple(xyz), yaw))
        return True

    def set_grip(self, value):
        self.grip_cmd = float(value)
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
        good = _topdown_pose((0.40, 0.01, -0.34))
        side = good.copy()
        side[:3, :3] = np.eye(3)
        far = _topdown_pose((0.70, 0.0, -0.34))

        selected = select_topdown_contact_pose(
            (np.stack([side, far, good]), np.array([0.99, 0.95, 0.80])),
            _cloud(),
            (0.40, 0.0, -0.40),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(2, selected["index"])
        np.testing.assert_allclose(selected["xyz"], (0.40, 0.01, -0.34), atol=1e-6)
        self.assertAlmostEqual(0.0, selected["yaw"])

    def test_selector_rejects_a_surface_close_above_the_object_body(self):
        surface = _topdown_pose((0.40, 0.0, -0.26))
        inserted = _topdown_pose((0.40, 0.0, -0.34))
        cloud = _cloud()
        cloud[:, 2] = np.linspace(-0.29, -0.26, len(cloud))

        selected = select_topdown_contact_pose(
            (np.stack([surface, inserted]), np.array([0.99, 0.80])),
            cloud,
            (0.40, 0.0, -0.40),
        )

        self.assertEqual(1, selected["index"])

    def test_contact_depth_scales_with_object_height(self):
        short = _cloud()
        short[:, 2] = np.linspace(-0.31, -0.26, len(short))
        tall = _cloud()
        tall[:, 2] = np.linspace(-0.46, -0.26, len(tall))

        short_z = object_contact_z(short, -0.31)
        tall_z = object_contact_z(tall, -0.46)

        self.assertGreater(short_z, tall_z)
        self.assertGreater(short_z, -0.31)
        self.assertGreater(tall_z, -0.46)

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
            return np.stack([_topdown_pose((0.37, 0.03, -0.34))]), np.array([0.9])

        observations = iter(
            (
                SimpleNamespace(fwd=0.41, lat=-0.01, z=-0.40, valid=True),
                SimpleNamespace(fwd=0.41, lat=-0.01, z=-0.34, valid=True),
            )
        )

        def observe():
            events.append("observe")
            return next(observations)

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
        self.assertEqual("observe", events[2])
        self.assertEqual("ggx-pregrasp", events[3][0])
        self.assertEqual("ggx-contact", events[4][0])
        np.testing.assert_allclose((0.39, 0.11), events[4][1][:2], atol=1e-9)
        self.assertAlmostEqual(object_contact_z(_cloud(), -0.40), events[4][1][2])
        self.assertEqual(("grip", 0.58), events[5])
        self.assertEqual("lift", events[6])
        self.assertEqual("observe", events[7])
        self.assertEqual(("grip", 0.703), events[8])
        self.assertTrue(runtime.poll_grasping())

    def test_hold_sensor_is_rejected_when_lift_motion_is_implausible(self):
        target = SimpleNamespace(
            fwd=0.40,
            lat=0.0,
            z=-0.40,
            valid=True,
            name="tumbler",
        )
        for after_z in (-0.40, -0.15):
            observations = iter(
                (
                    SimpleNamespace(fwd=0.40, lat=0.0, z=-0.40, valid=True),
                    SimpleNamespace(fwd=0.40, lat=0.0, z=after_z, valid=True),
                )
            )
            runtime = StagedGraspGenXPickRuntime(
                _Client(),
                log=lambda _msg: None,
                policy="graspgenx",
                grasp_generator=lambda _points: (
                    np.stack([_topdown_pose((0.40, 0.0, -0.34))]),
                    np.array([0.9]),
                ),
                target_observer=lambda: next(observations),
                points_fn=lambda _target, client=None: _cloud(),
                grasp_poll=1,
            )
            runtime._teleop = _Arm([])

            runtime._grasp(target)

            self.assertFalse(runtime.poll_grasping())


if __name__ == "__main__":
    unittest.main()
