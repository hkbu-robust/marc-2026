"""OpenVLA deterministic handoff and false-positive grasp gate."""

from __future__ import annotations

import pytest
pytest.importorskip('rclpy')

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from nearfield import LocalizedTarget

from manip_dataset.app import CollectorApp
from manip_dataset.depth_approach import (
    ApproachConfig,
    DepthApproachController,
    arm_center_error,
)
from manip_dataset.openvla_pick import (
    StagedOpenVLAPickRuntime,
    graspgenx_stage_goal,
    graspgenx_stage_waypoints,
    openvla_ee_is_safe,
    openvla_stage_goal,
    openvla_stage_waypoints,
    openvla_target_is_safe,
    target_moved_with_gripper,
)
from manip_dataset.pick_target import snapshot_pick_target, stable_pick_target


def _target(*, z=-0.25, fwd=0.50, lat=0.0, valid=True):
    return SimpleNamespace(
        z=z,
        fwd=fwd,
        lat=lat,
        valid=valid,
        name="tumbler",
    )


class _Client:
    def __init__(self, holding=True):
        self.holding = holding
        self.t = 0.0
        self.pose = (0.0, 0.0, 0.0)

    def is_grasping(self):
        return self.holding

    def sleep(self, dt):
        self.t += dt

    def get_world_pose(self):
        return self.pose


class _Arm:
    def __init__(self):
        self.cfg = SimpleNamespace(
            grip_close=0.68,
            dx=0.02,
            dy=0.02,
            dz=0.02,
            d_yaw=0.05,
            d_pitch=0.04,
        )
        self.staged = []
        self.lifts = 0
        self.commands = []
        self.ready_calls = 0
        self.camera_warms = []

    def stage_for_openvla(self, target):
        self.staged.append(target)
        return True

    def lift_for_grasp_verification(self):
        self.lifts += 1
        return True

    def apply_delta(self, **kwargs):
        self.commands.append(kwargs)

    def go_ready(self, **_kwargs):
        self.ready_calls += 1
        return True

    def warm_cameras(self, seconds):
        self.camera_warms.append(seconds)


class OpenVLAStagingTests(unittest.TestCase):
    def test_picker_and_collector_share_validated_target_snapshot(self):
        target = _target()
        target.conf = 0.91
        nearfield = SimpleNamespace(
            detector=None,
            snapshot=lambda name: target if name == "tumbler" else None,
        )
        logs = []

        result = snapshot_pick_target(nearfield, "tumbler", log=logs.append)

        self.assertIs(target, result)
        self.assertTrue(any("localized tumbler" in line for line in logs))

    def test_upright_tumbler_uses_its_farther_pick_reach_envelope(self):
        # Regression for ten stable 0.94-confidence samples previously rejected
        # only because the upright ceiling was still 0.63 m.
        upright = _target(fwd=0.671)
        upright.conf = 0.91
        upright.bbox_xyxy = (20.0, 10.0, 70.0, 140.0)
        nearfield = SimpleNamespace(detector=None, snapshot=lambda _name: upright)

        result = stable_pick_target(nearfield, "tumbler", log=lambda _line: None)

        self.assertIs(upright, result)

    def test_lying_tumbler_keeps_the_validated_nearer_pick_envelope(self):
        lying = _target(fwd=0.601)
        lying.conf = 0.91
        lying.bbox_xyxy = (10.0, 20.0, 110.0, 70.0)
        nearfield = SimpleNamespace(detector=None, snapshot=lambda _name: lying)

        result = stable_pick_target(nearfield, "tumbler", log=lambda _line: None)

        self.assertIsNone(result)

    def test_initial_pick_inherits_nearby_upright_approach_posture(self):
        # Close-range boxes can become almost square without the object falling.
        candidate = _target(fwd=0.593, lat=-0.017, z=-0.281)
        candidate.conf = 0.73
        candidate.bbox_xyxy = (20.0, 20.0, 120.0, 110.0)
        hint = _target(fwd=0.579, lat=-0.006, z=-0.370)
        hint.conf = 0.88
        hint.bbox_xyxy = (40.0, 10.0, 90.0, 140.0)
        nearfield = SimpleNamespace(detector=None, snapshot=lambda _name: candidate)
        logs = []

        result = stable_pick_target(
            nearfield,
            "tumbler",
            log=logs.append,
            posture_hint=hint,
        )

        self.assertIs(candidate, result)
        self.assertEqual("upright", candidate.grasp_posture)
        self.assertTrue(any("inherits approach posture=upright" in x for x in logs))

    def test_one_fresh_snapshot_can_confirm_maintained_approach_lock(self):
        candidate = _target(fwd=0.591, lat=0.010, z=-0.429)
        candidate.conf = 0.91
        candidate.bbox_xyxy = (40.0, 10.0, 90.0, 140.0)
        hint = _target(fwd=0.576, lat=0.006, z=-0.430)
        hint.conf = 0.93
        hint.bbox_xyxy = (40.0, 10.0, 90.0, 140.0)
        snapshots = iter((candidate,) + (None,) * 9)
        calls = []

        def snapshot(_name):
            calls.append(True)
            return next(snapshots)

        logs = []
        result = stable_pick_target(
            SimpleNamespace(detector=None, snapshot=snapshot),
            "tumbler",
            log=logs.append,
            posture_hint=hint,
        )

        self.assertIs(candidate, result)
        self.assertEqual(1, len(calls))
        self.assertEqual("upright", candidate.grasp_posture)
        self.assertTrue(any("seeded localization consensus" in x for x in logs))

    def test_initial_upright_hint_survives_a_truncated_horizontal_box(self):
        candidate = _target(fwd=0.593, lat=-0.017, z=-0.281)
        candidate.conf = 0.91
        candidate.bbox_xyxy = (10.0, 40.0, 150.0, 90.0)
        hint = _target(fwd=0.579, lat=-0.006, z=-0.370)
        hint.conf = 0.88
        hint.bbox_xyxy = (40.0, 10.0, 90.0, 140.0)
        nearfield = SimpleNamespace(detector=None, snapshot=lambda _name: candidate)

        result = stable_pick_target(
            nearfield,
            "tumbler",
            log=lambda _line: None,
            posture_hint=hint,
        )

        self.assertIs(candidate, result)
        self.assertEqual("upright", candidate.grasp_posture)

    def test_collector_stages_before_unlocking_manual_demonstration(self):
        target = _target()
        target.conf = 0.91
        arm = _Arm()
        arm.cfg.openvla_stage_back_m = 0.01
        arm.cfg.openvla_stage_min_forward_m = 0.16
        arm.cfg.openvla_stage_up_m = 0.225
        arm.cfg.primary_camera = "base_left"
        app = object.__new__(CollectorApp)
        app.target_object = "tumbler"
        app.nearfield = SimpleNamespace(
            detector=None,
            snapshot=lambda _name: target,
        )
        app.arm = arm
        app.recorder = SimpleNamespace(meta={})

        self.assertTrue(app._stage_manual_handoff())

        self.assertEqual([target], arm.staged)
        handoff = app.recorder.meta["openvla_handoff"]
        self.assertEqual("base_left", handoff["primary_camera"])
        self.assertEqual(0.01, handoff["stage_back_m"])
        self.assertEqual(0.16, handoff["stage_min_forward_m"])
        self.assertEqual(0.225, handoff["stage_up_m"])

    def test_collector_retries_failed_handoff_three_times(self):
        app = object.__new__(CollectorApp)
        app.arm = _Arm()
        outcomes = iter((False, False, True))
        attempts = []

        def stage():
            attempts.append(True)
            return next(outcomes)

        app._stage_manual_handoff = stage

        self.assertTrue(app._stage_manual_handoff_with_retries())
        self.assertEqual(3, len(attempts))
        self.assertEqual(2, app.arm.ready_calls)
        self.assertEqual([0.5, 0.5], app.arm.camera_warms)

    def test_collector_uses_near_lock_when_fresh_snapshot_misses(self):
        for object_name in ("tissue", "sunblock"):
            with self.subTest(object_name=object_name):
                target = _target(fwd=0.45, lat=0.01)
                target.name = object_name
                target.conf = 0.16
                arm = _Arm()
                arm.cfg.openvla_stage_back_m = 0.01
                arm.cfg.openvla_stage_min_forward_m = 0.16
                arm.cfg.openvla_stage_up_m = 0.225
                arm.cfg.primary_camera = "base_left"
                app = object.__new__(CollectorApp)
                app.target_object = object_name
                app.nearfield = SimpleNamespace(
                    detector=None,
                    snapshot=lambda _name: None,
                )
                app.approach = SimpleNamespace(locked_target=lambda _name: target)
                app.arm = arm
                app.recorder = SimpleNamespace(meta={})

                self.assertTrue(app._stage_manual_handoff())
                self.assertEqual([target], arm.staged)

    def test_near_lock_is_class_keyed_and_drives_range_from_world_pose(self):
        client = _Client()
        # This test isolates world-pose lock projection, not the approach stop
        # boundary; keep its projected 0.50 m sample outside the range floor.
        controller = DepthApproachController(
            client, cfg=ApproachConfig(already_near_m=0.44)
        )
        controller.set_target("sunblock")
        locked = LocalizedTarget(
            fwd=0.60,
            lat=0.01,
            z=-0.25,
            name="sunblock",
            conf=0.16,
            world_xy=(0.78, 0.0),
            valid=True,
            reason="ok",
        )
        det = {
            "bbox": (10.0, 10.0, 20.0, 20.0),
            "rgb_shape": (480, 640),
            "name": "sunblock",
            "conf": 0.16,
            "n_obj": 1,
        }

        controller._maybe_lock_near_target(det, locked)

        self.assertIs(locked, controller.locked_target("sunblock"))
        self.assertIsNone(controller.locked_target("tissue"))
        client.pose = (0.10, 0.0, 0.0)
        reused = controller._reuse_near_lock(None, (480, 640))
        self.assertIsNotNone(reused)
        projected = controller.locked_target("sunblock")
        self.assertAlmostEqual(0.50, projected.fwd)
        self.assertAlmostEqual(0.0, projected.lat)
        lin, ang = controller._command({
            "lat_n_obj": 1,
            "lat_err": reused["error"],
            "fwd": projected.fwd,
            "d_obj": None,
            "is_new": True,
        })
        self.assertGreater(lin, 0.0)
        self.assertEqual(0.0, ang)
        client.pose = (0.10, 0.40, 0.0)
        self.assertIsNone(controller._reuse_near_lock(None, (480, 640)))

    def test_near_flicker_lock_is_available_for_every_selected_class(self):
        for object_name in ("tumbler", "bottle", "tissue", "sunblock"):
            with self.subTest(object_name=object_name):
                client = _Client()
                controller = DepthApproachController(
                    client, cfg=ApproachConfig(already_near_m=0.49)
                )
                controller.set_target(object_name)
                localized = LocalizedTarget(
                    fwd=0.58,
                    lat=0.02,
                    z=-0.30,
                    name=object_name,
                    conf=0.21,
                    world_xy=(0.76, 0.02),
                    valid=True,
                    reason="ok",
                )
                det = {
                    "bbox": (250.0, 200.0, 350.0, 330.0),
                    "rgb_shape": (480, 640),
                    "name": object_name,
                    "conf": 0.21,
                    "n_obj": 1,
                }

                controller._maybe_lock_near_target(det, localized)
                self.assertIs(localized, controller.locked_target(object_name))
                reused = controller._reuse_near_lock(None, (480, 640))

                self.assertIsNotNone(reused)
                self.assertTrue(reused["near_lock_reused"])
                self.assertEqual(object_name, reused["name"])

    def test_logged_flicker_case_locks_and_drives_before_strict_center(self):
        controller = DepthApproachController(
            _Client(),
            cfg=ApproachConfig(
                stop_m=0.56,
                already_near_m=0.49,
                center_tol=0.05,
                realign=0.12,
                target_lock_acquire_pad_m=0.25,
                target_lock_drive_tol=0.10,
            ),
        )
        controller.set_target("tumbler")
        localized = LocalizedTarget(
            fwd=0.72,
            lat=0.05,
            z=-0.30,
            name="tumbler",
            conf=0.54,
            world_xy=(0.90, 0.05),
            valid=True,
            reason="ok",
        )
        det = {
            "bbox": (300.0, 200.0, 390.0, 330.0),
            "rgb_shape": (480, 640),
            "name": "tumbler",
            "conf": 0.54,
            "n_obj": 1,
        }

        controller._maybe_lock_near_target(det, localized)
        lin, ang = controller._command(
            {
                "lat_n_obj": 1,
                "lat_err": 0.08,
                "fwd": 0.72,
                "d_obj": 0.59,
                "target_lock_active": True,
                "is_new": True,
            }
        )

        self.assertIsNotNone(controller.locked_target("tumbler"))
        self.assertGreater(lin, 0.0)
        self.assertEqual(0.0, ang)

    def test_arm_center_error_targets_robot_axis_not_left_pixel_midpoint(self):
        # The historical pixel error can look centered while RGB-D says the
        # object is still 6 cm off the arm axis.
        pixel_err = -0.03

        error = arm_center_error(0.60, 0.06)

        self.assertAlmostEqual(-0.10, error)
        self.assertNotAlmostEqual(pixel_err, error)

    def test_range_boundary_blocks_forward_but_keeps_final_centering_yaw(self):
        controller = DepthApproachController(
            _Client(),
            cfg=ApproachConfig(
                already_near_m=0.70,
                upright_already_near_m=0.70,
                center_tol=0.03,
                yaw_gain=1.2,
                yaw_min=0.10,
                yaw_max=0.35,
                log_every=0,
            ),
        )
        controller._phase = "drive"
        off_axis = {
            "lat_n_obj": 1,
            "lat_err": -0.08,
            "fwd": 0.68,
            "name": "tumbler",
            "bbox": (20.0, 10.0, 70.0, 140.0),
            "is_new": True,
        }

        linear, angular = controller._command(off_axis)

        self.assertEqual(0.0, linear)
        self.assertGreater(angular, 0.0)
        self.assertEqual("align", controller._phase)
        centered = {**off_axis, "lat_err": -0.02}
        self.assertEqual((0.0, 0.0), controller._command(centered))

    def test_far_or_wrong_class_glimpse_cannot_establish_approach_lock(self):
        controller = DepthApproachController(
            _Client(), cfg=ApproachConfig(already_near_m=0.49)
        )
        controller.set_target("tumbler")
        det = {
            "bbox": (250.0, 200.0, 350.0, 330.0),
            "rgb_shape": (480, 640),
            "name": "tumbler",
            "conf": 0.21,
            "n_obj": 1,
        }
        far = LocalizedTarget(
            fwd=0.90,
            lat=0.0,
            z=-0.30,
            name="tumbler",
            world_xy=(1.08, 0.0),
            valid=True,
        )
        wrong_class = replace(far, fwd=0.58, name="bottle")
        no_world_pose = replace(far, fwd=0.58, world_xy=None)

        controller._maybe_lock_near_target(det, far)
        self.assertIsNone(controller.locked_target("tumbler"))
        controller._maybe_lock_near_target(det, wrong_class)
        self.assertIsNone(controller.locked_target("tumbler"))
        controller._maybe_lock_near_target(det, no_world_pose)
        self.assertIsNone(controller.locked_target("tumbler"))

    def test_upright_bbox_uses_farther_hard_approach_stop(self):
        controller = DepthApproachController(
            _Client(),
            cfg=ApproachConfig(
                already_near_m=0.49,
                upright_bbox_aspect=1.20,
                upright_already_near_m=0.70,
            ),
        )
        common = {
            "lat_n_obj": 1,
            "lat_err": 0.0,
            "fwd": 0.68,
            "d_obj": 0.45,
            "name": "tumbler",
        }

        self.assertTrue(
            controller._too_near({**common, "bbox": (10.0, 10.0, 50.0, 90.0)})
        )
        self.assertFalse(
            controller._too_near({**common, "bbox": (10.0, 10.0, 90.0, 50.0)})
        )

    def test_tall_non_tumbler_keeps_generic_approach_stop(self):
        controller = DepthApproachController(
            _Client(),
            cfg=ApproachConfig(
                already_near_m=0.49,
                upright_bbox_aspect=1.20,
                upright_already_near_m=0.70,
            ),
        )
        measurement = {
            "lat_n_obj": 1,
            "lat_err": 0.0,
            "fwd": 0.68,
            "d_obj": 0.45,
            "name": "bottle",
            "bbox": (10.0, 10.0, 50.0, 90.0),
        }

        self.assertFalse(controller._too_near(measurement))

    def test_default_upright_stop_preserves_side_grasp_workspace(self):
        cfg = ApproachConfig()

        self.assertEqual(0.70, cfg.upright_already_near_m)
        self.assertGreater(cfg.upright_already_near_m, cfg.already_near_m)


    def test_stage_goal_enters_recorded_base_camera_workspace(self):
        cfg = SimpleNamespace(
            openvla_stage_back_m=0.01,
            openvla_stage_min_forward_m=0.16,
            openvla_stage_up_m=0.225,
        )
        goal = openvla_stage_goal(_target(lat=-0.015), cfg)
        np.testing.assert_allclose(goal, [0.49, -0.015, -0.025], atol=1e-6)

    def test_stage_moves_forward_before_it_descends(self):
        cfg = SimpleNamespace(
            openvla_stage_back_m=0.01,
            openvla_stage_min_forward_m=0.16,
            openvla_stage_up_m=0.225,
        )
        current = np.array([0.329, 0.001, 0.094])
        waypoints = openvla_stage_waypoints(current, _target(lat=-0.015), cfg)
        self.assertEqual(["stage-forward", "stage-align", "stage-view"], [x[0] for x in waypoints])
        forward, align, view = (x[1] for x in waypoints)
        self.assertGreater(forward[0], current[0])
        self.assertEqual(current[2], forward[2])
        self.assertEqual(forward[0], align[0])
        self.assertEqual(align[0], view[0])
        self.assertLess(view[2], align[2])

    def test_stage_guarantees_large_forward_clearance_before_descent(self):
        cfg = SimpleNamespace(
            openvla_stage_back_m=0.01,
            openvla_stage_min_forward_m=0.16,
            openvla_stage_up_m=0.225,
        )
        current = np.array([0.329, 0.001, 0.094])
        waypoints = openvla_stage_waypoints(current, _target(fwd=0.40), cfg)

        forward, align, view = (waypoint for _label, waypoint in waypoints)
        self.assertAlmostEqual(0.489, forward[0])
        self.assertAlmostEqual(forward[0], align[0])
        self.assertAlmostEqual(forward[0], view[0])
        self.assertEqual(current[2], forward[2])
        self.assertEqual(current[2], align[2])
        self.assertLess(view[2], align[2])

    def test_graspgenx_stage_ends_directly_above_object_centre(self):
        cfg = SimpleNamespace(openvla_stage_up_m=0.225)
        current = np.array([0.329, 0.001, 0.094])
        target = _target(fwd=0.413, lat=-0.095, z=-0.303)

        goal = graspgenx_stage_goal(target, cfg)
        waypoints = graspgenx_stage_waypoints(current, target, cfg)
        forward, align, view = (waypoint for _label, waypoint in waypoints)

        np.testing.assert_allclose(goal, [0.413, -0.095, -0.078])
        self.assertEqual(current[2], forward[2])
        self.assertEqual(current[2], align[2])
        np.testing.assert_allclose(view[:2], [target.fwd, target.lat])

    def test_upright_graspgenx_stage_stays_behind_singular_full_reach(self):
        cfg = SimpleNamespace(openvla_stage_up_m=0.225)
        current = np.array([0.329, 0.001, 0.094])
        target = _target(fwd=0.591, lat=0.013, z=-0.315)

        goal = graspgenx_stage_goal(target, cfg, stage_back_m=0.05)
        waypoints = graspgenx_stage_waypoints(
            current, target, cfg, stage_back_m=0.05
        )

        np.testing.assert_allclose(goal, [0.541, 0.013, -0.090], atol=1e-9)
        self.assertTrue(all(point[0] == goal[0] for _label, point in waypoints))

    def test_finite_localization_is_not_rejected_by_guessed_target_ranges(self):
        self.assertTrue(openvla_target_is_safe(_target(fwd=0.325))[0])
        self.assertTrue(openvla_target_is_safe(_target(fwd=0.684))[0])
        self.assertTrue(openvla_target_is_safe(_target(z=-0.430))[0])
        self.assertFalse(openvla_target_is_safe(_target(z=float("nan")))[0])

    def test_only_forward_camera_keepout_gates_policy_motion(self):
        self.assertFalse(openvla_ee_is_safe([0.31, 0.0, 0.0])[0])
        self.assertTrue(openvla_ee_is_safe([0.42, 0.0, -0.13])[0])
        self.assertTrue(openvla_ee_is_safe([0.42, 0.5, -0.43])[0])
        self.assertTrue(openvla_ee_is_safe([0.70, 0.0, -0.43])[0])

    def test_observed_low_tumbler_height_produces_deterministic_stage(self):
        cfg = SimpleNamespace(
            openvla_stage_back_m=0.01,
            openvla_stage_min_forward_m=0.16,
            openvla_stage_up_m=0.225,
        )
        target = _target(fwd=0.545, lat=-0.022, z=-0.430)

        np.testing.assert_allclose(
            openvla_stage_goal(target, cfg),
            [0.535, -0.022, -0.205],
            atol=1e-6,
        )

    def test_target_must_rise_with_gripper(self):
        before = _target()
        self.assertTrue(target_moved_with_gripper(before, _target(z=-0.18, fwd=0.51)))
        self.assertFalse(target_moved_with_gripper(before, _target(z=-0.25)))
        self.assertFalse(target_moved_with_gripper(before, _target(z=-0.18, fwd=0.80)))
        self.assertFalse(target_moved_with_gripper(before, _target(valid=False)))

    def test_unverified_hold_is_not_grasp_success(self):
        runtime = StagedOpenVLAPickRuntime(
            _Client(holding=True),
            policy="openvla_remote",
            openvla_policy=SimpleNamespace(),
            target_observer=lambda: _target(z=-0.25),
            poll_dt=0.0,
            grasp_poll=1,
        )
        self.assertFalse(runtime.poll_grasping())

    def test_stages_before_policy_and_accepts_visual_lift(self):
        arm = _Arm()
        client = _Client(holding=True)
        policy = SimpleNamespace(
            stats=object(),
            act=lambda _obs: np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        )
        runtime = StagedOpenVLAPickRuntime(
            client,
            policy="openvla_remote",
            openvla_policy=policy,
            target_observer=lambda: _target(z=-0.18),
            policy_ticks=1,
            poll_dt=0.0,
            grasp_poll=1,
        )
        runtime._teleop = arm
        runtime._snapshot_obs = lambda _target: SimpleNamespace(
            ee_xyz=np.array([0.42, 0.0, -0.13])
        )
        target = _target()

        runtime._grasp(target)

        self.assertEqual([target], arm.staged)
        self.assertEqual(1, arm.lifts)
        self.assertTrue(runtime._visual_grasp_verified)
        self.assertTrue(runtime.poll_grasping())

    def test_false_hold_reopens_and_restages(self):
        arm = _Arm()
        client = _Client(holding=True)
        policy = SimpleNamespace(
            stats=object(),
            act=lambda _obs: np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        )
        ground = _target(z=-0.25)
        runtime = StagedOpenVLAPickRuntime(
            client,
            policy="openvla_remote",
            openvla_policy=policy,
            target_observer=lambda: ground,
            policy_ticks=1,
        )
        runtime._teleop = arm
        runtime._snapshot_obs = lambda _target: SimpleNamespace(
            ee_xyz=np.array([0.42, 0.0, -0.13])
        )

        runtime._grasp(ground)

        self.assertEqual(2, len(arm.staged))
        self.assertFalse(runtime._visual_grasp_verified)


if __name__ == "__main__":
    unittest.main()
