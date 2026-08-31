"""Focused tests for the deterministic-to-GraspGen-X handoff."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from manip_dataset.graspgenx_pick import (
    DEFAULT_FLOOR_CLEARANCE_M,
    DEFAULT_MAX_CLOUD_CENTER_SHIFT_M,
    DEFAULT_OVERHEAD_CLEARANCE_M,
    DEFAULT_STAGE_CLEARANCE_M,
    StagedGraspGenXPickRuntime,
    bounded_xy_center,
    contact_z_from_extent,
    filter_object_points,
    object_contact_z,
    object_vertical_extent,
    object_xy_footprint,
    select_topdown_contact_pose,
    upright_rear_insertion_waypoints,
    wrist_depth_object_observation,
    yolo_bbox_is_upright,
)
from manip_dataset.grasp_profiles import (
    LYING_TUMBLER,
    TUMBLER_UPRIGHT,
    resolve_grasp_profile,
)


def _topdown_pose(contact_xyz, yaw=0.0):
    pose = np.eye(4, dtype=np.float64)
    # +X closing, +Z approach; point local +Z down into the object.
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    pose[:3, :3] = np.array(
        [[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    contact = np.asarray(contact_xyz, dtype=np.float64)
    pose[:3, 3] = contact - pose[:3, 2] * 0.195
    return pose


def _cloud(n=80):
    x = np.linspace(0.36, 0.44, n)
    y = np.linspace(-0.035, 0.035, n)
    z = np.linspace(-0.39, -0.26, n)
    return np.stack([x, y, z], axis=1).astype(np.float32)


class _Client:
    def __init__(self, holding=True):
        self.t = 0.0
        self.holding = holding

    def is_grasping(self):
        return self.holding

    def sleep(self, dt):
        self.t += float(dt)

    def now_s(self):
        return self.t


class _Arm:
    def __init__(self, events):
        self.events = events
        self.cfg = SimpleNamespace(
            grip_close=0.58,
            grip_extra=0.04,
            openvla_stage_up_m=0.225,
        )
        self.grip_cmd = 0.0
        self.stage_target = None
        self.stage_profile = None
        self.settle_kwargs = None

    def stage_for_openvla(self, target):
        self.stage_target = target
        self.events.append("stage")
        return True

    def stage_for_graspgenx(self, target, *, profile=None):
        self.stage_target = target
        self.stage_profile = profile
        if profile is not None and profile.approach_style == "front_side":
            target.folded_pad_fwd = float(target.fwd) - profile.side_standoff_m
            target.folded_pad_lat = float(target.lat)
            target.folded_pad_z = (
                float(target.z) + profile.side_contact_fold_clearance_m
            )
        self.events.append("stage")
        return True

    def move_grasp_pose(self, xyz, yaw, label, **kwargs):
        self.events.append((label, tuple(xyz), yaw, kwargs))
        return True

    def move_grasp_line(self, xyz, yaw, label, **kwargs):
        self.events.append((label, tuple(xyz), yaw, kwargs))
        return True

    def stabilize_grasp_pose(self, label, **kwargs):
        self.settle_kwargs = kwargs
        self.events.append(label)
        return True

    def set_grip(self, value):
        self.grip_cmd = float(value)
        self.events.append(("grip", float(value)))
        return True

    def set_grip_immediate(self, value):
        self.grip_cmd = float(value)
        self.events.append(("grip-immediate", float(value)))
        return True

    def lift_for_grasp_verification(self):
        self.events.append("lift")
        return True

    def go_ready(self, open_grip=True):
        self.events.append(("ready", open_grip))
        return True


class GraspGenXStagingTests(unittest.TestCase):
    def test_flat_tumbler_selects_physically_validated_profile(self):
        target = SimpleNamespace(
            name="tumbler", bbox_xyxy=(10.0, 20.0, 110.0, 70.0)
        )

        profile = resolve_grasp_profile(target)

        self.assertIs(profile, LYING_TUMBLER)
        self.assertEqual("lying_tumbler", profile.name)
        self.assertEqual((0.32, 0.58), (profile.min_fwd_m, profile.max_fwd_m))
        self.assertEqual(0.0, profile.max_cloud_center_shift_m)
        self.assertEqual(0.03, profile.wrist_max_correction_m)

    def test_upright_tumbler_selects_validated_side_profile(self):
        upright = SimpleNamespace(
            name="tumbler", bbox_xyxy=(10.0, 20.0, 60.0, 120.0)
        )
        unknown = SimpleNamespace(name="tumbler")
        other_class = SimpleNamespace(
            name="bottle", bbox_xyxy=(10.0, 20.0, 110.0, 70.0)
        )

        profile = resolve_grasp_profile(upright)
        self.assertIs(profile, TUMBLER_UPRIGHT)
        self.assertEqual("tumbler_upright", profile.name)
        self.assertTrue(profile.physically_validated)
        self.assertEqual("front_side", profile.approach_style)
        self.assertEqual(0.10, profile.stage_back_m)
        self.assertEqual((0.32, 0.68), (profile.min_fwd_m, profile.max_fwd_m))
        self.assertEqual(0.44, profile.side_pregrasp_min_fwd_m)
        self.assertEqual(0.14, profile.side_clear_lift_m)
        self.assertEqual(0.65, profile.side_transit_tilt_rad)
        self.assertEqual(20.0, profile.side_transit_rate_hz)
        self.assertEqual(0.020, profile.side_transit_cart_step_m)
        self.assertEqual(0.055, profile.side_transit_joint_step_rad)
        self.assertEqual(1.10, profile.approach_tilt_rad)
        self.assertEqual(0.10, profile.side_contact_fold_clearance_m)
        self.assertEqual(0.015, profile.side_final_insertion_m)
        self.assertEqual(0.50, profile.side_fold_speed_rad_s)
        self.assertEqual(0.030, profile.pad_lateral_bias_m)
        self.assertEqual(0.0205, profile.pad_center_offset_m)
        self.assertEqual(0.35, profile.contact_depth_fraction)
        self.assertEqual(0.005, profile.center_pos_tolerance_m)
        self.assertEqual(0.05, profile.center_rot_tolerance_rad)
        self.assertEqual(0.004, profile.line_joint_step_rad)
        self.assertEqual(0.0, profile.max_cloud_center_shift_m)
        self.assertEqual(0.10, profile.side_standoff_m)
        self.assertEqual(profile.stage_back_m, profile.side_standoff_m)
        self.assertEqual(0.030, profile.wrist_max_correction_m)
        self.assertIsNone(resolve_grasp_profile(unknown))
        self.assertIsNone(resolve_grasp_profile(other_class))
        hinted_upright = SimpleNamespace(
            name="tumbler",
            bbox_xyxy=(10.0, 20.0, 110.0, 70.0),
            grasp_posture="upright",
        )
        self.assertIs(TUMBLER_UPRIGHT, resolve_grasp_profile(hinted_upright))

    def test_upright_side_approach_stays_rear_until_forward_insertion(self):
        contact = np.array((0.595, 0.037, -0.344))
        center_z = -0.068

        fold_high, pregrasp, final = upright_rear_insertion_waypoints(
            contact,
            TUMBLER_UPRIGHT.side_standoff_m,
            center_z,
        )

        self.assertAlmostEqual(pregrasp[0], fold_high[0])
        self.assertAlmostEqual(pregrasp[1], fold_high[1])
        self.assertAlmostEqual(
            contact[0] - TUMBLER_UPRIGHT.side_standoff_m,
            pregrasp[0],
        )
        self.assertAlmostEqual(contact[2], pregrasp[2])
        self.assertAlmostEqual(center_z, fold_high[2])
        self.assertGreater(fold_high[2], pregrasp[2])
        np.testing.assert_allclose(contact, final, atol=1e-12)

    def test_runtime_applies_lying_tumbler_profile_end_to_end(self):
        events = []
        messages = []
        cloud = _cloud()
        cloud[:, 0] += 0.02
        cloud[:, 1] += 0.03
        target = SimpleNamespace(
            fwd=0.40,
            lat=0.0,
            z=-0.40,
            valid=True,
            name="tumbler",
            bbox_xyxy=(10.0, 20.0, 110.0, 70.0),
        )
        runtime = StagedGraspGenXPickRuntime(
            _Client(),
            log=messages.append,
            policy="graspgenx",
            grasp_generator=lambda _points: (
                np.stack([_topdown_pose((0.42, 0.03, -0.34), yaw=0.55)]),
                np.array([0.9]),
            ),
            target_observer=lambda: SimpleNamespace(
                fwd=0.40, lat=0.0, z=-0.40, valid=True
            ),
            points_fn=lambda _target, client=None: cloud,
            grasp_poll=1,
        )
        arm = _Arm(events)
        runtime._teleop = arm

        runtime._grasp(target)

        contact = next(event for event in events if event[0] == "ggx-contact")
        np.testing.assert_allclose((0.40, 0.0), contact[1][:2], atol=1e-9)
        self.assertAlmostEqual(
            object_contact_z(
                cloud,
                -0.40,
                contact_depth_fraction=LYING_TUMBLER.contact_depth_fraction,
                floor_clearance_m=LYING_TUMBLER.floor_clearance_m,
            ),
            contact[1][2],
        )
        _support, top = object_vertical_extent(cloud, -0.40)
        self.assertAlmostEqual(
            top + LYING_TUMBLER.stage_clearance_m - arm.cfg.openvla_stage_up_m,
            arm.stage_target.z,
        )
        self.assertEqual(
            {
                "dwell_s": LYING_TUMBLER.contact_settle_s,
                "max_drift_m": LYING_TUMBLER.contact_max_drift_m,
            },
            arm.settle_kwargs,
        )
        self.assertTrue(any("grasp profile=lying_tumbler" in msg for msg in messages))

    def test_runtime_executes_upright_tumbler_as_oblique_side_grasp(self):
        events = []
        messages = []
        observed = []
        wrist_calls = []
        cloud = _cloud()
        cloud[:, 0] += 0.201
        target = SimpleNamespace(
            fwd=0.601,
            lat=0.0,
            z=-0.40,
            valid=True,
            name="tumbler",
            bbox_xyxy=(20.0, 10.0, 70.0, 140.0),
        )

        def generate(_points):
            events.append("inference")
            return (
                np.stack([_topdown_pose((0.601, 0.0, -0.34))]),
                np.array([0.9]),
            )

        runtime = StagedGraspGenXPickRuntime(
            _Client(),
            log=messages.append,
            policy="graspgenx",
            grasp_generator=generate,
            target_observer=lambda: observed.append(True),
            points_fn=lambda _target, client=None: cloud,
            grasp_poll=1,
        )
        arm = _Arm(events)
        runtime._teleop = arm
        runtime._wrist_depth_refinement = lambda *args, **kwargs: (
            wrist_calls.append(True) or None
        )

        runtime._grasp(target)

        contact = next(event for event in events if event[0] == "ggx-contact")
        self.assertLess(events.index("stage"), events.index("inference"))
        self.assertLess(events.index("inference"), events.index(contact))
        self.assertEqual([], observed)
        self.assertEqual([], wrist_calls)
        self.assertAlmostEqual(
            0.601 - TUMBLER_UPRIGHT.side_standoff_m
            + TUMBLER_UPRIGHT.side_final_insertion_m,
            contact[1][0],
            places=6,
        )
        self.assertAlmostEqual(
            TUMBLER_UPRIGHT.pad_lateral_bias_m,
            contact[1][1],
            places=6,
        )
        support_z, cloud_top_z = object_vertical_extent(cloud, -0.40)
        expected_z = contact_z_from_extent(
            support_z,
            cloud_top_z,
            contact_depth_fraction=TUMBLER_UPRIGHT.contact_depth_fraction,
            floor_clearance_m=TUMBLER_UPRIGHT.floor_clearance_m,
        )
        self.assertAlmostEqual(
            expected_z + TUMBLER_UPRIGHT.side_contact_fold_clearance_m,
            contact[1][2],
        )
        self.assertAlmostEqual(0.601, arm.stage_target.fwd)
        self.assertAlmostEqual(
            TUMBLER_UPRIGHT.pad_lateral_bias_m,
            arm.stage_target.lat,
        )
        self.assertAlmostEqual(expected_z, arm.stage_target.z)
        self.assertAlmostEqual(
            cloud_top_z + TUMBLER_UPRIGHT.stage_clearance_m,
            arm.stage_target.fold_z,
        )
        self.assertAlmostEqual(-0.5 * np.pi, arm.stage_target.grasp_yaw)
        self.assertAlmostEqual(
            TUMBLER_UPRIGHT.approach_tilt_rad,
            contact[3]["approach_tilt_rad"],
        )
        self.assertAlmostEqual(
            TUMBLER_UPRIGHT.pad_center_offset_m,
            contact[3]["pad_center_offset_m"],
        )
        self.assertAlmostEqual(
            TUMBLER_UPRIGHT.line_joint_step_rad,
            contact[3]["joint_step_rad"],
        )
        self.assertTrue(contact[3]["preserve_current_orientation"])
        self.assertEqual(0.002, contact[3]["cart_step_m"])
        self.assertEqual(20.0, contact[3]["control_rate_hz"])
        self.assertTrue(
            any(
                "grasp profile=tumbler_upright status=validated" in msg
                for msg in messages
            )
        )
        self.assertIs(TUMBLER_UPRIGHT, arm.stage_profile)
        self.assertTrue(
            any(
                "upright pad-center lateral bias object_lat=0.000 "
                "bias=+0.030 grasp_lat=0.030" in msg
                for msg in messages
            )
        )
        self.assertTrue(
            any(
                "upright handoff complete at folded rear pregrasp" in msg
                for msg in messages
            )
        )
        self.assertTrue(
            any(
                "GraspGen-X owns final forward insertion only" in msg
                for msg in messages
            )
        )

    def test_upright_side_pregrasp_inside_base_keepout_fails_before_tilt(self):
        events = []
        messages = []
        cloud = _cloud()
        cloud[:, 0] += 0.06
        target = SimpleNamespace(
            fwd=0.50,
            lat=0.0,
            z=-0.40,
            valid=True,
            name="tumbler",
            bbox_xyxy=(20.0, 10.0, 70.0, 140.0),
        )
        runtime = StagedGraspGenXPickRuntime(
            _Client(),
            log=messages.append,
            policy="graspgenx",
            grasp_generator=lambda _points: (
                np.stack([_topdown_pose((0.46, 0.0, -0.34))]),
                np.array([0.9]),
            ),
            target_observer=lambda: SimpleNamespace(
                fwd=0.50, lat=0.0, z=-0.40, valid=True
            ),
            points_fn=lambda _target, client=None: cloud,
            grasp_poll=1,
        )
        runtime._teleop = _Arm(events)

        runtime._grasp(target)

        self.assertFalse(
            any(
                event[0] == "ggx-center"
                for event in events
                if isinstance(event, tuple)
            )
        )
        self.assertTrue(any("inside base keepout" in msg for msg in messages))

    def test_wrist_depth_localizes_only_surface_near_base_prediction(self):
        depth = np.full((100, 100), 0.40, dtype=np.float32)
        depth[44:58, 54:68] = 0.20

        observed = wrist_depth_object_observation(
            depth,
            (100.0, 100.0, 50.0, 50.0),
            np.eye(4),
            (0.0, 0.0, 0.20),
            search_radius_m=0.10,
            depth_gate_m=0.04,
            min_points=12,
        )

        self.assertIsNotNone(observed)
        self.assertGreater(observed["n_points"], 12)
        self.assertAlmostEqual(0.020, observed["center"][0], delta=0.004)
        self.assertAlmostEqual(0.0, observed["center"][1], delta=0.004)
        self.assertAlmostEqual(0.20, observed["top_z"], places=5)

    def test_wrist_depth_rejects_background_outside_depth_gate(self):
        observed = wrist_depth_object_observation(
            np.full((80, 80), 0.40, dtype=np.float32),
            (100.0, 100.0, 40.0, 40.0),
            np.eye(4),
            (0.0, 0.0, 0.20),
            depth_gate_m=0.04,
        )

        self.assertIsNone(observed)

    def test_cloud_center_shift_is_bounded_around_stable_localization(self):
        centered = bounded_xy_center((0.40, -0.05), (0.38, 0.01), 0.015)

        self.assertAlmostEqual(
            0.015,
            float(np.linalg.norm(centered - np.array((0.40, -0.05)))),
            places=9,
        )
        self.assertGreater(centered[0], 0.38)
        self.assertLess(centered[1], 0.01)

    def test_tall_yolo_box_marks_upright_object(self):
        upright = SimpleNamespace(bbox_xyxy=(100.0, 50.0, 160.0, 180.0))
        lying = SimpleNamespace(bbox_xyxy=(100.0, 90.0, 240.0, 150.0))

        self.assertTrue(yolo_bbox_is_upright(upright))
        self.assertFalse(yolo_bbox_is_upright(lying))

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
        footprint = object_xy_footprint(_cloud())
        np.testing.assert_allclose(
            selected["xyz"][:2], footprint["center"], atol=1e-6
        )
        self.assertAlmostEqual(-0.34, selected["xyz"][2])

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

    def test_selector_prefers_narrow_fitting_axis_over_higher_score(self):
        rng = np.random.default_rng(11)
        cloud = np.column_stack(
            [
                rng.uniform(0.34, 0.46, 300),
                rng.uniform(-0.02, 0.02, 300),
                rng.uniform(-0.38, -0.27, 300),
            ]
        ).astype(np.float32)
        along_wide_axis = _topdown_pose((0.40, 0.0, -0.34), yaw=0.0)
        along_narrow_axis = _topdown_pose((0.40, 0.0, -0.34), yaw=np.pi / 2.0)

        selected = select_topdown_contact_pose(
            (
                np.stack([along_wide_axis, along_narrow_axis]),
                np.array([0.99, 0.50]),
            ),
            cloud,
            (0.40, 0.0, -0.40),
            max_grasp_width_m=0.12,
        )

        # The high-score candidate may provide the safe contact pose, but yaw
        # is finalized independently from the narrow footprint axis.
        self.assertEqual(0, selected["index"])
        self.assertLess(selected["width"], 0.05)
        self.assertAlmostEqual(
            np.pi / 2.0, abs(selected["yaw"]), delta=math.radians(3.0)
        )
        self.assertTrue(selected["directional"])

    def test_selector_executes_at_footprint_centre_not_an_object_end(self):
        rng = np.random.default_rng(17)
        cloud = np.column_stack(
            [
                rng.uniform(0.32, 0.48, 400),
                rng.uniform(-0.025, 0.025, 400),
                rng.uniform(-0.38, -0.27, 400),
            ]
        ).astype(np.float32)
        base_end = _topdown_pose((0.33, 0.0, -0.34), yaw=np.pi / 2.0)
        top_end = _topdown_pose((0.47, 0.0, -0.34), yaw=np.pi / 2.0)

        selected = select_topdown_contact_pose(
            (np.stack([base_end, top_end]), np.array([0.99, 0.95])),
            cloud,
            (0.40, 0.0, -0.40),
        )
        footprint = object_xy_footprint(cloud)

        np.testing.assert_allclose(
            selected["xyz"][:2], footprint["center"], atol=1e-6
        )
        self.assertGreater(selected["center_offset"], 0.05)

    def test_near_round_upright_footprint_uses_neutral_wrist_yaw(self):
        rng = np.random.default_rng(23)
        local = np.column_stack(
            [rng.uniform(-0.056, 0.056, 500), rng.uniform(-0.044, 0.044, 500)]
        )
        angle = 0.60
        c, s = np.cos(angle), np.sin(angle)
        xy = local @ np.array([[c, s], [-s, c]])
        cloud = np.column_stack(
            [xy[:, 0] + 0.40, xy[:, 1], rng.uniform(-0.44, -0.32, 500)]
        ).astype(np.float32)
        arbitrary_yaw = _topdown_pose((0.40, 0.0, -0.38), yaw=0.40)

        selected = select_topdown_contact_pose(
            (np.stack([arbitrary_yaw]), np.array([0.99])),
            cloud,
            (0.40, 0.0, -0.44),
        )

        self.assertFalse(selected["directional"])
        self.assertAlmostEqual(-np.pi / 2.0, selected["yaw"], places=6)

    def test_tall_bbox_can_force_neutral_yaw_for_skewed_upright_cloud(self):
        rng = np.random.default_rng(29)
        cloud = np.column_stack(
            [
                rng.uniform(0.35, 0.45, 500),
                rng.uniform(-0.018, 0.018, 500),
                rng.uniform(-0.43, -0.27, 500),
            ]
        ).astype(np.float32)
        arbitrary_yaw = _topdown_pose((0.40, 0.0, -0.35), yaw=0.0)

        selected = select_topdown_contact_pose(
            (np.stack([arbitrary_yaw]), np.array([0.99])),
            cloud,
            (0.40, 0.0, -0.43),
            force_neutral_yaw=True,
        )

        self.assertFalse(selected["directional"])
        self.assertAlmostEqual(-np.pi / 2.0, selected["yaw"], places=6)

    def test_contact_depth_scales_with_object_height(self):
        short = _cloud()
        short[:, 2] = np.linspace(-0.38, -0.26, len(short))
        tall = _cloud()
        tall[:, 2] = np.linspace(-0.46, -0.26, len(tall))

        short_z = object_contact_z(short, -0.38)
        tall_z = object_contact_z(tall, -0.46)

        self.assertGreater(short_z, tall_z)
        self.assertGreater(short_z, -0.38)
        self.assertGreater(tall_z, -0.46)

    def test_contact_respects_local_support_clearance(self):
        cloud = _cloud()
        support, top = object_vertical_extent(cloud, -0.40)

        contact = object_contact_z(cloud, -0.40)

        self.assertIsNotNone(contact)
        self.assertGreaterEqual(contact, support + DEFAULT_FLOOR_CLEARANCE_M)
        self.assertLessEqual(contact, top - 0.005)

    def test_contact_stays_above_the_raised_floor_collision_plane(self):
        cloud = _cloud()
        cloud[:, 2] = np.linspace(-0.40, -0.30, len(cloud))
        support, _top = object_vertical_extent(cloud, -0.40)

        contact = object_contact_z(cloud, -0.40)

        self.assertIsNotNone(contact)
        self.assertGreaterEqual(contact, support + 0.08)
        self.assertGreater(contact, support + 0.065)

    def test_contact_up_bias_is_optional_and_defaults_to_zero(self):
        cloud = _cloud()
        cloud[:, 2] = np.linspace(-0.50, -0.25, len(cloud))
        with patch.dict("os.environ", {}, clear=True):
            default_contact = object_contact_z(cloud, -0.50)
        with patch.dict(
            "os.environ", {"MARC_GRASPGENX_CONTACT_UP_BIAS_M": "0"}
        ):
            base_contact = object_contact_z(cloud, -0.50)
        with patch.dict(
            "os.environ", {"MARC_GRASPGENX_CONTACT_UP_BIAS_M": "0.03"}
        ):
            raised_contact = object_contact_z(cloud, -0.50)

        self.assertAlmostEqual(base_contact, default_contact, places=6)
        self.assertAlmostEqual(0.03, raised_contact - base_contact, places=6)

    def test_too_short_object_fails_closed_instead_of_entering_floor(self):
        cloud = _cloud()
        cloud[:, 2] = np.linspace(-0.30, -0.285, len(cloud))

        self.assertIsNone(object_contact_z(cloud, -0.30))

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
            return (
                np.stack([_topdown_pose((0.37, 0.03, -0.34), yaw=0.55)]),
                np.array([0.9]),
            )

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
        arm = _Arm(events)
        runtime._teleop = arm

        runtime._grasp(target)

        self.assertEqual("stage", events[0])
        self.assertEqual("infer", events[1])
        self.assertEqual("observe", events[2])
        self.assertEqual("ggx-center", events[3][0])
        self.assertEqual("ggx-pregrasp", events[4][0])
        self.assertEqual("ggx-contact", events[5][0])
        # The saved cloud supplies a bounded centering correction plus width/yaw.
        footprint = object_xy_footprint(_cloud())
        expected_xy = bounded_xy_center(
            (0.40, 0.0),
            footprint["center"],
            DEFAULT_MAX_CLOUD_CENTER_SHIFT_M,
        )
        np.testing.assert_allclose(expected_xy, events[5][1][:2], atol=1e-9)
        self.assertAlmostEqual(object_contact_z(_cloud(), -0.40), events[5][1][2])
        support, top = object_vertical_extent(_cloud(), -0.40)
        self.assertAlmostEqual(
            top + DEFAULT_STAGE_CLEARANCE_M,
            arm.stage_target.z + arm.cfg.openvla_stage_up_m,
        )
        np.testing.assert_allclose(
            arm.stage_target.fwd,
            expected_xy[0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            arm.stage_target.lat,
            expected_xy[1],
            atol=1e-9,
        )
        self.assertGreaterEqual(
            events[4][1][2], top + DEFAULT_OVERHEAD_CLEARANCE_M
        )
        self.assertAlmostEqual(top + DEFAULT_STAGE_CLEARANCE_M, events[3][1][2])
        np.testing.assert_allclose(events[3][1][:2], events[5][1][:2], atol=1e-9)
        self.assertGreaterEqual(events[5][1][2], support + DEFAULT_FLOOR_CLEARANCE_M)
        expected_yaw = footprint["yaw"]
        for event in events[3:6]:
            self.assertAlmostEqual(expected_yaw, event[2])
        self.assertEqual("ggx-contact-settle", events[6])
        self.assertEqual(("grip-immediate", 0.703), events[7])
        self.assertEqual("lift", events[8])
        self.assertEqual(9, len(events))
        self.assertTrue(runtime.poll_grasping())

    def test_confirmed_hold_skips_base_camera_reacquisition(self):
        target = SimpleNamespace(
            fwd=0.40,
            lat=0.0,
            z=-0.40,
            valid=True,
            name="tumbler",
        )
        observations = iter(
            (
                SimpleNamespace(fwd=0.40, lat=0.0, z=-0.40, valid=True),
                None,
            )
        )
        observer_calls = []

        def observe():
            observer_calls.append(True)
            return next(observations)

        client = _Client(holding=True)
        runtime = StagedGraspGenXPickRuntime(
            client,
            log=lambda _msg: None,
            policy="graspgenx",
            grasp_generator=lambda _points: (
                np.stack([_topdown_pose((0.40, 0.0, -0.34))]),
                np.array([0.9]),
            ),
            target_observer=observe,
            points_fn=lambda _target, client=None: _cloud(),
            grasp_poll=1,
        )
        runtime._teleop = _Arm([])

        runtime._grasp(target)

        self.assertEqual(1, len(observer_calls))
        self.assertTrue(runtime.poll_grasping())

    def test_visual_lift_verification_is_positive_when_hold_sensor_drops(self):
        runtime = StagedGraspGenXPickRuntime(
            _Client(holding=False),
            log=lambda _msg: None,
            policy="graspgenx",
            grasp_poll=1,
        )
        runtime._visual_grasp_verified = True

        self.assertTrue(runtime.poll_grasping())

    def test_visual_hold_evidence_is_consumed_by_basket_release(self):
        releases = []
        runtime = StagedGraspGenXPickRuntime(
            _Client(holding=False),
            log=lambda _msg: None,
            policy="graspgenx",
            place_fn=lambda: releases.append(True),
            grasp_poll=1,
        )
        runtime._visual_grasp_verified = True

        runtime._place()

        self.assertEqual([True], releases)
        self.assertFalse(runtime.poll_grasping())

    def test_graspgenx_transport_keeps_full_retention_close(self):
        runtime = StagedGraspGenXPickRuntime(
            _Client(holding=True),
            log=lambda _msg: None,
            policy="graspgenx",
            grasp_poll=1,
        )
        arm = _Arm([])
        arm.grip_cmd = 0.682
        runtime._teleop = arm

        runtime._seed_closed_grip()

        self.assertAlmostEqual(0.703, arm.grip_cmd)
        self.assertAlmostEqual(0.703, arm.grip_target)

    def test_large_post_stage_localization_jump_does_not_move_grasp_xy(self):
        events = []
        target = SimpleNamespace(
            fwd=0.40,
            lat=0.01,
            z=-0.40,
            valid=True,
            name="tumbler",
        )
        observations = iter(
            (SimpleNamespace(fwd=0.48, lat=0.01, z=-0.40, valid=True),)
        )
        runtime = StagedGraspGenXPickRuntime(
            _Client(holding=True),
            log=lambda _msg: None,
            policy="graspgenx",
            grasp_generator=lambda _points: (
                np.stack([_topdown_pose((0.40, 0.01, -0.34))]),
                np.array([0.9]),
            ),
            target_observer=lambda: next(observations),
            points_fn=lambda _target, client=None: _cloud(),
            grasp_poll=1,
        )
        runtime._teleop = _Arm(events)

        runtime._grasp(target)

        contact = next(event for event in events if event[0] == "ggx-contact")
        footprint = object_xy_footprint(_cloud())
        expected = bounded_xy_center(
            (0.40, 0.01), footprint["center"], DEFAULT_MAX_CLOUD_CENTER_SHIFT_M
        )
        np.testing.assert_allclose(contact[1][:2], expected, atol=1e-9)

    def test_fresh_localization_does_not_replace_bounded_saved_center(self):
        events = []
        target = SimpleNamespace(
            fwd=0.40,
            lat=-0.05,
            z=-0.40,
            valid=True,
            name="tumbler",
        )
        observations = iter(
            (SimpleNamespace(fwd=0.455, lat=-0.046, z=-0.40, valid=True),)
        )
        runtime = StagedGraspGenXPickRuntime(
            _Client(holding=True),
            log=lambda _msg: None,
            policy="graspgenx",
            grasp_generator=lambda _points: (
                np.stack([_topdown_pose((0.40, -0.05, -0.34))]),
                np.array([0.9]),
            ),
            target_observer=lambda: next(observations),
            points_fn=lambda _target, client=None: _cloud(),
            grasp_poll=1,
        )
        runtime._teleop = _Arm(events)

        runtime._grasp(target)

        contact = next(event for event in events if event[0] == "ggx-contact")
        footprint = object_xy_footprint(_cloud())
        expected = bounded_xy_center(
            (0.40, -0.05), footprint["center"], DEFAULT_MAX_CLOUD_CENTER_SHIFT_M
        )
        np.testing.assert_allclose(contact[1][:2], expected, atol=1e-9)

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
                _Client(holding=False),
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
