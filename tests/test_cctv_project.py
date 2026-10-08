"""Tests for CCTV contact-pixel projection onto a horizontal plane."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from marc_agent.command_parser import Relation
from marc_agent.perception.cctv_project import (
    CameraGeometry,
    CctvProjector,
    Tf2ClientLookup,
    _keep_cctv_transform,
    _start_tf_listener,
    contact_pixel,
    k_from_camera_info,
    directional_likelihood,
    pair_alignment_score,
    project_pixel,
    raised_plane_z,
    static_lookup,
)


IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _looking_along_z(z: float = -10.0, ground: float = 0.0) -> CameraGeometry:
    return CameraGeometry(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        rotation=IDENTITY,
        translation=(0.0, 0.0, z),
        ground_height=ground,
    )


class CctvProjectTests(unittest.TestCase):
    def test_image_center_projects_to_camera_xy_zero(self):
        lookup = static_lookup({"cam": _looking_along_z()})
        xyz = project_pixel("cam", 320.0, 240.0, lookup=lookup)
        self.assertIsNotNone(xyz)
        self.assertAlmostEqual(0.0, xyz[0], places=6)
        self.assertAlmostEqual(0.0, xyz[1], places=6)
        self.assertAlmostEqual(0.0, xyz[2], places=6)

    def test_offset_pixel_scales_with_focal_length(self):
        lookup = static_lookup({"cam": _looking_along_z()})
        xyz = project_pixel("cam", 820.0, 240.0, lookup=lookup)
        self.assertAlmostEqual(10.0, xyz[0], places=6)
        self.assertAlmostEqual(0.0, xyz[1], places=6)
        self.assertAlmostEqual(0.0, xyz[2], places=6)

    def test_contact_pixel_is_lower_half_centroid(self):
        self.assertEqual((320.0, 325.0), contact_pixel((280.0, 100.0, 360.0, 400.0)))

    def test_prostrate_contact_samples_higher_than_upright(self):
        box = (100.0, 100.0, 500.0, 400.0)
        upright = contact_pixel(box, pose_class="person_walking")
        prone = contact_pixel(box, pose_class="person_prone")
        self.assertAlmostEqual(300.0, upright[0])
        self.assertAlmostEqual(325.0, upright[1])
        self.assertAlmostEqual(300.0, prone[0])
        self.assertLess(prone[1], upright[1])
        self.assertAlmostEqual(280.0, prone[1])

    def test_bending_contact_samples_near_the_feet(self):
        self.assertEqual(
            (320.0, 415.0),
            contact_pixel((280.0, 100.0, 360.0, 400.0), "person_bending"),
        )

    def test_ground_lying_uses_feet_but_supported_lying_uses_torso(self):
        projector = CctvProjector(static_lookup({"cam": _looking_along_z()}))
        box = (280.0, 100.0, 360.0, 400.0)
        with patch(
            "marc_agent.perception.cctv_project.intersect_ray_plane",
            return_value=(0.0, 0.0, 0.0),
        ) as intersect:
            projector.project_contact("cam", box, pose_class="person_lying_down")
            self.assertEqual(385.0, intersect.call_args.args[2])
            projector.project_contact(
                "cam",
                box,
                pose_class="person_lying_down",
                relation=Relation.ON,
                support_class="bench",
            )
            self.assertEqual(280.0, intersect.call_args.args[2])

    def test_ground_bending_contact_pulls_slightly_toward_named_anchor(self):
        projector = CctvProjector(static_lookup({"cam": _looking_along_z()}))
        box = (280.0, 100.0, 360.0, 400.0)
        raw = projector.project_contact("cam", box, pose_class="person_bending")
        pulled = projector.project_contact(
            "cam",
            box,
            pose_class="person_bending",
            relation=Relation.NEAR,
            support_class="trash_can",
            support_box=(40.0, 100.0, 120.0, 400.0),
        )
        self.assertLess(pulled[0], raw[0])
        self.assertAlmostEqual(0.18, pulled[2])

    def test_ray_behind_camera_is_rejected(self):
        lookup = static_lookup({"cam": _looking_along_z(z=10.0, ground=0.0)})
        self.assertIsNone(project_pixel("cam", 320.0, 240.0, lookup=lookup))

    def test_on_uses_raised_plane_when_height_known(self):
        geometry = _looking_along_z()
        projector = CctvProjector(static_lookup({"cam": geometry}))
        xyz = projector.project_contact(
            "cam",
            (300.0, 210.0, 340.0, 250.0),
            relation=Relation.ON,
            support_class="picnic_table",
        )
        self.assertAlmostEqual(0.0, xyz[0], places=6)
        self.assertAlmostEqual(0.0, xyz[1], places=6)
        self.assertAlmostEqual(raised_plane_z(0.0, "picnic_table", Relation.ON), xyz[2], places=6)

    def test_on_alignment_prefers_contact_inside_support_footprint(self):
        projector = CctvProjector(static_lookup({"cam": _looking_along_z()}))
        inside = pair_alignment_score(
            projector,
            "cam",
            (300.0, 200.0, 340.0, 250.0),
            (200.0, 240.0, 440.0, 400.0),
            Relation.ON,
            "bench",
        )
        outside = pair_alignment_score(
            projector,
            "cam",
            (20.0, 200.0, 60.0, 250.0),
            (200.0, 240.0, 440.0, 400.0),
            Relation.ON,
            "bench",
        )
        self.assertIsNotNone(inside)
        self.assertIsNotNone(outside)
        self.assertGreater(inside, outside)

    def test_near_alignment_prefers_smaller_ground_distance(self):
        projector = CctvProjector(static_lookup({"cam": _looking_along_z()}))
        close = pair_alignment_score(
            projector,
            "cam",
            (300.0, 220.0, 340.0, 260.0),
            (280.0, 240.0, 360.0, 400.0),
            Relation.NEAR,
            "hydrant",
        )
        far = pair_alignment_score(
            projector,
            "cam",
            (20.0, 220.0, 60.0, 260.0),
            (280.0, 240.0, 360.0, 400.0),
            Relation.NEAR,
            "hydrant",
        )
        self.assertGreater(close, far)

    def test_directional_likelihood_prefers_the_near_instance(self):
        geometry = _looking_along_z()
        anchor = (0.0, 0.0, 5.0)
        near = directional_likelihood((1.0, 0.0, 5.0), anchor, geometry, Relation.RIGHT)
        far = directional_likelihood((4.0, 0.0, 5.0), anchor, geometry, Relation.RIGHT)
        self.assertGreater(near, far)
        self.assertIsNone(directional_likelihood((-1.0, 0.0, 5.0), anchor, geometry, Relation.RIGHT))

    def test_on_submit_snaps_xy_into_support_footprint(self):
        projector = CctvProjector(static_lookup({"cam": _looking_along_z()}))
        support = (200.0, 240.0, 440.0, 400.0)
        outside = (20.0, 200.0, 60.0, 250.0)
        raw = projector.project_contact(
            "cam", outside, relation=Relation.ON, support_class="bench"
        )
        snapped = projector.project_contact(
            "cam",
            outside,
            relation=Relation.ON,
            support_class="bench",
            support_box=support,
        )
        self.assertIsNotNone(raw)
        self.assertIsNotNone(snapped)
        self.assertNotAlmostEqual(raw[0], snapped[0], places=3)
        inside = projector.project_contact(
            "cam",
            (300.0, 200.0, 340.0, 250.0),
            relation=Relation.ON,
            support_class="bench",
            support_box=support,
        )
        xmin = min(inside[0], snapped[0])
        xmax = max(inside[0], snapped[0])
        self.assertGreaterEqual(snapped[0], xmin - 1e-6)
        self.assertLessEqual(snapped[0], xmax + 1e-6)

    def test_near_does_not_snap_into_support(self):
        projector = CctvProjector(static_lookup({"cam": _looking_along_z()}))
        support = (200.0, 240.0, 440.0, 400.0)
        outside = (20.0, 220.0, 60.0, 260.0)
        raw = projector.project_contact(
            "cam", outside, relation=Relation.NEAR, support_class="hydrant"
        )
        with_box = projector.project_contact(
            "cam",
            outside,
            relation=Relation.NEAR,
            support_class="hydrant",
            support_box=support,
        )
        self.assertEqual(raw, with_box)

    def test_unknown_support_falls_back_to_ground(self):
        self.assertEqual(16.5, raised_plane_z(16.5, "mystery_prop", Relation.ON))
        self.assertEqual(16.5, raised_plane_z(16.5, "picnic_table", Relation.NEAR))

    def test_k_from_camera_info_accepts_numpy_array(self):
        class Info:
            k = np.array(
                [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
                dtype=np.float64,
            )

        self.assertEqual((500.0, 500.0, 320.0, 240.0), k_from_camera_info(Info()))

    def test_tf2_lookup_returns_geometry_for_numpy_k_and_zero_ground(self):
        class Info:
            k = np.array(
                [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
                dtype=np.float64,
            )

        class Client:
            def __init__(self):
                self.info_calls = 0

            def get_cctv_info(self, camera_id):
                self.info_calls += 1
                return Info()

            def get_cctv_ground_height(self, camera_id):
                return 0.0

        client = Client()
        lookup = Tf2ClientLookup(client)
        lookup._ensure_tf = lambda: None
        lookup._lookup_tf = lambda _camera_id: (
            (0.0, 0.0, -10.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        geometry = lookup("rig_1_a")
        self.assertIsNotNone(geometry)
        self.assertEqual(500.0, geometry.fx)
        self.assertEqual(500.0, geometry.fy)
        self.assertEqual(320.0, geometry.cx)
        self.assertEqual(240.0, geometry.cy)
        self.assertEqual(0.0, geometry.ground_height)
        self.assertEqual(geometry, lookup("rig_1_a"))
        self.assertEqual(1, client.info_calls)

    def test_from_client_reuses_the_same_projector(self):
        class Client:
            pass

        client = Client()
        first = CctvProjector.from_client(client)
        second = CctvProjector.from_client(client)
        self.assertIs(first, second)
        self.assertIs(first.lookup, second.lookup)

    def test_ensure_tf_attaches_once_on_the_client(self):
        buffer, listener = object(), object()

        class Client:
            def subscribe(self, *args, **kwargs):
                return None

        client = Client()
        lookup = Tf2ClientLookup(client)
        with patch(
            "marc_agent.perception.cctv_project._start_tf_listener",
            return_value=(buffer, listener),
        ) as start:
            self.assertIs(buffer, lookup._ensure_tf())
            start.assert_called_once_with(client)
            self.assertIs(buffer, lookup._ensure_tf())
            start.assert_called_once()

    def test_cctv_tf_keeps_rig_frames_and_drops_robot_odom(self):
        class Transform:
            def __init__(self, parent, child):
                self.header = type("Header", (), {"frame_id": parent})()
                self.child_frame_id = child

        self.assertTrue(_keep_cctv_transform(Transform("world", "rig_1_a")))
        self.assertTrue(_keep_cctv_transform(Transform("rig_1", "rig_1_a")))
        self.assertTrue(_keep_cctv_transform(Transform("world", "cam_7")))
        self.assertFalse(_keep_cctv_transform(Transform("odom", "car_base_link")))
        # live /tf is robot traffic: only rig_ frames, as before
        self.assertTrue(_keep_cctv_transform(Transform("world", "rig_1_a"), static=False))
        self.assertFalse(_keep_cctv_transform(Transform("world", "cam_7"), static=False))
        self.assertFalse(_keep_cctv_transform(Transform("world", "Base_LiDAR"), static=False))

    def test_start_tf_listener_subscribes_static_and_live_without_transform_listener(self):
        captured = []

        class Client:
            def subscribe(self, topic, msg_type, callback, qos=None):
                captured.append(topic)
                return topic

        tf2_ros = types.ModuleType("tf2_ros")
        buffer_mod = types.ModuleType("tf2_ros.buffer")
        msgs = types.ModuleType("tf2_msgs")
        msg_mod = types.ModuleType("tf2_msgs.msg")
        sdk = types.ModuleType("marc_sdk")
        protocol = types.ModuleType("marc_sdk.protocol")
        protocol.QOS_TRANSIENT = "transient"
        protocol.QOS_RELIABLE = "reliable"
        sdk.protocol = protocol

        class FakeBuffer:
            pass

        class FakeTFMessage:
            pass

        buffer_mod.Buffer = FakeBuffer
        msg_mod.TFMessage = FakeTFMessage
        msgs.msg = msg_mod
        with patch.dict(
            sys.modules,
            {
                "tf2_ros": tf2_ros,
                "tf2_ros.buffer": buffer_mod,
                "tf2_msgs": msgs,
                "tf2_msgs.msg": msg_mod,
                "marc_sdk": sdk,
                "marc_sdk.protocol": protocol,
            },
        ):
            buffer, subs = _start_tf_listener(Client())
        self.assertIsInstance(buffer, FakeBuffer)
        self.assertEqual(["/tf_static", "/tf"], captured)
        self.assertEqual(("/tf_static", "/tf"), subs)


if __name__ == "__main__":
    unittest.main()
