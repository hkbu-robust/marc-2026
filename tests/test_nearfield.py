"""Near-field handoff: close-enough disc gate, inward crawl, bbox on target."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from marc_agent.retrieval_state import close_enough_to_search, nearfield_start_reason, yolo_may_center

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from nearfield import (  # noqa: E402
    MissingDetector,
    NearFieldAcquire,
    PEEK_RANGE_PAD_M,
    apply_lidar_gate,
    choose_lock_depth,
    gated_cmd_vel,
    localize_detection,
    peek_yolo_handoff,
    pick_nearfield_candidate,
)


class NearFieldHandoffTests(unittest.TestCase):
    def test_close_enough_when_half_metre_outside_three_metre_disc(self):
        center = (0.0, 0.0)
        pose = (3.5, 0.0)
        self.assertTrue(close_enough_to_search(pose, center, 3.0, 0.7))
        self.assertEqual(
            "close_enough",
            nearfield_start_reason(False, pose, center, 3.0, 0.7),
        )
        self.assertFalse(close_enough_to_search((4.0, 0.0), center, 3.0, 0.7))
        self.assertEqual(
            "skip_too_far",
            nearfield_start_reason(False, (4.0, 0.0), center, 3.0, 0.7),
        )
        self.assertEqual(
            "nav_reached",
            nearfield_start_reason(True, (10.0, 0.0), center, 3.0, 0.7),
        )

    def test_far_box_stays_skip_and_observe(self):
        center = (0.0, 0.0)
        far = (15.0, 0.0)
        self.assertEqual(
            "skip_too_far",
            nearfield_start_reason(False, far, center, 3.0, 0.7),
        )
        self.assertFalse(yolo_may_center(True, far, center, 3.0, 15.0, 0.7))
        self.assertTrue(yolo_may_center(True, (3.5, 0.0), center, 3.0, 2.0, 0.7))
        self.assertTrue(yolo_may_center(True, far, center, 3.0, 2.0, 0.7))
        acquire = NearFieldAcquire(object(), detector=object())
        acquire._phase = "scan"
        acquire._phase_t0 = 0.0
        lin, ang = acquire._command(1.0, 1, 0.2, 15.0, may_center=False)
        self.assertEqual(0.0, lin)
        self.assertEqual(0.0, ang)

    def test_inward_crawl_allowed_when_slightly_outside_disc(self):
        pose = (3.5, 0.0, math.pi)
        lin, _ang = gated_cmd_vel(pose, 0.15, 0.0, (0.0, 0.0), 3.0)
        self.assertGreater(lin, 0.0)
        away = (3.5, 0.0, 0.0)
        lin_out, _ang = gated_cmd_vel(away, 0.15, 0.0, (0.0, 0.0), 3.0)
        self.assertEqual(0.0, lin_out)

    def test_localized_target_keeps_bbox_and_depth(self):
        depth = np.full((48, 64), 0.40, dtype=np.float32)
        bbox = (20.0, 10.0, 40.0, 30.0)
        k = (50.0, 50.0, 32.0, 24.0)
        pose = (0.0, 0.0, 0.0)
        target = localize_detection(
            bbox, depth, (48, 64), k, pose, (0.0, 0.0), 5.0,
            conf=0.8, name="mug",
        )
        self.assertEqual(bbox, target.bbox_xyxy)
        self.assertIsNotNone(target.depth)
        self.assertEqual(k, target.k)
        self.assertEqual((48, 64), target.rgb_shape)

    def test_single_in_radius_box_wins_beside_person(self):
        localized = type("L", (), {"world_xy": (0.1, 0.0)})()
        candidate = {"confidence": 0.3, "xyxy": (1, 1, 2, 2)}
        picked = pick_nearfield_candidate(
            [(1, candidate, localized, 0.4)],
            people_xy=((0.0, 0.0),),
        )
        self.assertIs(candidate, picked[1])

    def test_two_boxes_prefer_closer_depth(self):
        near = type("L", (), {"world_xy": (1.0, 0.0)})()
        far = type("L", (), {"world_xy": (3.0, 0.0)})()
        close_box = {
            "confidence": 0.2,
            "xyxy": (1, 1, 2, 2),
            "class_id": 0,
            "class_name": "tumbler",
        }
        far_box = {
            "confidence": 0.99,
            "xyxy": (3, 3, 4, 4),
            "class_id": 0,
            "class_name": "tumbler",
        }
        picked = pick_nearfield_candidate(
            [
                (1, far_box, far, 2.5),
                (2, close_box, near, 0.35),
            ],
            people_xy=(),
            hint_xy=(0.0, 0.0),
        )
        self.assertIs(close_box, picked[1])

    def test_drive_does_not_stop_on_a_distance_quota(self):
        acquire = NearFieldAcquire(object(), detector=object())
        acquire._phase = "drive"
        acquire._phase_t0 = 0.0
        lin, _ang = acquire._command(2.0, 1, 0.0, 1.0)
        self.assertGreater(lin, 0.0)
        acquire._phase = "drive"
        acquire._phase_t0 = 0.0
        lin, _ang = acquire._command(0.2, 1, 0.4, 1.0)
        self.assertEqual(0.0, lin)
        self.assertEqual("align", acquire._phase)

    def test_lidar_gate_zeros_forward_on_frontal_stop(self):
        from marc_agent.lidar_safety import advise

        stop = advise(np.array([[0.5, 0.0, 0.4]], dtype=np.float32))
        lin, _ang = apply_lidar_gate(0.15, 0.0, stop)
        self.assertEqual(0.0, lin)
        clear = advise(None)
        lin, _ang = apply_lidar_gate(0.15, 0.0, clear)
        self.assertGreater(lin, 0.0)


class _RgbMsg:
    def __init__(self, arr):
        h, w = arr.shape[:2]
        self.height = h
        self.width = w
        self.encoding = "rgb8"
        self.step = w * 3
        self.data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()


class _DepthMsg:
    def __init__(self, arr):
        h, w = arr.shape[:2]
        self.height = h
        self.width = w
        self.encoding = "32fc1"
        self.data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()


class _PeekClient:
    def __init__(self, rgb, depth):
        self._rgb = rgb
        self._depth = depth
        self.cmds = []

    def get_robot_image(self, *_a, **_k):
        return self._rgb

    def get_robot_depth(self, *_a, **_k):
        return self._depth

    def get_world_pose(self):
        return (0.0, 0.0, 0.0)

    def send_cmd_vel(self, linear_x=0.0, angular_z=0.0, **_k):
        self.cmds.append((float(linear_x), float(angular_z)))

    def sleep(self, _dt):
        return None


class _PeekDetector:
    def __init__(self, candidates, boom=False):
        self.candidates = candidates
        self.boom = boom
        self.calls = 0

    def load(self):
        return True

    def set_target(self, _ttype):
        return None

    def detect_candidates(self, rgb, _ttype):
        self.calls += 1
        self.last_rgb = rgb
        if self.boom:
            raise AssertionError("detect must not run when pose is out of peek range")
        return {
            "status": "ok",
            "candidates": self.candidates,
            "rgb_shape": rgb.shape[:2],
        }


class PeekYoloHandoffTests(unittest.TestCase):
    def test_skips_when_pose_is_beyond_peek_pad(self):
        det = _PeekDetector([], boom=True)
        rgb = _RgbMsg(np.zeros((48, 64, 3), dtype=np.uint8))
        depth = _DepthMsg(np.full((48, 64), 1.5, dtype=np.float32))
        client = _PeekClient(rgb, depth)
        far = (0.0 + 3.0 + PEEK_RANGE_PAD_M + 1.0, 0.0, 0.0)
        self.assertFalse(
            peek_yolo_handoff(client, det, far, (0.0, 0.0), 3.0, "tumbler")
        )
        self.assertEqual(0, det.calls)

    def test_hands_off_when_bbox_depth_is_in_range(self):
        box = {"xyxy": [20, 10, 40, 30], "confidence": 0.9, "class_name": "tumbler"}
        det = _PeekDetector([box])
        rgb = _RgbMsg(np.zeros((48, 64, 3), dtype=np.uint8))
        depth = _DepthMsg(np.full((48, 64), 1.5, dtype=np.float32))
        client = _PeekClient(rgb, depth)
        pose = (5.0, 0.0, 0.0)
        self.assertTrue(
            peek_yolo_handoff(client, det, pose, (0.0, 0.0), 3.0, "tumbler")
        )
        self.assertEqual(1, det.calls)

    def test_existing_skip_too_far_and_yolo_may_center_fixtures_stay(self):
        center = (0.0, 0.0)
        far = (15.0, 0.0)
        self.assertEqual(
            "skip_too_far",
            nearfield_start_reason(False, far, center, 3.0, 0.7),
        )
        self.assertFalse(yolo_may_center(True, far, center, 3.0, 15.0, 0.7))
        self.assertTrue(yolo_may_center(True, (3.5, 0.0), center, 3.0, 2.0, 0.7))


class NearFieldSnapshotTests(unittest.TestCase):
    def _client(self, depth_m=0.40, rgb=None, depth=None):
        if rgb is None:
            rgb = _RgbMsg(np.zeros((48, 64, 3), dtype=np.uint8))
        if depth is None:
            depth = _DepthMsg(np.full((48, 64), float(depth_m), dtype=np.float32))
        return _PeekClient(rgb, depth)

    def test_snapshot_valid_without_reveal_disc(self):
        box = {
            "xyxy": [20, 10, 40, 30],
            "confidence": 0.9,
            "class_name": "tumbler",
            "class_id": 0,
        }
        det = _PeekDetector([box])
        client = self._client(0.40)
        target = NearFieldAcquire(client, detector=det).snapshot("tumbler")
        self.assertTrue(target.valid)
        self.assertEqual("tumbler", target.name)
        self.assertEqual(1, det.calls)
        self.assertTrue(all(lin == 0.0 for lin, _ang in client.cmds))

    def test_snapshot_pairs_rgb_with_the_selected_depth_frame(self):
        correct = _RgbMsg(np.full((48, 64, 3), 255, dtype=np.uint8))
        wrong = _RgbMsg(np.zeros((48, 64, 3), dtype=np.uint8))
        depth = _DepthMsg(np.full((48, 64), 0.40, dtype=np.float32))
        correct.header = type("Header", (), {"frame_id": "chassis"})()
        wrong.header = type("Header", (), {"frame_id": "closeup"})()
        depth.header = type("Header", (), {"frame_id": "chassis"})()
        client = _PeekClient(wrong, depth)
        client.get_robot_depth_hw = lambda _which, _shape: depth
        client.get_robot_image_frame = lambda _which, frame: (
            correct if frame == "chassis" else None
        )
        det = _PeekDetector([])

        NearFieldAcquire(client, detector=det).snapshot("tumbler")

        self.assertEqual(1, det.calls)
        self.assertEqual(255.0, float(det.last_rgb.mean()))

    def test_snapshot_fail_closed_without_box(self):
        det = _PeekDetector([])
        target = NearFieldAcquire(self._client(), detector=det).snapshot("tumbler")
        self.assertFalse(target.valid)
        self.assertEqual("not_detected", target.reason)

    def test_snapshot_fail_closed_without_depth(self):
        box = {
            "xyxy": [20, 10, 40, 30],
            "confidence": 0.9,
            "class_name": "tumbler",
            "class_id": 0,
        }
        det = _PeekDetector([box])
        client = self._client()
        client._depth = None
        target = NearFieldAcquire(client, detector=det).snapshot("tumbler")
        self.assertFalse(target.valid)
        self.assertEqual("no_depth", target.reason)
        self.assertEqual(0, det.calls)

    def test_snapshot_fail_closed_when_detector_missing(self):
        target = NearFieldAcquire(
            self._client(), detector=MissingDetector("no_weights"),
        ).snapshot("tumbler")
        self.assertFalse(target.valid)
        self.assertEqual("no_weights", target.reason)

    def test_snapshot_out_of_reach_is_invalid(self):
        box = {
            "xyxy": [20, 10, 40, 30],
            "confidence": 0.9,
            "class_name": "tumbler",
            "class_id": 0,
        }
        det = _PeekDetector([box])
        target = NearFieldAcquire(self._client(5.0), detector=det).snapshot("tumbler")
        self.assertFalse(target.valid)
        self.assertEqual("out_of_reach", target.reason)

    def test_choose_lock_depth_prefers_chassis_360(self):
        rgb = (480, 640)
        self.assertEqual(
            (360, 640),
            choose_lock_depth([(480, 640), (360, 640), (480, 640)], rgb),
        )
        self.assertIsNone(choose_lock_depth([(480, 640)], rgb))
        self.assertIsNone(choose_lock_depth([], rgb))


if __name__ == "__main__":
    unittest.main()
