"""Near-field handoff: close-enough disc gate, inward crawl, bbox on target."""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

from marc_agent.retrieval_state import close_enough_to_search, nearfield_start_reason

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from nearfield import (  # noqa: E402
    NearFieldAcquire,
    apply_lidar_gate,
    gated_cmd_vel,
    localize_detection,
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


if __name__ == "__main__":
    unittest.main()
