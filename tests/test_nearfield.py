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

from nearfield import gated_cmd_vel, localize_detection  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
