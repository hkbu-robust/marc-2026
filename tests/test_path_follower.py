"""Path follower interface verification."""

from __future__ import annotations

import math
import unittest

from marc_agent.path_follower import GoalSeekController, PathFollower


class GoalSeekTests(unittest.TestCase):
    def test_reached_when_close(self):
        ctrl = GoalSeekController()
        lin, ang, reached = ctrl.compute((0.0, 0.0, 0.0), (0.1, 0.0))
        self.assertTrue(reached)
        self.assertEqual(lin, 0.0)
        self.assertEqual(ang, 0.0)

    def test_rotate_in_place_for_large_heading(self):
        ctrl = GoalSeekController()
        # Goal behind the robot
        lin, ang, reached = ctrl.compute((0.0, 0.0, 0.0), (-5.0, 0.0))
        self.assertFalse(reached)
        self.assertEqual(lin, 0.0)
        self.assertNotEqual(ang, 0.0)

    def test_cruise_uses_max_linear(self):
        ctrl = GoalSeekController()
        lin, _, reached = ctrl.compute((0.0, 0.0, 0.0), (5.0, 0.0), cruise=True)
        self.assertFalse(reached)
        self.assertEqual(lin, ctrl.MAX_LINEAR)

    def test_decelerate_near_goal(self):
        ctrl = GoalSeekController()
        lin, _, reached = ctrl.compute((0.0, 0.0, 0.0), (1.0, 0.0), cruise=False)
        self.assertFalse(reached)
        self.assertLess(lin, ctrl.MAX_LINEAR)


class PathFollowerTests(unittest.TestCase):
    def test_empty_path_done(self):
        pf = PathFollower([])
        lin, ang, done, info = pf.compute((0.0, 0.0, 0.0))
        self.assertTrue(done)
        self.assertEqual(info["total"], 0)

    def test_single_waypoint_reaches(self):
        pf = PathFollower([(0.1, 0.0)])
        _, _, done, info = pf.compute((0.0, 0.0, 0.0))
        self.assertTrue(done)
        self.assertEqual(info["total"], 1)

    def test_advances_intermediate_waypoints(self):
        path = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0)]
        pf = PathFollower(path)
        # Near first waypoint -> advance
        lin, ang, done, info = pf.compute((0.0, 0.0, 0.0))
        self.assertFalse(done)
        self.assertGreaterEqual(info["idx"], 1)

    def test_passthrough_cruises_final(self):
        pf = PathFollower([(5.0, 0.0)], passthrough=True)
        lin, _, done, _ = pf.compute((0.0, 0.0, 0.0))
        self.assertFalse(done)
        self.assertEqual(lin, GoalSeekController.MAX_LINEAR)

    def test_straight_segment_positive_linear(self):
        pf = PathFollower([(3.0, 0.0), (6.0, 0.0)])
        lin, ang, done, info = pf.compute((0.0, 0.0, 0.0))
        self.assertFalse(done)
        self.assertGreater(lin, 0.0)
        self.assertLess(abs(ang), math.pi)


if __name__ == "__main__":
    unittest.main()
