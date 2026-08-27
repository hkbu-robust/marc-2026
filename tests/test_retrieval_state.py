"""RetrievalState and search-area approach-goal fixtures."""

from __future__ import annotations

import math
import os
import unittest
from types import SimpleNamespace

import yaml

from marc_agent.retrieval_state import (
    RetrievalState,
    close_enough_to_search,
    nearfield_start_reason,
    search_area_approach_goal,
)


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "nav_runtime_cases.yaml")


def _load():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class RetrievalStateTests(unittest.TestCase):
    def test_stores_mission_and_reveal_fields(self):
        state = RetrievalState()
        state.adopt_mission(SimpleNamespace(
            task_description="bring the umbrella",
            time_limit=180.0,
            owner_position=[-55.2, 142.1, 16.4],
        ))
        self.assertEqual("bring the umbrella", state.task_description)
        self.assertEqual(180.0, state.time_limit)
        self.assertEqual((-55.2, 142.1), state.owner_xy)

        state.adopt_grounding([1.0, 2.0, 3.0])
        self.assertEqual((1.0, 2.0), state.grounding_xy)
        self.assertEqual((1.0, 2.0), state.search_xy())

        state.adopt_reveal(SimpleNamespace(
            hint_center=[10.0, 4.0],
            hint_radius=3.5,
            target_type="umbrella",
            grounding_score=0.82,
        ))
        self.assertEqual((10.0, 4.0), state.hint_center_xy)
        self.assertEqual(3.5, state.hint_radius_m)
        self.assertEqual("umbrella", state.target_type)
        self.assertEqual(0.82, state.grounding_score)
        self.assertEqual((10.0, 4.0), state.search_xy())

    def test_does_not_navigate_semantics_owner_is_stored_only(self):
        state = RetrievalState()
        state.adopt_owner([9.0, 8.0])
        state.adopt_reveal(SimpleNamespace(
            hint_center=[1.0, 2.0], hint_radius=2.0, target_type="mug", grounding_score=1.0,
        ))
        self.assertEqual((9.0, 8.0), state.owner_xy)
        self.assertNotEqual(state.search_xy(), state.owner_xy)

    def test_approach_fixture_cases(self):
        cases = _load()["approach_cases"]
        self.assertGreaterEqual(len(cases), 8)
        for case in cases:
            with self.subTest(case=case["id"]):
                got = search_area_approach_goal(
                    case["hint_center"], case["hint_radius"], case["pose"])
                exp = case["expected"]
                self.assertLess(
                    math.hypot(got[0] - exp[0], got[1] - exp[1]),
                    0.03,
                    msg="%s got %s expected %s" % (case["id"], got, exp),
                )

    def test_runtime_steps_count_and_values(self):
        steps = _load()["runtime_steps"]
        self.assertGreaterEqual(len(steps), 25)
        self.assertEqual(len(steps), len({step["id"] for step in steps}))
        for step in steps:
            with self.subTest(step=step["id"]):
                got = search_area_approach_goal(
                    step["hint_center"], step["hint_radius"], step["pose"])
                exp = step["expected"]
                self.assertLess(
                    math.hypot(got[0] - exp[0], got[1] - exp[1]),
                    0.05,
                    msg="%s got %s expected %s" % (step["id"], got, exp),
                )
                center = step["hint_center"]
                radius = float(step["hint_radius"])
                dist_goal = math.hypot(got[0] - center[0], got[1] - center[1])
                dist_pose = math.hypot(step["pose"][0] - center[0], step["pose"][1] - center[1])
                if dist_pose > radius + 1e-6:
                    self.assertAlmostEqual(dist_goal, radius, delta=0.05)
                else:
                    self.assertEqual(got, (float(step["pose"][0]), float(step["pose"][1])))

    def test_close_enough_half_metre_outside_three_metre_disc(self):
        self.assertTrue(close_enough_to_search((3.5, 0.0), (0.0, 0.0), 3.0, 0.7))
        self.assertEqual(
            "close_enough",
            nearfield_start_reason(False, (3.5, 0.0), (0.0, 0.0), 3.0, 0.7),
        )


if __name__ == "__main__":
    unittest.main()
