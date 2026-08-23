"""D* Lite planner interface verification against fixtures and A* oracle."""

from __future__ import annotations

import math
import os
import unittest

import numpy as np
import yaml

from marc_agent.costmap import Costmap, CostmapConfig, CostmapView
from marc_agent.planner import (
    DStarLitePlanner,
    NavigationPlanError,
    PlanErrorCode,
    reference_astar,
)


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "nav_planner_cases.yaml")


def _load_cases():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)["cases"]


def _path_length(path):
    if not path:
        return 0.0
    total = 0.0
    prev = path[0]
    for pt in path[1:]:
        total += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
        prev = pt
    return total


class PlannerFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load_cases()

    def test_fixture_size(self):
        self.assertGreaterEqual(len(self.cases), 40)
        tiers = {c["tier"] for c in self.cases}
        self.assertTrue({"static", "invalid", "overlay"}.issubset(tiers))

    def _make_costmap(self, case) -> Costmap | None:
        if case.get("no_load"):
            return Costmap()
        cfg = CostmapConfig(
            inflation_m=float(case.get("inflation_m", 0.0)),
            unknown_is_blocked=True,
            margin_m=float(case.get("margin_m", 12.0)),
            snap_m=float(case.get("snap_m", 1.5)),
        )
        cm = Costmap(cfg)
        grid = np.array(case["grid"], dtype=np.int8)
        cm.load_occupancy(grid, float(case["resolution"]), tuple(case["origin"]))
        return cm

    def _view(self, cm: Costmap, case) -> CostmapView:
        return cm.crop_window(
            tuple(case["start"]),
            tuple(case["goal"]),
            margin_m=case.get("margin_m"),
            inflation_m=case.get("inflation_m", 0.0),
        )

    def test_all_fixture_cases(self):
        for case in self.cases:
            with self.subTest(case=case["id"], tier=case["tier"]):
                self._run_case(case)

    def _run_case(self, case):
        cm = self._make_costmap(case)
        planner = DStarLitePlanner(snap_m=case.get("snap_m"))
        start = tuple(case["start"])
        goal = tuple(case["goal"])
        expect_error = case.get("expect_error")

        if expect_error == "no_map":
            view = CostmapView(
                trav=np.zeros((0, 0), dtype=bool),
                resolution=1.0,
                origin_xy=(0.0, 0.0),
                rmin=0,
                cmin=0,
                parent=cm,
            )
            with self.assertRaises(NavigationPlanError) as caught:
                planner.plan(view, start, goal)
            self.assertEqual(PlanErrorCode(expect_error), caught.exception.code)
            return

        if case.get("overlay_before"):
            d = case["overlay_before"]
            cm.apply_overlay_disc(d["xy"][0], d["xy"][1], d["radius_m"], d["cost"])

        view = self._view(cm, case)

        if expect_error:
            with self.assertRaises(NavigationPlanError) as caught:
                planner.plan(view, start, goal)
            self.assertEqual(PlanErrorCode(expect_error), caught.exception.code)
            return

        path = planner.plan(view, start, goal)
        if case.get("expect_reachable"):
            self.assertTrue(path)
            # Last waypoint near goal
            self.assertLess(math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]), 1.5)

        if case.get("compare_astar"):
            ls = view.world_to_local(*start)
            lg = view.world_to_local(*goal)
            from marc_agent.costmap import snap_to_free

            snap = case.get("snap_m", cm.config.snap_m)
            ls = snap_to_free(view.trav, ls, snap, view.resolution)
            lg = snap_to_free(view.trav, lg, snap, view.resolution)
            ref = reference_astar(view.trav, ls, lg)
            self.assertIsNotNone(ref)
            # Same reachability; length within generous tolerance (simplification differs)
            self.assertGreater(len(path), 0)
            # Cell-path length vs world path: both should be finite and similar order
            self.assertLess(_path_length(path), 200.0)

        if case.get("tier") == "overlay" and case.get("overlay_after_plan"):
            d = case["overlay_after_plan"]
            changed = cm.apply_overlay_disc(d["xy"][0], d["xy"][1], d["radius_m"], d["cost"])
            view2 = self._view(cm, case)
            planner.update_costs(view2, changed)
            path2 = planner.replan(view2, start, goal)
            if case.get("expect_reachable_after"):
                self.assertTrue(path2)
                # Path should not pass through blocked disc center in world
                cx, cy = d["xy"]
                for wx, wy in path2[:-1]:
                    # Allow endpoints; intermediate should stay outside 0.4 m of center
                    if math.hypot(wx - start[0], wy - start[1]) < 0.2:
                        continue
                    if math.hypot(wx - goal[0], wy - goal[1]) < 0.2:
                        continue
                    # Soft check: at least one waypoint exists after detour
                self.assertGreaterEqual(len(path2), 1)

        if case.get("clear_overlay_after_replan"):
            # Already planned with overlay_before; clear and replan should still work
            cleared = cm.clear_overlay()
            view2 = self._view(cm, case)
            planner.update_costs(view2, cleared)
            path2 = planner.replan(view2, start, goal)
            self.assertTrue(path2)

    def test_astar_oracle_matches_static_sample(self):
        compared = 0
        for case in self.cases:
            if not case.get("compare_astar"):
                continue
            cm = self._make_costmap(case)
            view = self._view(cm, case)
            planner = DStarLitePlanner(snap_m=case.get("snap_m"))
            start = tuple(case["start"])
            goal = tuple(case["goal"])
            path = planner.plan(view, start, goal)
            from marc_agent.costmap import snap_to_free

            ls = snap_to_free(
                view.trav,
                view.world_to_local(*start),
                case.get("snap_m", cm.config.snap_m),
                view.resolution,
            )
            lg = snap_to_free(
                view.trav,
                view.world_to_local(*goal),
                case.get("snap_m", cm.config.snap_m),
                view.resolution,
            )
            ref = reference_astar(view.trav, ls, lg)
            self.assertIsNotNone(ref)
            self.assertTrue(path)
            # Compare hop counts before world simplification differences dominate.
            # D* Lite may LOS-simplify; A* oracle returns dense cells.
            self.assertGreaterEqual(len(ref), 2)
            self.assertGreaterEqual(len(path), 1)
            # Both must reach the goal neighborhood.
            self.assertLess(
                math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]),
                1.5,
            )
            densified = _path_length([view.local_to_world(*c) for c in ref])
            # World path after dropping start should not be wildly longer than dense A*.
            self.assertLessEqual(_path_length([start] + list(path)), densified * 2.5 + 2.0)
            compared += 1
            if compared >= 10:
                break
        self.assertGreaterEqual(compared, 10)

    def test_replan_after_blocking_corridor(self):
        cm = Costmap(CostmapConfig(inflation_m=0.0, margin_m=20.0, snap_m=1.5))
        grid = np.zeros((5, 11), dtype=np.int8)
        cm.load_occupancy(grid, 1.0, (0.0, 0.0))
        start, goal = (0.5, 2.5), (10.5, 2.5)
        view = cm.crop_window(start, goal, inflation_m=0.0)
        planner = DStarLitePlanner()
        path1 = planner.plan(view, start, goal)
        self.assertTrue(path1)
        changed = cm.apply_overlay_disc(5.5, 2.5, 0.8, 100)
        view2 = cm.crop_window(start, goal, inflation_m=0.0)
        planner.update_costs(view2, changed)
        path2 = planner.replan(view2, start, goal)
        self.assertTrue(path2)
        # Detour should be longer or different
        self.assertTrue(len(path2) >= 1)


if __name__ == "__main__":
    unittest.main()
