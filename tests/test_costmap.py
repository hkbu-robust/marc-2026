"""Costmap interface verification."""

from __future__ import annotations

import os
import unittest

import numpy as np
import yaml

from marc_agent.costmap import BLOCKED_COST, Costmap, CostmapConfig


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "nav_costmap_cases.yaml")


def _load_cases():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)["cases"]


class CostmapFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load_cases()

    def _build(self, case):
        cfg = CostmapConfig(
            inflation_m=float(case.get("inflation_m", 0.0)),
            unknown_is_blocked=bool(case.get("unknown_is_blocked", True)),
            margin_m=float(case.get("margin_m", 12.0)),
        )
        cm = Costmap(cfg)
        grid = np.array(case["grid"], dtype=np.int8)
        origin = tuple(case["origin"])
        cm.load_occupancy(grid, float(case["resolution"]), origin)
        if "set_overlay" in case:
            for row, col, cost in case["set_overlay"]:
                cm.set_overlay_cost(int(row), int(col), int(cost))
        if "overlay_disc" in case:
            d = case["overlay_disc"]
            cm.apply_overlay_disc(d["xy"][0], d["xy"][1], d["radius_m"], d["cost"])
        return cm

    def test_all_costmap_cases(self):
        for case in self.cases:
            with self.subTest(case=case["id"]):
                cm = self._build(case)
                for r, c in case.get("expect_blocked_parent", []):
                    self.assertGreaterEqual(cm.effective_cost(r, c), BLOCKED_COST)
                for r, c in case.get("expect_free_parent", []):
                    self.assertLess(cm.effective_cost(r, c), BLOCKED_COST)

                if "start" in case and "goal" in case:
                    view = cm.crop_window(
                        tuple(case["start"]),
                        tuple(case["goal"]),
                        margin_m=case.get("margin_m"),
                        inflation_m=case.get("inflation_m"),
                    )
                    for r, c in case.get("expect_free_local", []):
                        self.assertTrue(view.is_free(r, c), msg=(r, c))
                    for r, c in case.get("expect_local_free_false", []):
                        if view.in_bounds(r, c):
                            self.assertFalse(view.is_free(r, c), msg=(r, c))
                    for wx, wy in case.get("expect_crop_contains_world", []):
                        lr, lc = view.world_to_local(wx, wy)
                        self.assertTrue(view.in_bounds(lr, lc), msg=(wx, wy, lr, lc))

    def test_overlay_clear(self):
        cm = Costmap(CostmapConfig(inflation_m=0.0))
        cm.load_occupancy(np.zeros((5, 5), dtype=np.int8), 1.0, (0.0, 0.0))
        changed = cm.apply_overlay_disc(2.5, 2.5, 1.0, 100)
        self.assertTrue(changed)
        self.assertGreaterEqual(cm.effective_cost(2, 2), BLOCKED_COST)
        cleared = cm.clear_overlay()
        self.assertTrue(cleared)
        self.assertLess(cm.effective_cost(2, 2), BLOCKED_COST)

    def test_not_ready_crop_raises(self):
        cm = Costmap()
        with self.assertRaises(RuntimeError):
            cm.crop_window((0.0, 0.0), (1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
