"""Scripted-pick gripper: stop on is_grasping, latch carry, no Isaac."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from arm_pick import close_until_holding, latch_closed_grip  # noqa: E402


CEILING = 0.68
EXTRA = 0.02
THRESHOLD = 0.30


def _run_close(*, stop_on_grasp, holding_fn, hold_s=0.2):
    published = []

    def publish(grip):
        published.append(float(grip))

    latched = close_until_holding(
        start_grip=0.0,
        ceiling=CEILING,
        extra=EXTRA,
        ramp_s=0.5,
        hold_s=hold_s,
        rate_hz=10.0,
        is_grasping=lambda: holding_fn(published),
        publish=publish,
        sleep=lambda _dt: None,
        stop_on_grasp=stop_on_grasp,
    )
    return latched, published


class ArmPickGripTests(unittest.TestCase):
    def test_holding_mid_ramp_returns_cmd_plus_extra(self):
        def holding(published):
            return bool(published) and published[-1] >= THRESHOLD

        latched, published = _run_close(stop_on_grasp=True, holding_fn=holding)
        self.assertAlmostEqual(latched, published[-1])
        self.assertGreater(latched, THRESHOLD)
        self.assertLess(max(published), CEILING)
        first_latched = next(
            i for i, g in enumerate(published) if abs(g - latched) < 1e-9
        )
        self.assertTrue(
            all(abs(g - latched) < 1e-9 for g in published[first_latched:])
        )
        self.assertGreater(len(published) - first_latched, 1)

    def test_never_holding_ends_at_ceiling(self):
        latched, published = _run_close(
            stop_on_grasp=True, holding_fn=lambda _p: False
        )
        self.assertAlmostEqual(latched, CEILING)
        self.assertAlmostEqual(max(published), CEILING)

    def test_stop_off_publishes_ceiling_despite_hold(self):
        def always_hold(published):
            return bool(published)

        latched, published = _run_close(
            stop_on_grasp=False, holding_fn=always_hold
        )
        self.assertAlmostEqual(latched, CEILING)
        self.assertAlmostEqual(max(published), CEILING)
        self.assertGreaterEqual(len(published), 5)

    def test_latch_closed_grip_skips_release(self):
        seq = SimpleNamespace(
            _grip=0.58,
            _idx=2,
            _waypoints=[
                ("pre", [0.0, 0.0, 0.0], 0.0, 0.3, "cart"),
                ("descend", [0.0, 0.0, 0.0], 0.0, 0.2, "cart"),
                ("grasp", [0.4, 0.0, -0.3], 0.58, 1.0, "cart"),
                ("lift", [0.4, 0.0, 0.25], 0.58, 0.3, "cart"),
                ("over", [-0.29, 0.0, 0.25], 0.58, 0.3, "joint"),
                ("place", [-0.29, 0.0, 0.05], 0.58, 0.3, "cart"),
                ("release", [-0.29, 0.0, 0.05], 0.0, 1.0, "cart"),
            ],
        )
        latch_closed_grip(seq, 0.42)
        self.assertAlmostEqual(seq._grip, 0.42)
        by_label = {wp[0]: wp[2] for wp in seq._waypoints}
        self.assertAlmostEqual(by_label["pre"], 0.0)
        self.assertAlmostEqual(by_label["descend"], 0.0)
        self.assertAlmostEqual(by_label["grasp"], 0.42)
        self.assertAlmostEqual(by_label["lift"], 0.42)
        self.assertAlmostEqual(by_label["over"], 0.42)
        self.assertAlmostEqual(by_label["place"], 0.42)
        self.assertAlmostEqual(by_label["release"], 0.0)


if __name__ == "__main__":
    unittest.main()
