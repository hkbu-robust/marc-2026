"""Auto-approach overrun stops on arm-base fwd, not wall-clock cmd_vel distance."""

from __future__ import annotations

import os
import sys
import time
import unittest
from types import SimpleNamespace

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from manip_dataset.depth_approach import (  # noqa: E402
    ApproachConfig,
    BURST_SKIP_PAD_M,
    DepthApproachController,
    OVERRUN_MIN_S,
    OVERRUN_SLOW_MPS,
)


def _cfg(**kwargs):
    defaults = dict(
        crawl_mps=0.14,
        max_step_m=0.04,
        tick_s=0.05,
        already_near_m=0.16,
        stop_m=0.26,
        overrun_m=0.68,
        persist_k=3,
        log_every=0,
        burst_m=0.15,
        center_tol=0.05,
        realign=0.12,
        stale_stop_s=0.5,
    )
    defaults.update(kwargs)
    return ApproachConfig(**defaults)


class DepthApproachOverrunTests(unittest.TestCase):
    def _ctrl(self, measure=None, **kwargs):
        cmds = []
        client = SimpleNamespace(
            send_cmd_vel=lambda *a, **k: cmds.append((a, k)),
        )
        ctrl = DepthApproachController(client, _cfg(**kwargs))
        ctrl._cmds = cmds
        ctrl._frame_dt = 0.27
        ctrl.active = True
        ctrl._t0 = time.monotonic()
        if measure is not None:
            ctrl._measure = lambda: measure
        return ctrl

    def test_too_near_includes_already_near(self):
        ctrl = self._ctrl()
        m = dict(lat_n_obj=1, fwd=0.16, lat_err=0.0)
        self.assertTrue(ctrl._too_near(m))
        self.assertFalse(ctrl._too_near(dict(lat_n_obj=1, fwd=0.17, lat_err=0.0)))

    def test_overrun_cap_uses_slow_close_rate(self):
        ctrl = self._ctrl()
        remaining = 0.538 - 0.16
        cap = ctrl._overrun_cap_s(0.538)
        self.assertAlmostEqual(
            max(0.68 / ctrl._speed(), remaining / OVERRUN_SLOW_MPS, OVERRUN_MIN_S),
            cap,
            places=5,
        )
        self.assertGreater(cap, ctrl._overrun_duration_s())

    def test_overrun_stops_at_pick_range_before_cap(self):
        m = dict(lat_n_obj=1, fwd=0.15, d_obj=0.33, lat_err=0.02, is_new=True)
        ctrl = self._ctrl(measure=m)
        ctrl._overrun = True
        ctrl._overrun_t0 = time.monotonic()
        ctrl._overrun_need_s = 20.0
        self.assertTrue(ctrl.tick())
        self.assertFalse(ctrl.active)
        self.assertFalse(ctrl._overrun)

    def test_overrun_keeps_crawling_while_farther_than_pick_range(self):
        m = dict(lat_n_obj=1, fwd=0.31, d_obj=0.49, lat_err=0.02, is_new=True)
        ctrl = self._ctrl(measure=m)
        ctrl._overrun = True
        ctrl._overrun_t0 = time.monotonic()
        ctrl._overrun_need_s = 20.0
        self.assertFalse(ctrl.tick())
        self.assertTrue(ctrl.active)
        self.assertTrue(ctrl._overrun)
        self.assertTrue(any(
            k.get("linear_x", 0.0) > 0.0 for _a, k in ctrl._cmds if k
        ))

    def test_overrun_stops_at_cap_if_depth_stays_far(self):
        m = dict(lat_n_obj=1, fwd=0.31, d_obj=0.49, lat_err=0.02, is_new=True)
        ctrl = self._ctrl(measure=m)
        ctrl._overrun = True
        ctrl._overrun_t0 = time.monotonic() - 12.0
        ctrl._overrun_need_s = 11.0
        self.assertTrue(ctrl.tick())
        self.assertFalse(ctrl.active)

    def test_burst_does_not_pause_inside_stop_window(self):
        ctrl = self._ctrl()
        ctrl._phase = "drive"
        ctrl._phase_t0 = time.monotonic() - 10.0
        m = dict(lat_err=0.08, lat_n_obj=1, fwd=0.28, d_obj=0.46, is_new=True)
        lin, ang = ctrl._command(m)
        self.assertGreater(lin, 0.0)
        self.assertEqual(0.0, ang)
        self.assertEqual("drive", ctrl._phase)
        self.assertAlmostEqual(0.10, BURST_SKIP_PAD_M, places=6)
