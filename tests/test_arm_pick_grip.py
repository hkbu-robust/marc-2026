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

from arm_pick import close_until_holding, descend_should_stop, latch_closed_grip  # noqa: E402


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

    def test_descend_should_stop_only_on_hold_during_descend(self):
        self.assertTrue(descend_should_stop("descend", "move", lambda: True))
        self.assertFalse(descend_should_stop("descend", "move", lambda: False))
        self.assertFalse(descend_should_stop("descend", "dwell", lambda: True))
        self.assertFalse(descend_should_stop("grasp", "move", lambda: True))
        self.assertFalse(descend_should_stop("descend", "move", None))

    def test_place_in_basket_is_scoop_not_tuck(self):
        path = os.path.join(_ROOT, "demo", "manip_dataset", "arm_control.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertIn('n += self._joint_move("rotate over basket", q_over, on_step, speed)', src)
        self.assertIn("q_over[0] +=", src)
        self.assertNotIn("tuck over shoulder", src)
        self.assertNotIn("PLACE_TUCK_Q", src)
        self.assertNotIn(
            'n += self._joint_move("fold to ready", q_home, on_step, speed, wrap=True)',
            src,
        )
        hold = src.split("def place_in_basket", 1)[1]
        self.assertLess(
            hold.find('_grip_segment("hold"'),
            hold.find('n += self._joint_move("raise to carry height"'),
        )

    def test_seed_closed_grip_keeps_live_squeeze(self):
        from arm_pick import GRIPPER_JOINT_NAMES
        from marc_agent.pick_runtime import PickRuntime

        client = SimpleNamespace(
            get_arm_state=lambda: SimpleNamespace(
                name=list(GRIPPER_JOINT_NAMES),
                position=[0.332, 0.332],
            )
        )
        runtime = PickRuntime(client, log=lambda _m: None)
        arm = SimpleNamespace(
            grip_cmd=0.0,
            grip_target=0.0,
            cfg=SimpleNamespace(grip_close=0.68),
            _saw_grasp_after_close=False,
            _closed_since_last_open=False,
        )
        runtime._teleop = arm
        runtime._seed_closed_grip()
        self.assertAlmostEqual(0.332, arm.grip_cmd, places=3)
        self.assertAlmostEqual(0.68, arm.grip_target, places=3)
        self.assertGreaterEqual(arm.grip_target, 0.5 * arm.cfg.grip_close)

    def test_pick_skips_settle_end_while_holding(self):
        path = os.path.join(_ROOT, "demo", "arm_kin.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertIn("holding at last pose", src)
        self.assertIn("if self._grip > 1e-3:", src)

    def test_wrist_skip_logs_reason(self):
        from arm_pick import refine_grasp_z_from_wrist
        import numpy as np

        logs = []
        z = refine_grasp_z_from_wrist(
            SimpleNamespace(), None, np.zeros(6), {}, logs.append
        )
        self.assertIsNone(z)
        self.assertTrue(any("wrist z skipped" in line for line in logs))

    def test_scoop_place_stays_in_joint_limits(self):
        try:
            import placo  # noqa: F401
            from arm_pick import load_params
            from manip_dataset.arm_control import (
                ARM_JOINT_LOWER, ARM_JOINT_UPPER, ArmTeleop, TeleopConfig,
            )
        except Exception as exc:
            self.skipTest("placo/arm teleop unavailable: %s" % exc)
        arm = ArmTeleop(SimpleNamespace(), TeleopConfig(), log=lambda *_a, **_k: None)
        arm._place_params = load_params()
        poses = arm._place_poses()
        if poses is None:
            self.fail("scoop basket IK rejected; live place would keep holding")
        q_lift, q_over, q_drop = poses
        import numpy as np
        for q in poses:
            q = np.asarray(q, dtype=float)
            self.assertTrue(np.all(q >= ARM_JOINT_LOWER - 1e-6), q)
            self.assertTrue(np.all(q <= ARM_JOINT_UPPER + 1e-6), q)
        self.assertAlmostEqual(float(q_over[0] - q_lift[0]), 3.1416, places=3)
        self.assertAlmostEqual(float(q_over[4]), float(q_lift[4]), places=5)
        ee_drop = arm.kin.fk(q_drop)[:3, 3]
        self.assertAlmostEqual(
            float(arm._place_params["place_fwd"]), float(ee_drop[0]), places=2
        )

    def test_wrist_depth_lowers_z_to_floor(self):
        from arm_pick import grasp_z_from_wrist_depth, median_center_depth
        import numpy as np
        depth = np.full((32, 32), 0.20, dtype=np.float32)
        self.assertAlmostEqual(0.20, median_center_depth(depth), places=5)
        cam = np.eye(4, dtype=np.float64)
        cam[:3, 3] = (0.30, 0.0, 0.12)
        cam[:3, 2] = (0.0, 0.0, -1.0)
        z = grasp_z_from_wrist_depth(cam, 0.20, -0.28)
        self.assertAlmostEqual(-0.08, z, places=5)
        z_floor = grasp_z_from_wrist_depth(cam, 0.50, -0.28)
        self.assertAlmostEqual(-0.28, z_floor, places=5)

    def test_pick_pos_step_is_capped(self):
        path = os.path.join(_ROOT, "demo", "arm_pick.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertIn("POS_STEP_MAX_M", src)
        self.assertIn("stale skip dls", src)

    def test_run_motion_on_pre_replaces_center_window_z(self):
        path = os.path.join(_ROOT, "demo", "arm_pick.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertIn("on_pre=None", src)
        self.assertIn("if callable(on_pre):", src)
        self.assertIn("cmd = None", src)


if __name__ == "__main__":
    unittest.main()
