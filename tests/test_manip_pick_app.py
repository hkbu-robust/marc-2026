"""GraspGen-X pick tester: G runs PickRuntime, no HDF5 recorder."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

from marc_agent.pick_runtime import PickResult  # noqa: E402
from manip_dataset.pick_app import HELP_TEXT, PickTestApp, Phase  # noqa: E402


class _Approach:
    def __init__(self):
        self.target = None
        self.cancelled = 0
        self.started = 0
        self.cfg = SimpleNamespace(tick_s=0.05)

    def set_target(self, name):
        self.target = name

    def cancel(self):
        self.cancelled += 1

    def detector_ready(self):
        return True

    def ensure_detector(self):
        return True

    def start(self):
        self.started += 1
        return True

    def tick(self):
        return False

    def settle(self):
        return None


class _Nearfield:
    def __init__(self, target):
        self.detector = SimpleNamespace(
            load=lambda: True,
            unload=lambda: None,
        )
        self.target = target
        self.calls = 0

    def snapshot(self, name):
        self.calls += 1
        self.last_name = name
        return self.target


class ManipPickAppTests(unittest.TestCase):
    def test_g_runs_pick_runtime_on_snapshot(self):
        seen = []
        target = SimpleNamespace(
            valid=True, fwd=0.40, lat=0.0, z=-0.22, conf=0.8,
            name="tumbler", reason="ok",
        )
        runtime = SimpleNamespace(
            run=lambda t: seen.append(t) or PickResult(
                ok=True, reason="ok", attempts=1, grasped=True,
                basket_occupied=True, policy="graspgenx",
            ),
            _reacquire_fn=None,
        )
        client = SimpleNamespace(send_cmd_vel=lambda **_k: None)
        app = PickTestApp(
            client, "u1",
            arm=SimpleNamespace(),
            approach=_Approach(),
            nearfield=_Nearfield(target),
            pick_runtime=runtime,
        )
        app.phase = Phase.MANUAL_BASE
        app.handle_key("G")
        self.assertEqual(1, app.nearfield.calls)
        self.assertEqual([target], seen)
        self.assertEqual(target, runtime._reacquire_fn())

    def test_g_skips_pick_when_snapshot_invalid(self):
        seen = []
        runtime = SimpleNamespace(run=lambda t: seen.append(t))
        app = PickTestApp(
            SimpleNamespace(send_cmd_vel=lambda **_k: None),
            "u1",
            arm=SimpleNamespace(),
            approach=_Approach(),
            nearfield=_Nearfield(SimpleNamespace(valid=False, reason="not_detected")),
            pick_runtime=runtime,
        )
        app.handle_key("G")
        self.assertEqual([], seen)

    def test_snapshot_fail_prints_xyz(self):
        target = SimpleNamespace(
            valid=False, reason="out_of_reach",
            fwd=1.21, lat=0.02, z=0.01, conf=0.4, name="tumbler",
        )
        client = SimpleNamespace(send_cmd_vel=lambda **_k: None)
        app = PickTestApp(
            client, "u1",
            arm=SimpleNamespace(),
            approach=_Approach(),
            nearfield=_Nearfield(target),
            pick_runtime=SimpleNamespace(run=lambda _t: None, _reacquire_fn=None),
        )
        with patch("builtins.print") as printed:
            app.handle_key("G")
        text = " ".join(str(c.args[0]) for c in printed.call_args_list if c.args)
        self.assertIn("out_of_reach", text)
        self.assertIn("1.210", text)
        self.assertIn("0.020", text)

    def test_source_has_no_episode_recorder(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "demo", "manip_dataset", "pick_app.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertNotIn("EpisodeRecorder", src)
        self.assertNotIn("MARC_DATASET_ROOT", src)
        self.assertIn('ch == "G"', src)
        self.assertIn('ch == "H"', src)
        self.assertIn("GraspGen-X", HELP_TEXT)
        self.assertIn("H       wrist-YOLO pick + basket", HELP_TEXT)

    def test_g_unloads_yolo_before_graspgenx(self):
        target = SimpleNamespace(
            valid=True, fwd=0.40, lat=0.0, z=-0.22, conf=0.8,
            name="tumbler", reason="ok",
        )
        runtime = SimpleNamespace(
            run=lambda t: PickResult(
                ok=True, reason="ok", attempts=1, grasped=True,
                basket_occupied=True, policy="graspgenx",
            ),
            _reacquire_fn=None,
        )
        app = PickTestApp(
            SimpleNamespace(send_cmd_vel=lambda **_k: None),
            "u1",
            arm=SimpleNamespace(),
            approach=_Approach(),
            nearfield=_Nearfield(target),
            pick_runtime=runtime,
        )
        with patch("manip_dataset.pick_app.before_graspgenx") as before:
            app.handle_key("G")
        before.assert_called_once()

    def test_h_does_not_call_before_graspgenx(self):
        target = SimpleNamespace(
            valid=True, fwd=0.16, lat=0.0, z=-0.22, conf=0.8,
            name="tumbler", reason="ok",
        )
        seen = []
        runtime = SimpleNamespace(
            run=lambda t: seen.append(t) or PickResult(
                ok=True, reason="ok", attempts=1, grasped=True,
                basket_occupied=True, policy="wrist",
            ),
            _reacquire_fn=None,
        )
        detector = SimpleNamespace(load=lambda: True, unload=lambda: None)
        near = _Nearfield(target)
        near.detector = detector
        arm = SimpleNamespace()
        app = PickTestApp(
            SimpleNamespace(send_cmd_vel=lambda **_k: None),
            "u1",
            arm=arm,
            approach=_Approach(),
            nearfield=near,
            pick_runtime=runtime,
        )
        with patch("manip_dataset.pick_app.before_graspgenx") as before:
            app.handle_key("H")
        before.assert_not_called()
        self.assertEqual([target], seen)
        self.assertIs(detector, runtime._wrist_detector)
        self.assertIs(arm, runtime._teleop)
        nxt = runtime._reacquire_fn()
        self.assertEqual(target, nxt)
        before.assert_not_called()


if __name__ == "__main__":
    unittest.main()
