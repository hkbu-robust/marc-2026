"""GPU handoff: unload clears cached detectors/generators; pick success reloads YOLO."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

from marc_agent import gpu_handoff
from marc_agent.gpu_handoff import after_successful_pick, before_graspgenx, before_reacquire
from marc_agent.graspgenx_policy import unload_graspgenx
from marc_agent.perception import cctv_yolo
from marc_agent.pick_runtime import PickRuntime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)


class _FakeDetector:
    def __init__(self):
        self.model = object()
        self.loads = 0
        self.unloads = 0

    def load(self):
        self.loads += 1
        self.model = object()
        return True

    def unload(self):
        self.unloads += 1
        self.model = None


class GpuHandoffTests(unittest.TestCase):
    def test_cctv_unload_clears_singleton(self):
        fake = _FakeDetector()
        previous = cctv_yolo._detector
        cctv_yolo._detector = fake
        try:
            gpu_handoff.unload_cctv_yolo()
            self.assertIsNone(cctv_yolo._detector)
            self.assertIsNone(fake.model)
            self.assertEqual(1, fake.unloads)
        finally:
            cctv_yolo._detector = previous

    def test_graspgenx_unload_clears_module_cache(self):
        import marc_agent.graspgenx_policy as policy

        previous = policy._generator
        policy._generator = lambda _pts: None
        try:
            unload_graspgenx()
            self.assertIsNone(policy._generator)
        finally:
            policy._generator = previous

    def test_pick_runtime_unload_clears_generator(self):
        client = SimpleNamespace(is_running=True, sleep=lambda _dt: None)
        runtime = PickRuntime(client, log=lambda _m: None, policy="ik")
        runtime._grasp_generator = object()
        runtime.unload_generator()
        self.assertIsNone(runtime._grasp_generator)

    def test_yolo_tracker_unload_clears_model(self):
        from manip_dataset.yolo_detect import YoloTracker

        tracker = YoloTracker()
        tracker.model = object()
        tracker.unload()
        self.assertIsNone(tracker.model)

    def test_pick_success_unloads_graspgen_then_reloads_yolo(self):
        events = []
        detector = SimpleNamespace(
            load=lambda: events.append("yolo_load") or True,
            unload=lambda: events.append("yolo_unload"),
        )
        pick = SimpleNamespace(unload_generator=lambda: events.append("gx_unload"))
        after_successful_pick(pick, detector)
        self.assertEqual(["gx_unload", "yolo_load"], events)

    def test_before_grasp_unloads_nearfield(self):
        events = []
        detector = SimpleNamespace(unload=lambda: events.append("yolo_unload"))
        before_graspgenx(detector)
        self.assertEqual(["yolo_unload"], events)

    def test_reacquire_unloads_graspgen_and_reloads_yolo(self):
        events = []
        detector = SimpleNamespace(
            load=lambda: events.append("yolo_load") or True,
            unload=lambda: events.append("yolo_unload"),
        )
        pick = SimpleNamespace(unload_generator=lambda: events.append("gx_unload"))
        before_reacquire(pick, detector)
        self.assertEqual(["gx_unload", "yolo_load"], events)

    def test_unload_qwen_clears_cached_verifier(self):
        from marc_agent.perception import stage1_pipeline

        events = []
        fake = SimpleNamespace(unload=lambda: events.append("qwen_unload"))
        previous = stage1_pipeline._CACHED_VERIFIER
        stage1_pipeline._CACHED_VERIFIER = fake
        try:
            gpu_handoff.unload_qwen()
            self.assertIsNone(stage1_pipeline._CACHED_VERIFIER)
            self.assertEqual(["qwen_unload"], events)
        finally:
            stage1_pipeline._CACHED_VERIFIER = previous

    def test_participant_wires_handoff_hooks(self):
        path = os.path.join(_ROOT, "demo", "participant_app.py")
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertIn("unload_cctv_yolo", src)
        self.assertIn("unload_qwen", src)
        self.assertIn("before_graspgenx", src)
        self.assertIn("before_reacquire", src)
        self.assertIn("after_successful_pick", src)
        self.assertIn("nearfield_start_reason", src)
        self.assertIn("dump_stage_round", src)


if __name__ == "__main__":
    unittest.main()
