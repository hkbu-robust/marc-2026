"""Tests for Stage 1 CCTV fetch and stub detection."""

from __future__ import annotations

import os
import time
import types
import unittest
from unittest.mock import patch

from marc_agent.perception.cctv_round import fetch_cctv_round
from marc_agent.perception.stage1_detect import detect_stage1_round
from marc_agent.perception.stub_detections import default_stub_path, load_stub_detections
from marc_agent.types import PersonCandidate


class _FakeImage:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.encoding = "rgb8"
        self.step = width * 3
        self.data = bytes([128, 64, 32] * (width * height))


class _FakeClient:
    def __init__(self, camera_ids=None, images=None):
        self._camera_ids = camera_ids or ["rig_2_c", "rig_1_a"]
        self._images = images or {cam: _FakeImage(640, 480) for cam in self._camera_ids}

    def list_cctv(self):
        return list(self._camera_ids)

    def get_cctv_image(self, camera_id: str):
        return self._images.get(camera_id)


class FetchCctvRoundTests(unittest.TestCase):
    def test_fetches_each_camera_once(self):
        client = _FakeClient()
        frames, camera_ids = fetch_cctv_round(client)
        self.assertEqual(["rig_1_a", "rig_2_c"], camera_ids)
        self.assertEqual(set(camera_ids), set(frames))
        self.assertEqual((480, 640, 3), frames["rig_2_c"].rgb.shape)
        self.assertEqual((640, 480), (frames["rig_2_c"].width, frames["rig_2_c"].height))

    def test_skips_cameras_without_images(self):
        client = _FakeClient(images={"rig_2_c": _FakeImage(640, 480)})
        frames, camera_ids = fetch_cctv_round(
            client,
            discovery_timeout_s=0.0,
            discovery_settle_s=0.0,
        )
        self.assertEqual(["rig_1_a", "rig_2_c"], camera_ids)
        self.assertEqual(["rig_2_c"], list(frames))

    def test_waits_for_delayed_images(self):
        class DelayedImageClient(_FakeClient):
            def __init__(self):
                super().__init__(camera_ids=["rig_1_a", "rig_2_c"])
                self.started = time.monotonic()
                self.ready_after_s = 0.03

            def get_cctv_image(self, camera_id: str):
                if time.monotonic() - self.started < self.ready_after_s:
                    return None
                return _FakeImage(640, 480)

        client = DelayedImageClient()
        frames, camera_ids = fetch_cctv_round(
            client,
            discovery_timeout_s=0.2,
            discovery_settle_s=0.0,
            poll_interval_s=0.005,
        )
        self.assertEqual(["rig_1_a", "rig_2_c"], camera_ids)
        self.assertEqual(set(camera_ids), set(frames))

    def test_waits_for_delayed_intrinsics(self):
        class DelayedInfoClient(_FakeClient):
            def __init__(self):
                super().__init__(camera_ids=["rig_1_a"])
                self.info_calls = 0

            def get_cctv_info(self, camera_id: str):
                self.info_calls += 1
                if self.info_calls < 3:
                    return None
                return types.SimpleNamespace(k=[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0])

            def get_cctv_ground_height(self, camera_id: str):
                if self.info_calls < 3:
                    return None
                return 16.5

        client = DelayedInfoClient()
        frames, camera_ids = fetch_cctv_round(
            client,
            discovery_timeout_s=0.2,
            discovery_settle_s=0.0,
            poll_interval_s=0.005,
        )
        self.assertEqual(["rig_1_a"], camera_ids)
        self.assertEqual(["rig_1_a"], list(frames))
        self.assertGreaterEqual(client.info_calls, 3)

    def test_waits_for_first_round_camera_discovery(self):
        class DelayedClient(_FakeClient):
            def __init__(self):
                super().__init__(camera_ids=["rig_1_a", "rig_2_c"])
                self.calls = 0

            def list_cctv(self):
                self.calls += 1
                if self.calls < 3:
                    return []
                return super().list_cctv()

        client = DelayedClient()
        frames, camera_ids = fetch_cctv_round(
            client,
            discovery_timeout_s=0.1,
            discovery_settle_s=0.0,
            poll_interval_s=0.001,
        )
        self.assertEqual(["rig_1_a", "rig_2_c"], camera_ids)
        self.assertEqual(set(camera_ids), set(frames))

    def test_discover_hook_populates_cameras_during_wait(self):
        class DiscoveringClient(_FakeClient):
            def __init__(self):
                super().__init__(camera_ids=["rig_1_a"])
                self._camera_ids = []
                self._images = {}
                self.discover_calls = 0

            def _discover_cctv(self):
                self.discover_calls += 1
                if self.discover_calls < 3:
                    return
                self._camera_ids = ["rig_1_a"]
                self._images = {"rig_1_a": _FakeImage(640, 480)}

        client = DiscoveringClient()
        frames, camera_ids = fetch_cctv_round(
            client,
            discovery_timeout_s=0.2,
            discovery_settle_s=0.0,
            poll_interval_s=0.005,
        )
        self.assertEqual(["rig_1_a"], camera_ids)
        self.assertEqual(["rig_1_a"], list(frames))
        self.assertGreaterEqual(client.discover_calls, 3)


class StubDetectionTests(unittest.TestCase):
    def test_loads_fixture_candidates(self):
        round_ = load_stub_detections(default_stub_path())
        self.assertGreaterEqual(len(round_.objects), 2)
        self.assertEqual(1, len(round_.sar))
        mug = next(item for item in round_.objects if item.class_name == "mug")
        self.assertEqual("rig_2_c:mug:0", mug.candidate_id)
        self.assertEqual("accident", round_.sar[0].situation)

    def test_live_mode_retains_people_and_records_missing_frames(self):
        people = [
            PersonCandidate(
                "rig_2_c:person_prone:0",
                (10.0, 20.0, 30.0, 40.0),
                "person_prone",
                0.9,
                "rig_2_c",
                (640, 480),
            )
        ]
        previous_detector = os.environ.get("MARC_STAGE1_DETECTOR")
        previous_sar = os.environ.get("MARC_SAR_DETECTOR")
        previous_timeout = os.environ.get("MARC_CCTV_DISCOVERY_TIMEOUT_S")
        os.environ["MARC_STAGE1_DETECTOR"] = "yolo"
        os.environ["MARC_SAR_DETECTOR"] = "fallback"
        os.environ["MARC_CCTV_DISCOVERY_TIMEOUT_S"] = "0"
        try:
            client = _FakeClient(
                camera_ids=["rig_2_c", "rig_missing"],
                images={"rig_2_c": _FakeImage(640, 480)},
            )
            with patch(
                "marc_agent.perception.stage1_detect.detect_frames_with_yolo",
                return_value=([], people),
            ):
                round_ = detect_stage1_round(client)
        finally:
            if previous_detector is None:
                os.environ.pop("MARC_STAGE1_DETECTOR", None)
            else:
                os.environ["MARC_STAGE1_DETECTOR"] = previous_detector
            if previous_sar is None:
                os.environ.pop("MARC_SAR_DETECTOR", None)
            else:
                os.environ["MARC_SAR_DETECTOR"] = previous_sar
            if previous_timeout is None:
                os.environ.pop("MARC_CCTV_DISCOVERY_TIMEOUT_S", None)
            else:
                os.environ["MARC_CCTV_DISCOVERY_TIMEOUT_S"] = previous_timeout

        self.assertEqual(people, round_.people)
        self.assertEqual(2, round_.metrics.cameras_discovered)
        self.assertEqual(1, round_.metrics.frames_loaded)
        self.assertEqual(("missing_frame:rig_missing",), round_.metrics.failures)

    def test_filters_unknown_cameras(self):
        round_ = load_stub_detections(
            default_stub_path(),
            camera_ids=["rig_2_c"],
        )
        self.assertTrue(all(item.camera_id == "rig_2_c" for item in round_.objects))
        self.assertEqual([], round_.sar)

    def test_detect_stage1_round_uses_stub_by_default(self):
        prev = os.environ.get("MARC_STAGE1_DETECTOR")
        os.environ["MARC_STAGE1_DETECTOR"] = "stub"
        try:
            client = _FakeClient(camera_ids=["rig_2_c", "rig_1_a", "rig_3_b"])
            round_ = detect_stage1_round(client)
        finally:
            if prev is None:
                os.environ.pop("MARC_STAGE1_DETECTOR", None)
            else:
                os.environ["MARC_STAGE1_DETECTOR"] = prev

        self.assertEqual(["rig_1_a", "rig_2_c", "rig_3_b"], round_.camera_ids)
        self.assertIn("rig_2_c", round_.frames)
        self.assertTrue(any(item.class_name == "mug" for item in round_.objects))
        self.assertEqual(1, len(round_.sar))


if __name__ == "__main__":
    unittest.main()
