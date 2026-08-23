"""Tests for Stage 1 CCTV fetch and stub detection."""

from __future__ import annotations

import os
import unittest

from marc_agent.perception.cctv_round import fetch_cctv_round
from marc_agent.perception.stage1_detect import detect_stage1_round
from marc_agent.perception.stub_detections import default_stub_path, load_stub_detections


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
        frames, camera_ids = fetch_cctv_round(client)
        self.assertEqual(["rig_1_a", "rig_2_c"], camera_ids)
        self.assertEqual(["rig_2_c"], list(frames))


class StubDetectionTests(unittest.TestCase):
    def test_loads_fixture_candidates(self):
        round_ = load_stub_detections(default_stub_path())
        self.assertGreaterEqual(len(round_.objects), 2)
        self.assertEqual(1, len(round_.sar))
        mug = next(item for item in round_.objects if item.class_name == "mug")
        self.assertEqual("rig_2_c:mug:0", mug.candidate_id)
        self.assertEqual("accident", round_.sar[0].situation)

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
