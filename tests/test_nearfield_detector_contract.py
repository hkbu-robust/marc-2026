"""Public exact-target contract for the Stage 2 near-field detector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from demo.manip_dataset.yolo_detect import MAX_CANDIDATES, YoloConfig, YoloTracker


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Boxes:
    def __init__(self, xyxy=(), confidence=(), class_ids=()):
        self.xyxy = _Array(xyxy)
        self.conf = _Array(confidence)
        self.cls = _Array(class_ids)

    def __len__(self):
        return len(self.conf.value)


class _Model:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, _image, **kwargs):
        self.calls.append(kwargs)
        return [type("Result", (), {"boxes": self.boxes})()]


class NearfieldDetectorContractTests(unittest.TestCase):
    def _tracker(self, boxes) -> YoloTracker:
        tracker = YoloTracker(YoloConfig(device="cpu", candidate_limit=MAX_CANDIDATES))
        tracker.model = _Model(boxes)
        tracker.names = {0: "mug", 1: "tumbler"}
        tracker.device = "cpu"
        return tracker

    def test_unsupported_target_never_runs_all_classes(self):
        tracker = self._tracker(_Boxes())
        result = tracker.detect_candidates(np.zeros((48, 64, 3), dtype=np.uint8), "cup")
        self.assertEqual("unsupported_target", result["status"])
        self.assertEqual([], result["candidates"])
        self.assertEqual([], tracker.model.calls)

    def test_supported_target_without_model_is_detector_unavailable(self):
        tracker = YoloTracker(YoloConfig(ordered_classes=("mug", "tumbler")))
        result = tracker.detect_candidates(np.zeros((48, 64, 3), dtype=np.uint8), "mug")
        self.assertEqual("detector_unavailable", result["status"])

    def test_supported_but_missing_is_not_detected(self):
        tracker = self._tracker(_Boxes())
        result = tracker.detect_candidates(np.zeros((48, 64, 3), dtype=np.uint8), "mug")
        self.assertEqual("not_detected", result["status"])
        self.assertEqual([0], tracker.model.calls[0]["classes"])
        self.assertEqual(MAX_CANDIDATES, tracker.model.calls[0]["max_det"])

    def test_candidates_are_exact_sorted_bounded_and_original_pixels(self):
        count = MAX_CANDIDATES + 5
        boxes = [[-2, -3, 70, 50] for _ in range(count)]
        confidence = np.linspace(0.1, 0.9, count)
        tracker = self._tracker(_Boxes(boxes, confidence, [1] + [0] * (count - 1)))
        result = tracker.detect_candidates(np.zeros((48, 64, 3), dtype=np.uint8), "mug")
        self.assertEqual("ok", result["status"])
        self.assertEqual((48, 64), result["rgb_shape"])
        self.assertEqual(MAX_CANDIDATES, len(result["candidates"]))
        self.assertEqual(
            sorted((item["confidence"] for item in result["candidates"]), reverse=True),
            [item["confidence"] for item in result["candidates"]],
        )
        self.assertTrue(all(item["class_name"] == "mug" for item in result["candidates"]))
        self.assertTrue(all(item["class_id"] == 0 for item in result["candidates"]))
        self.assertEqual(list(range(1, MAX_CANDIDATES + 1)), [item["rank"] for item in result["candidates"]])
        self.assertEqual((0.0, 0.0, 64.0, 48.0), result["candidates"][0]["xyxy"])

    def test_detect_is_thin_first_candidate_wrapper(self):
        tracker = self._tracker(_Boxes([[10, 10, 30, 30]], [0.8], [0]))
        tracker.set_target("mug")
        result = tracker.detect(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertEqual("ok", result["status"])
        self.assertEqual((10.0, 10.0, 30.0, 30.0), result["bbox"])
        self.assertEqual("mug", result["name"])

    def test_config_reads_selected_bundle_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "model.pt"
            weights.touch()
            (root / "runtime_contract.json").write_text(json.dumps({
                "status": "provisional",
                "ordered_classes": ["mug", "tumbler"],
                "known_limitations": ["ordinary umbrella is absent"],
                "inference": {
                    "imgsz": 1280, "confidence": 0.2, "nms_iou": 0.5,
                    "end2end": False, "candidate_limit": 50, "quantize": "fp16",
                },
            }), encoding="utf-8")
            with patch.dict("os.environ", {"MARC_YOLO_WEIGHTS": str(weights)}, clear=False):
                config = YoloConfig.from_env()
            self.assertEqual(1280, config.imgsz)
            self.assertEqual(0.2, config.conf)
            self.assertEqual(MAX_CANDIDATES, config.candidate_limit)
            self.assertEqual("fp16", config.quantize)
            self.assertEqual(("mug", "tumbler"), config.ordered_classes)
            self.assertEqual("provisional", config.status)

    def test_checked_in_contract_is_provisional_ten_class(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "models/runtime/stage2-nearfield-detector/runtime_contract.json").read_text(
                encoding="utf-8"
            )
        )
        classes = list(payload["ordered_classes"])
        self.assertEqual(10, len(classes))
        self.assertIn("umbrella", classes)
        inference = payload["inference"]
        self.assertEqual(1920, inference["imgsz"])
        self.assertEqual(0.2, inference["confidence"])
        self.assertEqual("provisional", payload["status"])


if __name__ == "__main__":
    unittest.main()
