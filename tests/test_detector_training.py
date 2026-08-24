"""Small check for the only custom training logic: the Stage 1 taxonomy remap."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from training.train_cctv_detector import prepare_dataset
from training.evaluate_cctv_detector import _score, _select_head, _select_runtime


class DetectorDatasetTests(unittest.TestCase):
    def _source(self, root: Path, excluded_box: bool = False) -> Path:
        names = {key: f"class_{key}" for key in reversed(range(35))}
        names[4], names[5] = "umbrella", "long_umbrella"
        for split in ("train", "val"):
            for camera_id in range(1, 7):
                stem = f"scene_{camera_id + (0 if split == 'train' else 10):05d}_rig_{camera_id}_a"
                image = root / "images" / split / f"{stem}.png"
                label = root / "labels" / split / f"{stem}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"fixture")
                class_id = 4 if excluded_box and camera_id == 1 else 6
                label.write_text(f"{class_id} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        descriptor = root / "data.yaml"
        descriptor.write_text(yaml.safe_dump({"train": "images/train", "val": "images/val", "names": names}, sort_keys=False))
        return descriptor

    def test_remaps_above_exclusions_and_records_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "raw")
            output = root / "processed" / "data.yaml"
            stats = prepare_dataset(source, output, verify_hash=False)
            self.assertEqual(33, len(stats["names"]))
            self.assertEqual(12, sum(stats["boxes"].values()))
            self.assertEqual("4 0.5 0.5 0.1 0.1", next((output.parent / "labels" / "train").glob("*.txt")).read_text().strip())
            self.assertEqual(6, len(stats["cameras"]["train"]))
            linked = next((output.parent / "images" / "train").glob("*.png"))
            original = next((source.parent / "images" / "train").glob("*.png"))
            self.assertTrue(linked.samefile(original))

    def test_rejects_boxes_for_excluded_classes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "raw", excluded_box=True)
            with self.assertRaisesRegex(ValueError, "excluded class 4 has a box"):
                prepare_dataset(source, root / "processed" / "data.yaml", verify_hash=False)

    def test_refuses_to_write_inside_raw_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self._source(Path(temp) / "raw")
            with self.assertRaisesRegex(ValueError, "outside the raw dataset root"):
                prepare_dataset(source, source, verify_hash=False)

    def test_operational_score_matches_by_class_and_iou(self):
        frames = [{"image": Path("frame_rig_1_a.png"), "camera": "rig_1_a", "labels": [(0, 0.5, 0.5, 0.2, 0.2)]}]
        predictions = {"frame_rig_1_a.png": [
            {"class_id": 0, "conf": 0.9, "xyxy": [512, 288, 768, 432], "shape": (720, 1280)}
        ]}
        result = _score(frames, predictions, ["target"], conf=0.5, budget=1)
        self.assertEqual(1.0, result["recall"])
        self.assertEqual(1.0, result["precision"])
        self.assertEqual(1.0, result["cameras"]["rig_1_a"]["recall"])

    def test_operational_score_matches_in_confidence_order(self):
        frames = [{
            "image": Path("x.png"), "camera": "rig_1_a",
            "labels": [(0, 0.05, 0.05, 0.10, 0.10), (0, 0.07, 0.05, 0.10, 0.10)],
        }]
        predictions = {"x.png": [
            {"class_id": 0, "conf": 0.9, "xyxy": [-2, 0, 8, 10], "shape": (100, 100)},
            {"class_id": 0, "conf": 0.8, "xyxy": [0, 0, 10, 10], "shape": (100, 100)},
        ]}

        result = _score(frames, predictions, ["target"], conf=0.1, budget=2)

        self.assertEqual(1.0, result["recall"])
        self.assertEqual(1.0, result["precision"])

    def test_head_selection_uses_equal_candidate_budget(self):
        def result(end2end, confidence, budget, recall, nms_iou=0.7, macro_f1=None):
            bucket = {"recall": recall}
            return {
                "end2end": end2end, "nms_iou": nms_iou, "confidence": confidence,
                "candidate_limit": budget, "recall": recall, "macro_recall": recall,
                "macro_f1": recall if macro_f1 is None else macro_f1,
                "mean_inference_ms": 10.0, "classes": {"target": bucket},
                "cameras": {"rig_1_a": bucket}, "sizes": {"tiny": bucket, "small": bucket},
            }

        comparisons = [
            result(True, 0.001, 20, 0.9),
            result(False, 0.001, 20, 0.8),
            result(False, 0.1, 10, 0.99),
        ]
        self.assertTrue(_select_head(comparisons))

    def test_runtime_selection_considers_every_nms_setting(self):
        def result(nms_iou, macro_f1):
            bucket = {"recall": 0.9}
            return {
                "end2end": False, "nms_iou": nms_iou, "confidence": 0.3,
                "candidate_limit": 50, "recall": 0.9, "macro_recall": 0.9,
                "macro_f1": macro_f1, "mean_inference_ms": 10.0,
                "classes": {"target": bucket}, "cameras": {"rig_1_a": bucket},
                "sizes": {"tiny": bucket, "small": bucket},
            }

        comparisons = [result(0.7, 0.8), result(0.5, 0.9)]
        self.assertEqual(0.5, _select_runtime(comparisons, False)["nms_iou"])
        self.assertEqual(0.5, _select_runtime(list(reversed(comparisons)), False)["nms_iou"])


if __name__ == "__main__":
    unittest.main()
