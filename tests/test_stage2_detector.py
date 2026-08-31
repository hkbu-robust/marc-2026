"""Checks for the provisional Stage 2 robot-view dataset contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from training.evaluate_stage2_detector import (
    _best_available,
    _frames,
    _score,
    _select,
    _training_record,
)
from training.stage2_detector import prepare_dataset


class Stage2DatasetTests(unittest.TestCase):
    def test_mixed_images_are_remapped_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            names = [
                "tumbler", "cracker_box", "cola_can", "disposable_cup", "juice",
                "tissue", "mug", "sunblock", "pencilcase", "master_chef_can", "wood_block",
                "umbrella",
            ]
            for split, suffix, offset in (("train", ".jpg", 0), ("val", ".png", 10)):
                for index in range(3):
                    stem = f"{index + offset:05d}"
                    image = raw / "images" / split / f"{stem}{suffix}"
                    label = raw / "labels" / split / f"{stem}.txt"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (64, 48)).save(image)
                    label.write_text(
                        "" if index == 0 else
                        "9 0.5 0.5 0.1 0.1\n" if index == 1 else
                        "8 0.5 0.5 0.25 0.5\n11 0.5 0.5 0.1 0.25\n",
                        encoding="utf-8",
                    )
            source = raw / "data.yaml"
            source.write_text(yaml.safe_dump({
                "path": "/container/only",
                "train": "images/train", "val": "images/val", "names": dict(enumerate(names)),
            }), encoding="utf-8")
            before = {path.relative_to(raw): path.read_bytes() for path in raw.rglob("*") if path.is_file()}

            output = root / "derived" / "data.yaml"
            manifest = prepare_dataset(source, output, verify_source=False)

            self.assertEqual(before, {
                path.relative_to(raw): path.read_bytes() for path in raw.rglob("*") if path.is_file()
            })
            descriptor = yaml.safe_load(output.read_text(encoding="utf-8"))
            self.assertNotIn("path", descriptor)
            expected_names = [*names[:9], "umbrella"]
            self.assertEqual(expected_names, list(descriptor["names"].values()))
            self.assertEqual({"train": 3, "val": 3}, manifest["statistics"]["images"])
            self.assertEqual({"train": {"jpg": 3}, "val": {"png": 3}}, manifest["statistics"]["image_formats"])
            self.assertEqual({"train": 2, "val": 2}, manifest["statistics"]["empty_labels_after"])
            self.assertEqual({"master_chef_can": 2}, manifest["statistics"]["dropped_class_boxes"])
            self.assertEqual(0, manifest["split_checks"]["image_stem_overlap"])
            self.assertTrue((output.parent / "images/train/00000.jpg").samefile(raw / "images/train/00000.jpg"))
            self.assertEqual(
                "8 0.5 0.5 0.25 0.5\n9 0.5 0.5 0.1 0.25",
                (output.parent / "labels/train/00002.txt").read_text().strip(),
            )
            names_out, frames = _frames(output)
            self.assertEqual(expected_names, names_out)
            self.assertEqual(3, len(frames))
            self.assertEqual((48, 64), frames[0]["shape"])

    def test_refuses_output_inside_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "raw" / "data.yaml"
            source.parent.mkdir()
            source.write_text("names: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the raw dataset root"):
                prepare_dataset(source, source.parent / "derived/data.yaml", verify_source=False)

    def test_training_config_is_the_agreed_single_run(self):
        config = yaml.safe_load(
            (Path(__file__).parents[1] / "training/configs/stage2-roboview-yolo26x.yaml").read_text()
        )
        self.assertEqual(1920, config["imgsz"])
        self.assertEqual(4, config["batch"])
        self.assertEqual(100, config["epochs"])
        self.assertEqual(20, config["patience"])
        self.assertEqual(10, config["save_period"])
        self.assertTrue(config["cls_remap"])
        self.assertEqual(0.25, config["cls_pw"])
        self.assertEqual(0.0, config["mosaic"])
        self.assertEqual(0.0, config["mixup"])
        self.assertEqual(0.0, config["cutmix"])

    def test_selection_rejects_zero_recall(self):
        def result(macro_f1: float, recall: float) -> dict:
            bucket = {"recall": recall}
            return {
                "macro_f1": macro_f1,
                "recall": recall,
                "candidate_limit": 20,
                "classes": {"target": bucket},
                "sizes": {"tiny": bucket, "small": bucket},
            }

        self.assertEqual(0.8, _select([result(0.99, 0.0), result(0.8, 0.8)])["macro_f1"])
        self.assertEqual(0.8, _best_available([result(0.99, 0.0), result(0.8, 0.8)])["macro_f1"])

    def test_target_budget_is_applied_after_class_filtering(self):
        frame = {
            "image": Path("frame.png"), "camera": "all", "shape": (48, 64),
            "labels": [(0, 0.5, 0.5, 0.25, 0.25)],
        }
        distractors = [
            {"class_id": 1, "conf": 0.99 - index / 100, "xyxy": [0, 0, 5, 5]}
            for index in range(20)
        ]
        target = {"class_id": 0, "conf": 0.1, "xyxy": [24, 18, 40, 30]}
        score = _score([frame], {"frame.png": [*distractors, target]}, ["target", "other"], 0.01, 1)
        self.assertEqual(1.0, score["classes"]["target"]["recall"])
        self.assertEqual("after exact-target filtering", score["candidate_limit_applied"])

    def test_size_buckets_use_native_short_side(self):
        frame = {
            "image": Path("frame.png"), "camera": "all", "shape": (480, 640),
            "labels": [(0, 0.5, 0.5, 40 / 640, 8 / 480)],
        }
        prediction = {"class_id": 0, "conf": 0.9, "xyxy": [300, 236, 340, 244]}
        score = _score([frame], {"frame.png": [prediction]}, ["target"], 0.01, 20)
        self.assertEqual(1, score["sizes"]["tiny"]["gt"])
        self.assertEqual(1, score["classes"]["target"]["by_size"]["tiny"]["gt"])
        self.assertNotIn("small", score["sizes"])

    def test_smoke_record_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as temp:
            record = Path(temp) / "training_record.json"
            record.write_text('{"promotion_eligible": false}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "smoke checkpoints"):
                _training_record(record)


if __name__ == "__main__":
    unittest.main()
