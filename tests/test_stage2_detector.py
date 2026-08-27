"""Checks for the provisional Stage 2 robot-view dataset contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from training.stage2_detector import prepare_dataset


class Stage2DatasetTests(unittest.TestCase):
    def test_mixed_images_are_remapped_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            names = [
                "tumbler", "cracker_box", "cola_can", "disposable_cup", "juice",
                "tissue", "mug", "sunblock", "pencilcase", "master_chef_can", "wood_block",
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
                        "8 0.5 0.5 0.25 0.5\n",
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
            self.assertEqual(names[:9], list(descriptor["names"].values()))
            self.assertEqual({"train": 3, "val": 3}, manifest["statistics"]["images"])
            self.assertEqual({"train": {"jpg": 3}, "val": {"png": 3}}, manifest["statistics"]["image_formats"])
            self.assertEqual({"train": 2, "val": 2}, manifest["statistics"]["empty_labels_after"])
            self.assertEqual({"master_chef_can": 2}, manifest["statistics"]["dropped_class_boxes"])
            self.assertEqual(0, manifest["split_checks"]["image_stem_overlap"])
            self.assertTrue((output.parent / "images/train/00000.jpg").samefile(raw / "images/train/00000.jpg"))
            self.assertEqual("8 0.5 0.5 0.25 0.5", (output.parent / "labels/train/00002.txt").read_text().strip())

    def test_refuses_output_inside_source(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "raw" / "data.yaml"
            source.parent.mkdir()
            source.write_text("names: {}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside the raw dataset root"):
                prepare_dataset(source, source.parent / "derived/data.yaml", verify_source=False)


if __name__ == "__main__":
    unittest.main()
