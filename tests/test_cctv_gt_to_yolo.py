#!/usr/bin/env python3
"""YOLO converter keeps umbrella; drops long_umbrella, chef, and wood."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CONV = os.path.join(_ROOT, "demo", "vlm_dataset")
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)

import cctv_gt_to_yolo as y  # noqa: E402


class YoloExportTests(unittest.TestCase):
    def test_canonical_ids_keep_umbrellas_at_four_and_five(self):
        self.assertEqual(y.CANONICAL_NAMES[4], "umbrella")
        self.assertEqual(y.CANONICAL_NAMES[5], "long_umbrella")
        self.assertIn("master_chef_can", y.CANONICAL_NAMES)
        self.assertIn("wood_block", y.CANONICAL_NAMES)

    def test_keeps_umbrella_drops_excluded_classes(self):
        names = list(y.CANONICAL_NAMES)
        kept = y._write_yolo_line(
            {"class": "umbrella", "bbox": [10, 10, 40, 40]},
            names, 1280, 720)
        self.assertIsNotNone(kept)
        self.assertTrue(kept.startswith("4 "))
        for cls in (
                "long_umbrella", "closedlongumbrella",
                "master_chef_can", "wood_block", "wooden_block"):
            self.assertIsNone(y._write_yolo_line(
                {"class": cls, "bbox": [10, 10, 40, 40]},
                names, 1280, 720), cls)
        tumbler = y._write_yolo_line(
            {"class": "tumbler", "bbox": [10, 10, 40, 40]},
            names, 1280, 720)
        self.assertTrue(tumbler.startswith("0 "))

    def test_wooden_block_aliases_then_drops(self):
        self.assertEqual("wood_block", y._canon_class("wooden_block"))
        self.assertIn("wood_block", y.EXCLUDED_BOX_CLASSES)

    def test_drops_high_occlusion_keeps_missing_or_low(self):
        names = list(y.CANONICAL_NAMES)
        det = {"class": "tumbler", "bbox": [10, 10, 40, 40], "occlusion": 0.90}
        self.assertIsNone(y._write_yolo_line(det, names, 1280, 720))
        det["occlusion"] = 0.84
        self.assertIsNotNone(y._write_yolo_line(det, names, 1280, 720))
        del det["occlusion"]
        self.assertIsNotNone(y._write_yolo_line(det, names, 1280, 720))


if __name__ == "__main__":
    unittest.main()
