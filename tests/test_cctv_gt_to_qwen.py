#!/usr/bin/env python3
"""SAR / on-landmark labelling for the CCTV → Qwen2-VL converter."""

from __future__ import annotations

import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CONV = os.path.join(_ROOT, "demo", "vlm_dataset")
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)

import cctv_gt_to_qwen as q  # noqa: E402


def _det(cls, kind, bbox, xyz=None):
    d = {"class": cls, "kind": kind, "bbox": bbox}
    if xyz is not None:
        d["world_xyz"] = list(xyz)
    return d


def _payload(dets):
    return {"image": {"width": 1280, "height": 720}, "detections": dets}


class SarLabelTests(unittest.TestCase):
    def test_sitting_on_vending_is_not_normal(self):
        person = _det("person_sitting", "object", [600, 200, 700, 420],
                      xyz=(-62.77, 135.78, 17.94))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        bench = _det("bench", "landmark", [100, 400, 220, 520],
                     xyz=(-50.0, 140.0, 16.2))
        rng = random.Random(0)
        rows = q._sar_pairs_for_payload(
            _payload([person, vendor, bench]), rng, 4)
        self.assertTrue(rows)
        instruction, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "vending_machine")
        self.assertIn(rel, ("on", "above"))
        self.assertIn(sit, ("abnormal", "emergency", "accident"))
        self.assertNotEqual(sit, "normal")
        low = instruction.lower()
        self.assertTrue(
            "drink machine" in low or "vending machine" in low,
            instruction,
        )
        self.assertTrue("climbed" in low or "on top" in low, instruction)

    def test_above_wrong_place_is_always_abnormal(self):
        person = _det("person_reaching", "object", [600, 100, 700, 420],
                      xyz=(-62.81, 135.53, 18.75))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        for seed in range(20):
            rows = q._sar_pairs_for_payload(
                _payload([person, vendor]), random.Random(seed), 1)
            self.assertTrue(rows)
            _, _, _, situation, _, relation = rows[0]
            self.assertEqual(relation, "above")
            self.assertEqual(situation, "abnormal")

    def test_on_wrong_place_is_always_abnormal(self):
        person = _det("person_crouching", "object", [600, 200, 700, 420],
                      xyz=(-62.81, 135.53, 16.70))
        trash = _det("trash_can", "landmark", [580, 280, 720, 500],
                     xyz=(-62.81, 135.53, 16.20))
        for seed in range(20):
            rows = q._sar_pairs_for_payload(
                _payload([person, trash]), random.Random(seed), 1)
            _, _, _, situation, _, relation = rows[0]
            self.assertEqual(relation, "on")
            self.assertEqual(situation, "abnormal")

    def test_sitting_on_bench_is_normal(self):
        person = _det("person_sitting", "object", [100, 300, 180, 500],
                      xyz=(0.0, 0.0, 16.75))
        bench = _det("bench", "landmark", [80, 360, 200, 520],
                     xyz=(0.1, 0.05, 16.50))
        rng = random.Random(1)
        rows = q._sar_pairs_for_payload(_payload([person, bench]), rng, 2)
        self.assertTrue(rows)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "bench")
        self.assertEqual(sit, "normal")
        self.assertIn(rel, ("on", "above"))

    def test_sitting_beside_vending_on_ground_is_normal(self):
        # Same z as the machine origin → not on top (person on the ground).
        person = _det("person_sitting", "object", [200, 300, 280, 520],
                      xyz=(-64.5, 135.5, 16.20))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        rng = random.Random(2)
        rows = q._sar_pairs_for_payload(_payload([person, vendor]), rng, 2)
        self.assertTrue(rows)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "vending_machine")
        self.assertIsNone(rel)
        self.assertEqual(sit, "normal")

    def test_standing_beside_hydrant_is_not_on(self):
        # Old 1.6m XY / 0.12m Z window treated this pelvis-above-base pair as
        # "on" the hydrant. Side-by-side boxes must stay beside, not abnormal.
        person = _det("person_reaching", "object", [120, 200, 220, 520],
                      xyz=(-62.0, 135.5, 17.10))
        hydrant = _det("hydrant", "landmark", [400, 360, 460, 520],
                       xyz=(-61.2, 135.4, 16.20))
        rng = random.Random(8)
        rows = q._sar_pairs_for_payload(_payload([person, hydrant]), rng, 1)
        self.assertTrue(rows)
        _, _, _, sit, _, rel = rows[0]
        self.assertIsNone(rel)
        self.assertNotEqual(sit, "abnormal")

    def test_standing_in_front_of_vending_is_not_on(self):
        # Overlapping x, shared ground line, pelvis higher than machine origin.
        person = _det("person_walking", "object", [590, 240, 710, 500],
                      xyz=(-62.85, 135.50, 17.05))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        rng = random.Random(9)
        rows = q._sar_pairs_for_payload(_payload([person, vendor]), rng, 1)
        self.assertTrue(rows)
        _, _, _, sit, _, rel = rows[0]
        self.assertIsNone(rel)
        self.assertNotEqual(sit, "abnormal")

    def test_lying_on_picnic_table_is_normal(self):
        person = _det("person_lying_down", "object", [400, 200, 560, 320],
                      xyz=(-80.9, 96.5, 16.83))
        table = _det("picnic_table", "landmark", [380, 240, 580, 360],
                     xyz=(-80.7, 96.4, 15.65))
        rng = random.Random(3)
        rows = q._sar_pairs_for_payload(_payload([person, table]), rng, 2)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "picnic_table")
        self.assertEqual(sit, "normal")
        self.assertIn(rel, ("on", "above"))

    def test_lying_on_bench_is_abnormal(self):
        person = _det("person_lying_down", "object", [400, 200, 560, 320],
                      xyz=(-84.9, 111.9, 16.00))
        bench = _det("bench", "landmark", [380, 240, 580, 360],
                     xyz=(-84.1, 112.5, 15.48))
        rng = random.Random(4)
        rows = q._sar_pairs_for_payload(_payload([person, bench]), rng, 2)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(sit, "abnormal")
        self.assertIn(rel, ("on", "above"))

    def test_prone_on_ground_is_distress(self):
        person = _det("person_prone", "object", [200, 400, 360, 520],
                      xyz=(-49.3, 125.7, 16.74))
        bike = _det("bicycle", "landmark", [180, 430, 300, 540],
                    xyz=(-48.6, 128.8, 17.02))
        rng = random.Random(5)
        rows = q._sar_pairs_for_payload(_payload([person, bike]), rng, 2)
        _, _, _, sit, _, rel = rows[0]
        self.assertIsNone(rel)
        self.assertIn(sit, ("accident", "emergency"))

    def test_prefers_climber_over_ground_sitter(self):
        climber = _det("person_sitting", "object", [600, 200, 700, 420],
                       xyz=(-62.77, 135.78, 17.94))
        sitter = _det("person_sitting", "object", [100, 300, 180, 500],
                      xyz=(0.0, 0.0, 16.75))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        bench = _det("bench", "landmark", [80, 360, 200, 520],
                     xyz=(0.1, 0.05, 16.50))
        rng = random.Random(6)
        rows = q._sar_pairs_for_payload(
            _payload([sitter, climber, vendor, bench]), rng, 1)
        self.assertEqual(len(rows), 1)
        _, person, lm, sit, _, rel = rows[0]
        self.assertIs(person, climber)
        self.assertEqual(lm["class"], "vending_machine")
        self.assertNotEqual(sit, "normal")
        self.assertIn(rel, ("on", "above"))

    def test_object_on_uses_on_relation(self):
        cup = _det("disposable_cup", "object", [400, 300, 440, 360],
                   xyz=(1.0, 2.0, 16.9))
        bench = _det("bench", "landmark", [360, 320, 500, 420],
                     xyz=(1.05, 2.02, 16.5))
        rng = random.Random(7)
        rows = q._pairs_for_payload(_payload([cup, bench]), rng, 2)
        self.assertTrue(rows)
        instruction, _, lm, rel, _ = rows[0]
        self.assertEqual(lm["class"], "bench")
        self.assertEqual(rel, "on")
        self.assertIn(" on ", instruction.lower())

    def test_hanging_leg_vending_sitter_is_on(self):
        # Full-frame crop: person box shares the vending ground line, so the
        # 2D feet-above-base test fails. World Z still says they are on the roof.
        person = _det("person_sitting", "object", [219, 0, 1019, 719],
                      xyz=(-72.60, 135.18, 17.57))
        vendor = _det("vending_machine", "landmark", [0, 276, 1019, 719],
                      xyz=(-72.06, 134.99, 16.01))
        self.assertFalse(q._stacked_on_top(person, vendor))
        rows = q._sar_pairs_for_payload(
            _payload([person, vendor]), random.Random(0), 1)
        self.assertTrue(rows)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "vending_machine")
        self.assertIn(rel, ("on", "above"))
        self.assertNotEqual(sit, "normal")

    def test_reaching_on_vending_does_not_say_sitting(self):
        person = _det("person_reaching", "object", [600, 100, 700, 420],
                      xyz=(-62.81, 135.53, 18.75))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        for seed in range(40):
            rows = q._sar_pairs_for_payload(
                _payload([person, vendor]), random.Random(seed), 1)
            instruction = rows[0][0].lower()
            self.assertNotIn("sitting", instruction, instruction)

    def test_object_left_of_landmark(self):
        cup = _det("cola_can", "object", [200, 400, 240, 450],
                   xyz=(-1.0, 0.0, 16.5))
        bench = _det("bench", "landmark", [400, 400, 560, 500],
                     xyz=(0.0, 0.0, 16.5))
        payload = {
            "image": {"width": 1280, "height": 720},
            "camera": {"position": [0.0, -10.0, 20.0]},
            "detections": [cup, bench],
        }
        rows = q._pairs_for_payload(payload, random.Random(0), 2)
        self.assertTrue(rows)
        instruction, _, lm, rel, _ = rows[0]
        self.assertEqual(lm["class"], "bench")
        self.assertEqual(rel, "left")
        self.assertIn("left", instruction.lower())


if __name__ == "__main__":
    unittest.main()
