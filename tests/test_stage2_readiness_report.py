"""Focused checks for the Stage 2 readiness report update."""

from pathlib import Path
import unittest

from training.generate_stage2_readiness_report import _pick_blind, _view


class Stage2ReadinessReportTests(unittest.TestCase):
    def test_missing_baseline_class_is_explicitly_unsupported(self):
        result = _view(None, [[1, 2, 3, 4]], 0.3, 10)
        self.assertTrue(result["unsupported"])
        self.assertIsNone(result["matched"])
        self.assertEqual([], result["candidates"])

    def test_blind_cards_are_frame_disjoint(self):
        records = []
        for class_id, target in enumerate(("mug", "umbrella")):
            for frame_id in range(3):
                records.append({
                    "target": target,
                    "frame": {"image": Path(f"frame-{frame_id}.png"), "labels": [(class_id,)]},
                    "gt": [[1, 1, 9, 9]],
                    "stage2": {
                        "matched": 1, "false_candidates": frame_id,
                        "missed_gt": [], "candidates": [],
                    },
                })
        samples, _ = _pick_blind(records, ["mug", "umbrella"], set())
        images = [sample["frame"]["image"].name for sample in samples]
        self.assertEqual(len(images), len(set(images)))


if __name__ == "__main__":
    unittest.main()
