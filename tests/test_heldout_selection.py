"""Frozen selector arms (#89): the round from frozen candidates, raw vs final answers, scoring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.heldout_selection import exact_match, frozen_round, kernel_credit, raw_answer
from scripts.selector_arms_report import guard_counts


def _row(final, raw=None, kind="object", targets=("c:mug:0",), anchors=("c:bench:0",)):
    return {"kind": kind, "final": final, "raw": raw or final,
            "gold_target_ids": list(targets), "gold_anchor_ids": list(anchors)}


class FrozenRoundTest(unittest.TestCase):
    def test_people_and_objects_are_split_and_sized_from_the_frame(self):
        frame = SimpleNamespace(width=1280, height=720)
        case = SimpleNamespace(candidates=[
            {"candidate_id": "c:mug:0", "camera_id": "c", "class_name": "mug",
             "confidence": 0.9, "bbox_xyxy": [1, 2, 3, 4], "source": "zoom"},
            {"candidate_id": "c:person_sitting:0", "camera_id": "c",
             "class_name": "person_sitting", "confidence": 0.8, "bbox_xyxy": [5, 6, 7, 8],
             "source": "refine"},
        ])
        round_ = frozen_round(case, {"c": frame})
        self.assertEqual([o.candidate_id for o in round_.objects], ["c:mug:0"])
        self.assertEqual(round_.objects[0].source, "zoom")
        self.assertEqual(round_.objects[0].image_size, (1280, 720))
        self.assertEqual(round_.people[0].pose_class, "person_sitting")
        self.assertEqual(round_.camera_ids, ["c"])


class RawAnswerTest(unittest.TestCase):
    def test_guard_override_keeps_qwens_pair_as_the_raw_answer(self):
        notes = [{"decision": "geometry_override", "qwen_target": "c:mug:1", "qwen_anchor": "c:bench:1"},
                 {"decision": "selector", "outcome": "geometry_override", "selected": "c:mug:0"}]
        raw = raw_answer(notes, ("c:mug:0", "c:bench:0"))
        self.assertEqual((raw["target_id"], raw["anchor_id"]), ("c:mug:1", "c:bench:1"))

    def test_unusable_reply_is_no_answer(self):
        notes = [{"decision": "selector", "outcome": "fallback", "reason": "verifier_failed:ValueError",
                  "selected": "c:mug:0"}]
        raw = raw_answer(notes, ("c:mug:0", "c:bench:0"))
        self.assertIsNone(raw["target_id"])
        self.assertEqual(raw["outcome"], "fallback")

    def test_qwen_choice_and_geometry_routes_use_the_selected_pair(self):
        for outcome in ("qwen_choice", "geometry_only"):
            raw = raw_answer([{"decision": "selector", "outcome": outcome, "selected": "c:mug:0"}],
                             ("c:mug:0", "c:bench:0"))
            self.assertEqual((raw["target_id"], raw["anchor_id"]), ("c:mug:0", "c:bench:0"))

    def test_occlusion_hypothesis_has_no_selector_note(self):
        raw = raw_answer([], ("c:hyp:0", "c:bench:0"))
        self.assertEqual(raw["outcome"], "occlusion_hypothesis")


class ScoringTest(unittest.TestCase):
    def test_joint_match_needs_target_and_anchor(self):
        self.assertTrue(exact_match(_row({"target_id": "c:mug:0", "anchor_id": "c:bench:0"}))["joint"])
        self.assertFalse(exact_match(_row({"target_id": "c:mug:0", "anchor_id": "c:bench:1"}))["joint"])
        self.assertFalse(exact_match(_row({"target_id": None, "anchor_id": None}))["joint"])
        # No gold anchor ID: the target decides.
        self.assertTrue(exact_match(_row({"target_id": "c:mug:0", "anchor_id": "x"}, anchors=()))["joint"])

    def test_raw_and_final_are_scored_apart(self):
        row = _row({"target_id": "c:mug:0", "anchor_id": "c:bench:0"},
                   raw={"target_id": "c:mug:1", "anchor_id": "c:bench:0"})
        self.assertTrue(exact_match(row, "final")["joint"])
        self.assertFalse(exact_match(row, "raw")["joint"])

    def test_kernel_credit_attribution_and_anchor(self):
        gold = {"target_xyz": [0.0, 0.0, 0.0], "anchor_xyz": [0.0, 0.0, 0.0]}
        pick = {"target_xyz": [1.0, 0.0, 0.0], "anchor_xyz": [2.5, 0.0, 0.0]}
        self.assertEqual(kernel_credit(pick, gold, []), (1.0, 0.5))
        # A nearer distractor takes the attribution.
        self.assertEqual(kernel_credit(pick, gold, [[1.5, 0.0, 0.0]])[0], 0.0)
        far = {"target_xyz": [3.5, 0.0, 0.0], "anchor_xyz": [0.0, 0.0, 0.0]}
        self.assertTrue(np.isclose(kernel_credit(far, gold, [])[0], 0.5))
        self.assertEqual(kernel_credit({}, gold, []), (0.0, 0.0))


class SarPathTest(unittest.TestCase):
    """The SAR route (``select_sar_candidate``): one person, no anchor pair."""

    def test_sar_guard_keeps_qwens_person_as_the_raw_answer(self):
        notes = [{"decision": "geometry_override", "path": "sar", "reason": "landmark_geometry_override",
                  "guards": ["landmark_geometry_override"], "qwen_target": "c:person_prone:1",
                  "selected": "c:person_prone:0"},
                 {"decision": "selector", "path": "sar", "outcome": "geometry_override",
                  "selected": "c:person_prone:0"}]
        raw = raw_answer(notes, None)
        self.assertEqual(raw["target_id"], "c:person_prone:1")
        row = _row({"target_id": "c:person_prone:0"}, raw=raw, kind="sar",
                   targets=("c:person_prone:0",), anchors=("c:bench:0",))
        # Person exact match ignores the anchor; raw and final are scored apart.
        self.assertTrue(exact_match(row, "final")["joint"])
        self.assertFalse(exact_match(row, "raw")["joint"])
        self.assertEqual(guard_counts({"decisions": notes}), {"landmark_geometry_override": 1})

    def test_sar_qwen_choice_and_geometry_use_the_selected_person(self):
        for outcome in ("qwen_choice", "geometry_only"):
            raw = raw_answer([{"decision": "selector", "path": "sar", "outcome": outcome,
                               "selected": "c:person_sitting:0"}], None)
            self.assertEqual((raw["target_id"], raw["anchor_id"]), ("c:person_sitting:0", None))
        self.assertEqual(guard_counts({"decisions": []}), {})


if __name__ == "__main__":
    unittest.main()
