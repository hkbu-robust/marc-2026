"""The submitted SAR situation and the always-submit fallback (#114)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from marc_agent.command_parser import Intent, ParsedCommand, Situation
from marc_agent.experiment_log import decision_scope
from marc_agent.types import CctvDetectionRound, ObjectCandidate
from marc_agent.perception.stage1_pipeline import (
    IGNORED_CLASS_DUMMY_XYZ,
    fallback_grounding,
    submitted_situation,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sar_situation_88.json"


def _person(**fields):
    return ParsedCommand(intent=Intent.FIND, target_type="person", **fields)


class SituationTests(unittest.TestCase):
    def test_r03_route_matches_the_88_sar_rounds(self):
        rounds = json.loads(FIXTURE.read_text())["rounds"]
        self.assertEqual(140, len(rounds))
        correct = sum(
            submitted_situation(_person(scene_hint=row["scene_hint"]), row["pose"],
                                row["anchor_class"])[0] == row["expected"]
            for row in rounds
        )
        self.assertGreaterEqual(correct, 138)

    def test_text_hint_then_default_without_a_detected_pose(self):
        self.assertEqual(("emergency", "hint"),
                         submitted_situation(_person(situation=Situation.EMERGENCY), None, None))
        self.assertEqual(("normal", "default"), submitted_situation(_person(), None, None))

    def test_the_detected_pose_wins_over_the_text(self):
        # "face-down on the bench" reads as prone, but the detected pose is bending (sar03).
        with decision_scope("stage1", 1) as notes:
            value, source = submitted_situation(
                _person(situation=Situation.ACCIDENT), "person_bending", "bench")
        self.assertEqual(("abnormal", "r03"), (value, source))
        self.assertTrue([n for n in notes if n["decision"] == "situation" and n["source"] == "r03"])


class FallbackTests(unittest.TestCase):
    def test_an_unreadable_command_still_submits(self):
        with decision_scope("stage1", 1) as notes:
            grounding = fallback_grounding("do something", CctvDetectionRound(
                objects=[ObjectCandidate("rig_b:mug:0", "mug", (0, 0, 1, 1), 0.9, "rig_b", (640, 480))],
                camera_ids=["rig_a", "rig_b"]))
        self.assertEqual("rig_b", grounding.camera_id)  # the camera with the most candidates
        self.assertEqual("unknown", grounding.target_type)
        self.assertEqual(list(IGNORED_CLASS_DUMMY_XYZ), list(grounding.target_coord))
        self.assertTrue([n for n in notes if n["decision"] == "sentinel"])
        self.assertTrue([n for n in notes if n["decision"] == "field_source"
                         and n["target_type"] == "unknown"])

    def test_a_readable_command_keeps_its_fields(self):
        grounding = fallback_grounding("Find the mug on the bench.")
        self.assertEqual(("mug", "bench", "on"),
                         (grounding.target_type, grounding.landmark, grounding.relation))


if __name__ == "__main__":
    unittest.main()
