"""Qwen command reading under a JSON schema, with the rule fallback (#114)."""

from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from marc_agent import command_llm
from marc_agent.command_parser import Intent, Relation
from marc_agent.experiment_log import decision_scope

ORGANISER_OBJECTS = {"master_chef_can", "cracker_box", "mug", "wood_block", "cola_can",
                     "disposable_cup", "juice", "tissue", "pencilcase", "tumbler", "umbrella",
                     "long_umbrella", "sunblock"}
ORGANISER_LANDMARKS = {"bench", "bicycle", "hydrant", "kick_scooter", "picnic_table", "postbox",
                       "vending_machine", "trash_can", "police_car", "normal_car_1",
                       "normal_car_2", "sports_car", "sports_car_2", "suv", "taxi"}


def _record(**fields):
    base = {"intent": "find", "target_type": "mug", "landmark": "bench", "relation": "on",
            "pose_hint": None, "situation_hint": None, "scene_hint": None, "place_cues": []}
    return json.dumps({**base, **fields})


class SchemaTests(unittest.TestCase):
    def test_enums_are_the_organiser_catalogues(self):
        props = command_llm.schema()["properties"]
        self.assertEqual(ORGANISER_OBJECTS | {"person", "unknown"}, set(props["target_type"]["enum"]))
        self.assertEqual(ORGANISER_LANDMARKS, set(props["landmark"]["anyOf"][0]["enum"]))
        self.assertEqual({"near", "beside", "behind", "front", "left", "right", "on", "above"},
                         set(props["relation"]["anyOf"][0]["enum"]))
        self.assertEqual(9, len(props["pose_hint"]["anyOf"][0]["enum"]))
        self.assertEqual({"normal", "abnormal", "emergency", "accident"},
                         set(props["situation_hint"]["anyOf"][0]["enum"]))
        self.assertEqual({"find", "bring"}, set(props["intent"]["enum"]))
        self.assertFalse(command_llm.schema()["additionalProperties"])

    def test_unknown_classes_are_rejected(self):
        parsed, fixes = command_llm.to_parsed(json.loads(_record(target_type="laptop")), None)
        self.assertIsNone(parsed)


@mock.patch.dict(os.environ, {"MARC_COMMAND_PARSER": "qwen"})
class ReadCommandTests(unittest.TestCase):
    def test_a_valid_reply_is_used(self):
        reading = command_llm.read_command("Find the mug on the bench.", complete=lambda *a: _record())
        self.assertEqual("qwen", reading.source)
        self.assertEqual(("mug", "bench", Relation.ON),
                         (reading.parsed.target_type, reading.parsed.landmark, reading.parsed.relation))

    def test_a_landmark_the_text_does_not_name_is_dropped(self):
        reply = _record(target_type="tumbler", landmark="normal_car_1", relation=None,
                        scene_hint="parking", place_cues=["suv"])
        reading = command_llm.read_command(
            "I dropped my tumbler as I got out of my car in the parking lot.",
            complete=lambda *a: reply)
        self.assertIsNone(reading.parsed.landmark)
        self.assertEqual((), reading.parsed.place_cues)
        self.assertIn("landmark_not_in_text", reading.fixes)

    def test_left_behind_is_an_idiom_not_a_place(self):
        reply = _record(target_type="pencilcase", landmark="picnic_table", relation="behind")
        idiom = command_llm.read_command("I sat at the picnic table and left my pencil case behind.",
                                         complete=lambda *a: reply)
        place = command_llm.read_command("I left my pencil case behind the picnic table.",
                                         complete=lambda *a: reply)
        self.assertIsNone(idiom.parsed.relation)
        self.assertEqual(Relation.BEHIND, place.parsed.relation)

    def test_cross_field_rules_are_fixed_not_raised(self):
        reply = _record(target_type="mug", pose_hint="person_prone", landmark=None,
                        relation="on", place_cues=["bench"])
        reading = command_llm.read_command("whatever", complete=lambda *a: reply)
        self.assertIsNone(reading.parsed.relation)
        self.assertIn("pose_hint_on_object", reading.fixes)
        self.assertIn("relation_without_landmark", reading.fixes)

    def test_invalid_timeout_or_abstain_falls_back_to_rules(self):
        def timeout(*_):
            raise TimeoutError("slow")
        for complete in (lambda *a: "not json", timeout, lambda *a: _record(target_type="unknown")):
            with decision_scope("stage1", 1) as notes:
                reading = command_llm.read_command("Find the mug on the bench.",
                                                   expected_intent=Intent.FIND, complete=complete)
            self.assertEqual("rules", reading.source)
            self.assertEqual("mug", reading.parsed.target_type)
            self.assertTrue([n for n in notes if n["decision"] == "command_parse"])

    def test_the_rules_never_raise(self):
        for text in ("", "   ", "Find the laptop near the bench.", "Find the mug or tumbler.",
                     "asdf qwer", "Find the person walking normally but needed help."):
            reading = command_llm.read_command(text, complete=lambda *a: "{")
            self.assertIn(reading.source, {"rules", "partial"})

    def test_rules_correct_typos_only_when_a_word_was_not_understood(self):
        from marc_agent.command_parser import parse_command_lenient
        parsed, error = parse_command_lenient("Find the umberlla near the bench.",
                                              expected_intent=Intent.FIND)
        self.assertEqual(("umbrella", "bench", None), (parsed.target_type, parsed.landmark, error))
        for text, landmark, scene in (
            ("Find the mug on the bench near the beach.", "bench", None),
            ("Find the mug I dropped in the parking spot.", None, "parking_space"),
        ):
            parsed, _ = parse_command_lenient(text, expected_intent=Intent.FIND)
            self.assertEqual((landmark, scene), (parsed.landmark, parsed.scene_hint), text)

    def test_rules_mode_skips_qwen(self):
        with mock.patch.dict(os.environ, {"MARC_COMMAND_PARSER": "rules"}):
            reading = command_llm.read_command("Find the mug on the bench.",
                                               complete=lambda *a: self.fail("called Qwen"))
        self.assertEqual("rules", reading.source)


if __name__ == "__main__":
    unittest.main()
