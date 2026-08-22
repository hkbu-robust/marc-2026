"""Interface-level verification for the participant command parser."""

from __future__ import annotations

import os
import unittest

import yaml

from marc_agent.command_parser import (
    CommandParseError,
    Intent,
    LANDMARK_TYPES,
    MISSION_OBJECT_TYPES,
    ParseErrorCode,
    Relation,
    Situation,
    parse_command,
)


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "command_cases.yaml")


def _load_cases():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)["cases"]


def _as_dict(result):
    return {
        "intent": result.intent.value,
        "target_type": result.target_type,
        "landmark": result.landmark,
        "relation": result.relation.value if result.relation else None,
        "situation": result.situation.value if result.situation else None,
        "scene_hint": result.scene_hint,
    }


class CommandFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _load_cases()

    def test_fixture_size_and_provenance(self):
        practice = [case for case in self.cases if case["tier"] == "practice"]
        probes = [case for case in self.cases if case["tier"] != "practice"]
        self.assertEqual(39, len(practice))
        self.assertGreaterEqual(len(probes), 120)
        self.assertEqual(len(self.cases), len({case["id"] for case in self.cases}))
        self.assertTrue(
            {
                "direct", "aliases", "narrative", "reference", "morphology",
                "correction", "attachment", "sar", "surface", "disambiguation", "invalid",
            }.issubset({case["tier"] for case in probes})
        )

    def test_all_reviewed_commands(self):
        for case in self.cases:
            with self.subTest(case=case["id"], tier=case["tier"]):
                expected_intent = Intent(case["expected_intent"])
                expected_error = case.get("expected_error")
                if expected_error:
                    with self.assertRaises(CommandParseError) as caught:
                        parse_command(case["text"], expected_intent=expected_intent)
                    self.assertEqual(ParseErrorCode(expected_error), caught.exception.code)
                    continue
                result = parse_command(case["text"], expected_intent=expected_intent)
                self.assertEqual(case["expected"], _as_dict(result))
                if result.target_type == "person":
                    self.assertIsNone(result.relation)
                    self.assertIsNotNone(result.situation)
                else:
                    self.assertIsNone(result.situation)
                    if result.relation is not None:
                        self.assertIsNotNone(result.landmark)

    def test_fixture_covers_full_output_vocabulary(self):
        successful = [case["expected"] for case in self.cases if "expected" in case]
        self.assertEqual(
            MISSION_OBJECT_TYPES,
            {item["target_type"] for item in successful if item["target_type"] != "person"},
        )
        self.assertEqual(
            LANDMARK_TYPES,
            {item["landmark"] for item in successful if item["landmark"] is not None},
        )
        self.assertEqual(
            {item.value for item in Relation},
            {item["relation"] for item in successful if item["relation"] is not None},
        )
        self.assertEqual(
            {item.value for item in Situation},
            {item["situation"] for item in successful if item["situation"] is not None},
        )
        self.assertEqual(
            {item.value for item in Intent},
            {item["intent"] for item in successful},
        )

    def test_meaning_preserving_surface_mutations(self):
        bases = [case for case in self.cases if "expected" in case]
        transforms = (
            lambda text: text.upper(),
            lambda text: "   " + text + "   ",
            lambda text: text.replace(" ", "  "),
            lambda text: text.rstrip(".?!"),
        )
        for case in bases:
            expected = case["expected"]
            intent = Intent(case["expected_intent"])
            for transform in transforms:
                mutated = transform(case["text"])
                with self.subTest(case=case["id"], mutation=transform.__code__.co_firstlineno):
                    self.assertEqual(expected, _as_dict(parse_command(mutated, expected_intent=intent)))

    def test_approved_alias_mutations(self):
        substitutions = (
            ("pencil case", "pencilcase"),
            ("fire hydrant", "hydrant"),
            ("outdoor table", "picnic table"),
            ("drink machine", "vending machine"),
            ("bin", "trash can"),
            ("sunblock", "sunscreen"),
            ("post box", "postbox"),
            ("tumbler", "travel tumbler"),
        )
        successful = [case for case in self.cases if "expected" in case]
        for source, replacement in substitutions:
            case = next(case for case in successful if source in case["text"].casefold())
            start = case["text"].casefold().index(source)
            mutated = case["text"][:start] + replacement + case["text"][start + len(source):]
            with self.subTest(source=source, replacement=replacement, case=case["id"]):
                result = parse_command(mutated, expected_intent=Intent(case["expected_intent"]))
                self.assertEqual(case["expected"], _as_dict(result))

    def test_person_mention_is_not_automatically_sar(self):
        result = parse_command(
            "Someone left an umbrella near the bench; please find it.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("umbrella", result.target_type)
        self.assertEqual("bench", result.landmark)
        self.assertEqual(Relation.NEAR, result.relation)


if __name__ == "__main__":
    unittest.main()
