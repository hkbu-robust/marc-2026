"""Interface-level verification for the participant command parser."""

from __future__ import annotations

import os
import unittest
from dataclasses import asdict

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


def _assert_parsed(test, result, expected: dict):
    got = asdict(result)
    place_got = got.pop("place_cues")
    payload = dict(expected)
    if "place_cues" in payload:
        test.assertEqual(tuple(payload.pop("place_cues")), place_got)
    test.assertEqual(payload, got)


def _load_cases():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)["cases"]


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
                _assert_parsed(self, result, case["expected"])
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
                    _assert_parsed(
                        self,
                        parse_command(mutated, expected_intent=intent),
                        expected,
                    )

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
                _assert_parsed(self, result, case["expected"])

    def test_person_mention_is_not_automatically_sar(self):
        result = parse_command(
            "Someone left an umbrella near the bench; please find it.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("umbrella", result.target_type)
        self.assertEqual("bench", result.landmark)
        self.assertEqual(Relation.NEAR, result.relation)

    def test_possessive_person_mention_is_not_sar(self):
        cases = (
            ("Find someone's umbrella near the bench.", "umbrella", "bench", Relation.NEAR),
            ("Find the person's mug beside the taxi.", "mug", "taxi", Relation.BESIDE),
        )
        for text, target, landmark, relation in cases:
            with self.subTest(text=text):
                result = parse_command(text, expected_intent=Intent.FIND)
                self.assertEqual(target, result.target_type)
                self.assertEqual(landmark, result.landmark)
                self.assertEqual(relation, result.relation)

    def test_person_beneficiary_is_not_sar(self):
        cases = (
            ("Find for someone the umbrella near the bench.", "umbrella"),
            ("Please find for the person the mug beside the taxi.", "mug"),
        )
        for text, target in cases:
            with self.subTest(text=text):
                result = parse_command(text, expected_intent=Intent.FIND)
                self.assertEqual(target, result.target_type)

    def test_unrelated_location_mentions_are_not_target_facts(self):
        cases = (
            "I walked past the bench before losing my mug elsewhere. Please find the mug.",
            "I drove through the park before losing my mug elsewhere. Please find the mug.",
            "I rested at the park yesterday. Later I lost my mug elsewhere; please find the mug.",
            "I stood near the bench. Later I lost my mug elsewhere; please find the mug.",
            "I was in the park when my friend asked me to find the mug they lost elsewhere.",
        )
        for text in cases:
            with self.subTest(text=text):
                result = parse_command(text, expected_intent=Intent.FIND)
                self.assertIsNone(result.landmark)
                self.assertIsNone(result.scene_hint)

    def test_sar_scene_hint_does_not_describe_a_distractor(self):
        text = (
            "People are relaxing at the park and one is even lying on the grass; "
            "find the one near the trash can who is bent over and needs help."
        )
        result = parse_command(text, expected_intent=Intent.FIND)
        self.assertEqual("person", result.target_type)
        self.assertEqual("trash_can", result.landmark)
        self.assertEqual("park", result.scene_hint)

        distractor_cases = (
            (
                "Someone is waiting on the road; "
                "find the person near the bench who needs help.",
                "bench",
            ),
            (
                "A person is waiting in the parking lot; "
                "locate someone collapsed beside the postbox.",
                "postbox",
            ),
            (
                "A person is waiting on the road and another person is near the bench; "
                "find the one near the bench who needs help.",
                "bench",
            ),
            (
                "A person is in the park while another person is beside the postbox; "
                "locate the one beside the postbox who collapsed.",
                "postbox",
            ),
        )
        for command, landmark in distractor_cases:
            with self.subTest(command=command):
                parsed = parse_command(command, expected_intent=Intent.FIND)
                self.assertEqual(landmark, parsed.landmark)
                self.assertIsNone(parsed.scene_hint)

    def test_sar_scene_hint_follows_unambiguous_antecedent(self):
        cases = (
            ("A person collapsed in the parking lot; find them.", "parking"),
            ("Someone collapsed on the road; find them.", "road"),
            ("The person who needs help is in the park; find them.", "park"),
            ("A person collapsed in the parking lot; locate the person.", "parking"),
            ("Someone collapsed on the road; find that person.", "road"),
            ("One person is face-down and motionless in the parking lot; locate the one.", "parking"),
            ("A person collapsed in the parking lot; please locate them immediately.", "parking"),
            ("Someone collapsed on the road; could you find that person now?", "road"),
            ("A person collapsed in the parking lot; find the person who needs help.", "parking"),
            ("Someone collapsed on the road; locate the same person.", "road"),
            ("Someone collapsed on the road; locate the same person near the bench.", "road"),
            (
                "A person collapsed in the parking lot; "
                "find the same person beside the postbox.",
                "parking",
            ),
            ("One person collapsed in the parking lot; locate the one near the bench.", "parking"),
            (
                "One person is lying face-down and motionless in the parking lot; locate them.",
                "parking",
            ),
        )
        for text, scene_hint in cases:
            with self.subTest(text=text):
                result = parse_command(text, expected_intent=Intent.FIND)
                self.assertEqual("person", result.target_type)
                self.assertEqual(scene_hint, result.scene_hint)

    def test_ambiguous_person_antecedents_raise(self):
        cases = (
            "A person is walking normally and someone else is standing normally; find the one.",
            "A person collapsed and someone else needs help; find them.",
            "A person and someone are walking normally; locate the person.",
        )
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(CommandParseError) as caught:
                    parse_command(text, expected_intent=Intent.FIND)
                self.assertEqual(ParseErrorCode.AMBIGUOUS_TARGET, caught.exception.code)

        result = parse_command(
            "One person is walking normally in the park; find the one.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("person", result.target_type)
        self.assertEqual("park", result.scene_hint)

    def test_independent_black_box_regressions(self):
        result = parse_command(
            "Bring someone the umbrella behind the postbox.",
            expected_intent=Intent.BRING,
        )
        self.assertEqual("umbrella", result.target_type)
        self.assertEqual("postbox", result.landmark)
        self.assertEqual(Relation.BEHIND, result.relation)

        result = parse_command(
            "Locate the tissue for the person beside the bicycle.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("tissue", result.target_type)
        self.assertIsNone(result.landmark)
        self.assertIsNone(result.relation)

        with self.assertRaises(CommandParseError) as caught:
            parse_command(
                "Find the person walking normally but needing help.",
                expected_intent=Intent.FIND,
            )
        self.assertEqual(ParseErrorCode.CONFLICTING_SITUATION, caught.exception.code)

        result = parse_command(
            "The bench is near the taxi. Find the sunblock elsewhere.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("sunblock", result.target_type)
        self.assertIsNone(result.landmark)
        self.assertIsNone(result.relation)

        result = parse_command(
            "Find the PERSON—face down, completely still—on the road.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("person", result.target_type)
        self.assertEqual(Situation.ACCIDENT, result.situation)
        self.assertEqual("road", result.scene_hint)

    def test_major_target_evidence_regressions(self):
        with self.assertRaises(CommandParseError) as caught:
            parse_command(
                "A person is seated and someone is standing. Find the one near the bench "
                "while a third person is collapsed on the road.",
                expected_intent=Intent.FIND,
            )
        self.assertEqual(ParseErrorCode.UNKNOWN_SITUATION, caught.exception.code)

        cases = (
            ("Find the person walking normally but needed help.", ParseErrorCode.CONFLICTING_SITUATION),
            ("Find the person walking normally but requires help.", ParseErrorCode.CONFLICTING_SITUATION),
            ("Find the person walking normally but is in need of help.", ParseErrorCode.CONFLICTING_SITUATION),
        )
        for text, error_code in cases:
            with self.subTest(text=text):
                with self.assertRaises(CommandParseError) as caught:
                    parse_command(text, expected_intent=Intent.FIND)
                self.assertEqual(error_code, caught.exception.code)

        result = parse_command(
            "Bring the mug to someone collapsed on the road.",
            expected_intent=Intent.BRING,
        )
        self.assertEqual("mug", result.target_type)
        self.assertIsNone(result.scene_hint)

        sar_cases = (
            ("Find somebody face-down and immobile in a parking space.", Situation.ACCIDENT, "parking_space"),
            ("Find someone prone and motionless on the road.", Situation.ACCIDENT, "road"),
            ("Locate the person bending over, motionless, beside the hydrant.", Situation.EMERGENCY, None),
        )
        for text, situation, scene_hint in sar_cases:
            with self.subTest(text=text):
                result = parse_command(text, expected_intent=Intent.FIND)
                self.assertEqual(situation, result.situation)
                self.assertEqual(scene_hint, result.scene_hint)
                if "hydrant" in text:
                    self.assertEqual("hydrant", result.landmark)

        association_cases = (
            (
                "One person is fine near the taxi and another needs help on the road; "
                "locate the person on the road who needs help.",
                Situation.EMERGENCY,
                None,
                "road",
            ),
            (
                "One person fell on the road and another is walking normally near the bench. "
                "Find that person walking normally.",
                Situation.NORMAL,
                None,
                None,
            ),
        )
        for text, situation, landmark, scene_hint in association_cases:
            with self.subTest(text=text):
                result = parse_command(text, expected_intent=Intent.FIND)
                self.assertEqual(situation, result.situation)
                self.assertEqual(landmark, result.landmark)
                self.assertEqual(scene_hint, result.scene_hint)

        result = parse_command(
            "Find the mug. Beside the bench, a person is walking normally.",
            expected_intent=Intent.FIND,
        )
        self.assertIsNone(result.landmark)
        self.assertIsNone(result.relation)

        result = parse_command(
            "Near the bench in the park, find the person walking normally.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("bench", result.landmark)
        self.assertEqual("park", result.scene_hint)

        result = parse_command(
            "The bench near the taxi has my mug on it; please locate the mug.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual("bench", result.landmark)
        self.assertEqual(Relation.ON, result.relation)

        result = parse_command(
            "Bring the umbrella behind the postbox to the person beside the bicycle.",
            expected_intent=Intent.BRING,
        )
        self.assertEqual("umbrella", result.target_type)
        self.assertEqual("postbox", result.landmark)
        self.assertEqual(Relation.BEHIND, result.relation)

        result = parse_command(
            "Someone needs help on the road.",
            expected_intent=Intent.FIND,
        )
        self.assertEqual(Situation.EMERGENCY, result.situation)
        self.assertEqual("road", result.scene_hint)

    def test_parking_and_car_cues_use_vehicle_anchors_not_parser_cars(self):
        from marc_agent.command_parser import uses_vehicle_anchor

        parsed = parse_command(
            "I dropped my tumbler as I got out of my car in the parking lot.",
            expected_intent=Intent.FIND,
        )
        self.assertIsNone(parsed.landmark)
        self.assertEqual("parking", parsed.scene_hint)
        self.assertTrue(uses_vehicle_anchor(parsed, "I dropped my tumbler as I got out of my car in the parking lot."))

        floor = parse_command(
            "Pick up the tumbler dropped on the parking lot floor and bring it to the owner.",
            expected_intent=Intent.BRING,
        )
        self.assertIsNone(floor.landmark)
        self.assertEqual("parking", floor.scene_hint)
        self.assertFalse(
            uses_vehicle_anchor(
                floor,
                "Pick up the tumbler dropped on the parking lot floor and bring it to the owner.",
            )
        )

        empty = parse_command(
            "Find the tumbler dropped in the empty parking space.",
            expected_intent=Intent.FIND,
        )
        self.assertIsNone(empty.landmark)
        self.assertEqual("parking_space", empty.scene_hint)
        self.assertFalse(
            uses_vehicle_anchor(
                empty,
                "Find the tumbler dropped in the empty parking space.",
            )
        )
        from marc_agent.command_parser import implicit_anchor_classes

        from marc_agent.command_parser import PARKED_VEHICLE_LANDMARKS

        self.assertEqual(
            PARKED_VEHICLE_LANDMARKS,
            implicit_anchor_classes(
                empty, "Find the tumbler dropped in the empty parking space."
            ),
        )

    def test_any_vehicle_class_may_be_parked(self):
        from marc_agent.command_parser import (
            PARKED_VEHICLE_LANDMARKS,
            VEHICLE_LANDMARKS,
        )

        self.assertEqual(VEHICLE_LANDMARKS, PARKED_VEHICLE_LANDMARKS)
        self.assertIn("taxi", PARKED_VEHICLE_LANDMARKS)


class PlaceCueAndFallenOffTests(unittest.TestCase):
    def test_taxi_is_a_place_cue_when_bench_is_the_support(self):
        result = parse_command(
            "Find the person lying face-down on the bench near the taxi."
        )
        self.assertEqual("bench", result.landmark)
        self.assertEqual(("taxi",), result.place_cues)

    def test_fallen_off_attaches_the_support(self):
        result = parse_command("Find the person who fell off the bench.")
        self.assertEqual("person", result.target_type)
        self.assertEqual("bench", result.landmark)
        self.assertEqual((), result.place_cues)


if __name__ == "__main__":
    unittest.main()
