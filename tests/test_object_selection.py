"""Tests for lost-item object ranking across cameras and relations."""

from __future__ import annotations

import unittest

from marc_agent.command_parser import Intent, ParsedCommand, parse_command
from marc_agent.perception.object_selection import (
    build_object_board,
    format_object_verifier_prompt,
    parse_object_verifier_output,
    rank_object_candidates,
    select_object_candidate,
)
from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate


def _object(
    candidate_id: str,
    class_name: str,
    box,
    camera_id: str,
    confidence: float = 0.8,
) -> ObjectCandidate:
    return ObjectCandidate(
        candidate_id, class_name, box, confidence, camera_id, (640, 480)
    )


def _person(
    candidate_id: str,
    box,
    camera_id: str,
    pose: str = "person_sitting",
    confidence: float = 0.9,
) -> PersonCandidate:
    return PersonCandidate(
        candidate_id, box, pose, confidence, camera_id, (640, 480)
    )


def _on_table_vs_beside_bench() -> CctvDetectionRound:
    return CctvDetectionRound(
        objects=[
            _object(
                "rig_a:mug:0",
                "mug",
                (280.0, 200.0, 340.0, 255.0),
                "rig_a",
                0.7,
            ),
            _object(
                "rig_a:picnic_table:0",
                "picnic_table",
                (200.0, 250.0, 450.0, 400.0),
                "rig_a",
            ),
            _object(
                "rig_b:mug:0",
                "mug",
                (270.0, 300.0, 330.0, 360.0),
                "rig_b",
                0.99,
            ),
            _object(
                "rig_b:bench:0",
                "bench",
                (100.0, 300.0, 250.0, 400.0),
                "rig_b",
            ),
        ],
        camera_ids=["rig_a", "rig_b"],
    )


def _downward_projector(camera_id: str):
    from marc_agent.perception.cctv_project import (
        CameraGeometry,
        CctvProjector,
        static_lookup,
    )

    return CctvProjector(
        static_lookup(
            {
                camera_id: CameraGeometry(
                    fx=500.0,
                    fy=500.0,
                    cx=320.0,
                    cy=240.0,
                    rotation=(
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    ),
                    translation=(0.0, 0.0, -10.0),
                    ground_height=0.0,
                )
            }
        )
    )


class ObjectSelectionTests(unittest.TestCase):
    def test_on_prefers_mug_sitting_on_picnic_table(self):
        parsed = parse_command("Find the mug on the picnic table.")
        selected = select_object_candidate(parsed, _on_table_vs_beside_bench())
        self.assertIsNotNone(selected)
        self.assertEqual("rig_a:mug:0", selected.target.candidate_id)
        self.assertEqual("picnic_table", selected.anchor.class_name)
        self.assertEqual("object_relation", selected.method)

    def test_same_camera_pairs_only(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:mug:0", "mug", (280.0, 200.0, 340.0, 255.0), "rig_a"),
                _object(
                    "rig_b:picnic_table:0",
                    "picnic_table",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_b",
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )
        selected = select_object_candidate(parsed, round_)
        self.assertEqual("rig_a:mug:0", selected.target.candidate_id)
        self.assertIsNone(selected.anchor)
        self.assertEqual("target", selected.match)

    def test_no_landmark_takes_highest_confidence_tumbler(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_b:tumbler:0",
                    "tumbler",
                    (40.0, 40.0, 80.0, 120.0),
                    "rig_b",
                    0.4,
                ),
                _object(
                    "rig_a:tumbler:1",
                    "tumbler",
                    (100.0, 80.0, 140.0, 160.0),
                    "rig_a",
                    0.91,
                ),
                _object(
                    "rig_a:mug:0",
                    "mug",
                    (200.0, 80.0, 240.0, 140.0),
                    "rig_a",
                    0.99,
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )
        selected = select_object_candidate(parsed, round_)
        self.assertEqual("rig_a:tumbler:1", selected.target.candidate_id)
        self.assertIsNone(selected.anchor)

    def test_tie_breaks_on_candidate_id(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_z:tumbler:9", "tumbler", (10.0, 10.0, 20.0, 20.0), "rig_z", 0.8),
                _object("rig_a:tumbler:0", "tumbler", (30.0, 10.0, 40.0, 20.0), "rig_a", 0.8),
            ]
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:tumbler:0", ranked[0].target.candidate_id)

    def test_verifier_overrides_geometry_favorite(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:mug:0",
                    "mug",
                    (280.0, 200.0, 340.0, 255.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:mug:1",
                    "mug",
                    (20.0, 20.0, 50.0, 40.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:picnic_table:1",
                    "picnic_table",
                    (10.0, 45.0, 80.0, 90.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        geometry = select_object_candidate(parsed, round_)
        self.assertEqual("rig_a:mug:0", geometry.target.candidate_id)
        with self.assertLogs("marc_agent.object_selection", level="INFO") as captured:
            selected = select_object_candidate(
                parsed,
                round_,
                command_text="Find the mug on the picnic table.",
                verifier=lambda board: "rig_a:mug:1|rig_a:picnic_table:1",
            )
        self.assertEqual("rig_a:mug:1", selected.target.candidate_id)
        self.assertEqual("rig_a:picnic_table:1", selected.anchor.candidate_id)
        self.assertEqual("qwen", selected.method)
        self.assertIn("output=rig_a:mug:1|rig_a:picnic_table:1", "\n".join(captured.output))

    def test_invalid_verifier_output_falls_back_to_geometry(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:mug:0",
                    "mug",
                    (280.0, 200.0, 340.0, 255.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:mug:1",
                    "mug",
                    (20.0, 20.0, 50.0, 40.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:picnic_table:1",
                    "picnic_table",
                    (10.0, 45.0, 80.0, 90.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        with self.assertLogs("marc_agent.object_selection", level="INFO") as captured:
            selected = select_object_candidate(
                parsed,
                round_,
                command_text="Find the mug on the picnic table.",
                verifier=lambda board: "not a pair",
            )
        self.assertEqual("rig_a:mug:0", selected.target.candidate_id)
        self.assertEqual("object_relation", selected.method)
        self.assertIn("output=not a pair", "\n".join(captured.output))

    def test_single_pair_logs_skip_reason_and_does_not_call_verifier(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (40.0, 40.0, 80.0, 120.0),
                    "rig_a",
                    0.9,
                ),
            ],
            camera_ids=["rig_a"],
        )
        called = []
        with self.assertLogs("marc_agent.object_selection", level="INFO") as logged:
            selected = select_object_candidate(
                parsed,
                round_,
                command_text="Find the tumbler.",
                verifier=lambda board: called.append(1) or "nope",
            )
        self.assertEqual("rig_a:tumbler:0", selected.target.candidate_id)
        self.assertFalse(called)
        self.assertIn("verify=False reason=single_pair", "\n".join(logged.output))

    def test_no_landmark_verifier_picks_lower_confidence_tumbler(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_b:tumbler:0",
                    "tumbler",
                    (40.0, 40.0, 80.0, 120.0),
                    "rig_b",
                    0.4,
                ),
                _object(
                    "rig_a:tumbler:1",
                    "tumbler",
                    (100.0, 80.0, 140.0, 160.0),
                    "rig_a",
                    0.91,
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )
        selected = select_object_candidate(
            parsed,
            round_,
            command_text="Find the tumbler.",
            verifier=lambda board: "rig_b:tumbler:0|none",
        )
        self.assertEqual("rig_b:tumbler:0", selected.target.candidate_id)
        self.assertIsNone(selected.anchor)

    def test_relationless_pairs_nearest_landmark_and_calls_verifier(self):
        parsed = parse_command("Find the sunblock I left")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:sunblock:0",
                    "sunblock",
                    (200.0, 200.0, 240.0, 250.0),
                    "rig_a",
                    0.9,
                ),
                _object(
                    "rig_a:bench:0",
                    "bench",
                    (180.0, 240.0, 280.0, 360.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (10.0, 10.0, 80.0, 80.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual(1, len(ranked))
        self.assertEqual("rig_a:sunblock:0", ranked[0].target.candidate_id)
        self.assertEqual("rig_a:bench:0", ranked[0].anchor.candidate_id)
        calls = []
        with self.assertLogs("marc_agent.object_selection", level="INFO") as logged:
            selected = select_object_candidate(
                parsed,
                round_,
                command_text="Find the sunblock I left",
                verifier=lambda board: (
                    calls.append(board)
                    or "rig_a:sunblock:0|rig_a:bench:0"
                ),
            )
        self.assertEqual(1, len(calls))
        self.assertEqual("qwen", selected.method)
        self.assertEqual("rig_a:bench:0", selected.anchor.candidate_id)
        self.assertIn("verify=True reason=ok", "\n".join(logged.output))
        self.assertIn("Parsed relation: near", calls[0].prompt)

    def test_relationless_two_landmarks_nearer_wins(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (300.0, 200.0, 340.0, 250.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:hydrant:0",
                    "hydrant",
                    (40.0, 40.0, 90.0, 120.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:bench:0",
                    "bench",
                    (280.0, 240.0, 380.0, 360.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:bench:0", ranked[0].anchor.candidate_id)

    def test_relationless_verifier_overrides_nearest_pair(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (200.0, 200.0, 240.0, 250.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:tumbler:1",
                    "tumbler",
                    (40.0, 40.0, 80.0, 80.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:bench:0",
                    "bench",
                    (180.0, 240.0, 280.0, 360.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (10.0, 10.0, 90.0, 90.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual(2, len(ranked))
        other = ranked[1]
        selected = select_object_candidate(
            parsed,
            round_,
            command_text="Find the tumbler.",
            verifier=lambda _board: (
                f"{other.target.candidate_id}|{other.anchor.candidate_id}"
            ),
        )
        self.assertEqual(other.target.candidate_id, selected.target.candidate_id)
        self.assertEqual(other.anchor.candidate_id, selected.anchor.candidate_id)
        self.assertEqual("qwen", selected.method)

    def test_verifier_target_only_uses_best_ranked_pair(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = _on_table_vs_beside_bench()
        ranked = rank_object_candidates(parsed, round_)
        target_id, anchor_id = parse_object_verifier_output("rig_a:mug:0", ranked)
        self.assertEqual("rig_a:mug:0", target_id)
        self.assertEqual("rig_a:picnic_table:0", anchor_id)

    def test_parking_lot_car_cue_does_not_invent_a_vehicle_anchor(self):
        text = (
            "I think I dropped my tumbler on the ground as I got out of my car "
            "in the parking lot."
        )
        parsed = parse_command(text, expected_intent=Intent.FIND)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_7_a:tumbler:2",
                    "tumbler",
                    (40.0, 40.0, 80.0, 80.0),
                    "rig_7_a",
                    0.99,
                ),
                _object(
                    "rig_7_a:tumbler:3",
                    "tumbler",
                    (200.0, 300.0, 240.0, 340.0),
                    "rig_7_a",
                    0.4,
                ),
                _object(
                    "rig_7_a:suv:0",
                    "suv",
                    (180.0, 280.0, 400.0, 420.0),
                    "rig_7_a",
                    0.9,
                ),
            ],
            camera_ids=["rig_7_a"],
        )
        ranked = rank_object_candidates(parsed, round_, command_text=text)
        self.assertTrue(all(item.anchor is None for item in ranked))
        self.assertEqual("rig_7_a:tumbler:2", ranked[0].target.candidate_id)
        target_id, anchor_id = parse_object_verifier_output(
            "rig_7_a:tumbler:3|none", ranked
        )
        self.assertEqual("rig_7_a:tumbler:3", target_id)
        self.assertIsNone(anchor_id)
        calls = []
        selected = select_object_candidate(
            parsed,
            round_,
            command_text=text,
            verifier=lambda board: calls.append(board) or "rig_7_a:tumbler:3|none",
        )
        self.assertEqual([], calls)
        self.assertEqual("rig_7_a:tumbler:2", selected.target.candidate_id)
        self.assertIsNone(selected.anchor)

    def test_verifier_truncated_prefix_resolves_unique_target(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = _on_table_vs_beside_bench()
        ranked = rank_object_candidates(parsed, round_)
        target_id, anchor_id = parse_object_verifier_output("rig_a:mug:", ranked)
        self.assertEqual("rig_a:mug:0", target_id)
        self.assertEqual("rig_a:picnic_table:0", anchor_id)

    def test_verifier_thinking_trace_uses_last_valid_pair(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = _on_table_vs_beside_bench()
        ranked = rank_object_candidates(parsed, round_)
        target_id, anchor_id = parse_object_verifier_output(
            "<think>rig_b:mug:0|rig_b:bench:0 looks closer</think>\n"
            "The match is rig_a:mug:0|rig_a:picnic_table:0",
            ranked,
        )
        self.assertEqual("rig_a:mug:0", target_id)
        self.assertEqual("rig_a:picnic_table:0", anchor_id)

    def test_tumbler_on_bench_ignores_mug(self):
        parsed = parse_command("Find the tumbler on the bench.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_3:tumbler:0",
                    "tumbler",
                    (10.0, 300.0, 50.0, 360.0),
                    "rig_3",
                    0.95,
                ),
                _object(
                    "rig_3:bench:0",
                    "bench",
                    (200.0, 300.0, 400.0, 420.0),
                    "rig_3",
                ),
                _object(
                    "rig_1:mug:0",
                    "mug",
                    (280.0, 200.0, 340.0, 255.0),
                    "rig_1",
                    0.7,
                ),
                _object(
                    "rig_1:bench:0",
                    "bench",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_1",
                ),
            ],
            camera_ids=["rig_3", "rig_1"],
        )
        ranked = rank_object_candidates(parsed, round_)
        ids = [item.target.candidate_id for item in ranked]
        self.assertEqual(["rig_3:tumbler:0"], ids)

    def test_mug_command_does_not_alias_tumbler(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (280.0, 200.0, 340.0, 255.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        self.assertEqual((), rank_object_candidates(parsed, round_))

    def test_single_pair_does_not_call_verifier(self):
        parsed = parse_command("Find the tumbler on the bench.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (280.0, 200.0, 340.0, 255.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:bench:0",
                    "bench",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        calls = []
        selected = select_object_candidate(
            parsed,
            round_,
            verifier=lambda board: calls.append(board) or "x|y",
        )
        self.assertEqual([], calls)
        self.assertEqual("rig_a:tumbler:0", selected.target.candidate_id)
        self.assertEqual("object_relation", selected.method)

    def test_near_penalizes_target_sitting_on_landmark(self):
        parsed = parse_command("Find the tumbler near the fire hydrant.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (210.0, 180.0, 270.0, 350.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:tumbler:1",
                    "tumbler",
                    (300.0, 300.0, 340.0, 380.0),
                    "rig_a",
                    0.6,
                ),
                _object(
                    "rig_a:hydrant:0",
                    "hydrant",
                    (200.0, 200.0, 280.0, 420.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:tumbler:1", ranked[0].target.candidate_id)


    def test_parking_floor_drop_does_not_pair_with_vehicle(self):
        parsed = parse_command(
            "Pick up the tumbler dropped on the parking lot floor and bring it to the owner.",
        )
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (40.0, 300.0, 80.0, 380.0),
                    "rig_a",
                    0.5,
                ),
                _object(
                    "rig_a:police_car:0",
                    "police_car",
                    (90.0, 280.0, 400.0, 420.0),
                    "rig_a",
                    0.9,
                ),
                _object(
                    "rig_b:tumbler:0",
                    "tumbler",
                    (200.0, 200.0, 240.0, 280.0),
                    "rig_b",
                    0.99,
                ),
                _object(
                    "rig_b:bench:0",
                    "bench",
                    (180.0, 300.0, 420.0, 430.0),
                    "rig_b",
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertTrue(ranked)
        self.assertIsNone(ranked[0].anchor)
        self.assertTrue(all(item.anchor is None for item in ranked))
        self.assertEqual("rig_a:tumbler:0", ranked[0].target.candidate_id)

    def test_world_on_prefers_contact_inside_bench_footprint(self):
        from marc_agent.perception.cctv_project import (
            CameraGeometry,
            CctvProjector,
            static_lookup,
        )

        parsed = parse_command("Find the tumbler on the bench.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (10.0, 150.0, 400.0, 210.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:bench:0",
                    "bench",
                    (250.0, 200.0, 480.0, 400.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:tumbler:1",
                    "tumbler",
                    (180.0, 200.0, 230.0, 255.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:bench:1",
                    "bench",
                    (200.0, 250.0, 450.0, 410.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        image_ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:tumbler:0", image_ranked[0].target.candidate_id)
        projector = CctvProjector(
            static_lookup(
                {
                    "rig_a": CameraGeometry(
                        fx=500.0,
                        fy=500.0,
                        cx=320.0,
                        cy=240.0,
                        rotation=(
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                            (0.0, 0.0, 1.0),
                        ),
                        translation=(0.0, 0.0, -10.0),
                        ground_height=0.0,
                    )
                }
            )
        )
        world_ranked = rank_object_candidates(parsed, round_, projector=projector)
        self.assertEqual("rig_a:tumbler:1", world_ranked[0].target.candidate_id)

    def test_world_near_prefers_closer_tumbler_to_hydrant(self):
        from marc_agent.perception.cctv_project import (
            CameraGeometry,
            CctvProjector,
            static_lookup,
        )

        parsed = parse_command("Find the tumbler near the fire hydrant in the parking lot.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:2",
                    "tumbler",
                    (20.0, 200.0, 50.0, 240.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:tumbler:1",
                    "tumbler",
                    (300.0, 250.0, 330.0, 280.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:hydrant:0",
                    "hydrant",
                    (280.0, 240.0, 360.0, 400.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        projector = CctvProjector(
            static_lookup(
                {
                    "rig_a": CameraGeometry(
                        fx=500.0,
                        fy=500.0,
                        cx=320.0,
                        cy=240.0,
                        rotation=(
                            (1.0, 0.0, 0.0),
                            (0.0, 1.0, 0.0),
                            (0.0, 0.0, 1.0),
                        ),
                        translation=(0.0, 0.0, -10.0),
                        ground_height=0.0,
                    )
                }
            )
        )
        ranked = rank_object_candidates(
            parsed,
            round_,
            projector=projector,
            command_text="Find the tumbler near the fire hydrant in the parking lot.",
        )
        self.assertEqual("rig_a:tumbler:1", ranked[0].target.candidate_id)
        self.assertEqual("hydrant", ranked[0].anchor.class_name)

    def test_on_bench_beats_pavement_tumbler_that_only_shares_a_bench_camera(self):
        parsed = parse_command("Find the tumbler on the bench.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_3:tumbler:0",
                    "tumbler",
                    (10.0, 300.0, 50.0, 360.0),
                    "rig_3",
                    0.99,
                ),
                _object(
                    "rig_3:bench:0",
                    "bench",
                    (200.0, 300.0, 400.0, 420.0),
                    "rig_3",
                ),
                _object(
                    "rig_1:tumbler:0",
                    "tumbler",
                    (280.0, 200.0, 340.0, 255.0),
                    "rig_1",
                    0.25,
                ),
                _object(
                    "rig_1:bench:0",
                    "bench",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_1",
                ),
            ],
            camera_ids=["rig_3", "rig_1"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_1:tumbler:0", ranked[0].target.candidate_id)
        self.assertEqual("strict", ranked[0].match)
        self.assertTrue(all(item.match == "strict" for item in ranked))

    def test_empty_parking_space_ranks_against_vehicles(self):
        parsed = parse_command("Find the tumbler dropped in the empty parking space.")
        self.assertIsNone(parsed.landmark)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (100.0, 200.0, 130.0, 240.0),
                    "rig_a",
                    0.9,
                ),
                _object(
                    "rig_a:police_car:0",
                    "police_car",
                    (90.0, 180.0, 400.0, 420.0),
                    "rig_a",
                    0.9,
                ),
                _object(
                    "rig_a:tumbler:1",
                    "tumbler",
                    (500.0, 400.0, 530.0, 440.0),
                    "rig_a",
                    0.3,
                ),
                _object(
                    "rig_a:normal_car_1:0",
                    "normal_car_1",
                    (10.0, 10.0, 80.0, 80.0),
                    "rig_a",
                    0.99,
                ),
            ],
            people=[
                _person(
                    "rig_a:person_lying_down:0",
                    (70.0, 160.0, 220.0, 300.0),
                    "rig_a",
                    "person_lying_down",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(
            parsed,
            round_,
            command_text="Find the tumbler dropped in the empty parking space.",
        )
        self.assertEqual("rig_a:tumbler:1", ranked[0].target.candidate_id)
        self.assertIsNone(ranked[0].anchor)

    def test_same_camera_floor_drop_prefers_lower_box(self):
        text = (
            "Pick up the tumbler dropped on the parking lot floor "
            "and bring it to the owner."
        )
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam:tumbler:1",
                    "tumbler",
                    (200.0, 40.0, 280.0, 140.0),
                    "cam",
                    0.99,
                ),
                _object(
                    "cam:tumbler:0",
                    "tumbler",
                    (40.0, 320.0, 90.0, 410.0),
                    "cam",
                    0.4,
                ),
            ],
            camera_ids=["cam"],
        )
        ranked = rank_object_candidates(parsed, round_, command_text=text)
        self.assertEqual("cam:tumbler:0", ranked[0].target.candidate_id)
        self.assertIsNone(ranked[0].anchor)
        calls = []
        selected = select_object_candidate(
            parsed,
            round_,
            command_text=text,
            verifier=lambda board: calls.append(board) or "cam:tumbler:0|none",
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("cam:tumbler:0", selected.target.candidate_id)
        self.assertEqual("qwen", selected.method)

    def test_parking_occupancy_prefers_off_road_instance(self):
        text = "Find the tumbler in the parking lot."
        parsed = parse_command(text)
        self.assertEqual("parking", parsed.scene_hint)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam:tumbler:1",
                    "tumbler",
                    (400.0, 300.0, 460.0, 400.0),
                    "cam",
                    0.99,
                ),
                _object(
                    "cam:tumbler:0",
                    "tumbler",
                    (40.0, 80.0, 80.0, 140.0),
                    "cam",
                    0.4,
                ),
            ],
            camera_ids=["cam"],
        )
        projector = _downward_projector("cam")
        road = projector.project_contact("cam", (400.0, 300.0, 460.0, 400.0))
        lot = projector.project_contact("cam", (40.0, 80.0, 80.0, 140.0))
        self.assertIsNotNone(road)
        self.assertIsNotNone(lot)

        def occupancy_at(x, y):
            d_road = (x - road[0]) ** 2 + (y - road[1]) ** 2
            d_lot = (x - lot[0]) ** 2 + (y - lot[1]) ** 2
            return 0 if d_road <= d_lot else -1

        ranked = rank_object_candidates(
            parsed,
            round_,
            projector=projector,
            command_text=text,
            occupancy_at=occupancy_at,
        )
        self.assertEqual("cam:tumbler:0", ranked[0].target.candidate_id)
        self.assertIsNone(ranked[0].anchor)

    def test_place_cues_without_landmark_keep_cue_camera(self):
        parsed = ParsedCommand(
            intent=Intent.FIND,
            target_type="tumbler",
            place_cues=("taxi",),
        )
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam_a:tumbler:0",
                    "tumbler",
                    (40.0, 300.0, 80.0, 380.0),
                    "cam_a",
                    0.4,
                ),
                _object(
                    "cam_a:taxi:0",
                    "taxi",
                    (90.0, 280.0, 400.0, 420.0),
                    "cam_a",
                ),
                _object(
                    "cam_b:tumbler:0",
                    "tumbler",
                    (200.0, 200.0, 240.0, 280.0),
                    "cam_b",
                    0.99,
                ),
                _object(
                    "cam_b:bench:0",
                    "bench",
                    (180.0, 300.0, 420.0, 430.0),
                    "cam_b",
                ),
            ],
            camera_ids=["cam_a", "cam_b"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("cam_a:tumbler:0", ranked[0].target.candidate_id)

    def test_floor_language_does_not_override_on_table(self):
        text = "Find the mug dropped on the picnic table."
        parsed = parse_command(text)
        selected = select_object_candidate(
            parsed, _on_table_vs_beside_bench(), command_text=text
        )
        self.assertIsNotNone(selected)
        self.assertEqual("rig_a:mug:0", selected.target.candidate_id)
        self.assertEqual("picnic_table", selected.anchor.class_name)

    def test_verifier_prompt_matches_place_not_confidence(self):
        prompt = format_object_verifier_prompt(
            "Pick up the tumbler dropped on the parking lot floor.",
            "tumbler",
            None,
            None,
            ["cam:tumbler:0 | camera=cam"],
            scene_hint="parking",
        )
        self.assertIn("Do not prefer higher confidence", prompt)
        self.assertIn("Occupancy 0 is drivable road", prompt)
        self.assertIn("stall is vacant", prompt)
        self.assertIn("Parked cars (not a distant taxi)", prompt)
        self.assertIn("first listed id", prompt)
        self.assertNotIn("closer to the camera", prompt)

    def test_object_board_includes_world_xy_and_occupancy(self):
        text = "Find the tumbler in the parking lot."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam:tumbler:0",
                    "tumbler",
                    (40.0, 80.0, 80.0, 140.0),
                    "cam",
                    0.4,
                ),
            ],
            camera_ids=["cam"],
        )
        ranked = rank_object_candidates(parsed, round_, command_text=text)
        board = build_object_board(
            text,
            parsed,
            round_,
            ranked,
            max_crops=0,
            projector=_downward_projector("cam"),
            occupancy_at=lambda x, y: -1,
        )
        self.assertIn("world_xy=", board.prompt)
        self.assertIn("occupancy=-1", board.prompt)
        self.assertIn("owned=", board.prompt)
        self.assertIn("match=", board.prompt)
        self.assertIn("stall=", board.prompt)

    def test_empty_parking_two_instances_always_call_verifier(self):
        text = "Find the tumbler dropped in the empty parking space."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam:tumbler:0",
                    "tumbler",
                    (40.0, 80.0, 80.0, 140.0),
                    "cam",
                    0.9,
                ),
                _object(
                    "cam:tumbler:1",
                    "tumbler",
                    (400.0, 300.0, 440.0, 360.0),
                    "cam",
                    0.4,
                ),
            ],
            camera_ids=["cam"],
        )
        calls = []
        selected = select_object_candidate(
            parsed,
            round_,
            command_text=text,
            occupancy_at=lambda _x, _y: 0,
            verifier=lambda board: calls.append(board) or "cam:tumbler:1|none",
        )
        self.assertEqual(1, len(calls))
        self.assertIn("stall=", calls[0].prompt)
        self.assertEqual("cam:tumbler:1", selected.target.candidate_id)
        self.assertEqual("qwen", selected.method)

    def test_unique_empty_stall_skips_verifier(self):
        text = "Find the tumbler dropped in the empty parking space."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam:tumbler:0",
                    "tumbler",
                    (100.0, 200.0, 130.0, 240.0),
                    "cam",
                    0.9,
                ),
                _object(
                    "cam:police_car:0",
                    "police_car",
                    (90.0, 180.0, 400.0, 420.0),
                    "cam",
                    0.9,
                ),
                _object(
                    "cam:tumbler:1",
                    "tumbler",
                    (500.0, 400.0, 530.0, 440.0),
                    "cam",
                    0.3,
                ),
            ],
            camera_ids=["cam"],
        )
        calls = []
        selected = select_object_candidate(
            parsed,
            round_,
            command_text=text,
            occupancy_at=lambda _x, _y: 0,
            verifier=lambda board: calls.append(board) or "cam:tumbler:0|none",
        )
        self.assertEqual([], calls)
        self.assertEqual("cam:tumbler:1", selected.target.candidate_id)

    def test_parking_lot_camera_beats_park_camera_with_taxi(self):
        text = "Find the tumbler dropped in the empty parking space."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam_park:tumbler:0",
                    "tumbler",
                    (40.0, 300.0, 80.0, 380.0),
                    "cam_park",
                    0.99,
                ),
                _object(
                    "cam_park:taxi:0",
                    "taxi",
                    (90.0, 280.0, 400.0, 420.0),
                    "cam_park",
                ),
                _object(
                    "cam_park:bench:0",
                    "bench",
                    (20.0, 200.0, 200.0, 280.0),
                    "cam_park",
                ),
                _object(
                    "cam_lot:tumbler:0",
                    "tumbler",
                    (40.0, 80.0, 80.0, 140.0),
                    "cam_lot",
                    0.4,
                ),
                _object(
                    "cam_lot:suv:0",
                    "suv",
                    (200.0, 200.0, 400.0, 400.0),
                    "cam_lot",
                    0.9,
                ),
                _object(
                    "cam_lot:normal_car_1:0",
                    "normal_car_1",
                    (420.0, 40.0, 520.0, 120.0),
                    "cam_lot",
                    0.9,
                ),
            ],
            camera_ids=["cam_park", "cam_lot"],
        )
        ranked = rank_object_candidates(
            parsed,
            round_,
            command_text=text,
            occupancy_at=lambda _x, _y: 0,
        )
        self.assertEqual("cam_lot:tumbler:0", ranked[0].target.candidate_id)
        self.assertTrue(all(item.target.camera_id == "cam_lot" for item in ranked))

    def test_more_parked_cars_beats_higher_confidence_when_occupancy_ties(self):
        text = "Find the tumbler in the parking lot."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "cam_a:tumbler:0",
                    "tumbler",
                    (40.0, 300.0, 80.0, 380.0),
                    "cam_a",
                    0.99,
                ),
                _object(
                    "cam_a:suv:0",
                    "suv",
                    (200.0, 40.0, 280.0, 120.0),
                    "cam_a",
                ),
                _object(
                    "cam_b:tumbler:0",
                    "tumbler",
                    (40.0, 80.0, 80.0, 140.0),
                    "cam_b",
                    0.4,
                ),
                _object(
                    "cam_b:suv:0",
                    "suv",
                    (200.0, 200.0, 300.0, 300.0),
                    "cam_b",
                ),
                _object(
                    "cam_b:police_car:0",
                    "police_car",
                    (320.0, 40.0, 420.0, 140.0),
                    "cam_b",
                ),
            ],
            camera_ids=["cam_a", "cam_b"],
        )
        ranked = rank_object_candidates(
            parsed,
            round_,
            command_text=text,
            occupancy_at=lambda _x, _y: 0,
        )
        self.assertEqual("cam_b:tumbler:0", ranked[0].target.candidate_id)

    def test_unique_tumbler_beside_sitter_is_submitted(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (80.0, 260.0, 140.0, 340.0),
                    "rig_a",
                ),
            ],
            people=[
                _person(
                    "rig_a:person_sitting:0",
                    (40.0, 180.0, 200.0, 400.0),
                    "rig_a",
                    "person_sitting",
                ),
            ],
            camera_ids=["rig_a"],
        )
        selected = select_object_candidate(parsed, round_)
        self.assertIsNotNone(selected)
        self.assertEqual("rig_a:tumbler:0", selected.target.candidate_id)

    def test_two_cameras_one_tumbler_still_submits_when_possessed(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:0",
                    "tumbler",
                    (80.0, 260.0, 140.0, 340.0),
                    "rig_a",
                ),
                _object(
                    "rig_b:bench:0",
                    "bench",
                    (100.0, 200.0, 300.0, 360.0),
                    "rig_b",
                ),
            ],
            people=[
                _person(
                    "rig_a:person_walking:0",
                    (40.0, 180.0, 200.0, 400.0),
                    "rig_a",
                    "person_walking",
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )
        selected = select_object_candidate(parsed, round_)
        self.assertEqual("rig_a:tumbler:0", selected.target.candidate_id)

    def test_three_tumblers_prefers_unattended_foreground(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:owned",
                    "tumbler",
                    (80.0, 280.0, 130.0, 340.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:tumbler:far",
                    "tumbler",
                    (500.0, 20.0, 520.0, 45.0),
                    "rig_a",
                    0.95,
                ),
                _object(
                    "rig_a:tumbler:car",
                    "tumbler",
                    (400.0, 260.0, 470.0, 360.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:taxi:0",
                    "taxi",
                    (360.0, 180.0, 620.0, 430.0),
                    "rig_a",
                ),
            ],
            people=[
                _person(
                    "rig_a:person_lying_down:0",
                    (20.0, 200.0, 180.0, 400.0),
                    "rig_a",
                    "person_lying_down",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:tumbler:car", ranked[0].target.candidate_id)

    def test_clustered_unattended_prefers_closer_to_camera(self):
        parsed = parse_command("Find the tumbler.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:tumbler:near",
                    "tumbler",
                    (200.0, 240.0, 320.0, 400.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:tumbler:far",
                    "tumbler",
                    (80.0, 80.0, 110.0, 120.0),
                    "rig_a",
                    0.99,
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:tumbler:near", ranked[0].target.candidate_id)

    def test_on_table_falls_back_to_copresence_beside_trash_can(self):
        parsed = parse_command("Find the pencilcase on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:pencilcase:0",
                    "pencilcase",
                    (40.0, 300.0, 100.0, 360.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:trash_can:0",
                    "trash_can",
                    (10.0, 250.0, 90.0, 420.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (400.0, 40.0, 600.0, 200.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertTrue(ranked)
        self.assertEqual("rig_a:pencilcase:0", ranked[0].target.candidate_id)
        self.assertEqual("picnic_table", ranked[0].anchor.class_name)
        self.assertEqual("copresent", ranked[0].match)

    def test_strict_on_table_beats_copresent_trash_can_case(self):
        parsed = parse_command("Find the pencilcase on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    "rig_a:pencilcase:trash",
                    "pencilcase",
                    (40.0, 300.0, 100.0, 360.0),
                    "rig_a",
                    0.99,
                ),
                _object(
                    "rig_a:trash_can:0",
                    "trash_can",
                    (10.0, 250.0, 90.0, 420.0),
                    "rig_a",
                ),
                _object(
                    "rig_a:pencilcase:on",
                    "pencilcase",
                    (280.0, 200.0, 340.0, 255.0),
                    "rig_a",
                    0.4,
                ),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:pencilcase:on", ranked[0].target.candidate_id)
        self.assertEqual("strict", ranked[0].match)

    def test_named_landmark_other_camera_still_returns_target(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:mug:0", "mug", (280.0, 200.0, 340.0, 255.0), "rig_a"),
                _object(
                    "rig_b:picnic_table:0",
                    "picnic_table",
                    (200.0, 250.0, 450.0, 400.0),
                    "rig_b",
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )
        ranked = rank_object_candidates(parsed, round_)
        self.assertEqual("rig_a:mug:0", ranked[0].target.candidate_id)
        self.assertEqual("target", ranked[0].match)


if __name__ == "__main__":
    unittest.main()
