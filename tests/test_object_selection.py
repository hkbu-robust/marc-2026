"""Tests for lost-item object ranking across cameras and relations."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

import numpy as np

from marc_agent.command_parser import Intent, ParsedCommand, Relation, parse_command
from marc_agent.experiment_log import decision_scope
from marc_agent.perception.object_selection import (
    ObjectSelection,
    _context_crop,
    build_object_board,
    format_object_verifier_prompt,
    infer_anchor,
    parse_object_verifier_output,
    rank_object_candidates,
    select_object_candidate,
)
from marc_agent.types import (
    CctvDetectionRound,
    CctvFrame,
    ObjectCandidate,
    PersonCandidate,
)
from scripts.stage1_replay import projector_for

HF08_RUN02 = Path(__file__).resolve().parent / "fixtures" / "hf08_scene_evidence_run02.json"


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
    def test_detail_crop_stays_on_tiny_target_when_anchor_is_far(self):
        frame = CctvFrame(
            "cam", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
        )
        item = rank_object_candidates(
            parse_command("Find the tumbler near the fire hydrant."),
            CctvDetectionRound(
                objects=[
                    _object("cam:tumbler:0", "tumbler", (500, 300, 510, 310), "cam"),
                    _object("cam:hydrant:0", "hydrant", (20, 100, 100, 300), "cam"),
                ],
                camera_ids=["cam"],
            ),
        )[0]

        crop = np.asarray(_context_crop(frame, item))

        red = (crop[:, :, 0] > 200) & (crop[:, :, 1] < 80)
        ys, xs = np.where(red)
        self.assertGreater(xs.max() - xs.min(), 30)
        self.assertGreater(ys.max() - ys.min(), 30)

    def test_remote_qwen_gets_a_detail_crop_for_every_same_class_proposal(self):
        parsed = parse_command("Find the tumbler.")
        objects = [
            _object(
                f"rig_a:tumbler:{index}",
                "tumbler",
                (40.0 + 100 * index, 180.0, 90.0 + 100 * index, 260.0),
                "rig_a",
                0.99 - 0.1 * index,
            )
            for index in range(4)
        ]
        frame = CctvFrame(
            "rig_a", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
        )
        round_ = CctvDetectionRound(
            objects=objects,
            frames={"rig_a": frame},
            camera_ids=["rig_a"],
        )
        calls = []

        def verifier(board):
            calls.append(board)
            return "rig_a:tumbler:2|none"

        selected = select_object_candidate(
            parsed,
            round_,
            command_text="Find the tumbler.",
            verifier=verifier,
        )

        self.assertEqual("rig_a:tumbler:2", selected.target.candidate_id)
        self.assertEqual(4, len(set(calls[0].candidate_ids)))
        self.assertEqual(5, len(calls[0].images))
        self.assertTrue(all(image.size == (512, 512) for image in calls[0].images[1:]))
        self.assertIn("including a shoe", calls[0].prompt)

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
        self.assertEqual("deterministic_fallback", selected.method)
        self.assertIn("output=not a pair", "\n".join(captured.output))

    def test_single_pair_calls_configured_verifier(self):
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
                verifier=lambda board: called.append(1) or "rig_a:tumbler:0|none",
            )
        self.assertEqual("rig_a:tumbler:0", selected.target.candidate_id)
        self.assertEqual([1], called)
        self.assertEqual("qwen", selected.method)
        self.assertIn("verify=True reason=verifier_configured", "\n".join(logged.output))

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
        self.assertIn(
            "verify=True reason=verifier_configured",
            "\n".join(logged.output),
        )
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

    def test_verifier_rejects_anchor_only_reply(self):
        parsed = parse_command("Find the mug on the picnic table.")
        ranked = rank_object_candidates(parsed, _on_table_vs_beside_bench())
        with self.assertRaises(ValueError):
            parse_object_verifier_output("rig_a:picnic_table:0", ranked)

    def test_parking_lot_car_cue_pairs_with_vehicle_anchor(self):
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
        self.assertTrue(all(item.anchor.class_name == "suv" for item in ranked))
        self.assertEqual("rig_7_a:tumbler:3", ranked[0].target.candidate_id)
        target_id, anchor_id = parse_object_verifier_output(
            "rig_7_a:tumbler:3|rig_7_a:suv:0", ranked
        )
        self.assertEqual("rig_7_a:tumbler:3", target_id)
        self.assertEqual("rig_7_a:suv:0", anchor_id)
        calls = []
        selected = select_object_candidate(
            parsed,
            round_,
            command_text=text,
            verifier=lambda board: calls.append(board)
            or "rig_7_a:tumbler:3|rig_7_a:suv:0",
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("rig_7_a:tumbler:3", selected.target.candidate_id)
        self.assertEqual("rig_7_a:suv:0", selected.anchor.candidate_id)

    def test_verifier_truncated_prefix_resolves_unique_target(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = _on_table_vs_beside_bench()
        ranked = rank_object_candidates(parsed, round_)
        target_id, anchor_id = parse_object_verifier_output("rig_a:mug:", ranked)
        self.assertEqual("rig_a:mug:0", target_id)
        self.assertEqual("rig_a:picnic_table:0", anchor_id)

    def test_verifier_camera_prefix_with_anchor_resolves_unique_pair(self):
        parsed = parse_command("Find the mug on the picnic table.")
        ranked = rank_object_candidates(parsed, _on_table_vs_beside_bench())
        target_id, anchor_id = parse_object_verifier_output(
            "rig_a|rig_a:picnic_table:0", ranked
        )
        self.assertEqual("rig_a:mug:0", target_id)
        self.assertEqual("rig_a:picnic_table:0", anchor_id)

    def test_verifier_camera_prefix_uses_best_pair_within_that_camera(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:mug:0", "mug", (280, 200, 340, 255), "rig_a", 0.9),
                _object("rig_a:mug:1", "mug", (350, 200, 410, 255), "rig_a", 0.7),
                _object(
                    "rig_a:picnic_table:0",
                    "picnic_table",
                    (200, 250, 450, 400),
                    "rig_a",
                ),
            ],
            camera_ids=["rig_a"],
        )
        ranked = rank_object_candidates(parsed, round_)
        target_id, anchor_id = parse_object_verifier_output(
            "rig_a|rig_a:picnic_table:0", ranked
        )
        self.assertEqual(ranked[0].target.candidate_id, target_id)
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

    def test_named_class_stays_on_the_board_when_another_class_fits_the_relation(self):
        parsed = parse_command("Find the tumbler on the bench.")
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_bad:tumbler:0", "tumbler", (20, 20, 60, 70), "rig_bad"),
                _object("rig_bad:bench:0", "bench", (150, 200, 420, 390), "rig_bad"),
                _object("rig_good:mug:0", "mug", (250, 170, 290, 220), "rig_good"),
                _object("rig_good:bench:0", "bench", (150, 200, 420, 390), "rig_good"),
            ],
            frames={
                "rig_bad": CctvFrame("rig_bad", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)),
                "rig_good": CctvFrame("rig_good", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)),
            },
            camera_ids=["rig_bad", "rig_good"],
        )
        boards = []
        selected = select_object_candidate(
            parsed,
            round_,
            command_text="Find the tumbler on the bench.",
            verifier=lambda board: boards.append(board) or (
                "rig_good:mug:0|rig_good:bench:0"
            ),
        )
        self.assertTrue(boards)
        self.assertIn("rig_bad:tumbler:0", boards[0].candidate_ids)
        self.assertNotIn("rig_good:mug:0", boards[0].candidate_ids)
        self.assertEqual("rig_bad:tumbler:0", selected.target.candidate_id)

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

    def test_single_relation_pair_calls_configured_verifier(self):
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
            verifier=lambda board: calls.append(board) or (
                "rig_a:tumbler:0|rig_a:bench:0"
            ),
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("rig_a:tumbler:0", selected.target.candidate_id)
        self.assertEqual("qwen", selected.method)

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


    def test_parking_floor_drop_pairs_with_vehicle(self):
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
        self.assertEqual("police_car", ranked[0].anchor.class_name)
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
        # Without a projector the ranking falls back to image-space pair scores.
        # The 0.4-confidence tumbler used to win only through the police_car
        # +0.5 bonus (#113); on pair evidence alone the confident tumbler near
        # the normal car ranks first.
        self.assertEqual("rig_a:tumbler:0", ranked[0].target.candidate_id)
        self.assertEqual("rig_a:normal_car_1:0", ranked[0].anchor.candidate_id)
        # No vehicle class is preferred: renaming the cars changes nothing.
        renamed = CctvDetectionRound(
            objects=[
                replace(item, class_name={"police_car": "suv", "normal_car_1": "sports_car"}.get(
                    item.class_name, item.class_name))
                for item in round_.objects
            ],
            people=round_.people,
            camera_ids=round_.camera_ids,
        )
        swapped = rank_object_candidates(
            parsed, renamed,
            command_text="Find the tumbler dropped in the empty parking space.",
        )
        self.assertEqual(
            [(item.target.candidate_id, item.anchor.candidate_id) for item in ranked],
            [(item.target.candidate_id, item.anchor.candidate_id) for item in swapped],
        )

    def test_parking_anchor_is_the_nearest_vehicle_whatever_its_class(self):
        text = "Find the tumbler dropped in the empty parking space."
        parsed = parse_command(text)
        target = (300.0, 200.0, 320.0, 240.0)
        police = (10.0, 100.0, 200.0, 300.0)
        suv = (330.0, 100.0, 600.0, 300.0)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", target, "cam"),
                _object("cam:police_car:0", "police_car", police, "cam"),
                _object("cam:suv:0", "suv", suv, "cam"),
            ],
            camera_ids=["cam"],
        )

        class Projector:
            def project_contact(self, _camera_id, box, **_kwargs):
                return {
                    target: (0.0, 0.0, 0.0),
                    police: (10.0, 0.0, 0.0),
                    suv: (1.0, 0.0, 0.0),
                }[tuple(box)]

        ranked = rank_object_candidates(
            parsed, round_, command_text=text, projector=Projector()
        )
        self.assertEqual("cam:tumbler:0", ranked[0].target.candidate_id)
        self.assertEqual("cam:suv:0", ranked[0].anchor.candidate_id)
        with decision_scope("stage1", 1) as notes:
            selected = select_object_candidate(
                parsed, round_, command_text=text, projector=Projector()
            )
        self.assertEqual("cam:suv:0", selected.anchor.candidate_id)
        implied = [note for note in notes if note["decision"] == "implied_anchor"]
        self.assertEqual("cam:suv:0", implied[0]["chosen"])
        self.assertAlmostEqual(1.0, implied[0]["dist_m"])

    def test_implied_anchor_replaces_a_farther_ranked_anchor(self):
        text = "I think I dropped my tumbler on the ground as I got out of my car in the parking lot."
        parsed = parse_command(text, expected_intent=Intent.FIND)
        target = ObjectCandidate("cam:tumbler:0", "tumbler", (300.0, 200.0, 320.0, 240.0), 0.9, "cam", (640, 480))
        far = ObjectCandidate("cam:police_car:0", "police_car", (10.0, 100.0, 200.0, 300.0), 0.9, "cam", (640, 480))
        near = ObjectCandidate("cam:suv:0", "suv", (330.0, 100.0, 600.0, 300.0), 0.5, "cam", (640, 480))
        other_camera = ObjectCandidate("cam2:taxi:0", "taxi", (300.0, 200.0, 320.0, 240.0), 0.9, "cam2", (640, 480))

        class Projector:
            def project_contact(self, camera_id, box, **_kwargs):
                return {far.bbox_xyxy: (5.0, 0.0, 0.0), near.bbox_xyxy: (0.0, 2.0, 0.0)}.get(
                    tuple(box), (0.0, 0.0, 0.0)
                )

        selection = ObjectSelection(target=target, anchor=far, score=1.0)
        result = infer_anchor(selection, parsed, text, [target, far, near, other_camera], Projector())
        self.assertEqual("cam:suv:0", result.anchor.candidate_id)
        named = parse_command("Find the tumbler near the police car.")
        self.assertIs(selection, infer_anchor(selection, named, "", [near], Projector()))

    def _hf08_run02(self):
        case = json.loads(HF08_RUN02.read_text())
        objects, people = [], []
        for item in case["candidates"]:
            box = tuple(item["bbox_xyxy"])
            if item["class_name"].startswith("person_"):
                people.append(PersonCandidate(item["candidate_id"], box, item["class_name"],
                                              item["confidence"], item["camera_id"], (1280, 720)))
            else:
                objects.append(ObjectCandidate(item["candidate_id"], item["class_name"], box,
                                               item["confidence"], item["camera_id"], (1280, 720)))
        frames = {
            camera: CctvFrame(camera, 1280, 720, np.zeros((720, 1280, 3), dtype=np.uint8))
            for camera in case["camera_ids"]
        }
        round_ = CctvDetectionRound(objects=objects, people=people, frames=frames,
                                    camera_ids=case["camera_ids"])
        return case, round_

    def test_scene_evidence_guard_keeps_geometry_in_the_parking_camera_hf08(self):
        """#88 run-02 hf08: Qwen chose an umbrella in rig_2_b, which shows no parking lot."""
        case, round_ = self._hf08_run02()
        parsed = parse_command(case["command"], expected_intent=Intent.FIND)
        with decision_scope("stage1", 19) as notes:
            selected = select_object_candidate(
                parsed, round_, command_text=case["command"],
                verifier=lambda _board: case["qwen_target"], projector=projector_for(),
            )
        self.assertEqual("rig_7_a:umbrella:0", selected.target.candidate_id)
        override = [note for note in notes if note["decision"] == "geometry_override"]
        self.assertEqual("scene_evidence_guard", override[0]["reason"])

    def test_scene_evidence_guard_does_not_fire_within_one_camera(self):
        case, round_ = self._hf08_run02()
        parsed = parse_command(case["command"], expected_intent=Intent.FIND)
        with decision_scope("stage1", 19) as notes:
            selected = select_object_candidate(
                parsed, round_, command_text=case["command"],
                verifier=lambda _board: "rig_7_a:umbrella:0", projector=projector_for(),
            )
        self.assertEqual("rig_7_a:umbrella:0", selected.target.candidate_id)
        self.assertFalse([note for note in notes if note.get("reason") == "scene_evidence_guard"])

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
        self.assertIn("place all tie, prefer the higher-confidence detection", prompt)
        self.assertIn("Occupancy 0 is drivable road", prompt)
        self.assertIn("explicitly says empty space or empty stall", prompt)
        self.assertIn("painted parking bay", prompt)
        self.assertIn("Parking-lot or floor language alone", prompt)
        self.assertIn("must never override visible target identity", prompt)
        self.assertIn("Parked cars (not a distant taxi)", prompt)
        self.assertIn("Never let an exact class label override", prompt)
        self.assertIn("requested relation and landmark as mandatory", prompt)
        self.assertIn("For ON or ABOVE", prompt)
        self.assertIn("including a shoe", prompt)
        self.assertIn("remaining visually valid target instances", prompt)
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

    def test_unique_empty_stall_calls_configured_verifier(self):
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
            verifier=lambda board: calls.append(board) or "cam:tumbler:1|none",
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("cam:tumbler:1", selected.target.candidate_id)
        self.assertEqual("qwen", selected.method)

    def test_remote_verifier_compares_large_board_in_one_packed_request(self):
        text = "Find the umbrella."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object(
                    f"cam:umbrella:{index}",
                    "umbrella",
                    (40.0 + 100 * index, 200.0, 80.0 + 100 * index, 260.0),
                    "cam",
                    0.9 - 0.1 * index,
                )
                for index in range(4)
            ],
            frames={
                "cam": CctvFrame(
                    "cam", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
                )
            },
            camera_ids=["cam"],
        )
        seen = set()
        calls = []

        class Verifier:
            method = "qwen_remote"

            def __call__(self, board):
                calls.append(board)
                self.last_finish_reason = "stop"
                self.last_metrics = None
                seen.update(board.candidate_ids)
                self.assert_board(board)
                chosen = (
                    "cam:umbrella:3"
                    if "cam:umbrella:3" in board.candidate_ids
                    else board.candidate_ids[0]
                )
                return chosen

            @staticmethod
            def assert_board(board):
                assert len(board.candidate_ids) == 4
                assert len(board.images) == 1
                assert "visual=IMAGE_1:tile=O04" in board.prompt

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )
        self.assertEqual("cam:umbrella:3", selected.target.candidate_id)
        self.assertEqual({f"cam:umbrella:{index}" for index in range(4)}, seen)
        self.assertEqual(1, len(calls))

    def test_remote_verifier_screens_detector_classes_before_final_choice(self):
        text = "Find the mug."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:mug:0", "mug", (20, 100, 60, 160), "cam"),
                _object("cam:mug:1", "mug", (80, 100, 120, 160), "cam"),
                _object("cam:cola_can:0", "cola_can", (140, 100, 180, 160), "cam"),
                _object("cam:cola_can:1", "cola_can", (200, 100, 240, 160), "cam"),
            ],
            frames={"cam": CctvFrame(
                "cam", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
            )},
            camera_ids=["cam"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"
            last_metrics = None
            last_finish_reason = "stop"

            def __call__(self, board):
                calls.append(board)
                return "cam:mug:1"

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )
        self.assertEqual("cam:mug:1", selected.target.candidate_id)
        self.assertEqual(1, len(calls))
        self.assertTrue(all(item.target.class_name == "mug" for item in calls[0].ranked))
        self.assertEqual(2, len(calls[0].ranked))

    def test_remote_near_landmark_keeps_all_proposals_for_final_choice(self):
        text = "Find the umbrella near the trash can."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_a:umbrella:0", "umbrella", (20, 100, 60, 160), "cam_a"),
                _object("cam_b:umbrella:0", "umbrella", (80, 100, 120, 160), "cam_b"),
                _object("cam_a:trash_can:0", "trash_can", (10, 80, 100, 220), "cam_a"),
                _object("cam_b:trash_can:0", "trash_can", (60, 80, 160, 220), "cam_b"),
            ],
            frames={
                camera_id: CctvFrame(
                    camera_id, 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
                )
                for camera_id in ("cam_a", "cam_b")
            },
            camera_ids=["cam_a", "cam_b"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"
            last_metrics = None
            last_finish_reason = "stop"

            def __call__(self, board):
                calls.append(board)
                return "cam_b:umbrella:0"

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )

        self.assertEqual("cam_b:umbrella:0", selected.target.candidate_id)
        self.assertEqual(1, len(calls))
        self.assertEqual(2, len(calls[0].ranked))

    def test_remote_empty_parking_request_keeps_geometry_choice(self):
        text = "Find the umbrella dropped in the empty parking space."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_a:umbrella:0", "umbrella", (20, 100, 60, 160), "cam_a"),
                _object("cam_b:umbrella:0", "umbrella", (80, 100, 120, 160), "cam_b"),
                _object("cam_b:tumbler:0", "tumbler", (140, 100, 180, 160), "cam_b"),
                _object("cam_a:police_car:0", "police_car", (10, 50, 200, 200), "cam_a"),
                _object("cam_b:police_car:0", "police_car", (10, 50, 200, 200), "cam_b"),
            ],
            frames={
                camera_id: CctvFrame(
                    camera_id, 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
                )
                for camera_id in ("cam_a", "cam_b")
            },
            camera_ids=["cam_a", "cam_b"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"
            last_metrics = None
            last_finish_reason = "stop"

            def __call__(self, board):
                calls.append(board)
                return "cam_b:tumbler:0"

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )
        self.assertEqual("cam_b:umbrella:0", selected.target.candidate_id)
        self.assertEqual([], calls)
        self.assertEqual("object_relation", selected.method)

    def test_implied_anchor_command_is_decided_by_geometry(self):
        """#120 option (b): no named landmark but a car cue -> geometry, Qwen not asked (hf07)."""
        text = "Find the tumbler I dropped as I got out of my car in the parking lot."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", (20, 100, 60, 160), "cam"),
                _object("cam:tumbler:1", "tumbler", (500, 100, 540, 160), "cam"),
                _object("cam:police_car:0", "police_car", (10, 50, 200, 200), "cam"),
            ],
            camera_ids=["cam"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"

            def __call__(self, board):
                calls.append(board)
                return "cam:tumbler:1"

        geometry = rank_object_candidates(parsed, round_, command_text=text)[0]
        with decision_scope("stage1", 1) as notes:
            selected = select_object_candidate(parsed, round_, command_text=text, verifier=Verifier())
        self.assertEqual([], calls)
        self.assertEqual(geometry.target.candidate_id, selected.target.candidate_id)
        self.assertTrue([n for n in notes if n["decision"] == "selector"
                         and n.get("reason") == "implied_anchor_geometry"])

    def test_a_named_landmark_still_goes_to_qwen(self):
        text = "Find the tumbler near the police car."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", (20, 100, 60, 160), "cam"),
                _object("cam:tumbler:1", "tumbler", (150, 100, 190, 160), "cam"),
                _object("cam:police_car:0", "police_car", (10, 50, 200, 200), "cam"),
            ],
            camera_ids=["cam"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"

            def __call__(self, board):
                calls.append(board)
                return "cam:tumbler:1"

        select_object_candidate(parsed, round_, command_text=text, verifier=Verifier())
        self.assertTrue(calls)

    @patch.dict(os.environ, {"MARC_IMPLIED_ANCHOR_GEOMETRY": "0"})
    def test_remote_parking_lot_request_keeps_visual_choice(self):
        """With #120 option (b) off: Qwen decides implied-anchor commands."""
        text = "Find the tumbler I dropped as I got out of my car in the parking lot."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", (20, 100, 60, 160), "cam"),
                _object("cam:tumbler:1", "tumbler", (150, 100, 190, 160), "cam"),
                _object("cam:police_car:0", "police_car", (10, 50, 200, 200), "cam"),
            ],
            camera_ids=["cam"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"

            def __call__(self, board):
                calls.append(board)
                if "TARGET_ID|ANCHOR_ID" in board.prompt:
                    return "cam:tumbler:1|cam:police_car:0"
                return "cam:tumbler:1"

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )
        self.assertEqual("cam:tumbler:1", selected.target.candidate_id)
        self.assertEqual(1, len(calls))

    def test_remote_weak_parking_override_keeps_strong_geometry(self):
        text = "I dropped my tumbler as I got out of my car in the parking lot."
        parsed = parse_command(text, expected_intent=Intent.FIND)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", (20, 100, 60, 160), "cam", 0.9),
                _object("cam:tumbler:1", "tumbler", (500, 100, 505, 125), "cam", 0.08),
                _object("cam:police_car:0", "police_car", (10, 50, 200, 200), "cam"),
                _object("cam:sports_car:0", "sports_car", (400, 50, 620, 220), "cam"),
            ],
            camera_ids=["cam"],
        )

        class Verifier:
            method = "qwen_remote"

            def __call__(self, board):
                if "TARGET_ID|ANCHOR_ID" in board.prompt:
                    return "cam:tumbler:1|cam:sports_car:0"
                return "cam:tumbler:1"

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )

        self.assertEqual("cam:tumbler:0", selected.target.candidate_id)
        self.assertEqual("cam:police_car:0", selected.anchor.candidate_id)

    def test_remote_on_request_keeps_strict_support_over_copresence(self):
        text = "Find the tumbler on the bench."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_a:tumbler:0", "tumbler", (80, 80, 120, 100), "cam_a", 0.4),
                _object("cam_a:bench:0", "bench", (40, 100, 200, 220), "cam_a"),
                _object("cam_b:tumbler:0", "tumbler", (20, 300, 60, 340), "cam_b", 0.9),
                _object("cam_b:bench:0", "bench", (300, 100, 500, 220), "cam_b"),
            ],
            camera_ids=["cam_a", "cam_b"],
        )

        class Verifier:
            method = "qwen_remote"

            def __call__(self, _board):
                return "cam_b:tumbler:0"

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )
        self.assertEqual("cam_a:tumbler:0", selected.target.candidate_id)
        self.assertEqual("strict", selected.match)

    def test_a_very_low_confidence_target_no_longer_overrides_qwen(self):
        """#116: the <=0.003 forcing and the low-confidence guard are gone."""
        text = "Find the umbrella near the trash can."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_bad:umbrella:0", "umbrella", (210, 180, 240, 240), "cam_bad", 0.9),
                _object("cam_bad:trash_can:0", "trash_can", (200, 200, 300, 400), "cam_bad"),
                _object("cam_good:umbrella:0", "umbrella", (400, 200, 500, 400), "cam_good", 0.0025),
                _object("cam_good:trash_can:0", "trash_can", (400, 200, 500, 400), "cam_good"),
            ],
            camera_ids=["cam_bad", "cam_good"],
        )
        selected = select_object_candidate(
            parsed,
            round_,
            command_text=text,
            verifier=lambda _board: "cam_bad:umbrella:0|cam_bad:trash_can:0",
        )
        self.assertEqual("cam_bad:umbrella:0", selected.target.candidate_id)
        self.assertNotEqual("geometry_override", selected.method)

    def test_a_tiny_target_inside_its_landmark_no_longer_overrides_qwen(self):
        """#116: the class-pair contained-occlusion guard is gone."""
        text = "Find the umbrella near the trash can."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_bad:umbrella:0", "umbrella", (10, 10, 50, 60), "cam_bad", 0.9),
                _object("cam_bad:trash_can:0", "trash_can", (200, 100, 300, 350), "cam_bad"),
                _object("cam_good:umbrella:0", "umbrella", (420, 150, 435, 160), "cam_good", 0.031),
                _object("cam_good:trash_can:0", "trash_can", (400, 100, 500, 350), "cam_good"),
            ],
            camera_ids=["cam_bad", "cam_good"],
        )
        selected = select_object_candidate(
            parsed,
            round_,
            command_text=text,
            verifier=lambda _board: "cam_bad:umbrella:0|cam_bad:trash_can:0",
        )
        self.assertEqual("cam_bad:umbrella:0", selected.target.candidate_id)

    @patch.dict(os.environ, {"MARC_IMPLIED_ANCHOR_GEOMETRY": "0"})
    def test_remote_verifier_can_choose_alternate_implicit_anchor(self):
        text = "Pick up the tumbler dropped on the parking lot floor."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", (300, 220, 320, 250), "cam"),
                _object("cam:police_car:0", "police_car", (50, 100, 280, 300), "cam"),
                _object("cam:suv:0", "suv", (330, 100, 600, 300), "cam"),
            ],
            frames={
                "cam": CctvFrame(
                    "cam", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
                )
            },
            camera_ids=["cam"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"
            last_metrics = None
            last_finish_reason = "stop"

            def __call__(self, board):
                calls.append(board)
                if "TARGET_ID|ANCHOR_ID" in board.prompt:
                    return "cam:tumbler:0|cam:police_car:0"
                return "cam:tumbler:0"

        selected = select_object_candidate(
            parsed, round_, command_text=text, verifier=Verifier()
        )
        self.assertEqual("cam:police_car:0", selected.anchor.candidate_id)
        self.assertEqual(2, len(calls))
        self.assertIn("copy the chosen row as TARGET_ID|ANCHOR_ID", calls[-1].prompt)
        self.assertIn("anchor_owned=", calls[-1].prompt)
        self.assertIn("do not blindly choose the nearest vehicle", calls[-1].prompt)

    def test_packed_board_has_one_tile_per_physical_target(self):
        text = "Find the tumbler in the parking lot."
        parsed = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", (300, 250, 330, 290), "cam"),
                _object("cam:police_car:0", "police_car", (100, 180, 280, 360), "cam"),
                _object("cam:suv:0", "suv", (350, 180, 550, 360), "cam"),
            ],
            frames={"cam": CctvFrame(
                "cam", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
            )},
            camera_ids=["cam"],
        )
        ranked = rank_object_candidates(
            parsed, round_, command_text=text, visual_proposals=True
        )
        board = build_object_board(
            text,
            parsed,
            round_,
            ranked,
            max_crops=len(ranked),
            pack_cameras=True,
        )
        self.assertEqual(1, len(ranked))
        self.assertIn("visual=IMAGE_1:tile=O01", board.prompt)
        self.assertNotIn("visual=IMAGE_1:tile=O02", board.prompt)

    def test_a_taxi_counts_like_any_other_vehicle_for_parking(self):
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
        def ranking(round_):
            ranked = rank_object_candidates(
                parsed,
                round_,
                command_text=text,
                occupancy_at=lambda _x, _y: 0,
                visual_proposals=True,
            )
            return [(item.target.candidate_id, item.anchor and item.anchor.camera_id)
                    for item in ranked]

        # Where a vehicle class stands is the scene's business (#113): a taxi
        # ranks exactly as another car would in its place.
        as_car = CctvDetectionRound(
            objects=[
                replace(item, class_name="normal_car_2") if item.class_name == "taxi" else item
                for item in round_.objects
            ],
            camera_ids=round_.camera_ids,
        )
        self.assertEqual(ranking(as_car), ranking(round_))
        self.assertIn("cam_park:tumbler:0", [target for target, _ in ranking(round_)])

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

    def test_named_near_ranks_by_world_metres_not_pixel_score(self):
        text = "Find the umbrella next to the trash can."
        parsed = parse_command(text)
        park_umb = (200.0, 200.0, 260.0, 280.0)
        park_trash = (210.0, 210.0, 250.0, 300.0)
        owned_umb = (40.0, 300.0, 90.0, 380.0)
        owned_trash = (200.0, 200.0, 280.0, 360.0)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_park:umbrella:0", "umbrella", park_umb, "cam_park", 0.99),
                _object("cam_park:trash_can:0", "trash_can", park_trash, "cam_park"),
                _object("cam_own:umbrella:0", "umbrella", owned_umb, "cam_own", 0.4),
                _object("cam_own:trash_can:0", "trash_can", owned_trash, "cam_own"),
            ],
            camera_ids=["cam_park", "cam_own"],
        )

        class _XY:
            def project_contact(self, camera_id, box, **_kwargs):
                key = (camera_id, tuple(box))
                return {
                    ("cam_park", park_umb): (10.0, 0.0, 0.0),
                    ("cam_park", park_trash): (14.0, 0.0, 0.0),
                    ("cam_own", owned_umb): (0.0, 0.0, 0.0),
                    ("cam_own", owned_trash): (0.4, 0.0, 0.0),
                }.get(key)

        ranked = rank_object_candidates(
            parsed, round_, projector=_XY(), command_text=text,
        )
        self.assertEqual("cam_own:umbrella:0", ranked[0].target.candidate_id)

    def test_first_person_near_keeps_owned_pair(self):
        text = (
            "Find the umbrella I left leaning against the trash can."
        )
        parsed = parse_command(text)
        owned_umb = (80.0, 260.0, 140.0, 340.0)
        owned_trash = (150.0, 200.0, 250.0, 400.0)
        empty_umb = (200.0, 200.0, 240.0, 250.0)
        empty_trash = (205.0, 205.0, 235.0, 255.0)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_own:umbrella:0", "umbrella", owned_umb, "cam_own", 0.4),
                _object("cam_own:trash_can:0", "trash_can", owned_trash, "cam_own"),
                _object("cam_park:umbrella:0", "umbrella", empty_umb, "cam_park", 0.99),
                _object("cam_park:trash_can:0", "trash_can", empty_trash, "cam_park"),
            ],
            people=[
                _person(
                    "cam_own:person_standing:0",
                    (40.0, 180.0, 200.0, 400.0),
                    "cam_own",
                    "person_walking",
                ),
            ],
            camera_ids=["cam_own", "cam_park"],
        )

        class _XY:
            def project_contact(self, camera_id, box, **_kwargs):
                key = (camera_id, tuple(box))
                return {
                    ("cam_own", owned_umb): (0.0, 0.0, 0.0),
                    ("cam_own", owned_trash): (0.5, 0.0, 0.0),
                    ("cam_park", empty_umb): (20.0, 0.0, 0.0),
                    ("cam_park", empty_trash): (24.0, 0.0, 0.0),
                }.get(key)

        ranked = rank_object_candidates(
            parsed, round_, projector=_XY(), command_text=text,
        )
        self.assertEqual("cam_own:umbrella:0", ranked[0].target.candidate_id)

    def test_road_taxi_prefers_occupancy_road_and_closer_taxi(self):
        text = "Find the umbrella on the road next to the taxi."
        parsed = parse_command(text)
        self.assertEqual("road", parsed.scene_hint)
        self.assertEqual("taxi", parsed.landmark)
        park_umb = (40.0, 300.0, 80.0, 380.0)
        park_taxi = (90.0, 280.0, 400.0, 420.0)
        road_umb = (200.0, 200.0, 240.0, 250.0)
        road_taxi = (250.0, 180.0, 400.0, 360.0)
        round_ = CctvDetectionRound(
            objects=[
                _object("cam_park:umbrella:0", "umbrella", park_umb, "cam_park", 0.99),
                _object("cam_park:taxi:0", "taxi", park_taxi, "cam_park"),
                _object("cam_park:bench:0", "bench", (20.0, 20.0, 180.0, 120.0), "cam_park"),
                _object("cam_road:umbrella:0", "umbrella", road_umb, "cam_road", 0.4),
                _object("cam_road:taxi:0", "taxi", road_taxi, "cam_road"),
            ],
            camera_ids=["cam_park", "cam_road"],
        )

        class _XY:
            def project_contact(self, camera_id, box, **_kwargs):
                key = (camera_id, tuple(box))
                return {
                    ("cam_park", park_umb): (0.0, 10.0, 0.0),
                    ("cam_park", park_taxi): (8.0, 10.0, 0.0),
                    ("cam_road", road_umb): (0.0, 0.0, 0.0),
                    ("cam_road", road_taxi): (1.0, 0.0, 0.0),
                }.get(key)

        def occupancy_at(x, y):
            return -1 if y >= 5.0 else 0

        ranked = rank_object_candidates(
            parsed,
            round_,
            projector=_XY(),
            command_text=text,
            occupancy_at=occupancy_at,
        )
        self.assertEqual("cam_road:umbrella:0", ranked[0].target.candidate_id)


class ClassFirstTests(unittest.TestCase):
    def _round(self, mug_box, juice_box) -> CctvDetectionRound:
        return CctvDetectionRound(
            objects=[
                _object("rig_a:bench:0", "bench", (300.0, 180.0, 420.0, 420.0), "rig_a"),
                _object("rig_a:juice:0", "juice", juice_box, "rig_a", 0.95),
                _object("rig_a:mug:0", "mug", mug_box, "rig_a", 0.4),
            ],
            frames={"rig_a": CctvFrame(
                "rig_a", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8),
            )},
            camera_ids=["rig_a"],
        )

    def test_a_wrong_class_that_fits_the_relation_loses_to_the_named_class(self):
        parsed = ParsedCommand(
            Intent.FIND, "mug", landmark="bench", relation=Relation.LEFT,
        )
        ranked = rank_object_candidates(
            parsed,
            self._round((500.0, 220.0, 560.0, 280.0), (40.0, 220.0, 100.0, 280.0)),
            command_text="Find the mug to the left of the bench.",
            visual_proposals=True,
        )
        self.assertEqual("mug", ranked[0].target.class_name)

    def test_another_class_is_kept_when_the_named_class_is_absent(self):
        parsed = ParsedCommand(
            Intent.FIND, "mug", landmark="bench", relation=Relation.LEFT,
        )
        round_ = self._round((500.0, 220.0, 560.0, 280.0), (40.0, 220.0, 100.0, 280.0))
        round_.objects = [item for item in round_.objects if item.class_name != "mug"]
        ranked = rank_object_candidates(
            parsed, round_, command_text="Find the mug to the left of the bench.",
            visual_proposals=True,
        )
        self.assertEqual("juice", ranked[0].target.class_name)

    def test_qwen_cannot_submit_another_class_when_the_named_class_is_present(self):
        text = "Find the mug to the left of the bench."
        parsed = parse_command(text)
        round_ = self._round((40.0, 200.0, 90.0, 260.0), (120.0, 200.0, 170.0, 260.0))

        class Verifier:
            method = "qwen_remote"
            last_metrics = None
            last_finish_reason = "stop"

            def __call__(self, board):
                self.classes = {item.target.class_name for item in board.ranked}
                return "rig_a:juice:0"

        verifier = Verifier()
        with patch.dict(os.environ, {"MARC_CLASS_FIRST": "0", "MARC_CROSS_CLASS_GUARD": "1"}):
            selected = select_object_candidate(
                parsed, round_, command_text=text, verifier=verifier,
            )
        self.assertEqual({"mug"}, verifier.classes)
        self.assertEqual("mug", selected.target.class_name)


if __name__ == "__main__":
    unittest.main()
