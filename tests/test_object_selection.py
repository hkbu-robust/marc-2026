"""Tests for lost-item object ranking across cameras and relations."""

from __future__ import annotations

import unittest

from marc_agent.command_parser import parse_command
from marc_agent.perception.object_selection import (
    parse_object_verifier_output,
    rank_object_candidates,
    select_object_candidate,
)
from marc_agent.types import CctvDetectionRound, ObjectCandidate


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
        self.assertIsNone(select_object_candidate(parsed, round_))

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


    def test_verifier_target_only_uses_best_ranked_pair(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = _on_table_vs_beside_bench()
        ranked = rank_object_candidates(parsed, round_)
        target_id, anchor_id = parse_object_verifier_output("rig_a:mug:0", ranked)
        self.assertEqual("rig_a:mug:0", target_id)
        self.assertEqual("rig_a:picnic_table:0", anchor_id)

    def test_verifier_truncated_prefix_resolves_unique_target(self):
        parsed = parse_command("Find the mug on the picnic table.")
        round_ = _on_table_vs_beside_bench()
        ranked = rank_object_candidates(parsed, round_)
        target_id, anchor_id = parse_object_verifier_output("rig_a:mug:", ranked)
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


    def test_parking_scene_pairs_target_with_vehicle(self):
        parsed = parse_command(
            "Find the tumbler dropped in the empty parking space.",
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
        self.assertEqual("rig_a:tumbler:0", ranked[0].target.candidate_id)
        self.assertEqual("police_car", ranked[0].anchor.class_name)

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


if __name__ == "__main__":
    unittest.main()
