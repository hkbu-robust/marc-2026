"""Tests for composing Stage 1 GroundingResult payloads."""

from __future__ import annotations

import unittest

from marc_agent.perception.cctv_project import (
    CameraGeometry,
    CctvProjector,
    static_lookup,
)
from marc_agent.perception.stage1_pipeline import run_stage1_round
from marc_agent.types import (
    CctvDetectionRound,
    ObjectCandidate,
    PersonCandidate,
)


IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _geometry(camera_id: str = "rig_a") -> CctvProjector:
    return CctvProjector(
        static_lookup(
            {
                camera_id: CameraGeometry(
                    fx=500.0,
                    fy=500.0,
                    cx=320.0,
                    cy=240.0,
                    rotation=IDENTITY,
                    translation=(0.0, 0.0, -10.0),
                    ground_height=0.0,
                )
            }
        )
    )


def _person_round() -> CctvDetectionRound:
    return CctvDetectionRound(
        objects=[
            ObjectCandidate(
                "rig_a:trash_can:0",
                "trash_can",
                (300.0, 190.0, 430.0, 430.0),
                0.9,
                "rig_a",
                (640, 480),
            )
        ],
        people=[
            PersonCandidate(
                "rig_a:person:0",
                (30.0, 80.0, 130.0, 400.0),
                "person_walking",
                0.8,
                "rig_a",
                (640, 480),
            ),
            PersonCandidate(
                "rig_a:person:1",
                (315.0, 100.0, 405.0, 390.0),
                "person_bending",
                0.8,
                "rig_a",
                (640, 480),
            ),
        ],
        camera_ids=["rig_a"],
    )


def _object_round() -> CctvDetectionRound:
    return CctvDetectionRound(
        objects=[
            ObjectCandidate(
                "rig_a:mug:0",
                "mug",
                (280.0, 200.0, 340.0, 255.0),
                0.7,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_a:picnic_table:0",
                "picnic_table",
                (200.0, 250.0, 450.0, 400.0),
                0.9,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_a:mug:1",
                "mug",
                (20.0, 20.0, 50.0, 40.0),
                0.4,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_a:picnic_table:1",
                "picnic_table",
                (10.0, 45.0, 80.0, 90.0),
                0.5,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_b:mug:0",
                "mug",
                (270.0, 300.0, 330.0, 360.0),
                0.99,
                "rig_b",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_b:bench:0",
                "bench",
                (100.0, 300.0, 250.0, 400.0),
                0.9,
                "rig_b",
                (640, 480),
            ),
        ],
        camera_ids=["rig_a", "rig_b"],
    )


class Stage1RoundTests(unittest.TestCase):
    def test_sar_payload_keeps_parser_situation(self):
        def detector(client, *, stub_path=None):
            return _person_round()

        result = run_stage1_round(
            object(),
            "Find the person bent over near the trash can who needs urgent help.",
            detector=detector,
            verifier=lambda board: "rig_a:person:1|accident",
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("rig_a", result.grounding.camera_id)
        self.assertEqual("person", result.grounding.target_type)
        self.assertEqual("emergency", result.grounding.situation)
        self.assertEqual("trash_can", result.grounding.landmark)
        self.assertEqual("rig_a:person:1", result.target_id)
        self.assertEqual("rig_a:trash_can:0", result.anchor_id)
        self.assertEqual("qwen", result.method)
        self.assertEqual(3, len(result.grounding.target_coord))
        self.assertNotEqual([0.0, 0.0, 0.0], result.grounding.target_coord)

    def test_object_path_uses_verifier_pair(self):
        def detector(client, *, stub_path=None):
            return _object_round()

        result = run_stage1_round(
            object(),
            "Find the mug on the picnic table.",
            detector=detector,
            verifier=lambda board: "rig_a:mug:0|rig_a:picnic_table:0",
            projector=_geometry(),
        )
        self.assertIsNone(result.candidates.sar_selection)
        self.assertEqual("qwen", result.method)
        self.assertEqual("rig_a:mug:0", result.target_id)
        self.assertEqual("on", result.grounding.relation)
        self.assertEqual("picnic_table", result.grounding.landmark)
        self.assertIsNone(result.grounding.situation)
        self.assertGreater(result.grounding.target_coord[2], 0.0)

    def test_object_path_without_verifier_uses_geometry(self):
        def detector(client, *, stub_path=None):
            return _object_round()

        result = run_stage1_round(
            object(),
            "Find the mug on the picnic table.",
            detector=detector,
            projector=_geometry(),
        )
        self.assertEqual("object_relation", result.method)
        self.assertEqual("rig_a:mug:0", result.target_id)
        self.assertEqual("picnic_table", result.grounding.landmark)

    def test_missing_geometry_does_not_submit_zeros(self):
        def detector(client, *, stub_path=None):
            return _person_round()

        empty = CctvProjector(static_lookup({}))
        result = run_stage1_round(
            object(),
            "Find the person bent over near the trash can who needs urgent help.",
            detector=detector,
            verifier=lambda board: "rig_a:person:1|emergency",
            projector=empty,
        )
        self.assertIsNone(result.grounding)
        self.assertEqual("projection_failed", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
