"""Tests for composing Stage 1 GroundingResult payloads."""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from marc_agent.perception.cctv_project import (
    CameraGeometry,
    CctvProjector,
    static_lookup,
)
from marc_agent.perception.stage1_pipeline import (
    IGNORED_CLASS_DUMMY_XYZ,
    _project_selection,
    has_complete_camera_coverage,
    run_stage1_round,
    start_qwen_warmup,
    static_scene_key,
)
from marc_agent.perception.object_selection import ObjectSelection
from marc_agent.command_parser import Intent, Relation, parse_command
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


def test_paraphrased_static_scene_requests_share_a_key():
    first = parse_command("Find the umbrella next to the trash can.")
    second = parse_command(
        "I left my umbrella leaning against the trash can.",
        expected_intent=Intent.FIND,
    )
    assert static_scene_key(first) == static_scene_key(second)


def test_static_scene_cache_requires_every_discovered_camera_frame():
    incomplete = CctvDetectionRound(
        frames={"rig_a": object()}, camera_ids=["rig_a", "rig_b"]
    )
    complete = CctvDetectionRound(
        frames={"rig_a": object(), "rig_b": object()},
        camera_ids=["rig_a", "rig_b"],
    )

    assert not has_complete_camera_coverage(incomplete)
    assert has_complete_camera_coverage(complete)


def test_low_confidence_umbrella_at_trash_uses_occluder_geometry():
    target = ObjectCandidate(
        "rig_a:umbrella:0", "umbrella", (10, 10, 20, 20), 0.019,
        "rig_a", (1280, 720),
    )
    anchor = ObjectCandidate(
        "rig_a:trash_can:0", "trash_can", (100, 100, 200, 300), 0.9,
        "rig_a", (1280, 720),
    )

    class Projector:
        def __init__(self):
            self.boxes = []

        def project_contact(self, _camera_id, box, **_kwargs):
            self.boxes.append(box)
            return (0.0, 0.0, 0.0)

    projector = Projector()
    _project_selection(
        projector,
        parse_command("Find the umbrella next to the trash can."),
        ObjectSelection(target, anchor, 1.0),
        [target, anchor],
    )
    assert projector.boxes[0] == anchor.bbox_xyxy


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
                "rig_a:person_walking:0",
                (30.0, 80.0, 130.0, 400.0),
                "person_walking",
                0.8,
                "rig_a",
                (640, 480),
            ),
            PersonCandidate(
                "rig_a:person_bending:1",
                (280.0, 110.0, 350.0, 370.0),
                "person_bending",
                0.5,
                "rig_a",
                (640, 480),
            ),
            PersonCandidate(
                "rig_a:person_bending:0",
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
    def test_qwen_warmup_runs_in_the_background_and_logs_its_outcome(self):
        verifier = Mock()
        with patch.dict(os.environ, {"MARC_SAR_DETECTOR": "qwen-remote"}), patch(
            "marc_agent.perception.stage1_pipeline.RemoteQwenSarVerifier",
            return_value=verifier,
        ), patch("marc_agent.perception.stage1_pipeline.note_decision") as note:
            thread = start_qwen_warmup()
            thread.join(5.0)
        verifier.warm_up.assert_called_once()
        self.assertEqual("qwen_warmup", note.call_args.args[0])
        self.assertTrue(note.call_args.kwargs["ok"])
        self.assertIn("latency_s", note.call_args.kwargs)

    def test_qwen_warmup_failure_is_logged_not_raised(self):
        verifier = Mock()
        verifier.warm_up.side_effect = RuntimeError("vLLM down")
        with patch.dict(os.environ, {"MARC_SAR_DETECTOR": "qwen-remote"}), patch(
            "marc_agent.perception.stage1_pipeline.RemoteQwenSarVerifier",
            return_value=verifier,
        ), patch("marc_agent.perception.stage1_pipeline.note_decision") as note:
            start_qwen_warmup().join(5.0)
        self.assertFalse(note.call_args.kwargs["ok"])

    def test_no_qwen_warmup_without_the_remote_backend(self):
        with patch.dict(os.environ, {"MARC_SAR_DETECTOR": "fallback"}):
            self.assertIsNone(start_qwen_warmup())

    def test_selected_boxes_drive_target_and_anchor_projection(self):
        def detector(client, *, stub_path=None):
            return _object_round()

        projector = _geometry()
        result = run_stage1_round(
            object(),
            "Find the mug on the picnic table.",
            detector=detector,
            verifier=lambda board: "rig_a:mug:0|rig_a:picnic_table:0",
            projector=projector,
        )

        selection = result.candidates.object_selection
        expected_target = projector.project_contact(
            "rig_a",
            selection.target.bbox_xyxy,
            relation=Relation.ON,
            support_class="picnic_table",
            support_box=selection.anchor.bbox_xyxy,
        )
        expected_anchor = projector.project_contact("rig_a", selection.anchor.bbox_xyxy)
        self.assertIsNotNone(expected_target)
        self.assertIsNotNone(expected_anchor)
        self.assertEqual(list(expected_target), result.grounding.target_coord)
        self.assertEqual(list(expected_anchor), result.grounding.anchor_coord)

    def test_sar_payload_keeps_parser_situation(self):
        def detector(client, *, stub_path=None):
            return _person_round()

        result = run_stage1_round(
            object(),
            "Find the person bent over near the trash can who needs urgent help.",
            detector=detector,
            verifier=lambda board: "rig_a:person_bending:0|accident",
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("rig_a", result.grounding.camera_id)
        self.assertEqual("person", result.grounding.target_type)
        self.assertEqual("emergency", result.grounding.situation)
        self.assertEqual("trash_can", result.grounding.landmark)
        self.assertEqual("rig_a:person_bending:0", result.target_id)
        self.assertEqual("rig_a:trash_can:0", result.anchor_id)
        self.assertEqual("qwen", result.method)
        self.assertEqual(3, len(result.grounding.target_coord))
        self.assertNotEqual([0.0, 0.0, 0.0], result.grounding.target_coord)

    def test_sar_on_bench_uses_camera_ground_plane_without_height_guess(self):
        person = PersonCandidate(
            "rig_a:person_bending:0",
            (315.0, 100.0, 405.0, 390.0),
            "person_bending",
            0.9,
            "rig_a",
            (640, 480),
        )
        bench = ObjectCandidate(
            "rig_a:bench:0",
            "bench",
            (280.0, 180.0, 440.0, 410.0),
            0.9,
            "rig_a",
            (640, 480),
        )

        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[bench], people=[person], camera_ids=["rig_a"]
            )

        projector = _geometry()
        result = run_stage1_round(
            object(),
            "Find the person lying face-down on the bench.",
            detector=detector,
            verifier=lambda _board: person.candidate_id,
            projector=projector,
        )
        ground = projector.project_contact(
            "rig_a", person.bbox_xyxy, pose_class=person.pose_class
        )
        raised = projector.project_contact(
            "rig_a",
            person.bbox_xyxy,
            relation=Relation.ON,
            support_class="bench",
            pose_class=person.pose_class,
            support_box=bench.bbox_xyxy,
        )
        self.assertEqual(list(ground), result.grounding.target_coord)
        self.assertNotEqual(list(raised), result.grounding.target_coord)

    def test_sar_sitting_on_vending_keeps_existing_raised_support_geometry(self):
        person = PersonCandidate(
            "rig_a:person_sitting:0",
            (315.0, 100.0, 405.0, 390.0),
            "person_sitting",
            0.9,
            "rig_a",
            (640, 480),
        )
        vending = ObjectCandidate(
            "rig_a:vending_machine:0",
            "vending_machine",
            (280.0, 180.0, 440.0, 410.0),
            0.9,
            "rig_a",
            (640, 480),
        )

        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[vending], people=[person], camera_ids=["rig_a"]
            )

        projector = _geometry()
        result = run_stage1_round(
            object(),
            "Find the person sitting on top of the vending machine.",
            detector=detector,
            verifier=lambda _board: person.candidate_id,
            projector=projector,
        )
        expected = projector.project_contact(
            "rig_a",
            person.bbox_xyxy,
            relation=Relation.ON,
            support_class="vending_machine",
            pose_class=person.pose_class,
            support_box=vending.bbox_xyxy,
        )
        self.assertEqual(list(expected), result.grounding.target_coord)

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
            verifier=lambda board: "rig_a:person_bending:0|emergency",
            projector=empty,
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual(list(IGNORED_CLASS_DUMMY_XYZ), result.grounding.target_coord)
        self.assertNotEqual([0.0, 0.0, 0.0], result.grounding.target_coord)
        self.assertEqual("ignored_class_dummy", result.method)
        self.assertEqual("emergency", result.grounding.situation)
        self.assertEqual("trash_can", result.grounding.landmark)

    def test_umbrella_without_target_submits_dummy(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:vending_machine:0",
                        "vending_machine",
                        (200.0, 180.0, 420.0, 430.0),
                        0.92,
                        "rig_a",
                        (640, 480),
                    )
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Find the umbrella beside the vending machine.",
            detector=detector,
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual(list(IGNORED_CLASS_DUMMY_XYZ), result.grounding.target_coord)
        self.assertEqual(list(IGNORED_CLASS_DUMMY_XYZ), result.grounding.anchor_coord)
        self.assertEqual("umbrella", result.grounding.target_type)
        self.assertEqual("vending_machine", result.grounding.landmark)
        self.assertEqual("ignored_class_dummy", result.method)
        self.assertEqual("rig_a", result.grounding.camera_id)

    def test_pencilcase_without_target_submits_dummy(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:picnic_table:0",
                        "picnic_table",
                        (200.0, 180.0, 420.0, 430.0),
                        0.92,
                        "rig_a",
                        (640, 480),
                    )
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "I was taking notes at an outdoor table and left my pencil case behind.",
            detector=detector,
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual(list(IGNORED_CLASS_DUMMY_XYZ), result.grounding.target_coord)
        self.assertEqual("pencilcase", result.grounding.target_type)
        self.assertEqual("picnic_table", result.grounding.landmark)
        self.assertEqual("ignored_class_dummy", result.method)
        self.assertEqual("rig_a", result.grounding.camera_id)

    def test_bring_intent_reuses_object_pipeline(self):
        from marc_agent.command_parser import Intent

        def detector(client, *, stub_path=None):
            return _object_round()

        result = run_stage1_round(
            object(),
            "Bring the mug on the picnic table.",
            detector=detector,
            projector=_geometry(),
            expected_intent=Intent.BRING,
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("bring", result.candidates.parsed.intent.value)
        self.assertEqual("picnic_table", result.grounding.landmark)

    def test_stage1_accepts_an_explicit_bring_worded_grounding_round(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:tumbler:0", "tumbler", (40, 300, 80, 380),
                        0.7, "rig_a", (640, 480),
                    )
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Pick up the tumbler on the parking lot floor and bring it to the owner.",
            detector=detector,
            verifier=lambda _board: "rig_a:tumbler:0|none",
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("bring", result.candidates.parsed.intent.value)

    def test_relationless_named_landmark_submits_inferred_near_relation(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:pencilcase:0", "pencilcase", (260, 180, 310, 230),
                        0.7, "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:picnic_table:0", "picnic_table", (180, 230, 470, 410),
                        0.9, "rig_a", (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "I was taking notes at an outdoor table and left my pencil case behind.",
            detector=detector,
            verifier=lambda _board: (
                "rig_a:pencilcase:0|rig_a:picnic_table:0"
            ),
            projector=_geometry(),
        )
        self.assertEqual("near", result.grounding.relation)
        self.assertEqual("picnic_table", result.grounding.landmark)

    def test_parking_floor_drop_submits_nearest_vehicle_landmark(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:tumbler:0",
                        "tumbler",
                        (40.0, 300.0, 80.0, 380.0),
                        0.7,
                        "rig_a",
                        (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:police_car:0",
                        "police_car",
                        (90.0, 280.0, 400.0, 420.0),
                        0.9,
                        "rig_a",
                        (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Pick up the tumbler dropped on the parking lot floor and bring it to the owner.",
            detector=detector,
            projector=_geometry(),
            expected_intent=Intent.BRING,
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("police_car", result.grounding.landmark)
        self.assertEqual("near", result.grounding.relation)
        self.assertEqual("tumbler", result.grounding.target_type)
        self.assertEqual("rig_a:tumbler:0", result.target_id)
        self.assertNotEqual(result.grounding.anchor_coord, result.grounding.target_coord)

    def test_scene_inferred_park_anchor_submits_beside(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:sunblock:0", "sunblock", (80, 300, 100, 340),
                        0.9, "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:picnic_table:0", "picnic_table", (180, 220, 470, 420),
                        0.9, "rig_a", (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "I was resting in the sun at the park and left the sunblock I had been using.",
            detector=detector,
            projector=_geometry(),
        )
        self.assertEqual("picnic_table", result.grounding.landmark)
        self.assertEqual("beside", result.grounding.relation)

    def test_sar_without_landmark_does_not_project_nearby_object(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:cracker_box:0",
                        "cracker_box",
                        (300.0, 190.0, 430.0, 430.0),
                        0.9,
                        "rig_a",
                        (640, 480),
                    )
                ],
                people=[
                    PersonCandidate(
                        "rig_a:person_prone:0",
                        (30.0, 80.0, 130.0, 400.0),
                        "person_prone",
                        0.8,
                        "rig_a",
                        (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Several people are in the parking lot; one is completely face-down and immobile; find the most critical one.",
            detector=detector,
            verifier=lambda board: "rig_a:person_prone:0|accident",
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("", result.grounding.landmark)
        self.assertEqual("rig_a:person_prone:0", result.target_id)
        self.assertIsNone(result.anchor_id)
        self.assertEqual(result.grounding.anchor_coord, result.grounding.target_coord)

    def test_sar_scene_anchor_is_resolved_after_person_selection(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:taxi:0",
                        "taxi",
                        (300.0, 190.0, 430.0, 430.0),
                        0.9,
                        "rig_a",
                        (640, 480),
                    )
                ],
                people=[
                    PersonCandidate(
                        "rig_a:person_prone:0",
                        (30.0, 80.0, 130.0, 400.0),
                        "person_prone",
                        0.8,
                        "rig_a",
                        (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Several people are in the parking lot; one is completely face-down and immobile; find the most critical one.",
            detector=detector,
            verifier=lambda board: "rig_a:person_prone:0",
            projector=_geometry(),
        )

        self.assertIsNotNone(result.grounding)
        self.assertEqual("taxi", result.grounding.landmark)
        self.assertEqual("rig_a:person_prone:0", result.target_id)
        self.assertEqual("rig_a:taxi:0", result.anchor_id)
        self.assertNotEqual(
            result.grounding.anchor_coord,
            result.grounding.target_coord,
        )

    def test_pencilcase_does_not_submit_the_picnic_table(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:picnic_table:0",
                        "picnic_table",
                        (100.0, 200.0, 400.0, 400.0),
                        0.9,
                        "rig_a",
                        (640, 480),
                    )
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Find the pencil case near the picnic table.",
            detector=detector,
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("pencilcase", result.grounding.target_type)
        self.assertEqual("picnic_table", result.grounding.landmark)
        self.assertEqual("ignored_class_dummy", result.method)
        self.assertEqual(list(IGNORED_CLASS_DUMMY_XYZ), result.grounding.target_coord)
        self.assertEqual(list(IGNORED_CLASS_DUMMY_XYZ), result.grounding.anchor_coord)

    def test_empty_parking_space_submits_tumbler_with_nearest_car(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:tumbler:0",
                        "tumbler",
                        (100.0, 200.0, 130.0, 240.0),
                        0.8,
                        "rig_a",
                        (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:police_car:0",
                        "police_car",
                        (80.0, 160.0, 400.0, 420.0),
                        0.9,
                        "rig_a",
                        (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Find the tumbler dropped in the empty parking space.",
            detector=detector,
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertEqual("police_car", result.grounding.landmark)
        self.assertEqual("near", result.grounding.relation)
        self.assertEqual("tumbler", result.grounding.target_type)
        self.assertEqual("rig_a:tumbler:0", result.target_id)

    def test_sitting_on_vending_submits_person_relation_none(self):
        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:vending_machine:0",
                        "vending_machine",
                        (200.0, 180.0, 400.0, 420.0),
                        0.9,
                        "rig_a",
                        (640, 480),
                    )
                ],
                people=[
                    PersonCandidate(
                        "rig_a:person_sitting:0",
                        (240.0, 100.0, 320.0, 220.0),
                        "person_sitting",
                        0.9,
                        "rig_a",
                        (640, 480),
                    )
                ],
                camera_ids=["rig_a"],
            )

        result = run_stage1_round(
            object(),
            "Find the person sitting on top of the vending machine.",
            detector=detector,
            projector=_geometry(),
        )
        self.assertIsNotNone(result.grounding)
        self.assertIsNone(result.grounding.relation)
        self.assertEqual("vending_machine", result.grounding.landmark)
        self.assertEqual("rig_a:person_sitting:0", result.target_id)

    def test_clipped_low_confidence_target_projects_nearby_visible_proxy(self):
        table_box = (200.0, 220.0, 500.0, 420.0)
        proxy_box = (380.0, 150.0, 440.0, 260.0)

        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:pencilcase:0", "pencilcase",
                        (635.0, 300.0, 640.0, 340.0), 0.006,
                        "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:picnic_table:0", "picnic_table", table_box,
                        0.9, "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:trash_can:0", "trash_can", proxy_box,
                        0.9, "rig_a", (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        projector = _geometry()
        result = run_stage1_round(
            object(),
            "Find the pencil case near the picnic table.",
            detector=detector,
            projector=projector,
        )

        expected = projector.project_contact(
            "rig_a", (proxy_box[0], proxy_box[1], proxy_box[0], proxy_box[3])
        )
        self.assertEqual(list(expected), result.grounding.target_coord)

    def test_tiny_target_inside_other_object_projects_occluder(self):
        scooter_box = (200.0, 120.0, 320.0, 260.0)

        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:umbrella:0", "umbrella",
                        (215.0, 240.0, 226.0, 264.0), 0.9,
                        "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:taxi:0", "taxi", (300.0, 280.0, 600.0, 470.0),
                        0.9, "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:kick_scooter:0", "kick_scooter", scooter_box,
                        0.9, "rig_a", (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        projector = _geometry()
        result = run_stage1_round(
            object(),
            "Find the umbrella near the taxi.",
            detector=detector,
            projector=projector,
        )

        expected = projector.project_contact("rig_a", scooter_box)
        self.assertEqual(list(expected), result.grounding.target_coord)

    def test_tiny_tumbler_keeps_its_own_detection(self):
        tiny_box = (215.0, 240.0, 226.0, 264.0)
        scooter_box = (200.0, 120.0, 320.0, 260.0)

        def detector(client, *, stub_path=None):
            return CctvDetectionRound(
                objects=[
                    ObjectCandidate(
                        "rig_a:tumbler:0", "tumbler", tiny_box, 0.9,
                        "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:hydrant:0", "hydrant", (100, 200, 180, 400),
                        0.9, "rig_a", (640, 480),
                    ),
                    ObjectCandidate(
                        "rig_a:kick_scooter:0", "kick_scooter", scooter_box,
                        0.9, "rig_a", (640, 480),
                    ),
                ],
                camera_ids=["rig_a"],
            )

        projector = _geometry()
        result = run_stage1_round(
            object(),
            "Find the tumbler near the fire hydrant in the parking lot.",
            detector=detector,
            projector=projector,
        )

        expected = projector.project_contact("rig_a", tiny_box)
        self.assertEqual(list(expected), result.grounding.target_coord)


if __name__ == "__main__":
    unittest.main()
