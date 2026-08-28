"""Tests for bundled CCTV YOLO conversion and contract loading."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from marc_agent.command_parser import parse_command
from marc_agent.perception.bundle_paths import STAGE1_CCTV_CONTRACT, STAGE1_CCTV_WEIGHTS
from marc_agent.perception.cctv_yolo import (
    CctvYoloDetector,
    RuntimeContract,
    detections_to_candidates,
    person_candidates_to_sar,
    refine_detection_round,
)
from marc_agent.perception.pose_to_situation import situation_for_pose
from marc_agent.types import (
    CctvDetectionRound,
    CctvFrame,
    ObjectCandidate,
    PersonCandidate,
)


class CctvYoloTests(unittest.TestCase):
    def test_bundle_files_exist(self):
        self.assertTrue(STAGE1_CCTV_WEIGHTS.is_file(), "model.pt should be in the repo bundle")
        self.assertTrue(STAGE1_CCTV_CONTRACT.is_file())

    def test_runtime_contract_loads(self):
        with patch.dict(os.environ, {"MARC_CCTV_YOLO_CONF": ""}):
            contract = RuntimeContract.load()
        payload = json.loads(STAGE1_CCTV_CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(1920, contract.imgsz)
        self.assertAlmostEqual(0.3, contract.confidence)
        self.assertEqual(33, len(contract.ordered_classes))
        self.assertIn("mug", contract.ordered_classes)
        self.assertIn("person_reclining", contract.ordered_classes)
        self.assertEqual(
            ["master_chef_can", "wood_block", "umbrella", "long_umbrella"],
            payload["ignored_classes"],
        )
        self.assertNotIn("master_chef_can", contract.ordered_classes)
        self.assertNotIn("wood_block", contract.ordered_classes)
        self.assertNotIn("umbrella", contract.ordered_classes)

    def test_runtime_contract_reads_confidence_from_env(self):
        with patch.dict(os.environ, {"MARC_CCTV_YOLO_CONF": "0.15"}):
            contract = RuntimeContract.load()
        self.assertAlmostEqual(0.15, contract.confidence)

    def test_submission_compose_does_not_override_contract_confidence(self):
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertNotIn("MARC_CCTV_YOLO_CONF:", compose)

    def test_detections_to_candidates_retains_people_by_default(self):
        detections = [
            ("mug", 0.91, (10.0, 20.0, 30.0, 40.0)),
            ("person_reclining", 0.88, (100.0, 100.0, 200.0, 300.0)),
        ]
        objects, people = detections_to_candidates("rig_3_b", (1280, 720), detections)
        self.assertEqual(1, len(objects))
        self.assertEqual(1, len(people))
        self.assertEqual("mug", objects[0].class_name)
        self.assertEqual("rig_3_b:person_reclining:0", people[0].candidate_id)
        self.assertEqual("person_reclining", people[0].pose_class)

    def test_person_ids_index_per_pose_class(self):
        detections = [
            ("person_sitting", 0.9, (10.0, 10.0, 20.0, 40.0)),
            ("person_sitting", 0.8, (30.0, 10.0, 40.0, 40.0)),
            ("person_lying_down", 0.85, (50.0, 10.0, 80.0, 30.0)),
        ]
        _, people = detections_to_candidates("rig_a", (1280, 720), detections)
        self.assertEqual(
            [
                "rig_a:person_sitting:0",
                "rig_a:person_sitting:1",
                "rig_a:person_lying_down:0",
            ],
            [item.candidate_id for item in people],
        )

    def test_pose_only_sar_is_an_explicit_adapter(self):
        detections = [
            ("mug", 0.91, (10.0, 20.0, 30.0, 40.0)),
            ("person_reclining", 0.88, (100.0, 100.0, 200.0, 300.0)),
        ]
        objects, people = detections_to_candidates("rig_3_b", (1280, 720), detections)
        sar = person_candidates_to_sar(people)
        self.assertEqual(1, len(objects))
        self.assertEqual(1, len(sar))
        self.assertEqual("abnormal", sar[0].situation)

    def test_pose_situation_mapping(self):
        self.assertEqual("accident", situation_for_pose("person_prone"))
        self.assertEqual("normal", situation_for_pose("person_sitting"))

    def test_detector_processes_every_live_compatible_frame_once(self):
        detector = CctvYoloDetector(RuntimeContract.load())
        detector._predict_frame = Mock(
            side_effect=[
                [("person_prone", 0.9, (10.0, 20.0, 30.0, 40.0))],
                [("mug", 0.8, (50.0, 60.0, 80.0, 100.0))],
            ]
        )
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        frames = {
            "rig_b": CctvFrame("rig_b", 64, 48, rgb.copy()),
            "rig_a": CctvFrame("rig_a", 64, 48, rgb.copy()),
        }

        objects, people = detector.detect(frames)

        self.assertEqual(2, detector._predict_frame.call_count)
        self.assertEqual("rig_b:mug:0", objects[0].candidate_id)
        self.assertEqual("rig_a:person_prone:0", people[0].candidate_id)

    def test_recorded_bundle_output_preserves_every_candidate(self):
        example_path = (
            STAGE1_CCTV_CONTRACT.parent / "candidate_output_example.json"
        )
        payload = json.loads(Path(example_path).read_text(encoding="utf-8"))
        detections = [
            (
                item["class_name"],
                item["confidence"],
                tuple(item["xyxy"]),
            )
            for item in payload["candidates"]
        ]
        height, width = payload["image_shape"]

        objects, people = detections_to_candidates(
            payload["camera"],
            (width, height),
            detections,
        )

        self.assertEqual(len(detections), len(objects) + len(people))
        self.assertTrue(people)
        self.assertTrue(all(item.camera_id == "rig_3_b" for item in people))

    def test_landmark_crop_maps_boxes_back_to_the_full_frame(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {8: "pencilcase"}

        def fake_predict(rgb, *, confidence=None):
            self.assertLess(confidence, detector.contract.confidence)
            return [("pencilcase", 0.4, (5.0, 6.0, 15.0, 18.0))]

        detector._predict_frame = fake_predict
        frame = CctvFrame(
            "rig_3_b",
            1280,
            720,
            np.zeros((720, 1280, 3), dtype=np.uint8),
        )
        table = detections_to_candidates(
            "rig_3_b",
            (1280, 720),
            [("picnic_table", 0.9, (100.0, 200.0, 400.0, 500.0))],
        )[0][0]
        extra = detector.refine_around_landmarks(
            {"rig_3_b": frame},
            [table],
            "pencilcase",
            frozenset({"picnic_table"}),
        )
        self.assertEqual(1, len(extra))
        self.assertEqual("pencilcase", extra[0].class_name)
        self.assertEqual("rig_3_b", extra[0].camera_id)
        self.assertGreaterEqual(extra[0].bbox_xyxy[0], 5.0)

    def test_landmark_crop_recovers_a_missed_person(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {8: "person_reclining"}

        def fake_predict(rgb, *, confidence=None):
            self.assertLess(confidence, detector.contract.confidence)
            return [("person_reclining", 0.4, (5.0, 6.0, 55.0, 66.0))]

        detector._predict_frame = fake_predict
        frame = CctvFrame(
            "rig_3_b",
            1280,
            720,
            np.zeros((720, 1280, 3), dtype=np.uint8),
        )
        bench = ObjectCandidate(
            "rig_3_b:bench:0",
            "bench",
            (100.0, 200.0, 400.0, 500.0),
            0.9,
            "rig_3_b",
            (1280, 720),
        )

        extra = detector.refine_people_around_landmarks(
            {"rig_3_b": frame},
            [bench],
            [],
            frozenset({"bench"}),
        )

        self.assertEqual(1, len(extra))
        self.assertEqual("rig_3_b:person_reclining:0", extra[0].candidate_id)
        self.assertEqual("person_reclining", extra[0].pose_class)
        self.assertEqual("rig_3_b", extra[0].camera_id)

    def test_landmark_person_refinement_deduplicates_full_frame_people(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {8: "person_sitting"}
        detector._predict_frame = Mock(
            return_value=[("person_sitting", 0.4, (96.0, 96.0, 196.0, 296.0))]
        )
        frame = CctvFrame(
            "rig_a",
            640,
            480,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        trash_can = ObjectCandidate(
            "rig_a:trash_can:0",
            "trash_can",
            (300.0, 200.0, 400.0, 400.0),
            0.9,
            "rig_a",
            (640, 480),
        )
        existing = PersonCandidate(
            "rig_a:person_sitting:0",
            (300.0, 200.0, 400.0, 400.0),
            "person_sitting",
            0.8,
            "rig_a",
            (640, 480),
        )

        extra = detector.refine_people_around_landmarks(
            {"rig_a": frame},
            [trash_can],
            [existing],
            frozenset({"trash_can"}),
        )

        self.assertEqual([], extra)

    def test_person_round_runs_support_refinement_even_with_other_people(self):
        parsed = parse_command("Find the person lying on top of the bench.")
        bench = ObjectCandidate(
            "rig_a:bench:0",
            "bench",
            (100.0, 200.0, 400.0, 400.0),
            0.9,
            "rig_a",
            (640, 480),
        )
        distractor = PersonCandidate(
            "rig_a:person_walking:0",
            (20.0, 40.0, 80.0, 300.0),
            "person_walking",
            0.9,
            "rig_a",
            (640, 480),
        )
        recovered = PersonCandidate(
            "rig_a:person_lying_down:0",
            (120.0, 160.0, 360.0, 260.0),
            "person_lying_down",
            0.4,
            "rig_a",
            (640, 480),
        )
        round_ = CctvDetectionRound(
            objects=[bench],
            people=[distractor],
            camera_ids=["rig_a"],
        )
        detector = Mock()
        detector.refine_people_around_landmarks.return_value = [recovered]

        with patch("marc_agent.perception.cctv_yolo._detector", detector):
            refined = refine_detection_round(parsed, round_)

        self.assertEqual([distractor, recovered], refined.people)
        detector.refine_people_around_landmarks.assert_called_once_with(
            round_.frames,
            round_.objects,
            round_.people,
            frozenset({"bench"}),
        )


if __name__ == "__main__":
    unittest.main()
