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
    _letterbox_imgsz,
    _pad_box,
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
        self.assertEqual(34, len(contract.ordered_classes))
        self.assertIn("mug", contract.ordered_classes)
        self.assertIn("person_reclining", contract.ordered_classes)
        self.assertEqual(
            ["long_umbrella", "master_chef_can", "wood_block"],
            payload["ignored_classes"],
        )
        self.assertNotIn("master_chef_can", contract.ordered_classes)
        self.assertNotIn("wood_block", contract.ordered_classes)
        self.assertIn("umbrella", contract.ordered_classes)

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
            for item in payload["raw_global_candidates"]
        ]
        height, width = payload["image_shape"]

        objects, people = detections_to_candidates(
            payload["camera"],
            (width, height),
            detections,
        )

        self.assertEqual(len(detections), len(objects) + len(people))
        self.assertTrue(people)
        self.assertTrue(all(item.camera_id == payload["camera"] for item in people))

    def test_landmark_crop_maps_boxes_back_to_the_full_frame(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {8: "pencilcase"}

        def fake_predict(rgb, *, confidence=None, **_kwargs):
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

    def test_object_refine_crops_every_bench_on_cameras_missing_target(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {8: "tumbler"}
        detector._predict_frame = Mock(return_value=[])
        frames = {
            "rig_has": CctvFrame(
                "rig_has", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
            ),
            "rig_miss": CctvFrame(
                "rig_miss", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8)
            ),
        }
        objects = [
            ObjectCandidate(
                "rig_has:tumbler:0",
                "tumbler",
                (40.0, 40.0, 80.0, 90.0),
                0.8,
                "rig_has",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_has:bench:0",
                "bench",
                (20.0, 20.0, 200.0, 180.0),
                0.9,
                "rig_has",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_miss:bench:0",
                "bench",
                (30.0, 40.0, 120.0, 160.0),
                0.4,
                "rig_miss",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_miss:bench:1",
                "bench",
                (300.0, 80.0, 500.0, 260.0),
                0.9,
                "rig_miss",
                (640, 480),
            ),
        ]
        extra = detector.refine_around_landmarks(
            frames, objects, "tumbler", frozenset({"bench"})
        )
        self.assertEqual([], extra)
        self.assertEqual(2, detector._predict_frame.call_count)

    def test_object_refine_runs_when_another_camera_already_has_target(self):
        parsed = parse_command("Find the tumbler on the bench.")
        missed = ObjectCandidate(
            "rig_1:tumbler:0",
            "tumbler",
            (200.0, 200.0, 240.0, 250.0),
            0.2,
            "rig_1",
            (640, 480),
        )
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate(
                    "rig_3:tumbler:0",
                    "tumbler",
                    (10.0, 300.0, 40.0, 340.0),
                    0.9,
                    "rig_3",
                    (640, 480),
                ),
                ObjectCandidate(
                    "rig_3:bench:0",
                    "bench",
                    (200.0, 300.0, 400.0, 420.0),
                    0.9,
                    "rig_3",
                    (640, 480),
                ),
                ObjectCandidate(
                    "rig_1:bench:0",
                    "bench",
                    (180.0, 180.0, 400.0, 360.0),
                    0.9,
                    "rig_1",
                    (640, 480),
                ),
            ],
            camera_ids=["rig_3", "rig_1"],
        )
        detector = Mock()
        detector.refine_around_landmarks.return_value = [missed]
        with patch("marc_agent.perception.cctv_yolo._detector", detector):
            refined = refine_detection_round(parsed, round_)
        detector.refine_around_landmarks.assert_called_once()
        self.assertEqual(missed, refined.objects[-1])

    def test_landmark_crop_recovers_a_missed_person(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {8: "person_reclining"}

        def fake_predict(rgb, *, confidence=None, **_kwargs):
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
            return_value=[("person_sitting", 0.4, (30.0, 30.0, 130.0, 230.0))]
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

    def test_pad_box_uses_box_size_not_frame_size(self):
        padded = _pad_box((100.0, 200.0, 200.0, 300.0), (1920.0, 1080.0))
        self.assertEqual((85.0, 185.0, 215.0, 315.0), padded)

    def test_letterbox_imgsz_caps_refine_crops(self):
        self.assertEqual(224, _letterbox_imgsz(200, 180, cap=640))
        self.assertEqual(640, _letterbox_imgsz(800, 800, cap=640))
        self.assertEqual(1920, _letterbox_imgsz(1920, 1080, cap=1920))

    def test_small_crop_predict_uses_capped_imgsz(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.device = "cpu"
        detector.model = Mock()
        detector.names = {0: "mug"}
        result = Mock()
        result.boxes = None
        detector.model.predict.return_value = [result]
        crop = np.zeros((120, 80, 3), dtype=np.uint8)
        detector._predict_frame(crop, confidence=0.12, imgsz=224)
        self.assertEqual(224, detector.model.predict.call_args.kwargs["imgsz"])

    def test_full_frame_predict_keeps_contract_imgsz(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.device = "cpu"
        detector.model = Mock()
        detector.names = {0: "mug"}
        result = Mock()
        result.boxes = None
        detector.model.predict.return_value = [result]
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector._predict_frame(frame)
        self.assertEqual(1920, detector.model.predict.call_args.kwargs["imgsz"])

    def test_refine_uses_one_landmark_per_camera(self):
        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {}
        detector._predict_frame = Mock(return_value=[])
        frame = CctvFrame(
            "rig_a",
            640,
            480,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        bikes = [
            ObjectCandidate(
                "rig_a:bicycle:0",
                "bicycle",
                (40.0, 40.0, 80.0, 120.0),
                0.4,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_a:bicycle:1",
                "bicycle",
                (200.0, 80.0, 280.0, 200.0),
                0.9,
                "rig_a",
                (640, 480),
            ),
        ]
        detector.refine_people_around_landmarks(
            {"rig_a": frame},
            bikes,
            [],
            frozenset({"bicycle"}),
        )
        self.assertEqual(1, detector._predict_frame.call_count)
        crop = detector._predict_frame.call_args.args[0]
        self.assertLess(crop.shape[0], 480)
        self.assertLess(crop.shape[1], 640)
        self.assertLessEqual(detector._predict_frame.call_args.kwargs["imgsz"], 640)

    def test_refine_oom_keeps_full_frame_detections(self):
        parsed = parse_command(
            "Find the person fallen on the ground next to the toppled bicycle."
        )
        bike = ObjectCandidate(
            "rig_a:bicycle:0",
            "bicycle",
            (100.0, 200.0, 200.0, 400.0),
            0.9,
            "rig_a",
            (640, 480),
        )
        person = PersonCandidate(
            "rig_a:person_walking:0",
            (20.0, 40.0, 80.0, 300.0),
            "person_walking",
            0.9,
            "rig_a",
            (640, 480),
        )
        round_ = CctvDetectionRound(
            objects=[bike],
            people=[person],
            camera_ids=["rig_a"],
        )
        detector = Mock()
        detector.refine_people_around_landmarks.side_effect = RuntimeError(
            "CUDA out of memory. Tried to allocate 158.00 MiB"
        )
        with patch("marc_agent.perception.cctv_yolo._detector", detector):
            refined = refine_detection_round(parsed, round_)
        self.assertIs(round_, refined)
        self.assertEqual([person], refined.people)


if __name__ == "__main__":
    unittest.main()
