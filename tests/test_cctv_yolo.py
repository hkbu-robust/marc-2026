"""Tests for bundled CCTV YOLO conversion and contract loading."""

from __future__ import annotations

import unittest

from marc_agent.perception.bundle_paths import STAGE1_CCTV_CONTRACT, STAGE1_CCTV_WEIGHTS
from marc_agent.perception.cctv_yolo import RuntimeContract, detections_to_candidates
from marc_agent.perception.pose_to_situation import situation_for_pose


class CctvYoloTests(unittest.TestCase):
    def test_bundle_files_exist(self):
        self.assertTrue(STAGE1_CCTV_WEIGHTS.is_file(), "model.pt should be in the repo bundle")
        self.assertTrue(STAGE1_CCTV_CONTRACT.is_file())

    def test_runtime_contract_loads(self):
        contract = RuntimeContract.load()
        self.assertEqual(1920, contract.imgsz)
        self.assertAlmostEqual(0.3, contract.confidence)
        self.assertEqual(33, len(contract.ordered_classes))
        self.assertIn("mug", contract.ordered_classes)
        self.assertIn("person_reclining", contract.ordered_classes)

    def test_detections_to_candidates_objects_only_by_default(self):
        detections = [
            ("mug", 0.91, (10.0, 20.0, 30.0, 40.0)),
            ("person_reclining", 0.88, (100.0, 100.0, 200.0, 300.0)),
        ]
        objects, sar = detections_to_candidates("rig_3_b", (1280, 720), detections)
        self.assertEqual(1, len(objects))
        self.assertEqual([], sar)
        self.assertEqual("mug", objects[0].class_name)

    def test_detections_to_candidates_can_include_yolo_sar(self):
        detections = [
            ("mug", 0.91, (10.0, 20.0, 30.0, 40.0)),
            ("person_reclining", 0.88, (100.0, 100.0, 200.0, 300.0)),
        ]
        objects, sar = detections_to_candidates(
            "rig_3_b", (1280, 720), detections, include_sar=True
        )
        self.assertEqual(1, len(objects))
        self.assertEqual(1, len(sar))
        self.assertEqual("abnormal", sar[0].situation)

    def test_pose_situation_mapping(self):
        self.assertEqual("accident", situation_for_pose("person_prone"))
        self.assertEqual("normal", situation_for_pose("person_sitting"))


if __name__ == "__main__":
    unittest.main()
