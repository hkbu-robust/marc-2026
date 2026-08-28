#!/usr/bin/env python3
"""SAR / on-landmark labelling for the CCTV → Qwen2-VL converter."""

from __future__ import annotations

import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_CONV = os.path.join(_ROOT, "demo", "vlm_dataset")
if _CONV not in sys.path:
    sys.path.insert(0, _CONV)

import cctv_gt_to_qwen as q  # noqa: E402


def _det(cls, kind, bbox, xyz=None):
    d = {"class": cls, "kind": kind, "bbox": bbox}
    if xyz is not None:
        d["world_xyz"] = list(xyz)
    return d


def _payload(dets):
    return {"image": {"width": 1280, "height": 720}, "detections": dets}


class SarLabelTests(unittest.TestCase):
    def test_sitting_on_vending_is_not_normal(self):
        person = _det("person_sitting", "object", [600, 200, 700, 420],
                      xyz=(-62.77, 135.78, 17.94))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        bench = _det("bench", "landmark", [100, 400, 220, 520],
                     xyz=(-50.0, 140.0, 16.2))
        rng = random.Random(0)
        rows = q._sar_pairs_for_payload(
            _payload([person, vendor, bench]), rng, 4)
        self.assertTrue(rows)
        instruction, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "vending_machine")
        self.assertIn(rel, ("on", "above"))
        self.assertIn(sit, ("abnormal", "emergency", "accident"))
        self.assertNotEqual(sit, "normal")
        low = instruction.lower()
        self.assertTrue(
            "drink machine" in low or "vending machine" in low,
            instruction,
        )
        self.assertTrue("climbed" in low or "on top" in low, instruction)

    def test_above_wrong_place_is_always_abnormal(self):
        person = _det("person_reaching", "object", [600, 100, 700, 420],
                      xyz=(-62.81, 135.53, 18.75))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        for seed in range(20):
            rows = q._sar_pairs_for_payload(
                _payload([person, vendor]), random.Random(seed), 1)
            self.assertTrue(rows)
            _, _, _, situation, _, relation = rows[0]
            self.assertEqual(relation, "above")
            self.assertEqual(situation, "abnormal")

    def test_on_wrong_place_is_always_abnormal(self):
        person = _det("person_crouching", "object", [600, 200, 700, 420],
                      xyz=(-62.81, 135.53, 16.70))
        trash = _det("trash_can", "landmark", [580, 280, 720, 500],
                     xyz=(-62.81, 135.53, 16.20))
        for seed in range(20):
            rows = q._sar_pairs_for_payload(
                _payload([person, trash]), random.Random(seed), 1)
            _, _, _, situation, _, relation = rows[0]
            self.assertEqual(relation, "on")
            self.assertEqual(situation, "abnormal")

    def test_sitting_on_bench_is_normal(self):
        person = _det("person_sitting", "object", [100, 300, 180, 500],
                      xyz=(0.0, 0.0, 16.75))
        bench = _det("bench", "landmark", [80, 360, 200, 520],
                     xyz=(0.1, 0.05, 16.50))
        rng = random.Random(1)
        rows = q._sar_pairs_for_payload(_payload([person, bench]), rng, 2)
        self.assertTrue(rows)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "bench")
        self.assertEqual(sit, "normal")
        self.assertIn(rel, ("on", "above"))

    def test_sitting_beside_vending_on_ground_is_normal(self):
        # Same z as the machine origin → not on top (person on the ground).
        person = _det("person_sitting", "object", [200, 300, 280, 520],
                      xyz=(-64.5, 135.5, 16.20))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        rng = random.Random(2)
        rows = q._sar_pairs_for_payload(_payload([person, vendor]), rng, 2)
        self.assertTrue(rows)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "vending_machine")
        self.assertIsNone(rel)
        self.assertEqual(sit, "normal")

    def test_standing_beside_hydrant_is_not_on(self):
        # Old 1.6m XY / 0.12m Z window treated this pelvis-above-base pair as
        # "on" the hydrant. Side-by-side boxes must stay beside, not abnormal.
        person = _det("person_reaching", "object", [120, 200, 220, 520],
                      xyz=(-62.0, 135.5, 17.10))
        hydrant = _det("hydrant", "landmark", [400, 360, 460, 520],
                       xyz=(-61.2, 135.4, 16.20))
        rng = random.Random(8)
        rows = q._sar_pairs_for_payload(_payload([person, hydrant]), rng, 1)
        self.assertTrue(rows)
        _, _, _, sit, _, rel = rows[0]
        self.assertIsNone(rel)
        self.assertNotEqual(sit, "abnormal")

    def test_standing_in_front_of_vending_is_not_on(self):
        # Overlapping x, shared ground line, pelvis higher than machine origin.
        person = _det("person_walking", "object", [590, 240, 710, 500],
                      xyz=(-62.85, 135.50, 17.05))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        rng = random.Random(9)
        rows = q._sar_pairs_for_payload(_payload([person, vendor]), rng, 1)
        self.assertTrue(rows)
        _, _, _, sit, _, rel = rows[0]
        self.assertIsNone(rel)
        self.assertNotEqual(sit, "abnormal")

    def test_lying_on_picnic_table_is_abnormal(self):
        person = _det("person_lying_down", "object", [400, 200, 560, 320],
                      xyz=(-80.9, 96.5, 16.83))
        table = _det("picnic_table", "landmark", [380, 240, 580, 360],
                     xyz=(-80.7, 96.4, 15.65))
        rng = random.Random(3)
        rows = q._sar_pairs_for_payload(_payload([person, table]), rng, 2)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "picnic_table")
        self.assertEqual(sit, "abnormal")
        self.assertIn(rel, ("on", "above"))

    def test_lying_on_bench_is_abnormal(self):
        person = _det("person_lying_down", "object", [400, 200, 560, 320],
                      xyz=(-84.9, 111.9, 16.00))
        bench = _det("bench", "landmark", [380, 240, 580, 360],
                     xyz=(-84.1, 112.5, 15.48))
        rng = random.Random(4)
        rows = q._sar_pairs_for_payload(_payload([person, bench]), rng, 2)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(sit, "abnormal")
        self.assertIn(rel, ("on", "above"))

    def test_prone_on_ground_is_distress(self):
        person = _det("person_prone", "object", [200, 400, 360, 520],
                      xyz=(-49.3, 125.7, 16.74))
        bike = _det("bicycle", "landmark", [180, 430, 300, 540],
                    xyz=(-48.6, 128.8, 17.02))
        rng = random.Random(5)
        rows = q._sar_pairs_for_payload(_payload([person, bike]), rng, 2)
        _, _, _, sit, _, rel = rows[0]
        self.assertIsNone(rel)
        self.assertEqual(sit, "accident")

    def test_prefers_climber_over_ground_sitter(self):
        climber = _det("person_sitting", "object", [600, 200, 700, 420],
                       xyz=(-62.77, 135.78, 17.94))
        sitter = _det("person_sitting", "object", [100, 300, 180, 500],
                      xyz=(0.0, 0.0, 16.75))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        bench = _det("bench", "landmark", [80, 360, 200, 520],
                     xyz=(0.1, 0.05, 16.50))
        rng = random.Random(6)
        rows = q._sar_pairs_for_payload(
            _payload([sitter, climber, vendor, bench]), rng, 1)
        self.assertEqual(len(rows), 1)
        _, person, lm, sit, _, rel = rows[0]
        self.assertIs(person, climber)
        self.assertEqual(lm["class"], "vending_machine")
        self.assertNotEqual(sit, "normal")
        self.assertIn(rel, ("on", "above"))

    def test_object_on_uses_on_relation(self):
        cup = _det("disposable_cup", "object", [400, 300, 440, 360],
                   xyz=(1.0, 2.0, 16.9))
        bench = _det("bench", "landmark", [360, 320, 500, 420],
                     xyz=(1.05, 2.02, 16.5))
        rng = random.Random(7)
        rows = q._pairs_for_payload(_payload([cup, bench]), rng, 2)
        self.assertTrue(rows)
        instruction, _, lm, rel, _ = rows[0]
        self.assertEqual(lm["class"], "bench")
        self.assertEqual(rel, "on")
        self.assertIn(" on ", instruction.lower())

    def test_hanging_leg_vending_sitter_is_on(self):
        # Full-frame crop: person box shares the vending ground line, so the
        # 2D feet-above-base test fails. World Z still says they are on the roof.
        person = _det("person_sitting", "object", [219, 0, 1019, 719],
                      xyz=(-72.60, 135.18, 17.57))
        vendor = _det("vending_machine", "landmark", [0, 276, 1019, 719],
                      xyz=(-72.06, 134.99, 16.01))
        self.assertFalse(q._stacked_on_top(person, vendor))
        rows = q._sar_pairs_for_payload(
            _payload([person, vendor]), random.Random(0), 1)
        self.assertTrue(rows)
        _, _, lm, sit, _, rel = rows[0]
        self.assertEqual(lm["class"], "vending_machine")
        self.assertIn(rel, ("on", "above"))
        self.assertNotEqual(sit, "normal")

    def test_reaching_on_vending_does_not_say_sitting(self):
        person = _det("person_reaching", "object", [600, 100, 700, 420],
                      xyz=(-62.81, 135.53, 18.75))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        for seed in range(40):
            rows = q._sar_pairs_for_payload(
                _payload([person, vendor]), random.Random(seed), 1)
            instruction = rows[0][0].lower()
            self.assertNotIn("sitting", instruction, instruction)

    def test_object_left_of_landmark(self):
        cup = _det("cola_can", "object", [200, 400, 240, 450],
                   xyz=(-1.0, 0.0, 16.5))
        bench = _det("bench", "landmark", [400, 400, 560, 500],
                     xyz=(0.0, 0.0, 16.5))
        payload = {
            "image": {"width": 1280, "height": 720},
            "camera": {"position": [0.0, -10.0, 20.0]},
            "detections": [cup, bench],
        }
        rows = q._pairs_for_payload(payload, random.Random(0), 2)
        self.assertTrue(rows)
        instruction, _, lm, rel, _ = rows[0]
        self.assertEqual(lm["class"], "bench")
        self.assertEqual(rel, "left")
        self.assertIn("left", instruction.lower())

    def test_no_landmark_prone_is_accident(self):
        person = _det("person_prone", "object", [200, 400, 360, 520],
                      xyz=(-49.3, 125.7, 16.74))
        rows = q._sar_pairs_for_payload(
            _payload([person]), random.Random(0), 1)
        self.assertTrue(rows)
        _, target, lm, sit, _, rel = rows[0]
        self.assertIs(target, person)
        self.assertIsNone(lm)
        self.assertIsNone(rel)
        self.assertEqual(sit, "accident")

    def test_crouch_is_abnormal(self):
        person = _det("person_crouching", "object", [200, 300, 280, 520],
                      xyz=(-64.5, 135.5, 16.20))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        rows = q._sar_pairs_for_payload(
            _payload([person, vendor]), random.Random(0), 1)
        self.assertTrue(rows)
        _, _, _, sit, _, rel = rows[0]
        self.assertIsNone(rel)
        self.assertEqual(sit, "abnormal")

    def test_unlabeled_lying_down_is_dropped(self):
        person = _det("person_lying_down", "object", [200, 400, 360, 520],
                      xyz=(-49.3, 125.7, 16.74))
        walker = _det("person_walking", "object", [100, 300, 180, 500],
                      xyz=(0.0, 0.0, 16.75))
        rows = q._sar_pairs_for_payload(
            _payload([person]), random.Random(0), 2)
        self.assertEqual(rows, [])
        rows = q._sar_pairs_for_payload(
            _payload([person, walker]), random.Random(0), 2)
        self.assertTrue(rows)
        targets = [row[1]["class"] for row in rows]
        self.assertNotIn("person_lying_down", targets)
        self.assertIn("person_walking", targets)

    def test_labeled_lying_down_on_parking_is_kept(self):
        person = _det("person_lying_down", "object", [200, 400, 360, 520],
                      xyz=(-49.3, 125.7, 16.74))
        person["surface_class"] = "parking"
        rows = q._sar_pairs_for_payload(
            _payload([person]), random.Random(0), 1)
        self.assertTrue(rows)
        self.assertEqual(rows[0][3], "emergency")

    def test_recipe_target_frame_uses_runtime_verifier_prompt(self):
        person = _det("person_sitting", "object", [600, 200, 700, 420],
                      xyz=(-62.77, 135.78, 17.94))
        person["recipe_id"] = "sar02_vending_sit"
        person["command"] = "Find the person sitting on top of the vending machine."
        person["situation"] = "abnormal"
        walker = _det("person_walking", "object", [100, 300, 180, 500],
                      xyz=(0.0, 0.0, 16.75))
        vendor = _det("vending_machine", "landmark", [580, 280, 720, 500],
                      xyz=(-62.81, 135.53, 16.20))
        payload = _payload([walker, person, vendor])
        people = [walker, person]
        ids = q.person_candidate_id_list(people, "cam")
        cid = q.person_candidate_ids(people, "cam")[id(person)]
        row = q._sample(
            "/tmp/fake.png", payload, "cam", person["command"],
            person, vendor, "llama-factory", situation="abnormal",
            candidate_id=cid, candidate_ids=ids, people=people)
        user = row["messages"][0]["content"]
        self.assertIn("Original request:", user)
        self.assertIn("Candidates:", user)
        self.assertIn("pose=", user)
        self.assertIn("anchor=", user)
        self.assertIn("distance=", user)
        self.assertIn("overlap=", user)
        self.assertIn(cid, user)
        self.assertEqual(row["messages"][1]["content"], f"{cid}|abnormal")

    def test_missing_recipe_target_camera_emits_no_sar(self):
        walker = _det("person_walking", "object", [100, 300, 180, 500],
                      xyz=(0.0, 0.0, 16.75))
        rows = q._sar_pairs_for_payload(
            _payload([walker]), random.Random(0), 2,
            require_recipe_target=True)
        self.assertEqual(rows, [])

    def test_far_gt_support_is_ignored(self):
        person = _det("person_lying_down", "object", [200, 400, 360, 520],
                      xyz=(0.0, 0.0, 16.74))
        person["support_class"] = "bicycle"
        person["surface_class"] = "parking"
        person["recipe_id"] = "sar08_fallenbike_down"
        bike = _det("bicycle", "landmark", [1000, 10, 1100, 80],
                    xyz=(20.0, 20.0, 17.02))
        rows = q._sar_pairs_for_payload(
            _payload([person, bike]), random.Random(0), 1)
        self.assertTrue(rows)
        self.assertNotEqual(rows[0][3], "emergency")

    def test_convert_writes_split_sar_and_obj_jsonl(self):
        import json
        import shutil
        import tempfile
        from pathlib import Path

        tmp = tempfile.mkdtemp()
        try:
            scene = Path(tmp) / "scene_00001"
            scene.mkdir()
            (scene / "cam_tgt.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (scene / "cam_walk.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            person = _det("person_prone", "object", [10, 20, 30, 40],
                          xyz=(0.0, 0.0, 16.7))
            person["recipe_id"] = "sar07_parking_prone"
            person["surface_class"] = "parking"
            person["command"] = (
                "Find the person lying face-down and motionless in the parking lot."
            )
            walker = _det("person_walking", "object", [100, 300, 180, 500],
                          xyz=(1.0, 1.0, 16.7))
            cup = _det("disposable_cup", "object", [400, 300, 440, 360],
                       xyz=(1.0, 2.0, 16.9))
            bench = _det("bench", "landmark", [360, 320, 500, 420],
                         xyz=(1.05, 2.02, 16.5))
            tgt = _payload([person, walker, cup, bench])
            tgt["frame_id"] = "cam_tgt"
            walk = _payload([walker])
            walk["frame_id"] = "cam_walk"
            (scene / "cam_tgt.json").write_text(json.dumps(tgt))
            (scene / "cam_walk.json").write_text(json.dumps(walk))
            out = Path(tmp) / "out"
            q.convert(tmp, str(out), 0.0, 4, 0, "llama-factory")
            self.assertTrue((out / "train_sar.jsonl").exists())
            self.assertTrue((out / "val_sar.jsonl").exists())
            self.assertTrue((out / "train_obj.jsonl").exists())
            self.assertTrue((out / "val_obj.jsonl").exists())
            self.assertFalse((out / "train.jsonl").exists())
            sar_lines = (out / "train_sar.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(sar_lines), 1)
            sar = json.loads(sar_lines[0])
            user = sar["messages"][0]["content"]
            self.assertIn("Original request:", user)
            self.assertIn("Candidates:", user)
            self.assertIn("|accident", sar["messages"][1]["content"])
            obj_text = (out / "train_obj.jsonl").read_text()
            self.assertTrue(obj_text.strip())
            self.assertIn("disposable_cup", obj_text)
            self.assertNotIn("|accident", obj_text)
        finally:
            shutil.rmtree(tmp)

    def test_sar_assistant_is_id_pipe_situation_without_coords(self):
        person = _det("person_prone", "object", [10, 20, 30, 40])
        walker = _det("person_walking", "object", [100, 100, 200, 300])
        people = [walker, person]
        ids = q.person_candidate_id_list(people, "cam")
        self.assertEqual(ids[0], "cam:person_prone:0")
        self.assertEqual(ids[1], "cam:person_walking:0")
        cid = q.person_candidate_ids(people, "cam")[id(person)]
        self.assertEqual(cid, "cam:person_prone:0")
        row = q._sample(
            "/tmp/fake.png", {}, "cam", "Find the person lying face-down.",
            person, None, "llama-factory", situation="accident",
            candidate_id=cid, candidate_ids=ids)
        assistant = row["messages"][1]["content"]
        user = row["messages"][0]["content"]
        self.assertEqual(assistant, f"{cid}|accident")
        self.assertNotIn("grounding", assistant)
        self.assertNotIn("coord", assistant.lower())
        self.assertIn(cid, user)
        self.assertIn("Candidates:", user)

    def test_object_assistant_has_no_grounding_xyz(self):
        cup = _det("disposable_cup", "object", [400, 300, 440, 360],
                   xyz=(1.0, 2.0, 16.9))
        bench = _det("bench", "landmark", [360, 320, 500, 420],
                     xyz=(1.05, 2.02, 16.5))
        row = q._sample(
            "/tmp/fake.png", {}, "cam", "Find the disposable cup on the bench.",
            cup, bench, "llama-factory", rel="on")
        assistant = row["messages"][1]["content"]
        self.assertNotIn("grounding", assistant)
        self.assertNotIn("target_coord", assistant)
        self.assertNotIn("anchor_coord", assistant)
        self.assertIn("disposable_cup", assistant)

    def test_ambiguous_near_clones_are_skipped(self):
        cup_a = _det("cola_can", "object", [200, 400, 240, 450],
                     xyz=(-2.0, 0.0, 16.5))
        cup_b = _det("cola_can", "object", [260, 400, 300, 450],
                     xyz=(-1.5, 0.0, 16.5))
        bench = _det("bench", "landmark", [400, 400, 560, 500],
                     xyz=(0.0, 0.0, 16.5))
        rows = q._pairs_for_payload(
            _payload([cup_a, cup_b, bench]), random.Random(0), 4)
        self.assertEqual(rows, [])

    def test_long_umbrella_skipped_ordinary_umbrella_kept(self):
        self.assertTrue(q._is_long_umbrella("long_umbrella"))
        self.assertTrue(q._is_long_umbrella("closedlongumbrella"))
        self.assertFalse(q._is_long_umbrella("umbrella"))
        self.assertFalse(q._is_long_umbrella("master_chef_can"))


if __name__ == "__main__":
    unittest.main()
