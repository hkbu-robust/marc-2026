"""Frozen evaluation sets: relation rule, gold matching, geometry, and the freeze contract (#87)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.frozen_eval import case_store
from scripts.frozen_eval.geometry import geometry_from_gt, ground_point, project_point
from scripts.frozen_eval.heldout import assemble_case, stratum
from scripts.frozen_eval.public import propose
from scripts.frozen_eval.relations import object_choices, relations_between, sar_check

# rig_3_b as the generator reports it.
PAYLOAD = {
    "frame_id": "rig_3_b",
    "image": {"width": 1280, "height": 720},
    "camera": {
        "position": [-84.50272, 98.49848, 18.77523],
        "orientation_euler_deg": {"roll": 9.731, "pitch": -61.02, "yaw": -79.413},
        "focal_length_mm": 5.0, "horizontal_aperture_mm": 5.6, "vertical_aperture_mm": 3.15,
        "ground_height": 16.1,
    },
}


def _det(cls, iid, bbox, xyz, kind="object"):
    return {"class": cls, "instance_id": iid, "kind": kind, "bbox": bbox, "world_xyz": xyz}


class GeometryTests(unittest.TestCase):
    def test_gt_object_projects_into_its_box(self):
        # cracker_box_0 of a generated scene: GT box [1182, 476, 1232, 514].
        u, v = project_point(geometry_from_gt(PAYLOAD), [-81.0084, 96.8132, 16.4032])
        self.assertTrue(1182 - 20 <= u <= 1232 + 20 and 476 - 20 <= v <= 514 + 20, (u, v))

    def test_ground_point_inverts_projection(self):
        geom = geometry_from_gt(PAYLOAD)
        point = [-80.0, 99.0, 16.1]
        u, v = project_point(geom, point)
        hit = ground_point(geom, [u - 5, v - 10, u + 5, v])
        self.assertAlmostEqual(hit[0], point[0], places=3)
        self.assertAlmostEqual(hit[1], point[1], places=3)


class RelationTests(unittest.TestCase):
    def setUp(self):
        self.bench = _det("bench", "bench_0", [400, 300, 600, 400], [-80.0, 99.0, 16.3], "landmark")

    def test_distance_relations_are_predicates(self):
        mug = _det("mug", "mug_0", [700, 380, 710, 390], [-80.0, 100.5, 16.1])
        rels = relations_between(mug, self.bench, PAYLOAD)
        self.assertIn("beside", rels)
        self.assertIn("near", rels)
        far = _det("mug", "mug_1", [900, 380, 910, 390], [-80.0, 104.0, 16.1])
        self.assertEqual({"near"} & relations_between(far, self.bench, PAYLOAD), {"near"})
        self.assertNotIn("beside", relations_between(far, self.bench, PAYLOAD))

    def test_stacked_object_is_on(self):
        cup = _det("mug", "mug_0", [490, 290, 510, 310], [-80.0, 99.05, 16.75])
        self.assertIn("on", relations_between(cup, self.bench, PAYLOAD))

    def test_beside_kept_with_distractor_when_relation_separates(self):
        near = _det("mug", "mug_0", [700, 380, 710, 390], [-80.0, 100.5, 16.1])
        far = _det("mug", "mug_1", [900, 380, 910, 390], [-80.0, 104.0, 16.1])
        choices = object_choices(PAYLOAD, [near, far], [self.bench])
        beside = [c for c in choices["mug_0"] if c.relation == "beside"]
        self.assertEqual(len(beside), 1)
        self.assertTrue(beside[0].distractor_present)
        self.assertFalse(any(c.relation == "beside" for c in choices["mug_1"]))
        # "near" fits both mugs, so it describes neither.
        self.assertFalse(any(c.relation == "near" for c in choices["mug_0"] + choices["mug_1"]))


class SarCheckTests(unittest.TestCase):
    RECIPE = {"id": "sar01_trash_sit", "pose": "person_sitting", "support_class": "trash_can",
              "relation": "on"}

    def _scene(self, person_xyz, person_box):
        can = _det("trash_can", "trash_can_0", [600, 230, 700, 420], [-79.87, 98.08, 16.6], "landmark")
        person = _det("person_sitting", "p0", person_box, person_xyz)
        return person, can

    def test_seated_person_passes(self):
        person, can = self._scene([-79.85, 98.25, 17.0], [610, 120, 690, 300])
        check = sar_check(person, self.RECIPE, [person], [can])
        self.assertTrue(check.ok, check.reason)
        self.assertEqual(check.anchors, (can,))

    def test_rolled_away_support_fails(self):
        person, can = self._scene([-79.85, 99.2, 17.0], [610, 120, 690, 300])
        self.assertFalse(sar_check(person, self.RECIPE, [person], [can]).ok)

    def test_person_cut_at_top_fails(self):
        person, can = self._scene([-79.85, 98.25, 17.0], [610, 0, 690, 300])
        self.assertEqual(sar_check(person, self.RECIPE, [person], [can]).reason, "cut_at_top")


class GoldMatchingTests(unittest.TestCase):
    SPEC = {
        "case_id": "ho-s00001-mug_0", "kind": "object", "command": "Find the mug beside the bench.",
        "camera_id": "rig_3_b", "scene": 1, "gt_json": "x.json",
        "target": {"class": "mug", "instance_id": "mug_0", "bbox": [100, 100, 120, 120],
                   "world_xyz": [0, 0, 0]},
        "anchors": [{"class": "bench", "instance_id": "bench_0", "bbox": [200, 100, 300, 160],
                     "world_xyz": [1, 0, 0]}],
        "recipe_id": None, "relation": "beside", "landmark_class": "bench", "anchor_rule": "relation",
        "tags": {"stratum": "seen"},
    }

    def _detection(self, candidates):
        return {"parsed": {}, "candidates": candidates, "detector": {}, "decisions": []}

    def _cand(self, cid, cls, bbox, source="detector"):
        return {"candidate_id": cid, "camera_id": "rig_3_b", "class_name": cls,
                "confidence": 0.9, "bbox_xyxy": bbox, "source": source}

    def test_gold_ids_need_class_and_iou(self):
        case = assemble_case(self.SPEC, self._detection([
            self._cand("rig_3_b:mug:0", "mug", [101, 101, 121, 121]),
            self._cand("rig_3_b:cup:0", "disposable_cup", [100, 100, 120, 120]),
            self._cand("rig_3_b:bench:0", "bench", [200, 100, 300, 160]),
        ]))
        self.assertEqual(case["gold"]["target_ids"], ["rig_3_b:mug:0"])
        self.assertEqual(case["gold"]["anchor_ids"], ["rig_3_b:bench:0"])
        self.assertFalse(case["tags"]["detector_miss"])

    def test_missed_target_stays_tagged(self):
        case = assemble_case(self.SPEC, self._detection([
            self._cand("rig_3_b:mug:0", "mug", [300, 300, 320, 320]),
            self._cand("rig_3_b:mug:1", "mug", [100, 100, 120, 120], source="proxy"),
        ]))
        self.assertEqual(case["gold"]["target_ids"], ["rig_3_b:mug:1"])
        self.assertEqual(case["tags"]["proxy_count"], 1)
        missed = assemble_case(self.SPEC, self._detection([]))
        self.assertTrue(missed["tags"]["detector_miss"])
        self.assertTrue(missed["tags"]["anchor_miss"])

    def test_strata_thresholds(self):
        self.assertEqual([stratum(0), stratum(9), stratum(58)], ["unseen", "rare", "seen"])


class PublicProposalTests(unittest.TestCase):
    def test_nearest_per_camera_within_limit(self):
        geom = geometry_from_gt(PAYLOAD)
        expected = [-80.0, 99.0, 16.1]
        u, v = project_point(geom, expected)
        cands = [
            {"candidate_id": "rig_3_b:mug:0", "camera_id": "rig_3_b", "class_name": "mug",
             "bbox_xyxy": [u - 5, v - 10, u + 5, v]},
            {"candidate_id": "rig_3_b:mug:1", "camera_id": "rig_3_b", "class_name": "mug",
             "bbox_xyxy": [u - 500, v - 10, u - 490, v]},
        ]
        gold, evidence = propose(cands, {"rig_3_b": geom}, expected, lambda c: c["class_name"] == "mug")
        self.assertEqual(gold, ["rig_3_b:mug:0"])
        self.assertGreater(evidence["nearest"][1]["error_m"], 1.0)


class CaseStoreTests(unittest.TestCase):
    def _write(self, root: Path, set_name: str, case_id: str, confirmed=True):
        frame = root / f"{case_id}.png"
        Image.new("RGB", (4, 4)).save(frame)
        case_store.write_case(root, set_name, {
            "case_id": case_id, "kind": "object", "command": "Find the mug.", "parsed": {},
            "candidates": [], "tags": {"stratum": "seen"},
            "gold": {"target_ids": [], "camera_ids": ["rig_1_a"],
                     "confirmed_by": {"reviewer": "r"} if confirmed else None},
        }, {"rig_1_a": frame})

    def test_freeze_verify_and_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "heldout", "ho-1")
            self._write(root, "public", "pub-1")
            case_store.freeze(root)
            cases = case_store.load_set(root, "heldout")
            self.assertEqual([c.case_id for c in cases], ["ho-1"])
            self.assertEqual(cases[0].frame("rig_1_a").shape, (4, 4, 3))
            with self.assertRaises(FileExistsError):
                case_store.freeze(root)
            case_json = root / "heldout" / "cases" / "ho-1" / "case.json"
            raw = json.loads(case_json.read_text())
            raw["gold"]["target_ids"] = ["relabelled"]
            case_json.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                case_store.load_set(root, "heldout")
            self._write(root, "public", "pub-2")
            with self.assertRaises(ValueError):
                case_store.load_set(root, "public")

    def test_unconfirmed_public_gold_blocks_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "public", "pub-1", confirmed=False)
            with self.assertRaises(ValueError):
                case_store.freeze(root)


if __name__ == "__main__":
    unittest.main()
