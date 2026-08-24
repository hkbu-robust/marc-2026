#!/usr/bin/env python3
"""Checks for the dataset-gen on-landmark / above-mode trainer patch."""

from __future__ import annotations

import ast
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "simulation-platform" / "patches" / "patch_trainer_on_landmark.py"
SNAPSHOT = Path("/tmp/container_trainer.py")


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "patch_trainer_on_landmark", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OnLandmarkPatchTests(unittest.TestCase):
    def test_helper_and_positions_compile(self):
        patch = _load_patch()
        ast.parse(patch.ABOVE_HELPER)
        ast.parse("def outer():\n    def _deploy_into():\n" + patch.NEW_POSITIONS)
        ast.parse("def outer():\n    def _settle():\n" + patch.NEW_SNAP)
        ast.parse("def outer():\n    def _capture():\n" + patch.NEW_SAVE)
        self.assertIn("_marc_above_placed_n", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_surface_contact", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_construct", patch.ABOVE_HELPER)
        self.assertIn("def _marc_nudge_above_nearer", patch.ABOVE_HELPER)
        self.assertIn("for cx, cy, top_z in candidates", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_near_xy", patch.NEW_POSITIONS)
        self.assertIn("_marc_nudge_above_nearer", patch.NEW_SAVE)
        self.assertIn("drop_to_ground", patch.ABOVE_HELPER)
        self.assertIn("raycast_closest", patch.ABOVE_HELPER)
        self.assertIn("obj_pos[_climber_i] = list(lm_pos[_lm_i])",
                      patch.NEW_POSITIONS)
        self.assertIn("_n_placed > 0", patch.NEW_SAVE)
        self.assertIn("_n_stack > 0", patch.NEW_SAVE)
        self.assertIn("_top_m", patch.NEW_SAVE)
        self.assertIn("target={_target", patch.NEW_SAVE)
        self.assertIn("above-mode: target=", patch.ABOVE_HELPER)
        self.assertIn("target={_tgt}", patch.NEW_GT_LOG)
        self.assertIn("target={_tgt}", patch.NEW_AUTO_FAILURE)
        self.assertIn("_marc_above_target", patch.ABOVE_HELPER)
        self.assertIn("vending_machine", patch.NEW_POSITIONS)
        self.assertIn('placement.get("above")', patch.NEW_REPOSITION_PL)
        self.assertGreaterEqual(patch.NEW_SNAP.count("_marc_run_above_place"), 2)

    def test_contact_ok_requires_on_top_and_touching(self):
        patch = _load_patch()
        lo, hi = (-0.4, -0.4, 16.0), (0.4, 0.4, 18.0)
        top_z = 18.0
        self.assertTrue(
            patch.above_contact_ok((0.0, 0.0, 18.02), lo, hi, top_z))
        self.assertFalse(
            patch.above_contact_ok((0.0, 0.0, 16.05), lo, hi, top_z),
            "standing beside at ground height must not count as on top")
        self.assertFalse(
            patch.above_contact_ok((0.0, 0.0, 18.30), lo, hi, top_z),
            "30cm hover above the top is not contact")
        self.assertFalse(
            patch.above_contact_ok((2.0, 0.0, 18.02), lo, hi, top_z),
            "off the footprint is not on the landmark")

    def test_visible_stack_rejects_crop_and_beside(self):
        patch = _load_patch()
        on_vending = {
            "class": "person_sitting",
            "bbox": [600, 200, 700, 420],
        }
        vending = {
            "class": "vending_machine",
            "bbox": [580, 280, 720, 500],
        }
        self.assertTrue(patch.visible_above_stack(on_vending, vending, 720))
        clipped = {
            "class": "vending_machine",
            "bbox": [580, 0, 720, 400],
        }
        clipped_person = {
            "class": "person_sitting",
            "bbox": [600, 0, 700, 180],
        }
        self.assertFalse(
            patch.visible_above_stack(clipped_person, clipped, 720),
            "top-clipped vending roof must not count as stacked")
        beside = {
            "class": "person_walking",
            "bbox": [590, 240, 710, 500],
        }
        self.assertFalse(patch.visible_above_stack(beside, vending, 720))
        hydrant_side = {
            "class": "person_reaching",
            "bbox": [120, 200, 220, 520],
        }
        hydrant = {
            "class": "hydrant",
            "bbox": [400, 360, 460, 520],
        }
        self.assertFalse(patch.visible_above_stack(hydrant_side, hydrant, 720))

    def test_colocates_climber_with_target_cell(self):
        patch = _load_patch()
        source = (
            "def generate_cell_positions(area, n):\n"
            "    return [[float(i), 0.0, 0.0] for i in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = ["
            "{'name': 'vending_machine_0'}, {'name': 'bench_0'}]\n"
            "        objects = [\n"
            "            {'type': 'mug', 'count': 1},\n"
            "            {'type': 'person_reaching', 'count': 1,\n"
            "             'placement': {'above': True,\n"
            "                           'on_landmark_type': 'vending_machine'}},\n"
            "            {'type': 'person_walking', 'count': 1},\n"
            "        ]\n"
            "        n_lm = len(landmarks)\n"
            "        n_obj = sum(int(o.get('count', 1) or 1) for o in objects)\n"
            "        area = None\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos, obj_pos\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os}
        exec(compile(source, "<above-positions>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            lm_pos, obj_pos = namespace["outer"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(obj_pos[1], lm_pos[0])
        self.assertNotEqual(obj_pos[0], lm_pos[0])
        self.assertNotEqual(obj_pos[2], lm_pos[0])

    def test_headroom_puts_vending_mid_near_not_on_far_edge(self):
        patch = _load_patch()
        source = (
            "def generate_cell_positions(area, n):\n"
            "    return [[100.0, 100.0, 16.0] for _ in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = [{'name': 'vending_machine_0'}]\n"
            "        objects = [\n"
            "            {'type': 'person_sitting', 'count': 1,\n"
            "             'placement': {'above': True,\n"
            "                           'on_landmark_type': 'vending_machine'}},\n"
            "        ]\n"
            "        n_lm = 1\n"
            "        n_obj = 1\n"
            "        area = {\n"
            "            'top_left': [0.0, 20.0, 16.0],\n"
            "            'top_right': [10.0, 20.0, 16.0],\n"
            "            'bottom_right': [10.0, 0.0, 16.0],\n"
            "            'bottom_left': [0.0, 0.0, 16.0],\n"
            "        }\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos, obj_pos\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os, "random": __import__("random")}
        exec(compile(source, "<above-headroom>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            namespace["random"].seed(0)
            lm_pos, obj_pos = namespace["outer"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(lm_pos[0], obj_pos[0])
        # Far edge is y=20, near is y=0. Tall targets stay in the near-mid band.
        self.assertGreater(lm_pos[0][1], 2.0)
        self.assertLess(lm_pos[0][1], 10.0)
        self.assertGreater(lm_pos[0][0], 3.0)
        self.assertLess(lm_pos[0][0], 7.0)

    def test_replaces_stale_above_helper(self):
        patch = _load_patch()
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as fh:
            fh.write(
                "def other():\n    return 1\n\n"
                "def _marc_place_above_people(mission_id, objects, configurations=None):\n"
                "    globals()['_marc_above_placed_n'] = 0\n"
                "    return None\n"
            )
            path = fh.name
        try:
            self.assertTrue(patch._ensure_above_helper(path))
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("_marc_above_surface_contact", text)
            self.assertIn("_marc_above_construct", text)
            self.assertIn("def _marc_nudge_above_nearer", text)
            self.assertIn("raycast_closest", text)
            self.assertEqual(text.count("def _marc_place_above_people"), 1)
            self.assertFalse(patch._ensure_above_helper(path))
        finally:
            os.unlink(path)

    def test_snapshot_anchors_still_match(self):
        if not SNAPSHOT.is_file():
            self.skipTest("stock trainer snapshot not at %s" % SNAPSHOT)
        patch = _load_patch()
        src = SNAPSHOT.read_text(encoding="utf-8")
        for name in (
                "OLD_POSITIONS",
                "OLD_DEPLOY_LANDMARKS",
                "OLD_SAVE",
                "OLD_GT_LOG",
                "OLD_SNAP",
                "OLD_CAPTURE_RETURN",
                "OLD_INITIAL_WARN",
                "OLD_AUTO_REMAINING",
                "OLD_AUTO_ATTEMPT",
                "OLD_AUTO_FAILURE",
        ):
            self.assertIn(getattr(patch, name), src, name)
        patched = src.replace(patch.OLD_POSITIONS, patch.NEW_POSITIONS, 1)
        patched = patched.replace(patch.OLD_DEPLOY_LANDMARKS,
                                  patch.NEW_DEPLOY_LANDMARKS, 1)
        patched = patched.replace(patch.OLD_SAVE, patch.NEW_SAVE, 1)
        patched = patched.replace(patch.OLD_GT_LOG, patch.NEW_GT_LOG, 1)
        ast.parse(patched)


if __name__ == "__main__":
    unittest.main()
