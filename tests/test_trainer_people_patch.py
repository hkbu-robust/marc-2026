#!/usr/bin/env python3
"""Checks for the dataset-gen deterministic above-mode source patch."""

from __future__ import annotations

import importlib.util
import os
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "simulation-platform" / "patches" / "patch_trainer_people.py"


def _load_patch():
    spec = importlib.util.spec_from_file_location("patch_trainer_people", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AboveModePatchTests(unittest.TestCase):
    def test_one_person_per_eligible_landmark_type(self):
        patch = _load_patch()
        source = (
            "def build(landmarks, all_chars, all_poses):\n"
            "    catalog = {'persons': [], 'poses': []}\n"
            "    selected_keys = set()\n"
            "    people_state = {'enabled': True}\n"
            "    objects = []\n"
            "    char_by_key = {}\n"
            "    pose_by_key = {}\n"
            + patch.NEW_COMBOS
            + "    return landmarks, objects\n"
        )
        namespace = {"os": os, "random": random}
        exec(compile(source, "<above-mode-patch>", "exec"), namespace)

        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            landmarks = [
                {"name": "bench_0"},
                {"name": "hydrant_0"},
                {"name": "postbox_0"},
                {"name": "vending_machine_0"},
            ]
            _, people = namespace["build"](
                landmarks,
                ["female_student_01", "male_student_01", "professor_01"],
                ["person_reaching", "person_crouching", "person_sitting"],
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

        self.assertEqual(len(people), 3)
        targets = {p["placement"]["on_landmark_type"] for p in people}
        self.assertEqual(targets, {"hydrant", "postbox", "vending_machine"})
        self.assertTrue(all(p["placement"]["above"] for p in people))
        self.assertTrue(
            all(p["placement"]["z_offset_from_origin"] == 2.55 for p in people)
        )


if __name__ == "__main__":
    unittest.main()
