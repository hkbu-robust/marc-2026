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
    def test_cycles_one_climber_and_keeps_scene_density(self):
        patch = _load_patch()
        source = (
            "def build(landmarks, all_chars, all_poses):\n"
            "    catalog = {'persons': [], 'poses': []}\n"
            "    selected_keys = set()\n"
            "    people_state = {'enabled': True}\n"
            "    objects = [{'type': 'mug', 'count': 6}]\n"
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
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = (
                "vending_machine,trash_can,hydrant,postbox"
            )
            targets = []
            for _ in range(6):
                landmarks, specs = namespace["build"](
                    [{"name": "bench_0"}],
                    ["female_student_01", "male_student_01", "professor_01"],
                    ["person_reaching", "person_crouching", "person_sitting"],
                )
                lost = [o for o in specs if o.get("type") == "mug"]
                people = [o for o in specs if o.get("character")]
                self.assertEqual(lost, [{"type": "mug", "count": 6}])
                self.assertEqual(len(people), 3)
                self.assertIn("bench_0", {s["name"] for s in landmarks})
                climbers = [
                    p for p in people
                    if (p.get("placement") or {}).get("above")
                ]
                self.assertEqual(len(climbers), 1)
                placement = climbers[0]["placement"]
                targets.append(placement["on_landmark_type"])
                self.assertIn("%s_0" % placement["on_landmark_type"],
                              {s["name"] for s in landmarks})
                self.assertEqual(
                    landmarks[0]["name"],
                    "%s_0" % placement["on_landmark_type"],
                )
                self.assertTrue(placement["above"])
                self.assertEqual(placement["contact_clearance"], 0.03)
                if placement["on_landmark_type"] in (
                        "vending_machine", "postbox"):
                    self.assertIn(
                        climbers[0]["type"],
                        ("person_sitting", "person_crouching"),
                    )
                extras = [
                    p for p in people
                    if not (p.get("placement") or {}).get("above")
                ]
                for extra in extras:
                    self.assertFalse(extra.get("placement"))
        finally:
            os.environ.clear()
            os.environ.update(old)

        self.assertEqual(
            targets,
            ["vending_machine", "vending_machine", "vending_machine",
             "trash_can", "hydrant", "postbox"],
        )
        self.assertEqual(targets.count("vending_machine"), 3)

    def test_equal_weights_restore_even_cycle(self):
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
        exec(compile(source, "<above-weights>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = (
                "vending_machine,trash_can,hydrant,postbox"
            )
            os.environ["TRAINER_PEOPLE_ABOVE_WEIGHTS"] = (
                "vending_machine:1,trash_can:1,hydrant:1,postbox:1"
            )
            targets = []
            for _ in range(4):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}],
                    ["female_student_01"],
                    ["person_sitting"],
                )
                climbers = [
                    p for p in specs
                    if (p.get("placement") or {}).get("above")
                ]
                targets.append(climbers[0]["placement"]["on_landmark_type"])
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(
            targets,
            ["vending_machine", "trash_can", "hydrant", "postbox"],
        )


if __name__ == "__main__":
    unittest.main()
