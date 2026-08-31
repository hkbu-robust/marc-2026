#!/usr/bin/env python3
"""Checks for the dataset-gen deterministic above-mode source patch."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import tempfile
import unittest
from pathlib import Path

import yaml


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
        self.assertIn("person_walking", patch.NEW_COMBOS)
        self.assertIn("person_reaching", patch.NEW_COMBOS)
        self.assertIn("_marc_above_climber_pose", patch.NEW_COMBOS)
        self.assertIn("TRAINER_PEOPLE_CHARS", patch.NEW_COMBOS)
        self.assertIn("TRAINER_PEOPLE_POSES", patch.NEW_COMBOS)
        self.assertIn("_marc_filter_names", patch.NEW_COMBOS)
        self.assertIn("_marc_sit_sink", patch.NEW_COMBOS)
        self.assertIn("TRAINER_PEOPLE_SIT_SINK_SHORT", patch.NEW_COMBOS)
        self.assertIn("TRAINER_PEOPLE_SIT_SINK_TRASH", patch.NEW_COMBOS)
        self.assertIn("TRAINER_PEOPLE_VEND_BUTT_SINK", patch.NEW_COMBOS)
        self.assertIn("TRAINER_PEOPLE_CROUCH_SINK_HYDRANT", patch.NEW_COMBOS)

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
                    ["person_reaching", "person_crouching", "person_sitting",
                     "person_walking", "person_reclining", "person_lying_down"],
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
                if climbers[0]["type"] == "person_sitting":
                    if placement["on_landmark_type"] == "vending_machine":
                        self.assertAlmostEqual(
                            placement["z_offset_from_top"], 0.03 - 0.12)
                    elif placement["on_landmark_type"] == "trash_can":
                        self.assertAlmostEqual(
                            placement["z_offset_from_top"], 0.03 - 0.42)
                    else:
                        extra = 0.0
                        if placement["on_landmark_type"] in (
                                "postbox", "hydrant"):
                            extra = 0.35
                        self.assertAlmostEqual(
                            placement["z_offset_from_top"],
                            0.03 - 0.70 - extra)
                elif (climbers[0]["type"] == "person_crouching"
                      and placement["on_landmark_type"] == "hydrant"):
                    self.assertAlmostEqual(
                        placement["z_offset_from_top"], 0.03 - 0.18)
                else:
                    self.assertEqual(placement["z_offset_from_top"], 0.03)
                self.assertIn(
                    climbers[0]["type"],
                    ("person_crouching", "person_sitting", "person_walking",
                     "person_reaching"),
                )
                extras = [
                    p for p in people
                    if not (p.get("placement") or {}).get("above")
                ]
                for extra in extras:
                    pl = extra.get("placement") or {}
                    self.assertFalse(pl.get("above"))
                    self.assertIsNone(pl.get("on_landmark_type"))
        finally:
            os.environ.clear()
            os.environ.update(old)

        self.assertEqual(
            targets,
            ["vending_machine", "trash_can", "hydrant", "postbox",
             "vending_machine", "trash_can"],
        )
        self.assertEqual(targets.count("vending_machine"), 2)

    def test_climber_identity_cycles_catalog_chars(self):
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
        exec(compile(source, "<above-char-cycle>", "exec"), namespace)
        self.assertIn("random.shuffle(combos)", patch.NEW_COMBOS)
        self.assertIn(
            "(_above_mode or _recipe is not None)", patch.NEW_COMBOS)
        chars = [
            "male_student_01", "female_student_01",
            "professor_01", "student_weight_01",
        ]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = "vending_machine"
            os.environ["TRAINER_PEOPLE_ABOVE_WEIGHTS"] = "vending_machine:1"
            seen = []
            extras = []
            namespace["random"].seed(1)
            for _ in range(8):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}],
                    list(chars),
                    ["person_sitting", "person_crouching"],
                )
                climbers = [
                    p for p in specs
                    if (p.get("placement") or {}).get("above")
                ]
                self.assertEqual(len(climbers), 1)
                seen.append(climbers[0]["character"])
                extra_chars = [
                    p["character"] for p in specs
                    if p.get("character")
                    and not (p.get("placement") or {}).get("above")
                ]
                extras.append(sorted(extra_chars))
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertTrue(all(c in chars for c in seen))
        self.assertNotEqual(seen, chars + chars)
        self.assertGreater(len(set(seen)), 1)
        for i, extra_chars in enumerate(extras):
            self.assertEqual(len(extra_chars), 3)
            self.assertEqual(len(set(extra_chars)), 3)
            self.assertNotIn(seen[i], extra_chars)

    def test_drops_extra_climb_types_and_keeps_benches(self):
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
        exec(compile(source, "<above-collapse>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = "vending_machine"
            os.environ["TRAINER_PEOPLE_ABOVE_WEIGHTS"] = "vending_machine:1"
            landmarks, specs = namespace["build"](
                [
                    {"name": "bench_0"},
                    {"name": "hydrant_0"},
                    {"name": "trash_can_0"},
                    {"name": "vending_machine_1"},
                ],
                ["female_student_01"],
                ["person_sitting"],
            )
        finally:
            os.environ.clear()
            os.environ.update(old)
        names = {s["name"] for s in landmarks}
        self.assertEqual(names, {"vending_machine_1", "bench_0"})
        self.assertEqual(landmarks[0]["name"], "vending_machine_1")
        climbers = [
            p for p in specs if (p.get("placement") or {}).get("above")
        ]
        self.assertEqual(len(climbers), 1)
        self.assertEqual(
            climbers[0]["placement"]["on_landmark_type"], "vending_machine")

    def test_recipe_above_drops_extra_climb_types(self):
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
        exec(compile(source, "<recipe-collapse>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "10"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            landmarks, _specs = namespace["build"](
                [
                    {"name": "bench_0"},
                    {"name": "hydrant_0"},
                    {"name": "trash_can_0"},
                    {"name": "vending_machine_1"},
                ],
                ["female_student_01"],
                ["person_sitting", "person_walking"],
            )
        finally:
            os.environ.clear()
            os.environ.update(old)
        names = {s["name"] for s in landmarks}
        self.assertIn("vending_machine_1", names)
        self.assertIn("bench_0", names)
        self.assertNotIn("hydrant_0", names)
        self.assertNotIn("trash_can_0", names)
        climbers = [
            p for p in _specs if (p.get("placement") or {}).get("above")
        ]
        self.assertEqual(len(climbers), 1)
        self.assertEqual(
            climbers[0]["placement"]["on_landmark_type"], "vending_machine")

    def test_injects_climber_when_catalog_people_are_empty(self):
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
        exec(compile(source, "<above-fallback-climber>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ALL_CHARS"] = "0"
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = "vending_machine"
            os.environ["TRAINER_PEOPLE_ABOVE_WEIGHTS"] = "vending_machine:1"
            landmarks, specs = namespace["build"](
                [{"name": "bench_0"}],
                ["female_student_01"],
                ["person_reaching"],
            )
        finally:
            os.environ.clear()
            os.environ.update(old)
        people = [o for o in specs if o.get("character")]
        self.assertEqual(len(people), 1)
        self.assertTrue((people[0].get("placement") or {}).get("above"))
        self.assertEqual(
            people[0]["placement"]["on_landmark_type"], "vending_machine")
        self.assertIn("vending_machine_0", {s["name"] for s in landmarks})
        self.assertEqual(people[0]["type"], "person_reaching")

    def test_vending_climber_biases_sit_and_reach(self):
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
        exec(compile(source, "<above-vend-pose-bias>", "exec"), namespace)
        poses = [
            "person_reaching", "person_crouching", "person_sitting",
            "person_walking",
        ]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = "vending_machine"
            os.environ["TRAINER_PEOPLE_ABOVE_WEIGHTS"] = "vending_machine:1"
            namespace["random"].seed(7)
            vend = []
            for _ in range(80):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}],
                    ["female_student_01"],
                    list(poses),
                )
                climbers = [
                    p for p in specs
                    if (p.get("placement") or {}).get("above")
                ]
                vend.append(climbers[0]["type"])
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = "hydrant"
            os.environ["TRAINER_PEOPLE_ABOVE_WEIGHTS"] = "hydrant:1"
            namespace["random"].seed(7)
            hydrant = []
            for _ in range(40):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}],
                    ["female_student_01"],
                    list(poses),
                )
                climbers = [
                    p for p in specs
                    if (p.get("placement") or {}).get("above")
                ]
                hydrant.append(climbers[0]["type"])
        finally:
            os.environ.clear()
            os.environ.update(old)
        sit_or_reach = sum(
            1 for t in vend
            if t in ("person_sitting", "person_reaching"))
        self.assertGreaterEqual(sit_or_reach, 50)
        self.assertTrue(any(t == "person_reaching" for t in vend))
        self.assertTrue(any(t == "person_sitting" for t in vend))
        self.assertEqual(
            set(hydrant),
            {"person_crouching", "person_sitting", "person_walking"},
        )

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

    def test_chars_and_poses_flags_pin_climber(self):
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
        exec(compile(source, "<above-char-pose-flags>", "exec"), namespace)
        chars = [
            "male_student_01", "female_student_01",
            "professor_01", "student_weight_01",
        ]
        poses = [
            "person_reaching", "person_crouching", "person_sitting",
            "person_walking",
        ]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE_TYPES"] = "vending_machine"
            os.environ["TRAINER_PEOPLE_ABOVE_WEIGHTS"] = "vending_machine:1"
            os.environ["TRAINER_PEOPLE_CHARS"] = "student_weight_01"
            os.environ["TRAINER_PEOPLE_POSES"] = "person_sitting"
            _, specs = namespace["build"](
                [{"name": "bench_0"}], list(chars), list(poses),
            )
            people = [o for o in specs if o.get("character")]
            self.assertEqual(len(people), 1)
            self.assertEqual(people[0]["character"], "student_weight_01")
            self.assertEqual(people[0]["type"], "person_sitting")
            self.assertTrue((people[0].get("placement") or {}).get("above"))

            os.environ["TRAINER_PEOPLE_CHARS"] = "weight"
            os.environ["TRAINER_PEOPLE_POSES"] = "sitting"
            _, specs = namespace["build"](
                [{"name": "bench_0"}], list(chars), list(poses),
            )
            people = [o for o in specs if o.get("character")]
            self.assertEqual([p["character"] for p in people],
                             ["student_weight_01"])
            self.assertEqual(people[0]["type"], "person_sitting")

            os.environ["TRAINER_PEOPLE_CHARS"] = "male_student_01"
            os.environ["TRAINER_PEOPLE_POSES"] = "person_sitting,person_reaching"
            seen_chars = set()
            seen_poses = set()
            namespace["random"].seed(3)
            for _ in range(20):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
                people = [o for o in specs if o.get("character")]
                seen_chars.update(p["character"] for p in people)
                seen_poses.update(p["type"] for p in people)
            self.assertEqual(seen_chars, {"male_student_01"})
            self.assertTrue(
                seen_poses <= {"person_sitting", "person_reaching"})
            self.assertNotIn("female_student_01", seen_chars)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_seat_pool_includes_person_prone(self):
        patch = _load_patch()
        self.assertIn("person_prone", patch.NEW_COMBOS)
        self.assertIn("TRAINER_SAR_RECIPES", patch.NEW_COMBOS)
        self.assertIn("_marc_yolo_mode", patch.NEW_COMBOS)
        self.assertIn("TRAINER_YOLO_MODE", patch.NEW_COMBOS)
        self.assertIn("_marc_sar_recipe_cursor_start", patch.NEW_COMBOS)
        self.assertIn("TRAINER_SAR_RECIPE_CURSOR", patch.NEW_COMBOS)
        self.assertIn("_ON_SEAT_POSES", patch.NEW_COMBOS)

    def test_sar_recipes_pin_one_target_per_scene(self):
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
        exec(compile(source, "<sar-recipe-pin>", "exec"), namespace)
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_prone", "person_reaching",
        ]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "0"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            recipe_ids = []
            for _ in range(6):
                landmarks, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
                people = [o for o in specs if o.get("character")]
                self.assertEqual(len(people), 3)
                targets = []
                for person in people:
                    pl = person.get("placement") or {}
                    if (pl.get("on_landmark_type")
                            or pl.get("beside_landmark_type")
                            or pl.get("support_class")
                            or pl.get("surface_class")):
                        targets.append(person)
                self.assertEqual(len(targets), 1, people)
                recipe_ids.append((targets[0].get("placement") or {}).get(
                    "recipe_id"))
                names = {s["name"] for s in landmarks}
                pl = targets[0].get("placement") or {}
                if pl.get("beside_landmark_type") == "bicycle":
                    self.assertIn("bicycle_0", names)
                if pl.get("on_landmark_type"):
                    self.assertIn("%s_0" % pl["on_landmark_type"], names)
            self.assertTrue(any(recipe_ids))
            self.assertGreaterEqual(len(set(recipe_ids)), 2)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_beside_recipes_set_beside_landmark_type(self):
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
        exec(compile(source, "<sar-beside>", "exec"), namespace)
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_prone", "person_reaching", "person_bending",
        ]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "0"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            beside = set()
            for _ in range(14):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
                people = [o for o in specs if o.get("character")]
                for person in people:
                    typ = (person.get("placement") or {}).get(
                        "beside_landmark_type")
                    if typ:
                        beside.add(typ)
            self.assertIn("taxi", beside)
            self.assertIn("trash_can", beside)
            self.assertIn("bicycle", beside)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_recipe_dump_has_character_and_post_fallback_pose(self):
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
        exec(compile(source, "<sar-dump>", "exec"), namespace)
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_reaching",
        ]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        old = dict(os.environ)
        dump_path = "/tmp/marc_sar_recipe.json"
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "0"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            dumped = None
            target = None
            for _ in range(8):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
                people = [o for o in specs if o.get("character")]
                targets = [
                    o for o in people
                    if (o.get("placement") or {}).get("recipe_id")
                ]
                self.assertEqual(len(targets), 1, people)
                target = targets[0]
                with open(dump_path, encoding="utf-8") as f:
                    dumped = json.load(f)
                if str(dumped.get("recipe_id") or "").startswith("sar07"):
                    break
            self.assertIsNotNone(dumped)
            self.assertEqual(dumped["target_character"], target["character"])
            self.assertEqual(dumped["target_pose"], target["type"])
            self.assertEqual(dumped["target_pose"], "person_lying_down")
            self.assertNotEqual(dumped["target_pose"], "person_prone")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_climb_recipe_dump_sets_above(self):
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
        exec(compile(source, "<sar-above-dump>", "exec"), namespace)
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_prone", "person_reaching",
        ]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        old = dict(os.environ)
        dump_path = "/tmp/marc_sar_recipe.json"
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "10"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            namespace["build"](
                [{"name": "bench_0"}], list(chars), list(poses),
            )
            with open(dump_path, encoding="utf-8") as f:
                dumped = json.load(f)
            self.assertEqual(dumped.get("recipe_id"), "sarH3_climbed_machine")
            self.assertTrue(dumped.get("above"))
        finally:
            os.environ.clear()
            os.environ.update(old)
        namespace = {"os": os, "random": random}
        exec(compile(source, "<sar-above-dump-lie>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "2"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            namespace["build"](
                [{"name": "bench_0"}], list(chars), list(poses),
            )
            with open(dump_path, encoding="utf-8") as f:
                dumped = json.load(f)
            self.assertEqual(dumped.get("recipe_id"), "sar03_bench_lie")
            self.assertFalse(bool(dumped.get("above")))
            self.assertEqual(dumped.get("extra_landmarks"), ["taxi"])
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_yaml_miss_does_not_use_three_recipe_fallback(self):
        patch = _load_patch()
        self.assertIn("recipes disabled", patch.NEW_COMBOS)
        self.assertIn("return []", patch.NEW_COMBOS)
        compact = patch.NEW_COMBOS.replace(" ", "")
        self.assertNotIn('{"id":"sar01_trash_sit"', compact)
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
        exec(compile(source, "<sar-yaml-miss>", "exec"), namespace)
        poses = ["person_sitting", "person_walking", "person_lying_down"]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        old = dict(os.environ)
        cwd = os.getcwd()
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "0"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
            people = [o for o in specs if o.get("character")]
            recipe_ids = [
                (o.get("placement") or {}).get("recipe_id") for o in people
            ]
            self.assertTrue(people)
            self.assertTrue(all(rid is None for rid in recipe_ids), recipe_ids)
        finally:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(old)

    def test_recipe_cursor_env_continues_cycle(self):
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
        exec(compile(source, "<sar-cursor>", "exec"), namespace)
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_prone", "person_reaching", "person_bending",
        ]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "3513"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            _, specs = namespace["build"](
                [{"name": "bench_0"}], list(chars), list(poses),
            )
            people = [o for o in specs if o.get("character")]
            targets = [
                o for o in people
                if (o.get("placement") or {}).get("recipe_id")
            ]
            self.assertEqual(len(targets), 1, people)
            self.assertEqual(
                (targets[0].get("placement") or {}).get("recipe_id"),
                "sarH6_most_critical",
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_recipe_ids_filter_cycle(self):
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
        exec(compile(source, "<sar-filter>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "1"
            os.environ["TRAINER_SAR_RECIPE_IDS"] = (
                "sar01_trash_sit,sarH4_parking_trashsit")
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            _, specs = namespace["build"](
                [{"name": "bench_0"}], ["male_student_01"],
                ["person_sitting", "person_walking"],
            )
            target = next(
                o for o in specs if (o.get("placement") or {}).get("recipe_id"))
            self.assertEqual(
                target["placement"]["recipe_id"], "sarH4_parking_trashsit")
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_recipe_cursor_follows_existing_scene_dirs(self):
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
        exec(compile(source, "<sar-cursor-scan>", "exec"), namespace)
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_prone", "person_reaching", "person_bending",
        ]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ.pop("TRAINER_SAR_RECIPE_CURSOR", None)
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            os.environ["ENV_MARC_SCENARIO"] = "marc2026_chungmu"
            with tempfile.TemporaryDirectory() as tmp:
                scene = os.path.join(tmp, "marc2026_chungmu", "scene_03512")
                os.makedirs(scene)
                os.environ["MARC_TRAINER_DIR"] = tmp
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
            people = [o for o in specs if o.get("character")]
            targets = [
                o for o in people
                if (o.get("placement") or {}).get("recipe_id")
            ]
            self.assertEqual(len(targets), 1, people)
            self.assertEqual(
                (targets[0].get("placement") or {}).get("recipe_id"),
                "sarH6_most_critical",
            )
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_recipe_climber_identity_shuffles(self):
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
        exec(compile(source, "<sar-char-shuffle>", "exec"), namespace)
        chars = [
            "male_student_01", "female_student_01",
            "professor_01", "student_weight_01",
        ]
        poses = ["person_sitting", "person_walking", "person_lying_down"]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "0"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            seen = []
            namespace["random"].seed(1)
            for _ in range(8):
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
                targets = [
                    p for p in specs
                    if (p.get("placement") or {}).get("recipe_id")
                ]
                self.assertEqual(len(targets), 1, specs)
                seen.append(targets[0]["character"])
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertTrue(all(c in chars for c in seen))
        self.assertNotEqual(seen, [chars[0]] * len(seen))
        self.assertGreater(len(set(seen)), 1)

    def test_sar_recipes_yaml_relations_are_strings(self):
        path = ROOT / "simulation-platform" / "patches" / "sar_recipes.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        allowed = {
            "near", "beside", "behind", "front", "left", "right", "on", "above",
        }
        recipes = list(payload.get("recipes") or [])
        self.assertGreaterEqual(len(recipes), 14)
        for rec in recipes:
            rel = rec.get("relation")
            self.assertIsInstance(
                rel, str, "%s relation=%r" % (rec.get("id"), rel))
            self.assertIn(rel, allowed, rec.get("id"))
            self.assertIsNot(rel, True)

    def _exec_combos(self, name):
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
        exec(compile(source, name, "exec"), namespace)
        self.assertIn("def _marc_recipe_rel(", patch.NEW_COMBOS)
        return namespace

    def test_seat_lie_recipes_set_on_landmark_type(self):
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_prone", "person_reaching",
        ]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        cases = (
            (2, "sar03_bench_lie", "bench"),
            (3, "sar04_table_lie", "picnic_table"),
            (12, "sarH5_taxi_benchlie", "bench"),
        )
        old = dict(os.environ)
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            for cursor, recipe_id, landmark in cases:
                namespace = self._exec_combos(
                    "<sar-seat-lie-%s>" % recipe_id)
                os.environ["TRAINER_SAR_RECIPE_CURSOR"] = str(cursor)
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
                people = [o for o in specs if o.get("character")]
                targets = [
                    o for o in people
                    if (o.get("placement") or {}).get("recipe_id")
                ]
                self.assertEqual(len(targets), 1, people)
                pl = targets[0].get("placement") or {}
                self.assertEqual(pl.get("recipe_id"), recipe_id)
                self.assertEqual(pl.get("on_landmark_type"), landmark, recipe_id)
                self.assertFalse(pl.get("above"))
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_yolo_mode_seats_only_bench_or_table(self):
        namespace = self._exec_combos("<yolo-mode-seat>")
        poses = [
            "person_sitting", "person_walking", "person_lying_down",
            "person_prone", "person_reaching", "person_crouching",
            "person_reclining", "person_walking_away", "person_bending",
        ]
        chars = ["female_student_01", "male_student_01"]
        old = dict(os.environ)
        try:
            os.environ["TRAINER_YOLO_MODE"] = "1"
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "1"
            os.environ["TRAINER_PEOPLE_WRONG_PLACE_PROB"] = "1"
            os.environ["TRAINER_PEOPLE_ALL_CHARS"] = "1"
            for seed in range(24):
                namespace["random"].seed(seed)
                landmarks, specs = namespace["build"](
                    [
                        {"name": "bench_0"},
                        {"name": "vending_machine_0"},
                        {"name": "trash_can_0"},
                        {"name": "taxi_0"},
                    ],
                    chars,
                    poses,
                )
                people = [o for o in specs if o.get("character")]
                self.assertEqual(len(people), 2)
                for person in people:
                    pl = person.get("placement") or {}
                    self.assertFalse(pl.get("above"))
                    self.assertNotIn("recipe_id", pl)
                    lm = pl.get("on_landmark_type")
                    if lm is not None:
                        self.assertIn(lm, ("bench", "picnic_table"))
                        self.assertIn(
                            person["type"],
                            ("person_sitting", "person_reclining",
                             "person_lying_down", "person_prone"),
                        )
                self.assertIn("vending_machine_0", {s["name"] for s in landmarks})
                self.assertIn("trash_can_0", {s["name"] for s in landmarks})
        finally:
            os.environ.clear()
            os.environ.update(old)

    @unittest.skipIf(
        os.path.isfile("/marc-patches/sar_recipes.yaml"),
        "container recipe path shadows cwd yaml",
    )
    def test_unquoted_yaml_on_still_sets_on_landmark_type(self):
        namespace = self._exec_combos("<sar-yaml-true-on>")
        poses = ["person_lying_down", "person_walking", "person_sitting"]
        chars = ["female_student_01", "male_student_01", "professor_01"]
        yaml_text = (
            "recipes:\n"
            "  - id: sar03_bench_lie\n"
            "    pose: person_lying_down\n"
            "    support_landmark: bench\n"
            "    support_class: bench\n"
            "    relation: on\n"
            "    above: false\n"
        )
        old = dict(os.environ)
        cwd = os.getcwd()
        try:
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            os.environ["TRAINER_SAR_RECIPE_CURSOR"] = "0"
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_PEOPLE_ON_LANDMARK_PROB"] = "0"
            os.environ["TRAINER_FORCE_CLIMB_LANDMARK_PROB"] = "0"
            with tempfile.TemporaryDirectory() as tmp:
                rec_dir = Path(tmp) / "simulation-platform" / "patches"
                rec_dir.mkdir(parents=True)
                (rec_dir / "sar_recipes.yaml").write_text(
                    yaml_text, encoding="utf-8")
                loaded = yaml.safe_load(yaml_text)
                self.assertIs(loaded["recipes"][0]["relation"], True)
                os.chdir(tmp)
                _, specs = namespace["build"](
                    [{"name": "bench_0"}], list(chars), list(poses),
                )
            people = [o for o in specs if o.get("character")]
            targets = [
                o for o in people
                if (o.get("placement") or {}).get("recipe_id")
            ]
            self.assertEqual(len(targets), 1, people)
            pl = targets[0].get("placement") or {}
            self.assertEqual(pl.get("on_landmark_type"), "bench")
            with open("/tmp/marc_sar_recipe.json", encoding="utf-8") as f:
                dumped = json.load(f)
            self.assertEqual(dumped.get("relation"), "on")
        finally:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(old)


if __name__ == "__main__":
    unittest.main()
