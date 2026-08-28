#!/usr/bin/env python3
"""Ego YOLO planner: stable ids, skip aliases, umbrella bias, path guard."""

from __future__ import annotations

import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PATCH = os.path.join(_ROOT, "simulation-platform", "patches")
if _PATCH not in sys.path:
    sys.path.insert(0, _PATCH)

import ego_dataset_plan as plan  # noqa: E402


class LabelIdTests(unittest.TestCase):
    def test_umbrella_is_id_eleven_with_chef_wood_named(self):
        names = plan.label_names()
        self.assertEqual(names[9], "master_chef_can")
        self.assertEqual(names[10], "wood_block")
        self.assertEqual(names[11], "umbrella")
        self.assertEqual(len(names), 12)

    def test_manip_object_does_not_replace_label_ids(self):
        self.assertEqual(plan.label_names()[0], "tumbler")
        self.assertEqual(
            plan.force_classes({"EGO_FORCE_CLASSES": "umbrella",
                                "MARC_MANIP_OBJECT": "tumbler"}),
            ["umbrella", "tumbler"],
        )
        self.assertEqual(plan.label_names()[11], "umbrella")


class SkipTests(unittest.TestCase):
    def test_default_skip_aliases(self):
        skip = plan.skip_classes({})
        self.assertTrue(plan.is_skipped("long_umbrella", skip))
        self.assertTrue(plan.is_skipped("closedlongumbrella", skip))
        self.assertTrue(plan.is_skipped("master_chef_can", skip))
        self.assertTrue(plan.is_skipped("chef_can", skip))
        self.assertTrue(plan.is_skipped("wood_block", skip))
        self.assertTrue(plan.is_skipped("wooden_block", skip))
        self.assertFalse(plan.is_skipped("umbrella", skip))
        self.assertFalse(plan.is_skipped("tumbler", skip))

    def test_spawn_drops_chef_wood_keeps_umbrella(self):
        skip = plan.skip_classes({})
        force = plan.force_classes({"EGO_FORCE_CLASSES": "umbrella"})
        spawn = plan.spawn_types(plan.CANONICAL_NAMES, skip=skip, force=force)
        self.assertIn("umbrella", spawn)
        self.assertNotIn("master_chef_can", spawn)
        self.assertNotIn("wood_block", spawn)
        self.assertNotIn("long_umbrella", spawn)
        self.assertEqual(plan.label_names()[11], "umbrella")


class ForcePlanTests(unittest.TestCase):
    def test_every_scene_has_umbrella_and_skips_excluded(self):
        rng = random.Random(0)
        spawn = [n for n in plan.CANONICAL_NAMES
                 if n not in ("master_chef_can", "wood_block")]
        scenes = plan.plan_scene_classes(
            rng, spawn, 80, 1, 3, 0.0, force=["umbrella"], frac=0.85)
        self.assertEqual(len(scenes), 80)
        self.assertTrue(all(scenes))
        for bag in scenes:
            self.assertIn("umbrella", bag)
            self.assertNotIn("master_chef_can", bag)
            self.assertNotIn("wood_block", bag)
            self.assertNotIn("long_umbrella", bag)
        objs = [c for bag in scenes for c in bag]
        self.assertGreater(objs.count("umbrella") / len(objs), 0.7)
        self.assertTrue(any(c != "umbrella" for c in objs))

    def test_empty_frac_zero_when_force_unset_override(self):
        self.assertEqual(plan.empty_frac({"EGO_FORCE_CLASSES": "umbrella"}), 0.0)
        self.assertEqual(
            plan.empty_frac({"EGO_FORCE_CLASSES": "umbrella",
                             "EGO_EMPTY_FRAC": "0.2"}),
            0.2)
        self.assertEqual(plan.empty_frac({"EGO_FORCE_CLASSES": "none"}), 0.05)


class OutGuardTests(unittest.TestCase):
    def test_rejects_chungmu_and_other_scenario_dumps(self):
        with self.assertRaises(SystemExit) as ctx:
            plan.assert_ego_out(
                "/metacom2026/trainer_output/marc2026_chungmu")
        self.assertIn("chungmu", str(ctx.exception).lower())
        with self.assertRaises(SystemExit):
            plan.assert_ego_out(
                "/metacom2026/trainer_output/marc2026_demo/scene_00000")

    def test_accepts_ego_yolo(self):
        path = plan.assert_ego_out("/metacom2026/trainer_output/ego_yolo")
        self.assertTrue(path.endswith("ego_yolo"))
        repo = os.path.join(_ROOT, "trainer_output", "ego_yolo")
        plan.assert_ego_out(repo)

    def test_default_out_is_ego_yolo(self):
        self.assertTrue(plan.resolve_out_dir({}).endswith("ego_yolo"))
        self.assertTrue(
            plan.resolve_out_dir({"EGO_OUT": "/tmp/ego_yolo"}).endswith(
                "ego_yolo"))


if __name__ == "__main__":
    unittest.main()
