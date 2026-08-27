#!/usr/bin/env python3
"""Checks for YOLO-mode object injection in the instances patch."""

from __future__ import annotations

import importlib.util
import os
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "simulation-platform" / "patches" / "patch_trainer_instances.py"


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "patch_trainer_instances", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class InstancesPatchTests(unittest.TestCase):
    def test_yolo_mode_injects_umbrella_skips_excluded(self):
        patch = _load_patch()
        self.assertIn("def _marc_yolo_inject", patch.HELPER)
        self.assertIn("if _marc_yolo_inject():", patch.NEW_LISTS)
        self.assertIn('_MARC_YOLO_FORCE_DEFAULT = ("umbrella",)', patch.HELPER)
        self.assertIn("long_umbrella,master_chef_can,wood_block", patch.HELPER)
        source = (
            patch.HELPER
            + "def build(catalog, selected_keys):\n"
            + patch.NEW_LISTS
            + "    return objects, landmarks\n"
        )
        namespace = {"os": os, "random": random}
        exec(compile(source, "<instances-yolo>", "exec"), namespace)
        catalog = {
            "objects": [
                {"key": "tumbler", "spec": {"name": "tumbler", "type": "tumbler"}},
                {"key": "master_chef_can",
                 "spec": {"name": "master_chef_can", "type": "master_chef_can"}},
                {"key": "wood_block",
                 "spec": {"name": "wood_block", "type": "wood_block"}},
                {"key": "umbrella", "spec": {"name": "umbrella", "type": "umbrella"}},
                {"key": "long_umbrella",
                 "spec": {"name": "long_umbrella", "type": "long_umbrella"}},
                {"key": "closedlongumbrella",
                 "spec": {"name": "closedlongumbrella", "type": "closedlongumbrella"}},
            ],
            "landmarks": [
                {"key": "bench", "spec": {"name": "bench_0"}},
            ],
        }
        old = dict(os.environ)
        try:
            os.environ["TRAINER_YOLO_MODE"] = "1"
            os.environ.pop("TRAINER_SKIP_CLASSES", None)
            os.environ.pop("TRAINER_YOLO_FORCE_CLASSES", None)
            os.environ["TRAINER_MIN_INSTANCES"] = "4"
            os.environ["TRAINER_MAX_INSTANCES"] = "4"
            os.environ["TRAINER_CLASS_INCLUSION_PROBABILITY"] = "1"
            objects, _landmarks = namespace["build"](catalog, {"tumbler", "bench"})
        finally:
            os.environ.clear()
            os.environ.update(old)
        names = {str(o.get("name") or o.get("type")) for o in objects}
        self.assertIn("tumbler", names)
        self.assertIn("umbrella", names)
        self.assertNotIn("master_chef_can", names)
        self.assertNotIn("wood_block", names)
        self.assertNotIn("long_umbrella", names)
        self.assertNotIn("closedlongumbrella", names)
        self.assertEqual(
            next(o["count"] for o in objects if o.get("name") == "umbrella"),
            4,
        )

    def test_yolo_mode_force_spawns_umbrella_absent_from_catalog(self):
        patch = _load_patch()
        source = (
            patch.HELPER
            + "def build(catalog, selected_keys):\n"
            + patch.NEW_LISTS
            + "    return objects, landmarks\n"
        )
        namespace = {"os": os, "random": random}
        exec(compile(source, "<instances-force>", "exec"), namespace)
        catalog = {
            "objects": [
                {"key": "tumbler", "spec": {"name": "tumbler", "type": "tumbler"}},
            ],
            "landmarks": [
                {"key": "bench", "spec": {"name": "bench_0"}},
            ],
        }
        old = dict(os.environ)
        try:
            os.environ["TRAINER_YOLO_MODE"] = "1"
            os.environ.pop("TRAINER_SKIP_CLASSES", None)
            os.environ.pop("TRAINER_YOLO_FORCE_CLASSES", None)
            os.environ["TRAINER_MIN_INSTANCES"] = "4"
            os.environ["TRAINER_MAX_INSTANCES"] = "4"
            os.environ["TRAINER_CLASS_INCLUSION_PROBABILITY"] = "1"
            objects, _landmarks = namespace["build"](catalog, {"tumbler", "bench"})
        finally:
            os.environ.clear()
            os.environ.update(old)
        names = {str(o.get("name") or o.get("type")) for o in objects}
        self.assertIn("tumbler", names)
        self.assertIn("umbrella", names)
        self.assertNotIn("master_chef_can", names)
        self.assertNotIn("wood_block", names)
        umb = next(o for o in objects if o.get("name") == "umbrella")
        self.assertEqual(umb["type"], "umbrella")
        self.assertEqual(umb["count"], 4)

    def test_default_mode_does_not_inject_unselected_chef(self):
        patch = _load_patch()
        source = (
            patch.HELPER
            + "def build(catalog, selected_keys):\n"
            + patch.NEW_LISTS
            + "    return objects, landmarks\n"
        )
        namespace = {"os": os, "random": random}
        exec(compile(source, "<instances-default>", "exec"), namespace)
        catalog = {
            "objects": [
                {"key": "tumbler", "spec": {"name": "tumbler", "type": "tumbler"}},
                {"key": "master_chef_can",
                 "spec": {"name": "master_chef_can", "type": "master_chef_can"}},
            ],
            "landmarks": [
                {"key": "bench", "spec": {"name": "bench_0"}},
            ],
        }
        old = dict(os.environ)
        try:
            os.environ.pop("TRAINER_YOLO_MODE", None)
            os.environ["TRAINER_CLASS_INCLUSION_PROBABILITY"] = "1"
            objects, _landmarks = namespace["build"](catalog, {"tumbler", "bench"})
        finally:
            os.environ.clear()
            os.environ.update(old)
        names = {str(o.get("name") or o.get("type")) for o in objects}
        self.assertEqual(names, {"tumbler"})

    def test_upgrade_existing_adds_force_types_to_inject_without_them(self):
        patch = _load_patch()
        src = (
            "def _marc_yolo_inject():\n"
            "    return True\n"
            "\ndef _build_deploy_lists(catalog, selected_keys):\n"
            "    objects = []\n"
            "    objects.append(spec)\n"
            "    if _marc_yolo_inject():\n"
            "        _have = set()\n"
            "            _have.add(name)\n"
            "            _have.add(key)\n"
            "    landmarks = []\n"
        )
        upgraded, note = patch.upgrade_existing(src)
        self.assertIn("yolo force-type helpers", note)
        self.assertIn("yolo force-type spawn", note)
        self.assertIn("def _marc_yolo_force_types", upgraded)
        self.assertIn("for _want in _marc_yolo_force_types():", upgraded)

    def test_upgrade_existing_adds_yolo_mode_to_skip_only_patch(self):
        patch = _load_patch()
        src = (
            patch.OLD_SKIP
            + "\ndef _build_deploy_lists(catalog, selected_keys):\n"
            "    objects = []\n"
            "    for it in catalog[\"objects\"]:\n"
            "        if it[\"key\"] not in selected_keys:\n"
            "            continue\n"
            "        spec = dict(it[\"spec\"])\n"
            "        spec[\"count\"] = _marc_instance_count(\n"
            "            \"TRAINER_MIN_INSTANCES\", \"TRAINER_MAX_INSTANCES\", 4, 6)\n"
            "        objects.append(spec)\n"
            "    landmarks = []\n"
        )
        upgraded, note = patch.upgrade_existing(src)
        self.assertTrue(note)
        self.assertIn("def _marc_yolo_inject", upgraded)
        self.assertIn("if _marc_yolo_inject():", upgraded)
        self.assertIn("def _marc_yolo_force_types", upgraded)
        self.assertIn("for _want in _marc_yolo_force_types():", upgraded)
        self.assertNotIn("if _marc_yolo_mode():", upgraded)
        self.assertIn("long_umbrella,master_chef_can,wood_block", upgraded)
        self.assertNotIn('if "umbrella" in blob', upgraded)
        self.assertNotIn(
            'os.environ.get("TRAINER_SKIP_CLASSES", "umbrella,long_umbrella")',
            upgraded,
        )

    def test_upgrade_existing_is_idempotent_on_current_helpers(self):
        patch = _load_patch()
        src = patch.HELPER + "\ndef _build_deploy_lists():\n" + patch.NEW_LISTS
        upgraded, note = patch.upgrade_existing(src)
        self.assertEqual("", note)
        self.assertEqual(src, upgraded)

    def test_yolo_inject_not_shadowed_by_nested_people_helper(self):
        patch = _load_patch()
        source = (
            patch.HELPER
            + "def build(catalog, selected_keys):\n"
            + patch.NEW_LISTS
            + "    def _marc_yolo_mode():\n"
            + "        return False\n"
            + "    _yolo_mode = _marc_yolo_mode()\n"
            + "    return objects, landmarks, _yolo_mode\n"
        )
        namespace = {"os": os, "random": random}
        exec(compile(source, "<instances-shadow>", "exec"), namespace)
        catalog = {
            "objects": [
                {"key": "tumbler", "spec": {"name": "tumbler", "type": "tumbler"}},
                {"key": "umbrella", "spec": {"name": "umbrella", "type": "umbrella"}},
            ],
            "landmarks": [
                {"key": "bench", "spec": {"name": "bench_0"}},
            ],
        }
        old = dict(os.environ)
        try:
            os.environ["TRAINER_YOLO_MODE"] = "1"
            os.environ.pop("TRAINER_SKIP_CLASSES", None)
            os.environ["TRAINER_MIN_INSTANCES"] = "4"
            os.environ["TRAINER_MAX_INSTANCES"] = "4"
            os.environ["TRAINER_CLASS_INCLUSION_PROBABILITY"] = "1"
            objects, _landmarks, yolo_flag = namespace["build"](
                catalog, {"tumbler", "bench"})
        finally:
            os.environ.clear()
            os.environ.update(old)
        names = {str(o.get("name") or o.get("type")) for o in objects}
        self.assertIn("umbrella", names)
        self.assertNotIn("master_chef_can", names)
        self.assertFalse(yolo_flag)

    def test_upgrade_existing_rewrites_shadowed_inject_call(self):
        patch = _load_patch()
        src = (
            "def _marc_skip_object(it):\n"
            "    return False\n"
            "\ndef _build_deploy_lists(catalog, selected_keys):\n"
            "    objects = []\n"
            "    objects.append(spec)\n"
            "    if _marc_yolo_mode():\n"
            "        _have = set()\n"
            "    landmarks = []\n"
            "    def _marc_yolo_mode():\n"
            "        return False\n"
        )
        upgraded, note = patch.upgrade_existing(src)
        self.assertIn("unshadow yolo inject call", note)
        self.assertIn("    if _marc_yolo_inject():\n        _have = set()\n", upgraded)
        self.assertNotIn("    if _marc_yolo_mode():\n        _have = set()\n", upgraded)
        self.assertIn("def _marc_yolo_inject", upgraded)
        self.assertIn("    def _marc_yolo_mode():\n", upgraded)


if __name__ == "__main__":
    unittest.main()
