#!/usr/bin/env python3
"""R03 pose×support matrix (situation_rules 2026.1)."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marc_agent.perception.situation_rules import (  # noqa: E402
    POSE_FALLBACK,
    POSE_GROUPS,
    RULES_VERSION,
    SITUATION_MATRIX,
    SUPPORT_CLASSES,
    resolve_situation,
)


class SituationRulesTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(RULES_VERSION, "2026.1")

    def test_pose_group_by_support_class_matrix(self):
        for group, poses in POSE_GROUPS.items():
            for support_name, members in SUPPORT_CLASSES.items():
                expected = SITUATION_MATRIX[(group, support_name)]
                support = next(iter(members))
                for pose in poses:
                    if pose == "person_prone":
                        self.assertEqual(
                            resolve_situation(pose, support=support),
                            "accident",
                            f"{pose} x {support}",
                        )
                        continue
                    self.assertEqual(
                        resolve_situation(pose, support=support),
                        expected,
                        f"{pose} x {support} ({group}, {support_name})",
                    )

    def test_every_support_member_uses_its_class(self):
        pose_by_group = {
            "upright": "person_walking",
            "seated": "person_sitting",
            "low": "person_crouching",
            "prostrate": "person_bending",
        }
        for group, pose in pose_by_group.items():
            for support_name, members in SUPPORT_CLASSES.items():
                expected = SITUATION_MATRIX[(group, support_name)]
                for support in members:
                    self.assertEqual(
                        resolve_situation(pose, support=support),
                        expected,
                        f"{pose} x {support}",
                    )

    def test_prone_is_accident_for_every_support(self):
        self.assertEqual(resolve_situation("person_prone"), "accident")
        self.assertEqual(
            resolve_situation("person_prone", surface="parking"), "accident")
        for members in SUPPORT_CLASSES.values():
            for support in members:
                self.assertEqual(
                    resolve_situation("person_prone", support=support),
                    "accident",
                    support,
                )

    def test_unknown_support_uses_the_official_pose_fallback(self):
        for pose, expected in POSE_FALLBACK.items():
            self.assertEqual(resolve_situation(pose), expected, pose)
        self.assertEqual(resolve_situation("person_bending"), "emergency")
        self.assertEqual(resolve_situation("person_reclining"), "normal")
        self.assertEqual(resolve_situation("person_lying_down"), "abnormal")
        self.assertEqual(resolve_situation("not_a_pose"), "normal")

    def test_surface_used_when_support_unknown(self):
        self.assertEqual(
            resolve_situation("person_bending", surface="grass"), "normal")
        self.assertEqual(
            resolve_situation("person_bending", surface="parking"), "emergency")
        self.assertEqual(
            resolve_situation(
                "person_lying_down", support="picnic_table"), "abnormal")


if __name__ == "__main__":
    unittest.main()
