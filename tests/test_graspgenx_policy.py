"""Tests for the GraspGen-X pick seam without requiring the NVIDIA package."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from marc_agent.graspgenx_policy import (
    as_grasp,
    best_grasp,
    crop_depth_cloud,
    grasp_yaw_from_matrix,
    has_checkpoint,
    load_graspgenx,
    unload_graspgenx,
)


class GraspGenXPolicyTests(unittest.TestCase):
    def test_missing_weights_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(has_checkpoint(tmp))
            self.assertIsNone(load_graspgenx(tmp))

    def test_fake_generator_returns_pose(self):
        cloud = np.array([[0.4, 0.0, -0.3]], dtype=np.float32)
        pose = best_grasp(cloud, generate_fn=lambda _pts: (0.42, 0.01, -0.28))
        self.assertIsNotNone(pose)
        self.assertEqual((0.42, 0.01, -0.28), pose.xyz)

    def test_matrix_pose_uses_translation(self):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (0.35, -0.05, -0.22)
        pose = as_grasp(matrix)
        self.assertEqual((0.35, -0.05, -0.22), pose.xyz)

    def test_crop_empty_without_depth(self):
        cloud = crop_depth_cloud(None, (10, 10, 40, 40), (500, 500, 320, 240))
        self.assertEqual((0, 3), cloud.shape)

    def test_crop_unprojects_bbox(self):
        depth = np.full((48, 64), 1.0, dtype=np.float32)
        cloud = crop_depth_cloud(
            depth, (20, 10, 40, 30), (50.0, 50.0, 32.0, 24.0), stride=4
        )
        self.assertGreater(len(cloud), 0)
        self.assertEqual(3, cloud.shape[1])

    def test_bundled_runtime_dir_has_no_checkpoint(self):
        root = os.path.dirname(os.path.dirname(__file__))
        bundled = os.path.join(root, "models", "runtime", "graspgenx")
        self.assertFalse(has_checkpoint(bundled))

    def test_unload_clears_cached_generator(self):
        import marc_agent.graspgenx_policy as policy

        policy._generator = lambda _pts: (0.1, 0.0, 0.0)
        unload_graspgenx()
        self.assertIsNone(policy._generator)

    def test_gen_only_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            gen = os.path.join(tmp, "gen")
            os.mkdir(gen)
            with open(os.path.join(gen, "epoch_1.pth"), "wb") as handle:
                handle.write(b"ckpt")
            os.mkdir(os.path.join(tmp, "dis"))
            self.assertFalse(has_checkpoint(tmp))

    def test_both_gen_and_dis_are_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("gen", "dis"):
                folder = os.path.join(tmp, name)
                os.mkdir(folder)
                with open(os.path.join(folder, "epoch_1.pth"), "wb") as handle:
                    handle.write(b"ckpt")
            self.assertTrue(has_checkpoint(tmp))

    def test_yaw_from_identity_is_zero(self):
        self.assertAlmostEqual(0.0, grasp_yaw_from_matrix(np.eye(4)))

    def test_yaw_from_z_rotation(self):
        yaw = 0.7
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = np.cos(yaw)
        matrix[0, 1] = -np.sin(yaw)
        matrix[1, 0] = np.sin(yaw)
        matrix[1, 1] = np.cos(yaw)
        self.assertAlmostEqual(yaw, grasp_yaw_from_matrix(matrix), places=6)


if __name__ == "__main__":
    unittest.main()
