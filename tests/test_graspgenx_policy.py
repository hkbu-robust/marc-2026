"""Tests for the GraspGen-X pick seam without requiring the NVIDIA package."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from marc_agent.graspgenx_policy import (
    CROP_BBOX_INSET,
    MIN_CLOUD_POINTS,
    as_grasp,
    best_grasp,
    crop_depth_cloud,
    grasp_yaw_from_matrix,
    has_checkpoint,
    load_graspgenx,
    optical_cloud_to_arm_base,
    points_from_target,
    select_grasp_index,
    select_grasp_near,
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

    def test_points_from_target_empty_without_depth(self):
        target = SimpleNamespace(
            depth=None, bbox_xyxy=(8, 8, 56, 40), k=(50.0, 50.0, 32.0, 24.0)
        )
        cloud = points_from_target(target)
        self.assertEqual((0, 3), cloud.shape)

    def test_points_from_target_too_few_points(self):
        depth = np.full((48, 64), 1.0, dtype=np.float32)
        target = SimpleNamespace(
            depth=depth,
            bbox_xyxy=(20, 20, 24, 24),
            k=(50.0, 50.0, 32.0, 24.0),
            rgb_shape=None,
            R=None,
            t=None,
        )
        cloud = points_from_target(target)
        self.assertEqual((0, 3), cloud.shape)
        self.assertLess(
            len(crop_depth_cloud(depth, (20, 20, 24, 24), (50.0, 50.0, 32.0, 24.0))),
            MIN_CLOUD_POINTS,
        )

    def test_select_grasp_index_conf_longer_than_grasps(self):
        grasps = [np.eye(4)]
        conf = np.array([0.1, 0.9, 0.2])
        self.assertEqual(0, select_grasp_index(grasps, conf))

    def test_select_grasp_index_argmax_when_aligned(self):
        grasps = [0, 1, 2]
        conf = np.array([0.1, 0.9, 0.2])
        self.assertEqual(1, select_grasp_index(grasps, conf))

    def test_optical_cloud_converts_without_tf(self):
        optical = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
        cloud = optical_cloud_to_arm_base(optical, R=None, t=None)
        self.assertEqual((1, 3), cloud.shape)
        self.assertAlmostEqual(0.32, float(cloud[0, 0]), places=5)
        self.assertAlmostEqual(0.0, float(cloud[0, 1]), places=5)
        self.assertAlmostEqual(0.0, float(cloud[0, 2]), places=5)

    def test_points_from_target_crop_without_tf(self):
        depth = np.full((48, 64), 0.55, dtype=np.float32)
        target = SimpleNamespace(
            depth=depth,
            bbox_xyxy=(8, 8, 56, 40),
            k=(50.0, 50.0, 32.0, 24.0),
            rgb_shape=(48, 64),
            R=None,
            t=None,
        )
        cloud = points_from_target(target)
        self.assertGreaterEqual(len(cloud), MIN_CLOUD_POINTS)
        self.assertTrue(np.all(np.isfinite(cloud)))

    def test_crop_inset_shrinks_cloud(self):
        depth = np.full((48, 64), 1.0, dtype=np.float32)
        bbox = (8, 8, 56, 40)
        k = (50.0, 50.0, 32.0, 24.0)
        full = crop_depth_cloud(depth, bbox, k, stride=2, inset=0.0)
        inset = crop_depth_cloud(
            depth, bbox, k, stride=2, inset=CROP_BBOX_INSET
        )
        self.assertGreater(len(full), len(inset))
        self.assertGreater(len(inset), 0)

    def test_select_grasp_near_keeps_close_tcp(self):
        far = np.eye(4, dtype=np.float64)
        far[:3, 3] = (0.35, -0.176, -0.059)
        near = np.eye(4, dtype=np.float64)
        near[:3, 3] = (0.26, 0.0, -0.05)
        got = select_grasp_near(
            [far, near], [0.99, 0.40], (0.251, -0.005, -0.016)
        )
        self.assertIsNotNone(got)
        self.assertAlmostEqual(0.26, got.xyz[0], places=5)
        self.assertEqual(1, got.n_keep)

    def test_select_grasp_near_drops_far_tcp(self):
        far = np.eye(4, dtype=np.float64)
        far[:3, 3] = (0.35, -0.176, -0.059)
        got = select_grasp_near([far], [0.99], (0.251, -0.005, -0.016))
        self.assertIsNone(got)

    def test_select_grasp_near_drops_aerial_z(self):
        air = np.eye(4, dtype=np.float64)
        air[:3, 3] = (0.26, 0.0, 0.128)
        got = select_grasp_near([air], [0.99], (0.251, -0.005, -0.016))
        self.assertIsNone(got)

    def test_select_grasp_near_drops_flipped_yaw(self):
        flip = np.eye(4, dtype=np.float64)
        flip[:3, 3] = (0.26, 0.0, -0.05)
        flip[0, 0] = -1.0
        flip[1, 1] = -1.0
        got = select_grasp_near([flip], [0.99], (0.251, -0.005, -0.016))
        self.assertIsNone(got)

    def test_select_grasp_near_prefers_small_yaw(self):
        flip = np.eye(4, dtype=np.float64)
        flip[:3, 3] = (0.26, 0.0, -0.05)
        flip[0, 0] = -1.0
        flip[1, 1] = -1.0
        front = np.eye(4, dtype=np.float64)
        front[:3, 3] = (0.255, 0.0, -0.04)
        got = select_grasp_near(
            [flip, front], [0.99, 0.40], (0.251, -0.005, -0.016)
        )
        self.assertIsNotNone(got)
        self.assertAlmostEqual(0.255, got.xyz[0], places=5)

    def test_best_grasp_near_xyz_rejects_side_tcp(self):
        cloud = np.zeros((32, 3), dtype=np.float32)
        pose = best_grasp(
            cloud,
            generate_fn=lambda _pts: (0.35, -0.176, -0.059),
            near_xyz=(0.251, -0.005, -0.016),
        )
        self.assertIsNone(pose)

    def test_planner_kwargs_force_one_iteration(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "marc_agent",
            "graspgenx_policy.py",
        )
        with open(path, encoding="utf-8") as stream:
            src = stream.read()
        self.assertIn('"min_grasps": 1', src)
        self.assertIn('"max_tries": 1', src)
        self.assertNotIn("inspect.signature", src)


if __name__ == "__main__":
    unittest.main()
