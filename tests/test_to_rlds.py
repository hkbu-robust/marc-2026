"""OpenVLA-oriented NPZ export writes nested q01/q99/mask stats."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

import numpy as np

from marc_agent.openvla_actions import DEFAULT_UNNORM_KEY, build_dataset_statistics

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO = os.path.join(_ROOT, "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)


class ToRldsStatsTests(unittest.TestCase):
    def test_build_stats_schema(self):
        rng = np.random.default_rng(0)
        actions = rng.normal(scale=[0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 1.0], size=(50, 7))
        actions[:, 6] = (actions[:, 6] > 0).astype(np.float64)
        proprio = rng.normal(size=(50, 7))
        stats = build_dataset_statistics(
            actions, proprio, num_episodes=3, camera="auto",
        )
        block = stats[DEFAULT_UNNORM_KEY]["action"]
        self.assertEqual(7, len(block["q01"]))
        self.assertEqual(7, len(block["q99"]))
        self.assertEqual([True] * 6 + [False], block["mask"])
        self.assertEqual(50, stats[DEFAULT_UNNORM_KEY]["num_transitions"])
        self.assertEqual(3, stats[DEFAULT_UNNORM_KEY]["num_trajectories"])
        self.assertIn("proprio", stats[DEFAULT_UNNORM_KEY])
        self.assertEqual("ee_delta_arm_base_dx_dy_dz_droll_dpitch_dyaw_grip",
                         stats["action_convention"])
        self.assertEqual(256, stats["n_bins"])

    def test_convert_writes_nested_stats(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py required for convert()")
        from manip_dataset.to_rlds import convert

        with tempfile.TemporaryDirectory() as tmp:
            ep = os.path.join(tmp, "src", "ep_0001")
            os.makedirs(ep)
            n = 4
            image = np.zeros((n, 32, 40, 3), dtype=np.uint8)
            action = np.array(
                [[0.01, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0],
                 [0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            q_arm = np.zeros((n, 6), dtype=np.float32)
            q_grip = np.array([0.0, 0.0, 0.4, 0.5], dtype=np.float32)
            ee_pos = np.zeros((n, 3), dtype=np.float32)
            ee_quat = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
            with open(os.path.join(ep, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "success": True,
                    "language_instruction": "pick up the tumbler",
                    "object_name": "tumbler",
                    "task": "pick",
                }, handle)
            with h5py.File(os.path.join(ep, "episode.hdf5"), "w") as handle:
                steps = handle.create_group("steps")
                steps.create_dataset("image_base_left", data=image)
                steps.create_dataset("action", data=action)
                steps.create_dataset("q_arm", data=q_arm)
                steps.create_dataset("q_grip", data=q_grip)
                steps.create_dataset("ee_pos", data=ee_pos)
                steps.create_dataset("ee_quat", data=ee_quat)
                steps.create_dataset("grasping", data=np.array([0, 0, 1, 1]))
                steps.attrs["primary_camera"] = "base_left"
            out = os.path.join(tmp, "rlds")
            convert(os.path.join(tmp, "src"), out, image_size=16, camera="auto")
            with open(os.path.join(out, "dataset_statistics.json"), encoding="utf-8") as handle:
                stats = json.load(handle)
            self.assertIn(DEFAULT_UNNORM_KEY, stats)
            block = stats[DEFAULT_UNNORM_KEY]["action"]
            self.assertEqual(7, len(block["q01"]))
            self.assertEqual(7, len(block["q99"]))
            self.assertEqual([True, True, True, True, True, True, False], block["mask"])
            self.assertEqual(1, stats[DEFAULT_UNNORM_KEY]["num_trajectories"])
            self.assertEqual(4, stats[DEFAULT_UNNORM_KEY]["num_transitions"])
            npz = np.load(os.path.join(out, "episodes", "episode_0000.npz"), allow_pickle=True)
            self.assertEqual((4, 16, 16, 3), npz["image"].shape)
            self.assertEqual((4, 7), npz["action"].shape)
            self.assertEqual((4, 7), npz["state"].shape)
            self.assertEqual((4, 4), npz["ee_quat"].shape)


if __name__ == "__main__":
    unittest.main()
