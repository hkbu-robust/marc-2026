"""Storage contract for OpenVLA demonstration episodes."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import h5py
import numpy as np

from manip_dataset.recorder import EpisodeRecorder


class RecorderStorageTests(unittest.TestCase):
    def test_keeps_only_primary_rgb_in_hdf5_and_writes_no_frame_pngs(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = EpisodeRecorder(root, "u1", primary_camera="base_left")
            recorder.start_episode(object_name="tumbler")
            recorder.add_step(
                {
                    "image_base_left": np.full((8, 12, 3), 10, dtype=np.uint8),
                    "image_base_right": np.full((8, 12, 3), 20, dtype=np.uint8),
                    "image_gripper_left": np.full((8, 12, 3), 30, dtype=np.uint8),
                    "image_gripper_right": np.full((8, 12, 3), 40, dtype=np.uint8),
                    "depth_base": np.ones((8, 12), dtype=np.float32),
                    "depth_gripper": np.ones((8, 12), dtype=np.float32),
                    "q_arm": np.zeros(6, dtype=np.float32),
                    "q_grip": np.float32(0.42),
                    "q_inner": np.zeros(2, dtype=np.float32),
                    "ee_pos": np.zeros(3, dtype=np.float32),
                    "ee_quat": np.array([1, 0, 0, 0], dtype=np.float32),
                    "grasping": 1,
                    "t_sim": np.float64(1.0),
                    "t_wall": np.float64(2.0),
                },
                np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            )

            episode_dir = recorder.keep()

            with h5py.File(os.path.join(episode_dir, "episode.hdf5"), "r") as h5:
                keys = set(h5["steps"].keys())
                self.assertIn("image_base_left", keys)
                self.assertNotIn("image_base_right", keys)
                self.assertNotIn("image_gripper_left", keys)
                self.assertNotIn("image_gripper_right", keys)
                self.assertNotIn("depth_base", keys)
                self.assertNotIn("depth_gripper", keys)
                self.assertEqual("image_base_left", h5["steps"].attrs["rgb_views"])

            self.assertFalse(os.path.exists(os.path.join(episode_dir, "images")))
            with open(os.path.join(episode_dir, "meta.json")) as handle:
                meta = json.load(handle)
            self.assertEqual({}, meta["png_counts"])
            self.assertNotIn("images_dir", meta)


if __name__ == "__main__":
    unittest.main()
