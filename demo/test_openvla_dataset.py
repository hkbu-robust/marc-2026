"""Tests for the MARC OpenVLA export contract."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from manip_dataset.openvla_dataset import (
    DatasetValidationError,
    build_openvla_state,
    validate_export,
    validate_pick_metadata,
)


class OpenVlaDatasetContractTests(unittest.TestCase):
    def test_six_joint_state_is_padded_before_gripper(self):
        q_arm = np.arange(18, dtype=np.float32).reshape(3, 6)
        q_grip = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        state = build_openvla_state(q_arm, q_grip)
        self.assertEqual((3, 8), state.shape)
        np.testing.assert_array_equal(state[:, :6], q_arm)
        np.testing.assert_array_equal(state[:, 6], np.zeros(3))
        np.testing.assert_array_equal(state[:, 7], q_grip)

    def test_pick_metadata_rejects_unrecorded_place_semantics(self):
        with self.assertRaisesRegex(DatasetValidationError, "pick-only language"):
            validate_pick_metadata({
                "task": "pick",
                "object_name": "tumbler",
                "language_instruction": "pick up the tumbler and put it in the basket",
                "includes_place": False,
            })

    def test_converter_emits_and_validates_canonical_contract(self):
        try:
            import h5py
        except ImportError:
            self.skipTest("h5py required for converter test")
        from manip_dataset.to_rlds import convert

        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "source")
            episode = os.path.join(source, "Episode 001")
            output = os.path.join(tmp, "export")
            os.makedirs(episode)
            count = 4
            with open(os.path.join(episode, "meta.json"), "w", encoding="utf-8") as handle:
                json.dump({
                    "success": True,
                    "task": "pick",
                    "includes_place": False,
                    "object_name": "tumbler",
                    "language_instruction": "pick up the tumbler",
                }, handle)
            with h5py.File(os.path.join(episode, "episode.hdf5"), "w") as handle:
                steps = handle.create_group("steps")
                steps.attrs["primary_camera"] = "base_left"
                steps.create_dataset(
                    "image_base_left",
                    data=np.zeros((count, 24, 32, 3), dtype=np.uint8),
                )
                action = np.zeros((count, 7), dtype=np.float32)
                action[2:, -1] = 1.0
                steps.create_dataset("action", data=action)
                steps.create_dataset("q_arm", data=np.ones((count, 6), dtype=np.float32))
                steps.create_dataset(
                    "q_grip", data=np.linspace(0.0, 0.3, count, dtype=np.float32)
                )
                steps.create_dataset("ee_pos", data=np.zeros((count, 3), dtype=np.float32))
                steps.create_dataset(
                    "ee_quat",
                    data=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (count, 1)),
                )
                steps.create_dataset("grasping", data=np.array([0, 0, 1, 1]))

            convert(source, output, image_size=16, camera="base_left")
            report = validate_export(output)
            self.assertEqual((1, count, "base_left"), (
                report.episodes, report.transitions, report.camera,
            ))
            with np.load(
                os.path.join(output, "episodes", "episode_0000.npz"),
                allow_pickle=True,
            ) as data:
                self.assertEqual((count, 8), data["state"].shape)
                np.testing.assert_array_equal(data["state"][:, 6], np.zeros(count))
                np.testing.assert_array_equal(data["action"][:, -1], [1, 1, 0, 0])
            with open(os.path.join(output, "dataset_statistics.json"), encoding="utf-8") as handle:
                stats = json.load(handle)
            self.assertEqual(8, len(stats["marc_pnp"]["proprio"]["q01"]))


if __name__ == "__main__":
    unittest.main()
