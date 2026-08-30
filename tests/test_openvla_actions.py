"""OpenVLA token codec, quaternion apply, gripper command, nested stats."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import numpy as np
import yaml

from marc_agent.openvla_actions import (
    ACTION_DIM,
    DEFAULT_UNNORM_KEY,
    GRIPPER_CLOSE,
    ActionStats,
    action_to_token_ids,
    build_dataset_statistics,
    compose_ee_quat,
    gripper_command,
    gripper_should_close,
    load_action_stats,
    openvla_prompt,
    pick_instruction,
    quat_mul,
    rotvec_to_quat,
    tokens_to_action,
)


_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openvla_action_cases.yaml")


def _load():
    with open(_FIXTURE, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class OpenVLAActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _load()
        cls.stats = ActionStats(
            q01=cls.fx["q01"],
            q99=cls.fx["q99"],
            mask=cls.fx["mask"],
            n_bins=cls.fx["n_bins"],
            unnorm_key=cls.fx["unnorm_key"],
            vocab_size=cls.fx["vocab_size"],
        )

    def test_decode_token_ids(self):
        vocab = int(self.fx["vocab_size"])
        for case in self.fx["decode_cases"]:
            with self.subTest(case=case["id"]):
                action = tokens_to_action(
                    case["token_ids"], self.stats, vocab_size=vocab,
                )
                self.assertEqual((ACTION_DIM,), action.shape)
                np.testing.assert_allclose(action, case["action"], atol=1e-12)
                self.assertEqual(
                    bool(case["grip_close"]),
                    gripper_should_close(float(action[6])),
                )
                self.assertAlmostEqual(
                    float(case["grip_cmd"]),
                    gripper_command(float(action[6])),
                )

    def test_round_trip_within_one_bin(self):
        raw = np.array([0.01, -0.02, 0.005, 0.04, -0.03, 0.02, 1.0], dtype=np.float64)
        tokens = action_to_token_ids(raw, self.stats)
        self.assertEqual((ACTION_DIM,), tokens.shape)
        decoded = tokens_to_action(tokens, self.stats)
        # Gripper is unmasked: 1.0 digitizes onto the last bin center, not 1.0.
        np.testing.assert_allclose(decoded[:6], raw[:6], atol=0.002)
        self.assertGreater(float(decoded[6]), 0.5)
        # Bin centers re-encode stably (the last-edge token 31744 decodes to 31745).
        stable = tokens_to_action(action_to_token_ids(decoded, self.stats), self.stats)
        np.testing.assert_allclose(stable, decoded, atol=1e-12)

    def test_quat_cases(self):
        for case in self.fx["quat_cases"]:
            with self.subTest(case=case["id"]):
                got = compose_ee_quat(
                    case["q"], case["droll"], case["dpitch"], case["dyaw"],
                )
                np.testing.assert_allclose(got, case["expected"], atol=1e-12)

    def test_quat_matches_rotvec_identity_compose(self):
        delta = rotvec_to_quat([0.1, -0.2, 0.3])
        composed = compose_ee_quat([1.0, 0.0, 0.0, 0.0], 0.1, -0.2, 0.3)
        np.testing.assert_allclose(composed, delta, atol=1e-12)
        q = np.array([0.9238795325112867, 0.0, 0.0, 0.3826834323650898])
        got = compose_ee_quat(q, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(got, quat_mul(q, [1.0, 0.0, 0.0, 0.0]), atol=1e-12)

    def test_grip_cases(self):
        for case in self.fx["grip_cases"]:
            open_is_high = case["open_is_high"]
            with self.subTest(value=case["value"], open_is_high=open_is_high):
                self.assertEqual(
                    bool(case["close"]),
                    gripper_should_close(case["value"], open_is_high=open_is_high),
                )
                self.assertAlmostEqual(
                    float(case["cmd"]),
                    gripper_command(
                        case["value"], open_is_high=open_is_high, close=GRIPPER_CLOSE,
                    ),
                )

    def test_nested_stats_round_trip(self):
        actions = np.array(
            [
                [0.01, 0.0, -0.01, 0.0, 0.0, 0.02, 0.0],
                [0.02, -0.01, 0.0, 0.01, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        proprio = np.array(
            [[0.1, 0.2, -1.0, 0.0, -0.5, 1.5, 0.0],
             [0.1, 0.3, -1.1, 0.0, -0.4, 1.5, 0.4]],
            dtype=np.float64,
        )
        payload = build_dataset_statistics(
            actions, proprio, num_episodes=2, image_size=224, camera="base_left",
        )
        self.assertIn(DEFAULT_UNNORM_KEY, payload)
        block = payload[DEFAULT_UNNORM_KEY]["action"]
        self.assertEqual([True, True, True, True, True, True, False], block["mask"])
        self.assertEqual(2, payload[DEFAULT_UNNORM_KEY]["num_transitions"])
        self.assertEqual(2, payload[DEFAULT_UNNORM_KEY]["num_trajectories"])
        self.assertIn("q01", block)
        self.assertIn("q99", block)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dataset_statistics.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            loaded = load_action_stats(path)
        self.assertEqual(DEFAULT_UNNORM_KEY, loaded.unnorm_key)
        np.testing.assert_allclose(loaded.q01, block["q01"])
        np.testing.assert_allclose(loaded.q99, block["q99"])
        np.testing.assert_array_equal(loaded.mask, block["mask"])

    def test_mean_std_only_stats_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dataset_statistics.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({
                    "action_mean": [0.0] * 7,
                    "action_std": [1.0] * 7,
                }, handle)
            with self.assertRaises(ValueError):
                load_action_stats(path)

    def test_prompt_and_instruction(self):
        self.assertEqual(
            "In: What action should the robot take to pick up the tumbler?\nOut:",
            openvla_prompt("pick up the tumbler"),
        )
        target = type("T", (), {"name": "mug"})()
        self.assertEqual("pick up the mug", pick_instruction(target))
        self.assertEqual("pick up the object", pick_instruction(None))

    def test_yaw90_norm(self):
        q = compose_ee_quat([1, 0, 0, 0], 0.0, 0.0, math.pi / 2)
        self.assertAlmostEqual(1.0, float(np.linalg.norm(q)), places=12)


if __name__ == "__main__":
    unittest.main()
