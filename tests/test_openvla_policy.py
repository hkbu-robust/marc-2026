"""OpenVLA policy stub and weight-loader fallback (no 7B download)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from marc_agent.openvla_actions import ACTION_DIM, build_dataset_statistics
from marc_agent.openvla_policy import (
    GeometricGraspStub,
    HuggingFaceOpenVLAPolicy,
    PolicyObservation,
    RemoteOpenVLAPolicy,
    load_openvla,
    resolve_weights_path,
)


class OpenVLAPolicyTests(unittest.TestCase):
    def test_geom_stub_moves_then_closes(self):
        stub = GeometricGraspStub((0.40, 0.0, -0.30), step_m=0.05, close_m=0.04)
        far = PolicyObservation(ee_xyz=np.array([0.10, 0.0, 0.0]))
        action = stub.act(far)
        self.assertEqual((ACTION_DIM,), action.shape)
        self.assertGreater(float(action[0]), 0.0)
        self.assertEqual(0.0, float(action[6]))
        near = PolicyObservation(ee_xyz=np.array([0.40, 0.0, -0.30]))
        close = stub.act(near)
        self.assertEqual(1.0, float(close[6]))
        np.testing.assert_allclose(close[:3], 0.0, atol=1e-6)

    def test_load_without_weights_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_openvla(tmp))
            self.assertIsNone(resolve_weights_path(tmp))

    def test_load_missing_path_is_none(self):
        self.assertIsNone(load_openvla(os.path.join("no", "such", "openvla.pt")))

    def test_source_dir_without_snapshot_is_not_weights(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bundled = os.path.join(root, "models", "runtime", "openvla")
        self.assertTrue(os.path.isdir(bundled))
        self.assertIsNone(resolve_weights_path(bundled))

    def test_fake_hf_dir_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_type": "openvla"}, handle)
            with open(os.path.join(tmp, "dataset_statistics.json"), "w", encoding="utf-8") as handle:
                json.dump(build_dataset_statistics(np.zeros((8, 7))), handle)
            self.assertEqual(tmp, resolve_weights_path(tmp))
            self.assertIsNone(load_openvla(tmp))

    def test_fake_hf_dir_without_stats_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_type": "openvla"}, handle)
            self.assertIsNone(load_openvla(tmp))

    def test_lora_only_dir_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_type": "openvla"}, handle)
            with open(os.path.join(tmp, "adapter_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"peft_type": "LORA"}, handle)
            with open(os.path.join(tmp, "adapter_model.safetensors"), "wb") as handle:
                handle.write(b"not-a-model")
            with open(os.path.join(tmp, "dataset_statistics.json"), "w", encoding="utf-8") as handle:
                json.dump(build_dataset_statistics(np.zeros((4, 7))), handle)
            self.assertIsNone(load_openvla(tmp))

    def test_act_without_image_is_zero(self):
        stats_payload = build_dataset_statistics(np.zeros((4, 7)))
        from marc_agent.openvla_actions import load_action_stats
        stats = load_action_stats(stats_payload)
        policy = HuggingFaceOpenVLAPolicy(model=None, processor=None, stats=stats)
        zeros = policy.act(PolicyObservation(instruction="pick up the mug"))
        self.assertEqual((ACTION_DIM,), zeros.shape)
        np.testing.assert_allclose(zeros, 0.0)

    def test_remote_vllm_decodes_seven_action_tokens(self):
        seen = {}

        def post(url, body, headers, timeout_s):
            seen.update(
                url=url,
                payload=json.loads(body),
                authorization=headers["Authorization"],
                timeout_s=timeout_s,
            )
            return json.dumps({
                "choices": [{
                    "token_ids": [31877, 31858, 31921, 31883, 31744, 31885, 31872],
                }],
            }).encode()

        policy = RemoteOpenVLAPolicy.from_env({}, post=post)
        action = policy.act(PolicyObservation(
            rgb=np.zeros((32, 48, 3), dtype=np.uint8),
            instruction="pick up the tumbler",
        ))
        self.assertEqual((ACTION_DIM,), action.shape)
        self.assertEqual(
            "http://10.0.0.200:8003/v1/chat/completions", seen["url"],
        )
        self.assertEqual(ACTION_DIM, seen["payload"]["max_tokens"])
        self.assertTrue(seen["payload"]["return_token_ids"])
        self.assertLess(float(action[6]), 0.5)


if __name__ == "__main__":
    unittest.main()
