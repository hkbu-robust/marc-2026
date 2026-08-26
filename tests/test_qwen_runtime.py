"""Tests for Qwen configuration and offline bundle readiness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from marc_agent.perception.bundle_paths import STAGE1_QWEN_CONTRACT
from marc_agent.perception.qwen_runtime import (
    QwenBackend,
    QwenConfigurationError,
    QwenModelUnavailable,
    QwenRemoteConfig,
    QwenRuntimeConfig,
    bundle_status,
    configured_backend,
    configured_max_crops,
    require_local_bundle,
)


class QwenRuntimeTests(unittest.TestCase):
    def _config(self, model_dir: Path) -> QwenRuntimeConfig:
        return QwenRuntimeConfig(
            model_dir=model_dir,
            dtype="bfloat16",
            device_map="auto",
            attention_implementation="sdpa",
            max_new_tokens=8,
        )

    def test_contract_is_configurable_before_model_arrives(self):
        config = QwenRuntimeConfig.load(environ={})
        self.assertTrue(STAGE1_QWEN_CONTRACT.is_file())
        self.assertEqual("bfloat16", config.dtype)
        self.assertEqual("sdpa", config.attention_implementation)
        self.assertEqual(8, config.max_new_tokens)
        self.assertFalse(bundle_status(config).ready)

    def test_environment_can_point_at_later_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = QwenRuntimeConfig.load(environ={"MARC_QWEN_MODEL_DIR": tmp})
            self.assertEqual(Path(tmp).resolve(), config.model_dir)

    def test_backend_defaults_local_and_accepts_remote(self):
        self.assertEqual(QwenBackend.LOCAL, configured_backend({}))
        self.assertEqual(
            QwenBackend.REMOTE,
            configured_backend({"MARC_QWEN_BACKEND": "remote"}),
        )
        with self.assertRaises(QwenConfigurationError):
            configured_backend({"MARC_QWEN_BACKEND": "cloud"})

    def test_remote_config_requires_url_and_key(self):
        with self.assertRaisesRegex(QwenConfigurationError, "BASE_URL"):
            QwenRemoteConfig.load(environ={})
        with self.assertRaisesRegex(QwenConfigurationError, "API_KEY"):
            QwenRemoteConfig.load(
                environ={"MARC_QWEN_BASE_URL": "http://10.0.0.2:8000/v1"}
            )

    def test_remote_config_builds_chat_endpoint(self):
        config = QwenRemoteConfig.load(
            environ={
                "MARC_QWEN_BASE_URL": "http://10.0.0.2:8000/v1/",
                "MARC_QWEN_API_KEY": "secret",
                "MARC_QWEN_MODEL": "qwen-test",
                "MARC_QWEN_TIMEOUT_S": "4.5",
                "MARC_QWEN_JPEG_QUALITY": "80",
            }
        )
        self.assertEqual(
            "http://10.0.0.2:8000/v1/chat/completions",
            config.chat_completions_url,
        )
        self.assertEqual("http://10.0.0.2:8000/v1/models", config.models_url)
        self.assertEqual("qwen-test", config.model_name)
        self.assertEqual(4.5, config.timeout_s)
        self.assertEqual(80, config.jpeg_quality)

    def test_remote_timeout_defaults_to_ninety_seconds(self):
        config = QwenRemoteConfig.load(
            environ={
                "MARC_QWEN_BASE_URL": "http://10.0.0.2:8000/v1",
                "MARC_QWEN_API_KEY": "secret",
            }
        )
        self.assertEqual(90.0, config.timeout_s)

    def test_remote_generation_defaults_to_forty_eight_tokens(self):
        config = QwenRemoteConfig.load(
            environ={
                "MARC_QWEN_BASE_URL": "http://10.0.0.2:8000/v1",
                "MARC_QWEN_API_KEY": "secret",
            }
        )
        self.assertEqual(48, config.max_new_tokens)

    def test_crop_limit_is_bounded(self):
        self.assertEqual(12, configured_max_crops({}))
        self.assertEqual(0, configured_max_crops({"MARC_QWEN_MAX_CROPS": "0"}))
        with self.assertRaises(QwenConfigurationError):
            configured_max_crops({"MARC_QWEN_MAX_CROPS": "-1"})

    def test_complete_sharded_snapshot_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                (model_dir / name).write_text("{}", encoding="utf-8")
            (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights-1")
            (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"weights-2")
            index = {
                "weight_map": {
                    "visual.weight": "model-00001-of-00002.safetensors",
                    "language.weight": "model-00002-of-00002.safetensors",
                }
            }
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )

            config = self._config(model_dir)
            status = bundle_status(config)
            self.assertTrue(status.ready)
            self.assertEqual(2, len(status.weight_files))
            self.assertEqual(config, require_local_bundle(config))

    def test_incomplete_snapshot_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(Path(tmp))
            with self.assertRaisesRegex(QwenModelUnavailable, "config.json"):
                require_local_bundle(config)

    def test_git_lfs_pointer_is_not_a_materialized_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer_config.json",
            ):
                (model_dir / name).write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
                "size 123\n",
                encoding="utf-8",
            )
            status = bundle_status(self._config(model_dir))
            self.assertFalse(status.ready)
            self.assertTrue(any("Git LFS" in item for item in status.missing))


if __name__ == "__main__":
    unittest.main()
