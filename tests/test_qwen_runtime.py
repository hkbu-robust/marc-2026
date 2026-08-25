"""Tests for Qwen configuration and offline bundle readiness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from marc_agent.perception.bundle_paths import STAGE1_QWEN_CONTRACT
from marc_agent.perception.qwen_runtime import (
    QwenModelUnavailable,
    QwenRuntimeConfig,
    bundle_status,
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
