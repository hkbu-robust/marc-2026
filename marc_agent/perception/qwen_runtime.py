"""Configuration and offline loading for the bundled Stage 1 Qwen verifier."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from marc_agent.perception.bundle_paths import (
    STAGE1_QWEN_CONTRACT,
    STAGE1_QWEN_MODEL_DIR,
)


class QwenModelUnavailable(RuntimeError):
    """Raised when the configured local model snapshot is incomplete."""


@dataclass(frozen=True)
class QwenRuntimeConfig:
    model_dir: Path
    dtype: str
    device_map: str
    attention_implementation: str
    max_new_tokens: int

    @classmethod
    def load(
        cls,
        contract_path: Path = STAGE1_QWEN_CONTRACT,
        *,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "QwenRuntimeConfig":
        env = os.environ if environ is None else environ
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        inference = payload.get("inference") or {}
        configured_dir = env.get("MARC_QWEN_MODEL_DIR")
        model_dir = (
            Path(configured_dir).expanduser()
            if configured_dir
            else STAGE1_QWEN_MODEL_DIR
        )
        return cls(
            model_dir=model_dir.resolve(),
            dtype=str(env.get("MARC_QWEN_DTYPE") or inference.get("dtype", "bfloat16")),
            device_map=str(
                env.get("MARC_QWEN_DEVICE_MAP") or inference.get("device_map", "auto")
            ),
            attention_implementation=str(
                env.get("MARC_QWEN_ATTN_IMPLEMENTATION")
                or inference.get("attention_implementation", "sdpa")
            ),
            max_new_tokens=int(
                env.get("MARC_QWEN_MAX_NEW_TOKENS")
                or inference.get("max_new_tokens", 8)
            ),
        )


@dataclass(frozen=True)
class QwenBundleStatus:
    model_dir: Path
    missing: tuple[str, ...]
    weight_files: tuple[Path, ...]

    @property
    def ready(self) -> bool:
        return not self.missing


_REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        return path.read_bytes()[:80].startswith(
            b"version https://git-lfs.github.com/spec/v1"
        )
    except OSError:
        return False


def bundle_status(config: Optional[QwenRuntimeConfig] = None) -> QwenBundleStatus:
    """Inspect the local snapshot without importing Torch or Transformers."""
    cfg = config or QwenRuntimeConfig.load()
    missing: list[str] = []

    if not cfg.model_dir.is_dir():
        return QwenBundleStatus(cfg.model_dir, ("model directory",), ())

    for name in _REQUIRED_SNAPSHOT_FILES:
        path = cfg.model_dir / name
        if not path.is_file():
            missing.append(name)

    index_path = cfg.model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = sorted(set((index.get("weight_map") or {}).values()))
        except (OSError, ValueError, AttributeError) as exc:
            missing.append(f"valid model.safetensors.index.json ({exc})")
            shard_names = []
        if not shard_names:
            missing.append("weight shards declared by model.safetensors.index.json")
        weight_files = tuple(cfg.model_dir / name for name in shard_names)
        missing.extend(
            name
            for name, path in zip(shard_names, weight_files)
            if not path.is_file()
        )
    else:
        weight_files = tuple(sorted(cfg.model_dir.glob("*.safetensors")))
        if not weight_files:
            missing.append("model.safetensors or model.safetensors.index.json")

    for path in weight_files:
        if path.is_file() and _is_lfs_pointer(path):
            missing.append(f"materialized Git LFS object for {path.name}")

    return QwenBundleStatus(cfg.model_dir, tuple(missing), weight_files)


def require_local_bundle(
    config: Optional[QwenRuntimeConfig] = None,
) -> QwenRuntimeConfig:
    """Return a usable config or explain exactly which local artifacts are absent."""
    cfg = config or QwenRuntimeConfig.load()
    status = bundle_status(cfg)
    if not status.ready:
        details = ", ".join(status.missing)
        raise QwenModelUnavailable(
            f"Qwen model bundle is not ready at {status.model_dir}: missing {details}. "
            "Copy the complete pinned snapshot there and run the bundle check again."
        )
    return cfg


def load_local_qwen(config: Optional[QwenRuntimeConfig] = None):
    """Load the processor and model from disk without any Hub fallback."""
    cfg = require_local_bundle(config)
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Qwen inference requires torch, transformers, accelerate, and safetensors "
            "in the participant image."
        ) from exc

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    try:
        dtype = dtypes[cfg.dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported MARC_QWEN_DTYPE: {cfg.dtype!r}") from exc

    processor = AutoProcessor.from_pretrained(
        str(cfg.model_dir),
        local_files_only=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(cfg.model_dir),
        local_files_only=True,
        dtype=dtype,
        device_map=cfg.device_map,
        attn_implementation=cfg.attention_implementation,
    )
    model.eval()
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the offline Qwen model bundle")
    parser.parse_args()
    status = bundle_status()
    if status.ready:
        print(
            f"Qwen bundle ready: {status.model_dir} "
            f"({len(status.weight_files)} weight files)"
        )
        return 0
    print(f"Qwen bundle not ready: {status.model_dir}")
    for item in status.missing:
        print(f"- missing: {item}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
