"""Configuration and offline loading for the bundled Stage 1 Qwen verifier."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse

from marc_agent.perception.bundle_paths import (
    STAGE1_QWEN_CONTRACT,
    STAGE1_QWEN_MODEL_DIR,
)


class QwenModelUnavailable(RuntimeError):
    """Raised when the configured local model snapshot is incomplete."""


class QwenConfigurationError(ValueError):
    """Raised when the selected Qwen backend cannot be configured safely."""


class QwenBackend(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


def configured_backend(
    environ: Optional[Mapping[str, str]] = None,
) -> QwenBackend:
    """Return the Qwen adapter selected for this process."""
    env = os.environ if environ is None else environ
    raw = str(env.get("MARC_QWEN_BACKEND") or QwenBackend.LOCAL.value).strip().lower()
    try:
        return QwenBackend(raw)
    except ValueError as exc:
        choices = ", ".join(item.value for item in QwenBackend)
        raise QwenConfigurationError(
            f"Unsupported MARC_QWEN_BACKEND {raw!r}; choose {choices}."
        ) from exc


def configured_max_crops(
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    """Bound per-person crops while annotated views still cover every candidate."""
    env = os.environ if environ is None else environ
    try:
        value = int(env.get("MARC_QWEN_MAX_CROPS") or 12)
    except (TypeError, ValueError) as exc:
        raise QwenConfigurationError("MARC_QWEN_MAX_CROPS must be an integer") from exc
    if value < 0:
        raise QwenConfigurationError("MARC_QWEN_MAX_CROPS must be non-negative")
    return value


@dataclass(frozen=True)
class QwenRemoteConfig:
    """Connection and payload limits for an OpenAI-compatible vLLM server."""

    base_url: str
    api_key: str
    model_name: str
    timeout_s: float
    max_new_tokens: int
    wall_timeout_s: float = 90.0

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    @classmethod
    def load(
        cls,
        contract_path: Path = STAGE1_QWEN_CONTRACT,
        *,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "QwenRemoteConfig":
        env = os.environ if environ is None else environ
        json.loads(contract_path.read_text(encoding="utf-8"))
        stream = str(env.get("STREAM") or "").strip().lower() == "true"
        base_url = str(
            "http://10.0.0.200:8004/v1"
            if stream
            else env.get("MARC_QWEN_BASE_URL") or "http://127.0.0.1:8004/v1"
        ).strip().rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise QwenConfigurationError(
                "MARC_QWEN_BASE_URL must be an http(s) URL such as "
                "http://192.168.1.20:8004/v1"
            )
        api_key = str(env.get("MARC_QWEN_API_KEY") or "").strip()
        if not api_key:
            raise QwenConfigurationError("MARC_QWEN_API_KEY is required in remote mode")
        model_name = str(env.get("MARC_QWEN_MODEL") or "qwen3-vl-8b").strip()
        if not model_name:
            raise QwenConfigurationError("MARC_QWEN_MODEL must not be empty")
        try:
            timeout_s = float(env.get("MARC_QWEN_TIMEOUT_S") or 90.0)
            wall_timeout_s = float(env.get("MARC_QWEN_WALL_TIMEOUT_S") or 90.0)
            max_new_tokens = int(env.get("MARC_QWEN_MAX_NEW_TOKENS") or 48)
        except (TypeError, ValueError) as exc:
            raise QwenConfigurationError(
                "Qwen timeout and token limit must be numeric"
            ) from exc
        if timeout_s <= 0:
            raise QwenConfigurationError("MARC_QWEN_TIMEOUT_S must be positive")
        if wall_timeout_s <= 0:
            raise QwenConfigurationError("MARC_QWEN_WALL_TIMEOUT_S must be positive")
        if max_new_tokens <= 0:
            raise QwenConfigurationError("MARC_QWEN_MAX_NEW_TOKENS must be positive")
        return cls(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout_s=timeout_s,
            max_new_tokens=max_new_tokens,
            wall_timeout_s=wall_timeout_s,
        )


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
                or inference.get("max_new_tokens", 48)
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


_REQUIRED_SNAPSHOT_FILES = ("config.json", "tokenizer_config.json")


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
    if not any(
        (cfg.model_dir / name).is_file()
        for name in ("processor_config.json", "preprocessor_config.json")
    ):
        missing.append("processor_config.json or preprocessor_config.json")

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


def _qwen_model_cls():
    """Qwen3-VL class for transformers 4.57.1 (AutoModelForMultimodalLM is v5)."""
    try:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    except ImportError:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration


def load_local_qwen(config: Optional[QwenRuntimeConfig] = None):
    """Load the processor and model from disk without any Hub fallback."""
    cfg = require_local_bundle(config)
    try:
        import torch
        from transformers import AutoProcessor

        model_cls = _qwen_model_cls()
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
    model = model_cls.from_pretrained(
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
