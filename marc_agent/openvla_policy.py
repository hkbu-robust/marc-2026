"""OpenVLA-oriented grasp policy: protocol, geometric stub, HuggingFace load.

The graded ``agent`` image already ships torch + transformers. Real inference is
a lazy optional load of a local HF snapshot under ``models/runtime/openvla`` (or
``MARC_OPENVLA_WEIGHTS``). Missing weights, stats, or a failed import returns
``None`` so the caller falls back to IK. Do not download at runtime.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

import numpy as np

from marc_agent.openvla_actions import (
    ACTION_DIM,
    DEFAULT_INSTRUCTION,
    DEFAULT_UNNORM_KEY,
    DEFAULT_VOCAB_SIZE,
    IMAGE_SIZE,
    ActionStats,
    load_action_stats,
    openvla_prompt,
    resize_rgb,
    resolve_stats_path,
    tokens_to_action,
)

log = logging.getLogger("participant_app")

DEFAULT_STEP_M = 0.02
DEFAULT_CLOSE_M = 0.04
DEFAULT_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "runtime", "openvla",
)

_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "model.pt",
    "openvla.pt",
)

DEFAULT_REMOTE_BASE_URL = "http://10.0.0.200:8003/v1"
DEFAULT_REMOTE_API_KEY = "marc-openvla-secret"
DEFAULT_REMOTE_MODEL = "openvla-7b"
BRIDGE_ORIG_Q01 = (
    -0.02872725307941437, -0.04170349963009357, -0.026093858778476715,
    -0.08092105075716972, -0.09288699507713317, -0.20718276381492615, 0.0,
)
BRIDGE_ORIG_Q99 = (
    0.028309678435325586, 0.040855254605412394, 0.040161586627364146,
    0.08192047759890528, 0.07792850524187081, 0.20382574498653397, 1.0,
)


@dataclass
class PolicyObservation:
    """One policy tick: third-person RGB, arm proprio, optional EE pose."""

    rgb: Optional[np.ndarray] = None
    proprio: Optional[np.ndarray] = None  # q(6) + grip
    ee_xyz: Optional[np.ndarray] = None
    ee_quat: Optional[np.ndarray] = None  # wxyz
    instruction: str = DEFAULT_INSTRUCTION


class GraspPolicy(Protocol):
    def act(self, obs: PolicyObservation) -> np.ndarray:
        """Return OpenVLA-style action ``(dx, dy, dz, droll, dpitch, dyaw, grip)``."""


class GeometricGraspStub:
    """No-weights stand-in: EE deltas toward an arm_base target, then close.

    Used with ``MARC_PICK_POLICY=openvla_geom`` to exercise the action executor
    without a model. Not the scoring default.
    """

    def __init__(self, target_xyz, step_m: float = DEFAULT_STEP_M,
                 close_m: float = DEFAULT_CLOSE_M):
        self.target = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        self.step_m = float(step_m)
        self.close_m = float(close_m)

    def act(self, obs: PolicyObservation) -> np.ndarray:
        ee = np.zeros(3, dtype=np.float64)
        if obs is not None and obs.ee_xyz is not None:
            ee = np.asarray(obs.ee_xyz, dtype=np.float64).reshape(3)
        delta = self.target - ee
        dist = float(np.linalg.norm(delta))
        if dist > self.step_m and dist > 1e-9:
            delta = delta * (self.step_m / dist)
        grip = 1.0 if dist <= self.close_m else 0.0
        return np.array(
            [delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip],
            dtype=np.float32,
        )


def resolve_weights_path(path: Optional[str] = None) -> Optional[str]:
    raw = path or os.environ.get("MARC_OPENVLA_WEIGHTS") or DEFAULT_WEIGHTS_DIR
    raw = os.path.expanduser(str(raw))
    if not raw:
        return None
    if os.path.isfile(raw):
        return raw
    if os.path.isdir(raw):
        if os.path.isfile(os.path.join(raw, "config.json")):
            return raw
        for name in _WEIGHT_FILES:
            cand = os.path.join(raw, name)
            if os.path.isfile(cand):
                return raw if name.endswith(".index.json") else cand
    return None


def _is_lora_only(directory: str) -> bool:
    if not os.path.isfile(os.path.join(directory, "adapter_config.json")):
        return False
    has_full = any(
        os.path.isfile(os.path.join(directory, name)) for name in _WEIGHT_FILES
    )
    return not has_full


def _weights_dir(weights: str) -> str:
    return weights if os.path.isdir(weights) else os.path.dirname(os.path.abspath(weights))


class HuggingFaceOpenVLAPolicy:
    """Local ``AutoModelForVision2Seq`` wrapper. Never hits the Hub."""

    def __init__(self, model, processor, stats: ActionStats, *,
                 device: str = "cpu", dtype=None, vocab_size: int = DEFAULT_VOCAB_SIZE):
        self.model = model
        self.processor = processor
        self.stats = stats
        self.device = device
        self.dtype = dtype
        self.vocab_size = int(vocab_size)
        self.unnorm_key = stats.unnorm_key

    @classmethod
    def from_pretrained(cls, path: str, stats: ActionStats) -> "HuggingFaceOpenVLAPolicy":
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        load_dir = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(
            load_dir, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            load_dir, trust_remote_code=True, local_files_only=True, torch_dtype=dtype,
        )
        model = model.to(device)
        model.eval()
        payload = {stats.unnorm_key: {"action": stats.to_norm_dict()}}
        existing = getattr(model, "norm_stats", None)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(payload)
            model.norm_stats = merged
        else:
            try:
                model.norm_stats = payload
            except Exception:
                pass
        vocab = int(
            getattr(model, "vocab_size", None)
            or getattr(getattr(model, "config", None), "vocab_size", None)
            or stats.vocab_size
            or DEFAULT_VOCAB_SIZE
        )
        return cls(
            model, processor, stats,
            device=device, dtype=dtype, vocab_size=vocab,
        )

    def act(self, obs: PolicyObservation) -> np.ndarray:
        instruction = DEFAULT_INSTRUCTION
        rgb = None
        if obs is not None:
            rgb = obs.rgb
            instruction = obs.instruction or DEFAULT_INSTRUCTION
        if rgb is None:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        import torch

        image = _to_pil(rgb)
        prompt = openvla_prompt(instruction)
        processed = self.processor(prompt, image)
        if hasattr(processed, "to"):
            kwargs = {}
            if self.dtype is not None:
                kwargs["dtype"] = self.dtype
            processed = processed.to(self.device, **kwargs)
        elif isinstance(processed, dict):
            processed = {
                key: (
                    value.to(self.device, dtype=self.dtype)
                    if hasattr(value, "to") and value.is_floating_point()
                    else value.to(self.device) if hasattr(value, "to") else value
                )
                for key, value in processed.items()
            }
        predict = getattr(self.model, "predict_action", None)
        if callable(predict):
            with torch.inference_mode():
                action = predict(
                    **processed, unnorm_key=self.unnorm_key, do_sample=False,
                )
            return np.asarray(action, dtype=np.float32).reshape(-1)[:ACTION_DIM]
        generate = getattr(self.model, "generate", None)
        if not callable(generate):
            raise RuntimeError("OpenVLA model has neither predict_action nor generate")
        with torch.inference_mode():
            generated = generate(
                **processed, max_new_tokens=self.stats.dim, do_sample=False,
            )
        token_ids = generated[0, -self.stats.dim:].detach().cpu().numpy()
        action = tokens_to_action(token_ids, self.stats, vocab_size=self.vocab_size)
        return np.asarray(action, dtype=np.float32).reshape(-1)


class RemoteOpenVLAPolicy:
    """OpenVLA action-token client for an OpenAI-compatible vLLM server."""

    def __init__(self, base_url: str, api_key: str, model: str, *,
                 timeout_s: float = 30.0, post=None):
        parsed = urlparse(str(base_url))
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("MARC_OPENVLA_BASE_URL must be an http(s) URL")
        if float(timeout_s) <= 0:
            raise ValueError("MARC_OPENVLA_TIMEOUT_S must be positive")
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.model = str(model)
        self.timeout_s = float(timeout_s)
        self._post = post or self._url_post
        self.stats = ActionStats(
            q01=BRIDGE_ORIG_Q01,
            q99=BRIDGE_ORIG_Q99,
            mask=(True, True, True, True, True, True, False),
            unnorm_key="bridge_orig",
        )

    @classmethod
    def from_env(cls, environ: Optional[Mapping[str, str]] = None, *, post=None):
        env = os.environ if environ is None else environ
        try:
            timeout_s = float(env.get("MARC_OPENVLA_TIMEOUT_S", "30"))
        except (TypeError, ValueError) as exc:
            raise ValueError("MARC_OPENVLA_TIMEOUT_S must be a number") from exc
        return cls(
            env.get("MARC_OPENVLA_BASE_URL", DEFAULT_REMOTE_BASE_URL),
            env.get("MARC_OPENVLA_API_KEY", DEFAULT_REMOTE_API_KEY),
            env.get("MARC_OPENVLA_MODEL", DEFAULT_REMOTE_MODEL),
            timeout_s=timeout_s,
            post=post,
        )

    @staticmethod
    def _url_post(url: str, body: bytes, headers: Mapping[str, str], timeout_s: float):
        request = Request(url, data=body, headers=dict(headers), method="POST")
        # The simulator and policy server share a LAN; ambient HTTP proxies must
        # not intercept private-address image streams.
        with build_opener(ProxyHandler({})).open(request, timeout=timeout_s) as response:
            return response.read()

    def act(self, obs: PolicyObservation) -> np.ndarray:
        if obs is None or obs.rgb is None:
            return np.zeros(ACTION_DIM, dtype=np.float32)
        buffer = io.BytesIO()
        _to_pil(obs.rgb).save(buffer, format="JPEG", quality=85)
        image_url = "data:image/jpeg;base64," + base64.b64encode(
            buffer.getvalue()
        ).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": obs.instruction or DEFAULT_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "max_tokens": ACTION_DIM,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        }
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        raw = self._post(
            self.base_url + "/chat/completions",
            body,
            {
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            self.timeout_s,
        )
        try:
            response = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            token_ids = response["choices"][0]["token_ids"][-ACTION_DIM:]
            if len(token_ids) != ACTION_DIM or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or not self.stats.vocab_size - self.stats.n_bins <= token < self.stats.vocab_size
                for token in token_ids
            ):
                raise ValueError("invalid action token ids")
        except (AttributeError, IndexError, KeyError, TypeError, ValueError,
                UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OpenVLA response did not contain seven action token ids") from exc
        return tokens_to_action(
            token_ids, self.stats, vocab_size=self.stats.vocab_size,
        ).astype(np.float32)


def _to_pil(rgb: np.ndarray):
    arr = resize_rgb(np.asarray(rgb), IMAGE_SIZE)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        max_v = float(np.nanmax(arr)) if arr.size else 0.0
        if max_v <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(arr[..., :3])


def load_openvla(path: Optional[str] = None) -> Optional[GraspPolicy]:
    """Load an OpenVLA policy if a local snapshot and stats exist; else ``None``."""
    weights = resolve_weights_path(path)
    if weights is None:
        log.info(
            "[openvla] no weights at %s -- IK fallback",
            path or os.environ.get("MARC_OPENVLA_WEIGHTS") or DEFAULT_WEIGHTS_DIR,
        )
        return None
    directory = _weights_dir(weights)
    if _is_lora_only(directory):
        log.warning("[openvla] LoRA adapter at %s without merged base -- IK fallback", directory)
        return None
    stats_path = resolve_stats_path(directory, default_dir=DEFAULT_WEIGHTS_DIR)
    if stats_path is None:
        log.warning("[openvla] no dataset_statistics.json for %s -- IK fallback", weights)
        return None
    unnorm_key = (
        os.environ.get("MARC_OPENVLA_UNNORM_KEY") or DEFAULT_UNNORM_KEY
    ).strip() or DEFAULT_UNNORM_KEY
    try:
        stats = load_action_stats(stats_path, unnorm_key=unnorm_key)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("[openvla] stats at %s unusable (%s) -- IK fallback", stats_path, exc)
        return None
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("[openvla] weights at %s but torch is unavailable -- IK fallback", weights)
        return None
    try:
        import transformers  # noqa: F401
    except ImportError:
        log.warning(
            "[openvla] weights at %s but transformers is unavailable -- IK fallback", weights,
        )
        return None
    try:
        return HuggingFaceOpenVLAPolicy.from_pretrained(weights, stats)
    except Exception as exc:
        log.warning("[openvla] load failed from %s (%s) -- IK fallback", weights, exc)
        return None
