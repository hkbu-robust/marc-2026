"""OpenVLA discrete-token codec: bins, unnormalize, quaternion apply, gripper.

Matches openvla/openvla ``ActionTokenizer`` and HuggingFace ``predict_action``
``bounds_q99`` unnormalize. No torch dependency.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np

ACTION_DIM = 7
N_BINS = 256
MIN_ACTION = -1.0
MAX_ACTION = 1.0
DEFAULT_VOCAB_SIZE = 32000  # Llama; last 256 tokens are action bins
DEFAULT_UNNORM_KEY = "marc_pnp"
DEFAULT_INSTRUCTION = "pick up the object"
IMAGE_SIZE = 224
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.703
ACTION_CONVENTION = "ee_delta_arm_base_dx_dy_dz_droll_dpitch_dyaw_grip_0_closed_1_open"
# EE dims use q01/q99; canonical binary gripper stays unnormalized (mask=false).
DEFAULT_ACTION_MASK = (True, True, True, True, True, True, False)


def action_bins(n_bins: int = N_BINS) -> np.ndarray:
    return np.linspace(MIN_ACTION, MAX_ACTION, int(n_bins))


def action_bin_centers(n_bins: int = N_BINS) -> np.ndarray:
    edges = action_bins(n_bins)
    return (edges[:-1] + edges[1:]) / 2.0


def openvla_prompt(instruction: Optional[str] = None) -> str:
    text = str(instruction or "").strip() or DEFAULT_INSTRUCTION
    return f"In: What action should the robot take to {text}?\nOut:"


def pick_instruction(target) -> str:
    """Language string matching the manip collector: ``pick up the {name}``."""
    if target is None:
        return DEFAULT_INSTRUCTION
    if isinstance(target, str):
        name = target.strip()
        return f"pick up the {name}" if name else DEFAULT_INSTRUCTION
    for attr in ("language_instruction", "instruction"):
        raw = getattr(target, attr, None)
        if raw:
            return str(raw).strip()
    name = getattr(target, "name", None) or getattr(target, "target_type", None)
    if name:
        return f"pick up the {name}"
    return DEFAULT_INSTRUCTION


def resize_rgb(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Center-crop to square then nearest-neighbor resize (matches to_rlds)."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w = arr.shape[:2]
    if h == size and w == size:
        return np.ascontiguousarray(arr)
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = arr[y0:y0 + side, x0:x0 + side]
    yy = np.linspace(0, side - 1, size).astype(np.int32)
    xx = np.linspace(0, side - 1, size).astype(np.int32)
    return np.ascontiguousarray(crop[yy][:, xx])


@dataclass
class ActionStats:
    """Per-dataset bounds used by OpenVLA ``unnorm_key``."""

    q01: np.ndarray
    q99: np.ndarray
    mask: np.ndarray
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    n_bins: int = N_BINS
    unnorm_key: str = DEFAULT_UNNORM_KEY
    vocab_size: int = DEFAULT_VOCAB_SIZE

    def __post_init__(self) -> None:
        self.q01 = np.asarray(self.q01, dtype=np.float64).reshape(-1)
        self.q99 = np.asarray(self.q99, dtype=np.float64).reshape(-1)
        self.mask = np.asarray(self.mask, dtype=bool).reshape(-1)
        dim = int(self.q01.size)
        if self.q99.size != dim or self.mask.size != dim:
            raise ValueError("q01, q99, and mask must share one action dimension")
        for name in ("min", "max", "mean", "std"):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, np.asarray(val, dtype=np.float64).reshape(-1))

    @property
    def dim(self) -> int:
        return int(self.q01.size)

    def to_norm_dict(self) -> dict:
        payload = {
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "mask": [bool(x) for x in self.mask],
        }
        for name in ("min", "max", "mean", "std"):
            val = getattr(self, name)
            if val is not None:
                payload[name] = val.tolist()
        return payload


def _as_float_vec(raw: Any, dim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if dim is not None and arr.size != dim:
        raise ValueError(f"expected length {dim}, got {arr.size}")
    return arr


def _axis_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).clip(min=1e-6).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }


def build_dataset_statistics(
    actions: np.ndarray,
    proprio: Optional[np.ndarray] = None,
    *,
    unnorm_key: str = DEFAULT_UNNORM_KEY,
    image_size: int = IMAGE_SIZE,
    camera: str = "auto",
    num_episodes: int = 1,
    n_bins: int = N_BINS,
) -> dict:
    """Nested OpenVLA ``dataset_statistics.json`` payload."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    action_block = _axis_stats(actions)
    mask = list(DEFAULT_ACTION_MASK)
    if actions.shape[1] != ACTION_DIM:
        mask = [True] * int(actions.shape[1])
        if mask:
            mask[-1] = False
    action_block["mask"] = mask
    nested = {
        "action": action_block,
        "num_transitions": int(actions.shape[0]),
        "num_trajectories": int(num_episodes),
    }
    if proprio is not None:
        proprio_arr = np.asarray(proprio, dtype=np.float64)
        if proprio_arr.ndim == 1:
            proprio_arr = proprio_arr.reshape(1, -1)
        nested["proprio"] = _axis_stats(proprio_arr)
        nested["proprio"]["mask"] = [True] * int(proprio_arr.shape[1])
    return {
        unnorm_key: nested,
        "action_dim": int(actions.shape[1]),
        "n_bins": int(n_bins),
        "unnorm_key": unnorm_key,
        "num_episodes": int(num_episodes),
        "num_steps": int(actions.shape[0]),
        "image_size": int(image_size),
        "action_convention": ACTION_CONVENTION,
        "camera": camera,
    }


def load_action_stats(
    path: Union[str, Mapping[str, Any]],
    unnorm_key: str = DEFAULT_UNNORM_KEY,
) -> ActionStats:
    """Load nested OpenVLA stats. Requires q01/q99; mean/std-only is rejected."""
    if isinstance(path, Mapping):
        data = dict(path)
    else:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    key = str(unnorm_key or data.get("unnorm_key") or DEFAULT_UNNORM_KEY)
    block = None
    if key in data and isinstance(data[key], dict) and "action" in data[key]:
        block = data[key]["action"]
    elif "action" in data and isinstance(data["action"], dict) and "q01" in data["action"]:
        block = data["action"]
        key = str(data.get("unnorm_key") or key)
    if not isinstance(block, dict) or "q01" not in block or "q99" not in block:
        raise ValueError(
            "dataset_statistics.json missing nested %s.action.q01/q99" % key
        )
    q01 = _as_float_vec(block["q01"])
    q99 = _as_float_vec(block["q99"], dim=int(q01.size))
    if "mask" in block:
        mask = np.asarray(block["mask"], dtype=bool).reshape(-1)
        if mask.size != q01.size:
            raise ValueError("mask length does not match q01")
    else:
        mask = np.ones(q01.size, dtype=bool)
        if q01.size:
            mask[-1] = False
    extra = {}
    for name in ("min", "max", "mean", "std"):
        if name in block:
            extra[name] = _as_float_vec(block[name], dim=int(q01.size))
    n_bins = int(data.get("n_bins") or N_BINS)
    vocab = int(data.get("vocab_size") or DEFAULT_VOCAB_SIZE)
    return ActionStats(
        q01=q01, q99=q99, mask=mask, n_bins=n_bins,
        unnorm_key=key, vocab_size=vocab, **extra,
    )


def decode_token_ids_to_normalized(
    token_ids: Sequence[int],
    *,
    n_bins: int = N_BINS,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> np.ndarray:
    """Map Llama action token ids to bin centers in [-1, 1]."""
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    centers = action_bin_centers(n_bins)
    discretized = int(vocab_size) - ids
    idx = np.clip(discretized - 1, a_min=0, a_max=centers.shape[0] - 1)
    return centers[idx]


def unnormalize_action(normalized: np.ndarray, stats: ActionStats) -> np.ndarray:
    """``bounds_q99``: masked dims use q01/q99; unmasked (gripper) stay in [-1, 1]."""
    x = np.asarray(normalized, dtype=np.float64).reshape(-1)
    if x.size != stats.dim:
        raise ValueError(f"action dim {x.size} != stats dim {stats.dim}")
    low, high, mask = stats.q01, stats.q99, stats.mask
    scaled = 0.5 * (x + 1.0) * (high - low) + low
    return np.where(mask, scaled, x).astype(np.float64)


def normalize_action(action: np.ndarray, stats: ActionStats) -> np.ndarray:
    x = np.asarray(action, dtype=np.float64).reshape(-1)
    if x.size != stats.dim:
        raise ValueError(f"action dim {x.size} != stats dim {stats.dim}")
    low, high, mask = stats.q01, stats.q99, stats.mask
    span = high - low
    span = np.where(np.abs(span) < 1e-12, 1.0, span)
    scaled = np.clip(2.0 * (x - low) / span - 1.0, MIN_ACTION, MAX_ACTION)
    return np.where(mask, scaled, np.clip(x, MIN_ACTION, MAX_ACTION)).astype(np.float64)


def action_to_token_ids(
    action: np.ndarray,
    stats: ActionStats,
    *,
    vocab_size: Optional[int] = None,
) -> np.ndarray:
    vocab = int(vocab_size if vocab_size is not None else stats.vocab_size)
    normalized = np.clip(normalize_action(action, stats), MIN_ACTION, MAX_ACTION)
    discretized = np.digitize(normalized, action_bins(stats.n_bins))
    return (vocab - discretized).astype(np.int64)


def tokens_to_action(
    token_ids: Sequence[int],
    stats: ActionStats,
    *,
    vocab_size: Optional[int] = None,
) -> np.ndarray:
    vocab = int(vocab_size if vocab_size is not None else stats.vocab_size)
    normalized = decode_token_ids_to_normalized(
        token_ids, n_bins=stats.n_bins, vocab_size=vocab,
    )
    return unnormalize_action(normalized, stats)


def gripper_should_close(grip: float, *, open_is_high: bool = True) -> bool:
    """Interpret canonical OpenVLA ``0=closed, 1=open`` (or an explicit stub)."""
    value = float(grip)
    return value < 0.5 if open_is_high else value >= 0.5


def gripper_command(
    grip: float,
    *,
    open_is_high: bool = True,
    close: float = GRIPPER_CLOSE,
    open_: float = GRIPPER_OPEN,
) -> float:
    if gripper_should_close(grip, open_is_high=open_is_high):
        return float(close)
    return float(open_)


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def rotvec_to_quat(rv) -> np.ndarray:
    vec = np.asarray(rv, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vec))
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = vec / angle
    half = angle / 2.0
    s = np.sin(half)
    return np.array(
        [np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s],
        dtype=np.float64,
    )


def compose_ee_quat(q, droll: float, dpitch: float, dyaw: float) -> np.ndarray:
    """Apply OpenVLA rpy/axis-angle delta to a wxyz EE quaternion."""
    return quat_mul(q, rotvec_to_quat([droll, dpitch, dyaw]))


def resolve_stats_path(
    weights: Optional[str] = None,
    default_dir: Optional[str] = None,
) -> Optional[str]:
    """``dataset_statistics.json`` next to weights, else under ``default_dir``."""
    candidates = []
    if weights:
        base = weights if os.path.isdir(weights) else os.path.dirname(weights)
        if base:
            candidates.append(os.path.join(base, "dataset_statistics.json"))
    if default_dir:
        candidates.append(os.path.join(default_dir, "dataset_statistics.json"))
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None
