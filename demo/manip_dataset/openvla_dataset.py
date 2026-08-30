"""Canonical MARC-to-OpenVLA dataset contract and validation.

The converter, offline validator, and optional loader smoke tests all use this
module so the schema has one source of truth.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np


DATASET_NAME = "marc_pnp"
SCHEMA_VERSION = "marc_openvla_pick_v2"
TFDS_VERSION = "1.2.0"
ACTION_DIM = 7
MARC_ARM_JOINTS = 6
OPENVLA_JOINT_SLOTS = 7
PROPRIO_DIM = OPENVLA_JOINT_SLOTS + 1
GRIPPER_ACTION_MASK = [True] * 6 + [False]
DEFAULT_CAMERA = "base_left"
DEFAULT_IMAGE_KEY = "image_base_left"

_PLACE_LANGUAGE = re.compile(
    r"\b(?:place|placed|placing|put|puts|drop|drops|deliver|delivers|basket)\b",
    re.IGNORECASE,
)


class DatasetValidationError(ValueError):
    """Raised when an export does not satisfy the OpenVLA contract."""


@dataclass(frozen=True)
class ValidationReport:
    episodes: int
    transitions: int
    camera: str


def _text(value) -> str:
    """Read text from JSON values or zero-dimensional NumPy arrays."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return ""
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value or "").strip()


def build_openvla_state(q_arm: np.ndarray, q_grip: np.ndarray) -> np.ndarray:
    """Return OpenVLA JOINT state: seven joint slots plus gripper state.

    MARC has six arm joints, so joint slot seven is deliberately zero padded.
    This matches OpenVLA's ``StateEncoding.JOINT`` eight-value convention.
    """
    arm = np.asarray(q_arm, dtype=np.float32)
    grip = np.asarray(q_grip, dtype=np.float32)
    if arm.ndim != 2 or arm.shape[1] != MARC_ARM_JOINTS:
        raise DatasetValidationError(
            f"q_arm must have shape (N, {MARC_ARM_JOINTS}), got {arm.shape}"
        )
    if grip.ndim == 2 and grip.shape[1] == 1:
        grip = grip[:, 0]
    if grip.ndim != 1 or grip.shape[0] != arm.shape[0]:
        raise DatasetValidationError(
            f"q_grip must have shape ({arm.shape[0]},) or ({arm.shape[0]}, 1), "
            f"got {grip.shape}"
        )
    if not np.isfinite(arm).all() or not np.isfinite(grip).all():
        raise DatasetValidationError("proprioception contains non-finite values")
    padding = np.zeros((arm.shape[0], OPENVLA_JOINT_SLOTS - MARC_ARM_JOINTS), dtype=np.float32)
    return np.concatenate([arm, padding, grip.reshape(-1, 1)], axis=1)


def validate_pick_metadata(meta: Mapping) -> tuple[str, str, str]:
    """Validate that an episode label describes only the recorded pick phase."""
    task = _text(meta.get("task")).lower()
    instruction = _text(meta.get("language_instruction"))
    object_name = _text(meta.get("object_name"))
    if task != "pick":
        raise DatasetValidationError(f"task must be 'pick', got {task or '<empty>'!r}")
    if not object_name:
        raise DatasetValidationError("object_name is required")
    if not instruction:
        raise DatasetValidationError("language_instruction is required")
    if bool(meta.get("includes_place", False)):
        raise DatasetValidationError("includes_place must be false for pick-only actions")
    match = _PLACE_LANGUAGE.search(instruction)
    if match:
        raise DatasetValidationError(
            f"pick-only language may not mention {match.group(0)!r}"
        )
    normalized_instruction = instruction.lower().replace("_", " ")
    normalized_object = object_name.lower().replace("_", " ")
    if normalized_object not in normalized_instruction:
        raise DatasetValidationError(
            "language_instruction must name object_name "
            f"({object_name!r} not found in {instruction!r})"
        )
    return instruction, object_name, task


def _require_vector(block: Mapping, key: str, size: int, errors: list[str], prefix: str) -> None:
    raw = block.get(key)
    try:
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        errors.append(f"{prefix}.{key} is not numeric")
        return
    if arr.size != size:
        errors.append(f"{prefix}.{key} must have length {size}, got {arr.size}")
    elif not np.isfinite(arr).all():
        errors.append(f"{prefix}.{key} contains non-finite values")


def validate_export(
    export_dir: str,
    *,
    expected_camera: Optional[str] = DEFAULT_CAMERA,
) -> ValidationReport:
    """Validate an NPZ export, manifest, builder, and normalization statistics."""
    errors: list[str] = []
    manifest_path = os.path.join(export_dir, "manifest.json")
    stats_path = os.path.join(export_dir, "dataset_statistics.json")
    builder_path = os.path.join(export_dir, "marc_pnp_dataset_builder.py")
    for path in (manifest_path, stats_path, builder_path):
        if not os.path.isfile(path):
            errors.append(f"missing {os.path.basename(path)}")
    if errors:
        raise DatasetValidationError("; ".join(errors))

    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(stats_path, encoding="utf-8") as handle:
        stats = json.load(handle)
    with open(builder_path, encoding="utf-8") as handle:
        builder_source = handle.read()
    try:
        image_size = int(stats.get("image_size", -1))
    except (TypeError, ValueError):
        image_size = -1
    if image_size <= 0:
        errors.append("statistics image_size must be a positive integer")

    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"manifest.schema_version must be {SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("action_dim") != ACTION_DIM:
        errors.append(f"manifest.action_dim must be {ACTION_DIM}")
    if manifest.get("state_dim") != PROPRIO_DIM:
        errors.append(f"manifest.state_dim must be {PROPRIO_DIM}")
    if manifest.get("tfds_version") != TFDS_VERSION:
        errors.append(f"manifest.tfds_version must be {TFDS_VERSION!r}")
    builder_checks = {
        f'Version("{TFDS_VERSION}")': "TFDS builder version",
        f'shape=({PROPRIO_DIM},), dtype=np.float32': "TFDS state feature",
        f'shape=({ACTION_DIM},), dtype=np.float32': "TFDS action feature",
    }
    for snippet, label in builder_checks.items():
        if snippet not in builder_source:
            errors.append(f"{label} does not match the manifest contract")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("manifest.episodes must be a non-empty list")
        episodes = []

    expected_image_key = None
    if expected_camera:
        expected_image_key = expected_camera
        if not expected_image_key.startswith("image_"):
            expected_image_key = f"image_{expected_image_key}"

    transition_count = 0
    manifest_paths: set[str] = set()
    for position, episode in enumerate(episodes):
        prefix = f"episode[{position}]"
        relative_path = _text(episode.get("path"))
        if not relative_path:
            errors.append(f"{prefix}.path is required")
            continue
        manifest_paths.add(relative_path)
        path = os.path.join(export_dir, relative_path)
        if not os.path.isfile(path):
            errors.append(f"{prefix} missing file {relative_path}")
            continue
        camera = _text(episode.get("camera"))
        if expected_image_key and camera != expected_image_key:
            errors.append(
                f"{prefix}.camera must be {expected_image_key!r}, got {camera!r}"
            )
        try:
            validate_pick_metadata(episode)
        except DatasetValidationError as exc:
            errors.append(f"{prefix}: {exc}")
        if episode.get("includes_place") is not False:
            errors.append(f"{prefix}.includes_place must be explicitly false")

        try:
            with np.load(path, allow_pickle=True) as data:
                required = {
                    "image", "state", "action", "ee_pos", "ee_quat",
                    "language_instruction", "object_name", "task", "includes_place",
                }
                missing = sorted(required.difference(data.files))
                if missing:
                    errors.append(f"{prefix} missing arrays: {', '.join(missing)}")
                    continue
                image = np.asarray(data["image"])
                state = np.asarray(data["state"])
                action = np.asarray(data["action"])
                count = int(action.shape[0]) if action.ndim else 0
                transition_count += count
                if image.ndim != 4 or image.shape[-1] != 3 or image.shape[0] != count:
                    errors.append(f"{prefix}.image must have shape (N, H, W, 3), got {image.shape}")
                elif image.dtype != np.uint8:
                    errors.append(f"{prefix}.image must be uint8, got {image.dtype}")
                elif image_size > 0 and image.shape[1:3] != (image_size, image_size):
                    errors.append(
                        f"{prefix}.image must be {image_size}x{image_size}, got {image.shape[1:3]}"
                    )
                if count <= 0:
                    errors.append(f"{prefix} has no transitions")
                if action.shape != (count, ACTION_DIM):
                    errors.append(f"{prefix}.action must have shape (N, {ACTION_DIM}), got {action.shape}")
                elif not np.isfinite(action).all():
                    errors.append(f"{prefix}.action contains non-finite values")
                elif not np.all(np.isclose(action[:, -1], 0.0) | np.isclose(action[:, -1], 1.0)):
                    errors.append(f"{prefix}.action gripper values must be binary")
                if state.shape != (count, PROPRIO_DIM):
                    errors.append(f"{prefix}.state must have shape (N, {PROPRIO_DIM}), got {state.shape}")
                elif not np.isfinite(state).all():
                    errors.append(f"{prefix}.state contains non-finite values")
                elif not np.allclose(state[:, MARC_ARM_JOINTS], 0.0):
                    errors.append(f"{prefix}.state joint padding slot must be zero")
                ee_pos = np.asarray(data["ee_pos"])
                ee_quat = np.asarray(data["ee_quat"])
                if ee_pos.shape != (count, 3) or not np.isfinite(ee_pos).all():
                    errors.append(f"{prefix}.ee_pos must be finite with shape (N, 3)")
                if ee_quat.shape != (count, 4) or not np.isfinite(ee_quat).all():
                    errors.append(f"{prefix}.ee_quat must be finite with shape (N, 4)")
                if int(episode.get("num_steps", -1)) != count:
                    errors.append(f"{prefix}.num_steps does not match its NPZ")
                npz_meta = {
                    "language_instruction": data["language_instruction"],
                    "object_name": data["object_name"],
                    "task": data["task"],
                    "includes_place": data["includes_place"].item(),
                }
                try:
                    instruction, object_name, task = validate_pick_metadata(npz_meta)
                    if instruction != _text(episode.get("language_instruction")):
                        errors.append(f"{prefix} language differs between manifest and NPZ")
                    if object_name != _text(episode.get("object_name")):
                        errors.append(f"{prefix} object_name differs between manifest and NPZ")
                    if task != _text(episode.get("task")).lower():
                        errors.append(f"{prefix} task differs between manifest and NPZ")
                except DatasetValidationError as exc:
                    errors.append(f"{prefix} NPZ metadata: {exc}")
        except (OSError, ValueError) as exc:
            errors.append(f"{prefix} cannot be read: {exc}")

    disk_paths = {
        os.path.relpath(path, export_dir)
        for path in glob.glob(os.path.join(export_dir, "episodes", "episode_*.npz"))
    }
    if disk_paths != manifest_paths:
        errors.append("manifest episode paths do not exactly match exported NPZ files")

    block = stats.get(DATASET_NAME, {})
    action_block = block.get("action", {}) if isinstance(block, Mapping) else {}
    proprio_block = block.get("proprio", {}) if isinstance(block, Mapping) else {}
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        _require_vector(action_block, key, ACTION_DIM, errors, f"{DATASET_NAME}.action")
        _require_vector(proprio_block, key, PROPRIO_DIM, errors, f"{DATASET_NAME}.proprio")
    if action_block.get("mask") != GRIPPER_ACTION_MASK:
        errors.append(f"{DATASET_NAME}.action.mask must be {GRIPPER_ACTION_MASK}")
    if proprio_block.get("mask") != [True] * PROPRIO_DIM:
        errors.append(f"{DATASET_NAME}.proprio.mask must contain {PROPRIO_DIM} true values")
    if stats.get("action_dim") != ACTION_DIM:
        errors.append(f"statistics action_dim must be {ACTION_DIM}")
    if int(stats.get("num_episodes", -1)) != len(episodes):
        errors.append("statistics num_episodes does not match manifest")
    if int(stats.get("num_steps", -1)) != transition_count:
        errors.append("statistics num_steps does not match exported transitions")
    if int(block.get("num_trajectories", -1)) != len(episodes):
        errors.append(f"{DATASET_NAME}.num_trajectories does not match manifest")
    if int(block.get("num_transitions", -1)) != transition_count:
        errors.append(f"{DATASET_NAME}.num_transitions does not match exported transitions")
    if expected_camera and stats.get("camera") != expected_camera:
        errors.append(
            f"statistics camera must be {expected_camera!r}, got {stats.get('camera')!r}"
        )

    if errors:
        raise DatasetValidationError("\n- " + "\n- ".join(errors))
    return ValidationReport(len(episodes), transition_count, expected_camera or "mixed")


def validate_tfds(tfds_root: str) -> ValidationReport:
    """Load one materialized sample through TensorFlow Datasets."""
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise DatasetValidationError(
            "tensorflow_datasets is required for the TFDS smoke test"
        ) from exc
    builder = tfds.builder(DATASET_NAME, data_dir=tfds_root)
    info = builder.info
    splits = set(info.splits)
    if "train" not in splits:
        raise DatasetValidationError("materialized TFDS dataset has no train split")
    sample = next(iter(tfds.as_numpy(builder.as_dataset(split="train").take(1))))
    steps = sample["steps"]
    action = np.asarray(steps["action"])
    state = np.asarray(steps["observation"]["state"])
    image = np.asarray(steps["observation"]["image"])
    if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise DatasetValidationError(f"TFDS action shape is {action.shape}, expected (N, {ACTION_DIM})")
    if state.ndim != 2 or state.shape[-1] != PROPRIO_DIM:
        raise DatasetValidationError(f"TFDS state shape is {state.shape}, expected (N, {PROPRIO_DIM})")
    if image.ndim != 4 or image.shape[-1] != 3:
        raise DatasetValidationError(f"TFDS image shape is invalid: {image.shape}")
    transitions = int(sum(split.num_examples for split in info.splits.values()))
    return ValidationReport(episodes=transitions, transitions=int(action.shape[0]), camera=DEFAULT_CAMERA)


def validate_openvla_loader(openvla_dir: str, tfds_root: str) -> None:
    """Read one sample through OpenVLA's official RLDSDataset pipeline."""
    checkout = os.path.abspath(openvla_dir)
    if not os.path.isdir(os.path.join(checkout, "prismatic")):
        raise DatasetValidationError(f"not an OpenVLA checkout: {checkout}")
    sys.path.insert(0, checkout)
    try:
        from prismatic.vla.datasets import RLDSDataset
        from prismatic.vla.datasets.rlds.oxe.configs import (
            ActionEncoding,
            OXE_DATASET_CONFIGS,
            StateEncoding,
        )
        from prismatic.vla.datasets.rlds.oxe.transforms import OXE_STANDARDIZATION_TRANSFORMS
    except ImportError as exc:  # pragma: no cover - depends on OpenVLA environment
        raise DatasetValidationError(
            "OpenVLA and its Python dependencies must be installed for the loader smoke test"
        ) from exc

    OXE_DATASET_CONFIGS[DATASET_NAME] = {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    }
    OXE_STANDARDIZATION_TRANSFORMS[DATASET_NAME] = lambda trajectory: trajectory
    dataset = RLDSDataset(
        Path(tfds_root),
        DATASET_NAME,
        batch_transform=lambda sample: sample,
        resize_resolution=(224, 224),
        shuffle_buffer_size=32,
        train=True,
        image_aug=False,
    )
    sample = next(iter(dataset))
    action = np.asarray(sample["action"])
    image = np.asarray(sample["observation"]["image_primary"])
    language = sample.get("task", {}).get("language_instruction")
    if action.shape[-1] != ACTION_DIM:
        raise DatasetValidationError(f"OpenVLA loader produced action shape {action.shape}")
    if image.shape[-1] != 3:
        raise DatasetValidationError(f"OpenVLA loader produced image shape {image.shape}")
    if language is None or np.asarray(language).size == 0:
        raise DatasetValidationError("OpenVLA loader produced no language instruction")
