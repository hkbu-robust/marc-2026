#!/usr/bin/env python3
"""Convert successful marc_pnp HDF5 episodes to an OpenVLA RLDS source.

Writes:
  <out>/episodes/episode_XXXX.npz   — arrays per episode
  <out>/dataset_statistics.json     — nested marc_pnp q01/q99/mask
  <out>/manifest.json               — episode index + language
  <out>/marc_pnp_dataset_builder.py — materializes the source as TFDS/RLDS

Conversion does not require tensorflow_datasets. Run the emitted builder from
an OpenVLA environment to materialize TFDS/RLDS. LoRA fine-tuning should use
unnorm_key=marc_pnp.

Usage:
  python3 -m manip_dataset.to_rlds \\
    --in demo/datasets/marc_pnp \\
    --out demo/datasets/marc_pnp_rlds \\
    --image-size 224
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import List, Tuple

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marc_agent.openvla_actions import (  # noqa: E402
    DEFAULT_UNNORM_KEY,
    IMAGE_SIZE,
    build_dataset_statistics,
    resize_rgb,
)
try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise SystemExit("h5py required: pip install h5py") from e


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize without OpenCV/Pillow dependency."""
    return resize_rgb(img, size)


CAMERA_CHOICES = ("auto", "base_left", "base_right", "gripper_left", "gripper_right")
# Episodes recorded before the stereo split stored the left eye under a short name.
_LEGACY = {"image_base_left": "image_base", "image_gripper_left": "image_gripper"}
_PREFERENCE = ("image_base_left", "image_gripper_left", "image_base_right",
               "image_gripper_right")


def _pick_image_key(steps, camera: str) -> str:
    """Resolve the requested view, falling back to whichever one was recorded."""
    preferred = str(steps.attrs.get("primary_camera", "base_left")) if camera == "auto" \
        else camera
    wanted = f"image_{preferred}"
    order = [wanted, _LEGACY.get(wanted)] + [k for k in _PREFERENCE if k != wanted]
    order += [_LEGACY[k] for k in _PREFERENCE if k in _LEGACY]
    for key in order:
        if key and key in steps:
            return key
    raise ValueError("episode has no RGB view")


def _load_episode(ep_dir: str, camera: str = "auto") -> Tuple[dict, dict]:
    meta_path = os.path.join(ep_dir, "meta.json")
    h5_path = os.path.join(ep_dir, "episode.hdf5")
    with open(meta_path) as f:
        meta = json.load(f)
    if not meta.get("success", False):
        raise ValueError("not a successful episode")
    with h5py.File(h5_path, "r") as f:
        steps = f["steps"]
        image_key = _pick_image_key(steps, camera)
        data = {
            "image_key": image_key,
            "image": np.asarray(steps[image_key]),
            "action": np.asarray(steps["action"], dtype=np.float32),
            "q_arm": np.asarray(steps["q_arm"], dtype=np.float32),
            "q_grip": np.asarray(steps["q_grip"], dtype=np.float32),
            "ee_pos": np.asarray(steps["ee_pos"], dtype=np.float32),
            "ee_quat": np.asarray(steps["ee_quat"], dtype=np.float32),
            "grasping": np.asarray(steps["grasping"]),
        }
    return meta, data


def convert(in_dir: str, out_dir: str, image_size: int = IMAGE_SIZE, camera: str = "auto"):
    os.makedirs(os.path.join(out_dir, "episodes"), exist_ok=True)
    episodes = []
    all_actions: List[np.ndarray] = []
    all_states: List[np.ndarray] = []

    subdirs = sorted(
        d for d in os.listdir(in_dir)
        if os.path.isdir(os.path.join(in_dir, d))
        and os.path.isfile(os.path.join(in_dir, d, "meta.json"))
        and os.path.isfile(os.path.join(in_dir, d, "episode.hdf5"))
    )

    idx = 0
    for name in subdirs:
        ep_dir = os.path.join(in_dir, name)
        try:
            with open(os.path.join(ep_dir, "meta.json")) as f:
                meta_peek = json.load(f)
            if not meta_peek.get("success", False):
                continue
            meta, data = _load_episode(ep_dir, camera)
        except Exception as e:
            print(f"[skip] {name}: {e}")
            continue

        images = np.stack(
            [_resize_rgb(data["image"][i], image_size) for i in range(len(data["action"]))],
            axis=0,
        )
        # proprio state for OpenVLA-style observation.state: q_arm(6) + grip
        state = np.concatenate(
            [data["q_arm"], data["q_grip"].reshape(-1, 1)], axis=1
        ).astype(np.float32)

        out_path = os.path.join(out_dir, "episodes", f"episode_{idx:04d}.npz")
        np.savez_compressed(
            out_path,
            image=images,
            state=state,
            action=data["action"],
            ee_pos=data["ee_pos"],
            ee_quat=data["ee_quat"],
            language_instruction=meta.get("language_instruction", ""),
            object_name=meta.get("object_name", ""),
            task=meta.get("task", "pick"),
        )
        episodes.append({
            "index": idx,
            "source": name,
            "path": f"episodes/episode_{idx:04d}.npz",
            "num_steps": int(data["action"].shape[0]),
            "language_instruction": meta.get("language_instruction", ""),
            "object_name": meta.get("object_name", ""),
            "task": meta.get("task", "pick"),
            "camera": data["image_key"],
        })
        all_actions.append(data["action"])
        all_states.append(state)
        print(f"[ok] {name} -> episode_{idx:04d}.npz "
              f"({data['action'].shape[0]} steps, {data['image_key']})")
        idx += 1

    if not all_actions:
        raise SystemExit(f"no successful episodes found under {in_dir}")

    actions = np.concatenate(all_actions, axis=0)
    proprio = np.concatenate(all_states, axis=0)
    stats = build_dataset_statistics(
        actions,
        proprio,
        unnorm_key=DEFAULT_UNNORM_KEY,
        image_size=image_size,
        camera=camera,
        num_episodes=len(episodes),
    )
    with open(os.path.join(out_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"episodes": episodes}, f, indent=2)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "marc_pnp_dataset_builder.py"),
        os.path.join(out_dir, "marc_pnp_dataset_builder.py"),
    )
    action_block = stats[DEFAULT_UNNORM_KEY]["action"]
    print(f"[done] {len(episodes)} episodes -> {out_dir}")
    print(f"[stats] action_q01={action_block['q01']}")
    print(f"[stats] action_q99={action_block['q99']}")
    print(f"[stats] action_mask={action_block['mask']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True, help="marc_pnp success root")
    ap.add_argument("--out", dest="out_dir", required=True, help="export root")
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument(
        "--camera",
        choices=CAMERA_CHOICES,
        default="auto",
        help="view exported as `image` (auto = each episode's recorded primary_camera)",
    )
    args = ap.parse_args()
    convert(args.in_dir, args.out_dir, args.image_size, args.camera)


if __name__ == "__main__":
    main()
