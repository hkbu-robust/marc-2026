#!/usr/bin/env python3
"""Convert successful marc_pnp HDF5 episodes to an OpenVLA-oriented export.

Writes:
  <out>/episodes/episode_XXXX.npz   — arrays per episode
  <out>/dataset_statistics.json     — action mean/std
  <out>/manifest.json               — episode index + language

Does not require tensorflow_datasets at convert time (NumPy export). Point an
OXE/RLDS builder at these files, or load npz directly in a custom OpenVLA
dataset transform.

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
from typing import List, Tuple

import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise SystemExit("h5py required: pip install h5py") from e


def _resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbor resize without OpenCV/Pillow dependency."""
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    # center-crop to square then NN resize
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    # NN scale
    yy = (np.linspace(0, side - 1, size)).astype(np.int32)
    xx = (np.linspace(0, side - 1, size)).astype(np.int32)
    return np.ascontiguousarray(crop[yy][:, xx])


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


def convert(in_dir: str, out_dir: str, image_size: int = 224, camera: str = "auto"):
    os.makedirs(os.path.join(out_dir, "episodes"), exist_ok=True)
    episodes = []
    all_actions: List[np.ndarray] = []

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
        )
        episodes.append({
            "index": idx,
            "source": name,
            "path": f"episodes/episode_{idx:04d}.npz",
            "num_steps": int(data["action"].shape[0]),
            "language_instruction": meta.get("language_instruction", ""),
            "camera": data["image_key"],
        })
        all_actions.append(data["action"])
        print(f"[ok] {name} -> episode_{idx:04d}.npz "
              f"({data['action'].shape[0]} steps, {data['image_key']})")
        idx += 1

    if not all_actions:
        raise SystemExit(f"no successful episodes found under {in_dir}")

    actions = np.concatenate(all_actions, axis=0)
    stats = {
        "action_dim": 7,
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).clip(min=1e-6).tolist(),
        "num_episodes": len(episodes),
        "num_steps": int(actions.shape[0]),
        "image_size": image_size,
        "action_convention": "ee_delta_arm_base_dx_dy_dz_droll_dpitch_dyaw_grip",
        "camera": camera,
    }
    with open(os.path.join(out_dir, "dataset_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"episodes": episodes}, f, indent=2)
    print(f"[done] {len(episodes)} episodes -> {out_dir}")
    print(f"[stats] action_mean={stats['action_mean']}")
    print(f"[stats] action_std={stats['action_std']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True, help="marc_pnp success root")
    ap.add_argument("--out", dest="out_dir", required=True, help="export root")
    ap.add_argument("--image-size", type=int, default=224)
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
