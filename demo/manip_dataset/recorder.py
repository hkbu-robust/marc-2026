"""In-memory episode buffer and HDF5 writer."""

from __future__ import annotations

import json
import os
import re
import struct
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover — ImportError or numpy ABI mismatch
    h5py = None


# Language must match the labeled motion: ARM_TELEOP grasp only. Place (`P`) is
# not written into the HDF5, so "…and put it in the basket" would be a lie.
INSTRUCTION_TEMPLATES = [
    "pick up the {obj}",
    "grasp the {obj}",
    "pick the {obj}",
    "grab the {obj}",
]

DEFAULT_OBJECT = "tumbler"


# Folder name for fine-tuning corpora: Episode 001, Episode 002, …
EPISODE_DIR_RE = re.compile(r"^Episode\s+(\d+)$")
EPISODE_DIR_FMT = "Episode {:03d}"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: str, image: np.ndarray) -> None:
    """Write a uint8 HxW or HxWx3 array as a PNG (stdlib only — no Pillow in the image)."""
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        color_type = 0  # greyscale
        row = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        h, w, _ = arr.shape
        color_type = 2  # RGB
        row = arr.reshape(h, w * 3)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        return write_png(path, arr[:, :, 0])
    else:
        raise ValueError(f"unsupported image shape for PNG: {arr.shape}")
    raw = b"".join(b"\x00" + row[i].tobytes() for i in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(data)


def _next_episode_dir(parent: str) -> Tuple[str, str]:
    """Return (folder_name, absolute_path) for the next Episode NNN under parent."""
    os.makedirs(parent, exist_ok=True)
    highest = 0
    for name in os.listdir(parent):
        m = EPISODE_DIR_RE.match(name)
        if m and os.path.isdir(os.path.join(parent, name)):
            highest = max(highest, int(m.group(1)))
    eid = EPISODE_DIR_FMT.format(highest + 1)
    return eid, os.path.join(parent, eid)


class EpisodeRecorder:
    def __init__(self, root_dir: str, team_id: str, primary_camera: str = "base_left"):
        self.root_dir = root_dir
        self.failed_dir = os.path.join(os.path.dirname(root_dir.rstrip("/")), "marc_pnp_failed")
        self.team_id = team_id
        self.primary_camera = primary_camera
        self.steps: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}
        self.recording = False
        self._episode_idx = 0

    @property
    def primary_key(self) -> str:
        return f"image_{self.primary_camera}"

    def start_episode(self, extra_meta: Optional[dict] = None, object_name: Optional[str] = None):
        self.steps = []
        self.recording = True
        self._episode_idx += 1
        obj = object_name or os.environ.get("MARC_TARGET_OBJECT", DEFAULT_OBJECT)
        obj_display = obj.replace("_", " ")
        lang = INSTRUCTION_TEMPLATES[(self._episode_idx - 1) % len(INSTRUCTION_TEMPLATES)].format(obj=obj_display)
        self.meta = {
            "episode_id": f"pending_{self._episode_idx:04d}",
            "team_id": self.team_id,
            "language_instruction": lang,
            "object_name": obj,
            "action_convention": "ee_delta_arm_base_dx_dy_dz_droll_dpitch_dyaw_grip",
            "primary_camera": self.primary_camera,
            "includes_place": False,
            "task": "pick",
            "success": False,
            "started_at": time.time(),
        }
        if extra_meta:
            self.meta.update(extra_meta)

    def add_step(self, obs: dict, action: np.ndarray):
        if not self.recording:
            return
        step = {
            "action": np.asarray(action, dtype=np.float32).reshape(7,),
            "q_arm": obs["q_arm"],
            "q_grip": obs["q_grip"],
            "q_inner": obs.get("q_inner", np.zeros(0, dtype=np.float32)),
            "ee_pos": obs["ee_pos"],
            "ee_quat": obs["ee_quat"],
            "grasping": np.int8(obs["grasping"]),
            "t_sim": obs["t_sim"],
            "t_wall": obs["t_wall"],
        }
        # Vanilla OpenVLA consumes one primary RGB stream. Keep exactly that image;
        # secondary RGB and depth made episodes hundreds of MiB larger but were never
        # read by conversion or fine-tuning.
        step[self.primary_key] = obs.get(self.primary_key)
        self.steps.append(step)

    def discard(self, save_failed: bool = True) -> Optional[str]:
        self.recording = False
        self.meta["success"] = False
        self.meta["finished_at"] = time.time()
        self.meta["num_steps"] = len(self.steps)
        path = None
        if save_failed and self.steps:
            path = self._write(self.failed_dir)
        self.steps = []
        return path

    def keep(self) -> str:
        self.recording = False
        self.meta["success"] = True
        self.meta["finished_at"] = time.time()
        self.meta["num_steps"] = len(self.steps)
        path = self._write(self.root_dir)
        self.steps = []
        return path

    def _drop_odd_shaped_steps(self):
        """Keep only steps whose primary RGB matches the episode's dominant resolution.

        A single differently-sized camera frame would otherwise abort the whole write.
        Dropping the step (rather than the dataset) keeps every array aligned by index.
        """
        key = self.primary_key
        counts: Dict[Any, int] = {}
        for s in self.steps:
            v = s.get(key)
            shape = None if v is None else np.asarray(v).shape
            counts[shape] = counts.get(shape, 0) + 1
        if len(counts) <= 1:
            return
        keep_shape = max(counts.items(), key=lambda kv: kv[1])[0]
        kept = [
            s for s in self.steps
            if (None if s.get(key) is None else np.asarray(s[key]).shape) == keep_shape
        ]
        odd = ", ".join(f"{sh}x{c}" for sh, c in counts.items() if sh != keep_shape)
        print(f"[record] dropped {len(self.steps) - len(kept)} step(s) with mismatched "
              f"{key} shape (kept {keep_shape}, dropped {odd})")
        self.steps = kept

    def _write(self, parent: str) -> str:
        if h5py is None:
            raise RuntimeError("h5py is required to write episodes (pip/apt install h5py)")
        if not self.steps:
            raise RuntimeError("no steps to write")
        self._drop_odd_shaped_steps()
        if not self.steps:
            raise RuntimeError("no steps to write")
        eid, out_dir = _next_episode_dir(parent)
        os.makedirs(out_dir, exist_ok=True)
        self.meta["episode_id"] = eid
        self.meta["num_steps"] = len(self.steps)
        self.meta["filled_frames"] = {}
        self.meta["includes_place"] = False
        self.meta["task"] = "pick"
        grasp_steps = [i for i, step in enumerate(self.steps) if step["grasping"]]
        grasp_index = grasp_steps[0] if grasp_steps else len(self.steps) - 1
        grasp_step = self.steps[grasp_index]
        self.meta["grasp_capture"] = {
            "step": grasp_index,
            "ee_pose_arm_base": {
                "position": np.asarray(grasp_step["ee_pos"], dtype=float).tolist(),
                "quaternion": np.asarray(grasp_step["ee_quat"], dtype=float).tolist(),
                "quaternion_order": "wxyz",
            },
            "object_in_gripper": bool(grasp_steps),
            "gripper": os.environ.get("MARC_GRASPGENX_GRIPPER", "robotiq_2f_140"),
        }
        self.meta["training_formats"] = {
            "openvla": "RLDS source via manip_dataset.to_rlds",
            "graspgenx": "capture requires exact object mesh and object pose annotation",
        }
        h5_path = os.path.join(out_dir, "episode.hdf5")
        meta_path = os.path.join(out_dir, "meta.json")

        with h5py.File(h5_path, "w") as f:
            n = len(self.steps)
            g = f.create_group("steps")
            def _stack_images(key) -> bool:
                """Write a view only if every step has it at one resolution."""
                frames = []
                for s in self.steps:
                    v = s.get(key)
                    if v is None:
                        return False
                    frames.append(np.asarray(v))
                if len({fr.shape for fr in frames}) != 1:
                    return False
                g.create_dataset(
                    key,
                    data=np.stack(frames, axis=0),
                    compression="gzip",
                    compression_opts=4,
                )
                return True

            # The primary view is the complete OpenVLA visual observation.
            written = []
            if _stack_images(self.primary_key):
                written.append(self.primary_key)
            else:
                print(f"[record] WARNING: primary view {self.primary_key} incomplete — "
                      f"episode not usable for OpenVLA training")
            g.attrs["primary_camera"] = self.primary_camera
            g.attrs["rgb_views"] = ",".join(written)
            print(f"[record] rgb view: {', '.join(written) or 'none'}")

            g.create_dataset("q_arm", data=np.stack([s["q_arm"] for s in self.steps]))
            g.create_dataset("q_grip", data=np.array([s["q_grip"] for s in self.steps], dtype=np.float32))
            # inner fingers may vary length — store first length only if consistent
            inners = [np.asarray(s["q_inner"], dtype=np.float32) for s in self.steps]
            if inners and all(x.shape == inners[0].shape for x in inners):
                g.create_dataset("q_inner", data=np.stack(inners, axis=0))
            g.create_dataset("ee_pos", data=np.stack([s["ee_pos"] for s in self.steps]))
            g.create_dataset("ee_quat", data=np.stack([s["ee_quat"] for s in self.steps]))
            g.create_dataset("grasping", data=np.array([s["grasping"] for s in self.steps], dtype=np.int8))
            g.create_dataset("action", data=np.stack([s["action"] for s in self.steps]))
            g.create_dataset("t_sim", data=np.array([s["t_sim"] for s in self.steps], dtype=np.float64))
            g.create_dataset("t_wall", data=np.array([s["t_wall"] for s in self.steps], dtype=np.float64))
            f.attrs["num_steps"] = n
            f.attrs["episode_id"] = eid
            f.attrs["primary_camera"] = self.primary_camera
            f.attrs["language_instruction"] = self.meta["language_instruction"]
            f.attrs["object_name"] = self.meta.get("object_name", DEFAULT_OBJECT)

        # HDF5 is the single source. Per-frame PNG copies previously duplicated every
        # RGB byte and are deliberately not emitted.
        self.meta["png_counts"] = {}

        with open(meta_path, "w") as fp:
            json.dump(self.meta, fp, indent=2)
        return out_dir
