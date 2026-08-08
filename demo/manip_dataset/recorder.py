"""In-memory episode buffer and HDF5 writer."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover — ImportError or numpy ABI mismatch
    h5py = None


LANGUAGE_PARAPHRASES = [
    "pick up the tumbler and place it in the basket",
    "grasp the cup and put it into the rear basket",
    "pick the tumbler object and drop it in the basket",
    "grab the tumbler and place it in the robot basket",
]


class EpisodeRecorder:
    def __init__(self, root_dir: str, team_id: str):
        self.root_dir = root_dir
        self.failed_dir = os.path.join(os.path.dirname(root_dir.rstrip("/")), "marc_pnp_failed")
        self.team_id = team_id
        self.steps: List[Dict[str, Any]] = []
        self.meta: Dict[str, Any] = {}
        self.recording = False
        self._episode_idx = 0

    def start_episode(self, extra_meta: Optional[dict] = None):
        self.steps = []
        self.recording = True
        self._episode_idx += 1
        eid = f"{time.strftime('%Y%m%d_%H%M%S')}_{self._episode_idx:04d}_{uuid.uuid4().hex[:8]}"
        lang = LANGUAGE_PARAPHRASES[(self._episode_idx - 1) % len(LANGUAGE_PARAPHRASES)]
        self.meta = {
            "episode_id": eid,
            "team_id": self.team_id,
            "language_instruction": lang,
            "action_convention": "ee_delta_arm_base_dx_dy_dz_droll_dpitch_dyaw_grip",
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
            "image_gripper": obs["image_gripper"],
            "image_base": obs.get("image_base"),
            "depth_gripper": obs.get("depth_gripper"),
            "depth_base": obs.get("depth_base"),
            "q_arm": obs["q_arm"],
            "q_grip": obs["q_grip"],
            "q_inner": obs.get("q_inner", np.zeros(0, dtype=np.float32)),
            "ee_pos": obs["ee_pos"],
            "ee_quat": obs["ee_quat"],
            "grasping": np.int8(obs["grasping"]),
            "t_sim": obs["t_sim"],
            "t_wall": obs["t_wall"],
        }
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
        """Keep only steps whose gripper RGB matches the episode's dominant resolution.

        A single differently-sized camera frame would otherwise abort the whole write.
        Dropping the step (rather than the dataset) keeps every array aligned by index.
        """
        counts: Dict[Any, int] = {}
        for s in self.steps:
            shape = np.asarray(s["image_gripper"]).shape
            counts[shape] = counts.get(shape, 0) + 1
        if len(counts) <= 1:
            return
        keep_shape = max(counts.items(), key=lambda kv: kv[1])[0]
        kept = [s for s in self.steps if np.asarray(s["image_gripper"]).shape == keep_shape]
        print(f"[record] dropped {len(self.steps) - len(kept)} step(s) with mismatched "
              f"image shape (kept {keep_shape})")
        self.steps = kept

    def _write(self, parent: str) -> str:
        if h5py is None:
            raise RuntimeError("h5py is required to write episodes (pip/apt install h5py)")
        if not self.steps:
            raise RuntimeError("no steps to write")
        self._drop_odd_shaped_steps()
        if not self.steps:
            raise RuntimeError("no steps to write")
        self.meta["num_steps"] = len(self.steps)
        eid = self.meta["episode_id"]
        out_dir = os.path.join(parent, eid)
        os.makedirs(out_dir, exist_ok=True)
        h5_path = os.path.join(out_dir, "episode.hdf5")
        meta_path = os.path.join(out_dir, "meta.json")

        with h5py.File(h5_path, "w") as f:
            n = len(self.steps)
            g = f.create_group("steps")
            # Assume constant image resolution within an episode.
            g.create_dataset(
                "image_gripper",
                data=np.stack([s["image_gripper"] for s in self.steps], axis=0),
                compression="gzip",
                compression_opts=4,
            )
            if self.steps[0]["image_base"] is not None:
                bases = []
                ok = True
                for s in self.steps:
                    if s["image_base"] is None or s["image_base"].shape != self.steps[0]["image_base"].shape:
                        ok = False
                        break
                    bases.append(s["image_base"])
                if ok:
                    g.create_dataset(
                        "image_base",
                        data=np.stack(bases, axis=0),
                        compression="gzip",
                        compression_opts=4,
                    )

            def _stack_optional(key, dtype):
                arrays = []
                for s in self.steps:
                    v = s.get(key)
                    if v is None:
                        return
                    arrays.append(np.asarray(v, dtype=dtype))
                if not arrays:
                    return
                shapes = {a.shape for a in arrays}
                if len(shapes) != 1:
                    return
                g.create_dataset(key, data=np.stack(arrays, axis=0), compression="gzip")

            _stack_optional("depth_gripper", np.float32)
            _stack_optional("depth_base", np.float32)

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

        with open(meta_path, "w") as fp:
            json.dump(self.meta, fp, indent=2)
        return out_dir
