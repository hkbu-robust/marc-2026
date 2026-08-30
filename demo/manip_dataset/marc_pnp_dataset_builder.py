#!/usr/bin/env python3
"""Materialize a ``to_rlds.py`` export as the ``marc_pnp`` TFDS/RLDS dataset."""

from __future__ import annotations

import argparse
import glob
import hashlib
import os

import numpy as np
import tensorflow_datasets as tfds


def split_episode_paths(pattern, val_fraction=0.1):
    """Deterministically split whole episodes; never leak frames across splits."""
    paths = sorted(glob.glob(pattern))
    if len(paths) < 2:
        return paths, []
    val_count = max(1, round(len(paths) * float(val_fraction)))
    shuffled = sorted(
        paths,
        key=lambda path: hashlib.sha256(os.path.basename(path).encode()).digest(),
    )
    val = set(shuffled[:val_count])
    return [path for path in paths if path not in val], [path for path in paths if path in val]


class MarcPnp(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.2.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(None, None, 3), dtype=np.uint8,
                            encoding_format="png",
                        ),
                        # OpenVLA StateEncoding.JOINT: seven joint slots + gripper.
                        "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        source = os.path.join(str(dl_manager.manual_dir), "episodes", "episode_*.npz")
        train, val = split_episode_paths(source)
        splits = {"train": self._generate_examples(train)}
        if val:
            # OpenVLA looks specifically for a split named "val".
            splits["val"] = self._generate_examples(val)
        return splits

    def _generate_examples(self, paths):
        for path in paths:
            data = np.load(path, allow_pickle=True)
            count = len(data["action"])
            instruction = str(data["language_instruction"].item())
            steps = []
            for index in range(count):
                last = index == count - 1
                steps.append({
                    "observation": {
                        "image": data["image"][index],
                        "state": data["state"][index],
                    },
                    "action": data["action"][index],
                    "discount": np.float32(1.0),
                    "reward": np.float32(last),
                    "is_first": index == 0,
                    "is_last": last,
                    "is_terminal": last,
                    "language_instruction": instruction,
                })
            yield os.path.basename(path), {
                "steps": steps,
                "episode_metadata": {"file_path": path},
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-dir", required=True, help="to_rlds.py output")
    parser.add_argument("--data-dir", required=True, help="TFDS output root")
    args = parser.parse_args()
    MarcPnp(data_dir=args.data_dir).download_and_prepare(
        download_config=tfds.download.DownloadConfig(manual_dir=args.manual_dir)
    )


if __name__ == "__main__":
    main()
