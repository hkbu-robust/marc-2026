#!/usr/bin/env python3
"""Materialize a ``to_rlds.py`` export as the ``marc_pnp`` TFDS/RLDS dataset."""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import tensorflow_datasets as tfds


class MarcPnp(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

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
                        "state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
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
        return {"train": self._generate_examples(source)}

    def _generate_examples(self, pattern):
        for path in sorted(glob.glob(pattern)):
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
