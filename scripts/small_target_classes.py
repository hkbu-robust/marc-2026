"""Derive the detector's small target classes from its own training labels (#115).

    python -m scripts.small_target_classes [--out marc_agent/perception/data/small_targets.json]

A class is small when its median ground-truth box covers less than
``MAX_MEDIAN_AREA_FRAC`` of the frame in the training labels of the dataset the
runtime contract names. Areas are relative (YOLO labels are normalised), so the
rule holds at any frame resolution. The classes split into two groups with a
7.5x gap between them (largest small class 1.8e-4, smallest other class 1.35e-3);
the threshold is the gap's log-midpoint, rounded.

The tiled refinement pass runs only for these classes. The detector bundle itself
(weights, contract, checksums) is not modified.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

from marc_agent.perception.bundle_paths import STAGE1_CCTV_CONTRACT
from marc_agent.perception.cctv_yolo import SMALL_TARGETS_PATH

MAX_MEDIAN_AREA_FRAC = 5e-4


def median_area_fractions(dataset_yaml: Path) -> tuple[dict[str, float], dict[str, int]]:
    import yaml

    names = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))["names"]
    areas = collections.defaultdict(list)
    for label in sorted((dataset_yaml.parent / "labels" / "train").glob("*.txt")):
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                areas[names[int(parts[0])]].append(float(parts[3]) * float(parts[4]))
    return (
        {name: float(np.median(values)) for name, values in areas.items()},
        {name: len(values) for name, values in areas.items()},
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contract", type=Path, default=STAGE1_CCTV_CONTRACT)
    parser.add_argument("--out", type=Path, default=SMALL_TARGETS_PATH)
    args = parser.parse_args(argv)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    dataset = Path(os.path.expanduser(contract["dataset"]["path"]))
    medians, counts = median_area_fractions(dataset)
    small = sorted(name for name, area in medians.items() if area < MAX_MEDIAN_AREA_FRAC)
    payload = {
        "rule": "median training-label box area / frame area < max_median_area_frac",
        "max_median_area_frac": MAX_MEDIAN_AREA_FRAC,
        "classes": small,
        "dataset": str(dataset),
        "dataset_sha256": contract["dataset"]["sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "median_area_frac": {name: round(medians[name], 7) for name in sorted(medians)},
        "train_boxes": {name: counts[name] for name in sorted(counts)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"classes": small, "out": str(args.out)}, indent=1))


if __name__ == "__main__":
    main()
