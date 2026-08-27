"""Prepare the provisional Stage 2 robot-view detector dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


SOURCE_REVISION = "6723cca004d05de4cde72827923941605502fdcc"
SOURCE_DESCRIPTOR_SHA256 = "28827a048bbe1adc80f790dbd3411f1baacb7a44ebc59dbe9328dfae1edfbfa5"
EXPECTED_IMAGES = {"train": 2048, "val": 400}
EXCLUDED_CLASSES = frozenset({"master_chef_can", "wood_block"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(root: Path) -> tuple[str, str]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True
    ).strip()
    return revision, status


def _names(value: object) -> list[str]:
    if isinstance(value, list):
        names = [str(name) for name in value]
    elif isinstance(value, dict):
        mapping = {int(key): str(name) for key, name in value.items()}
        if sorted(mapping) != list(range(len(mapping))):
            raise ValueError("source class IDs must be contiguous from zero")
        names = [mapping[key] for key in range(len(mapping))]
    else:
        raise ValueError("source descriptor names must be a list or mapping")
    if len(names) != len(set(names)):
        raise ValueError("source class names must be unique")
    return names


def _pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    images: dict[str, Path] = {}
    for path in (root / "images" / split).iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in images:
                raise ValueError(f"duplicate image stem in {split}: {path.stem}")
            images[path.stem] = path
    labels = {path.stem: path for path in (root / "labels" / split).glob("*.txt")}
    if images.keys() != labels.keys():
        raise ValueError(
            f"{split} pairing mismatch: missing labels={sorted(images.keys() - labels.keys())[:5]}, "
            f"missing images={sorted(labels.keys() - images.keys())[:5]}"
        )
    return [(images[stem], labels[stem]) for stem in sorted(images)]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_dataset(source_yaml: Path, output_yaml: Path, *, verify_source: bool = True) -> dict:
    source_yaml, output_yaml = source_yaml.resolve(), output_yaml.resolve()
    source_root, output_root = source_yaml.parent, output_yaml.parent
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("derived dataset must be outside the raw dataset root")

    before_state = _git_state(source_root) if verify_source else None
    if verify_source:
        if before_state != (SOURCE_REVISION, ""):
            raise ValueError(f"raw source is not the clean pinned revision: {before_state}")
        if sha256(source_yaml) != SOURCE_DESCRIPTOR_SHA256:
            raise ValueError("raw source descriptor checksum does not match the pinned revision")

    source = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source_names = _names(source.get("names"))
    excluded_ids = {index for index, name in enumerate(source_names) if name in EXCLUDED_CLASSES}
    if {source_names[index] for index in excluded_ids} != EXCLUDED_CLASSES:
        raise ValueError("raw source must contain both excluded classes")
    retained_ids = [index for index in range(len(source_names)) if index not in excluded_ids]
    id_map = {old: new for new, old in enumerate(retained_ids)}
    names = [source_names[index] for index in retained_ids]
    if len(names) != 9:
        raise ValueError(f"expected nine provisional classes, got {names}")

    output_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {
        "images": {},
        "image_formats": {},
        "boxes_before": {},
        "boxes_after": {},
        "empty_labels_before": {},
        "empty_labels_after": {},
        "class_boxes": Counter(),
        "dropped_class_boxes": Counter(),
    }
    split_stems: dict[str, set[str]] = {}
    short_sides: list[float] = []

    for split in ("train", "val"):
        pairs = _pairs(source_root, split)
        if verify_source and len(pairs) != EXPECTED_IMAGES[split]:
            raise ValueError(f"unexpected {split} image count: {len(pairs)}")
        image_dir, label_dir = output_root / "images" / split, output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        expected_images = {image.name for image, _ in pairs}
        expected_labels = {label.name for _, label in pairs}
        extras = (
            {path.name for path in image_dir.iterdir() if path.is_file()} - expected_images
        ) | ({path.name for path in label_dir.glob("*.txt")} - expected_labels)
        if extras:
            raise ValueError(f"unexpected files in derived {split}: {sorted(extras)[:5]}")

        formats, boxes_before, boxes_after = Counter(), 0, 0
        empty_before = empty_after = 0
        split_stems[split] = {image.stem for image, _ in pairs}
        for image, label in pairs:
            linked = image_dir / image.name
            if linked.exists() and not linked.samefile(image):
                raise ValueError(f"derived image is not linked to its pinned source: {linked}")
            if not linked.exists():
                os.link(image, linked)
            formats[image.suffix.lower().lstrip(".")] += 1

            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            empty_before += not lines
            output: list[str] = []
            with Image.open(image) as opened:
                width, height = opened.size
            for line_number, line in enumerate(lines, 1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label}:{line_number}: expected five YOLO fields")
                try:
                    old_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError as error:
                    raise ValueError(f"{label}:{line_number}: invalid YOLO label") from error
                if not 0 <= old_id < len(source_names):
                    raise ValueError(f"{label}:{line_number}: class ID out of range")
                if not all(math.isfinite(value) and 0 <= value <= 1 for value in coords):
                    raise ValueError(f"{label}:{line_number}: invalid normalized box")
                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError(f"{label}:{line_number}: box width and height must be positive")
                boxes_before += 1
                if old_id in excluded_ids:
                    stats["dropped_class_boxes"][source_names[old_id]] += 1
                    continue
                new_id = id_map[old_id]
                output.append(" ".join((str(new_id), *fields[1:])))
                stats["class_boxes"][names[new_id]] += 1
                boxes_after += 1
                short_sides.append(min(coords[2] * width, coords[3] * height))
            empty_after += not output
            derived_label = label_dir / label.name
            derived_label.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
        stats["images"][split] = len(pairs)
        stats["image_formats"][split] = dict(sorted(formats.items()))
        stats["boxes_before"][split] = boxes_before
        stats["boxes_after"][split] = boxes_after
        stats["empty_labels_before"][split] = empty_before
        stats["empty_labels_after"][split] = empty_after
    descriptor = {
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(names)),
    }
    output_yaml.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")

    size_counts = Counter(
        "under_16px" if side < 16 else "16_to_31px" if side < 32 else "32px_or_more"
        for side in short_sides
    )
    stats["class_boxes"] = dict(stats["class_boxes"])
    stats["dropped_class_boxes"] = dict(stats["dropped_class_boxes"])
    stats["retained_box_short_side_px"] = {
        "median": statistics.median(short_sides),
        "buckets": dict(size_counts),
    }
    manifest = {
        "status": "provisional",
        "source": {
            "repository": "https://huggingface.co/ahmad-rev0/robust-roboView-yolo",
            "revision": SOURCE_REVISION,
            "descriptor_sha256": sha256(source_yaml),
        },
        "derived": {
            "descriptor": "data.yaml",
            "descriptor_sha256": sha256(output_yaml),
            "ordered_classes": names,
            "excluded_classes": sorted(EXCLUDED_CLASSES),
        },
        "statistics": stats,
        "split_checks": {
            "image_stem_overlap": len(split_stems["train"] & split_stems["val"]),
            "independent_test_split": False,
        },
        "known_limitations": [
            "No test split exists; validation must not be represented as final readiness evidence.",
            "Train and validation share scenario/background families, so the split is weak.",
            "Ordinary umbrella is absent; this provisional dataset covers nine of ten scored classes.",
            "Empty-label intent metadata is absent, so intentional negatives cannot be separated from unknown empties.",
        ],
    }
    _write_json(output_root / "dataset_manifest.json", manifest)

    if verify_source and _git_state(source_root) != before_state:
        raise RuntimeError("raw source changed while preparing the derived dataset")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--no-verify-source", action="store_true")
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.source_data, args.data, verify_source=not args.no_verify_source
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
