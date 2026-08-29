"""Prepare or train the provisional Stage 2 robot-view detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


SOURCE_REPOSITORY = "https://huggingface.co/datasets/roBUst-marc-2026/robust-nearfield-yoloWithUmb"
SOURCE_REVISION = "e52c3156f004f6b07bdec044465c9d1be59ad4f3"
SOURCE_DESCRIPTOR_SHA256 = "ee810912b28da56806cdd3b768d2f8cb76373950e7d1784310bdc7d9f456dd6a"
EXPECTED_IMAGES = {"train": 2266, "val": 454}
EXCLUDED_CLASSES = frozenset({"master_chef_can", "wood_block"})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
CONFIG = Path(__file__).with_name("configs") / "stage2-roboview-yolo26x.yaml"
STAGE1_SEED_SHA256 = "12712d12e3321dffeb49296285a7d99f8fab12b46aa99a808b5d3cd1da80da22"
STAGE1_SEED_RUN = "stage1-cctv-yolo26x-1920-umbrella-ceccc62"


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
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


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
    if len(names) != 10 or names[-1] != "umbrella":
        raise ValueError(f"expected ten target classes ending with umbrella, got {names}")

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
            "repository": SOURCE_REPOSITORY,
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
            "Empty-label intent metadata is absent, so intentional negatives cannot be separated from unknown empties.",
        ],
    }
    _write_json(output_root / "dataset_manifest.json", manifest)

    if verify_source and _git_state(source_root) != before_state:
        raise RuntimeError("raw source changed while preparing the derived dataset")
    return manifest


def train_detector(args: argparse.Namespace) -> None:
    import torch
    import torchvision
    from ultralytics import YOLO, __version__ as ultralytics_version
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    data, checkpoint = args.data.resolve(), args.model.resolve()
    if not checkpoint.is_file() or sha256(checkpoint) != STAGE1_SEED_SHA256:
        raise ValueError("--model must be the original selected Stage 1 last.pt")
    manifest_path = data.with_name("dataset_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = manifest["derived"]["ordered_classes"]
    checked = check_det_dataset(str(data), autodownload=False)
    if list(checked["names"].values()) != names:
        raise ValueError("dataset class order does not match its manifest")
    model = YOLO(checkpoint)
    seed_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    missing = [name for name in names if name not in seed_names]
    if missing:
        raise ValueError(f"Stage 1 seed is missing transferable classes: {missing}")

    overrides = {
        "model": str(checkpoint),
        "data": str(data),
        "project": str(args.project.resolve()),
        "name": args.name,
        "device": args.device,
    }
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not config.get("cls_remap"):
        raise ValueError("Stage 2 training requires name-based cls_remap")
    if args.smoke:
        overrides.update(epochs=1, patience=1, imgsz=640, batch=1, workers=0, cache=False, fraction=0.01)
    resolved = vars(get_cfg(overrides={**config, **overrides}))
    run_dir = args.project.resolve() / args.name
    record = {
        "status": "disposable_smoke" if args.smoke else "provisional",
        "promotion_eligible": not args.smoke,
        "completed": False,
        "seed_checkpoint": {
            "path": str(checkpoint),
            "sha256": STAGE1_SEED_SHA256,
            "selected_run": STAGE1_SEED_RUN,
            "ordered_classes": seed_names,
        },
        "dataset": {"path": str(data), **manifest},
        "resolved_args": resolved,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics_version,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
            ).strip(),
        },
    }
    if args.preflight:
        print(json.dumps(record, indent=2, default=str))
        return
    if run_dir.exists():
        raise FileExistsError(f"refusing to increment an existing run: {run_dir}")
    if not torch.cuda.is_available() or str(args.device).lower() == "cpu":
        raise RuntimeError("Stage 2 training requires CUDA")

    record_path = run_dir / "training_record.json"

    def on_train_start(trainer) -> None:
        trained_names = list(trainer.model.names.values())
        if trained_names != names:
            raise RuntimeError(f"expected remapped ten-class head, got {trained_names}")
        torch.cuda.reset_peak_memory_stats()
        record["resolved_args"] = vars(trainer.args)
        record["trained_class_order"] = trained_names
        record["resolved_optimizer"] = {
            "name": type(trainer.optimizer).__name__,
            "parameter_groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in trainer.optimizer.param_groups
            ],
        }
        record["outputs"] = {"best": str(trainer.best), "last": str(trainer.last)}
        _write_json(record_path, record)

    def on_train_end(trainer) -> None:
        record["outputs"] = {"best": str(trainer.best), "last": str(trainer.last)}
        record["peak_training_memory_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        record["final_metrics"] = trainer.metrics
        record["completed"] = True
        _write_json(record_path, record)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_end", on_train_end)
    model.train(cfg=str(args.config), **overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-data", required=True, type=Path)
    prepare.add_argument("--data", required=True, type=Path)
    prepare.add_argument("--no-verify-source", action="store_true")
    train = commands.add_parser("train")
    train.add_argument("--model", required=True, type=Path)
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--project", required=True, type=Path)
    train.add_argument("--name", required=True)
    train.add_argument("--config", type=Path, default=CONFIG)
    train.add_argument("--device", default="0")
    train.add_argument("--preflight", action="store_true")
    train.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.command == "train":
        train_detector(args)
        return
    manifest = prepare_dataset(
        args.source_data, args.data, verify_source=not args.no_verify_source
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
