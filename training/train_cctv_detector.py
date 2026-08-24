"""Prepare the Stage 1 CCTV dataset and launch the fixed YOLO26x training job."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml


CONFIG = Path(__file__).with_name("configs") / "stage1-cctv-yolo26x.yaml"
SOURCE_DATA_SHA256 = "88dee1597d7a25d7b1dc23bc7814da3669ee28e750ccf22a42c15499eca1d66c"
SOURCE_MODEL_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"
SOURCE_DATA_REVISION = "075767315653d3286fb636e9f278397ae73d84c9"
EXPECTED_COUNTS = {"images": {"train": 3733, "val": 415}, "boxes": {"train": 139505, "val": 15311}}
REMOVED_IDS = {4, 5}
IGNORED_CLASSES = ("master_chef_can", "wood_block", "umbrella", "long_umbrella")
CAMERA_RE = re.compile(r"rig_(\d+)_([a-z])$")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(root: Path, split: str) -> list[tuple[Path, Path]]:
    images = {path.stem: path for path in (root / "images" / split).glob("*.png")}
    labels = {path.stem: path for path in (root / "labels" / split).glob("*.txt")}
    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())[:5]
        missing_images = sorted(labels.keys() - images.keys())[:5]
        raise ValueError(f"{split} pairing mismatch: missing labels={missing_labels}, missing images={missing_images}")
    return [(images[key], labels[key]) for key in sorted(images)]


def _camera(stem: str) -> str:
    match = CAMERA_RE.search(stem)
    if not match:
        raise ValueError(f"cannot identify camera from {stem!r}")
    return f"rig_{match.group(1)}_{match.group(2)}"


def _remap_label(source: Path, destination: Path, class_count: int) -> tuple[int, Counter]:
    output, counts = [], Counter()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{source}:{line_number}: expected five YOLO fields")
        try:
            old_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{source}:{line_number}: invalid YOLO label") from error
        if old_id in REMOVED_IDS:
            raise ValueError(f"{source}:{line_number}: excluded class {old_id} has a box")
        new_id = old_id - 2 if old_id > max(REMOVED_IDS) else old_id
        if not 0 <= new_id < class_count:
            raise ValueError(f"{source}:{line_number}: class {old_id} is out of bounds")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{source}:{line_number}: invalid normalized box")
        output.append(" ".join((str(new_id), *fields[1:])))
        counts[new_id] += 1
    destination.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return len(output), counts


def prepare_dataset(source_yaml: Path, output_yaml: Path, *, verify_hash: bool = True) -> dict:
    source_yaml, output_yaml = source_yaml.resolve(), output_yaml.resolve()
    if verify_hash and sha256(source_yaml) != SOURCE_DATA_SHA256:
        raise ValueError(f"unexpected source descriptor checksum: {source_yaml}")
    source_root, output_root = source_yaml.parent, output_yaml.parent
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("derived dataset must be outside the raw dataset root")
    if verify_hash:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
        changed = set(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=source_root, text=True).splitlines())
        if revision != SOURCE_DATA_REVISION or changed - {"data.yaml"}:
            raise ValueError(f"unexpected source dataset revision or tracked changes: {revision}, {sorted(changed)}")
    source = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    source["names"] = {int(key): value for key, value in source["names"].items()}
    if sorted(source["names"]) != list(range(35)):
        raise ValueError("source descriptor must contain contiguous class IDs 0..34")
    if tuple(source["names"][key] for key in sorted(REMOVED_IDS)) != ("umbrella", "long_umbrella"):
        raise ValueError("source classes 4 and 5 are not the expected exclusions")
    names = [source["names"][key] for key in range(35) if key not in REMOVED_IDS]
    if len(names) != 33 or len(set(names)) != 33:
        raise ValueError("derived class map must contain 33 unique names")

    output_root.mkdir(parents=True, exist_ok=True)
    stats = {"images": {}, "boxes": {}, "cameras": {}, "class_boxes": Counter()}
    scene_ids = {}
    for split in ("train", "val"):
        pairs = _pairs(source_root, split)
        image_dir = output_root / "images" / split
        image_dir.parent.mkdir(exist_ok=True)
        if image_dir.is_symlink():
            image_dir.unlink()
        image_dir.mkdir(exist_ok=True)
        expected_images = {image.name for image, _ in pairs}
        existing_images = {path.name for path in image_dir.glob("*.png")}
        if existing_images - expected_images:
            raise ValueError(f"unexpected images in {image_dir}: {sorted(existing_images - expected_images)[:5]}")
        for image, _ in pairs:
            linked = image_dir / image.name
            if linked.exists() and not linked.samefile(image):
                raise ValueError(f"{linked} is not linked to {image}")
            if not linked.exists():
                os.link(image, linked)
        label_dir = output_root / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        cameras, split_scenes, boxes = Counter(), set(), 0
        for image, label in pairs:
            cameras[_camera(image.stem)] += 1
            split_scenes.add(image.stem.split("_rig_", 1)[0])
            count, class_counts = _remap_label(label, label_dir / label.name, len(names))
            boxes += count
            stats["class_boxes"].update(class_counts)
        _pairs(output_root, split)
        stats["images"][split] = len(pairs)
        stats["boxes"][split] = boxes
        stats["cameras"][split] = dict(sorted(cameras.items()))
        scene_ids[split] = split_scenes
    if scene_ids["train"] & scene_ids["val"]:
        raise ValueError("train and validation scenes overlap")
    if set(stats["cameras"]["train"]) != set(stats["cameras"]["val"]) or len(stats["cameras"]["train"]) != 6:
        raise ValueError("all six cameras must occur in both splits")
    if verify_hash and {key: stats[key] for key in ("images", "boxes")} != EXPECTED_COUNTS:
        raise ValueError("source dataset counts do not match the pinned revision")

    descriptor = {
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(names)),
        "ignored_classes": list(IGNORED_CLASSES),
        "source_revision": SOURCE_DATA_REVISION,
        "source_data_yaml_sha256": sha256(source_yaml),
    }
    output_yaml.write_text(yaml.safe_dump(descriptor, sort_keys=False), encoding="utf-8")
    stats["class_boxes"] = {names[key]: stats["class_boxes"][key] for key in range(len(names))}
    stats["names"] = names
    stats["ignored_classes"] = list(IGNORED_CLASSES)
    (output_root / "dataset_summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[1]
    dirty_paths = subprocess.check_output(
        ["git", "status", "--short", "--", "training", ".gitattributes"], cwd=root, text=True
    ).splitlines()
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "dirty_paths": dirty_paths,
    }


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="official yolo26x.pt checkpoint")
    source.add_argument("--resume", type=Path, help="last.pt from this run")
    parser.add_argument("--source-data", required=True, type=Path, help="raw 35-class data.yaml")
    parser.add_argument("--data", required=True, type=Path, help="derived 33-class data.yaml to create/validate")
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="0")
    parser.add_argument("--preflight", action="store_true", help="prepare and load inputs without training")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    stats = prepare_dataset(args.source_data, args.data)
    checked = check_det_dataset(str(args.data), autodownload=False)
    checkpoint = (args.resume or args.model).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.model and sha256(checkpoint) != SOURCE_MODEL_SHA256:
        raise ValueError(f"unexpected source checkpoint checksum: {checkpoint}")
    model = YOLO(checkpoint)
    overrides = {
        "model": str(checkpoint),
        "data": str(args.data.resolve()),
        "project": str(args.project.resolve()),
        "name": args.name,
        "device": args.device,
    }
    resolved = vars(get_cfg(overrides={**yaml.safe_load(args.config.read_text(encoding="utf-8")), **overrides}))
    preflight = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "dataset": {"path": str(args.data.resolve()), **stats},
        "resolved_args": resolved,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "ultralytics": ultralytics_version,
        },
        "determinism_note": "Ultralytics warns that cache=ram may make deterministic training non-deterministic.",
        "ultralytics_dataset": {key: checked[key] for key in ("path", "train", "val", "nc", "names")},
    }
    if args.preflight:
        print(json.dumps(preflight, indent=2, default=str))
        return

    run_dir = args.project.resolve() / args.name
    if not args.resume and run_dir.exists():
        raise FileExistsError(f"refusing to increment an existing production run: {run_dir}")
    record_path = run_dir / "training_record.json"
    git_state = _git_state()
    if git_state["dirty_paths"]:
        raise RuntimeError(f"commit the training implementation before production: {git_state['dirty_paths']}")
    if args.resume:
        if not record_path.is_file():
            raise FileNotFoundError(f"resume requires the original training record: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        recorded_last = Path(record["outputs"]["last"]).resolve()
        if checkpoint != recorded_last:
            raise ValueError(f"resume checkpoint must be this run's recorded last.pt: {recorded_last}")
        record.setdefault("resume_events", []).append({
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "git": git_state,
        })
    else:
        record = {**preflight, "git": git_state}

    def on_train_start(trainer) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        record["resolved_args"] = vars(trainer.args)
        record["resolved_optimizer"] = {
            "name": type(trainer.optimizer).__name__,
            "parameter_groups": [
                {key: value for key, value in group.items() if key != "params"}
                for group in trainer.optimizer.param_groups
            ],
        }
        record["outputs"] = {"best": str(trainer.best), "last": str(trainer.last)}
        _json(record_path, record)

    def on_train_end(trainer) -> None:
        record["outputs"] = {"best": str(trainer.best), "last": str(trainer.last)}
        record["peak_training_memory_gib"] = (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
        )
        record["final_metrics"] = trainer.metrics
        _json(record_path, record)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_end", on_train_end)
    if args.resume:
        model.train(resume=True, device=args.device)
    else:
        model.train(cfg=str(args.config), **overrides)


if __name__ == "__main__":
    main()
