"""Evaluate Stage 1 YOLO26 checkpoints with the global candidate contract."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image


CAMERA_RE = __import__("re").compile(r"(rig_\d+_[a-z])$")
SIZE_BUCKETS = {
    "lt8": "native-image box short side < 8 px",
    "8_to_lt16": "native-image box short side >= 8 px and < 16 px",
    "16_to_lt32": "native-image box short side >= 16 px and < 32 px",
    "ge32": "native-image box short side >= 32 px",
    "tiny": "native-image box short side < 16 px (aggregate)",
}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _frames(data_yaml: Path) -> tuple[list[str], list[dict]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names_map = {int(key): str(value) for key, value in data["names"].items()}
    if sorted(names_map) != list(range(len(names_map))):
        raise ValueError("data.yaml names must be contiguous class IDs from 0")
    names = [names_map[key] for key in range(len(names_map))]
    root = data_yaml.parent
    images = sorted(
        path for path in (root / data["val"]).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    frames = []
    for image in images:
        match = CAMERA_RE.search(image.stem)
        labels = []
        label_path = root / "labels" / "val" / f"{image.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, cx, cy, width, height = map(float, line.split())
            labels.append((int(class_id), cx, cy, width, height))
        with Image.open(image) as opened:
            shape = (opened.height, opened.width)
        frames.append({
            "image": image,
            "camera": match.group(1) if match else "metadata-unavailable",
            "camera_metadata_source": "source filename" if match else "unavailable",
            "labels": labels,
            "shape": shape,
        })
    return names, frames


def _iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if not intersection:
        return 0.0
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def _bucket(short_side: float) -> str:
    if short_side < 8:
        return "lt8"
    if short_side < 16:
        return "8_to_lt16"
    if short_side < 32:
        return "16_to_lt32"
    return "ge32"


def _distribution(values: list[float]) -> dict:
    if not values:
        return {key: None for key in ("count", "min", "p10", "p25", "median", "p75", "p90", "p95", "max", "mean")}
    ordered = sorted(values)
    percentile = lambda q: ordered[round((len(ordered) - 1) * q)]
    return {
        "count": len(ordered), "min": ordered[0], "p10": percentile(0.10),
        "p25": percentile(0.25), "median": statistics.median(ordered),
        "p75": percentile(0.75), "p90": percentile(0.90), "p95": percentile(0.95), "max": ordered[-1],
        "mean": statistics.mean(ordered),
    }


def _result(bucket: Counter) -> dict:
    recall = bucket["matched"] / bucket["gt"] if bucket["gt"] else None
    precision = bucket["matched"] / bucket["predictions"] if bucket["predictions"] else None
    f1 = (
        2 * recall * precision / (recall + precision)
        if recall is not None and precision is not None and recall + precision else 0.0
    )
    return {
        **bucket,
        "false_candidates": max(0, bucket["predictions"] - bucket["matched"]),
        "recall": recall, "precision": precision, "f1": f1,
    }


def _score(
    frames: list[dict], predictions: dict[str, list[dict]], names: list[str],
    conf: float, budget: int, *, supported_names: set[str] | None = None,
    include_frames: bool = False,
) -> dict:
    """Apply confidence and candidate limits once to the global all-class list."""
    supported_names = supported_names or set(names)
    name_to_id = {name: class_id for class_id, name in enumerate(names)}
    supported_ids = {name_to_id[name] for name in supported_names}
    totals, classes, cameras, sizes = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    class_cameras, class_sizes = defaultdict(lambda: defaultdict(Counter)), defaultdict(lambda: defaultdict(Counter))
    matched_confidences, frame_counts, frame_details = defaultdict(list), [], []

    for frame in frames:
        height, width = frame.get("shape") or (
            predictions[frame["image"].name][0].get("shape", (720, 1280))
            if predictions[frame["image"].name] else (720, 1280)
        )
        gt = []
        for class_id, cx, cy, box_width, box_height in frame["labels"]:
            if class_id not in supported_ids:
                continue
            short_side = min(box_width * width, box_height * height)
            size = _bucket(short_side)
            gt.append({
                "class_id": class_id, "class_name": names[class_id], "size": size,
                "short_side": short_side,
                "xyxy": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                         (cx + box_width / 2) * width, (cy + box_height / 2) * height],
            })
        selected = sorted(
            (item for item in predictions[frame["image"].name] if item["conf"] >= conf),
            key=lambda item: item["conf"], reverse=True,
        )[:budget]
        frame_counts.append(len(selected))
        matched_gt, candidate_details = set(), []
        for prediction in selected:
            class_id = prediction["class_id"]
            matches = [
                (_iou(truth["xyxy"], prediction["xyxy"]), gi)
                for gi, truth in enumerate(gt)
                if gi not in matched_gt and truth["class_id"] == class_id
            ]
            overlap, gi = max(matches, default=(0.0, -1))
            matched = overlap >= 0.5
            if matched:
                matched_gt.add(gi)
                matched_confidences[class_id].append(prediction["conf"])
            best_any = max(
                ((_iou(truth["xyxy"], prediction["xyxy"]), truth["class_name"]) for truth in gt),
                default=(0.0, None),
            )
            if class_id in supported_ids:
                totals["predictions"] += 1
                classes[class_id]["predictions"] += 1
                cameras[frame["camera"]]["predictions"] += 1
                class_cameras[class_id][frame["camera"]]["predictions"] += 1
            if include_frames:
                candidate_details.append({
                    **prediction, "class_name": names[class_id], "matched": matched,
                    "matched_iou": overlap if matched else None,
                    "confused_with": best_any[1] if not matched and best_any[0] >= 0.5 else None,
                    "best_any_iou": best_any[0],
                })
        totals["candidate_load"] += len(selected)
        totals["matched"] += len(matched_gt)
        totals["gt"] += len(gt)
        for gi, truth in enumerate(gt):
            matched = gi in matched_gt
            for counter in (
                classes[truth["class_id"]], cameras[frame["camera"]], sizes[truth["size"]],
                class_cameras[truth["class_id"]][frame["camera"]],
                class_sizes[truth["class_id"]][truth["size"]],
            ):
                counter["gt"] += 1
                counter["matched"] += matched
            sizes["tiny"]["gt"] += truth["size"] in {"lt8", "8_to_lt16"}
            sizes["tiny"]["matched"] += matched and truth["size"] in {"lt8", "8_to_lt16"}
            class_sizes[truth["class_id"]]["tiny"]["gt"] += truth["size"] in {"lt8", "8_to_lt16"}
            class_sizes[truth["class_id"]]["tiny"]["matched"] += matched and truth["size"] in {"lt8", "8_to_lt16"}
        if include_frames:
            frame_details.append({
                "image": frame["image"].name, "source_image": str(frame["image"]),
                "camera": frame["camera"], "shape": list(frame["shape"]),
                "ground_truth": [
                    {**truth, "matched": index in matched_gt} for index, truth in enumerate(gt)
                ],
                "candidates": candidate_details,
                "candidate_count": len(selected), "hits_cap": len(selected) == budget,
            })

    class_results = {}
    for class_id, name in enumerate(names):
        if name not in supported_names:
            continue
        values = _result(classes[class_id])
        values.update({
            "by_camera": {key: _result(value) for key, value in sorted(class_cameras[class_id].items())},
            "by_size": {key: _result(class_sizes[class_id][key]) for key in SIZE_BUCKETS},
            "tiny_recall": _result(class_sizes[class_id]["tiny"])["recall"],
            "matched_confidence": _distribution(matched_confidences[class_id]),
        })
        class_results[name] = values
    recalls = [value["recall"] for value in class_results.values() if value["recall"] is not None]
    candidate_distribution = _distribution([float(value) for value in frame_counts])
    candidate_distribution["histogram"] = dict(sorted(Counter(frame_counts).items()))
    candidate_distribution["frames_hitting_cap"] = sum(value == budget for value in frame_counts)
    result = {
        "confidence": conf, "candidate_limit": budget,
        "candidate_limit_applied": "global all-class confidence-descending list before per-class scoring",
        "size_bucket_definitions": SIZE_BUCKETS,
        "gt": totals["gt"], "predictions": totals["predictions"],
        "matched": totals["matched"], "false_candidates": max(0, totals["predictions"] - totals["matched"]),
        "candidate_load": totals["candidate_load"],
        "candidates_per_frame": candidate_distribution,
        "recall": totals["matched"] / totals["gt"] if totals["gt"] else None,
        "precision": totals["matched"] / totals["predictions"] if totals["predictions"] else 0.0,
        "macro_recall": statistics.mean(recalls),
        "macro_precision": statistics.mean(value["precision"] or 0.0 for value in class_results.values()),
        "macro_f1": statistics.mean(value["f1"] for value in class_results.values()),
        "classes": class_results,
        "cameras": {key: _result(value) for key, value in sorted(cameras.items())},
        "sizes": {key: _result(sizes[key]) for key in SIZE_BUCKETS},
    }
    if include_frames:
        result["frames"] = frame_details
    return result


def _score_targeted(
    frames: list[dict], predictions: dict[str, list[dict]], names: list[str], conf: float, budget: int
) -> dict:
    """Preserve Stage 2 exact-target scoring without sharing Stage 1 candidate semantics."""
    totals, classes, cameras, sizes = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for frame in frames:
        height, width = frame.get("shape") or (
            predictions[frame["image"].name][0].get("shape", (720, 1280))
            if predictions[frame["image"].name] else (720, 1280)
        )
        truths = [{
            "class_id": class_id,
            "xyxy": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                     (cx + box_width / 2) * width, (cy + box_height / 2) * height],
            "size": _bucket(min(box_width * width, box_height * height)),
        } for class_id, cx, cy, box_width, box_height in frame["labels"]]
        for class_id in range(len(names)):
            class_truths = [truth for truth in truths if truth["class_id"] == class_id]
            selected = sorted(
                (item for item in predictions[frame["image"].name]
                 if item["class_id"] == class_id and item["conf"] >= conf),
                key=lambda item: item["conf"], reverse=True,
            )[:budget]
            matched_gt = set()
            for prediction in selected:
                overlap, gi = max(
                    ((_iou(truth["xyxy"], prediction["xyxy"]), gi)
                     for gi, truth in enumerate(class_truths) if gi not in matched_gt),
                    default=(0.0, -1),
                )
                if overlap >= 0.5:
                    matched_gt.add(gi)
            matched = len(matched_gt)
            totals.update(gt=len(class_truths), predictions=len(selected), matched=matched)
            classes[class_id].update(gt=len(class_truths), predictions=len(selected), matched=matched)
            cameras[frame["camera"]].update(gt=len(class_truths), predictions=len(selected), matched=matched)
            for gi, truth in enumerate(class_truths):
                sizes[truth["size"]]["gt"] += 1
                sizes[truth["size"]]["matched"] += gi in matched_gt
                if truth["size"] in {"lt8", "8_to_lt16"}:
                    sizes["tiny"]["gt"] += 1
                    sizes["tiny"]["matched"] += gi in matched_gt
    class_results = {name: _result(classes[class_id]) for class_id, name in enumerate(names)}
    return {
        "confidence": conf, "candidate_limit": budget,
        "candidate_limit_applied": "after exact-target filtering",
        "gt": totals["gt"], "predictions": totals["predictions"], "matched": totals["matched"],
        "recall": totals["matched"] / totals["gt"] if totals["gt"] else None,
        "precision": totals["matched"] / totals["predictions"] if totals["predictions"] else 0.0,
        "macro_recall": statistics.mean(value["recall"] or 0.0 for value in class_results.values()),
        "macro_f1": statistics.mean(value["f1"] for value in class_results.values()),
        "classes": class_results,
        "cameras": {key: _result(value) for key, value in cameras.items()},
        "sizes": {key: _result(value) for key, value in sizes.items()},
    }


def _worst_recall(item: dict) -> float:
    return min(value["recall"] for value in item["classes"].values() if value["recall"] is not None)


def _worst_camera(item: dict) -> float:
    values = [value["recall"] for value in item.get("cameras", {}).values() if value["recall"] is not None]
    return min(values) if values else 0.0


def _selection_rank(item: dict) -> tuple:
    return (
        item["macro_f1"], _worst_recall(item), item["classes"].get("umbrella", {}).get("recall") or 0,
        item["sizes"]["tiny"]["recall"], _worst_camera(item), item["recall"],
    )


def _head_rank(item: dict) -> tuple:
    return (
        item["macro_recall"], _worst_recall(item), item["classes"].get("umbrella", {}).get("recall") or 0,
        item["sizes"]["tiny"]["recall"], _worst_camera(item), item["recall"],
    )


def _eligible(item: dict) -> bool:
    return all((value["recall"] or 0) > 0 for value in item["classes"].values())


def _select(items) -> dict:
    eligible = [item for item in items if _eligible(item)]
    if not eligible:
        raise ValueError("no candidate detects every supported class")
    return max(eligible, key=lambda item: (*_selection_rank(item), -item["candidate_limit"]))


def _best_available(items) -> dict:
    items = list(items)
    eligible = [item for item in items if _eligible(item)]
    return max(eligible or items, key=lambda item: (*_selection_rank(item), -item["candidate_limit"]))


def _select_head(comparisons: list[dict]) -> bool:
    floor = [item for item in comparisons if item["confidence"] == 0.001 and item["candidate_limit"] == 20]
    best = []
    for end2end in (True, False):
        candidates = [item for item in floor if item["end2end"] == end2end and _eligible(item)]
        if candidates:
            best.append(max(candidates, key=_head_rank))
    if not best:
        raise ValueError("neither inference head detects every supported class at confidence 0.001/top-20")
    return max(best, key=lambda item: (*_head_rank(item), not item["end2end"]))["end2end"]


def _select_runtime(comparisons: list[dict], end2end: bool) -> dict:
    return _select(item for item in comparisons if item["end2end"] == end2end)


def _training_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("promotion_eligible") is False:
        raise ValueError("disposable smoke checkpoints cannot produce a runtime contract")
    if not record.get("final_metrics") or record.get("peak_training_memory_gib") is None:
        raise ValueError("training record does not prove a completed run")
    results_path = path.with_name("results.csv")
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(
        not math.isfinite(float(value)) for row in rows for value in row.values() if value not in (None, "")
    ):
        raise ValueError("completed run requires finite, non-empty results.csv")
    outputs = {key: Path(value).resolve() for key, value in record.get("outputs", {}).items()}
    if set(outputs) != {"best", "last"} or any(not output.is_file() for output in outputs.values()):
        raise ValueError("completed run requires readable best.pt and last.pt")
    epochs = [int(float(row["epoch"])) for row in rows]
    final_epoch = epochs[-1]
    expected_epochs = int(record["resolved_args"]["epochs"])
    if epochs != list(range(1, final_epoch + 1)) or final_epoch > expected_epochs:
        raise ValueError("results.csv epochs are not a contiguous completed training history")
    save_period = int(record["resolved_args"].get("save_period") or 0)
    expected = {
        outputs["last"].with_name(f"epoch{epoch}.pt") for epoch in range(0, final_epoch, save_period)
    } if save_period > 0 else set()
    missing = sorted(str(path) for path in expected if not path.is_file())
    if missing:
        raise ValueError(f"completed run is missing periodic checkpoints: {missing}")
    run_name = path.parent.name
    live_trainers = []
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "train_cctv_detector" in command and run_name in command:
            live_trainers.append({"pid": int(cmdline_path.parent.name), "command": command})
    if live_trainers:
        raise ValueError(f"trainer is still running for {run_name}: {live_trainers}")
    record.update({
        "completed": True, "completed_epoch": final_epoch,
        "results_csv": str(results_path.resolve()),
        "completion_proof": {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "normal_end_callback_evidence": True,
            "live_trainers": live_trainers,
        },
    })
    return record


def _predict(model, frames: list[dict], names: list[str], *, imgsz: int, batch: int,
             device: str, end2end: bool, nms_iou: float, max_det: int,
             quantize: str | None) -> tuple[dict, dict]:
    """Run one all-class inference pass and map model classes into the canonical map."""
    import torch

    model_names = dict(model.names)
    canonical = {name: class_id for class_id, name in enumerate(names)}
    class_map = {int(class_id): canonical[name] for class_id, name in model_names.items() if name in canonical}
    paths = [str(frame["image"]) for frame in frames]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started, speeds, predictions = time.perf_counter(), [], {}
    for offset in range(0, len(paths), batch):
        results = model.predict(
            paths[offset:offset + batch], imgsz=imgsz, batch=batch, device=device,
            conf=0.001, iou=nms_iou, max_det=max_det, end2end=end2end,
            quantize=quantize, verbose=False,
        )
        for frame, result in zip(frames[offset:offset + batch], results):
            speeds.append(float(result.speed["inference"]))
            predictions[frame["image"].name] = [
                {"xyxy": box.tolist(), "conf": float(conf), "class_id": class_map[int(class_id)]}
                for box, conf, class_id in zip(
                    result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu()
                ) if int(class_id) in class_map
            ]
    return predictions, {
        "collection_wall_seconds": time.perf_counter() - started,
        "collection_mean_inference_ms": statistics.mean(speeds),
        "collection_batch": batch, "quantize": quantize,
        "collection_peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None
        ),
    }


def _ap_summary(metrics) -> dict:
    rows = metrics.summary()
    return {
        "save_dir": str(metrics.save_dir), "classes": rows,
        "generic": metrics.results_dict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-record", type=Path)
    parser.add_argument("--old-model", type=Path, default=Path("models/runtime/stage1-cctv-detector/model.pt"))
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", nargs="+", type=float, default=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3))
    parser.add_argument("--candidate-limits", nargs="+", type=int, default=(20, 50))
    parser.add_argument("--nms-iou", nargs="+", type=float, default=(0.5, 0.7))
    parser.add_argument("--quantize", choices=("none", "fp16"), default="fp16")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.imgsz != 1920:
        parser.error("Stage 1 checkpoint selection is fixed to the canonical 1920 profile")
    if not {20, 50}.issubset(args.candidate_limits):
        parser.error("--candidate-limits must include top-20 and top-50")
    if any(value < 0.001 for value in args.confidence) or any(value <= 0 for value in args.candidate_limits):
        parser.error("invalid confidence or candidate limit")
    if not args.preflight and args.training_record is None:
        parser.error("--training-record is required for final provenance")

    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    names, frames = _frames(args.data.resolve())
    descriptor = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if len(names) != 34 or "umbrella" not in names or descriptor.get("ignored_classes") != [
        "long_umbrella", "master_chef_can", "wood_block"
    ]:
        raise ValueError("expected the corrected contiguous 34-class Stage 1 dataset")
    expected_cameras = {"rig_1_a", "rig_2_b", "rig_2_c", "rig_3_b", "rig_3_c", "rig_7_a"}
    if {frame["camera"] for frame in frames} != expected_cameras:
        raise ValueError("corrected Stage 1 validation must contain explicit metadata for all six cameras")
    check_det_dataset(str(args.data.resolve()), autodownload=False)
    checkpoints = [checkpoint.resolve() for checkpoint in args.checkpoints]
    if len(checkpoints) != len(set(checkpoints)):
        raise ValueError("checkpoint arguments must be unique")
    quantize = None if str(args.device).lower() == "cpu" or args.quantize == "none" else args.quantize
    if args.preflight:
        for checkpoint in checkpoints:
            model_names = list(dict(YOLO(checkpoint).names).values())
            if model_names != names:
                raise ValueError(f"checkpoint class order mismatch: {checkpoint}")
        print(json.dumps({"checkpoints": list(map(str, checkpoints)), "images": len(frames), "names": names}, indent=2))
        return

    training_record = _training_record(args.training_record)
    recorded_dataset = training_record.get("dataset", {})
    if Path(recorded_dataset.get("path", "")).resolve() != args.data.resolve():
        raise ValueError("evaluation descriptor differs from the recorded training dataset")
    if recorded_dataset.get("names") != names or recorded_dataset.get("ignored_classes") != descriptor["ignored_classes"]:
        raise ValueError("training record taxonomy differs from the corrected dataset")
    resolved = training_record.get("resolved_args", {})
    from training.train_cctv_detector import (
        EXPECTED_COUNTS, SOURCE_DATA_REVISION, SOURCE_DATA_SHA256, SOURCE_MODEL_SHA256,
    )
    expected_recipe = {
        "imgsz": 1920, "epochs": 100, "batch": 4, "patience": 20,
        "save_period": 10, "rect": True, "amp": True, "deterministic": True,
        "seed": 0, "scale": 0.2, "mosaic": 0.0, "mixup": 0.0,
        "cutmix": 0.0, "multi_scale": 0.0, "fraction": 1.0, "resume": False,
    }
    if (
        any(resolved.get(key) != value for key, value in expected_recipe.items())
        or training_record.get("checkpoint_sha256") != SOURCE_MODEL_SHA256
        or descriptor.get("source_revision") != SOURCE_DATA_REVISION
        or descriptor.get("source_data_yaml_sha256") != SOURCE_DATA_SHA256
        or any(recorded_dataset.get(key) != EXPECTED_COUNTS[key] for key in ("images", "boxes"))
    ):
        raise ValueError("training record is not the pinned non-disposable YOLO26x/1920 recipe")
    weights_dir = Path(training_record["outputs"]["last"]).resolve().parent
    saved = {path.resolve() for path in weights_dir.glob("*.pt")}
    if any(path.parent != weights_dir for path in checkpoints) or set(checkpoints) != saved:
        raise ValueError(f"evaluation must include every and only saved checkpoint from {weights_dir}")

    screening, screening_comparisons = [], []
    for checkpoint in checkpoints:
        model = YOLO(checkpoint)
        if list(dict(model.names).values()) != names:
            raise ValueError(f"checkpoint class order mismatch: {checkpoint}")
        predictions, profile = _predict(
            model, frames, names, imgsz=1920, batch=args.batch, device=args.device,
            end2end=False, nms_iou=0.7, max_det=max(args.candidate_limits), quantize=quantize,
        )
        score = {
            "checkpoint": str(checkpoint), "end2end": False, "nms_iou": 0.7,
            **profile, **_score(frames, predictions, names, 0.001, 50),
        }
        zero_recall = [
            name for name, values in score["classes"].items()
            if (values["recall"] or 0) == 0
        ]
        score.update({"eligible": not zero_recall, "rejection": zero_recall})
        screening_comparisons.append(score)
        screening.append(score)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_checkpoint = _select(screening)["checkpoint"]
    comparisons, prediction_cache = [], {}
    for end2end in (True, False):
        model = YOLO(selected_checkpoint)
        for nms_iou in ((0.7,) if end2end else args.nms_iou):
            predictions, profile = _predict(
                model, frames, names, imgsz=1920, batch=args.batch, device=args.device,
                end2end=end2end, nms_iou=nms_iou, max_det=max(args.candidate_limits), quantize=quantize,
            )
            prediction_cache[(end2end, nms_iou)] = predictions
            comparisons.extend({
                "checkpoint": selected_checkpoint, "end2end": end2end, "nms_iou": nms_iou,
                **profile, **_score(frames, predictions, names, conf, limit),
            } for conf in (0.001, *args.confidence) for limit in args.candidate_limits)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_head = _select_head(comparisons)
    selected_runtime = _select_runtime(comparisons, selected_head)
    selected_nms = selected_runtime["nms_iou"]
    selected_predictions = prediction_cache[(selected_head, selected_nms)]
    operational = _score(
        frames, selected_predictions, names, selected_runtime["confidence"],
        selected_runtime["candidate_limit"], include_frames=True,
    )
    confidence_floor_operational = _score(
        frames, selected_predictions, names, 0.001,
        selected_runtime["candidate_limit"], include_frames=True,
    )

    metrics = YOLO(selected_checkpoint).val(
        data=str(args.data.resolve()), imgsz=1920, batch=args.batch, device=args.device,
        end2end=selected_head, iou=selected_nms, conf=0.001, max_det=300, plots=True,
        project=str(args.output.parent),
        name=f"{Path(selected_checkpoint).stem}-{'e2e' if selected_head else 'nms'}-selected-val",
    )
    ap = _ap_summary(metrics)
    del metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    old_model = YOLO(args.old_model.resolve())
    old_names = set(dict(old_model.names).values())
    shared_names = set(names) & old_names
    if shared_names != set(names) - {"umbrella"}:
        raise ValueError("published model does not provide the expected 33 shared classes")
    old_predictions, old_profile = _predict(
        old_model, frames, names, imgsz=1920, batch=args.batch, device=args.device,
        end2end=selected_head, nms_iou=selected_nms,
        max_det=selected_runtime["candidate_limit"], quantize=quantize,
    )
    old_shared = _score(
        frames, old_predictions, names, selected_runtime["confidence"],
        selected_runtime["candidate_limit"], supported_names=shared_names, include_frames=True,
    )
    new_shared = _score(
        frames, selected_predictions, names, selected_runtime["confidence"],
        selected_runtime["candidate_limit"], supported_names=shared_names,
    )
    class_deltas = {}
    for name in sorted(shared_names):
        old_values, new_values = old_shared["classes"][name], new_shared["classes"][name]
        class_deltas[name] = {
            "old": {key: old_values[key] for key in ("precision", "recall", "f1")},
            "new": {key: new_values[key] for key in ("precision", "recall", "f1")},
            **{
                key: None if old_values[key] is None or new_values[key] is None
                else new_values[key] - old_values[key]
                for key in ("precision", "recall", "f1")
            },
        }
    camera_deltas = {
        camera: {
            key: new_shared["cameras"][camera][key] - old_shared["cameras"][camera][key]
            for key in ("recall", "f1")
        } for camera in new_shared["cameras"]
    }
    size_deltas = {
        size: new_shared["sizes"][size]["recall"] - old_shared["sizes"][size]["recall"]
        for size in SIZE_BUCKETS
    }

    checkpoint_sha = _sha256(Path(selected_checkpoint))
    runtime_settings = {
        "imgsz": 1920, "end2end": selected_head,
        "nms_iou": None if selected_head else selected_nms,
        "confidence": selected_runtime["confidence"],
        "candidate_limit": selected_runtime["candidate_limit"], "quantize": quantize,
    }
    runtime_contract = {
        "checkpoint": "model.pt", "checkpoint_sha256": None,
        "selected_training_checkpoint": {"name": Path(selected_checkpoint).name, "sha256": checkpoint_sha},
        "dataset": {
            "path": str(args.data.resolve()), "sha256": _sha256(args.data.resolve()),
            "source_revision": descriptor.get("source_revision"),
            "source_data_yaml_sha256": descriptor.get("source_data_yaml_sha256"),
        },
        "ordered_classes": names,
        "ignored_classes": ["long_umbrella", "master_chef_can", "wood_block"],
        "preprocessing": {
            "input": "full-frame PIL.Image RGB or NumPy uint8 HWC RGB",
            "resize": "Ultralytics letterbox to imgsz=1920",
            "coordinates": "pixel xyxy in the original image",
        },
        "inference": {"full_frame": True, **runtime_settings},
        "output": {"type": "global detections", "fields": ["class_id", "class_name", "confidence", "xyxy"]},
        "local_checkpoint_load_verified": False, "offline_bundle_load_verified": False,
    }
    report = {
        "dataset": str(args.data.resolve()), "imgsz": 1920, "images": len(frames),
        "evaluator": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__)), "argv": sys.argv},
        "training_completion": {
            "completed": True, "completed_epoch": training_record["completed_epoch"],
            "final_metrics": training_record["final_metrics"],
            "peak_training_memory_gib": training_record["peak_training_memory_gib"],
            "results_csv": training_record["results_csv"],
            "completion_proof": training_record["completion_proof"],
        },
        "selection": {
            "checkpoint_rule": "one common 1920 one-to-many/NMS IoU 0.7, confidence 0.001/global top-50 score per checkpoint; then macro per-class F1, worst-class recall, umbrella recall, tiny <16px recall, worst-camera recall, overall recall; reject any zero-recall class",
            "head_rule": "confidence 0.001/top-20 equal global budget: macro recall; worst class; umbrella; tiny; worst camera; overall recall",
            "runtime_rule": "validation-only confidence/NMS/top-20 or top-50 using the checkpoint ranking hierarchy",
            "checkpoint": selected_checkpoint, "runtime": runtime_settings,
        },
        "screening": screening, "screening_comparisons": screening_comparisons,
        "head_and_runtime_comparisons": comparisons,
        "generic_validation": ap, "operational": operational,
        "confidence_floor_operational": confidence_floor_operational,
        "old_published_comparison": {
            "model": str(args.old_model.resolve()), "model_sha256": _sha256(args.old_model.resolve()),
            "umbrella": "unsupported: the published 33-class head cannot emit ordinary umbrella",
            "shared_classes": sorted(shared_names), "old": old_shared, "new": new_shared,
            "class_deltas": class_deltas, "camera_deltas": camera_deltas,
            "size_recall_deltas": size_deltas, "old_collection_profile": old_profile,
        },
        "runtime_contract": runtime_contract,
        "known_limitations": [
            "No independent test split exists; the same validation split was used for selection and reporting.",
            "Camera identifiers are source filename metadata; no independent camera manifest was supplied.",
            "Human blind full-frame review is pending; this report cannot claim final replacement readiness.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    contract_path = args.output.with_name("runtime_contract.json")
    contract_path.write_text(json.dumps(runtime_contract, indent=2) + "\n", encoding="utf-8")
    training_record.update({
        "completed": True, "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": checkpoint_sha, "evaluation": report["evaluator"],
        "evaluation_report": str(args.output.resolve()), "runtime_contract": str(contract_path.resolve()),
        "selected_validation_plots": ap["save_dir"],
    })
    args.training_record.write_text(json.dumps(training_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
