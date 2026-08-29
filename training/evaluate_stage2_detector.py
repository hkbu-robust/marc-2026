"""Evaluate ten-class Stage 2 YOLO26 checkpoints and runtime settings."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


CAMERA_RE = re.compile(r"(rig_\d+_[a-z])$")
RUNTIME_IMAGE_SIZES = (640, 1280, 1920)
RECALL_LIMITS = (1, 5, 10, 20)
SIZE_BUCKETS = {
    "tiny": "native-image box short side < 16 px",
    "small": "native-image box short side >= 16 px and < 32 px",
    "large": "native-image box short side >= 32 px",
}
EXPECTED_NAMES = (
    "tumbler", "cracker_box", "cola_can", "disposable_cup", "juice",
    "tissue", "mug", "sunblock", "pencilcase", "umbrella",
)
DATASET_REVISION = "e52c3156f004f6b07bdec044465c9d1be59ad4f3"
SEED_SHA256 = "12712d12e3321dffeb49296285a7d99f8fab12b46aa99a808b5d3cd1da80da22"
MATERIAL_REGRESSION = 0.05


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _frames(data_yaml: Path) -> tuple[list[str], list[dict]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names_map = {int(key): value for key, value in data["names"].items()}
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
        for line in (root / "labels" / "val" / f"{image.stem}.txt").read_text().splitlines():
            class_id, cx, cy, width, height = map(float, line.split())
            labels.append((int(class_id), cx, cy, width, height))
        with Image.open(image) as opened:
            shape = (opened.height, opened.width)
        frames.append({
            "image": image,
            "camera": match.group(1) if match else "all",
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


def _score(frames: list[dict], predictions: dict[str, list[dict]], names: list[str], conf: float, budget: int) -> dict:
    """Score the exact-target interface, applying each budget after class filtering."""
    totals, classes, cameras, sizes = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    class_sizes = defaultdict(lambda: defaultdict(Counter))
    for frame in frames:
        height, width = frame.get("shape") or (
            predictions[frame["image"].name][0]["shape"]
            if predictions[frame["image"].name] else (720, 1280)
        )
        gt = []
        for class_id, cx, cy, box_width, box_height in frame["labels"]:
            pixel_width, pixel_height = box_width * width, box_height * height
            short_side = min(pixel_width, pixel_height)
            gt.append({
                "class_id": class_id,
                "xyxy": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                           (cx + box_width / 2) * width, (cy + box_height / 2) * height],
                "size": "tiny" if short_side < 16 else "small" if short_side < 32 else "large",
            })
        if not gt:
            totals["empty_label_frames"] += 1
        frame_predictions = predictions[frame["image"].name]
        for class_id in range(len(names)):
            truths = [truth for truth in gt if truth["class_id"] == class_id]
            selected = sorted(
                (
                    item for item in frame_predictions
                    if item["class_id"] == class_id and item["conf"] >= conf
                ),
                key=lambda item: item["conf"], reverse=True,
            )[:budget]
            matched_gt = set()
            for prediction in selected:
                matches = [
                    (_iou(truth["xyxy"], prediction["xyxy"]), gi)
                    for gi, truth in enumerate(truths)
                    if gi not in matched_gt
                ]
                overlap, gi = max(matches, default=(0.0, -1))
                if overlap >= 0.5:
                    matched_gt.add(gi)
            matched = len(matched_gt)
            totals.update(gt=len(truths), predictions=len(selected), matched=matched, requests=1)
            if truths:
                totals["present_target_requests"] += 1
            else:
                totals["absent_target_requests"] += 1
                totals["absent_target_false_candidates"] += len(selected)
                totals["absent_target_requests_with_candidates"] += bool(selected)
            if not gt:
                totals["empty_label_false_candidates"] += len(selected)
            class_bucket = classes[class_id]
            class_bucket.update(gt=len(truths), predictions=len(selected), matched=matched)
            camera_bucket = cameras[frame["camera"]]
            camera_bucket.update(gt=len(truths), predictions=len(selected), matched=matched)
            for gi, truth in enumerate(truths):
                size_bucket = sizes[truth["size"]]
                size_bucket["gt"] += 1
                size_bucket["matched"] += gi in matched_gt
                class_size = class_sizes[class_id][truth["size"]]
                class_size["gt"] += 1
                class_size["matched"] += gi in matched_gt

    def result(bucket: Counter) -> dict:
        recall = bucket["matched"] / bucket["gt"] if bucket["gt"] else None
        precision = bucket["matched"] / bucket["predictions"] if bucket["predictions"] else None
        f1 = 2 * recall * precision / (recall + precision) if recall is not None and precision is not None and recall + precision else 0.0
        return {
            **bucket,
            "false_candidates": max(0, bucket["predictions"] - bucket["matched"]),
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }

    class_results = {}
    for class_id, name in enumerate(names):
        class_results[name] = {
            **result(classes[class_id]),
            "by_size": {bucket: result(class_sizes[class_id][bucket]) for bucket in SIZE_BUCKETS},
        }
    recalls = [value["recall"] for value in class_results.values() if value["recall"] is not None]
    macro_recall = statistics.mean(recalls)
    macro_precision = statistics.mean(value["precision"] or 0.0 for value in class_results.values())
    return {
        "confidence": conf,
        "candidate_limit": budget,
        "candidate_limit_applied": "after exact-target filtering",
        "size_bucket_definitions": SIZE_BUCKETS,
        "gt": totals["gt"],
        "predictions": totals["predictions"],
        "false_candidates": max(0, totals["predictions"] - totals["matched"]),
        "recall": totals["matched"] / totals["gt"],
        "precision": totals["matched"] / totals["predictions"] if totals["predictions"] else 0.0,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": statistics.mean(value["f1"] for value in class_results.values()),
        "classes": class_results,
        "cameras": {key: result(value) for key, value in sorted(cameras.items())},
        "sizes": {key: result(value) for key, value in sizes.items()},
        "target_requests": {
            "total": totals["requests"],
            "present": totals["present_target_requests"],
            "absent": totals["absent_target_requests"],
            "absent_with_candidates": totals["absent_target_requests_with_candidates"],
            "false_candidates_when_absent": totals["absent_target_false_candidates"],
        },
        "empty_label_frames": {
            "frames": totals["empty_label_frames"],
            "false_candidates": totals["empty_label_false_candidates"],
            "label": "empty-label frames; intentional-negative status is unknown",
        },
    }


def _recall_rank(item: dict) -> tuple:
    """Rank candidate coverage without letting aggregate recall hide weak classes."""
    return (
        item["macro_recall"],
        min(value["recall"] for value in item["classes"].values() if value["recall"] is not None),
        item["sizes"].get("tiny", {}).get("recall") or 0,
        item["sizes"].get("small", {}).get("recall") or 0,
        item["recall"],
    )


def _select_head(comparisons: list[dict]) -> bool:
    """Compare each head's best NMS setting at the same confidence and candidate budget."""
    equal_budget = [
        item for item in comparisons
        if item["confidence"] == 0.001 and item["candidate_limit"] == 20
    ]
    best_per_head = [
        _best_available(item for item in equal_budget if item["end2end"] == end2end)
        for end2end in (True, False)
    ]
    eligible = [item for item in best_per_head if _eligible(item)]
    if not eligible:
        raise ValueError("neither inference head detects every supported class")
    return max(
        eligible,
        key=lambda item: (*_selection_rank(item), not item["end2end"], -item["mean_inference_ms"]),
    )["end2end"]


def _select_runtime(comparisons: list[dict], end2end: bool) -> dict:
    """Select settings across every candidate configuration for the chosen head."""
    return _select(item for item in comparisons if item["end2end"] == end2end)


def _selection_rank(item: dict) -> tuple:
    return (
        item["macro_f1"],
        min(value["recall"] for value in item["classes"].values() if value["recall"] is not None),
        item["sizes"].get("tiny", {}).get("recall") or 0,
        item["sizes"].get("small", {}).get("recall") or 0,
        item["recall"],
    )


def _select(items) -> dict:
    eligible = [item for item in items if _eligible(item)]
    if not eligible:
        raise ValueError("no candidate detects every supported class")
    return _best(eligible)


def _eligible(item: dict) -> bool:
    return all((value["recall"] or 0) > 0 for value in item["classes"].values())


def _best(items) -> dict:
    return max(items, key=lambda item: (*_selection_rank(item), -item["candidate_limit"]))


def _best_available(items) -> dict:
    items = list(items)
    eligible = [item for item in items if _eligible(item)]
    return _best(eligible or items)


def _training_record(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("promotion_eligible", True):
        raise ValueError("disposable smoke checkpoints cannot produce a runtime contract")
    if record.get("completed") is not True or not record.get("final_metrics"):
        raise ValueError("training record does not prove a completed run")
    if record.get("peak_training_memory_gib") is None:
        raise ValueError("training record is missing final peak-memory evidence")
    results_path = path.with_name("results.csv")
    with results_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("completed run requires a non-empty results.csv")
    if any(
        not math.isfinite(float(value))
        for row in rows for value in row.values() if value not in (None, "")
    ):
        raise ValueError("results.csv contains non-finite values")
    epochs = [int(float(row["epoch"])) for row in rows]
    if epochs != list(range(epochs[0], epochs[0] + len(epochs))):
        raise ValueError("results.csv epochs are not contiguous")
    if record.get("trained_class_order") != list(EXPECTED_NAMES):
        raise ValueError("training record does not prove the exact ten-class order")
    if not record.get("resolved_optimizer"):
        raise ValueError("training record is missing the resolved optimizer")
    if record.get("dataset", {}).get("source", {}).get("revision") != DATASET_REVISION:
        raise ValueError("training record points to the wrong dataset revision")
    if record.get("seed_checkpoint", {}).get("sha256") != SEED_SHA256:
        raise ValueError("training record points to the wrong Stage 1 seed")
    outputs = {key: Path(value).resolve() for key, value in record.get("outputs", {}).items()}
    if set(outputs) != {"best", "last"} or any(not output.is_file() for output in outputs.values()):
        raise ValueError("completed run requires readable best.pt and last.pt")
    final_epoch = int(float(rows[-1]["epoch"]))
    save_period = int(record.get("resolved_args", {}).get("save_period") or 0)
    expected_periodic = {
        outputs["last"].with_name(f"epoch{epoch}.pt")
        for epoch in range(0, final_epoch, save_period)
    } if save_period > 0 else set()
    missing = sorted(str(checkpoint) for checkpoint in expected_periodic if not checkpoint.is_file())
    if missing:
        raise ValueError(f"completed run is missing periodic checkpoints: {missing}")
    return record


def _predict_targeted(model, frames: list[dict], names: list[str], *, imgsz: int, batch: int,
                      device: str, end2end: bool, nms_iou: float, max_det: int,
                      quantize: str | None) -> tuple[dict, dict]:
    """Run each exact-class request so max_det is applied after target filtering."""
    import torch

    paths = [str(frame["image"]) for frame in frames]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started, speeds = time.perf_counter(), []
    predictions = {frame["image"].name: [] for frame in frames}
    for requested_class in range(len(names)):
        for offset in range(0, len(paths), batch):
            results = model.predict(
                paths[offset:offset + batch], imgsz=imgsz, batch=batch, device=device,
                conf=0.001, iou=nms_iou, max_det=max_det, end2end=end2end,
                classes=[requested_class], quantize=quantize, verbose=False,
            )
            for frame, result in zip(frames[offset:offset + batch], results):
                shape = tuple(result.orig_shape)
                speeds.append(result.speed["inference"])
                predictions[frame["image"].name].extend(
                    {
                        "xyxy": box.tolist(), "conf": float(conf),
                        "class_id": int(class_id), "shape": shape,
                    }
                    for box, conf, class_id in zip(
                        result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu()
                    )
                )
    return predictions, {
        "wall_seconds": time.perf_counter() - started,
        "mean_inference_ms": statistics.mean(speeds),
        "target_requests": len(speeds),
        "batch": batch,
        "quantize": quantize,
        "peak_inference_memory_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-record", type=Path, help="training_record.json to finalize")
    parser.add_argument("--baseline-stage2", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", nargs="+", type=float, default=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3))
    parser.add_argument("--candidate-limits", nargs="+", type=int, default=(10, 20))
    parser.add_argument("--nms-iou", nargs="+", type=float, default=(0.5, 0.7))
    parser.add_argument("--quantize", choices=("none", "fp16"), default="fp16")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if 20 not in args.candidate_limits:
        parser.error("--candidate-limits must include 20 for equal-budget checkpoint selection")
    if any(confidence < 0.001 for confidence in args.confidence):
        parser.error("--confidence values must be at least the 0.001 collection floor")
    if any(limit > 20 for limit in args.candidate_limits):
        parser.error("--candidate-limits cannot exceed the runtime contract's top-20 bound")
    if not args.preflight and args.training_record is None:
        parser.error("--training-record is required for final checkpoint provenance")

    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    names, frames = _frames(args.data.resolve())
    if tuple(names) != EXPECTED_NAMES:
        raise ValueError(f"expected exact Stage 2 class order {list(EXPECTED_NAMES)}, got {names}")
    quantize = None if str(args.device).lower() == "cpu" or args.quantize == "none" else args.quantize
    check_det_dataset(str(args.data.resolve()), autodownload=False)
    if not names or not frames:
        raise ValueError("expected a non-empty validation dataset")
    checkpoints = [checkpoint.resolve() for checkpoint in args.checkpoints]
    baseline = YOLO(args.baseline_stage2.resolve())
    baseline_names = list(baseline.names.values()) if isinstance(baseline.names, dict) else list(baseline.names)
    if baseline_names != names[:-1]:
        raise ValueError(f"baseline must contain the nine shared classes in order, got {baseline_names}")
    del baseline
    if args.preflight:
        for checkpoint in checkpoints:
            model = YOLO(checkpoint)
            model_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
            if model_names != names:
                raise ValueError(f"checkpoint class order does not match the derived dataset: {checkpoint}")
        print(json.dumps({
            "checkpoints": [str(item) for item in checkpoints], "images": len(frames),
            "names": names, "quantize": quantize,
        }, indent=2))
        return
    training_record = _training_record(args.training_record)
    weights_dir = Path(training_record["outputs"]["last"]).resolve().parent
    if any(checkpoint.parent != weights_dir for checkpoint in checkpoints):
        raise ValueError(f"all evaluated checkpoints must belong to the recorded run: {weights_dir}")
    saved_checkpoints = {path.resolve() for path in weights_dir.glob("*.pt")}
    if set(checkpoints) != saved_checkpoints:
        missing = sorted(str(path) for path in saved_checkpoints - set(checkpoints))
        raise ValueError(f"evaluation must include every saved checkpoint; missing={missing}")

    screening, screening_comparisons = [], []
    for checkpoint in checkpoints:
        model = YOLO(checkpoint)
        model_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if model_names != names:
            raise ValueError(f"checkpoint class order does not match the derived dataset: {checkpoint}")
        predictions, profile = _predict_targeted(
            model, frames, names, imgsz=args.imgsz, batch=args.batch, device=args.device,
            end2end=False, nms_iou=0.7, max_det=max(args.candidate_limits), quantize=quantize,
        )
        checkpoint_comparisons = [
            {
                "checkpoint": str(checkpoint), "end2end": False, "nms_iou": 0.7,
                **profile, **_score(frames, predictions, names, confidence, budget),
            }
            for confidence in (0.001, *args.confidence)
            for budget in args.candidate_limits
        ]
        screening_comparisons.extend(checkpoint_comparisons)
        screened = _best_available(checkpoint_comparisons)
        screened["eligible"] = _eligible(screened)
        screened["rejection_reasons"] = [] if screened["eligible"] else [
            "zero recall: " + ", ".join(
                name for name, values in screened["classes"].items() if not (values["recall"] or 0)
            )
        ]
        screening.append(screened)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_checkpoint = _select(screening)["checkpoint"]
    comparisons, ap = [], []
    for end2end in (True, False):
        # Prediction fuses the selected head; a fresh model keeps the alternate branch available.
        head_model = YOLO(selected_checkpoint)
        for nms_iou in ((0.7,) if end2end else args.nms_iou):
            predictions, profile = _predict_targeted(
                head_model, frames, names, imgsz=args.imgsz, batch=args.batch, device=args.device,
                end2end=end2end, nms_iou=nms_iou, max_det=max(args.candidate_limits),
                quantize=quantize,
            )
            for conf in (0.001, *args.confidence):
                for budget in args.candidate_limits:
                    comparisons.append({
                        "checkpoint": selected_checkpoint, "end2end": end2end, "nms_iou": nms_iou,
                        **profile, **_score(frames, predictions, names, conf, budget),
                    })
        del head_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selected_head = _select_head(comparisons)
    selected_runtime = _select_runtime(comparisons, selected_head)
    for end2end in (True, False):
        # Validation fuses under torch.inference_mode(); use a fresh model so its tensors are never reused by prediction.
        metrics = YOLO(selected_checkpoint).val(
            data=str(args.data.resolve()), imgsz=args.imgsz, batch=args.batch, device=args.device,
            end2end=end2end, iou=0.7, conf=0.001, max_det=300, plots=end2end == selected_head,
            project=str(args.output.parent), name=f"{Path(selected_checkpoint).stem}-{'e2e' if end2end else 'nms'}-val",
        )
        ap.append({
            "checkpoint": selected_checkpoint,
            "end2end": end2end,
            "save_dir": str(metrics.save_dir),
            "classes": metrics.summary(),
            "mean": metrics.results_dict,
        })
        del metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    size_comparisons, size_predictions = [], {}
    for runtime_imgsz in RUNTIME_IMAGE_SIZES:
        size_model = YOLO(selected_checkpoint)
        predictions, profile = _predict_targeted(
            size_model, frames, names, imgsz=runtime_imgsz, batch=args.batch,
            device=args.device, end2end=selected_head,
            nms_iou=selected_runtime["nms_iou"], max_det=max(RECALL_LIMITS), quantize=quantize,
        )
        size_predictions[runtime_imgsz] = predictions
        size_comparisons.append({
            "imgsz": runtime_imgsz,
            "end2end": selected_head,
            "nms_iou": selected_runtime["nms_iou"],
            **profile,
            **_score(
                frames, predictions, names, selected_runtime["confidence"],
                selected_runtime["candidate_limit"],
            ),
        })
        del size_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    selected_imgsz = _select(size_comparisons)["imgsz"]
    runtime_settings = {
        key: selected_runtime[key] for key in ("end2end", "nms_iou", "confidence", "candidate_limit")
    }
    runtime_settings["imgsz"] = selected_imgsz
    runtime_settings["quantize"] = quantize
    if runtime_settings["end2end"]:
        runtime_settings["nms_iou"] = None
    selected_predictions = size_predictions[selected_imgsz]
    recall_at = {
        str(limit): _score(
            frames, selected_predictions, names, selected_runtime["confidence"], limit
        )
        for limit in RECALL_LIMITS
    }
    example_frame, example_class_id = next(
        (frame, class_id)
        for frame in frames for class_id, *_ in frame["labels"]
        if any(
            item["class_id"] == class_id and item["conf"] >= selected_runtime["confidence"]
            for item in selected_predictions[frame["image"].name]
        )
    )
    example_candidates = sorted(
        (
            item for item in selected_predictions[example_frame["image"].name]
            if item["class_id"] == example_class_id
            and item["conf"] >= selected_runtime["confidence"]
        ),
        key=lambda item: item["conf"], reverse=True,
    )[:selected_runtime["candidate_limit"]]
    candidate_output_example = {
        "status": "ok" if example_candidates else "not_detected",
        "target_type": names[example_class_id],
        "image": example_frame["image"].name,
        "camera": example_frame["camera"],
        "rgb_shape": list(example_frame["shape"]),
        "candidates": [
            {
                "rank": rank, "class_id": item["class_id"],
                "class_name": names[item["class_id"]], "confidence": item["conf"],
                "xyxy": item["xyxy"],
            }
            for rank, item in enumerate(example_candidates, 1)
        ],
    }
    with Path(selected_checkpoint).open("rb") as checkpoint_stream:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_stream, "sha256").hexdigest()
    dataset_manifest_path = args.data.with_name("dataset_manifest.json")
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest_path.is_file() else {}
    )
    required_limitations = list(dict.fromkeys([
        *dataset_manifest.get("known_limitations", []),
        "The same validation split is used for model selection and readiness reporting.",
        "Human blind review is pending.",
        "This ten-class replacement candidate does not close issue #16 or establish final challenge readiness.",
    ]))

    shared_names = names[:-1]
    baseline_model = YOLO(args.baseline_stage2.resolve())
    baseline_predictions, baseline_profile = _predict_targeted(
        baseline_model, frames, shared_names, imgsz=selected_imgsz, batch=args.batch,
        device=args.device, end2end=selected_head,
        nms_iou=selected_runtime["nms_iou"] or 0.7,
        max_det=selected_runtime["candidate_limit"], quantize=quantize,
    )
    baseline_score = _score(
        frames, baseline_predictions, shared_names,
        selected_runtime["confidence"], selected_runtime["candidate_limit"],
    )
    new_shared_score = _score(
        frames, selected_predictions, shared_names,
        selected_runtime["confidence"], selected_runtime["candidate_limit"],
    )
    regression_classes = {}
    for name in shared_names:
        old, new = baseline_score["classes"][name], new_shared_score["classes"][name]
        recall_delta, f1_delta = new["recall"] - old["recall"], new["f1"] - old["f1"]
        regression_classes[name] = {
            "baseline": old, "candidate": new,
            "recall_delta": recall_delta, "f1_delta": f1_delta,
            "material_regression": recall_delta < -MATERIAL_REGRESSION or f1_delta < -MATERIAL_REGRESSION,
        }
    selected_operational = next(item for item in size_comparisons if item["imgsz"] == selected_imgsz)
    selected_ap = next(item for item in ap if item["end2end"] == selected_head)
    ap_by_name = {item["Class"]: item for item in selected_ap["classes"]}
    umbrella = selected_operational["classes"]["umbrella"]
    umbrella_highlight = {
        **{key: umbrella[key] for key in ("gt", "matched", "predictions", "false_candidates", "precision", "recall", "f1", "by_size")},
        "misses": umbrella["gt"] - umbrella["matched"],
        "ap50": ap_by_name["umbrella"]["mAP50"],
        "ap50_95": ap_by_name["umbrella"]["mAP50-95"],
    }
    runtime_contract = {
        "status": dataset_manifest.get("status", "selected"),
        "checkpoint": selected_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "ordered_classes": names,
        "preprocessing": {
            "input": "full-frame PIL.Image RGB or NumPy uint8 HWC RGB",
            "resize": f"Ultralytics letterbox to imgsz={selected_imgsz}",
            "internal": "RGB HWC to CHW, FP16/FP32, divide by 255",
        },
        "inference": {
            "full_frame": True,
            **runtime_settings,
        },
        "local_checkpoint_load_verified": True,
        "offline_bundle_load_verified": False,
        "known_limitations": required_limitations,
    }
    report = {
        "dataset": str(args.data.resolve()), "imgsz": args.imgsz, "images": len(frames),
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "argv": sys.argv,
        },
        "selection": {
            "checkpoint_rule": "screen every saved checkpoint on validation: macro F1, worst-class recall, tiny recall, small recall, overall recall, then smaller candidate limit; reject zero-recall classes",
            "head_rule": "compare both heads on validation at confidence 0.001/top-20 using the same hierarchy; reject zero-recall classes",
            "runtime_rule": "select confidence, NMS IoU, and top-10/20 on validation using the same hierarchy",
            "resolution_rule": "compare 640/1280/1920 on validation using macro F1, worst-class recall, tiny recall, small recall, then overall recall",
            "checkpoint": selected_checkpoint,
            "runtime": runtime_settings,
        },
        "screening": screening,
        "screening_comparisons": screening_comparisons,
        "ap": ap,
        "comparisons": comparisons,
        "runtime_size_comparisons": size_comparisons,
        "target_conditioned_recall_at": recall_at,
        "umbrella_highlight": umbrella_highlight,
        "baseline_comparison": {
            "checkpoint": str(args.baseline_stage2.resolve()),
            "checkpoint_sha256": _sha256(args.baseline_stage2),
            "ordered_classes": shared_names,
            "settings": runtime_settings,
            "profile": baseline_profile,
            "baseline": baseline_score,
            "candidate": new_shared_score,
            "per_class": regression_classes,
            "material_regression_threshold": MATERIAL_REGRESSION,
            "material_regressions": [name for name, values in regression_classes.items() if values["material_regression"]],
            "umbrella": {"status": "unsupported"},
        },
        "candidate_output_example": candidate_output_example,
        "known_limitations": required_limitations,
        "runtime_contract": runtime_contract,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    contract_path = args.output.with_name("runtime_contract.json")
    contract_path.write_text(json.dumps(runtime_contract, indent=2) + "\n", encoding="utf-8")
    training_record.update({
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "evaluation": report["evaluator"],
        "evaluation_report": str(args.output.resolve()),
        "runtime_contract": str(contract_path.resolve()),
        "selected_validation_plots": next(
            item["save_dir"] for item in ap if item["end2end"] == selected_head
        ),
        "known_limitations": required_limitations,
    })
    args.training_record.write_text(json.dumps(training_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
