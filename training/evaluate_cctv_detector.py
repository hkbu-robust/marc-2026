"""Evaluate YOLO26 checkpoints, heads, settings, and runtime resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


CAMERA_RE = re.compile(r"(rig_\d+_[a-z])$")
RUNTIME_IMAGE_SIZES = (640, 1280, 1920)


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
    totals, classes, cameras, sizes = Counter(), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for frame in frames:
        height, width = frame.get("shape") or (
            predictions[frame["image"].name][0]["shape"]
            if predictions[frame["image"].name] else (720, 1280)
        )
        gt = []
        for class_id, cx, cy, box_width, box_height in frame["labels"]:
            pixel_width, pixel_height = box_width * width, box_height * height
            gt.append({
                "class_id": class_id,
                "xyxy": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                           (cx + box_width / 2) * width, (cy + box_height / 2) * height],
                "size": "tiny" if pixel_width * pixel_height < 16**2 else "small" if pixel_width * pixel_height < 32**2 else "large",
            })
        selected = sorted(
            (item for item in predictions[frame["image"].name] if item["conf"] >= conf),
            key=lambda item: item["conf"], reverse=True,
        )[:budget]
        matched_gt, matched_pred = set(), set()
        for pi, prediction in enumerate(selected):
            matches = [
                (_iou(truth["xyxy"], prediction["xyxy"]), gi)
                for gi, truth in enumerate(gt)
                if gi not in matched_gt and truth["class_id"] == prediction["class_id"]
            ]
            if matches:
                overlap, gi = max(matches)
            else:
                overlap = 0.0
            if overlap >= 0.5:
                matched_gt.add(gi)
                matched_pred.add(pi)
        totals.update(gt=len(gt), predictions=len(selected), matched=len(matched_gt))
        for gi, truth in enumerate(gt):
            for bucket in (classes[truth["class_id"]], cameras[frame["camera"]], sizes[truth["size"]]):
                bucket["gt"] += 1
                bucket["matched"] += gi in matched_gt
        for prediction in selected:
            classes[prediction["class_id"]]["predictions"] += 1
        cameras[frame["camera"]]["predictions"] += len(selected)

    def result(bucket: Counter) -> dict:
        recall = bucket["matched"] / bucket["gt"] if bucket["gt"] else None
        precision = bucket["matched"] / bucket["predictions"] if bucket["predictions"] else None
        f1 = 2 * recall * precision / (recall + precision) if recall is not None and precision is not None and recall + precision else 0.0
        return {**bucket, "recall": recall, "precision": precision, "f1": f1}

    class_results = {names[key]: result(value) for key, value in classes.items()}
    recalls = [value["recall"] for value in class_results.values() if value["recall"] is not None]
    macro_recall = statistics.mean(recalls)
    macro_precision = statistics.mean(value["precision"] or 0.0 for value in class_results.values())
    return {
        "confidence": conf,
        "candidate_limit": budget,
        "gt": totals["gt"],
        "predictions": totals["predictions"],
        "recall": totals["matched"] / totals["gt"],
        "precision": totals["matched"] / totals["predictions"] if totals["predictions"] else 0.0,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": statistics.mean(value["f1"] for value in class_results.values()),
        "classes": class_results,
        "cameras": {key: result(value) for key, value in sorted(cameras.items())},
        "sizes": {key: result(value) for key, value in sizes.items()},
    }


def _recall_rank(item: dict) -> tuple:
    """Rank candidate coverage without letting aggregate recall hide weak classes or cameras."""
    return (
        item["macro_recall"],
        min(value["recall"] for value in item["classes"].values() if value["recall"] is not None),
        item["sizes"].get("tiny", {}).get("recall") or 0,
        item["sizes"].get("small", {}).get("recall") or 0,
        min(value["recall"] for value in item["cameras"].values()),
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
    return record


def _predict(model, frames: list[dict], *, imgsz: int, batch: int, device: str,
             end2end: bool, nms_iou: float, max_det: int) -> tuple[dict, dict]:
    """Run one validation-image inference pass and return reusable predictions plus its profile."""
    import torch

    paths = [str(frame["image"]) for frame in frames]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started, speeds, predictions = time.perf_counter(), [], {}
    for offset in range(0, len(paths), batch):
        results = model.predict(
            paths[offset:offset + batch], imgsz=imgsz, batch=batch, device=device,
            conf=0.001, iou=nms_iou, max_det=max_det, end2end=end2end,
            quantize=None if str(device) == "cpu" else "fp16", verbose=False,
        )
        for frame, result in zip(frames[offset:offset + batch], results):
            shape = tuple(result.orig_shape)
            speeds.append(result.speed["inference"])
            predictions[frame["image"].name] = [
                {"xyxy": box.tolist(), "conf": float(conf), "class_id": int(class_id), "shape": shape}
                for box, conf, class_id in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu())
            ]
    return predictions, {
        "wall_seconds": time.perf_counter() - started,
        "mean_inference_ms": statistics.mean(speeds),
        "peak_inference_memory_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-record", type=Path, help="training_record.json to finalize")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", nargs="+", type=float, default=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3))
    parser.add_argument("--candidate-limits", nargs="+", type=int, default=(10, 20))
    parser.add_argument("--nms-iou", nargs="+", type=float, default=(0.5, 0.7))
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
    check_det_dataset(str(args.data.resolve()), autodownload=False)
    if not names or not frames:
        raise ValueError("expected a non-empty validation dataset")
    checkpoints = [checkpoint.resolve() for checkpoint in args.checkpoints]
    if args.preflight:
        for checkpoint in checkpoints:
            model = YOLO(checkpoint)
            model_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
            if model_names != names:
                raise ValueError(f"checkpoint class order does not match the derived dataset: {checkpoint}")
        print(json.dumps({"checkpoints": [str(item) for item in checkpoints], "images": len(frames), "names": names}, indent=2))
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
        predictions, profile = _predict(
            model, frames, imgsz=args.imgsz, batch=args.batch, device=args.device,
            end2end=False, nms_iou=0.7, max_det=max(args.candidate_limits),
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
        screening.append(_best_available(checkpoint_comparisons))

    selected_checkpoint = _select(screening)["checkpoint"]
    comparisons, ap = [], []
    for end2end in (True, False):
        # Prediction fuses the selected head; a fresh model keeps the alternate branch available.
        head_model = YOLO(selected_checkpoint)
        for nms_iou in ((0.7,) if end2end else args.nms_iou):
            predictions, profile = _predict(
                head_model, frames, imgsz=args.imgsz, batch=args.batch, device=args.device,
                end2end=end2end, nms_iou=nms_iou, max_det=max(args.candidate_limits),
            )
            for conf in (0.001, *args.confidence):
                for budget in args.candidate_limits:
                    comparisons.append({
                        "checkpoint": selected_checkpoint, "end2end": end2end, "nms_iou": nms_iou,
                        **profile, **_score(frames, predictions, names, conf, budget),
                    })
    for end2end in (True, False):
        # Validation fuses under torch.inference_mode(); use a fresh model so its tensors are never reused by prediction.
        metrics = YOLO(selected_checkpoint).val(
            data=str(args.data.resolve()), imgsz=args.imgsz, batch=args.batch, device=args.device,
            end2end=end2end, iou=0.7, conf=0.001, max_det=300, plots=False,
            project=str(args.output.parent), name=f"{Path(selected_checkpoint).stem}-{'e2e' if end2end else 'nms'}-val",
        )
        ap.append({"checkpoint": selected_checkpoint, "end2end": end2end, "classes": metrics.summary(), "mean": metrics.results_dict})

    selected_head = _select_head(comparisons)
    selected_runtime = _select_runtime(comparisons, selected_head)
    size_comparisons = []
    for runtime_imgsz in RUNTIME_IMAGE_SIZES:
        predictions, profile = _predict(
            YOLO(selected_checkpoint), frames, imgsz=runtime_imgsz, batch=args.batch,
            device=args.device, end2end=selected_head,
            nms_iou=selected_runtime["nms_iou"], max_det=selected_runtime["candidate_limit"],
        )
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
    selected_imgsz = _select(size_comparisons)["imgsz"]
    runtime_settings = {
        key: selected_runtime[key] for key in ("end2end", "nms_iou", "confidence", "candidate_limit")
    }
    runtime_settings["imgsz"] = selected_imgsz
    if runtime_settings["end2end"]:
        runtime_settings["nms_iou"] = None
    example_result = YOLO(selected_checkpoint).predict(
        str(frames[0]["image"]), imgsz=selected_imgsz, device=args.device, conf=selected_runtime["confidence"],
        iou=selected_runtime["nms_iou"], max_det=selected_runtime["candidate_limit"],
        end2end=selected_runtime["end2end"],
        quantize=None if str(args.device) == "cpu" else "fp16", verbose=False,
    )[0]
    candidate_output_example = {
        "image": frames[0]["image"].name,
        "camera": frames[0]["camera"],
        "image_shape": list(example_result.orig_shape),
        "candidates": [
            {"class_id": int(class_id), "class_name": names[int(class_id)], "confidence": float(conf), "xyxy": box.tolist()}
            for box, conf, class_id in zip(example_result.boxes.xyxy.cpu(), example_result.boxes.conf.cpu(), example_result.boxes.cls.cpu())
        ],
    }
    with Path(selected_checkpoint).open("rb") as checkpoint_stream:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_stream, "sha256").hexdigest()
    dataset_manifest_path = args.data.with_name("dataset_manifest.json")
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest_path.is_file() else {}
    )
    runtime_contract = {
        "status": dataset_manifest.get("status", "selected"),
        "checkpoint": selected_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "ordered_classes": names,
        "preprocessing": {
            "input": "full-frame PIL.Image RGB (NumPy callers must provide OpenCV BGR uint8 HWC)",
            "resize": f"Ultralytics letterbox to imgsz={selected_imgsz}",
            "internal": "BGR to RGB, HWC to CHW, FP16/FP32, divide by 255",
        },
        "inference": {
            "full_frame": True,
            **runtime_settings,
        },
        "local_checkpoint_load_verified": True,
        "offline_bundle_load_verified": False,
        "known_limitations": dataset_manifest.get("known_limitations", []),
    }
    report = {
        "dataset": str(args.data.resolve()), "imgsz": args.imgsz, "images": len(frames),
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
        "candidate_output_example": candidate_output_example,
        "runtime_contract": runtime_contract,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    contract_path = args.output.with_name("runtime_contract.json")
    contract_path.write_text(json.dumps(runtime_contract, indent=2) + "\n", encoding="utf-8")
    training_record.update({
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "evaluation_report": str(args.output.resolve()),
        "runtime_contract": str(contract_path.resolve()),
    })
    args.training_record.write_text(json.dumps(training_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
