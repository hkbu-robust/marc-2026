"""Evaluate Stage 1 checkpoints on both YOLO26 heads and select runtime settings."""

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


CAMERA_RE = re.compile(r"(rig_\d+_[a-z])$")


def _frames(data_yaml: Path) -> tuple[list[str], list[dict]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = [data["names"][key] for key in range(33)]
    root = data_yaml.parent
    images = sorted((root / data["val"]).glob("*.png"))
    frames = []
    for image in images:
        match = CAMERA_RE.search(image.stem)
        if not match:
            raise ValueError(f"cannot identify camera from {image.name}")
        labels = []
        for line in (root / "labels" / "val" / f"{image.stem}.txt").read_text().splitlines():
            class_id, cx, cy, width, height = map(float, line.split())
            labels.append((int(class_id), cx, cy, width, height))
        frames.append({"image": image, "camera": match.group(1), "labels": labels})
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
        height, width = predictions[frame["image"].name][0]["shape"] if predictions[frame["image"].name] else (720, 1280)
        gt = []
        for class_id, cx, cy, box_width, box_height in frame["labels"]:
            pixel_width, pixel_height = box_width * width, box_height * height
            gt.append({
                "class_id": class_id,
                "xyxy": [(cx - box_width / 2) * width, (cy - box_height / 2) * height,
                           (cx + box_width / 2) * width, (cy + box_height / 2) * height],
                "size": "tiny" if pixel_width * pixel_height < 16**2 else "small" if pixel_width * pixel_height < 32**2 else "large",
            })
        selected = [item for item in predictions[frame["image"].name] if item["conf"] >= conf][:budget]
        edges = sorted(
            ((_iou(truth["xyxy"], pred["xyxy"]), gi, pi) for gi, truth in enumerate(gt)
             for pi, pred in enumerate(selected) if truth["class_id"] == pred["class_id"]),
            reverse=True,
        )
        matched_gt, matched_pred = set(), set()
        for overlap, gi, pi in edges:
            if overlap < 0.5:
                break
            if gi not in matched_gt and pi not in matched_pred:
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
        return {**bucket, "recall": recall, "precision": precision}

    class_results = {names[key]: result(value) for key, value in classes.items()}
    recalls = [value["recall"] for value in class_results.values() if value["recall"] is not None]
    precisions = [value["precision"] for value in class_results.values() if value["precision"] is not None]
    macro_recall = statistics.mean(recalls)
    macro_precision = statistics.mean(precisions) if precisions else 0.0
    return {
        "confidence": conf,
        "candidate_limit": budget,
        "gt": totals["gt"],
        "predictions": totals["predictions"],
        "recall": totals["matched"] / totals["gt"],
        "precision": totals["matched"] / totals["predictions"] if totals["predictions"] else 0.0,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": 2 * macro_recall * macro_precision / (macro_recall + macro_precision) if macro_recall + macro_precision else 0.0,
        "classes": class_results,
        "cameras": {key: result(value) for key, value in sorted(cameras.items())},
        "sizes": {key: result(value) for key, value in sizes.items()},
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
    parser.add_argument("--candidate-limits", nargs="+", type=int, default=(10, 20, 30, 50))
    parser.add_argument("--nms-iou", nargs="+", type=float, default=(0.5, 0.7))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if 20 not in args.candidate_limits:
        parser.error("--candidate-limits must include 20 for equal-budget checkpoint selection")
    if any(confidence < 0.001 for confidence in args.confidence):
        parser.error("--confidence values must be at least the 0.001 collection floor")
    if not args.preflight and args.training_record is None:
        parser.error("--training-record is required for final checkpoint provenance")

    import torch
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    names, frames = _frames(args.data.resolve())
    check_det_dataset(str(args.data.resolve()), autodownload=False)
    if len(names) != 33 or not frames:
        raise ValueError("expected a non-empty 33-class validation dataset")
    models = [(checkpoint.resolve(), YOLO(checkpoint.resolve())) for checkpoint in args.checkpoints]
    if args.preflight:
        print(json.dumps({"checkpoints": [str(item[0]) for item in models], "images": len(frames), "names": names}, indent=2))
        return
    training_record = json.loads(args.training_record.read_text(encoding="utf-8"))
    weights_dir = Path(training_record["outputs"]["last"]).resolve().parent
    if any(checkpoint.parent != weights_dir for checkpoint, _ in models):
        raise ValueError(f"all evaluated checkpoints must belong to the recorded run: {weights_dir}")
    for checkpoint, model in models:
        model_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        if model_names != names:
            raise ValueError(f"checkpoint class order does not match the derived dataset: {checkpoint}")

    comparisons, ap = [], []
    paths = [str(frame["image"]) for frame in frames]
    for checkpoint, model in models:
        for end2end in (True, False):
            metrics = model.val(
                data=str(args.data.resolve()), imgsz=args.imgsz, batch=args.batch, device=args.device,
                end2end=end2end, iou=0.7, conf=0.001, max_det=300, plots=False,
                project=str(args.output.parent), name=f"{checkpoint.stem}-{'e2e' if end2end else 'nms'}-val",
            )
            ap.append({"checkpoint": str(checkpoint), "end2end": end2end, "classes": metrics.summary(), "mean": metrics.results_dict})
            for nms_iou in ((0.7,) if end2end else args.nms_iou):
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                started, speeds, predictions = time.perf_counter(), [], {}
                for offset in range(0, len(paths), args.batch):
                    results = model.predict(
                        paths[offset:offset + args.batch], imgsz=args.imgsz, batch=args.batch, device=args.device,
                        conf=0.001, iou=nms_iou, max_det=max(args.candidate_limits), end2end=end2end,
                        quantize=None if str(args.device) == "cpu" else "fp16", verbose=False,
                    )
                    for frame, result in zip(frames[offset:offset + args.batch], results):
                        shape = tuple(result.orig_shape)
                        speeds.append(result.speed["inference"])
                        predictions[frame["image"].name] = [
                            {"xyxy": box.tolist(), "conf": float(conf), "class_id": int(class_id), "shape": shape}
                            for box, conf, class_id in zip(result.boxes.xyxy.cpu(), result.boxes.conf.cpu(), result.boxes.cls.cpu())
                        ]
                elapsed = time.perf_counter() - started
                for conf in (0.001, *args.confidence):
                    for budget in args.candidate_limits:
                        comparisons.append({
                            "checkpoint": str(checkpoint), "end2end": end2end, "nms_iou": nms_iou,
                            "wall_seconds": elapsed, "mean_inference_ms": statistics.mean(speeds),
                            "peak_inference_memory_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None,
                            **_score(frames, predictions, names, conf, budget),
                        })

    checkpoint_candidates = [
        item for item in comparisons
        if not item["end2end"] and item["nms_iou"] == 0.7
        and item["confidence"] == 0.001 and item["candidate_limit"] == 20
    ]
    selected_checkpoint = max(checkpoint_candidates, key=lambda item: (
        item["macro_recall"],
        min(value["recall"] for value in item["classes"].values() if value["recall"] is not None),
        item["sizes"].get("tiny", {}).get("recall") or 0,
        item["sizes"].get("small", {}).get("recall") or 0,
        min(value["recall"] for value in item["cameras"].values()),
        item["recall"],
    ))["checkpoint"]
    runtime_candidates = [item for item in comparisons if item["checkpoint"] == selected_checkpoint]
    selected_runtime = max(
        runtime_candidates,
        key=lambda item: (item["macro_f1"], item["macro_recall"], item["recall"], not item["end2end"]),
    )
    runtime_settings = {
        key: selected_runtime[key] for key in ("end2end", "nms_iou", "confidence", "candidate_limit")
    }
    if runtime_settings["end2end"]:
        runtime_settings["nms_iou"] = None
    selected_model = YOLO(selected_checkpoint)
    example_result = selected_model.predict(
        paths[0], imgsz=args.imgsz, device=args.device, conf=selected_runtime["confidence"],
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
    runtime_contract = {
        "checkpoint": selected_checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "ordered_classes": names,
        "preprocessing": {
            "input": "full-frame PIL.Image RGB (NumPy callers must provide OpenCV BGR uint8 HWC)",
            "resize": f"Ultralytics letterbox to imgsz={args.imgsz}",
            "internal": "BGR to RGB, HWC to CHW, FP16/FP32, divide by 255",
        },
        "inference": {
            "full_frame": True,
            **runtime_settings,
        },
        "local_checkpoint_load_verified": True,
        "offline_bundle_load_verified": False,
    }
    report = {
        "dataset": str(args.data.resolve()), "imgsz": args.imgsz, "images": len(frames),
        "selection": {
            "checkpoint_rule": "one-to-many top-20 at confidence 0.001/NMS IoU 0.7: macro, worst-class, tiny, small, worst-camera, then overall recall",
            "runtime_rule": "macro F1, then macro and overall recall",
            "checkpoint": selected_checkpoint,
            "runtime": runtime_settings,
        },
        "ap": ap,
        "comparisons": comparisons,
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
