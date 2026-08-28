"""Verify and benchmark the packaged Stage 2 detector through its runtime seam."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from demo.manip_dataset.yolo_detect import MAX_CANDIDATES, YoloConfig, YoloTracker
from demo.nearfield import YoloDetector, _candidate_detection, localize_detection


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["MARC_YOLO_WEIGHTS"] = str(args.weights.resolve())
    config = YoloConfig.from_env()
    tracker = YoloTracker(config)
    adapter = YoloDetector(tracker)
    if not adapter.load():
        raise RuntimeError(tracker._last_fail)

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    loaded_order = tuple(tracker.names[index] for index in sorted(tracker.names))
    if loaded_order != config.ordered_classes:
        raise ValueError(f"loaded class order differs from contract: {loaded_order}")

    statuses = {}
    for target in config.ordered_classes:
        result = adapter.detect_candidates(rgb, target)
        candidates = result["candidates"]
        if any(item["class_name"] != target for item in candidates):
            raise ValueError(f"non-target candidate returned for {target}")
        confidences = [item["confidence"] for item in candidates]
        if confidences != sorted(confidences, reverse=True):
            raise ValueError(f"candidates are not confidence-descending for {target}")
        if len(candidates) > min(MAX_CANDIDATES, config.candidate_limit):
            raise ValueError(f"candidate limit exceeded for {target}")
        if [item["rank"] for item in candidates] != list(range(1, len(candidates) + 1)):
            raise ValueError(f"candidate ranks are invalid for {target}")
        statuses[target] = {"status": result["status"], "candidates": len(candidates)}

    unsupported = adapter.detect_candidates(rgb, "long_umbrella")
    blank = adapter.detect_candidates(np.zeros_like(rgb), args.target)
    if unsupported["status"] != "unsupported_target" or unsupported["candidates"]:
        raise ValueError("unsupported target did not fail closed")
    if blank["status"] != "not_detected" or blank["candidates"]:
        raise ValueError(f"blank frame did not produce not_detected: {blank['status']}")

    direct = adapter.detect_candidates(rgb, args.target)
    if direct["status"] != "ok":
        raise ValueError(f"representative target was not detected: {direct['status']}")
    tracker.set_target(args.target)
    legacy = adapter.detect(rgb)
    if legacy["status"] != "ok" or legacy["bbox"] != direct["candidates"][0]["xyxy"]:
        raise ValueError("legacy detect() is not the first-candidate wrapper")

    height, width = rgb.shape[:2]
    depth = np.full((height, width), 0.60, dtype=np.float32)
    k = (0.8 * width, 0.8 * width, 0.5 * width, 0.5 * height)
    first = direct["candidates"][0]
    localized = localize_detection(
        first["xyxy"], depth, (height, width), k, (0.0, 0.0, 0.0), None, 0,
        conf=first["confidence"], name=first["class_name"],
    )
    nearfield = _candidate_detection(
        direct, depth, k, (0.0, 0.0, 0.0), localized.world_xy, 0.01,
    )
    if nearfield is None or nearfield["rank"] != first["rank"]:
        raise ValueError("NearFieldAcquire candidate/radius selection rejected the packaged output")

    import torch
    import ultralytics

    for _ in range(args.warmup):
        adapter.detect_candidates(rgb, args.target)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        adapter.detect_candidates(rgb, args.target)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)

    timings_sorted = sorted(timings)
    p95_index = min(len(timings_sorted) - 1, int(0.95 * len(timings_sorted)))
    report = {
        "verified": True,
        "offline": True,
        "weights": str(args.weights.resolve()),
        "weights_sha256": _sha256(args.weights),
        "image": str(args.image.resolve()),
        "target": args.target,
        "runtime_path": "YoloDetector -> YoloTracker.detect_candidates -> NearFieldAcquire candidate/radius selection",
        "ordered_classes": list(loaded_order),
        "settings": {
            "imgsz": config.imgsz,
            "confidence": config.conf,
            "nms_iou": config.nms_iou,
            "end2end": config.end2end,
            "candidate_limit": config.candidate_limit,
            "quantize": config.quantize,
            "batch": 1,
        },
        "checks": {
            "exact_target_only": True,
            "confidence_descending": True,
            "bounded_candidates": True,
            "unsupported_target": unsupported["status"],
            "blank_frame": blank["status"],
            "no_all_class_fallback": True,
            "legacy_first_candidate": True,
            "nearfield_adapter": True,
            "nearfield_candidate_radius_selection": True,
            "per_class_probe": statuses,
        },
        "candidate_output_example": direct,
        "benchmark": {
            "warmup_iterations": args.warmup,
            "timed_iterations": args.iterations,
            "mean_ms": statistics.mean(timings),
            "median_ms": statistics.median(timings),
            "p95_ms": timings_sorted[p95_index],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else None,
        },
        "machine": {
            "device": tracker.device,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
