"""Verify and benchmark a staged Stage 1 detector through its runtime seam."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from marc_agent.perception.cctv_yolo import CctvYoloDetector, RuntimeContract
from marc_agent.types import CctvFrame


EXPECTED_IGNORED = ("long_umbrella", "master_chef_can", "wood_block")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _payload(bundle: Path) -> dict:
    return json.loads((bundle / "runtime_contract.json").read_text(encoding="utf-8"))


def _contract(bundle: Path, payload: dict, settings: dict) -> RuntimeContract:
    return RuntimeContract(
        weights=(bundle / "model.pt").resolve(),
        imgsz=int(settings["imgsz"]), confidence=float(settings["confidence"]),
        nms_iou=None if settings.get("nms_iou") is None else float(settings["nms_iou"]),
        candidate_limit=int(settings["candidate_limit"]), end2end=bool(settings["end2end"]),
        use_fp16=str(settings.get("quantize", "")).lower() == "fp16",
        ordered_classes=tuple(payload["ordered_classes"]),
        ignored_classes=tuple(payload["ignored_classes"]),
    )


def _p95(values: list[float]) -> float:
    return sorted(values)[round(0.95 * (len(values) - 1))]


def _run(
    contract: RuntimeContract, rgb: np.ndarray, camera: str, warmup: int, iterations: int
) -> tuple[dict, list, list, list, str]:
    import torch

    detector = CctvYoloDetector(contract)
    detector.load()
    resolved_device = detector.device
    cuda_index = int(resolved_device.removeprefix("cuda:")) if resolved_device != "cpu" else None
    detections = detector._predict_frame(rgb)
    height, width = rgb.shape[:2]
    objects, people = detector.detect({camera: CctvFrame(camera, width, height, rgb)})
    for _ in range(warmup):
        detector._predict_frame(rgb)
    if cuda_index is not None:
        torch.cuda.synchronize(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        detector._predict_frame(rgb)
        if cuda_index is not None:
            torch.cuda.synchronize(cuda_index)
        timings.append((time.perf_counter() - started) * 1000.0)
    benchmark = {
        "warmup_iterations": warmup, "timed_iterations": iterations, "batch": 1,
        "mean_ms": statistics.mean(timings), "median_ms": statistics.median(timings),
        "p95_ms": _p95(timings),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(cuda_index) / 1024**3 if cuda_index is not None else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(cuda_index) / 1024**3 if cuda_index is not None else None,
    }
    detector.unload()
    if cuda_index is not None:
        torch.cuda.empty_cache()
    return benchmark, detections, objects, people, resolved_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-bundle", required=True, type=Path)
    parser.add_argument("--published-bundle", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    os.environ.update({"YOLO_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    import torch
    import ultralytics
    from ultralytics import YOLO as _YOLO

    _ = _YOLO  # Fully load lazy Ultralytics modules before denying subprocesses.

    candidate_payload = _payload(args.candidate_bundle)
    old_payload = _payload(args.published_bundle)
    settings = candidate_payload["inference"]
    descriptor = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    canonical_classes = tuple(str(descriptor["names"][index]) for index in range(len(descriptor["names"])))
    classes = tuple(candidate_payload["ordered_classes"])
    dataset_identity = candidate_payload.get("dataset", {})
    if (
        len(canonical_classes) != 34 or "umbrella" not in canonical_classes
        or tuple(descriptor["ignored_classes"]) != EXPECTED_IGNORED
        or classes != canonical_classes
        or tuple(candidate_payload["ignored_classes"]) != EXPECTED_IGNORED
        or Path(dataset_identity.get("path", "")).resolve() != args.data.resolve()
        or dataset_identity.get("sha256") != _sha256(args.data.resolve())
    ):
        raise ValueError("candidate taxonomy is not the exact corrected 34-class contract")
    if "umbrella" in old_payload["ordered_classes"]:
        raise ValueError("published artifact unexpectedly claims umbrella support")
    candidate_weights = args.candidate_bundle / "model.pt"
    if _sha256(candidate_weights) != candidate_payload["checkpoint_sha256"]:
        raise ValueError("candidate deployment checksum differs from runtime contract")

    def deny_network(event, _args):
        if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo", "subprocess.Popen", "os.system"}:
            raise RuntimeError(f"network/process escape disabled during runtime verification: {event}")

    sys.addaudithook(deny_network)
    socket.setdefaulttimeout(0.01)
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    camera_match = __import__("re").search(r"(rig_\d+_[a-z])$", args.image.stem)
    camera = camera_match.group(1) if camera_match else "metadata-unavailable"
    candidate_contract = _contract(args.candidate_bundle, candidate_payload, settings)
    old_contract = _contract(args.published_bundle, old_payload, settings)
    candidate_benchmark, detections, objects, people, candidate_device = _run(
        candidate_contract, rgb, camera, args.warmup, args.iterations
    )
    old_benchmark, old_detections, _old_objects, _old_people, old_device = _run(
        old_contract, rgb, camera, args.warmup, args.iterations
    )
    confidences = [item[1] for item in detections]
    if not detections:
        raise ValueError("verification frame produced no candidate-output example")
    if confidences != sorted(confidences, reverse=True) or len(detections) > settings["candidate_limit"]:
        raise ValueError("candidate output is not globally confidence-descending and bounded")
    height, width = rgb.shape[:2]
    candidates = []
    for rank, (name, confidence, xyxy) in enumerate(detections, 1):
        class_id = classes.index(name)
        x1, y1, x2, y2 = xyxy
        if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
            raise ValueError("candidate coordinates are not in the original image")
        candidates.append({
            "rank": rank, "class_id": class_id, "class_name": name,
            "confidence": confidence, "xyxy": list(xyxy),
        })
    if len(objects) + len(people) != len(candidates):
        raise ValueError("existing Stage 1 consumer conversion dropped candidates")
    consumer_objects = [dataclasses.asdict(item) for item in objects]
    consumer_people = [dataclasses.asdict(item) for item in people]
    expected_object_fields = {"candidate_id", "class_name", "bbox_xyxy", "confidence", "camera_id", "image_size"}
    expected_person_fields = {"candidate_id", "bbox_xyxy", "pose_class", "confidence", "camera_id", "image_size"}
    if any(set(item) != expected_object_fields for item in consumer_objects):
        raise ValueError("public ObjectCandidate fields differ from the Stage 1 consumer contract")
    if any(set(item) != expected_person_fields for item in consumer_people):
        raise ValueError("public PersonCandidate fields differ from the Stage 1 consumer contract")
    report = {
        "verified": True, "offline": True,
        "network_guard": "Python socket connect/bind/send and subprocess escape denied; offline environment flags set",
        "verifier": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__)), "argv": sys.argv},
        "candidate_bundle": str(args.candidate_bundle.resolve()),
        "candidate_weights_sha256": _sha256(candidate_weights),
        "selected_training_checkpoint_sha256": candidate_payload["selected_training_checkpoint"]["sha256"],
        "published_bundle": str(args.published_bundle.resolve()),
        "published_weights_sha256": _sha256(args.published_bundle / "model.pt"),
        "image": str(args.image.resolve()), "image_shape": [height, width],
        "dataset": str(args.data.resolve()), "dataset_sha256": _sha256(args.data.resolve()),
        "dataset_identity": candidate_payload["dataset"],
        "settings": settings, "ordered_classes": list(classes),
        "ignored_classes": list(EXPECTED_IGNORED),
        "checks": {
            "local_path_load": True, "exact_34_class_order": True, "exact_three_exclusions": True,
            "global_confidence_descending": True, "bounded_candidates": True,
            "original_image_coordinates": True,
            "consumer_contract": "CctvYoloDetector.detect -> ObjectCandidate and PersonCandidate",
            "old_umbrella": "unsupported: published 33-class head cannot emit ordinary umbrella",
        },
        "candidate_output_example": {
            "camera": camera, "image": args.image.name, "image_shape": [height, width],
            "raw_global_candidates": candidates,
            "consumer_output": {"objects": consumer_objects, "people": consumer_people},
        },
        "benchmark": candidate_benchmark, "published_benchmark": old_benchmark,
        "machine": {
            "device": candidate_device, "published_device": old_device,
            "gpu": torch.cuda.get_device_name(int(candidate_device.removeprefix("cuda:"))) if candidate_device != "cpu" else None,
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        },
        "benchmark_delta": {
            "mean_ms": candidate_benchmark["mean_ms"] - old_benchmark["mean_ms"],
            "median_ms": candidate_benchmark["median_ms"] - old_benchmark["median_ms"],
            "p95_ms": candidate_benchmark["p95_ms"] - old_benchmark["p95_ms"],
            "peak_allocated_gib": None if candidate_benchmark["peak_allocated_gib"] is None else candidate_benchmark["peak_allocated_gib"] - old_benchmark["peak_allocated_gib"],
            "peak_reserved_gib": None if candidate_benchmark["peak_reserved_gib"] is None else candidate_benchmark["peak_reserved_gib"] - old_benchmark["peak_reserved_gib"],
        },
        "candidate_count": len(candidates), "published_candidate_count": len(old_detections),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
