"""Per-round debug folders: images, bbox overlays, command, detections, grounding."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from marc_agent.types import CctvDetectionRound

log = logging.getLogger("marc_agent.round_dump")

_DEFAULT_ROOT = "artifacts/rounds"


def dump_enabled() -> bool:
    raw = (os.environ.get("MARC_ROUND_DUMP") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def dump_root() -> Path:
    return Path(os.environ.get("MARC_ROUND_DUMP_DIR") or _DEFAULT_ROOT)


def round_dir(name: str) -> Path:
    return dump_root() / name


def stage1_dir(round_num: int) -> Path:
    return round_dir("stage1") / f"round-{int(round_num):03d}"


def stage2_dir(concern: Optional[str] = None) -> Path:
    root = round_dir("stage2")
    return root / concern if concern else root


def _write_text(path: Path, text: str) -> None:
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _candidate_row(item) -> dict:
    row = {
        "candidate_id": item.candidate_id,
        "bbox_xyxy": [float(v) for v in item.bbox_xyxy],
        "confidence": float(item.confidence),
        "camera_id": item.camera_id,
        "image_size": list(item.image_size),
    }
    class_name = getattr(item, "class_name", None)
    if class_name is not None:
        row["class_name"] = class_name
    pose = getattr(item, "pose_class", None)
    if pose is not None:
        row["pose_class"] = pose
    return row


def detections_payload(detections: CctvDetectionRound) -> dict:
    return {
        "camera_ids": list(detections.camera_ids),
        "frames_loaded": sorted(detections.frames),
        "objects": [_candidate_row(item) for item in detections.objects],
        "people": [_candidate_row(item) for item in detections.people],
        "metrics": asdict(detections.metrics) if detections.metrics is not None else None,
    }


def _draw_boxes(rgb, width: int, height: int, boxes: list[tuple]) -> object:
    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    sx = image.width / max(1, int(width))
    sy = image.height / max(1, int(height))
    for bbox, color, label in boxes:
        box = tuple(
            round(value * (sx if index % 2 == 0 else sy))
            for index, value in enumerate(bbox)
        )
        draw.rectangle(box, outline=color, width=3)
        if label:
            draw.text((box[0] + 3, box[1] + 3), str(label), fill=color)
    return image


def save_camera_frames(
    out_dir: Path,
    detections: CctvDetectionRound,
    *,
    target_id: Optional[str] = None,
    anchor_id: Optional[str] = None,
) -> int:
    if not detections.frames:
        return 0
    try:
        from PIL import Image
    except ImportError:
        log.warning("[round-dump] PIL not installed; cannot save frames")
        return 0
    cam_dir = out_dir / "cameras"
    cam_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    by_cam: dict[str, list] = {}
    for item in detections.objects:
        by_cam.setdefault(item.camera_id, []).append(item)
    for item in detections.people:
        by_cam.setdefault(item.camera_id, []).append(item)
    for camera_id, frame in detections.frames.items():
        raw = cam_dir / f"{camera_id}.png"
        Image.fromarray(frame.rgb, mode="RGB").save(raw)
        boxes = []
        for item in by_cam.get(camera_id, []):
            color = "yellow"
            if item.candidate_id == target_id:
                color = "red"
            elif item.candidate_id == anchor_id:
                color = "cyan"
            elif getattr(item, "pose_class", None) is not None:
                color = "orange"
            boxes.append((item.bbox_xyxy, color, item.candidate_id))
        overlay = _draw_boxes(frame.rgb, frame.width, frame.height, boxes)
        overlay.save(cam_dir / f"{camera_id}_boxes.png")
        saved += 1
    return saved


def _grounding_payload(grounding) -> Optional[dict]:
    if grounding is None:
        return None
    return {
        "camera_id": getattr(grounding, "camera_id", None),
        "target_type": getattr(grounding, "target_type", None),
        "target_coord": getattr(grounding, "target_coord", None),
        "anchor_coord": getattr(grounding, "anchor_coord", None),
        "landmark": getattr(grounding, "landmark", None),
        "relation": getattr(grounding, "relation", None),
        "situation": getattr(grounding, "situation", None),
    }


def _save_qwen_artifacts(out_dir: Path, object_sel, sar) -> None:
    qwen_bits = _qwen_transcript(object_sel, sar)
    qwen_path = out_dir / "qwen.txt"
    if qwen_bits:
        _write_text(qwen_path, qwen_bits)
    elif qwen_path.exists():
        qwen_path.unlink()
    for pattern in ("qwen_object_*.png", "qwen_sar_*.png"):
        for stale in out_dir.glob(pattern):
            stale.unlink()
    for label, result in (("object", object_sel), ("sar", sar)):
        for index, image in enumerate(
            getattr(result, "evidence_images", ()) if result is not None else (),
            start=1,
        ):
            image.save(out_dir / f"qwen_{label}_{index:02d}.png")


def dump_stage_artifacts(out_dir: Path, round_result) -> Optional[Path]:
    """Save only image/text evidence; structured facts belong in events.jsonl."""
    if not dump_enabled():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = getattr(round_result, "candidates", None)
    detections = getattr(candidates, "detections", CctvDetectionRound())
    _save_qwen_artifacts(
        out_dir,
        getattr(candidates, "object_selection", None),
        getattr(candidates, "sar_selection", None),
    )
    saved = save_camera_frames(
        out_dir,
        detections,
        target_id=getattr(round_result, "target_id", None),
        anchor_id=getattr(round_result, "anchor_id", None),
    )
    log.info("[round-dump] %s cameras=%d -> %s", out_dir.name, saved, out_dir.resolve())
    return out_dir


def dump_stage_round(
    out_dir: Path,
    round_result,
    *,
    extra: Optional[dict] = None,
) -> Optional[Path]:
    """Write one Stage 1/2 pipeline result into ``out_dir``."""
    if not dump_enabled():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = getattr(round_result, "candidates", None)
    command = getattr(candidates, "command_text", "") if candidates is not None else ""
    parsed = getattr(candidates, "parsed", None) if candidates is not None else None
    detections = (
        getattr(candidates, "detections", CctvDetectionRound())
        if candidates is not None
        else CctvDetectionRound()
    )
    _write_text(out_dir / "command.txt", command or "")
    _write_json(out_dir / "parsed.json", asdict(parsed) if parsed is not None else None)
    _write_json(out_dir / "detections.json", detections_payload(detections))
    grounding = getattr(round_result, "grounding", None)
    _write_json(
        out_dir / "grounding.json",
        {
            "submitted": _grounding_payload(grounding),
            "failure_reason": getattr(round_result, "failure_reason", None)
            or (getattr(candidates, "failure_reason", None) if candidates else None),
            "method": getattr(round_result, "method", None),
            "target_id": getattr(round_result, "target_id", None),
            "anchor_id": getattr(round_result, "anchor_id", None),
        },
    )
    selection = {
        "selected_candidate_id": (
            candidates.selected_candidate_id if candidates is not None else None
        ),
        "method": getattr(round_result, "method", None),
        "failure_reason": getattr(round_result, "failure_reason", None),
        "total_ms": getattr(candidates, "total_ms", None) if candidates else None,
    }
    object_sel = getattr(candidates, "object_selection", None) if candidates else None
    if object_sel is not None:
        selection["object"] = {
            "target_id": object_sel.target.candidate_id,
            "anchor_id": object_sel.anchor.candidate_id if object_sel.anchor else None,
            "score": object_sel.score,
            "method": object_sel.method,
        }
    sar = getattr(candidates, "sar_selection", None) if candidates else None
    if sar is not None:
        selection["sar"] = {
            "status": sar.status.value if sar.status is not None else None,
            "method": sar.method,
            "reason": sar.reason,
            "requested_situation": sar.requested_situation,
            "selected_id": sar.selected.candidate_id if sar.selected else None,
        }
    _save_qwen_artifacts(out_dir, object_sel, sar)
    if extra:
        selection["extra"] = extra
    _write_json(out_dir / "summary.json", selection)
    saved = save_camera_frames(
        out_dir,
        detections,
        target_id=getattr(round_result, "target_id", None),
        anchor_id=getattr(round_result, "anchor_id", None),
    )
    log.info("[round-dump] %s cameras=%d -> %s", out_dir.name, saved, out_dir.resolve())
    return out_dir


def _qwen_transcript(object_sel, sar) -> str:
    parts = []
    if object_sel is not None:
        prompt = getattr(object_sel, "prompt", None)
        raw = getattr(object_sel, "raw_output", None)
        if prompt:
            parts.append("### object prompt\n" + str(prompt).rstrip())
        if raw:
            parts.append("### object reply\n" + str(raw).rstrip())
    if sar is not None:
        prompt = getattr(sar, "prompt", None)
        raw = getattr(sar, "raw_output", None)
        if prompt:
            parts.append("### sar prompt\n" + str(prompt).rstrip())
        if raw:
            parts.append("### sar reply\n" + str(raw).rstrip())
    return "\n\n".join(parts)


def dump_json(out_dir: Path, name: str, payload) -> None:
    if not dump_enabled() or out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / name, payload)


def dump_nearfield_frame(
    out_dir: Optional[Path],
    name: str,
    rgb,
    bbox=None,
    *,
    label: str = "",
) -> None:
    if not dump_enabled() or out_dir is None or rgb is None:
        return
    try:
        from PIL import Image
    except ImportError:
        return
    dest = out_dir / "nearfield"
    dest.mkdir(parents=True, exist_ok=True)
    if bbox is not None:
        h, w = rgb.shape[:2]
        image = _draw_boxes(rgb, w, h, [(tuple(bbox), "red", label or "yolo")])
    else:
        image = Image.fromarray(rgb, mode="RGB")
    image.save(dest / name)


def maybe_dump_cctv_round(round_: CctvDetectionRound, *, round_num: int) -> Path | None:
    """Backward-compatible raw-frame dump; prefer dump_stage_round."""
    if not dump_enabled():
        raw = (os.environ.get("MARC_DUMP_CCTV") or "").strip().lower()
        if raw not in {"1", "true", "yes", "on"}:
            return None
    out_dir = stage1_dir(round_num)
    save_camera_frames(out_dir, round_)
    return out_dir
