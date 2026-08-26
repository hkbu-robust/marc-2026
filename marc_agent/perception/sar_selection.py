"""Query-conditioned SAR candidate evidence, Qwen verification, and fallback."""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional

from marc_agent.command_parser import ParsedCommand
from marc_agent.perception.pose_to_situation import situation_for_pose
from marc_agent.perception.qwen_runtime import (
    QwenRuntimeConfig,
    configured_max_crops,
    load_local_qwen,
)
from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate

_SITUATIONS = frozenset({"normal", "abnormal", "emergency", "accident"})
log = logging.getLogger("marc_agent.sar_selection")


class SelectionStatus(str, Enum):
    SELECTED = "selected"
    NO_MATCH = "no_match"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class CandidateEvidence:
    person: PersonCandidate
    anchor: Optional[ObjectCandidate]
    normalized_distance: Optional[float]
    horizontal_overlap: float
    vertical_gap: Optional[float]
    deterministic_score: float


@dataclass(frozen=True)
class CandidateBoard:
    """Bounded visual evidence and finite candidate IDs for one verifier call."""

    images: tuple[object, ...]
    prompt: str
    evidence: tuple[CandidateEvidence, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.person.candidate_id for item in self.evidence)


@dataclass(frozen=True)
class VerifierMetrics:
    backend: str
    image_count: int
    latency_ms: Optional[float]
    request_bytes: Optional[int] = None


@dataclass(frozen=True)
class SarSelectionResult:
    status: SelectionStatus
    selected: Optional[PersonCandidate]
    requested_situation: Optional[str]
    evidence_situation: Optional[str]
    method: str
    reason: str
    raw_output: Optional[str] = None
    verifier_metrics: Optional[VerifierMetrics] = None


Verifier = Callable[[CandidateBoard], str]


def _verifier_metrics(verifier: Verifier, board: CandidateBoard) -> VerifierMetrics:
    backend = str(getattr(verifier, "method", "qwen"))
    remote = getattr(verifier, "last_metrics", None)
    if remote is not None:
        return VerifierMetrics(
            backend=backend,
            image_count=int(getattr(remote, "image_count", len(board.images))),
            latency_ms=float(getattr(remote, "latency_ms")),
            request_bytes=int(getattr(remote, "request_bytes")),
        )
    local_latency = getattr(verifier, "last_latency_ms", None)
    return VerifierMetrics(
        backend=backend,
        image_count=len(board.images),
        latency_ms=float(local_latency) if local_latency is not None else None,
    )


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _horizontal_overlap(
    person_box: tuple[float, float, float, float],
    anchor_box: tuple[float, float, float, float],
) -> float:
    overlap = max(
        0.0,
        min(person_box[2], anchor_box[2]) - max(person_box[0], anchor_box[0]),
    )
    width = max(
        1.0,
        min(person_box[2] - person_box[0], anchor_box[2] - anchor_box[0]),
    )
    return min(1.0, overlap / width)


def _nearest_anchor(
    person: PersonCandidate,
    objects: list[ObjectCandidate],
    requested_landmark: Optional[str],
) -> tuple[Optional[ObjectCandidate], Optional[float]]:
    same_camera = [item for item in objects if item.camera_id == person.camera_id]
    if requested_landmark:
        same_camera = [
            item for item in same_camera if item.class_name == requested_landmark
        ]
    if not same_camera:
        return None, None

    px, py = _center(person.bbox_xyxy)
    diagonal = max(1.0, math.hypot(*person.image_size))
    ranked = sorted(
        same_camera,
        key=lambda item: (
            math.hypot(
                _center(item.bbox_xyxy)[0] - px,
                _center(item.bbox_xyxy)[1] - py,
            ),
            item.candidate_id,
        ),
    )
    anchor = ranked[0]
    ax, ay = _center(anchor.bbox_xyxy)
    return anchor, math.hypot(ax - px, ay - py) / diagonal


def candidate_evidence(
    command: ParsedCommand,
    round_: CctvDetectionRound,
) -> tuple[CandidateEvidence, ...]:
    """Calculate inspectable pose and person/anchor geometry for every person."""
    requested_situation = command.situation.value if command.situation else None
    evidence: list[CandidateEvidence] = []
    for person in round_.people:
        anchor, distance = _nearest_anchor(person, round_.objects, command.landmark)
        overlap = (
            _horizontal_overlap(person.bbox_xyxy, anchor.bbox_xyxy)
            if anchor
            else 0.0
        )
        vertical_gap = (
            (anchor.bbox_xyxy[1] - person.bbox_xyxy[3]) / max(1.0, person.image_size[1])
            if anchor
            else None
        )
        pose_situation = situation_for_pose(
            person.pose_class,
            support=anchor.class_name if anchor else None,
        )
        score = person.confidence
        if requested_situation and pose_situation == requested_situation:
            score += 2.0
        if command.landmark:
            if anchor is None:
                score -= 1.0
            else:
                score += max(0.0, 1.0 - 3.0 * float(distance))
                score += 0.5 * overlap
                if vertical_gap is not None and abs(vertical_gap) < 0.08:
                    score += 0.35
        evidence.append(
            CandidateEvidence(
                person=person,
                anchor=anchor,
                normalized_distance=distance,
                horizontal_overlap=overlap,
                vertical_gap=vertical_gap,
                deterministic_score=score,
            )
        )
    return tuple(
        sorted(
            evidence,
            key=lambda item: (-item.deterministic_score, item.person.candidate_id),
        )
    )


def _bounded_image(rgb, *, max_side: int):
    from PIL import Image

    image = Image.fromarray(rgb, mode="RGB")
    if max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def _annotated_view(frame, evidence: tuple[CandidateEvidence, ...]):
    from PIL import ImageDraw

    image = _bounded_image(frame.rgb, max_side=960)
    sx = image.width / frame.width
    sy = image.height / frame.height
    draw = ImageDraw.Draw(image)
    for item in evidence:
        person = item.person
        if person.camera_id != frame.camera_id:
            continue
        box = tuple(
            round(value * (sx if index % 2 == 0 else sy))
            for index, value in enumerate(person.bbox_xyxy)
        )
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), person.candidate_id, fill="red")
        if item.anchor:
            anchor_box = tuple(
                round(value * (sx if index % 2 == 0 else sy))
                for index, value in enumerate(item.anchor.bbox_xyxy)
            )
            draw.rectangle(anchor_box, outline="cyan", width=3)
            draw.text(
                (anchor_box[0] + 3, anchor_box[1] + 3),
                item.anchor.candidate_id,
                fill="cyan",
            )
    return image


def _context_crop(frame, item: CandidateEvidence):
    from PIL import ImageDraw

    boxes = [item.person.bbox_xyxy]
    if item.anchor:
        boxes.append(item.anchor.bbox_xyxy)
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad = 0.2 * max(x2 - x1, y2 - y1, 32.0)
    crop_box = (
        max(0, math.floor(x1 - pad)),
        max(0, math.floor(y1 - pad)),
        min(frame.width, math.ceil(x2 + pad)),
        min(frame.height, math.ceil(y2 + pad)),
    )
    image = _bounded_image(
        frame.rgb,
        max_side=max(frame.width, frame.height),
    ).crop(crop_box)
    image.thumbnail((640, 640))
    draw = ImageDraw.Draw(image)
    draw.text((4, 4), item.person.candidate_id, fill="red")
    return image


def format_sar_verifier_prompt(
    command_text: str,
    requested: Optional[str],
    landmark: Optional[str],
    scene_hint: Optional[str],
    rows: Iterable[str],
) -> str:
    """User text for the SAR verifier (training and runtime share this string)."""
    req = requested or "unspecified"
    return (
        "Select the single numbered person candidate that best matches the request. "
        "The request is language to satisfy, not visual ground truth. Use the images and "
        "detector metadata together. Reply with exactly CANDIDATE_ID|SITUATION and nothing "
        "else. SITUATION must be normal, abnormal, emergency, or accident.\n"
        f"Original request: {command_text}\n"
        f"Parsed requested situation: {req}\n"
        f"Parsed requested landmark: {landmark or 'none'}\n"
        f"Parsed scene hint: {scene_hint or 'none'}\n"
        "Candidates:\n" + "\n".join(rows)
    )


def build_candidate_board(
    command_text: str,
    command: ParsedCommand,
    round_: CctvDetectionRound,
    *,
    max_crops: Optional[int] = None,
) -> CandidateBoard:
    """Build bounded visual evidence while keeping every candidate in the prompt."""
    evidence = candidate_evidence(command, round_)
    crop_limit = configured_max_crops() if max_crops is None else max_crops
    if crop_limit < 0:
        raise ValueError("max_crops must be non-negative")
    images: list[object] = []
    for camera_id in sorted({item.person.camera_id for item in evidence}):
        frame = round_.frames.get(camera_id)
        if frame is not None:
            images.append(_annotated_view(frame, evidence))
    for item in evidence[:crop_limit]:
        frame = round_.frames.get(item.person.camera_id)
        if frame is not None:
            images.append(_context_crop(frame, item))

    rows = []
    for item in evidence:
        anchor = item.anchor
        rows.append(
            " | ".join(
                (
                    item.person.candidate_id,
                    f"camera={item.person.camera_id}",
                    f"pose={item.person.pose_class}",
                    f"confidence={item.person.confidence:.3f}",
                    f"anchor={anchor.class_name if anchor else 'none'}",
                    f"anchor_id={anchor.candidate_id if anchor else 'none'}",
                    (
                        f"distance={item.normalized_distance:.4f}"
                        if item.normalized_distance is not None
                        else "distance=none"
                    ),
                    f"overlap={item.horizontal_overlap:.4f}",
                )
            )
        )
    requested = command.situation.value if command.situation else "unspecified"
    prompt = format_sar_verifier_prompt(
        command_text,
        requested,
        command.landmark,
        command.scene_hint,
        rows,
    )
    return CandidateBoard(tuple(images), prompt, evidence)


def parse_verifier_output(
    raw_output: str,
    candidate_ids: tuple[str, ...],
) -> tuple[str, str]:
    """Validate the complete finite Qwen response; no free-form output is accepted."""
    text = (raw_output or "").strip()
    if text.count("|") != 1 or any(char.isspace() for char in text):
        raise ValueError("output must be exactly CANDIDATE_ID|SITUATION")
    candidate_id, situation = text.split("|", 1)
    if candidate_id not in candidate_ids:
        raise ValueError(f"unknown candidate ID: {candidate_id!r}")
    if situation not in _SITUATIONS:
        raise ValueError(f"unknown situation: {situation!r}")
    return candidate_id, situation


class QwenSarVerifier:
    """Lazy local-only Transformers adapter for one candidate-board generation."""

    def __init__(self, config: Optional[QwenRuntimeConfig] = None):
        self.config = config or QwenRuntimeConfig.load()
        self._model = None
        self._processor = None
        self.last_latency_ms: Optional[float] = None

    method = "qwen_local"

    def __call__(self, board: CandidateBoard) -> str:
        started = time.perf_counter()
        try:
            if self._model is None or self._processor is None:
                self._model, self._processor = load_local_qwen(self.config)

            content = [{"type": "image", "image": image} for image in board.images]
            content.append({"type": "text", "text": board.prompt})
            messages = [{"role": "user", "content": content}]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)

            import torch

            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=self.config.max_new_tokens,
                )
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated)
            ]
            text = self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        log.info(
            "[qwen-local] images=%d latency=%.0fms",
            len(board.images),
            self.last_latency_ms,
        )
        return text


def select_sar_candidate(
    command_text: str,
    command: ParsedCommand,
    round_: CctvDetectionRound,
    *,
    verifier: Optional[Verifier] = None,
) -> SarSelectionResult:
    """Select one person with Qwen or the deterministic bounded fallback."""
    requested = command.situation.value if command.situation else None
    if command.target_type != "person" or requested is None:
        return SarSelectionResult(
            SelectionStatus.UNCERTAIN,
            None,
            requested,
            None,
            "none",
            "not_a_complete_sar_request",
        )
    if not round_.people:
        return SarSelectionResult(
            SelectionStatus.NO_MATCH,
            None,
            requested,
            None,
            "none",
            "no_person_candidates",
        )

    board = build_candidate_board(command_text, command, round_)
    metrics = None
    if verifier is not None:
        try:
            raw_output = verifier(board)
            metrics = _verifier_metrics(verifier, board)
            candidate_id, evidence_situation = parse_verifier_output(
                raw_output, board.candidate_ids
            )
            selected = next(
                item.person
                for item in board.evidence
                if item.person.candidate_id == candidate_id
            )
            return SarSelectionResult(
                SelectionStatus.SELECTED,
                selected,
                requested,
                evidence_situation,
                str(getattr(verifier, "method", "qwen")),
                "valid_model_output",
                raw_output,
                metrics,
            )
        except Exception as exc:
            metrics = _verifier_metrics(verifier, board)
            fallback_reason = f"verifier_failed:{type(exc).__name__}"
            raw_output = locals().get("raw_output")
    else:
        fallback_reason = "verifier_not_configured"
        raw_output = None

    best = board.evidence[0]
    return SarSelectionResult(
        SelectionStatus.SELECTED,
        best.person,
        requested,
        situation_for_pose(
            best.person.pose_class,
            support=best.anchor.class_name if best.anchor else None,
        ),
        "deterministic_fallback",
        fallback_reason,
        raw_output,
        metrics,
    )
