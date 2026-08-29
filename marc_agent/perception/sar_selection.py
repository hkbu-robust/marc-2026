"""Query-conditioned SAR candidate evidence, Qwen verification, and fallback."""

from __future__ import annotations

import math
import logging
import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable, Optional

from marc_agent.command_parser import (
    LANDMARK_TYPES,
    SEATING_LANDMARKS,
    ParsedCommand,
    Relation,
)
from marc_agent.perception.camera_gate import scene_classes
from marc_agent.perception.pose_to_situation import situation_for_pose
from marc_agent.perception.qwen_reply import call_verifier, last_parsed
from marc_agent.perception.qwen_remote import pack_camera_evidence
from marc_agent.perception.qwen_runtime import (
    QwenRuntimeConfig,
    configured_max_crops,
    load_local_qwen,
)
from marc_agent.perception.situation_rules import POSE_GROUPS, SUPPORT_CLASSES
from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate

_SITUATIONS = frozenset({"normal", "abnormal", "emergency", "accident"})
_P_ID_RE = re.compile(r"^P(\d+)$", re.IGNORECASE)
SHORTLIST_CAP = 8
_FACE_DOWN_POSES = frozenset(
    {"person_prone", "person_lying_down", "person_bending"}
)
_SEATING_LYING_POSES = (
    (POSE_GROUPS["prostrate"] - frozenset({"person_bending"}))
    | frozenset({"person_reclining"})
)
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
    camera_evidence: tuple[str, ...]
    deterministic_score: float
    world_xy: Optional[tuple[float, float]] = None
    world_dist_m: Optional[float] = None


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
    prompt: Optional[str] = None


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
    allowed_classes: frozenset[str],
) -> tuple[Optional[ObjectCandidate], Optional[float]]:
    same_camera = [item for item in objects if item.camera_id == person.camera_id]
    if not allowed_classes:
        return None, None
    same_camera = [
        item for item in same_camera if item.class_name in allowed_classes
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


def resolve_sar_anchor(
    command: ParsedCommand,
    person: PersonCandidate,
    round_: CctvDetectionRound,
) -> Optional[ObjectCandidate]:
    """Resolve projection anchor after person selection, never before it."""
    classes = (
        frozenset({command.landmark})
        if command.landmark
        else (LANDMARK_TYPES if command.scene_hint else frozenset())
    )
    anchor, _ = _nearest_anchor(person, round_.objects, classes)
    return anchor


_ON_TOP_CUE = re.compile(
    r"\b(?:sitting|sat|climbed|climbing)\b|"
    r"\bon\s+top\s+of\b"
)


def _on_top_non_mount(command: ParsedCommand, command_text: str) -> bool:
    landmark = command.landmark
    if not landmark or landmark not in SUPPORT_CLASSES["non_mount"]:
        return False
    return bool(_ON_TOP_CUE.search(command_text or ""))


def _on_top_bonus(
    person: PersonCandidate,
    anchor: ObjectCandidate,
    overlap: float,
) -> float:
    tcy = _center(person.bbox_xyxy)[1]
    acy = _center(anchor.bbox_xyxy)[1]
    seated = person.pose_class in POSE_GROUPS["seated"]
    above = tcy < acy
    bonus = 6.0 * overlap
    if seated:
        bonus += 8.0
    if above:
        bonus += 4.0
    return bonus


_POSE_WORD_BENDING = re.compile(r"\b(?:bent|bending)\b")
_POSE_WORD_PRONE = re.compile(r"\b(?:face[\s-]*down|prone)\b")
_ON_SUPPORT_CUE = re.compile(
    r"\b(?:on\s+top\s+of|sitting|sat|climbed|climbing|lying|lie)\b"
)


def projection_relation(
    command: ParsedCommand, command_text: str = ""
) -> Optional[Relation]:
    """Internal ON for projecting a person onto a support. Not submitted."""
    if command.relation is not None:
        return command.relation
    if command.target_type != "person":
        return None
    if _on_top_non_mount(command, command_text):
        return Relation.ON
    if command.landmark in SEATING_LANDMARKS and _ON_SUPPORT_CUE.search(
        command_text or ""
    ):
        return Relation.ON
    return None


def _pose_word_bonus(person: PersonCandidate, command: ParsedCommand, text: str) -> float:
    requested = command.situation.value if command.situation else None
    seating = command.landmark in SEATING_LANDMARKS
    bonus = 0.0
    if _POSE_WORD_BENDING.search(text) and person.pose_class == "person_bending":
        bonus += 1.5
    face_down = bool(_POSE_WORD_PRONE.search(text))
    if face_down and seating:
        if person.pose_class in {"person_prone", "person_lying_down"}:
            bonus += 0.8
        elif person.pose_class == "person_bending":
            bonus -= 1.0
    elif face_down or requested == "accident":
        if person.pose_class == "person_prone":
            bonus += 2.0
        elif person.pose_class == "person_lying_down":
            bonus -= 0.5
    return bonus


def _soft_camera_evidence(
    command: ParsedCommand,
    person: PersonCandidate,
    objects: list[ObjectCandidate],
) -> tuple[tuple[str, ...], float]:
    """Return additive camera cues without making any person ineligible."""
    visible = {
        item.class_name for item in objects if item.camera_id == person.camera_id
    }
    evidence: list[str] = []
    score = 0.0
    if command.landmark and command.landmark in visible:
        evidence.append(f"landmark:{command.landmark}")
    for cue in command.place_cues:
        if cue in visible:
            evidence.append(f"place:{cue}")
            score += 0.35
    matches = sorted(visible & scene_classes(command.scene_hint))
    if matches:
        evidence.extend(f"scene:{item}" for item in matches)
        score += min(2.0, 0.5 * len(matches))
    return tuple(evidence), score


def candidate_evidence(
    command: ParsedCommand,
    round_: CctvDetectionRound,
    command_text: str = "",
) -> tuple[CandidateEvidence, ...]:
    """Calculate inspectable pose and person/anchor geometry for every person."""
    requested_situation = command.situation.value if command.situation else None
    on_top = _on_top_non_mount(command, command_text)
    text = command_text or ""
    surface = command.scene_hint
    evidence: list[CandidateEvidence] = []
    for person in round_.people:
        if command.landmark:
            anchor, distance = _nearest_anchor(
                person,
                round_.objects,
                frozenset({command.landmark}),
            )
        else:
            anchor, distance = None, None
        camera_evidence, camera_score = _soft_camera_evidence(
            command,
            person,
            round_.objects,
        )
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
            surface=surface,
        )
        score = person.confidence
        score += camera_score
        if requested_situation and pose_situation == requested_situation:
            score += 2.0
        score += _pose_word_bonus(person, command, text)
        if command.landmark:
            if anchor is None:
                score -= 1.0
            else:
                score += max(0.0, 1.0 - 3.0 * float(distance))
                score += 0.5 * overlap
                if vertical_gap is not None and abs(vertical_gap) < 0.08:
                    score += 0.35
                if on_top:
                    score += _on_top_bonus(person, anchor, overlap)
        elif anchor is not None and distance is not None:
            score += max(0.0, 0.6 - 2.0 * float(distance))
        evidence.append(
            CandidateEvidence(
                person=person,
                anchor=anchor,
                normalized_distance=distance,
                horizontal_overlap=overlap,
                vertical_gap=vertical_gap,
                camera_evidence=camera_evidence,
                deterministic_score=score,
            )
        )
    return tuple(
        sorted(
            evidence,
            key=lambda item: (-item.deterministic_score, item.person.candidate_id),
        )
    )


_POSE_CUE_CLASSES = (
    (re.compile(r"\b(?:bent|bending)\b"), frozenset({"person_bending"})),
    (
        re.compile(r"\b(?:sitting|sat|climbed|climbing)\b"),
        frozenset({"person_sitting"}),
    ),
    (re.compile(r"\b(?:lying)\b"), frozenset({"person_lying_down"})),
    (re.compile(r"\b(?:collapsed|fell|fallen)\b"), POSE_GROUPS["prostrate"]),
    (re.compile(r"\b(?:face[\s-]*down|prone)\b"), frozenset({"person_prone"})),
)


def _named_pose_classes(
    text: str, command: Optional[ParsedCommand] = None
) -> frozenset[str]:
    """YOLO pose classes named in the command. The last cue wins when several appear."""
    chosen: frozenset[str] = frozenset()
    last_end = -1
    last_is_face_down = False
    for pattern, classes in _POSE_CUE_CLASSES:
        for match in pattern.finditer(text or ""):
            if match.end() >= last_end:
                last_end = match.end()
                chosen = classes
                last_is_face_down = classes == frozenset({"person_prone"})
    if command is not None and command.landmark in SEATING_LANDMARKS:
        if last_is_face_down:
            return _FACE_DOWN_POSES
        if chosen & {"person_lying_down", "person_prone"}:
            return _SEATING_LYING_POSES
    return chosen


def _world_xy(projector, camera_id: str, box, pose_class: Optional[str] = None):
    if projector is None:
        return None
    xyz = projector.project_contact(camera_id, box, pose_class=pose_class)
    if xyz is None:
        return None
    return (float(xyz[0]), float(xyz[1]))


def _with_world(item: CandidateEvidence, projector) -> CandidateEvidence:
    if projector is None:
        return item
    person_xy = _world_xy(
        projector,
        item.person.camera_id,
        item.person.bbox_xyxy,
        item.person.pose_class,
    )
    dist = None
    if person_xy is not None and item.anchor is not None:
        anchor_xy = _world_xy(
            projector,
            item.anchor.camera_id,
            item.anchor.bbox_xyxy,
        )
        if anchor_xy is not None:
            dist = math.hypot(person_xy[0] - anchor_xy[0], person_xy[1] - anchor_xy[1])
    return replace(item, world_xy=person_xy, world_dist_m=dist)


def _cap_and_order(items: list[CandidateEvidence]) -> tuple[CandidateEvidence, ...]:
    return tuple(items[:SHORTLIST_CAP])


def shortlist_for_verifier(
    evidence: tuple[CandidateEvidence, ...],
    command: ParsedCommand,
    command_text: str = "",
    projector=None,
) -> tuple[CandidateEvidence, ...]:
    """Keep the strongest evidence without treating detector labels as truth."""
    del command, command_text
    return _cap_and_order([_with_world(item, projector) for item in evidence])


def _fallback_pick(
    shortlist: tuple[CandidateEvidence, ...],
    ranked: tuple[CandidateEvidence, ...],
) -> CandidateEvidence:
    if not shortlist:
        return ranked[0]
    return min(
        shortlist,
        key=lambda item: (
            float("inf") if item.world_dist_m is None else item.world_dist_m,
            -item.deterministic_score,
            item.person.candidate_id,
        ),
    )


def _selected_from_item(
    item: CandidateEvidence,
    requested: Optional[str],
    method: str,
    reason: str,
    raw_output: Optional[str] = None,
    metrics: Optional[VerifierMetrics] = None,
    prompt: Optional[str] = None,
) -> SarSelectionResult:
    return SarSelectionResult(
        SelectionStatus.SELECTED,
        item.person,
        requested,
        situation_for_pose(
            item.person.pose_class,
            support=item.anchor.class_name if item.anchor else None,
        ),
        method,
        reason,
        raw_output,
        metrics,
        prompt,
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
    from PIL import Image, ImageDraw

    x1, y1, x2, y2 = item.person.bbox_xyxy
    side = min(
        max(frame.width, frame.height),
        max(160.0, 2.0 * max(x2 - x1, y2 - y1)),
    )
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    crop_box = (
        max(0, math.floor(cx - side / 2.0)),
        max(0, math.floor(cy - side / 2.0)),
        min(frame.width, math.ceil(cx + side / 2.0)),
        min(frame.height, math.ceil(cy + side / 2.0)),
    )
    image = _bounded_image(
        frame.rgb, max_side=max(frame.width, frame.height)
    ).crop(crop_box)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    sx = 512 / (crop_box[2] - crop_box[0])
    sy = 512 / (crop_box[3] - crop_box[1])
    box = (
        round((x1 - crop_box[0]) * sx),
        round((y1 - crop_box[1]) * sy),
        round((x2 - crop_box[0]) * sx),
        round((y2 - crop_box[1]) * sy),
    )
    draw.rectangle(box, outline="red", width=3)
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
        "Compare these listed people against the original request. Candidates are ordered "
        "by weak detector and scene evidence, but you must verify them visually. Pose and landmark proximity named "
        "in the request beat list order. world_dist_m is metres on the ground to the "
        "parsed landmark when known; it is geometry, not the answer by itself. The "
        "request is language to satisfy, not visual ground truth. Use the images and "
        "detector metadata together. The middle token of each CANDIDATE_ID is the detector "
        "pose (person_sitting, person_lying_down, person_prone, …); prefer the pose that "
        "matches the request. Every listed candidate is eligible; detail_crop=no means "
        "the candidate appears only in its annotated camera view. Each visual= field maps "
        "the row to an exact IMAGE and, when present, its yellow crop tile. Reply with "
        "exactly CANDIDATE_ID and nothing else.\n"
        f"Original request: {command_text}\n"
        f"Parsed requested situation: {req}\n"
        f"Parsed requested landmark: {landmark or 'none'}\n"
        f"Parsed scene hint: {scene_hint or 'none'}\n"
        "Prefer candidates whose camera contains the parsed landmark and any "
        "place cues named in the request.\n"
        "Candidates:\n" + "\n".join(rows)
    )


def _detail_crop_evidence(
    evidence: tuple[CandidateEvidence, ...],
    crop_limit: int,
) -> tuple[CandidateEvidence, ...]:
    """Allocate detail crops camera-first, then fill in shortlist order."""
    if crop_limit <= 0:
        return ()
    selected: list[CandidateEvidence] = []
    selected_ids: set[str] = set()
    covered_cameras: set[str] = set()
    for item in evidence:
        camera_id = item.person.camera_id
        if camera_id in covered_cameras:
            continue
        selected.append(item)
        selected_ids.add(item.person.candidate_id)
        covered_cameras.add(camera_id)
        if len(selected) == crop_limit:
            return tuple(selected)
    for item in evidence:
        if item.person.candidate_id in selected_ids:
            continue
        selected.append(item)
        if len(selected) == crop_limit:
            break
    return tuple(selected)


def build_candidate_board(
    command_text: str,
    command: ParsedCommand,
    round_: CctvDetectionRound,
    *,
    max_crops: Optional[int] = None,
    projector=None,
) -> CandidateBoard:
    """Build bounded images for the command-conditioned shortlist."""
    ranked = candidate_evidence(command, round_, command_text)
    evidence = shortlist_for_verifier(ranked, command, command_text, projector)
    return _assemble_board(command_text, command, round_, evidence, max_crops)


def _assemble_board(
    command_text: str,
    command: ParsedCommand,
    round_: CctvDetectionRound,
    evidence: tuple[CandidateEvidence, ...],
    max_crops: Optional[int],
    pack_cameras: bool = False,
) -> CandidateBoard:
    crop_limit = configured_max_crops() if max_crops is None else max_crops
    if crop_limit < 0:
        raise ValueError("max_crops must be non-negative")
    detail_evidence = _detail_crop_evidence(evidence, crop_limit)
    detail_ids = {item.person.candidate_id for item in detail_evidence}
    aliases = {
        item.person.candidate_id: f"P{index:02d}"
        for index, item in enumerate(evidence, start=1)
    }
    images: list[object] = []
    visual_refs: dict[str, str] = {}
    for camera_id in sorted({item.person.camera_id for item in evidence}):
        frame = round_.frames.get(camera_id)
        if frame is None:
            continue
        image_number = len(images) + 1
        camera_crops = tuple(
            (aliases[item.person.candidate_id], _context_crop(frame, item))
            for item in detail_evidence
            if item.person.camera_id == camera_id
        )
        if pack_cameras:
            images.append(
                pack_camera_evidence(_annotated_view(frame, evidence), camera_crops)
            )
            for item in detail_evidence:
                if item.person.camera_id == camera_id:
                    visual_refs[item.person.candidate_id] = (
                        f"IMAGE_{image_number}:tile={aliases[item.person.candidate_id]}"
                    )
        else:
            images.append(_annotated_view(frame, evidence))
            for item in detail_evidence:
                if item.person.camera_id != camera_id:
                    continue
                images.append(_context_crop(frame, item))
                visual_refs[item.person.candidate_id] = (
                    f"IMAGE_{image_number},detail=IMAGE_{len(images)}"
                )
        for item in evidence:
            if item.person.camera_id == camera_id:
                visual_refs.setdefault(
                    item.person.candidate_id, f"IMAGE_{image_number}:overview"
                )

    rows = []
    for index, item in enumerate(evidence, start=1):
        anchor = item.anchor
        rows.append(
            " | ".join(
                (
                    f"P{index:02d}",
                    item.person.candidate_id,
                    f"camera={item.person.camera_id}",
                    f"pose={item.person.pose_class}",
                    f"confidence={item.person.confidence:.3f}",
                    f"visual={visual_refs.get(item.person.candidate_id, 'none')}",
                    f"anchor={anchor.class_name if anchor else 'none'}",
                    f"anchor_id={anchor.candidate_id if anchor else 'none'}",
                    (
                        "camera_evidence=" + ",".join(item.camera_evidence)
                        if item.camera_evidence
                        else "camera_evidence=none"
                    ),
                    (
                        "detail_crop=yes"
                        if item.person.candidate_id in detail_ids
                        else "detail_crop=no"
                    ),
                    (
                        f"distance={item.normalized_distance:.4f}"
                        if item.normalized_distance is not None
                        else "distance=none"
                    ),
                    (
                        f"world_xy={item.world_xy[0]:.2f},{item.world_xy[1]:.2f}"
                        if item.world_xy is not None
                        else "world_xy=none"
                    ),
                    (
                        f"world_dist_m={item.world_dist_m:.2f}"
                        if item.world_dist_m is not None
                        else "world_dist_m=none"
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
) -> tuple[str, Optional[str]]:
    """Validate a finite candidate ID; tolerate thinking, prose, and a situation suffix."""
    return last_parsed(
        raw_output,
        lambda token: _parse_sar_token(token, candidate_ids),
    )


def _parse_sar_token(
    token: str, candidate_ids: tuple[str, ...]
) -> tuple[str, Optional[str]]:
    text = (token or "").strip()
    if not text:
        raise ValueError("output must be exactly CANDIDATE_ID")
    parts = text.split("|")
    situation = parts[-1] if parts[-1] in _SITUATIONS else None
    raw_id = "|".join(parts[:-1]) if situation else text
    return _resolve_candidate_id(raw_id, candidate_ids), situation


def _resolve_candidate_id(raw_id: str, candidate_ids: tuple[str, ...]) -> str:
    numbered = _P_ID_RE.fullmatch((raw_id or "").strip())
    if numbered:
        index = int(numbered.group(1)) - 1
        if 0 <= index < len(candidate_ids):
            return candidate_ids[index]
    if raw_id in candidate_ids:
        return raw_id
    joined = raw_id.replace("|", ":")
    if joined in candidate_ids:
        return joined
    prefixed = [item for item in candidate_ids if item.startswith(raw_id)]
    unique = set(prefixed)
    if len(unique) == 1:
        return next(iter(unique))
    parts = joined.split(":")
    if len(parts) >= 3:
        camera, index = parts[0], parts[-1]
        matches = [
            item
            for item in candidate_ids
            if item.startswith(camera + ":") and item.endswith(":" + index)
        ]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"unknown candidate ID: {raw_id!r}")


def _chat_template_kwargs() -> dict:
    return {"enable_thinking": False}


class QwenSarVerifier:
    """Lazy local-only Transformers adapter for one candidate-board generation."""

    method = "qwen_local"

    def __init__(self, config: Optional[QwenRuntimeConfig] = None):
        self.config = config or QwenRuntimeConfig.load()
        self._model = None
        self._processor = None
        self.last_latency_ms: Optional[float] = None
        self.budget_timeout_s: Optional[float] = None

    def set_budget_timeout(self, timeout_s: float) -> None:
        self.budget_timeout_s = float(timeout_s)

    def warmup(self) -> None:
        """Load weights once so the first round does not pay the 8B disk cost."""
        if self._model is None or self._processor is None:
            self._model, self._processor = load_local_qwen(self.config)

    def unload(self) -> None:
        """Drop GPU tensors. The next warmup or generate reloads from disk."""
        self._model = None
        self._processor = None

    def __call__(self, board: CandidateBoard) -> str:
        started = time.perf_counter()
        try:
            self.warmup()

            content = [{"type": "image", "image": image} for image in board.images]
            content.append({"type": "text", "text": board.prompt})
            messages = [{"role": "user", "content": content}]
            template_kw = dict(
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            try:
                inputs = self._processor.apply_chat_template(
                    messages, **template_kw, **_chat_template_kwargs(),
                )
            except TypeError:
                inputs = self._processor.apply_chat_template(messages, **template_kw)
            inputs = inputs.to(self._model.device)

            import torch

            generate_kw = dict(
                do_sample=False,
                max_new_tokens=self.config.max_new_tokens,
            )
            if self.budget_timeout_s is not None and self.budget_timeout_s > 0:
                generate_kw["max_time"] = float(self.budget_timeout_s)
            with torch.inference_mode():
                try:
                    generated = self._model.generate(**inputs, **generate_kw)
                except TypeError:
                    generate_kw.pop("max_time", None)
                    generated = self._model.generate(**inputs, **generate_kw)
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
    projector=None,
) -> SarSelectionResult:
    """Select one person with Qwen or the deterministic bounded fallback."""
    requested = command.situation.value if command.situation else None
    if command.target_type != "person" or requested is None:
        log.info(
            "[vlm] skipped reason=incomplete_sar_request target_type=%s situation=%s",
            command.target_type,
            requested,
        )
        return SarSelectionResult(
            SelectionStatus.UNCERTAIN,
            None,
            requested,
            None,
            "none",
            "not_a_complete_sar_request",
        )
    if not round_.people:
        log.info("[vlm] skipped reason=no_person_candidates")
        return SarSelectionResult(
            SelectionStatus.NO_MATCH,
            None,
            requested,
            None,
            "none",
            "no_person_candidates",
        )

    ranked = candidate_evidence(command, round_, command_text)
    shortlist = shortlist_for_verifier(ranked, command, command_text, projector)
    poses = _named_pose_classes(command_text, command)
    log.info(
        "[vlm] shortlist n=%d of %d poses=%s landmark=%s",
        len(shortlist),
        len(ranked),
        ",".join(sorted(poses)) or "all",
        command.landmark or "none",
    )
    if verifier is None and len(shortlist) <= 1:
        item = shortlist[0] if shortlist else ranked[0]
        reason = "unique_shortlist" if shortlist else "empty_shortlist"
        log.info(
            "[vlm] skipped reason=%s candidate=%s",
            reason,
            item.person.candidate_id,
        )
        board = (
            _assemble_board(command_text, command, round_, shortlist, None)
            if shortlist
            else None
        )
        return _selected_from_item(
            item,
            requested,
            "deterministic_fallback",
            reason,
            prompt=board.prompt if board is not None else None,
        )

    contenders = shortlist if shortlist else ranked
    board = _assemble_board(command_text, command, round_, contenders, None)
    metrics = None
    if verifier is not None:
        try:
            method = str(getattr(verifier, "method", "qwen"))
            board = _assemble_board(
                command_text,
                command,
                round_,
                contenders,
                len(contenders) if method == "molmo_remote" else None,
                pack_cameras=method == "molmo_remote",
            )
            (parsed, raw_output) = call_verifier(
                verifier,
                board,
                lambda text: parse_verifier_output(text, board.candidate_ids),
            )
            candidate_id, evidence_situation = parsed
            metrics = _verifier_metrics(verifier, board)
            selected = next(
                item.person
                for item in contenders
                if item.person.candidate_id == candidate_id
            )
            log.info(
                "[vlm] validated output candidate=%s situation=%s backend=%s output=%s",
                candidate_id,
                evidence_situation,
                str(getattr(verifier, "method", "qwen")),
                " ".join((raw_output or "").split()),
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
                board.prompt,
            )
        except Exception as exc:
            metrics = _verifier_metrics(verifier, board)
            fallback_reason = f"verifier_failed:{type(exc).__name__}"
            raw_output = locals().get("raw_output")
            if raw_output is None:
                raw_output = getattr(exc, "raw_output", None)
            preview = (
                " ".join(raw_output.split()) if isinstance(raw_output, str) else None
            )
            log.warning(
                "[vlm] verifier failed backend=%s error=%s detail=%s output=%s; "
                "using deterministic fallback",
                str(getattr(verifier, "method", "qwen")),
                type(exc).__name__,
                exc,
                preview,
            )
    else:
        fallback_reason = "verifier_not_configured"
        raw_output = None
        log.info("[vlm] verifier not configured; using deterministic fallback")

    best = _fallback_pick(shortlist, ranked)
    return _selected_from_item(
        best,
        requested,
        "deterministic_fallback",
        fallback_reason,
        raw_output,
        metrics,
        board.prompt,
    )
