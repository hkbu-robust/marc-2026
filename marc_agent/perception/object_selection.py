"""Query-conditioned object ranking from YOLO candidates.

Selects one lost-item instance (and an optional same-camera landmark) using
image-space geometry for the eight official relations, then optional world-metre
rerank. An optional zero-shot verifier may pick among two or more ranked pairs;
geometry remains the fallback. A one-pair board is not sent to the verifier.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from marc_agent.command_parser import ParsedCommand, Relation
from marc_agent.perception.cctv_project import pair_alignment_score
from marc_agent.perception.qwen_runtime import configured_max_crops
from marc_agent.types import CctvDetectionRound, ObjectCandidate

log = logging.getLogger("marc_agent.object_selection")

ObjectVerifier = Callable[["ObjectBoard"], str]

# YOLO confuses small drinkware; expand only when a landmark can disambiguate.
TARGET_CLASS_ALIASES = {
    "tumbler": frozenset({"tumbler", "mug", "disposable_cup"}),
}
OBJECT_BOARD_LIMIT = 12
_ALIAS_SCORE_SCALE = 0.85


@dataclass(frozen=True)
class ObjectSelection:
    """One ranked object (and optional landmark) for a lost-item command."""

    target: ObjectCandidate
    anchor: Optional[ObjectCandidate]
    score: float
    method: str = "object_relation"


@dataclass(frozen=True)
class ObjectBoard:
    """Bounded visual evidence for one object-relation verifier call."""

    images: tuple[object, ...]
    prompt: str
    ranked: tuple[ObjectSelection, ...]

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.target.candidate_id for item in self.ranked)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _horizontal_overlap(
    target_box: tuple[float, float, float, float],
    anchor_box: tuple[float, float, float, float],
) -> float:
    overlap = max(
        0.0,
        min(target_box[2], anchor_box[2]) - max(target_box[0], anchor_box[0]),
    )
    width = max(
        1.0,
        min(target_box[2] - target_box[0], anchor_box[2] - anchor_box[0]),
    )
    return min(1.0, overlap / width)


def _relation_score(
    target: ObjectCandidate,
    anchor: ObjectCandidate,
    relation: Optional[Relation],
) -> float:
    overlap = _horizontal_overlap(target.bbox_xyxy, anchor.bbox_xyxy)
    tcx, tcy = _center(target.bbox_xyxy)
    acx, acy = _center(anchor.bbox_xyxy)
    img_w, img_h = float(target.image_size[0]), float(target.image_size[1])
    nx = (tcx - acx) / max(1.0, img_w)
    ny = (tcy - acy) / max(1.0, img_h)
    dist = math.hypot(nx, ny)
    t_bottom = target.bbox_xyxy[3]
    a_top = anchor.bbox_xyxy[1]
    contact = (t_bottom - a_top) / max(1.0, img_h)
    conf = 0.1 * target.confidence

    if relation is None or relation in (Relation.NEAR, Relation.BESIDE):
        score = 1.0 - dist
        if relation is Relation.BESIDE:
            score += 0.4 * (1.0 - min(1.0, abs(ny) * 4.0))
            score -= 0.5 * overlap
        elif relation is Relation.NEAR:
            on_contact = max(0.0, 1.0 - abs(contact) * 8.0)
            score -= 0.8 * overlap * (0.5 + 0.5 * on_contact)
        return score + conf

    if relation in (Relation.ON, Relation.ABOVE):
        score = 1.5 * overlap
        if tcy < acy:
            score += 0.8
        if relation is Relation.ON:
            score += max(0.0, 1.0 - abs(contact) * 8.0)
        else:
            gap = (a_top - t_bottom) / max(1.0, img_h)
            if gap > 0.0:
                score += min(1.0, gap * 6.0)
        return score + conf

    if relation is Relation.LEFT:
        return (0.5 - nx) + 0.3 * (1.0 - min(1.0, abs(ny) * 3.0)) + conf
    if relation is Relation.RIGHT:
        return (nx + 0.5) + 0.3 * (1.0 - min(1.0, abs(ny) * 3.0)) + conf
    if relation is Relation.FRONT:
        return ny + 0.2 * (1.0 - min(1.0, abs(nx) * 2.0)) + conf
    if relation is Relation.BEHIND:
        return -ny + 0.2 * (1.0 - min(1.0, abs(nx) * 2.0)) + conf
    return -dist + conf


def _target_classes(parsed: ParsedCommand) -> frozenset[str]:
    names = frozenset({parsed.target_type})
    if parsed.landmark is None:
        return names
    extra = TARGET_CLASS_ALIASES.get(parsed.target_type)
    return names if extra is None else names | extra


def _class_scale(class_name: str, target_type: str) -> float:
    return 1.0 if class_name == target_type else _ALIAS_SCORE_SCALE


def _rerank_by_world(
    ranked: tuple[ObjectSelection, ...],
    parsed: ParsedCommand,
    projector,
) -> tuple[ObjectSelection, ...]:
    def key(item: ObjectSelection) -> tuple:
        if item.anchor is None:
            return (1, 0.0, -item.score, item.target.candidate_id)
        world = pair_alignment_score(
            projector,
            item.target.camera_id,
            item.target.bbox_xyxy,
            item.anchor.bbox_xyxy,
            parsed.relation,
            item.anchor.class_name,
        )
        if world is None:
            return (1, 0.0, -item.score, item.target.candidate_id)
        return (0, -world, -item.score, item.target.candidate_id)

    return tuple(sorted(ranked, key=key))


def rank_object_candidates(
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    *,
    projector=None,
) -> tuple[ObjectSelection, ...]:
    """Rank matching object (and same-camera landmark) pairs across cameras."""
    targets = [
        item for item in round_.objects if item.class_name in _target_classes(parsed)
    ]
    if not targets:
        return ()

    if not parsed.landmark:
        ranked = [
            ObjectSelection(target=item, anchor=None, score=item.confidence)
            for item in targets
        ]
        ranked.sort(key=lambda item: (-item.score, item.target.candidate_id))
        return tuple(ranked)[:OBJECT_BOARD_LIMIT]

    landmarks = [
        item for item in round_.objects if item.class_name == parsed.landmark
    ]
    selected: list[ObjectSelection] = []
    for target in targets:
        same_camera = [
            item for item in landmarks if item.camera_id == target.camera_id
        ]
        scale = _class_scale(target.class_name, parsed.target_type)
        for anchor in same_camera:
            selected.append(
                ObjectSelection(
                    target=target,
                    anchor=anchor,
                    score=scale * _relation_score(target, anchor, parsed.relation),
                )
            )
    selected.sort(
        key=lambda item: (
            -item.score,
            item.target.candidate_id,
            item.anchor.candidate_id if item.anchor else "",
        )
    )
    ranked = tuple(selected)
    if projector is not None:
        ranked = _rerank_by_world(ranked, parsed, projector)
    return ranked[:OBJECT_BOARD_LIMIT]


def _pair_key(item: ObjectSelection) -> tuple[str, Optional[str]]:
    return (
        item.target.candidate_id,
        item.anchor.candidate_id if item.anchor is not None else None,
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


def _annotated_view(frame, ranked: tuple[ObjectSelection, ...]):
    from PIL import ImageDraw

    image = _bounded_image(frame.rgb, max_side=960)
    sx = image.width / frame.width
    sy = image.height / frame.height
    draw = ImageDraw.Draw(image)
    for item in ranked:
        if item.target.camera_id != frame.camera_id:
            continue
        box = tuple(
            round(value * (sx if index % 2 == 0 else sy))
            for index, value in enumerate(item.target.bbox_xyxy)
        )
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), item.target.candidate_id, fill="red")
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


def _context_crop(frame, item: ObjectSelection):
    from PIL import ImageDraw

    boxes = [item.target.bbox_xyxy]
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
    draw.text((4, 4), item.target.candidate_id, fill="red")
    return image


def format_object_verifier_prompt(
    command_text: str,
    target_type: str,
    relation: Optional[str],
    landmark: Optional[str],
    rows: Iterable[str],
) -> str:
    """User text for the object-relation verifier."""
    return (
        "Select the single object candidate that best matches the spatial request. "
        "The request is language to satisfy, not visual ground truth. Use the images and "
        "detector metadata together. Similar drinkware may be labeled tumbler, mug, or "
        "disposable_cup; choose the instance that matches the request. NEAR and BESIDE "
        "mean beside the landmark, not sitting on top of it. Reply with exactly "
        "TARGET_ID|ANCHOR_ID and nothing else. ANCHOR_ID must be none when the request "
        "has no landmark.\n"
        f"Original request: {command_text}\n"
        f"Parsed target type: {target_type}\n"
        f"Parsed relation: {relation or 'none'}\n"
        f"Parsed landmark: {landmark or 'none'}\n"
        "Candidates:\n" + "\n".join(rows)
    )


def build_object_board(
    command_text: str,
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    ranked: tuple[ObjectSelection, ...],
    *,
    max_crops: Optional[int] = None,
) -> ObjectBoard:
    """Build bounded visual evidence while keeping every ranked pair in the prompt."""
    crop_limit = configured_max_crops() if max_crops is None else max_crops
    if crop_limit < 0:
        raise ValueError("max_crops must be non-negative")
    images: list[object] = []
    for camera_id in sorted({item.target.camera_id for item in ranked}):
        frame = round_.frames.get(camera_id)
        if frame is not None:
            images.append(_annotated_view(frame, ranked))
    for item in ranked[:crop_limit]:
        frame = round_.frames.get(item.target.camera_id)
        if frame is not None:
            images.append(_context_crop(frame, item))
    rows = []
    for item in ranked:
        anchor = item.anchor
        rows.append(
            " | ".join(
                (
                    item.target.candidate_id,
                    f"camera={item.target.camera_id}",
                    f"class={item.target.class_name}",
                    f"confidence={item.target.confidence:.3f}",
                    f"anchor={anchor.class_name if anchor else 'none'}",
                    f"anchor_id={anchor.candidate_id if anchor else 'none'}",
                    f"score={item.score:.4f}",
                )
            )
        )
    prompt = format_object_verifier_prompt(
        command_text,
        parsed.target_type,
        parsed.relation.value if parsed.relation else None,
        parsed.landmark,
        rows,
    )
    return ObjectBoard(tuple(images), prompt, ranked)


def parse_object_verifier_output(
    raw_output: str,
    ranked: tuple[ObjectSelection, ...],
) -> tuple[str, Optional[str]]:
    """Validate a finite object reply against the ranked pair list.

    Accepts ``TARGET_ID|ANCHOR_ID``, a bare ``TARGET_ID`` (uses the best ranked
    pair for that target), or a unique truncated prefix such as ``rig_3_c:tumbler:``.
    """
    text = (raw_output or "").strip()
    if not text:
        raise ValueError("output must be exactly TARGET_ID|ANCHOR_ID")
    text = text.splitlines()[0].strip().split()[0]
    allowed = {_pair_key(item) for item in ranked}
    if "|" in text:
        if text.count("|") != 1:
            raise ValueError("output must be exactly TARGET_ID|ANCHOR_ID")
        target_id, anchor_id = text.split("|", 1)
        resolved_anchor = None if anchor_id in {"", "none"} else anchor_id
        if (target_id, resolved_anchor) not in allowed:
            raise ValueError(f"unknown pair: {target_id}|{anchor_id}")
        return target_id, resolved_anchor
    return _resolve_target_only(text, ranked)


def _resolve_target_only(
    text: str, ranked: tuple[ObjectSelection, ...]
) -> tuple[str, Optional[str]]:
    exact = [item for item in ranked if item.target.candidate_id == text]
    if not exact:
        prefixed = [
            item for item in ranked if item.target.candidate_id.startswith(text)
        ]
        unique_ids = {item.target.candidate_id for item in prefixed}
        if len(unique_ids) != 1:
            raise ValueError("output must be exactly TARGET_ID|ANCHOR_ID")
        exact = [item for item in prefixed if item.target.candidate_id in unique_ids]
    item = exact[0]
    return (
        item.target.candidate_id,
        item.anchor.candidate_id if item.anchor is not None else None,
    )


def select_object_candidate(
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    *,
    command_text: str = "",
    verifier: Optional[ObjectVerifier] = None,
    projector=None,
) -> Optional[ObjectSelection]:
    """Return the best matching object, optionally verified, or None."""
    ranked = rank_object_candidates(parsed, round_, projector=projector)
    if not ranked:
        return None
    geometry = ranked[0]
    should_verify = verifier is not None and len(ranked) >= 2
    log.info(
        "[object-selection] pairs=%d classes=%s verify=%s geometry=%s",
        len(ranked),
        ",".join(sorted(_target_classes(parsed))),
        should_verify,
        geometry.target.candidate_id,
    )
    if not should_verify:
        return geometry

    board = build_object_board(command_text, parsed, round_, ranked)
    method = str(getattr(verifier, "method", "qwen"))
    try:
        raw_output = verifier(board)
        target_id, anchor_id = parse_object_verifier_output(raw_output, ranked)
        selected = next(
            item
            for item in ranked
            if _pair_key(item) == (target_id, anchor_id)
        )
        log.info(
            "[qwen] object verified target=%s anchor=%s backend=%s output=%s",
            target_id,
            anchor_id or "none",
            method,
            " ".join(raw_output.split()),
        )
        return ObjectSelection(
            selected.target, selected.anchor, selected.score, method
        )
    except Exception as exc:
        raw_output = locals().get("raw_output")
        preview = " ".join(raw_output.split()) if isinstance(raw_output, str) else None
        log.warning(
            "[qwen] object verifier failed backend=%s error=%s detail=%s "
            "output=%s; using geometry ranking",
            method,
            type(exc).__name__,
            exc,
            preview,
        )
        return geometry
