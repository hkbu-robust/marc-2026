"""Query-conditioned object ranking from YOLO candidates.

Selects one lost-item instance (and an optional same-camera landmark) using a
strict-then-loose scene ladder: relation geometry, then co-presence with the
named landmark, then any target instance. Place language ranks empty bays,
not camera proximity. A unique instance is submitted even beside a person.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from marc_agent.command_parser import (
    LANDMARK_TYPES,
    PARKING_SCENES,
    ParsedCommand,
    Relation,
    implicit_anchor_classes,
)
from marc_agent.perception.camera_gate import apply_camera_gate
from marc_agent.perception.cctv_project import pair_alignment_score
from marc_agent.perception.lost_item_priors import (
    distinct_target_count,
    evidence_for_candidate,
    parked_vehicle_count,
    prior_key,
    stall_is_empty,
)
from marc_agent.perception.qwen_reply import call_verifier, last_parsed
from marc_agent.perception.qwen_runtime import configured_max_crops
from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate

log = logging.getLogger("marc_agent.object_selection")

ObjectVerifier = Callable[["ObjectBoard"], str]
OccupancyAt = Callable[[float, float], Optional[int]]

OBJECT_BOARD_LIMIT = 12
_FLOOR_SPLIT = 0.15
_OFF_ROAD_SCENES = frozenset(PARKING_SCENES | {"park"})
_FLOOR_PLACE_RE = re.compile(
    r"\b(?:on the (?:floor|ground|pavement)|lot floor)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ObjectSelection:
    """One ranked object (and optional landmark) for a lost-item command."""

    target: ObjectCandidate
    anchor: Optional[ObjectCandidate]
    score: float
    method: str = "object_relation"
    raw_output: Optional[str] = None
    prompt: Optional[str] = None
    match: str = "target"


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
        if anchor.class_name == "police_car":
            score += 0.5
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


def _relation_compatible(
    target: ObjectCandidate,
    anchor: ObjectCandidate,
    relation: Optional[Relation],
) -> bool:
    """True when image geometry actually matches the parsed relation."""
    overlap = _horizontal_overlap(target.bbox_xyxy, anchor.bbox_xyxy)
    tcx, tcy = _center(target.bbox_xyxy)
    acx, acy = _center(anchor.bbox_xyxy)
    img_w = max(1.0, float(target.image_size[0]))
    img_h = max(1.0, float(target.image_size[1]))
    nx = (tcx - acx) / img_w
    ny = (tcy - acy) / img_h
    dist = math.hypot(nx, ny)
    t_bottom = target.bbox_xyxy[3]
    a_top = anchor.bbox_xyxy[1]
    contact = (t_bottom - a_top) / img_h

    if relation in (Relation.ON, Relation.ABOVE):
        if overlap < 0.2 or tcy >= acy:
            return False
        if relation is Relation.ON:
            return abs(contact) <= 0.25
        gap = (a_top - t_bottom) / img_h
        return gap > 0.0
    if relation in (Relation.NEAR, Relation.BESIDE, None):
        return dist <= 0.55
    if relation is Relation.LEFT:
        return nx < 0.0
    if relation is Relation.RIGHT:
        return nx > 0.0
    if relation is Relation.FRONT:
        return ny > 0.0
    if relation is Relation.BEHIND:
        return ny < 0.0
    return True


def _target_classes(parsed: ParsedCommand) -> frozenset[str]:
    return frozenset({parsed.target_type})


def _near_relation(parsed: ParsedCommand, command_text: str) -> Optional[Relation]:
    del command_text
    return parsed.relation or Relation.NEAR


def _pixel_dist(left: ObjectCandidate, right: ObjectCandidate) -> float:
    lx, ly = _center(left.bbox_xyxy)
    rx, ry = _center(right.bbox_xyxy)
    return math.hypot(lx - rx, ly - ry)


def _infer_nearest_pairs(
    targets: list[ObjectCandidate],
    objects: Iterable[ObjectCandidate],
) -> list[ObjectSelection]:
    """Pair each target with the nearest same-camera landmark (any official class)."""
    landmarks = [
        item for item in objects
        if item.class_name in LANDMARK_TYPES
    ]
    selected: list[ObjectSelection] = []
    for target in targets:
        same = [
            item for item in landmarks
            if item.camera_id == target.camera_id
            and item.candidate_id != target.candidate_id
        ]
        if not same:
            continue
        anchor = min(same, key=lambda item: _pixel_dist(target, item))
        selected.append(
            ObjectSelection(
                target=target,
                anchor=anchor,
                score=_relation_score(target, anchor, Relation.NEAR),
                match="target",
            )
        )
    return selected


def _strict_pairs(
    targets: list[ObjectCandidate],
    landmarks: list[ObjectCandidate],
    relation: Optional[Relation],
) -> list[ObjectSelection]:
    selected: list[ObjectSelection] = []
    for target in targets:
        same = [item for item in landmarks if item.camera_id == target.camera_id]
        for anchor in same:
            if not _relation_compatible(target, anchor, relation):
                continue
            selected.append(
                ObjectSelection(
                    target=target,
                    anchor=anchor,
                    score=_relation_score(target, anchor, relation),
                    match="strict",
                )
            )
    return selected


def _copresent_pairs(
    targets: list[ObjectCandidate],
    landmarks: list[ObjectCandidate],
) -> list[ObjectSelection]:
    selected: list[ObjectSelection] = []
    for target in targets:
        same = [item for item in landmarks if item.camera_id == target.camera_id]
        if not same:
            continue
        anchor = min(same, key=lambda item: _pixel_dist(target, item))
        selected.append(
            ObjectSelection(
                target=target,
                anchor=anchor,
                score=_relation_score(target, anchor, Relation.NEAR),
                match="copresent",
            )
        )
    return selected


def _finish_rank(
    selected: list[ObjectSelection] | tuple[ObjectSelection, ...],
    parsed: ParsedCommand,
    command_text: str,
    projector,
    occupancy_at: Optional[OccupancyAt],
    people: Iterable[PersonCandidate],
    *,
    score_first: bool,
    use_place: bool = False,
    objects: Iterable[ObjectCandidate] = (),
) -> tuple[ObjectSelection, ...]:
    ranked = list(selected)
    apply_priors = distinct_target_count(ranked) >= 2 and not use_place
    evidences = (
        [evidence_for_candidate(item.target, people, projector) for item in ranked]
        if apply_priors
        else []
    )
    demote = apply_priors and any(not item.owned for item in evidences)
    by_id = {id(item): evidence for item, evidence in zip(ranked, evidences)}
    object_list = list(objects)
    people_list = list(people)

    def prior(item: ObjectSelection) -> tuple:
        if not apply_priors:
            return (0, 0.0, 0.0)
        return prior_key(by_id[id(item)], demote_owned=demote)

    def ids(item: ObjectSelection) -> tuple:
        return (
            item.target.candidate_id,
            item.anchor.candidate_id if item.anchor else "",
        )

    def place_parts(item: ObjectSelection) -> tuple[float, float]:
        place, floor, _, _ = _place_features(
            item, parsed, command_text, projector, occupancy_at
        )
        return place, floor

    def stall_rank(item: ObjectSelection) -> int:
        return 0 if stall_is_empty(
            item.target, people_list, object_list, projector
        ) else 1

    def lot_rank(item: ObjectSelection) -> int:
        return -parked_vehicle_count(item.target.camera_id, object_list)

    def world_parts(item: ObjectSelection) -> tuple:
        if item.anchor is None:
            return (1, 0.0)
        world = pair_alignment_score(
            projector,
            item.target.camera_id,
            item.target.bbox_xyxy,
            item.anchor.bbox_xyxy,
            parsed.relation or Relation.NEAR,
            item.anchor.class_name,
        )
        if world is None:
            return (1, 0.0)
        return (0, -world)

    if use_place:
        ranked.sort(
            key=lambda item: (
                stall_rank(item),
                -place_parts(item)[0],
                lot_rank(item),
                -place_parts(item)[1],
                -item.score,
                *ids(item),
            )
        )
    elif projector is not None and any(item.anchor is not None for item in ranked):
        ranked.sort(
            key=lambda item: (
                *world_parts(item),
                prior(item),
                -item.score,
                *ids(item),
            )
        )
    elif score_first:
        ranked.sort(
            key=lambda item: (-item.score, prior(item), *ids(item))
        )
    else:
        ranked.sort(
            key=lambda item: (prior(item), -item.score, *ids(item))
        )
    return tuple(ranked)[:OBJECT_BOARD_LIMIT]


def _world_unique_near(
    ranked: tuple[ObjectSelection, ...],
    parsed: ParsedCommand,
    projector,
    command_text: str,
) -> bool:
    """True when the top NEAR pair is clearly closer in world metres."""
    if projector is None or len(ranked) < 2:
        return False
    relation = _near_relation(parsed, command_text)
    if relation not in (Relation.NEAR, Relation.BESIDE, None):
        return False
    scores = []
    for item in ranked[:2]:
        if item.anchor is None:
            return False
        world = pair_alignment_score(
            projector,
            item.target.camera_id,
            item.target.bbox_xyxy,
            item.anchor.bbox_xyxy,
            relation or Relation.NEAR,
            item.anchor.class_name,
        )
        if world is None:
            return False
        scores.append(world)
    return scores[0] >= scores[1] * 1.4 + 1e-9


def occupancy_lookup(grid) -> Optional[OccupancyAt]:
    """Sample occupancy at world XY. 0=free road, 100=occupied, -1=unknown."""
    if grid is None:
        return None
    info = getattr(grid, "info", None)
    data = getattr(grid, "data", None)
    if info is None or data is None:
        return None
    try:
        width = int(info.width)
        height = int(info.height)
        resolution = float(info.resolution)
        origin = info.origin.position
        ox = float(origin.x)
        oy = float(origin.y)
    except (AttributeError, TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return None

    def at(x: float, y: float) -> Optional[int]:
        col = int((float(x) - ox) / resolution)
        row = int((float(y) - oy) / resolution)
        if col < 0 or row < 0 or col >= width or row >= height:
            return None
        index = row * width + col
        if index < 0 or index >= len(data):
            return None
        return int(data[index])

    return at


def _floor_place_request(command_text: str) -> bool:
    return bool(_FLOOR_PLACE_RE.search(command_text or ""))


def _place_only_request(parsed: ParsedCommand, command_text: str) -> bool:
    """True when ranking should stay on occupancy/floor, not a guessed landmark."""
    return parsed.scene_hint is not None or _floor_place_request(command_text)


def _place_board_ranked(
    ranked: tuple[ObjectSelection, ...],
    parsed: ParsedCommand,
    command_text: str,
    round_: CctvDetectionRound,
    projector,
) -> tuple[ObjectSelection, ...]:
    """Keep occupied bays off the verifier board when a vacant stall exists."""
    if not _place_only_request(parsed, command_text):
        return ranked
    empty = tuple(
        item
        for item in ranked
        if stall_is_empty(item.target, round_.people, round_.objects, projector)
    )
    return empty if empty else ranked


def _image_floor_frac(target: ObjectCandidate) -> float:
    height = max(1.0, float(target.image_size[1]))
    return float(target.bbox_xyxy[3]) / height


def _occupancy_place_score(scene_hint: Optional[str], cell: Optional[int]) -> float:
    if cell is None or not scene_hint:
        return 0.0
    if scene_hint in _OFF_ROAD_SCENES:
        if cell < 0:
            return 1.0
        if cell >= 100:
            return 0.3
        return 0.0
    if scene_hint == "road":
        return 1.0 if cell == 0 else 0.0
    return 0.0


def _target_world_xy(projector, target: ObjectCandidate):
    if projector is None:
        return None
    project = getattr(projector, "project_contact", None)
    if not callable(project):
        return None
    xyz = project(target.camera_id, target.bbox_xyxy)
    if xyz is None:
        return None
    return (float(xyz[0]), float(xyz[1]))


def _place_features(
    item: ObjectSelection,
    parsed: ParsedCommand,
    command_text: str,
    projector,
    occupancy_at: Optional[OccupancyAt],
) -> tuple[float, float, Optional[int], Optional[tuple[float, float]]]:
    floor = (
        _image_floor_frac(item.target)
        if _floor_place_request(command_text)
        else 0.0
    )
    world = _target_world_xy(projector, item.target)
    cell = None
    if occupancy_at is not None and world is not None:
        cell = occupancy_at(world[0], world[1])
    place = _occupancy_place_score(parsed.scene_hint, cell)
    return place, floor, cell, world


def _rank_unanchored(
    targets: list[ObjectCandidate],
    parsed: ParsedCommand,
    command_text: str,
    projector,
    occupancy_at: Optional[OccupancyAt],
    people: Iterable[PersonCandidate] = (),
    objects: Iterable[ObjectCandidate] = (),
    *,
    use_place: bool = False,
) -> tuple[ObjectSelection, ...]:
    ranked = [
        ObjectSelection(
            target=item, anchor=None, score=item.confidence, match="target"
        )
        for item in targets
    ]
    return _finish_rank(
        ranked,
        parsed,
        command_text,
        projector,
        occupancy_at,
        people,
        score_first=False,
        use_place=use_place,
        objects=objects,
    )


def _place_unique(
    ranked: tuple[ObjectSelection, ...],
    parsed: ParsedCommand,
    command_text: str,
    projector,
    occupancy_at: Optional[OccupancyAt],
) -> bool:
    """True when occupancy or floor height clearly splits the top two instances."""
    if len(ranked) < 2:
        return False
    if ranked[0].anchor is not None or ranked[1].anchor is not None:
        return False
    place0, floor0, _, _ = _place_features(
        ranked[0], parsed, command_text, projector, occupancy_at
    )
    place1, floor1, _, _ = _place_features(
        ranked[1], parsed, command_text, projector, occupancy_at
    )
    if occupancy_at is not None and place0 > place1 + 1e-9:
        return True
    return _floor_place_request(command_text) and (floor0 - floor1) >= _FLOOR_SPLIT


def rank_object_candidates(
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    *,
    projector=None,
    command_text: str = "",
    occupancy_at: Optional[OccupancyAt] = None,
) -> tuple[ObjectSelection, ...]:
    """Rank matching object (and same-camera landmark) pairs across cameras."""
    gated = apply_camera_gate(parsed, round_)
    targets = [
        item
        for item in gated.objects
        if item.class_name in _target_classes(parsed)
    ]
    if not targets:
        return ()
    people = list(gated.people)
    place_only = _place_only_request(parsed, command_text) and parsed.landmark is None

    def unanchored(subset: list[ObjectCandidate]) -> tuple[ObjectSelection, ...]:
        return _rank_unanchored(
            subset,
            parsed,
            command_text,
            projector,
            occupancy_at,
            people,
            gated.objects,
            use_place=place_only,
        )

    if place_only:
        return unanchored(targets)

    anchor_names = implicit_anchor_classes(parsed, command_text)
    if not anchor_names:
        if _place_only_request(parsed, command_text):
            return _rank_unanchored(
                targets,
                parsed,
                command_text,
                projector,
                occupancy_at,
                people,
                gated.objects,
                use_place=True,
            )
        inferred = _infer_nearest_pairs(targets, gated.objects)
        if not inferred:
            return unanchored(targets)
        anchored_ids = {item.target.candidate_id for item in inferred}
        leftover = [item for item in targets if item.candidate_id not in anchored_ids]
        selected = list(inferred)
        if leftover:
            selected.extend(unanchored(leftover))
        selected.sort(
            key=lambda item: (
                0 if item.anchor is not None else 1,
                -item.score,
                item.target.candidate_id,
                item.anchor.candidate_id if item.anchor else "",
            )
        )
        return _finish_rank(
            selected,
            parsed,
            command_text,
            projector,
            occupancy_at,
            people,
            score_first=False,
            objects=gated.objects,
        )

    relation = parsed.relation or Relation.NEAR
    landmarks = [
        item for item in gated.objects if item.class_name in anchor_names
    ]
    strict = _strict_pairs(targets, landmarks, relation)
    if strict:
        strict.sort(
            key=lambda item: (
                -item.score,
                item.target.candidate_id,
                item.anchor.candidate_id if item.anchor else "",
            )
        )
        return _finish_rank(
            strict,
            parsed,
            command_text,
            projector,
            occupancy_at,
            people,
            score_first=True,
            objects=gated.objects,
        )
    copresent = _copresent_pairs(targets, landmarks)
    if copresent:
        return _finish_rank(
            copresent,
            parsed,
            command_text,
            projector,
            occupancy_at,
            people,
            score_first=False,
            objects=gated.objects,
        )
    return unanchored(targets)


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
    scene_hint: Optional[str] = None,
) -> str:
    """User text for the object-relation verifier."""
    return (
        "Select the single object candidate that best matches the spatial request. "
        "The request is language to satisfy, not visual ground truth. Use the images and "
        "detector metadata together. Do not prefer higher confidence or the first listed "
        "id. When the request names a place (floor, parking lot, empty space, stall, park, "
        "road, walkway), pick the instance in that place. If it names an empty space, lot, "
        "floor, or stall, pick the instance whose stall is vacant (no person and no car "
        "parked in that bay). A closer, larger, or higher-confidence box in an occupied "
        "stall is wrong. Occupancy 0 is drivable road; -1 is off-road "
        "(lot, grass, curb). Parked cars (not a distant taxi) mark a parking-lot "
        "camera. NEAR and BESIDE "
        "mean beside the landmark, not sitting on top of it. When several instances "
        "exist and the request is not a place, prefer an unattended one; items next to "
        "a person are usually theirs. A unique instance is enough even beside a person. "
        "If the object is not on the named support, still pick it when it shares the "
        "landmark's camera. Reply with exactly "
        "TARGET_ID|ANCHOR_ID and nothing else. ANCHOR_ID must be none when the request "
        "has no landmark.\n"
        f"Original request: {command_text}\n"
        f"Parsed target type: {target_type}\n"
        f"Parsed relation: {relation or 'none'}\n"
        f"Parsed landmark: {landmark or 'none'}\n"
        f"Parsed scene hint: {scene_hint or 'none'}\n"
        "Candidates:\n" + "\n".join(rows)
    )


def build_object_board(
    command_text: str,
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    ranked: tuple[ObjectSelection, ...],
    *,
    max_crops: Optional[int] = None,
    projector=None,
    occupancy_at: Optional[OccupancyAt] = None,
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
        fields = [
            item.target.candidate_id,
            f"camera={item.target.camera_id}",
            f"class={item.target.class_name}",
            f"confidence={item.target.confidence:.3f}",
            f"anchor={anchor.class_name if anchor else 'none'}",
            f"anchor_id={anchor.candidate_id if anchor else 'none'}",
            f"score={item.score:.4f}",
            f"match={item.match}",
        ]
        evidence = evidence_for_candidate(item.target, round_.people, projector)
        fields.append(f"owned={'true' if evidence.owned else 'false'}")
        gap = evidence.person_gap
        gap_text = "inf" if not math.isfinite(gap) else f"{gap:.1f}"
        fields.append(f"person_gap={gap_text}")
        fields.append(f"camera_prox={evidence.camera_prox:.4f}")
        empty = stall_is_empty(
            item.target, round_.people, round_.objects, projector
        )
        fields.append(f"stall={'empty' if empty else 'occupied'}")
        _, _, cell, world = _place_features(
            item, parsed, command_text, projector, occupancy_at
        )
        if world is not None:
            fields.append(f"world_xy={world[0]:.2f},{world[1]:.2f}")
        if occupancy_at is not None:
            occ = "missing" if cell is None else str(int(cell))
            fields.append(f"occupancy={occ}")
        rows.append(" | ".join(fields))
    relation = parsed.relation.value if parsed.relation else None
    if relation is None and any(item.anchor is not None for item in ranked):
        relation = Relation.NEAR.value
    prompt = format_object_verifier_prompt(
        command_text,
        parsed.target_type,
        relation,
        parsed.landmark,
        rows,
        scene_hint=parsed.scene_hint,
    )
    return ObjectBoard(tuple(images), prompt, ranked)


def parse_object_verifier_output(
    raw_output: str,
    ranked: tuple[ObjectSelection, ...],
) -> tuple[str, Optional[str]]:
    """Validate a finite object reply against the ranked pair list.

    Accepts ``TARGET_ID|ANCHOR_ID``, a bare ``TARGET_ID`` (uses the best ranked
    pair for that target), or a unique truncated prefix such as ``rig_3_c:tumbler:``.
    Thinking traces and extra prose are ignored; the last valid token wins.
    """
    return last_parsed(
        raw_output,
        lambda token: _parse_object_token(token, ranked),
    )


def _parse_object_token(
    text: str, ranked: tuple[ObjectSelection, ...]
) -> tuple[str, Optional[str]]:
    text = (text or "").strip()
    if not text:
        raise ValueError("output must be exactly TARGET_ID|ANCHOR_ID")
    allowed = {_pair_key(item) for item in ranked}
    if "|" in text:
        if text.count("|") != 1:
            raise ValueError("output must be exactly TARGET_ID|ANCHOR_ID")
        target_id, anchor_id = text.split("|", 1)
        resolved_anchor = None if anchor_id in {"", "none"} else anchor_id
        if (target_id, resolved_anchor) not in allowed:
            if resolved_anchor is None:
                return _resolve_target_only(target_id, ranked)
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
    occupancy_at: Optional[OccupancyAt] = None,
) -> Optional[ObjectSelection]:
    """Return the best matching object, optionally verified, or None."""
    ranked = rank_object_candidates(
        parsed,
        round_,
        projector=projector,
        command_text=command_text,
        occupancy_at=occupancy_at,
    )
    if not ranked:
        return None
    geometry = ranked[0]
    skip_reason = "ok"
    should_verify = False
    inferred = parsed.relation is None
    place = _place_only_request(parsed, command_text)
    board_ranked = ranked
    if verifier is None:
        skip_reason = "no_verifier"
    elif place:
        board_ranked = _place_board_ranked(
            ranked, parsed, command_text, round_, projector
        )
        if distinct_target_count(board_ranked) >= 2:
            should_verify = True
        elif distinct_target_count(ranked) >= 2:
            skip_reason = "unique_empty_stall"
        else:
            skip_reason = "single_pair"
    elif inferred and (
        len(ranked) >= 2 or geometry.anchor is not None
    ):
        should_verify = True
    elif len(ranked) < 2:
        skip_reason = "single_pair"
    elif _world_unique_near(ranked, parsed, projector, command_text):
        skip_reason = "world_unique_near"
    elif _place_unique(
        ranked, parsed, command_text, projector, occupancy_at
    ):
        skip_reason = "place_unique"
    else:
        should_verify = True
    log.info(
        "[object-selection] pairs=%d classes=%s verify=%s reason=%s "
        "geometry=%s occupancy=%s",
        len(ranked),
        ",".join(sorted(_target_classes(parsed))),
        should_verify,
        skip_reason,
        geometry.target.candidate_id,
        "ok" if occupancy_at is not None else "missing",
    )
    if not should_verify:
        return geometry

    board = build_object_board(
        command_text,
        parsed,
        round_,
        board_ranked,
        projector=projector,
        occupancy_at=occupancy_at,
    )
    method = str(getattr(verifier, "method", "qwen"))
    try:
        (target_id, anchor_id), raw_output = call_verifier(
            verifier,
            board,
            lambda text: parse_object_verifier_output(text, board_ranked),
        )
        selected = next(
            item
            for item in board_ranked
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
            selected.target,
            selected.anchor,
            selected.score,
            method,
            raw_output,
            board.prompt,
            selected.match,
        )
    except Exception as exc:
        raw_output = locals().get("raw_output")
        if raw_output is None:
            raw_output = getattr(exc, "raw_output", None)
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
