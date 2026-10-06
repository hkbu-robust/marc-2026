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
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

from marc_agent.experiment_log import note_decision
from marc_agent.command_parser import (
    LANDMARK_TYPES,
    MISSION_OBJECT_TYPES,
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
    speaker_left_item,
    stall_is_empty,
)
from marc_agent.perception.qwen_reply import call_verifier, last_parsed
from marc_agent.perception.qwen_remote import pack_camera_evidence
from marc_agent.perception.qwen_runtime import configured_max_crops
from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate

log = logging.getLogger("marc_agent.object_selection")

ObjectVerifier = Callable[["ObjectBoard"], str]
OccupancyAt = Callable[[float, float], Optional[int]]

OBJECT_BOARD_LIMIT = 12
OCCLUSION_PROXY_MAX_CONF = 0.003
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
    evidence_images: tuple[object, ...] = ()


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


def _target_candidates(
    parsed: ParsedCommand,
    objects: Iterable[ObjectCandidate],
    *,
    visual_proposals: bool,
) -> list[ObjectCandidate]:
    classes = MISSION_OBJECT_TYPES if visual_proposals else _target_classes(parsed)
    return [item for item in objects if item.class_name in classes]


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
    demote = (
        apply_priors
        and any(not item.owned for item in evidences)
        and not speaker_left_item(command_text)
    )
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

    def target_rank(item: ObjectSelection) -> int:
        return 0 if item.target.class_name == parsed.target_type else 1

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
        relation = parsed.relation or Relation.NEAR
        if relation in (Relation.NEAR, Relation.BESIDE, None):
            dist = _pair_world_dist_m(item, projector)
            if dist is None:
                return (1, 0.0)
            return (0, dist)
        world = pair_alignment_score(
            projector,
            item.target.camera_id,
            item.target.bbox_xyxy,
            item.anchor.bbox_xyxy,
            relation,
            item.anchor.class_name,
        )
        if world is None:
            return (1, 0.0)
        return (0, -world)

    def road_parts(item: ObjectSelection) -> tuple:
        if parsed.scene_hint != "road":
            return (0,)
        place, _floor, _cell, _world = _place_features(
            item, parsed, command_text, projector, occupancy_at
        )
        return (-place,)

    if use_place:
        ranked.sort(
            key=lambda item: (
                target_rank(item),
                stall_rank(item),
                -place_parts(item)[0],
                lot_rank(item),
                -place_parts(item)[1],
                -item.score,
                *ids(item),
            )
        )
    elif projector is not None and any(item.anchor is not None for item in ranked):
        if parsed.scene_hint == "road":
            ranked.sort(
                key=lambda item: (
                    target_rank(item),
                    *road_parts(item),
                    *world_parts(item),
                    -item.score,
                    *ids(item),
                )
            )
        else:
            ranked.sort(
                key=lambda item: (
                    target_rank(item),
                    *world_parts(item),
                    prior(item),
                    -item.score,
                    *ids(item),
                )
            )
    elif score_first:
        ranked.sort(
            key=lambda item: (target_rank(item), -item.score, prior(item), *ids(item))
        )
    else:
        ranked.sort(
            key=lambda item: (target_rank(item), prior(item), -item.score, *ids(item))
        )
    # Keep one row per physical target. The ordering above already chose its
    # best anchor; repeating a target for every anchor only wastes VLM context.
    unique = []
    seen_targets = set()
    for item in ranked:
        target_id = item.target.candidate_id
        if target_id in seen_targets:
            continue
        seen_targets.add(target_id)
        preferred = item
        if parsed.scene_hint in PARKING_SCENES:
            preferred = next(
                (
                    candidate
                    for candidate in ranked
                    if candidate.target.candidate_id == target_id
                    and candidate.anchor is not None
                    and candidate.anchor.class_name == "police_car"
                ),
                item,
            )
        unique.append(preferred)
    return tuple(unique[:OBJECT_BOARD_LIMIT])


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


def _pair_world_dist_m(item: ObjectSelection, projector) -> Optional[float]:
    """Ground metres from target to the paired landmark. None if either end fails."""
    if projector is None or item.anchor is None:
        return None
    target_xy = _target_world_xy(projector, item.target)
    anchor_xy = _target_world_xy(projector, item.anchor)
    if target_xy is None or anchor_xy is None:
        return None
    return math.hypot(target_xy[0] - anchor_xy[0], target_xy[1] - anchor_xy[1])


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


def rank_object_candidates(
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    *,
    projector=None,
    command_text: str = "",
    occupancy_at: Optional[OccupancyAt] = None,
    visual_proposals: bool = False,
) -> tuple[ObjectSelection, ...]:
    """Rank matching object (and same-camera landmark) pairs across cameras."""
    gated = (
        apply_camera_gate(parsed, round_)
        if not visual_proposals or (parsed.scene_hint and not parsed.landmark)
        else round_
    )
    targets = _target_candidates(
        parsed, gated.objects, visual_proposals=visual_proposals
    )
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
                item.target.class_name != parsed.target_type,
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
    for index, item in enumerate(ranked, start=1):
        if item.target.camera_id != frame.camera_id:
            continue
        label = f"O{index:02d}"
        box = tuple(
            round(value * (sx if index % 2 == 0 else sy))
            for index, value in enumerate(item.target.bbox_xyxy)
        )
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), label, fill="red")
        if item.anchor:
            anchor_box = tuple(
                round(value * (sx if index % 2 == 0 else sy))
                for index, value in enumerate(item.anchor.bbox_xyxy)
            )
            draw.rectangle(anchor_box, outline="cyan", width=3)
            draw.text(
                (anchor_box[0] + 3, anchor_box[1] + 3),
                f"{label}-A",
                fill="cyan",
            )
    return image


def _context_crop(frame, item: ObjectSelection, label: str = ""):
    from PIL import Image, ImageDraw

    x1, y1, x2, y2 = item.target.bbox_xyxy
    if item.anchor is not None:
        ax1, ay1, ax2, ay2 = item.anchor.bbox_xyxy
        left, top = min(x1, ax1), min(y1, ay1)
        right, bottom = max(x2, ax2), max(y2, ay2)
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        target_side = 3.0 * max(x2 - x1, y2 - y1)
        side = min(
            max(frame.width, frame.height),
            max(
                128.0,
                min(
                    1.35 * max(right - left, bottom - top),
                    target_side,
                ),
            ),
        )
        # Relation stays visible in the annotated overview. Do not let a large
        # or distant anchor turn this target-detail tile into another overview.
        if side < max(right - left, bottom - top):
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    else:
        side = min(
            max(frame.width, frame.height),
            max(128.0, 3.0 * max(x2 - x1, y2 - y1)),
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
    draw.text((4, 4), label or item.target.candidate_id, fill="red")
    if item.anchor is not None:
        anchor_box = (
            round((ax1 - crop_box[0]) * sx),
            round((ay1 - crop_box[1]) * sy),
            round((ax2 - crop_box[0]) * sx),
            round((ay2 - crop_box[1]) * sy),
        )
        draw.rectangle(anchor_box, outline="cyan", width=3)
        draw.text((anchor_box[0] + 3, anchor_box[1] + 3), f"{label}-A", fill="cyan")
    return image


def format_object_verifier_prompt(
    command_text: str,
    target_type: str,
    relation: Optional[str],
    landmark: Optional[str],
    rows: Iterable[str],
    scene_hint: Optional[str] = None,
    choose_anchor: bool = False,
) -> str:
    """User text for the object-relation verifier."""
    task = (
        "The red target is already fixed and is identical in every row. Compare the "
        "different cyan anchors and select the pair whose visible anchor and spatial "
        "context best explain the request. If the request mentions an owner, their car, "
        "or getting out, anchor_owned and anchor_person_gap help identify the vehicle "
        "associated with that person; do not blindly choose the nearest vehicle. "
        if choose_anchor
        else "Select the single object candidate that best matches the spatial request. "
    )
    answer = (
        "Reply with exactly one listed TARGET_ID|ANCHOR_ID pair and nothing else."
        if choose_anchor
        else "Reply with exactly one listed TARGET_ID and nothing else."
    )
    final = (
        "copy the chosen row as TARGET_ID|ANCHOR_ID"
        if choose_anchor
        else "copy only the chosen row's first field (TARGET_ID)"
    )
    return (
        task
        + "YOLO class labels are fallible priors, not facts. Never let an exact class "
        "label override a visibly wrong relation or place. First silently inspect every red "
        "TARGET_ID box and its detail crop. Each visual= field maps the row to an exact "
        "IMAGE and, when present, its yellow crop tile. Discard boxes that do not visually "
        "contain "
        "the parsed target type, including a shoe, body part, background patch, or other "
        "object mislabeled as the target. Also reject a cyan anchor whose visible class "
        "does not match its metadata. Never select a false positive because it has higher "
        "confidence or better geometry. Then choose the best relation among the remaining "
        "visually valid target instances. The request is language to satisfy, not visual "
        "ground truth. Use the images and detector metadata together. Do not prefer the "
        "first listed id. After target identity, relation, landmark, and place all tie, "
        "prefer the higher-confidence detection. When the request names a place (floor, parking "
        "lot, empty space, stall, park, "
        "road, walkway), pick the instance in that place. Only when the request explicitly "
        "says empty space or empty stall, prefer a vacant bay (no person and no parked car). "
        "An empty parking space is a painted parking bay beside a parked vehicle, not a "
        "sidewalk or park edge where a car merely appears in the background. "
        "Parking-lot or floor language alone does not require an empty stall. Stall metadata "
        "is weak derived evidence and must never override visible target identity or relation. "
        "Occupancy 0 is drivable road; -1 is off-road "
        "(lot, grass, curb). Parked cars (not a distant taxi) mark a parking-lot "
        "camera. For dropped, lost, or floor requests, an owned=true candidate touching "
        "a person is strong negative evidence unless the request says it is carried or held. "
        "NEAR and BESIDE "
        "mean beside the landmark, not sitting on top of it. When several instances "
        "exist, decide from the requested target, relation, landmark, and place. "
        "A unique instance is enough even beside a person. "
        "Treat the requested relation and landmark as mandatory. match=strict means the "
        "detector geometry supports that relation; match=copresent is only a fallback when "
        "no candidate visibly satisfies it. For ON or ABOVE, reject any red target visibly "
        f"separate from its cyan support. {answer}\n"
        f"Original request: {command_text}\n"
        f"Parsed target type: {target_type}\n"
        f"Parsed relation: {relation or 'none'}\n"
        f"Parsed landmark: {landmark or 'none'}\n"
        f"Parsed scene hint: {scene_hint or 'none'}\n"
        "Candidates:\n"
        + "\n".join(rows)
        + f"\nFINAL ANSWER: {final}."
    )


def build_object_board(
    command_text: str,
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    ranked: tuple[ObjectSelection, ...],
    *,
    max_crops: Optional[int] = None,
    pack_cameras: bool = False,
    projector=None,
    occupancy_at: Optional[OccupancyAt] = None,
    choose_anchor: bool = False,
) -> ObjectBoard:
    """Build bounded visual evidence while keeping every ranked pair in the prompt."""
    crop_limit = configured_max_crops() if max_crops is None else max_crops
    if crop_limit < 0:
        raise ValueError("max_crops must be non-negative")
    cropped = []
    for index, item in enumerate(ranked, start=1):
        if len(cropped) >= crop_limit:
            break
        if round_.frames.get(item.target.camera_id) is not None:
            cropped.append((f"O{index:02d}", item))

    images: list[object] = []
    visual_refs: dict[str, str] = {}
    for camera_id in sorted({item.target.camera_id for item in ranked}):
        frame = round_.frames.get(camera_id)
        if frame is None:
            continue
        image_number = len(images) + 1
        camera_crops = tuple(
            (label, _context_crop(frame, item, label))
            for label, item in cropped
            if item.target.camera_id == camera_id
        )
        if pack_cameras:
            images.append(
                pack_camera_evidence(_annotated_view(frame, ranked), camera_crops)
            )
            for label, item in cropped:
                if item.target.camera_id == camera_id:
                    visual_refs[_pair_key(item)] = (
                        f"IMAGE_{image_number}:tile={label}"
                    )
        else:
            images.append(_annotated_view(frame, ranked))
            for label, item in cropped:
                if item.target.camera_id != camera_id:
                    continue
                images.append(_context_crop(frame, item, label))
                visual_refs[_pair_key(item)] = (
                    f"IMAGE_{image_number},detail=IMAGE_{len(images)}"
                )
        for item in ranked:
            if item.target.camera_id == camera_id:
                visual_refs.setdefault(
                    _pair_key(item), f"IMAGE_{image_number}:overview"
                )
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
            f"visual={visual_refs.get(_pair_key(item), 'none')}",
        ]
        evidence = evidence_for_candidate(item.target, round_.people, projector)
        fields.append(f"owned={'true' if evidence.owned else 'false'}")
        gap = evidence.person_gap
        gap_text = "inf" if not math.isfinite(gap) else f"{gap:.1f}"
        fields.append(f"person_gap={gap_text}")
        fields.append(f"camera_prox={evidence.camera_prox:.4f}")
        if choose_anchor and anchor is not None:
            anchor_evidence = evidence_for_candidate(
                anchor, round_.people, projector
            )
            anchor_gap = anchor_evidence.person_gap
            anchor_gap_text = (
                "inf" if not math.isfinite(anchor_gap) else f"{anchor_gap:.1f}"
            )
            fields.append(
                f"anchor_owned={'true' if anchor_evidence.owned else 'false'}"
            )
            fields.append(f"anchor_person_gap={anchor_gap_text}")
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
        choose_anchor=choose_anchor,
    )
    return ObjectBoard(tuple(images), prompt, ranked)


def _anchor_variants(
    selected: ObjectSelection,
    parsed: ParsedCommand,
    command_text: str,
    round_: CctvDetectionRound,
) -> tuple[ObjectSelection, ...]:
    """Expose alternate implicit anchors after Qwen has fixed the target."""
    if parsed.landmark is not None:
        return ()
    names = implicit_anchor_classes(parsed, command_text)
    anchors = [
        item
        for item in round_.objects
        if item.camera_id == selected.target.camera_id
        and item.class_name in names
        and item.candidate_id != selected.target.candidate_id
    ]
    if len(anchors) < 2:
        return ()
    relation = parsed.relation or Relation.NEAR
    pairs = _strict_pairs([selected.target], anchors, relation)
    if len(pairs) < 2:
        pairs = _copresent_pairs([selected.target], anchors)
    if len(pairs) < 2:
        return ()
    pairs.sort(key=lambda item: (-item.score, item.anchor.candidate_id))
    return tuple(pairs[:OBJECT_BOARD_LIMIT])


def parse_object_pair_output(
    raw_output: str,
    ranked: tuple[ObjectSelection, ...],
) -> tuple[str, Optional[str]]:
    """Require an exact pair when Qwen is resolving an implicit anchor."""
    allowed = {_pair_key(item) for item in ranked}

    def parse(token: str) -> tuple[str, Optional[str]]:
        if token.count("|") != 1:
            raise ValueError("output must be exactly TARGET_ID|ANCHOR_ID")
        target_id, anchor_id = token.split("|", 1)
        pair = (target_id, None if anchor_id in {"", "none"} else anchor_id)
        if pair not in allowed:
            raise ValueError("output must be one listed TARGET_ID|ANCHOR_ID pair")
        return pair

    return last_parsed(raw_output, parse)


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


def parse_object_target_output(
    raw_output: str,
    ranked: tuple[ObjectSelection, ...],
) -> tuple[str, Optional[str]]:
    """Accept one exact target ID and retain its deterministically ranked anchor."""
    by_target = {item.target.candidate_id: item for item in ranked}

    def parse(token: str) -> tuple[str, Optional[str]]:
        item = by_target.get(token)
        if item is None:
            raise ValueError("output must be exactly one listed TARGET_ID")
        return _pair_key(item)

    return last_parsed(raw_output, parse)


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
            matches = [
                item
                for item in ranked
                if item.target.candidate_id.startswith(target_id)
                and (item.anchor.candidate_id if item.anchor else None) == resolved_anchor
            ]
            if matches:
                return _pair_key(matches[0])
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
        if not prefixed:
            raise ValueError("output must be exactly TARGET_ID|ANCHOR_ID")
        exact = prefixed
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
        visual_proposals=verifier is not None,
    )
    if not ranked:
        return None
    geometry = ranked[0]
    occluded_geometry = next(
        (
            item
            for item in ranked
            if parsed.target_type == "umbrella"
            and item.anchor is not None
            and item.anchor.class_name == "trash_can"
            and item.target.confidence <= 0.05
            and (item.target.bbox_xyxy[2] - item.target.bbox_xyxy[0])
            * (item.target.bbox_xyxy[3] - item.target.bbox_xyxy[1])
            <= 400
            and item.anchor.bbox_xyxy[0]
            <= _center(item.target.bbox_xyxy)[0]
            <= item.anchor.bbox_xyxy[2]
            and item.anchor.bbox_xyxy[1]
            <= _center(item.target.bbox_xyxy)[1]
            <= item.anchor.bbox_xyxy[3]
        ),
        None,
    )
    if occluded_geometry is not None:
        geometry = occluded_geometry
    geometry = next(
        (
            item
            for item in ranked
            if item.target.class_name == parsed.target_type
            and item.target.confidence <= OCCLUSION_PROXY_MAX_CONF
        ),
        geometry,
    )
    parking_geometry = (
        parsed.scene_hint == "parking_space"
        and str(getattr(verifier, "method", "qwen")) == "qwen_remote"
    )
    if verifier is None or parking_geometry:
        log.info(
            "[object-selection] pairs=%d classes=%s verify=False reason=%s "
            "geometry=%s occupancy=%s",
            len(ranked),
            ",".join(sorted(_target_classes(parsed))),
            "parking_geometry" if parking_geometry else "no_verifier",
            geometry.target.candidate_id,
            "ok" if occupancy_at is not None else "missing",
        )
        note_decision(
            "selector", path="object", outcome="geometry_only",
            reason="parking_geometry" if parking_geometry else "no_verifier",
            selected=geometry.target.candidate_id,
        )
        return geometry
    log.info(
        "[object-selection] pairs=%d classes=%s verify=%s reason=%s "
        "geometry=%s occupancy=%s",
        len(ranked),
        ",".join(sorted(_target_classes(parsed))),
        True,
        "verifier_configured",
        geometry.target.candidate_id,
        "ok" if occupancy_at is not None else "missing",
    )

    method = str(getattr(verifier, "method", "qwen"))
    try:
        remote = method == "qwen_remote"
        contenders = ranked
        screened: dict[
            tuple[str, Optional[str]], tuple[ObjectSelection, str, ObjectBoard]
        ] = {}

        def verify(
            items: tuple[ObjectSelection, ...], *, choose_anchor: bool = False
        ):
            candidate_board = build_object_board(
                command_text,
                parsed,
                round_,
                items,
                max_crops=len(items) if remote else None,
                pack_cameras=remote,
                projector=projector,
                occupancy_at=occupancy_at,
                choose_anchor=choose_anchor,
            )
            parsed_reply, reply = call_verifier(
                verifier,
                candidate_board,
                lambda text: (
                    parse_object_pair_output(text, items)
                    if choose_anchor
                    else parse_object_target_output(text, items)
                    if remote
                    else parse_object_verifier_output(text, items)
                ),
            )
            chosen = next(item for item in items if _pair_key(item) == parsed_reply)
            return chosen, reply, candidate_board

        if (
            remote
            and parsed.scene_hint is None
            and parsed.relation is not Relation.NEAR
        ):
            groups: dict[str, list[ObjectSelection]] = {}
            for item in contenders:
                groups.setdefault(item.target.class_name, []).append(item)
            survivors: list[ObjectSelection] = []
            for class_name, group in groups.items():
                batch = tuple(group)
                if len(batch) == 1:
                    survivors.extend(batch)
                    continue
                try:
                    result = verify(batch)
                    winner = result[0]
                    survivors.append(winner)
                    screened[_pair_key(winner)] = result
                    log.info(
                        "[vlm] object class-screen class=%s candidates=%d selected=%s",
                        class_name,
                        len(batch),
                        winner.target.candidate_id,
                    )
                except Exception as exc:
                    log.warning(
                        "[vlm] object class-screen failed class=%s candidates=%d "
                        "error=%s; keeping all",
                        class_name,
                        len(batch),
                        type(exc).__name__,
                    )
                    note_decision(
                        "class_screen_failed", class_name=class_name,
                        candidates=len(batch), error=type(exc).__name__,
                    )
                    survivors.extend(batch)
            contenders = tuple(survivors)

        if remote and len(contenders) == 1 and _pair_key(contenders[0]) in screened:
            selected, raw_output, board = screened[_pair_key(contenders[0])]
        else:
            selected, raw_output, board = verify(contenders)
        if remote:
            anchors = _anchor_variants(
                selected, parsed, command_text, round_
            )
            if anchors:
                selected, raw_output, board = verify(
                    anchors, choose_anchor=True
                )
        qwen_pick = selected
        guard = None
        if (
            parsed.relation in (Relation.ON, Relation.ABOVE)
            and selected.match != "strict"
            and geometry.match == "strict"
        ):
            guard = "support_geometry_guard"
            selected = geometry
            log.info(
                "[vlm] support geometry guard selected=%s",
                selected.target.candidate_id,
            )
        elif occluded_geometry is not None:
            guard = "contained_occlusion_guard"
            selected = geometry
            log.info(
                "[vlm] contained-occlusion geometry guard selected=%s",
                selected.target.candidate_id,
            )
        elif (
            geometry.target.class_name == parsed.target_type
            and geometry.target.confidence < 0.02
        ):
            guard = "low_confidence_guard"
            selected = geometry
            log.info(
                "[vlm] low-confidence geometry guard selected=%s",
                selected.target.candidate_id,
            )
        elif (
            selected.target.confidence < 0.15
            and geometry.target.class_name == parsed.target_type
            and geometry.target.confidence >= 0.5
        ):
            guard = "weak_override_guard"
            selected = geometry
            log.info(
                "[vlm] weak-override geometry guard selected=%s",
                selected.target.candidate_id,
            )
        target_id, anchor_id = _pair_key(selected)
        if guard is not None:
            note_decision(
                "geometry_override", path="object", reason=guard,
                qwen_target=qwen_pick.target.candidate_id,
                qwen_anchor=_pair_key(qwen_pick)[1],
                selected=target_id, selected_anchor=anchor_id,
            )
        note_decision(
            "selector", path="object", backend=method,
            outcome="geometry_override" if guard is not None else "qwen_choice",
            reason=guard, selected=target_id,
        )
        log.info(
            "[vlm] object verified target=%s anchor=%s backend=%s output=%s",
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
            getattr(verifier, "last_prompt", board.prompt),
            selected.match,
            tuple(board.images),
        )
    except Exception as exc:
        raw_output = locals().get("raw_output")
        if raw_output is None:
            raw_output = getattr(exc, "raw_output", None)
        preview = " ".join(raw_output.split()) if isinstance(raw_output, str) else None
        log.warning(
            "[vlm] object verifier failed backend=%s error=%s detail=%s "
            "output=%s; using geometry ranking",
            method,
            type(exc).__name__,
            exc,
            preview,
        )
        note_decision(
            "selector", path="object", backend=method, outcome="fallback",
            reason=f"verifier_failed:{type(exc).__name__}",
            selected=geometry.target.candidate_id,
        )
        return replace(geometry, method="deterministic_fallback")
