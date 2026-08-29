"""Keep object ranking on cameras YOLO already saw for this request.

Cameras are chosen from detected class names, not from practice-scene rig ids.
An empty gate must not fail closed: callers keep the unfiltered round.

SAR person selection deliberately uses these scene classes only as soft evidence.
"""

from __future__ import annotations

from typing import Optional

from marc_agent.command_parser import (
    PARKED_VEHICLE_LANDMARKS,
    PARKING_SCENES,
    SEATING_LANDMARKS,
    ParsedCommand,
)
from marc_agent.types import CctvDetectionRound


def scene_classes(scene_hint: Optional[str]) -> frozenset[str]:
    """YOLO classes that typically appear in a parsed scene."""
    if scene_hint in PARKING_SCENES:
        return PARKED_VEHICLE_LANDMARKS
    if scene_hint == "park":
        return SEATING_LANDMARKS
    if scene_hint == "road":
        return frozenset({"taxi"})
    return frozenset()


def _cameras_with(objects, names: frozenset[str]) -> set[str]:
    return {item.camera_id for item in objects if item.class_name in names}


def _target_cameras(parsed: ParsedCommand, round_: CctvDetectionRound) -> set[str]:
    if parsed.target_type == "person":
        return {person.camera_id for person in round_.people}
    return {
        item.camera_id
        for item in round_.objects
        if item.class_name == parsed.target_type
    }


def _narrow(current: set[str], extra: set[str]) -> set[str]:
    if not extra:
        return current
    if not current:
        return set(extra)
    narrowed = current & extra
    return narrowed if narrowed else current


def eligible_cameras(parsed: ParsedCommand, round_: CctvDetectionRound) -> frozenset[str]:
    """Cameras that contain the object and/or the named landmark or place cue."""
    target_cams = _target_cameras(parsed, round_)
    support_cams = (
        _cameras_with(round_.objects, frozenset({parsed.landmark}))
        if parsed.landmark
        else set()
    )
    scene_cams = _cameras_with(round_.objects, scene_classes(parsed.scene_hint))
    if parsed.landmark:
        cams = set(support_cams)
        cams = _narrow_place_cues(cams, parsed, round_)
        cams = _narrow(cams, scene_cams)
        cams = _narrow(cams, target_cams)
        if cams:
            return frozenset(cams)
        if target_cams:
            return frozenset(target_cams)
        if support_cams:
            return frozenset(support_cams)
        return frozenset()

    cams = set(target_cams)
    cams = _narrow(cams, scene_cams)
    if parsed.target_type != "person":
        cams = _narrow_place_cues(cams, parsed, round_)
    if cams:
        return frozenset(cams)
    if target_cams:
        return frozenset(target_cams)
    return frozenset(scene_cams)


def _narrow_place_cues(
    cams: set[str], parsed: ParsedCommand, round_: CctvDetectionRound
) -> set[str]:
    for cue in parsed.place_cues:
        cue_cams = _cameras_with(round_.objects, frozenset({cue}))
        cams = _narrow(cams, cue_cams)
    return cams


def apply_camera_gate(
    parsed: ParsedCommand, round_: CctvDetectionRound
) -> CctvDetectionRound:
    """Drop people and objects whose camera is outside the YOLO gate.

    If the gate would remove every remaining target, the original round is kept.
    """
    cameras = eligible_cameras(parsed, round_)
    if not cameras:
        return round_

    people = [item for item in round_.people if item.camera_id in cameras]
    objects = [item for item in round_.objects if item.camera_id in cameras]
    if parsed.target_type == "person":
        if round_.people and not people:
            return round_
    else:
        had_target = any(
            item.class_name == parsed.target_type for item in round_.objects
        )
        kept_target = any(item.class_name == parsed.target_type for item in objects)
        if had_target and not kept_target:
            return round_

    frames = {
        camera_id: frame
        for camera_id, frame in round_.frames.items()
        if camera_id in cameras
    } or round_.frames
    camera_ids = [item for item in round_.camera_ids if item in cameras] or list(
        round_.camera_ids
    )
    return CctvDetectionRound(
        objects=objects,
        people=people,
        sar=list(round_.sar),
        frames=frames,
        camera_ids=camera_ids,
        metrics=round_.metrics,
    )
