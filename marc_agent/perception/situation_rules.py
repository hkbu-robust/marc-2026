"""Public R03 pose×support situation rules (version 2026.1).

The submitted GroundingResult.situation comes from this matrix applied to the selected
person's detected pose, with the anchor's class as support (#114,
``stage1_pipeline.submitted_situation``); the text's situation is only a fallback.
Unknown support uses the pose fallback below, then ``abnormal`` — never invent
``emergency`` without a known road/parking, non-mount, or fallen-vehicle support.
"""

from __future__ import annotations

from typing import Optional

RULES_VERSION = "2026.1"
SITUATIONS = ("normal", "abnormal", "emergency", "accident")
DEFAULT_SITUATION = "abnormal"

POSE_GROUPS = {
    "upright": frozenset({"person_walking", "person_walking_away", "person_reaching"}),
    "seated": frozenset({"person_sitting", "person_reclining"}),
    "low": frozenset({"person_crouching"}),
    "prostrate": frozenset({"person_bending", "person_lying_down", "person_prone"}),
}

SUPPORT_CLASSES = {
    "seating": frozenset({"bench", "picnic_table", "chair"}),
    "grass_lawn": frozenset({"grass", "lawn", "park"}),
    "road_parking": frozenset({
        "walkway", "sidewalk", "road", "parking", "parking_space",
    }),
    "non_mount": frozenset({
        "trash_can", "vending_machine", "postbox", "hydrant", "car",
        "police_car", "normal_car_1", "normal_car_2",
        "sports_car", "sports_car_2", "suv", "taxi",
    }),
    "fallen_vehicle": frozenset({"bicycle", "kick_scooter"}),
}

# (pose_group, support_class) -> situation. person_prone is applied first.
SITUATION_MATRIX = {
    ("upright", "seating"): "abnormal",
    ("upright", "grass_lawn"): "normal",
    ("upright", "road_parking"): "normal",
    ("upright", "non_mount"): "abnormal",
    ("upright", "fallen_vehicle"): "abnormal",
    ("seated", "seating"): "normal",
    ("seated", "grass_lawn"): "normal",
    ("seated", "road_parking"): "abnormal",
    ("seated", "non_mount"): "abnormal",
    ("seated", "fallen_vehicle"): "abnormal",
    ("low", "seating"): "abnormal",
    ("low", "grass_lawn"): "abnormal",
    ("low", "road_parking"): "abnormal",
    ("low", "non_mount"): "abnormal",
    ("low", "fallen_vehicle"): "abnormal",
    ("prostrate", "seating"): "abnormal",
    ("prostrate", "grass_lawn"): "normal",
    ("prostrate", "road_parking"): "emergency",
    ("prostrate", "non_mount"): "emergency",
    ("prostrate", "fallen_vehicle"): "emergency",
}

# Pose-only fallback when support/surface is unknown. Does not invent emergency.
POSE_FALLBACK = {
    "person_prone": "accident",
    "person_lying_down": "abnormal",
    "person_bending": "abnormal",
    "person_crouching": "abnormal",
    "person_reclining": "abnormal",
    "person_sitting": "normal",
    "person_reaching": "normal",
    "person_walking": "normal",
    "person_walking_away": "normal",
}


def _norm(name: Optional[str]) -> str:
    return (name or "").strip().lower()


def pose_group(pose: str) -> Optional[str]:
    name = _norm(pose)
    for group, members in POSE_GROUPS.items():
        if name in members:
            return group
    return None


def support_class(name: Optional[str]) -> Optional[str]:
    token = _norm(name)
    if not token:
        return None
    if token.startswith("person_"):
        return None
    for cls, members in SUPPORT_CLASSES.items():
        if token in members:
            return cls
    return None


def resolve_situation(
    pose: str,
    support: Optional[str] = None,
    surface: Optional[str] = None,
) -> str:
    """Map pose × support/surface to a four-way situation.

    ``person_prone`` is always ``accident``. A physical support class (landmark)
    wins over a ground ``surface``. Unknown support uses :data:`POSE_FALLBACK`.
    """
    pose_name = _norm(pose)
    if pose_name == "person_prone" or pose_name.startswith("person_prone"):
        return "accident"

    support_cls = support_class(support) or support_class(surface)
    group = pose_group(pose_name)
    if group is not None and support_cls is not None:
        mapped = SITUATION_MATRIX.get((group, support_cls))
        if mapped:
            return mapped
    return POSE_FALLBACK.get(pose_name, DEFAULT_SITUATION)
