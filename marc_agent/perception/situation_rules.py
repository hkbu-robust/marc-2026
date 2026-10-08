"""Public R03 pose×support situation rules (version 2026.1).

The submitted GroundingResult.situation comes from this matrix applied to the selected
person's detected pose, with the anchor's class as support (#114,
``stage1_pipeline.submitted_situation``); the text's situation is only a fallback.
Unknown support uses the pose fallback from the organisers' people_poses.yaml
(bending is emergency, reclining is normal), then ``normal``.
"""

from __future__ import annotations

from typing import Optional

RULES_VERSION = "2026.1"
SITUATIONS = ("normal", "abnormal", "emergency", "accident")
DEFAULT_SITUATION = "normal"

POSE_GROUPS = {
    "upright": frozenset({"person_walking", "person_walking_away", "person_reaching"}),
    "seated": frozenset({"person_sitting", "person_reclining"}),
    "low": frozenset({"person_crouching"}),
    "prostrate": frozenset({"person_bending", "person_lying_down", "person_prone"}),
}

SUPPORT_CLASSES = {
    "seating": frozenset({"bench", "picnic_table", "chair"}),
    "rest_ground": frozenset({"grass", "lawn"}),
    "transit": frozenset({
        "walkway", "sidewalk", "road", "parking", "parking_space",
    }),
    "non_mount": frozenset({
        "trash_can", "vending_machine", "postbox", "hydrant", "car",
    }),
    "vehicle_fallen": frozenset({"bicycle", "kick_scooter"}),
}

# (pose_group, support_class) -> situation. person_prone is applied first.
SITUATION_MATRIX = {
    ("upright", "seating"): "abnormal",
    ("upright", "rest_ground"): "normal",
    ("upright", "transit"): "normal",
    ("upright", "non_mount"): "abnormal",
    ("upright", "vehicle_fallen"): "abnormal",
    ("seated", "seating"): "normal",
    ("seated", "rest_ground"): "normal",
    ("seated", "transit"): "abnormal",
    ("seated", "non_mount"): "abnormal",
    ("seated", "vehicle_fallen"): "abnormal",
    ("low", "seating"): "abnormal",
    ("low", "rest_ground"): "abnormal",
    ("low", "transit"): "abnormal",
    ("low", "non_mount"): "abnormal",
    ("low", "vehicle_fallen"): "abnormal",
    ("prostrate", "seating"): "abnormal",
    ("prostrate", "rest_ground"): "normal",
    ("prostrate", "transit"): "emergency",
    ("prostrate", "non_mount"): "emergency",
    ("prostrate", "vehicle_fallen"): "emergency",
}

# Pose-only fallback when support/surface is unknown. Does not invent emergency.
POSE_FALLBACK = {
    "person_prone": "accident",
    "person_lying_down": "abnormal",
    "person_bending": "emergency",
    "person_crouching": "abnormal",
    "person_reclining": "normal",
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
