"""Map YOLO person-pose classes to challenge SAR situation categories."""

from __future__ import annotations

# Heuristic aligned with demo/vlm_dataset/cctv_gt_to_qwen.py (_SAR_SITUATION).
POSE_TO_SITUATION: dict[str, str] = {
    "person_prone": "accident",
    "person_lying_down": "accident",
    "person_bending": "emergency",
    "person_crouching": "emergency",
    "person_reclining": "abnormal",
    "person_sitting": "normal",
    "person_reaching": "normal",
    "person_walking": "normal",
    "person_walking_away": "normal",
}

_DEFAULT_SITUATION = "abnormal"


def is_person_pose(class_name: str) -> bool:
    return (class_name or "").strip().lower().startswith("person_")


def situation_for_pose(class_name: str) -> str:
    name = (class_name or "").strip().lower()
    return POSE_TO_SITUATION.get(name, _DEFAULT_SITUATION)
