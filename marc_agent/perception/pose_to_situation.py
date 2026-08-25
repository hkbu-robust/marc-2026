"""Map YOLO person-pose classes to challenge SAR situation categories.

Delegates to the versioned R03 matrix in ``situation_rules``. Pose-only calls
use the documented fallback (never invent ``emergency`` without support).
"""

from __future__ import annotations

from typing import Optional

from marc_agent.perception.situation_rules import (
    DEFAULT_SITUATION,
    POSE_FALLBACK,
    resolve_situation,
)

# Backward-compatible alias of the pose-only fallback table.
POSE_TO_SITUATION: dict[str, str] = dict(POSE_FALLBACK)
_DEFAULT_SITUATION = DEFAULT_SITUATION


def is_person_pose(class_name: str) -> bool:
    return (class_name or "").strip().lower().startswith("person_")


def situation_for_pose(
    class_name: str,
    support: Optional[str] = None,
    surface: Optional[str] = None,
) -> str:
    return resolve_situation(class_name, support=support, surface=surface)
