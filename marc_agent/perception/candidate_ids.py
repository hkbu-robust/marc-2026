"""Stable Stage 1 candidate IDs: ``{camera}:{class}:{index}``.

Person IDs use the detector pose class (``person_sitting``, ``person_lying_down``,
…) so the token Qwen copies carries pose. Object IDs use the object class the
same way. Index is per-camera per-class, matching ``tumbler:0``, ``tumbler:1``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def object_candidate_id(camera_id: str, class_name: str, index: int) -> str:
    name = (class_name or "").strip() or "object"
    return f"{camera_id}:{name}:{int(index)}"


def person_candidate_id(camera_id: str, pose_class: str, index: int) -> str:
    pose = (pose_class or "").strip() or "person"
    return f"{camera_id}:{pose}:{int(index)}"


def assign_person_ids(camera_id: str, pose_classes: Sequence[str]) -> list[str]:
    """IDs for people already in display order; index is per pose class."""
    counts: dict[str, int] = defaultdict(int)
    ids: list[str] = []
    for pose in pose_classes:
        key = (pose or "").strip() or "person"
        idx = counts[key]
        counts[key] += 1
        ids.append(person_candidate_id(camera_id, key, idx))
    return ids
