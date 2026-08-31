"""Validated class-and-posture grasp calibrations.

Only physically demonstrated profiles belong here. Unknown combinations return
``None`` so the existing generic, environment-configured path remains fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GraspProfile:
    name: str
    stage_clearance_m: float
    overhead_clearance_m: float
    floor_clearance_m: float
    contact_depth_fraction: float
    max_cloud_center_shift_m: float
    wrist_search_radius_m: float
    wrist_depth_gate_m: float
    wrist_min_points: int
    wrist_max_disagreement_m: float
    wrist_max_correction_m: float
    contact_settle_s: float
    contact_max_drift_m: float


LYING_TUMBLER = GraspProfile(
    name="lying_tumbler",
    stage_clearance_m=0.23,
    overhead_clearance_m=0.08,
    floor_clearance_m=0.065,
    contact_depth_fraction=0.50,
    max_cloud_center_shift_m=0.0,
    wrist_search_radius_m=0.10,
    wrist_depth_gate_m=0.06,
    wrist_min_points=20,
    wrist_max_disagreement_m=0.07,
    wrist_max_correction_m=0.03,
    contact_settle_s=0.50,
    contact_max_drift_m=0.006,
)


def bbox_is_upright(target, min_aspect: float = 1.20) -> bool:
    bbox = getattr(target, "bbox_xyxy", None)
    if bbox is None or len(bbox) < 4:
        return False
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    return width > 1.0 and height / width >= float(min_aspect)


def resolve_grasp_profile(target) -> Optional[GraspProfile]:
    """Return a demonstrated profile for one detected class/posture, if any."""
    name = str(getattr(target, "name", "") or "").strip().lower()
    bbox = getattr(target, "bbox_xyxy", None)
    # Do not infer "lying" merely because posture metadata is absent.  A
    # demonstrated profile is selected only from an observed class and bbox.
    if (
        name == "tumbler"
        and bbox is not None
        and len(bbox) >= 4
        and not bbox_is_upright(target)
    ):
        return LYING_TUMBLER
    return None
