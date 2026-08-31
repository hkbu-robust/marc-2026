"""Class-and-posture grasp calibrations.

Profiles state whether they have completed a physical placement. Unknown
combinations return ``None`` so the generic environment path remains fail-closed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class GraspProfile:
    name: str
    physically_validated: bool
    min_fwd_m: float
    max_fwd_m: float
    approach_already_near_m: Optional[float]
    approach_style: str
    stage_back_m: float
    approach_tilt_rad: float
    side_transit_tilt_rad: float
    side_transit_rate_hz: float
    side_transit_cart_step_m: float
    side_transit_joint_step_rad: float
    side_floor_load: bool
    side_contact_fold_clearance_m: float
    side_final_insertion_m: float
    side_fold_speed_rad_s: float
    side_standoff_m: float
    side_pregrasp_min_fwd_m: float
    side_clear_lift_m: float
    pad_lateral_bias_m: float
    pad_center_offset_m: float
    center_pos_tolerance_m: float
    center_rot_tolerance_rad: float
    line_joint_step_rad: float
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
    physically_validated=True,
    min_fwd_m=0.32,
    max_fwd_m=0.58,
    approach_already_near_m=None,
    approach_style="top_down",
    stage_back_m=0.0,
    approach_tilt_rad=0.0,
    side_transit_tilt_rad=0.0,
    side_transit_rate_hz=20.0,
    side_transit_cart_step_m=0.004,
    side_transit_joint_step_rad=0.0125,
    side_floor_load=False,
    side_contact_fold_clearance_m=0.0,
    side_final_insertion_m=0.0,
    side_fold_speed_rad_s=0.0,
    side_standoff_m=0.0,
    side_pregrasp_min_fwd_m=0.0,
    side_clear_lift_m=0.0,
    pad_lateral_bias_m=0.0,
    pad_center_offset_m=0.0205,
    center_pos_tolerance_m=0.03,
    center_rot_tolerance_rad=0.10,
    line_joint_step_rad=0.0125,
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


# Upright calibration derived from the observed cloud/front-surface offset and
# checked through a complete physical grip and basket placement.
TUMBLER_UPRIGHT = GraspProfile(
    name="tumbler_upright",
    physically_validated=True,
    min_fwd_m=0.32,
    # The guarded coordinated-fold path was checked through 0.671 m using the
    # real URDF. Keep a narrow 9 mm ceiling above that observed target.
    max_fwd_m=0.68,
    approach_already_near_m=None,
    approach_style="front_side",
    stage_back_m=0.10,
    # Keep the absolute contact tilt as a conservative fallback for generic
    # pose helpers. The upright handoff itself finishes with a direct distal
    # wrist arc and preserves that measured orientation during insertion.
    approach_tilt_rad=1.10,
    side_transit_tilt_rad=0.65,
    # Collision-clear pregrasp travel is capped at 20 Hz and runs near (but
    # below) the Gen3 wrist's 1.22 rad/s URDF limit. Contact motion keeps its
    # smaller 30 Hz leash.
    side_transit_rate_hz=20.0,
    side_transit_cart_step_m=0.020,
    side_transit_joint_step_rad=0.055,
    side_floor_load=False,
    # Stop the descent at the safe transition. Fold in place there, then make
    # only a slow 15 mm level advance before closing; never restore the lower
    # cloud-contact Z after the fold.
    side_contact_fold_clearance_m=0.10,
    side_final_insertion_m=0.015,
    side_fold_speed_rad_s=0.50,
    side_standoff_m=0.10,
    side_pregrasp_min_fwd_m=0.44,
    # The folded orientation is not reachable at the nominal ready height: the
    # third joint bottoms out first. A 14 cm vertical clear move gives the wrist
    # enough room to fold beside the chassis before it extends toward the cup.
    side_clear_lift_m=0.14,
    # Arm-frame +Y is robot/camera-left (REP-103). Keep the empirical
    # upright correction positive so it moves the pad midpoint left.
    pad_lateral_bias_m=0.030,
    pad_center_offset_m=0.0205,
    center_pos_tolerance_m=0.005,
    center_rot_tolerance_rad=0.05,
    line_joint_step_rad=0.004,
    stage_clearance_m=0.23,
    overhead_clearance_m=0.08,
    floor_clearance_m=0.045,
    contact_depth_fraction=0.35,
    max_cloud_center_shift_m=0.0,
    wrist_search_radius_m=0.10,
    wrist_depth_gate_m=0.06,
    wrist_min_points=20,
    wrist_max_disagreement_m=0.07,
    wrist_max_correction_m=0.030,
    contact_settle_s=0.50,
    contact_max_drift_m=0.006,
)


def _copied_profile(
    source: GraspProfile, name: str
) -> GraspProfile:
    """Copy a calibrated motion under a class-specific experimental identity."""
    overrides = {}
    if source.approach_style == "top_down":
        # The copied overhead classes have not validated the tumbler's deep
        # midpoint insertion. Stop within the upper object envelope and close
        # there; this remains geometric and never waits for physical contact.
        overrides.update(
            contact_depth_fraction=0.30,
            floor_clearance_m=max(0.075, source.floor_clearance_m),
        )
    return replace(
        source,
        name=name,
        physically_validated=False,
        **overrides,
    )


# These classes inherit both demonstrated tumbler motions. Their separate
# identities keep physical validation local to each class instead of falsely
# marking all copies validated when only the tumbler has completed placement.
LYING_MUG = _copied_profile(LYING_TUMBLER, "lying_mug")
UPRIGHT_MUG = _copied_profile(TUMBLER_UPRIGHT, "upright_mug")
LYING_COLA_CAN = _copied_profile(LYING_TUMBLER, "lying_cola_can")
UPRIGHT_COLA_CAN = _copied_profile(TUMBLER_UPRIGHT, "upright_cola_can")
LYING_JUICE = _copied_profile(LYING_TUMBLER, "lying_juice")
UPRIGHT_JUICE = _copied_profile(TUMBLER_UPRIGHT, "upright_juice")
LYING_DISPOSABLE_CUP = _copied_profile(
    LYING_TUMBLER, "lying_disposable_cup"
)
UPRIGHT_DISPOSABLE_CUP = _copied_profile(
    TUMBLER_UPRIGHT, "upright_disposable_cup"
)
LYING_UMBRELLA = _copied_profile(LYING_TUMBLER, "lying_umbrella")
UPRIGHT_UMBRELLA = _copied_profile(TUMBLER_UPRIGHT, "upright_umbrella")
LYING_SUNBLOCK = _copied_profile(LYING_TUMBLER, "lying_sunblock")
# A sunblock is too short for the tumbler's elevated side pinch. Keep the same
# collision-clear upright transit, but finish with the open gripper at a guarded
# floor-level pad height and load the object fully between the fingers before
# closing. This remains isolated from every other copied upright profile.
UPRIGHT_SUNBLOCK = replace(
    _copied_profile(TUMBLER_UPRIGHT, "upright_sunblock"),
    approach_already_near_m=0.65,
    approach_tilt_rad=1.45,
    side_floor_load=True,
    side_contact_fold_clearance_m=0.0,
    side_final_insertion_m=0.10,
    floor_clearance_m=0.035,
    contact_settle_s=0.75,
)

# Pencil cases and tissues always use the top-down/lying motion, even when a
# tall or folded detection box looks upright.
LYING_PENCILCASE = _copied_profile(LYING_TUMBLER, "lying_pencilcase")
LYING_TISSUE = _copied_profile(LYING_TUMBLER, "lying_tissue")

# Cracker-box posture is intentionally inverted: standing boxes expose a
# narrow top footprint suitable for the lying-tumbler motion. A flat box has no
# safe copied grasp because its width exceeds the gripper; fail closed until a
# physical flip-and-reacquire sequence is calibrated.
UPRIGHT_CRACKER_BOX = _copied_profile(
    LYING_TUMBLER, "upright_cracker_box"
)
LYING_CRACKER_BOX = replace(
    LYING_TUMBLER,
    name="lying_cracker_box",
    physically_validated=False,
    approach_style="flip_required",
)


_POSTURE_PROFILES = {
    "tumbler": (LYING_TUMBLER, TUMBLER_UPRIGHT),
    "mug": (LYING_MUG, UPRIGHT_MUG),
    "cola_can": (LYING_COLA_CAN, UPRIGHT_COLA_CAN),
    "juice": (LYING_JUICE, UPRIGHT_JUICE),
    "disposable_cup": (LYING_DISPOSABLE_CUP, UPRIGHT_DISPOSABLE_CUP),
    "umbrella": (LYING_UMBRELLA, UPRIGHT_UMBRELLA),
    "sunblock": (LYING_SUNBLOCK, UPRIGHT_SUNBLOCK),
}

_ALWAYS_LYING_PROFILES = {
    "pencilcase": LYING_PENCILCASE,
    "tissue": LYING_TISSUE,
}

_ALWAYS_UPRIGHT_PROFILES = {
    # Sunblock is narrow enough in either resting posture for the side-style
    # handoff; avoid the copied overhead descent entirely.
    "sunblock": UPRIGHT_SUNBLOCK,
}


def _canonical_name(name) -> str:
    normalized = (
        str(name or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    return "umbrella" if normalized == "long_umbrella" else normalized


def bbox_xyxy_is_upright(bbox, min_aspect: float = 1.20) -> bool:
    if bbox is None or len(bbox) < 4:
        return False
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    return width > 1.0 and height / width >= float(min_aspect)


def bbox_is_upright(target, min_aspect: float = 1.20) -> bool:
    return bbox_xyxy_is_upright(
        getattr(target, "bbox_xyxy", None), min_aspect=min_aspect
    )


def is_upright_tumbler(name, bbox, min_aspect: float = 1.20) -> bool:
    """Classify the one posture with a distinct base and arm workspace."""
    if str(name or "").strip().lower() != "tumbler":
        return False
    return bbox_xyxy_is_upright(bbox, min_aspect=min_aspect)


def uses_upright_side_profile(name, bbox, min_aspect: float = 1.20) -> bool:
    """Whether this class and observed posture use the farther side workspace."""
    canonical = _canonical_name(name)
    if canonical in _ALWAYS_UPRIGHT_PROFILES:
        return True
    return (
        canonical in _POSTURE_PROFILES
        and bbox_xyxy_is_upright(bbox, min_aspect=min_aspect)
    )


def upright_side_approach_near_override(
    name, bbox, min_aspect: float = 1.20
) -> Optional[float]:
    """Return a calibrated class-only auto-approach distance, if one exists."""
    canonical = _canonical_name(name)
    profile = _ALWAYS_UPRIGHT_PROFILES.get(canonical)
    if profile is None:
        profiles = _POSTURE_PROFILES.get(canonical)
        if profiles is None or not bbox_xyxy_is_upright(
            bbox, min_aspect=min_aspect
        ):
            return None
        profile = profiles[1]
    return profile.approach_already_near_m


def resolve_grasp_profile(target) -> Optional[GraspProfile]:
    """Resolve all class/posture policy behind one fail-closed interface."""
    name = _canonical_name(getattr(target, "name", ""))
    posture = str(getattr(target, "grasp_posture", "") or "").strip().lower()
    bbox = getattr(target, "bbox_xyxy", None)

    always_lying = _ALWAYS_LYING_PROFILES.get(name)
    if always_lying is not None:
        return always_lying

    always_upright = _ALWAYS_UPRIGHT_PROFILES.get(name)
    if always_upright is not None:
        return always_upright

    if name == "cracker_box":
        if posture == "lying":
            return LYING_CRACKER_BOX
        if posture == "upright":
            return UPRIGHT_CRACKER_BOX
        if bbox is not None and len(bbox) >= 4:
            return (
                UPRIGHT_CRACKER_BOX
                if bbox_xyxy_is_upright(bbox)
                else LYING_CRACKER_BOX
            )
        return None

    profiles = _POSTURE_PROFILES.get(name)
    if profiles is None:
        return None
    lying_profile, upright_profile = profiles
    if posture in {"upright", "lying"}:
        return upright_profile if posture == "upright" else lying_profile
    # Do not infer "lying" merely because posture metadata is absent.  A
    # demonstrated profile is selected only from an observed class and bbox.
    if bbox is not None and len(bbox) >= 4:
        return (
            upright_profile
            if bbox_xyxy_is_upright(bbox)
            else lying_profile
        )
    return None
