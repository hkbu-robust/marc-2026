"""Shared near-field target snapshot used at policy and demonstration handoff."""

from __future__ import annotations

import math
import os

from marc_agent.gpu_handoff import reload_nearfield_detector

from .grasp_profiles import resolve_grasp_profile


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def nearfield_for_approach(client, approach):
    """Build the same RGB-D localizer around an approach controller's detector."""
    from nearfield import NearFieldAcquire, YoloDetector

    return NearFieldAcquire(client, detector=YoloDetector(approach._yolo))


def snapshot_pick_target(nearfield, target_object: str, log=print):
    """Return one validated arm-frame target with identical picker/collector logs."""
    detector = getattr(nearfield, "detector", None)
    reload_nearfield_detector(detector)
    log(f"[pick] snapshot {target_object}...")
    target = nearfield.snapshot(target_object)
    if target is None or not getattr(target, "valid", False):
        reason = "invalid" if target is None else (target.reason or "invalid")
        if target is None:
            log(f"[pick] snapshot failed ({reason})")
        else:
            log(
                "[pick] snapshot failed (%s) fwd=%.3f lat=%.3f z=%.3f"
                % (
                    reason,
                    float(getattr(target, "fwd", 0.0) or 0.0),
                    float(getattr(target, "lat", 0.0) or 0.0),
                    float(getattr(target, "z", 0.0) or 0.0),
                )
            )
        return None
    log(
        "[pick] localized %s fwd=%.3f lat=%.3f z=%.3f conf=%.2f"
        % (
            getattr(target, "name", "") or target_object,
            float(getattr(target, "fwd", 0.0) or 0.0),
            float(getattr(target, "lat", 0.0) or 0.0),
            float(getattr(target, "z", 0.0) or 0.0),
            float(getattr(target, "conf", 0.0) or 0.0),
        )
    )
    return target


def stable_pick_target(
    nearfield,
    target_object: str,
    log=print,
    *,
    visual: bool = False,
    require_reach: bool = True,
    posture_hint=None,
):
    """Return two agreeing, confident YOLO RGB-D localizations or fail closed."""
    min_conf = _env_float(
        "MARC_GRASPGENX_VISUAL_MIN_CONF" if visual
        else "MARC_GRASPGENX_MIN_LOCALIZE_CONF",
        0.3 if visual else 0.45,
    )
    generic_min_fwd = _env_float("MARC_GRASPGENX_MIN_FWD_M", 0.32)
    generic_max_fwd = _env_float("MARC_GRASPGENX_MAX_FWD_M", 0.58)
    attempts = max(2, int(_env_float("MARC_GRASPGENX_SNAPSHOT_ATTEMPTS", 10)))
    xy_tol = _env_float("MARC_GRASPGENX_LOCALIZE_XY_TOL_M", 0.04)
    z_tol = _env_float("MARC_GRASPGENX_LOCALIZE_Z_TOL_M", 0.05)
    hint_profile = (
        None if posture_hint is None else resolve_grasp_profile(posture_hint)
    )
    hint_name = str(getattr(posture_hint, "name", "") or "").strip().lower()
    safe = []
    if hint_profile is not None and getattr(posture_hint, "valid", False) is True:
        try:
            hint_values = (
                float(getattr(posture_hint, "fwd", float("nan"))),
                float(getattr(posture_hint, "lat", float("nan"))),
                float(getattr(posture_hint, "z", float("nan"))),
                float(getattr(posture_hint, "conf", 0.0) or 0.0),
            )
        except (TypeError, ValueError):
            hint_values = (float("nan"),) * 4
        hint_in_reach = (
            float(hint_profile.min_fwd_m)
            <= hint_values[0]
            <= float(hint_profile.max_fwd_m)
        )
        if (
            hint_name == str(target_object or "").strip().lower()
            and all(math.isfinite(value) for value in hint_values)
            and hint_values[3] >= min_conf
            and (not require_reach or hint_in_reach)
        ):
            # Auto-approach maintained this class-keyed RGB-D lock immediately
            # before stopping. Treat it as the first independent observation;
            # one agreeing post-stop chassis snapshot then completes consensus.
            safe.append(posture_hint)
            log("[pick] seeded localization consensus from maintained approach lock")
    for _ in range(attempts):
        candidate = snapshot_pick_target(nearfield, target_object, log=log)
        profile = None if candidate is None else resolve_grasp_profile(candidate)
        use_hint = False
        if candidate is not None and hint_profile is not None:
            candidate_name = str(getattr(candidate, "name", "") or "").strip().lower()
            hint_dxy = math.hypot(
                float(candidate.fwd) - float(posture_hint.fwd),
                float(candidate.lat) - float(posture_hint.lat),
            )
            use_hint = (
                hint_profile.approach_style == "front_side"
                and candidate_name == hint_name
                and hint_dxy <= max(0.10, xy_tol)
            )
            if use_hint:
                profile = hint_profile
        min_fwd = (
            generic_min_fwd if profile is None else float(profile.min_fwd_m)
        )
        max_fwd = (
            generic_max_fwd if profile is None else float(profile.max_fwd_m)
        )
        if (
            candidate is None
            or not (float(candidate.conf) >= min_conf)
            or (require_reach and not min_fwd <= float(candidate.fwd) <= max_fwd)
        ):
            continue
        for previous in safe:
            dxy = math.hypot(
                float(candidate.fwd) - float(previous.fwd),
                float(candidate.lat) - float(previous.lat),
            )
            if dxy <= xy_tol and abs(float(candidate.z) - float(previous.z)) <= z_tol:
                if use_hint:
                    setattr(candidate, "grasp_posture", "upright")
                    log(
                        "[pick] stable localization inherits approach posture=upright"
                    )
                return candidate
        safe.append(candidate)
    return None
