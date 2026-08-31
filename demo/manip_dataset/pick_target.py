"""Shared near-field target snapshot used at policy and demonstration handoff."""

from __future__ import annotations

import math
import os

from marc_agent.gpu_handoff import reload_nearfield_detector


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
):
    """Return two agreeing, confident YOLO RGB-D localizations or fail closed."""
    min_conf = _env_float(
        "MARC_GRASPGENX_VISUAL_MIN_CONF" if visual
        else "MARC_GRASPGENX_MIN_LOCALIZE_CONF",
        0.3 if visual else 0.45,
    )
    min_fwd = _env_float("MARC_GRASPGENX_MIN_FWD_M", 0.32)
    max_fwd = _env_float("MARC_GRASPGENX_MAX_FWD_M", 0.58)
    attempts = max(2, int(_env_float("MARC_GRASPGENX_SNAPSHOT_ATTEMPTS", 10)))
    xy_tol = _env_float("MARC_GRASPGENX_LOCALIZE_XY_TOL_M", 0.04)
    z_tol = _env_float("MARC_GRASPGENX_LOCALIZE_Z_TOL_M", 0.05)
    safe = []
    for _ in range(attempts):
        candidate = snapshot_pick_target(nearfield, target_object, log=log)
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
                return candidate
        safe.append(candidate)
    return None
