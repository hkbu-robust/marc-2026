"""Shared near-field target snapshot used at policy and demonstration handoff."""

from __future__ import annotations

from marc_agent.gpu_handoff import reload_nearfield_detector


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
