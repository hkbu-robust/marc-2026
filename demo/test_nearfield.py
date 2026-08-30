#!/usr/bin/env python3
"""Numpy-only checks for near-field RGB-D localization (no ROS / torch).

Run from repo root or demo/:
  python3 demo/test_nearfield.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from nearfield import (
    ARM_BASE_FWD,
    BASE_CAMERA_TO_ARM_R,
    BASE_CAMERA_TO_ARM_T,
    BASE_CAMERA_TF_FRAME,
    MissingDetector,
    NearFieldAcquire,
    _candidate_detection,
    base_camera_to_car_transform,
    depth_in_bbox,
    gated_cmd_vel,
    localize_detection,
    optical_to_car_base,
    car_to_arm_base,
)


class _Fail(Exception):
    pass


def _check(cond, msg):
    if not cond:
        raise _Fail(msg)


class _FakeClient:
    def __init__(self):
        self.cmds = []
        self.is_running = True
        self._t = 0.0
        self.node = None

    def now_s(self):
        return self._t

    def sleep(self, seconds):
        self._t += float(seconds)

    def send_cmd_vel(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, twist=None):
        self.cmds.append((float(linear_x), float(angular_z)))

    def get_world_pose(self):
        return (0.0, 0.0, 0.0)

    def get_robot_image(self, which="base_left"):
        return None

    def get_robot_depth(self, which="base"):
        return None

    def get_robot_info(self, which="base_left"):
        return None


def test_depth_in_bbox_ignores_bad_center_pixel():
    h, w = 40, 40
    depth = np.full((h, w), 0.80, dtype=np.float32)
    # One unchecked center pixel would be 12 m (or NaN); the inset cluster is 0.80.
    depth[h // 2, w // 2] = 12.0
    bbox = (2.0, 2.0, 38.0, 38.0)
    d = depth_in_bbox(depth, bbox, (h, w), percentile=30.0, min_valid=4)
    _check(d is not None, "expected a depth statistic")
    _check(abs(d - 0.80) < 0.05, "cluster depth should win over the center spike, got %s" % d)
    # Single-pixel read of the center would have been 12 m.
    _check(d < 2.0, "must not return the lone center spike")


def test_radius_gate_rejects_far_box():
    h, w = 48, 64
    depth = np.full((h, w), 0.70, dtype=np.float32)
    bbox = (20.0, 16.0, 44.0, 40.0)
    k = (50.0, 50.0, 32.0, 24.0)  # fx, fy, cx, cy
    pose = (0.0, 0.0, 0.0)
    hint = (0.0, 0.0)
    inside = localize_detection(
        bbox, depth, (h, w), k, pose, hint, hint_radius=3.0, conf=0.9, name="cup",
    )
    _check(inside.valid, "near box should be inside a 3 m disc, got %s" % inside.reason)
    far = localize_detection(
        bbox, depth, (h, w), k, pose, hint_xy=(20.0, 0.0), hint_radius=1.0,
        conf=0.9, name="cup",
    )
    _check(not far.valid, "box far from hint should be rejected")
    _check(far.reason == "outside_radius", "reason=%s" % far.reason)


def test_optical_to_arm_base_known_Rt():
    # Identity optical->car: R maps (X,Y,Z) to (Z,-X,-Y) explicitly via fallback.
    fwd, lat, up = optical_to_car_base(0.10, -0.20, 0.80)
    _check(abs(fwd - 0.80) < 1e-9, "fallback fwd=%s" % fwd)
    _check(abs(lat - (-0.10)) < 1e-9, "fallback lat=%s" % lat)
    _check(abs(up - 0.20) < 1e-9, "fallback up=%s" % up)
    af, al, au = car_to_arm_base(fwd, lat, up)
    _check(abs(af - (0.80 - ARM_BASE_FWD)) < 1e-9, "arm fwd=%s" % af)
    _check(abs(al - lat) < 1e-9, "arm lat")

    # Known R: identity, t = (0.5, 0.1, 0.0) — p_car = p_optical + t
    R = np.eye(3)
    t = np.array([0.5, 0.1, 0.0])
    f2, l2, u2 = optical_to_car_base(0.0, 0.0, 0.0, R=R, t=t)
    _check(abs(f2 - 0.5) < 1e-9 and abs(l2 - 0.1) < 1e-9 and abs(u2) < 1e-9,
           "R,t identity offset got %s" % ((f2, l2, u2),))


def test_base_camera_calibration_and_tf_alias():
    # The formerly used origin-free fallback would report 0.505 - 0.18 = 0.325m.
    # The measured transform correctly places that centre-ray point around 0.696m.
    R, t = base_camera_to_car_transform()
    p_car = R @ np.array([0.0, 0.0, 0.505]) + t
    p_arm = p_car - np.array([ARM_BASE_FWD, 0.0, 0.0])
    _check(abs(p_arm[0] - 0.696) < 0.01, "calibrated fwd=%s" % p_arm[0])
    _check(p_arm[2] < -0.20, "camera pitch/height must affect z, got %s" % p_arm[2])

    acq = NearFieldAcquire.__new__(NearFieldAcquire)
    acq._tf = {
        BASE_CAMERA_TF_FRAME: (
            "arm_base_link",
            BASE_CAMERA_TO_ARM_T.copy(),
            BASE_CAMERA_TO_ARM_R.copy(),
        )
    }
    msg = type("Msg", (), {})()
    msg.header = type("Header", (), {"frame_id": "base_camera"})()
    R_alias, t_alias = acq._cam_to_car(msg)
    _check(np.allclose(R_alias, BASE_CAMERA_TO_ARM_R), "base_camera alias rotation")
    _check(np.allclose(t_alias, t), "base_camera alias translation")


def test_gated_cmd_zeros_linear_outside_disc():
    hint = (0.0, 0.0)
    # Inside, a small step stays in: keep linear.
    lin, ang = gated_cmd_vel((0.0, 0.0, 0.0), 0.15, 0.0, hint, 2.0, dt=0.1)
    _check(lin > 0.1, "inside disc should allow crawl, got lin=%s" % lin)
    # Already outside: force linear 0.
    lin2, _ = gated_cmd_vel((5.0, 0.0, 0.0), 0.15, 0.2, hint, 1.0, dt=0.1)
    _check(lin2 == 0.0, "outside disc must zero linear, got %s" % lin2)
    # Inside but the step would leave: zero linear.
    lin3, _ = gated_cmd_vel((0.90, 0.0, 0.0), 1.0, 0.0, hint, 1.0, dt=0.5, slack=0.0)
    _check(lin3 == 0.0, "step leaving the disc must zero linear, got %s" % lin3)


def test_fixture_box_localizes():
    h, w = 48, 64
    depth = np.full((h, w), 0.60, dtype=np.float32)
    # Box centred in the image so optical X,Y ~ 0, Z = 0.60 -> car fwd=0.60, lat=0
    bbox = (24.0, 16.0, 40.0, 32.0)
    k = (80.0, 80.0, 32.0, 24.0)
    pose = (10.0, 20.0, 0.0)
    hint = (10.6, 20.0)
    t = localize_detection(
        bbox, depth, (h, w), k, pose, hint, hint_radius=2.0, conf=0.8, name="tumbler",
    )
    _check(t.valid, "expected valid target, got %s (fwd=%s lat=%s)" % (t.reason, t.fwd, t.lat))
    _check(abs(t.lat) < 0.15, "centred box should be near lat 0, got %s" % t.lat)
    _check(0.15 <= t.fwd <= 1.0, "fwd in pick range, got %s" % t.fwd)
    _check(t.world_xy is not None, "world_xy required")
    _check(abs(t.world_xy[0] - (10.0 + 0.60)) < 0.15, "world x ~ pose + optical Z, got %s" % (t.world_xy,))


def test_missing_detector_fail_closed():
    c = _FakeClient()
    acq = NearFieldAcquire(c, MissingDetector("no_detector"))
    t = acq.acquire(None, deadline=c.now_s() + 5.0)
    _check(not t.valid, "missing detector must be invalid")
    _check(t.reason == "no_detector", "reason=%s" % t.reason)
    pos = [lin for lin, _ in c.cmds if lin > 1e-6]
    _check(not pos, "missing detector must never command linear, got %s" % c.cmds)


def test_candidate_selection_skips_box_outside_reveal():
    depth = np.full((48, 64), 0.60, dtype=np.float32)
    result = {
        "rgb_shape": (48, 64),
        "candidates": [
            {"xyxy": (55, 20, 63, 28), "confidence": 0.95, "class_id": 0, "class_name": "mug"},
            {"xyxy": (28, 20, 36, 28), "confidence": 0.80, "class_id": 0, "class_name": "mug"},
        ],
    }
    selected = _candidate_detection(
        result, depth, (50.0, 50.0, 32.0, 24.0), (0.0, 0.0, 0.0),
        (0.60, 0.0), 0.01,
    )
    _check(selected is not None, "expected an in-radius candidate")
    _check(selected["rank"] == 2, "outside-radius top candidate must be skipped")


def test_candidate_selection_needs_depth_and_prefers_hint_distance():
    depth = np.full((48, 64), 0.60, dtype=np.float32)
    result = {
        "rgb_shape": (48, 64),
        "candidates": [
            {"xyxy": (48, 20, 56, 28), "confidence": 0.95, "class_id": 0, "class_name": "mug"},
            {"xyxy": (28, 20, 36, 28), "confidence": 0.80, "class_id": 0, "class_name": "mug"},
        ],
    }
    _check(
        _candidate_detection(result, None, (50, 50, 32, 24), (0, 0, 0), (0.6, 0), 1.0) is None,
        "candidate without depth must not drive acquisition",
    )
    _check(
        _candidate_detection(result, depth, (50, 50, 32, 24), None, (0.6, 0), 1.0) is None,
        "candidate without chassis pose must not bypass reveal containment",
    )
    selected = _candidate_detection(
        result, depth, (50, 50, 32, 24), (0, 0, 0), (0.60, 0.0), 1.0,
    )
    _check(selected["rank"] == 2, "candidate nearest the reveal hint must beat confidence")


def test_no_depth_never_commands_linear_motion():
    acq = NearFieldAcquire.__new__(NearFieldAcquire)
    acq._phase, acq._phase_t0, acq._align_ok, acq._scan_sign = "align", 0.0, 1, 1.0
    linear, _ = acq._command(1.0, n_obj=1, err=0.0, fwd_obj=None)
    _check(linear == 0.0, "missing depth must never command positive linear motion")


def main():
    tests = [
        test_depth_in_bbox_ignores_bad_center_pixel,
        test_radius_gate_rejects_far_box,
        test_optical_to_arm_base_known_Rt,
        test_base_camera_calibration_and_tf_alias,
        test_gated_cmd_zeros_linear_outside_disc,
        test_fixture_box_localizes,
        test_missing_detector_fail_closed,
        test_candidate_selection_skips_box_outside_reveal,
        test_candidate_selection_needs_depth_and_prefers_hint_distance,
        test_no_depth_never_commands_linear_motion,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print("ok  %s" % fn.__name__)
        except _Fail as e:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, e))
        except Exception as e:
            failed += 1
            print("ERROR %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    if failed:
        print("%d failed" % failed)
        return 1
    print("all %d passed" % len(tests))
    return 0


if __name__ == "__main__":
    sys.exit(main())
