"""Wrist-camera Stage 2 YOLO pick: center the EE, then bbox depth for z/yaw.

Used by PickRuntime policy ``wrist`` (H key). GraspGen-X is not loaded.
Lidar is not used: the nav z filter would drop a tumbler.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np

_DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

DEFAULT_STEP_M = 0.012
DEFAULT_TOL = 0.08
DEFAULT_MAX_STEPS = 25
# Downward wrist cam (URDF rpy π,π,0): image-right → -lat, image-down → +fwd.
DEFAULT_SIGN_X = -1.0
DEFAULT_SIGN_Y = 1.0
MIN_PCA_POINTS = 8
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 0.80


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def image_errors(det) -> Optional[tuple]:
    """(err_x, err_y) in half-axis units, or None if there is no box."""
    if det is None or int(det.get("n_obj") or 0) <= 0:
        return None
    err_x = det.get("error")
    if err_x is None:
        return None
    shape = det.get("rgb_shape")
    cy = det.get("cy")
    if shape is None or cy is None or int(shape[0]) < 2:
        return float(err_x), 0.0
    h = float(shape[0])
    err_y = float(np.clip((float(cy) - 0.5 * (h - 1.0)) / (0.5 * (h - 1.0)), -1.0, 1.0))
    return float(err_x), err_y


def delta_from_errors(err_x, err_y, step_m: float = DEFAULT_STEP_M) -> tuple:
    """Map image-right / image-down errors onto arm_base (fwd, lat) metres."""
    step = max(0.001, float(step_m))
    sx = _env_float("MARC_WRIST_SIGN_X", DEFAULT_SIGN_X)
    sy = _env_float("MARC_WRIST_SIGN_Y", DEFAULT_SIGN_Y)
    dx = float(np.clip(sy * float(err_y) * step, -step, step))
    dy = float(np.clip(sx * float(err_x) * step, -step, step))
    return dx, dy


def _decode_rgb(msg):
    if msg is None:
        return None
    if isinstance(msg, np.ndarray) and msg.size > 0:
        return msg
    try:
        from manip_dataset.images import image_to_rgb
        rgb = image_to_rgb(msg)
        if rgb is not None:
            return rgb
    except Exception:
        pass
    try:
        from nearfield import _image_to_rgb
        return _image_to_rgb(msg)
    except Exception:
        return None


def _gripper_rgb(client):
    getter = getattr(client, "get_robot_image", None)
    if not callable(getter):
        return None
    try:
        return _decode_rgb(getter("gripper_left"))
    except Exception:
        return None


def center_over_bbox(
    client,
    detector,
    arm,
    name,
    *,
    tol: float = DEFAULT_TOL,
    max_steps: int = DEFAULT_MAX_STEPS,
    step_m: float = DEFAULT_STEP_M,
    log=None,
) -> bool:
    """YOLO on gripper_left + EE-delta until the box is centered. False = no box."""
    if detector is None or arm is None:
        return False
    setter = getattr(detector, "set_target", None)
    if callable(setter):
        setter(name)
    last_err = None
    for i in range(max(1, int(max_steps))):
        rgb = _gripper_rgb(client)
        if rgb is None:
            if callable(log):
                log("[wrist] no gripper RGB")
            return False
        det = detector.detect(rgb)
        errs = image_errors(det)
        if errs is None:
            if callable(log):
                log("[wrist] no gripper box")
            return False
        err_x, err_y = errs
        last_err = (err_x, err_y)
        if abs(err_x) <= float(tol) and abs(err_y) <= float(tol):
            if callable(log):
                log(
                    "[wrist] center err_x=%.3f err_y=%.3f steps=%d"
                    % (err_x, err_y, i)
                )
            return True
        dx, dy = delta_from_errors(err_x, err_y, step_m=step_m)
        arm.apply_delta(dx=dx, dy=dy, dz=0.0)
    if callable(log) and last_err is not None:
        log("[wrist] center timeout err_x=%.3f err_y=%.3f" % last_err)
    return True


def yaw_from_points_xy(points) -> Optional[float]:
    """Principal-axis yaw in arm_base XY, wrapped to [-pi/2, pi/2]."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < MIN_PCA_POINTS or pts.shape[1] < 2:
        return None
    xy = pts[:, :2]
    xy = xy - xy.mean(axis=0)
    cov = xy.T @ xy / max(1.0, float(xy.shape[0]))
    vals, vecs = np.linalg.eigh(cov)
    if float(vals[-1]) < 1e-8:
        return None
    axis = vecs[:, -1]
    yaw = math.atan2(float(axis[1]), float(axis[0]))
    while yaw > 0.5 * math.pi:
        yaw -= math.pi
    while yaw < -0.5 * math.pi:
        yaw += math.pi
    return float(yaw)


def _pinhole(k) -> Optional[tuple]:
    if k is None:
        return None
    km = np.asarray(k, dtype=np.float64).reshape(-1)
    if km.size >= 9:
        fx, fy, cx, cy = float(km[0]), float(km[4]), float(km[2]), float(km[5])
    elif km.size >= 4:
        fx, fy, cx, cy = float(km[0]), float(km[1]), float(km[2]), float(km[3])
    else:
        return None
    if fx <= 1.0 or fy <= 1.0:
        return None
    return fx, fy, cx, cy


def _bbox_ij(depth, bbox, rgb_shape):
    arr = np.asarray(depth)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0 or bbox is None:
        return None, None
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if rgb_shape is not None:
        sy = h / max(1.0, float(rgb_shape[0]))
        sx = w / max(1.0, float(rgb_shape[1]))
        x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
    ix1 = max(0, int(math.floor(x1)))
    iy1 = max(0, int(math.floor(y1)))
    ix2 = min(w, int(math.ceil(x2)))
    iy2 = min(h, int(math.ceil(y2)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None, None
    return arr, (ix1, iy1, ix2, iy2)


def _bbox_optical_points(depth, bbox, rgb_shape, k, stride: int = 2) -> np.ndarray:
    arr, ij = _bbox_ij(depth, bbox, rgb_shape)
    if arr is None or ij is None:
        return np.zeros((0, 3), dtype=np.float64)
    pin = _pinhole(k)
    if pin is None:
        return np.zeros((0, 3), dtype=np.float64)
    fx, fy, cx, cy = pin
    ix1, iy1, ix2, iy2 = ij
    ys, xs = np.mgrid[iy1:iy2:stride, ix1:ix2:stride]
    z = arr[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z > DEPTH_MIN_M) & (z < DEPTH_MAX_M)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def yaw_from_bbox_depth(depth, bbox, rgb_shape, k, cam_T) -> Optional[float]:
    """PCA yaw of gripper-depth points inside ``bbox``, in arm_base."""
    optical = _bbox_optical_points(depth, bbox, rgb_shape, k)
    if optical.shape[0] < MIN_PCA_POINTS or cam_T is None:
        return None
    T = np.asarray(cam_T, dtype=np.float64)
    if T.shape != (4, 4):
        return None
    ones = np.ones((optical.shape[0], 1), dtype=np.float64)
    arm = (T @ np.concatenate([optical, ones], axis=1).T).T[:, :3]
    return yaw_from_points_xy(arm)


def _bbox_median_depth(depth, bbox, rgb_shape) -> Optional[float]:
    arr, ij = _bbox_ij(depth, bbox, rgb_shape)
    if arr is None or ij is None:
        return None
    ix1, iy1, ix2, iy2 = ij
    crop = arr[iy1:iy2, ix1:ix2]
    valid = crop[np.isfinite(crop) & (crop > DEPTH_MIN_M) & (crop < DEPTH_MAX_M)]
    if valid.size < 4:
        return None
    return float(np.median(valid))


def _rewrite_xy(seq, x, y) -> None:
    for i, wp in enumerate(seq._waypoints):
        label, ee, grip, hold, mode = wp
        if label in ("pre", "descend", "grasp", "lift"):
            ee = np.asarray(ee, dtype=np.float64).copy()
            ee[0] = float(x)
            ee[1] = float(y)
            seq._waypoints[i] = (label, ee, grip, hold, mode)
    tpos = getattr(seq, "_target_pos", None)
    if tpos is not None:
        tpos = np.asarray(tpos, dtype=np.float64).copy()
        tpos[0] = float(x)
        tpos[1] = float(y)
        seq._target_pos = tpos


def _rewrite_z(seq, z) -> None:
    from arm_pick import _rewrite_grasp_z
    _rewrite_grasp_z(seq, z)


def refine_at_pre(client, detector, arm, seq, p, kin, log) -> None:
    """At pre dwell: center on gripper YOLO, then bbox depth z + PCA yaw."""
    emit = log if callable(log) else None
    name = str(p.get("target_name") or os.environ.get("MARC_TARGET_OBJECT") or "tumbler")
    ok = center_over_bbox(client, detector, arm, name, log=emit)
    if not ok and emit:
        emit("[wrist] center failed — keeping seed XY")

    from arm_pick import (
        _camera_depth_T,
        _gripper_camera_k,
        _gripper_depth_array,
        _read_arm_q,
        grasp_z_from_wrist_depth,
        load_params,
    )

    q = _read_arm_q(client)
    ee = None
    if q is not None and kin is not None:
        try:
            ee = kin.fk(q)[:3, 3]
        except Exception:
            ee = None
    if ee is not None:
        _rewrite_xy(seq, float(ee[0]), float(ee[1]))
        p["grasp_fwd"] = float(ee[0])
        p["grasp_lat"] = float(ee[1])

    depth = _gripper_depth_array(client)
    rgb = _gripper_rgb(client)
    det = None
    if detector is not None and rgb is not None:
        setter = getattr(detector, "set_target", None)
        if callable(setter):
            setter(name)
        det = detector.detect(rgb)
    bbox = None if det is None else det.get("bbox")
    rgb_shape = None if det is None else det.get("rgb_shape")
    cam_T = _camera_depth_T(kin, q) if q is not None else None
    k = _gripper_camera_k(client)
    floor_z = float((p or load_params()).get("grasp_down", -0.28))
    d = _bbox_median_depth(depth, bbox, rgb_shape)
    z = None
    if d is not None and cam_T is not None:
        uv = None
        if bbox is not None:
            uv = (
                0.5 * (float(bbox[0]) + float(bbox[2])),
                0.5 * (float(bbox[1]) + float(bbox[3])),
            )
        z = grasp_z_from_wrist_depth(cam_T, d, floor_z, k=k, uv=uv)
    if z is not None:
        _rewrite_z(seq, z)
        p["grasp_down"] = float(z)

    yaw = yaw_from_bbox_depth(depth, bbox, rgb_shape, k, cam_T)
    if yaw is not None:
        p["grasp_yaw"] = float(yaw)
        try:
            import arm_kin
            base_quat = arm_kin.R_to_quat(kin.fk(arm_kin.DEFAULT_ARM_Q)[:3, :3])
            hold = arm_kin.quat_mul(
                base_quat, arm_kin.rotvec_to_quat(np.array([0.0, 0.0, float(yaw)]))
            )
            n = float(np.linalg.norm(hold))
            if n > 0:
                seq._default_quat = hold / n
        except Exception:
            pass

    if emit:
        emit(
            "[wrist] ee=(%.3f,%.3f,%.3f) z=%s yaw=%s"
            % (
                float(p.get("grasp_fwd", float("nan"))),
                float(p.get("grasp_lat", float("nan"))),
                float(ee[2]) if ee is not None else float("nan"),
                "--" if z is None else "%.3f" % z,
                "--" if yaw is None else "%.2f" % yaw,
            )
        )
