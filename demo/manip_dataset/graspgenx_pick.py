"""Staged GraspGen-X grasping from the robot base RGB-D camera.

The near-field detector owns global localization.  This module consumes its
saved base-depth crop, stages the arm into the repeatable camera handoff pose,
asks GraspGen-X for a top-down grasp, converts the returned gripper-base pose to
the arm's finger-pad contact frame, and executes only the short final approach.
"""

from __future__ import annotations

import math
import logging
import os
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

from marc_agent.graspgenx_policy import (
    MIN_CLOUD_POINTS,
    _unpack_grasps,
    checkpoint_dir,
    gripper_assets_dir,
    gripper_name,
    has_checkpoint,
    points_from_target,
)
from marc_agent.pick_runtime import PickRuntime, _grasp_floor_z, grasp_xyz_in_reach

from .openvla_pick import target_moved_with_gripper


DEFAULT_OBJECT_RADIUS_M = 0.14
DEFAULT_SUPPORT_BAND_M = 0.015
DEFAULT_TOPDOWN_TILT_DEG = 35.0
DEFAULT_TCP_DEPTH_M = 0.195
DEFAULT_FINAL_APPROACH_M = 0.06
DEFAULT_MIN_FWD_M = 0.32
DEFAULT_MAX_FWD_M = 0.58
DEFAULT_VERIFY_MIN_RISE_M = 0.035
DEFAULT_VERIFY_MAX_XY_DRIFT_M = 0.12
DEFAULT_SNAP_CLOSE = 0.68

_generator: Optional[Callable[[np.ndarray], object]] = None
log = logging.getLogger("participant_app")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _wrap_half_pi(yaw: float) -> float:
    """Normalize the symmetric two-finger closing axis to [-pi/2, pi/2]."""
    value = float(yaw)
    while value > 0.5 * math.pi:
        value -= math.pi
    while value < -0.5 * math.pi:
        value += math.pi
    return value


def filter_object_points(points, target) -> np.ndarray:
    """Apply arm-frame locality and support-plane filtering to a bbox cloud."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < MIN_CLOUD_POINTS or points.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    points = points[:, :3]
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] < MIN_CLOUD_POINTS:
        return np.zeros((0, 3), dtype=np.float32)

    radius = max(
        0.05,
        _env_float("MARC_GRASPGENX_OBJECT_RADIUS_M", DEFAULT_OBJECT_RADIUS_M),
    )
    cx = float(getattr(target, "fwd", np.median(points[:, 0])))
    cy = float(getattr(target, "lat", np.median(points[:, 1])))
    local = (
        (np.abs(points[:, 0] - cx) <= radius)
        & (np.abs(points[:, 1] - cy) <= radius)
    )
    if int(np.count_nonzero(local)) >= MIN_CLOUD_POINTS:
        points = points[local]

    # The pavement/table is the lowest horizontal surface in arm_base (+Z up).
    # Remove it only when it forms a real band and leaves a viable object cloud.
    band = max(
        0.005,
        _env_float("MARC_GRASPGENX_SUPPORT_BAND_M", DEFAULT_SUPPORT_BAND_M),
    )
    floor_z = float(np.percentile(points[:, 2], 8.0))
    on_support = points[:, 2] <= floor_z + band
    above = points[~on_support]
    support_fraction = float(np.mean(on_support))
    if above.shape[0] >= MIN_CLOUD_POINTS and support_fraction >= 0.08:
        points = above

    return np.ascontiguousarray(points, dtype=np.float32)


def segmented_object_points(target, client=None) -> np.ndarray:
    """Return a local, support-plane-reduced object cloud in ``arm_base``.

    YOLO supplies the segmentation prior (its inset box); robust arm-frame XY
    gating removes unrelated background, then a dense lowest-Z support band is
    removed only when enough above-plane object points remain.  This preserves
    small objects instead of allowing floor pixels to dominate GraspGen-X.
    """
    points = points_from_target(target, client=client)
    return filter_object_points(points, target)


def _score_at(scores, index: int) -> float:
    if scores is None:
        return 0.0
    flat = np.asarray(scores).reshape(-1)
    if index < 0 or index >= flat.size:
        return 0.0
    value = float(flat[index])
    return value if math.isfinite(value) else 0.0


def select_topdown_contact_pose(
    raw,
    points,
    target_xyz,
    *,
    fingertip_depth_m: float = DEFAULT_TCP_DEPTH_M,
    max_tilt_deg: float = DEFAULT_TOPDOWN_TILT_DEG,
    max_xy_m: float = DEFAULT_OBJECT_RADIUS_M,
):
    """Select a reachable top-down contact pose from GraspGen-X output.

    GraspGen-X matrices locate the Robotiq base.  Its +Z is the approach axis
    and +X is the finger closing axis.  The arm IK instead locates the midpoint
    of the finger pads, so the returned translation is shifted 0.195 m along
    local +Z before reach filtering and execution.
    """
    grasps, scores = _unpack_grasps(raw)
    if grasps is None or len(grasps) == 0:
        return None
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] < MIN_CLOUD_POINTS:
        return None
    tx, ty = float(target_xyz[0]), float(target_xyz[1])
    z_lo = float(np.min(cloud[:, 2])) - 0.04
    z_hi = float(np.max(cloud[:, 2])) + 0.12
    min_down = math.cos(math.radians(float(max_tilt_deg)))
    candidates = []
    for index, raw_pose in enumerate(grasps):
        matrix = np.asarray(raw_pose, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            continue
        approach = matrix[:3, 2]
        norm = float(np.linalg.norm(approach))
        if norm <= 1e-9:
            continue
        approach = approach / norm
        # A safe overhead grasp approaches downward in arm_base (+Z is up).
        if float(approach[2]) > -min_down:
            continue
        contact = matrix[:3, 3] + approach * float(fingertip_depth_m)
        dxy = math.hypot(float(contact[0]) - tx, float(contact[1]) - ty)
        if dxy > float(max_xy_m):
            continue
        if not (z_lo <= float(contact[2]) <= z_hi):
            continue
        closing = matrix[:2, 0]
        if float(np.linalg.norm(closing)) <= 1e-9:
            continue
        yaw = _wrap_half_pi(math.atan2(float(closing[1]), float(closing[0])))
        score = _score_at(scores, index)
        candidates.append((score, -dxy, -abs(yaw), contact, yaw, matrix, index))
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[:3], reverse=True)
    score, neg_dxy, _neg_yaw, contact, yaw, matrix, index = candidates[0]
    return {
        "xyz": tuple(float(v) for v in contact),
        "yaw": float(yaw),
        "score": float(score),
        "dyolo": float(-neg_dxy),
        "n_keep": len(candidates),
        "index": int(index),
        "matrix": matrix,
    }


def load_local_graspgenx(path: Optional[str] = None):
    """Load the already-bundled local sampler; never downloads anything."""
    global _generator
    if _generator is not None:
        return _generator
    directory = checkpoint_dir(path)
    if not has_checkpoint(directory):
        return None
    os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", directory)
    try:
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.samplers import run_planner_on_object
        from graspgenx.utils.checkpoint_io import load_model_cfg

        cfg = load_model_cfg(
            os.path.join(directory, "gen"),
            os.path.join(directory, "dis"),
            None,
            None,
        )
        assets = gripper_assets_dir()
        if not assets:
            return None
        sampler = GraspGenXSampler(cfg, gripper_name(), assets_dir=assets)
    except Exception as exc:
        log.warning(
            "[graspgenx] local sampler load failed (%s). Stop other GPU agent "
            "containers before running the pick tester.",
            exc,
        )
        return None

    def generate(points: np.ndarray):
        planner = (os.environ.get("MARC_GRASPGENX_PLANNER") or "graspmoe").strip()
        grasps, conf, _tags, _obb = run_planner_on_object(
            np.asarray(points, dtype=np.float32),
            sampler,
            planner=planner,
            grasp_threshold=_env_float("MARC_GRASPGENX_THRESHOLD", -1.0),
            num_grasps=max(16, int(_env_float("MARC_GRASPGENX_NUM_GRASPS", 64))),
            topk_num_grasps=max(8, int(_env_float("MARC_GRASPGENX_TOPK", 32))),
        )
        return grasps, conf

    _generator = generate
    return generate


def unload_local_graspgenx() -> None:
    global _generator
    _generator = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class StagedGraspGenXPickRuntime(PickRuntime):
    """Deterministic stage, GraspGen-X pose/yaw, short approach, snap, lift."""

    def __init__(self, *args, target_observer=None, points_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_observer = target_observer
        self._points_fn = points_fn or segmented_object_points
        self._visual_grasp_verified = False

    def _observe_target(self):
        if not callable(self._target_observer):
            return None
        try:
            return self._target_observer()
        except Exception as exc:
            self._emit("[pick] GraspGen-X visual verification failed (%s)" % exc)
            return None

    def _grasp(self, target) -> None:
        self._visual_grasp_verified = False
        localized_before = SimpleNamespace(
            fwd=float(target.fwd),
            lat=float(target.lat),
            z=float(target.z),
            valid=True,
        )
        try:
            points = self._points_fn(target, client=self.client)
        except TypeError:
            points = self._points_fn(target)
        n_points = 0 if points is None else int(np.asarray(points).shape[0])
        if n_points < MIN_CLOUD_POINTS:
            self._emit(
                "[pick] GraspGen-X segmented cloud too small n_pts=%d -- fail closed"
                % n_points
            )
            return

        arm = self._teleop_arm()
        if not arm.stage_for_openvla(target):
            self._emit("[pick] GraspGen-X deterministic staging failed")
            return
        self._emit(
            "[pick] GraspGen-X handoff: staged in base-camera view; "
            "sampling saved base-depth cloud"
        )

        generator = self._grasp_generator or load_local_graspgenx()
        if generator is None:
            self._emit("[pick] GraspGen-X unavailable in local image -- fail closed")
            return
        self._grasp_generator = generator
        try:
            raw = generator(np.asarray(points, dtype=np.float32))
        except Exception as exc:
            self._emit("[pick] GraspGen-X inference failed (%s) -- fail closed" % exc)
            return
        selected = select_topdown_contact_pose(
            raw,
            points,
            (float(target.fwd), float(target.lat), float(target.z)),
            fingertip_depth_m=_env_float(
                "MARC_GRASPGENX_TCP_DEPTH_M", DEFAULT_TCP_DEPTH_M
            ),
            max_tilt_deg=_env_float(
                "MARC_GRASPGENX_TOPDOWN_TILT_DEG", DEFAULT_TOPDOWN_TILT_DEG
            ),
            max_xy_m=_env_float(
                "MARC_GRASPGENX_OBJECT_RADIUS_M", DEFAULT_OBJECT_RADIUS_M
            ),
        )
        if selected is None:
            self._emit(
                "[pick] GraspGen-X produced no safe top-down contact n_pts=%d"
                % n_points
            )
            return

        fwd, lat, z = selected["xyz"]
        z = max(float(z), float(_grasp_floor_z()))
        min_fwd = _env_float("MARC_GRASPGENX_MIN_FWD_M", DEFAULT_MIN_FWD_M)
        max_fwd = _env_float("MARC_GRASPGENX_MAX_FWD_M", DEFAULT_MAX_FWD_M)
        if not grasp_xyz_in_reach((fwd, lat, z)) or not (min_fwd <= fwd <= max_fwd):
            self._emit(
                "[pick] GraspGen-X contact out of safe reach "
                "fwd=%.3f lat=%.3f z=%.3f" % (fwd, lat, z)
            )
            return
        yaw = float(selected["yaw"])
        target.fwd, target.lat, target.z = float(fwd), float(lat), float(z)
        target.grasp_yaw = yaw
        self._emit(
            "[pick] ggx-contact fwd=%.3f lat=%.3f z=%.3f yaw=%.2f "
            "score=%.3f n_pts=%d n_keep=%d dyolo=%.3f"
            % (
                fwd, lat, z, yaw, float(selected["score"]), n_points,
                int(selected["n_keep"]), float(selected["dyolo"]),
            )
        )

        above = max(
            0.03,
            _env_float("MARC_GRASPGENX_FINAL_APPROACH_M", DEFAULT_FINAL_APPROACH_M),
        )
        if not arm.move_grasp_pose((fwd, lat, z + above), yaw, "ggx-pregrasp"):
            return
        if not arm.move_grasp_pose((fwd, lat, z), yaw, "ggx-contact"):
            return
        # Runtime calibration is deliberately a primitive, not a learned output.
        if not arm.set_grip(DEFAULT_SNAP_CLOSE):
            return
        if not arm.lift_for_grasp_verification():
            self._emit("[pick] GraspGen-X verification lift failed")
            return

        after = self._observe_target()
        if after is None or getattr(after, "valid", False) is not True:
            self._emit("[pick] GraspGen-X grasp rejected: target not visible after lift")
            return
        valid = target_moved_with_gripper(
            localized_before,
            after,
            min_rise_m=_env_float(
                "MARC_GRASPGENX_VERIFY_MIN_RISE_M", DEFAULT_VERIFY_MIN_RISE_M
            ),
            max_xy_drift_m=_env_float(
                "MARC_GRASPGENX_VERIFY_MAX_XY_DRIFT_M",
                DEFAULT_VERIFY_MAX_XY_DRIFT_M,
            ),
        )
        rise = float(after.z) - float(localized_before.z)
        drift = math.hypot(
            float(after.fwd) - float(localized_before.fwd),
            float(after.lat) - float(localized_before.lat),
        )
        self._visual_grasp_verified = bool(valid)
        self._emit(
            "[pick] GraspGen-X visual grasp %s rise=%.3fm xy_drift=%.3fm"
            % ("verified" if valid else "rejected", rise, drift)
        )

    def poll_grasping(self):
        held = super().poll_grasping()
        if held is True and not self._visual_grasp_verified:
            self._emit("[pick] rejecting hold=True without base-camera lift evidence")
            return False
        return held

    def unload_generator(self) -> None:
        super().unload_generator()
        unload_local_graspgenx()

    def _recover(self) -> None:
        try:
            ok = self._teleop_arm().go_ready(open_grip=True)
            self._emit("[pick] GraspGen-X recovery -> ready (%s)" % (
                "OK" if ok else "timeout"
            ))
        except Exception as exc:
            self._emit("[pick] GraspGen-X recovery failed: %s" % exc)
        finally:
            self._teleop = None
