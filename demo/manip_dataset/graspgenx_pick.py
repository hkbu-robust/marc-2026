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
from marc_agent.pick_runtime import PickRuntime

from .grasp_profiles import bbox_is_upright, resolve_grasp_profile
from .openvla_pick import target_moved_with_gripper


DEFAULT_OBJECT_RADIUS_M = 0.14
DEFAULT_SUPPORT_BAND_M = 0.015
DEFAULT_TOPDOWN_TILT_DEG = 35.0
DEFAULT_TCP_DEPTH_M = 0.195
DEFAULT_FINAL_APPROACH_M = 0.06
DEFAULT_STAGE_CLEARANCE_M = 0.23
DEFAULT_OVERHEAD_CLEARANCE_M = 0.08
DEFAULT_FLOOR_CLEARANCE_M = 0.08
DEFAULT_MAX_GRASP_WIDTH_M = 0.12
# Cropped non-tumbler clouds are often slightly inflated by the support plane.
# An observed 1.46-aspect cola can still has a stable narrow PCA axis; keep
# near-round objects neutral while allowing those elongated objects to yaw.
DEFAULT_MIN_DIRECTIONAL_ASPECT = 1.30
DEFAULT_UPRIGHT_BBOX_ASPECT = 1.20
DEFAULT_ISOTROPIC_YAW = -0.5 * math.pi
DEFAULT_MIN_INSERTION_FRACTION = 0.30
DEFAULT_CONTACT_DEPTH_FRACTION = 0.35
DEFAULT_CONTACT_IK_BIAS_M = 0.0
DEFAULT_CONTACT_UP_BIAS_M = 0.0
DEFAULT_CONTACT_FWD_BIAS_M = 0.0
DEFAULT_CONTACT_LAT_BIAS_M = 0.0
DEFAULT_MAX_RELOCALIZATION_DRIFT_M = 0.04
DEFAULT_MAX_CLOUD_CENTER_SHIFT_M = 0.015
DEFAULT_WRIST_SEARCH_RADIUS_M = 0.10
DEFAULT_WRIST_DEPTH_GATE_M = 0.06
DEFAULT_WRIST_MAX_CORRECTION_M = 0.02
DEFAULT_WRIST_MAX_DISAGREEMENT_M = 0.07
DEFAULT_WRIST_MIN_POINTS = 20
DEFAULT_MIN_FWD_M = 0.32
DEFAULT_MAX_FWD_M = 0.58
DEFAULT_MAX_ABS_LAT_M = 0.45
DEFAULT_VERIFY_MIN_RISE_M = 0.02
DEFAULT_VERIFY_MAX_RISE_M = 0.20
DEFAULT_VERIFY_MAX_XY_DRIFT_M = 0.20
DEFAULT_RETENTION_CLOSE = 0.703
DEFAULT_GRASP_CLOSE = 0.703
DEFAULT_HOLD_CONFIRM_SAMPLES = 3

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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return int(default)


def _wrap_half_pi(yaw: float) -> float:
    """Normalize the symmetric two-finger closing axis to [-pi/2, pi/2]."""
    value = float(yaw)
    while value > 0.5 * math.pi:
        value -= math.pi
    while value < -0.5 * math.pi:
        value += math.pi
    return value


def yolo_bbox_is_upright(target, min_aspect: float = DEFAULT_UPRIGHT_BBOX_ASPECT) -> bool:
    """Use the unobstructed detector box to distinguish standing from lying objects."""
    return bbox_is_upright(target, min_aspect)


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


def object_vertical_extent(points, target_z: float) -> tuple[float, float]:
    """Estimate the local support and object top in the arm-base frame."""
    z_values = np.asarray(points, dtype=np.float64)[:, 2]
    top = float(np.percentile(z_values, 95.0))
    support = min(float(np.percentile(z_values, 5.0)), float(target_z))
    return support, top


def contact_z_from_extent(
    support: float,
    top: float,
    *,
    contact_depth_fraction: Optional[float] = None,
    floor_clearance_m: Optional[float] = None,
) -> Optional[float]:
    """Choose a floor-safe pad midpoint from a measured vertical extent."""
    height = max(0.02, float(top) - float(support))
    configured_fraction = (
        _env_float(
            "MARC_GRASPGENX_CONTACT_DEPTH_FRACTION",
            DEFAULT_CONTACT_DEPTH_FRACTION,
        )
        if contact_depth_fraction is None
        else float(contact_depth_fraction)
    )
    fraction = min(0.80, max(0.25, configured_fraction))
    contact = float(top) - fraction * height - _env_float(
        "MARC_GRASPGENX_CONTACT_IK_BIAS_M", DEFAULT_CONTACT_IK_BIAS_M
    )
    configured_clearance = (
        _env_float(
            "MARC_GRASPGENX_FLOOR_CLEARANCE_M", DEFAULT_FLOOR_CLEARANCE_M
        )
        if floor_clearance_m is None
        else float(floor_clearance_m)
    )
    z_lo = float(support) + max(0.01, configured_clearance)
    z_hi = float(top) - 0.005
    if z_lo > z_hi:
        return None
    contact = max(z_lo, contact) + max(
        0.0,
        _env_float(
            "MARC_GRASPGENX_CONTACT_UP_BIAS_M", DEFAULT_CONTACT_UP_BIAS_M
        ),
    )
    return min(z_hi, contact)


def object_contact_z(
    points,
    target_z: float,
    *,
    contact_depth_fraction: Optional[float] = None,
    floor_clearance_m: Optional[float] = None,
) -> Optional[float]:
    """Choose a finger-pad height without entering the local support plane.

    The lower object envelope is a conservative support-plane estimate: if the
    bbox crop did not retain pavement pixels, treating the object's visible
    bottom as support only raises the grasp. Return ``None`` when the observed
    object is too short to satisfy both floor clearance and object contact.
    """
    support, top = object_vertical_extent(points, target_z)
    return contact_z_from_extent(
        support,
        top,
        contact_depth_fraction=contact_depth_fraction,
        floor_clearance_m=floor_clearance_m,
    )


def closing_axis_width(points, closing_xy) -> float:
    """Robust object width along one candidate finger-closing axis."""
    cloud = np.asarray(points, dtype=np.float64)
    axis = np.asarray(closing_xy, dtype=np.float64).reshape(-1)[:2]
    norm = float(np.linalg.norm(axis))
    if cloud.ndim != 2 or cloud.shape[1] < 2 or norm <= 1e-9:
        return float("inf")
    projected = cloud[:, :2] @ (axis / norm)
    return float(np.percentile(projected, 95.0) - np.percentile(projected, 5.0))


def object_xy_footprint(points) -> dict:
    """Fit a robust top-down footprint to the YOLO-cropped object cloud.

    The axis-aligned RGB box supplies the object crop but cannot encode an
    in-plane object angle. PCA on the deprojected crop supplies that angle;
    percentile extents keep a few depth-edge outliers from defining the width.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] < MIN_CLOUD_POINTS or cloud.shape[1] < 2:
        raise ValueError("object cloud is too small for an XY footprint")
    xy = cloud[:, :2]
    seed = np.median(xy, axis=0)
    centered = xy - seed
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    _values, axes = np.linalg.eigh(covariance)
    projected = centered @ axes
    lo = np.percentile(projected, 5.0, axis=0)
    hi = np.percentile(projected, 95.0, axis=0)
    extents = hi - lo
    narrow_i = int(np.argmin(extents))
    long_i = 1 - narrow_i
    center = seed + axes @ (0.5 * (lo + hi))
    narrow_axis = axes[:, narrow_i]
    width = float(extents[narrow_i])
    length = float(extents[long_i])
    aspect = length / max(width, 1e-6)
    yaw = _wrap_half_pi(
        math.atan2(float(narrow_axis[1]), float(narrow_axis[0]))
    )
    return {
        "center": center,
        "yaw": float(yaw),
        "width": width,
        "length": length,
        "aspect": float(aspect),
    }


def bounded_xy_center(reference_xy, cloud_xy, max_shift_m: float) -> np.ndarray:
    """Use cloud centering without allowing a visible-surface snap off target."""
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(2)
    cloud = np.asarray(cloud_xy, dtype=np.float64).reshape(2)
    delta = cloud - reference
    distance = float(np.linalg.norm(delta))
    limit = max(0.0, float(max_shift_m))
    if distance <= limit or distance <= 1e-9:
        return cloud.copy()
    return reference + delta * (limit / distance)


def upright_rear_insertion_waypoints(
    contact_xyz, rear_standoff_m: float, fold_z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold and descend at a fixed rear point, then insert straight forward.

    The upright path must never move over the object before the wrist is fully
    folded. ``rear_standoff_m`` is forward-axis clearance from contact,
    independent of the GraspGen-X yaw and the profile's wrist tilt.
    """
    contact = np.asarray(contact_xyz, dtype=np.float64).reshape(3)
    pregrasp = contact.copy()
    pregrasp[0] -= max(0.0, float(rear_standoff_m))
    fold_high = pregrasp.copy()
    fold_high[2] = max(float(fold_z), float(pregrasp[2]))
    return fold_high, pregrasp, contact


def wrist_depth_object_observation(
    depth,
    k,
    cam_T,
    expected_xyz,
    *,
    search_radius_m: float = DEFAULT_WRIST_SEARCH_RADIUS_M,
    depth_gate_m: float = DEFAULT_WRIST_DEPTH_GATE_M,
    min_points: int = DEFAULT_WRIST_MIN_POINTS,
) -> Optional[dict]:
    """Measure the expected object's visible surface from the wrist depth image.

    The base-camera pose predicts a pixel and depth in the wrist camera. Only a
    small metric window around that prediction is accepted, and points must also
    agree in depth. This prevents the floor or fingers from becoming a new target.
    """
    arr = np.asarray(depth) if depth is not None else np.zeros((0, 0))
    if arr.ndim == 3:
        arr = arr[..., 0]
    T = np.asarray(cam_T, dtype=np.float64) if cam_T is not None else np.zeros((0, 0))
    km = np.asarray(k, dtype=np.float64).reshape(-1) if k is not None else np.zeros(0)
    if arr.ndim != 2 or arr.size == 0 or T.shape != (4, 4):
        return None
    if km.size >= 9:
        fx, fy, cx, cy = float(km[0]), float(km[4]), float(km[2]), float(km[5])
    elif km.size >= 4:
        fx, fy, cx, cy = (float(value) for value in km[:4])
    else:
        return None
    if fx <= 1.0 or fy <= 1.0:
        return None
    expected = np.append(np.asarray(expected_xyz, dtype=np.float64).reshape(3), 1.0)
    try:
        optical_expected = np.linalg.inv(T) @ expected
    except np.linalg.LinAlgError:
        return None
    predicted_depth = float(optical_expected[2])
    if not (0.05 < predicted_depth < 0.80):
        return None
    u = fx * float(optical_expected[0]) / predicted_depth + cx
    v = fy * float(optical_expected[1]) / predicted_depth + cy
    radius = max(0.02, float(search_radius_m))
    rx = max(2, int(math.ceil(fx * radius / predicted_depth)))
    ry = max(2, int(math.ceil(fy * radius / predicted_depth)))
    h, w = arr.shape
    x1, x2 = max(0, int(math.floor(u)) - rx), min(w, int(math.ceil(u)) + rx + 1)
    y1, y2 = max(0, int(math.floor(v)) - ry), min(h, int(math.ceil(v)) + ry + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    ys, xs = np.mgrid[y1:y2:2, x1:x2:2]
    values = arr[ys, xs].astype(np.float64)
    gate = max(0.01, float(depth_gate_m))
    valid = (
        np.isfinite(values)
        & (values > 0.05)
        & (values < 0.80)
        & (np.abs(values - predicted_depth) <= gate)
    )
    if int(np.count_nonzero(valid)) < max(4, int(min_points)):
        return None
    z = values[valid]
    px = xs[valid].astype(np.float64)
    py = ys[valid].astype(np.float64)
    optical = np.column_stack(
        ((px - cx) * z / fx, (py - cy) * z / fy, z, np.ones_like(z))
    )
    arm_points = (T @ optical.T).T[:, :3]
    # Keep the visible object surface near the predicted top, not support below it.
    expected_top = float(expected[2])
    surface = arm_points[
        np.abs(arm_points[:, 2] - expected_top) <= gate
    ]
    if surface.shape[0] < max(4, int(min_points)):
        return None
    center = np.median(surface[:, :2], axis=0)
    return {
        "center": center,
        "top_z": float(np.percentile(surface[:, 2], 90.0)),
        "depth_m": float(np.median(z)),
        "n_points": int(surface.shape[0]),
        "predicted_depth_m": predicted_depth,
    }


def select_topdown_contact_pose(
    raw,
    points,
    target_xyz,
    *,
    fingertip_depth_m: float = DEFAULT_TCP_DEPTH_M,
    max_tilt_deg: float = DEFAULT_TOPDOWN_TILT_DEG,
    max_xy_m: float = DEFAULT_OBJECT_RADIUS_M,
    max_grasp_width_m: float = DEFAULT_MAX_GRASP_WIDTH_M,
    force_neutral_yaw: bool = False,
    diagnostics: Optional[dict] = None,
):
    """Select a reachable top-down contact pose from GraspGen-X output.

    GraspGen-X matrices locate the Robotiq base.  Its +Z is the approach axis
    and +X is the finger closing axis.  The arm IK instead locates the midpoint
    of the finger pads, so the returned translation is shifted 0.195 m along
    local +Z before reach filtering and execution.
    """
    grasps, scores = _unpack_grasps(raw)
    if grasps is None or len(grasps) == 0:
        if diagnostics is not None:
            diagnostics.update(reason="no_grasps", total=0)
        return None
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] < MIN_CLOUD_POINTS:
        if diagnostics is not None:
            diagnostics.update(reason="small_cloud", total=len(grasps))
        return None
    tx, ty = float(target_xyz[0]), float(target_xyz[1])
    footprint = object_xy_footprint(cloud)
    center_xy = np.asarray(footprint["center"], dtype=np.float64)
    center_dxy = math.hypot(float(center_xy[0]) - tx, float(center_xy[1]) - ty)
    if center_dxy > float(max_xy_m):
        if diagnostics is not None:
            diagnostics.update(
                reason="footprint_center",
                total=len(grasps),
                center_dxy=center_dxy,
                max_xy=float(max_xy_m),
            )
        return None
    directional = (
        not force_neutral_yaw
        and footprint["aspect"]
        >= _env_float(
            "MARC_GRASPGENX_MIN_DIRECTIONAL_ASPECT",
            DEFAULT_MIN_DIRECTIONAL_ASPECT,
        )
    )
    top_z = float(np.percentile(cloud[:, 2], 95.0))
    bottom_z = min(float(np.percentile(cloud[:, 2], 5.0)), float(target_xyz[2]))
    height = max(0.02, top_z - bottom_z)
    z_lo = bottom_z - 0.01
    z_hi = top_z - height * _env_float(
        "MARC_GRASPGENX_MIN_INSERTION_FRACTION",
        DEFAULT_MIN_INSERTION_FRACTION,
    )
    min_down = math.cos(math.radians(float(max_tilt_deg)))
    rejected = {
        "shape": 0,
        "tilt": 0,
        "xy": 0,
        "z": 0,
        "closing": 0,
        "width": 0,
    }
    contact_z_min = float("inf")
    contact_z_max = float("-inf")
    candidates = []
    for index, raw_pose in enumerate(grasps):
        matrix = np.asarray(raw_pose, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            rejected["shape"] += 1
            continue
        approach = matrix[:3, 2]
        norm = float(np.linalg.norm(approach))
        if norm <= 1e-9:
            rejected["tilt"] += 1
            continue
        approach = approach / norm
        # A safe overhead grasp approaches downward in arm_base (+Z is up).
        if float(approach[2]) > -min_down:
            rejected["tilt"] += 1
            continue
        contact = matrix[:3, 3] + approach * float(fingertip_depth_m)
        contact_z_min = min(contact_z_min, float(contact[2]))
        contact_z_max = max(contact_z_max, float(contact[2]))
        dxy = math.hypot(float(contact[0]) - tx, float(contact[1]) - ty)
        if dxy > float(max_xy_m):
            rejected["xy"] += 1
            continue
        if not (z_lo <= float(contact[2]) <= z_hi):
            rejected["z"] += 1
            continue
        closing = matrix[:2, 0]
        if float(np.linalg.norm(closing)) <= 1e-9:
            rejected["closing"] += 1
            continue
        if directional:
            # Finalize yaw from the whole bbox-cropped object footprint. This
            # prevents a high-scoring end grasp from putting one finger on the
            # base and the other on the top of an elongated object.
            yaw = float(footprint["yaw"])
            width = float(footprint["width"])
        else:
            # Upright/near-round footprints have no meaningful PCA direction.
            # Keep the physical jaw axis at its default world yaw, which avoids
            # a large arbitrary wrist rotation for a rotationally symmetric item.
            yaw = _wrap_half_pi(
                _env_float("MARC_GRASPGENX_ISOTROPIC_YAW", DEFAULT_ISOTROPIC_YAW)
            )
            width = closing_axis_width(
                cloud, (math.cos(yaw), math.sin(yaw))
            )
        if width > float(max_grasp_width_m):
            rejected["width"] += 1
            continue
        score = _score_at(scores, index)
        center_offset = float(np.linalg.norm(contact[:2] - center_xy))
        # Aperture fit is a hard constraint. Prefer the candidate closest to
        # the footprint centre before model confidence; execution then snaps
        # XY to that centre cross-section and uses the finalized narrow yaw.
        candidates.append(
            (
                -width,
                -center_offset,
                score,
                -dxy,
                -abs(yaw),
                contact,
                yaw,
                matrix,
                index,
            )
        )
    if not candidates:
        if diagnostics is not None:
            diagnostics.update(
                reason="all_filtered",
                total=len(grasps),
                rejected=rejected,
                z_window=(z_lo, z_hi),
                contact_z_range=(contact_z_min, contact_z_max),
                footprint=(
                    float(footprint["width"]),
                    float(footprint["length"]),
                    float(footprint["aspect"]),
                ),
                directional=bool(directional),
                center_dxy=center_dxy,
            )
        return None
    candidates.sort(key=lambda row: row[:5], reverse=True)
    (
        neg_width,
        neg_center_offset,
        score,
        neg_dxy,
        _neg_yaw,
        contact,
        yaw,
        matrix,
        index,
    ) = candidates[0]
    centered_contact = np.asarray(contact, dtype=np.float64).copy()
    centered_contact[:2] = center_xy
    return {
        "xyz": tuple(float(v) for v in centered_contact),
        "candidate_xyz": tuple(float(v) for v in contact),
        "yaw": float(yaw),
        "score": float(score),
        "dyolo": float(-neg_dxy),
        "width": float(-neg_width),
        "length": float(footprint["length"]),
        "aspect": float(footprint["aspect"]),
        "center_offset": float(-neg_center_offset),
        "directional": bool(directional),
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
            topk_num_grasps=-1,
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

    def __init__(
        self,
        *args,
        target_observer=None,
        visual_target_observer=None,
        points_fn=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._target_observer = target_observer
        self._visual_target_observer = visual_target_observer or target_observer
        self._points_fn = points_fn or segmented_object_points
        self._visual_grasp_verified = False

    def _observe_target(self, observer=None):
        observer = observer or self._target_observer
        if not callable(observer):
            return None
        try:
            return observer()
        except Exception as exc:
            self._emit("[pick] GraspGen-X visual verification failed (%s)" % exc)
            return None

    def _confirmed_hold_after_lift(self) -> bool:
        """Require consecutive hold samples before camera-based verification."""
        samples = max(
            1,
            _env_int(
                "MARC_GRASPGENX_HOLD_CONFIRM_SAMPLES",
                DEFAULT_HOLD_CONFIRM_SAMPLES,
            ),
        )
        for index in range(samples):
            try:
                if self.client.is_grasping() is not True:
                    return False
            except Exception:
                return False
            if index + 1 < samples:
                self._sleep(self.poll_dt)
        return True

    def _wrist_depth_refinement(
        self, arm, expected_xyz, *, profile=None, allow_fwd_correction: bool = True
    ) -> Optional[dict]:
        """Fuse a bounded local wrist-depth observation with the base-camera pose."""
        search_radius_m = (
            profile.wrist_search_radius_m
            if profile is not None
            else _env_float(
                "MARC_GRASPGENX_WRIST_SEARCH_RADIUS_M",
                DEFAULT_WRIST_SEARCH_RADIUS_M,
            )
        )
        depth_gate_m = (
            profile.wrist_depth_gate_m
            if profile is not None
            else _env_float(
                "MARC_GRASPGENX_WRIST_DEPTH_GATE_M",
                DEFAULT_WRIST_DEPTH_GATE_M,
            )
        )
        min_points = (
            profile.wrist_min_points
            if profile is not None
            else _env_int(
                "MARC_GRASPGENX_WRIST_MIN_POINTS",
                DEFAULT_WRIST_MIN_POINTS,
            )
        )
        try:
            from arm_pick import _camera_depth_T, _gripper_camera_k, _gripper_depth_array
            from .arm_control import read_arm_q

            q = read_arm_q(self.client)
            if q is None:
                raise RuntimeError("no arm joint state")
            observation = wrist_depth_object_observation(
                _gripper_depth_array(self.client),
                _gripper_camera_k(self.client),
                _camera_depth_T(arm.kin, q),
                expected_xyz,
                search_radius_m=search_radius_m,
                depth_gate_m=depth_gate_m,
                min_points=max(4, min_points),
            )
        except Exception as exc:
            self._emit("[pick] wrist-depth unavailable (%s); keeping base pose" % exc)
            return None
        if observation is None:
            self._emit("[pick] wrist-depth found no consistent object surface")
            return None
        base_xy = np.asarray(expected_xyz[:2], dtype=np.float64)
        wrist_xy = np.asarray(observation["center"], dtype=np.float64)
        disagreement = float(np.linalg.norm(wrist_xy - base_xy))
        max_disagreement_m = (
            profile.wrist_max_disagreement_m
            if profile is not None
            else _env_float(
                "MARC_GRASPGENX_WRIST_MAX_DISAGREEMENT_M",
                DEFAULT_WRIST_MAX_DISAGREEMENT_M,
            )
        )
        limit = max(0.01, max_disagreement_m)
        if disagreement > limit:
            self._emit(
                "[pick] wrist-depth rejected disagreement=%.3fm limit=%.3fm"
                % (disagreement, limit)
            )
            return None
        max_correction_m = (
            profile.wrist_max_correction_m
            if profile is not None
            else _env_float(
                "MARC_GRASPGENX_WRIST_MAX_CORRECTION_M",
                DEFAULT_WRIST_MAX_CORRECTION_M,
            )
        )
        correction_target = wrist_xy.copy()
        if not allow_fwd_correction:
            # A front-facing wrist camera observes the near surface of an
            # upright body, not its forward centre. Keep cloud-derived forward
            # centering while still accepting wrist lateral correction.
            correction_target[0] = base_xy[0]
        fused_xy = bounded_xy_center(
            base_xy, correction_target, max(0.0, max_correction_m)
        )
        observation["fused_center"] = fused_xy
        observation["disagreement_m"] = disagreement
        self._emit(
            "[pick] wrist-depth n=%d depth=%.3fm predicted=%.3fm "
            "raw_delta=(%+.3f,%+.3f) applied_delta=(%+.3f,%+.3f) top_z=%.3f"
            % (
                observation["n_points"],
                observation["depth_m"],
                observation["predicted_depth_m"],
                wrist_xy[0] - base_xy[0],
                wrist_xy[1] - base_xy[1],
                fused_xy[0] - base_xy[0],
                fused_xy[1] - base_xy[1],
                observation["top_z"],
            )
        )
        return observation

    def _grasp(self, target) -> None:
        self._visual_grasp_verified = False
        profile = resolve_grasp_profile(target)
        side_profile = bool(
            profile is not None and profile.approach_style == "front_side"
        )
        self._emit(
            "[pick] grasp profile=%s status=%s"
            % (
                profile.name if profile is not None else "generic_unvalidated",
                "unvalidated"
                if profile is None
                else (
                    "validated" if profile.physically_validated else "experimental"
                ),
            )
        )
        if profile is not None and profile.approach_style == "flip_required":
            self._emit(
                "[pick] lying cracker_box requires calibrated flip-and-reacquire "
                "sequence -- no unsafe wide pinch attempted"
            )
            return
        contact_fwd_bias = _env_float(
            "MARC_GRASPGENX_CONTACT_FWD_BIAS_M", DEFAULT_CONTACT_FWD_BIAS_M
        )
        contact_lat_bias = _env_float(
            "MARC_GRASPGENX_CONTACT_LAT_BIAS_M", DEFAULT_CONTACT_LAT_BIAS_M
        )
        cloud_localization = SimpleNamespace(
            fwd=float(target.fwd),
            lat=float(target.lat),
            z=float(target.z),
            valid=True,
        )
        localized_before = cloud_localization
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

        cloud_footprint = object_xy_footprint(points)
        cloud_center = np.asarray(cloud_footprint["center"], dtype=np.float64)
        localized_xy = np.array(
            [cloud_localization.fwd, cloud_localization.lat], dtype=np.float64
        )
        max_cloud_center_shift = max(
            0.0,
            profile.max_cloud_center_shift_m
            if profile is not None
            else _env_float(
                "MARC_GRASPGENX_MAX_CLOUD_CENTER_SHIFT_M",
                DEFAULT_MAX_CLOUD_CENTER_SHIFT_M,
            ),
        )
        execution_center = bounded_xy_center(
            localized_xy, cloud_center, max_cloud_center_shift
        )

        arm = self._teleop_arm()
        support_z, top_z = object_vertical_extent(points, float(target.z))
        stage_clearance = max(
            0.10,
            profile.stage_clearance_m
            if profile is not None
            else _env_float(
                "MARC_GRASPGENX_STAGE_CLEARANCE_M", DEFAULT_STAGE_CLEARANCE_M
            ),
        )
        side_contact_z = None
        side_object_lat = None
        side_grasp_lat = None
        if side_profile:
            # Build the upright handoff only from the saved, unoccluded base
            # cloud. The arm reaches the folded rear pregrasp before inference.
            if profile.side_floor_load:
                # A short object must be loaded from floor level rather than
                # pinched at the tumbler's elevated side-contact fraction. The
                # value is the pad midpoint; the guarded wrist fold still keeps
                # all tracked gripper frames above the measured support plane.
                side_contact_z = float(support_z) + float(
                    profile.floor_clearance_m
                )
            else:
                side_contact_z = contact_z_from_extent(
                    support_z,
                    top_z,
                    contact_depth_fraction=profile.contact_depth_fraction,
                    floor_clearance_m=profile.floor_clearance_m,
                )
            if side_contact_z is None:
                self._emit(
                    "[pick] GraspGen-X object too short for upright side handoff "
                    "support_z=%.3f top_z=%.3f" % (support_z, top_z)
                )
                return
            side_object_lat = float(execution_center[1]) + contact_lat_bias
            side_grasp_lat = side_object_lat + float(profile.pad_lateral_bias_m)
            stage_target = SimpleNamespace(
                fwd=float(execution_center[0]) + contact_fwd_bias,
                lat=side_grasp_lat,
                z=float(side_contact_z),
                support_z=float(support_z),
                fold_z=float(top_z + stage_clearance),
                grasp_yaw=DEFAULT_ISOTROPIC_YAW,
                valid=True,
            )
            self._emit(
                "[pick] upright pre-inference plan object_lat=%.3f "
                "bias=%+.3f grasp_lat=%.3f contact_z=%.3f"
                % (
                    side_object_lat,
                    profile.pad_lateral_bias_m,
                    side_grasp_lat,
                    side_contact_z,
                )
            )
            if profile.side_floor_load:
                self._emit(
                    "[pick] sunblock floor-load plan fully_open=True "
                    "support_z=%.3f pad_z=%.3f level_advance=%.3fm"
                    % (
                        support_z,
                        side_contact_z,
                        profile.side_final_insertion_m,
                    )
                )
        else:
            # Overhead profiles retain the demonstrated camera handoff. Feed a
            # synthetic Z so its stage is measured from the observed object top.
            stage_up = float(arm.cfg.openvla_stage_up_m)
            stage_input_z = top_z + stage_clearance - stage_up
            stage_target = SimpleNamespace(
                fwd=float(execution_center[0]) + contact_fwd_bias,
                lat=float(execution_center[1]) + contact_lat_bias,
                z=float(stage_input_z),
                valid=True,
            )
        stage_graspgenx = getattr(arm, "stage_for_graspgenx", None)
        if stage_graspgenx is None:
            staged = arm.stage_for_openvla(stage_target)
        else:
            staged = stage_graspgenx(stage_target, profile=profile)
        if not staged:
            self._emit("[pick] GraspGen-X deterministic staging failed")
            return
        if side_profile:
            self._emit(
                "[pick] GraspGen-X upright handoff complete at folded rear "
                "pregrasp; deploying inference on saved base-depth cloud"
            )
        else:
            self._emit(
                "[pick] GraspGen-X handoff support_z=%.3f top_z=%.3f "
                "stage_z=%.3f clearance=%.3f; sampling saved base-depth cloud"
                % (support_z, top_z, top_z + stage_clearance, stage_clearance)
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
        selection_diagnostics = {}
        selected = select_topdown_contact_pose(
            raw,
            points,
            (
                float(target.fwd) + contact_fwd_bias,
                float(target.lat) + contact_lat_bias,
                float(target.z),
            ),
            fingertip_depth_m=_env_float(
                "MARC_GRASPGENX_TCP_DEPTH_M", DEFAULT_TCP_DEPTH_M
            ),
            max_tilt_deg=_env_float(
                "MARC_GRASPGENX_TOPDOWN_TILT_DEG", DEFAULT_TOPDOWN_TILT_DEG
            ),
            max_xy_m=_env_float(
                "MARC_GRASPGENX_OBJECT_RADIUS_M", DEFAULT_OBJECT_RADIUS_M
            ),
            max_grasp_width_m=_env_float(
                "MARC_GRASPGENX_MAX_GRASP_WIDTH_M", DEFAULT_MAX_GRASP_WIDTH_M
            ),
            # Only the copied side-grasp motion needs neutral closing yaw.
            # Tissue, pencilcase, and upright cracker-box deliberately retain
            # the lying profile's narrow-axis top-down selection.
            force_neutral_yaw=side_profile,
            diagnostics=selection_diagnostics,
        )
        if selected is None:
            self._emit(
                "[pick] GraspGen-X produced no safe top-down contact n_pts=%d "
                "diagnostics=%s" % (n_points, selection_diagnostics)
            )
            return

        if side_profile:
            # The folded arm occupies the base-camera view. Re-detecting here
            # can lock onto the arm and reclassify an untouched upright object.
            # The deterministic handoff therefore keeps its saved localization.
            localized_before = cloud_localization
            drift_fwd = 0.0
            drift_lat = 0.0
            self._emit(
                "[pick] upright final approach keeps pre-handoff localization; "
                "base-camera relocalization disabled while arm is folded"
            )
        else:
            fresh = self._observe_target()
            if fresh is None or getattr(fresh, "valid", False) is not True:
                self._emit("[pick] GraspGen-X lost target before final approach")
                return
            localized_before = SimpleNamespace(
                fwd=float(fresh.fwd),
                lat=float(fresh.lat),
                z=float(fresh.z),
                valid=True,
            )
            # The saved depth cloud owns width/yaw, but its sparse footprint
            # centre can be biased by the visible surface. Gate fresh RGB-D
            # independently per axis for overhead profiles.
            drift_fwd = localized_before.fwd - cloud_localization.fwd
            drift_lat = localized_before.lat - cloud_localization.lat
            max_relocalization_drift = max(
                0.0,
                _env_float(
                    "MARC_GRASPGENX_MAX_RELOCALIZATION_DRIFT_M",
                    DEFAULT_MAX_RELOCALIZATION_DRIFT_M,
                ),
            )
            if (
                abs(drift_fwd) > max_relocalization_drift
                or abs(drift_lat) > max_relocalization_drift
            ):
                self._emit(
                    "[pick] GraspGen-X ignoring post-stage localization jump "
                    "fwd=%+.3fm lat=%+.3fm limit=%.3fm"
                    % (drift_fwd, drift_lat, max_relocalization_drift)
                )
        # The cloud still owns width and yaw, but a partial visible surface can
        # move its fitted centre several centimetres away from a stable detection.
        # Apply only the bounded correction already used for deterministic staging.
        fwd = float(execution_center[0]) + contact_fwd_bias
        object_lat = float(execution_center[1]) + contact_lat_bias
        lat = float(side_grasp_lat) if side_profile else object_lat
        raw_cloud_delta = cloud_center - localized_xy
        applied_center_delta = execution_center - localized_xy
        z = (
            float(side_contact_z)
            if side_profile
            else object_contact_z(
                points,
                localized_before.z,
                contact_depth_fraction=(
                    profile.contact_depth_fraction if profile is not None else None
                ),
                floor_clearance_m=(
                    profile.floor_clearance_m if profile is not None else None
                ),
            )
        )
        if z is None:
            self._emit(
                "[pick] GraspGen-X object too short for floor-safe contact "
                "support_z=%.3f top_z=%.3f" % (support_z, top_z)
            )
            return
        if side_profile:
            folded_names = (
                "folded_pad_fwd", "folded_pad_lat", "folded_pad_z"
            )
            if not all(hasattr(stage_target, name) for name in folded_names):
                self._emit(
                    "[pick] upright stage did not report its post-fold pad pose "
                    "-- fail closed"
                )
                return
            folded_pregrasp = np.array(
                (
                    float(stage_target.folded_pad_fwd),
                    float(stage_target.folded_pad_lat),
                    float(stage_target.folded_pad_z),
                ),
                dtype=np.float64,
            )
            cloud_contact_z = float(z)
            # The final upright correction is level. Tumbler-shaped profiles
            # use a tiny correction; the sunblock floor-load profile traverses
            # its whole rear standoff while the gripper remains fully open.
            # The lower cloud Z must never be restored after the guarded fold.
            final_insertion = float(
                getattr(
                    stage_target,
                    "folded_final_insertion_m",
                    profile.side_final_insertion_m,
                )
            )
            fwd = float(folded_pregrasp[0]) + final_insertion
            lat = float(folded_pregrasp[1])
            z = float(folded_pregrasp[2])
            self._emit(
                "[pick] upright measured folded precontact height %.3f "
                "(cloud contact %.3f); final level advance %.3fm"
                % (z, cloud_contact_z, final_insertion)
            )
        min_fwd = (
            _env_float("MARC_GRASPGENX_MIN_FWD_M", DEFAULT_MIN_FWD_M)
            if profile is None
            else float(profile.min_fwd_m)
        )
        max_fwd = (
            _env_float("MARC_GRASPGENX_MAX_FWD_M", DEFAULT_MAX_FWD_M)
            if profile is None
            else float(profile.max_fwd_m)
        )
        max_abs_lat = _env_float(
            "MARC_GRASPGENX_MAX_ABS_LAT_M", DEFAULT_MAX_ABS_LAT_M
        )
        if not (min_fwd <= fwd <= max_fwd and abs(lat) <= max_abs_lat):
            self._emit(
                "[pick] GraspGen-X contact out of safe reach "
                "fwd=%.3f lat=%.3f z=%.3f" % (fwd, lat, z)
            )
            return
        yaw = float(selected["yaw"])
        target.fwd, target.lat, target.z = float(fwd), float(lat), float(z)
        target.grasp_yaw = yaw
        self._emit(
            "[pick] ggx-contact fwd=%.3f lat=%.3f z=%.3f yaw=%.2f mode=%s "
            "footprint=%.3fx%.3f aspect=%.2f center_shift=%.3f "
            "score=%.3f n_pts=%d n_keep=%d dyolo=%.3f "
            "source=bounded-cloud-center raw_cloud_delta=(%+.3f,%+.3f) "
            "applied_delta=(%+.3f,%+.3f) drift=(%+.3f,%+.3f)"
            % (
                fwd, lat, z, yaw,
                "upright-side-profile"
                if side_profile
                else (
                    "narrow-axis"
                    if selected["directional"]
                    else "upright-neutral"
                ),
                float(selected["width"]),
                float(selected["length"]), float(selected["aspect"]),
                float(selected["center_offset"]),
                float(selected["score"]), n_points,
                int(selected["n_keep"]), float(selected["dyolo"]),
                raw_cloud_delta[0], raw_cloud_delta[1],
                applied_center_delta[0], applied_center_delta[1],
                drift_fwd, drift_lat,
            )
        )

        above = max(
            0.03,
            _env_float("MARC_GRASPGENX_FINAL_APPROACH_M", DEFAULT_FINAL_APPROACH_M),
        )
        overhead_clearance = max(
            0.04,
            profile.overhead_clearance_m
            if profile is not None
            else _env_float(
                "MARC_GRASPGENX_OVERHEAD_CLEARANCE_M",
                DEFAULT_OVERHEAD_CLEARANCE_M,
            ),
        )
        approach_tilt = (
            float(profile.approach_tilt_rad) if side_profile else 0.0
        )
        pad_center_offset = (
            float(profile.pad_center_offset_m) if profile is not None else 0.0
        )
        center_pos_tolerance = (
            float(profile.center_pos_tolerance_m)
            if profile is not None
            else None
        )
        center_rot_tolerance = (
            float(profile.center_rot_tolerance_rad)
            if profile is not None
            else None
        )
        line_joint_step = (
            float(profile.line_joint_step_rad) if profile is not None else None
        )
        if side_profile:
            pregrasp = folded_pregrasp
        else:
            pregrasp = np.array(
                (fwd, lat, max(z + above, top_z + overhead_clearance)),
                dtype=np.float64,
            )
        self._emit(
            "[pick] ggx-safe-z support=%.3f top=%.3f "
            "pregrasp=(%.3f,%.3f,%.3f) contact=%.3f "
            "floor_clearance=%.3f tilt=%.2f"
            % (
                support_z,
                top_z,
                pregrasp[0],
                pregrasp[1],
                pregrasp[2],
                z,
                z - support_z,
                approach_tilt,
            )
        )
        # Upright profiles are already folded and descended at the deterministic
        # handoff. Overhead profiles retain the established center/refine path.
        center_z = top_z + stage_clearance
        if side_profile:
            wrist = None
            self._emit(
                "[pick] upright side handoff already folded and descended; "
                "skipping occluded camera refinement"
            )
        else:
            if not arm.move_grasp_pose(
                (fwd, lat, center_z),
                yaw,
                "ggx-center",
                approach_tilt_rad=approach_tilt,
                pad_center_offset_m=pad_center_offset,
                position_tolerance_m=center_pos_tolerance,
                rotation_tolerance_rad=center_rot_tolerance,
            ):
                return
            wrist = self._wrist_depth_refinement(
                arm,
                (fwd, lat, top_z),
                profile=profile,
                allow_fwd_correction=True,
            )
        if wrist is not None:
            fused_xy = np.asarray(wrist["fused_center"], dtype=np.float64)
            wrist_top_z = float(wrist["top_z"])
            if float(np.linalg.norm(fused_xy - np.array((fwd, lat)))) > 1e-4:
                if not arm.move_grasp_line(
                    (float(fused_xy[0]), float(fused_xy[1]), center_z),
                    yaw,
                    "ggx-wrist-center",
                    approach_tilt_rad=approach_tilt,
                    pad_center_offset_m=pad_center_offset,
                    joint_step_rad=line_joint_step,
                ):
                    return
            fwd, lat = float(fused_xy[0]), float(fused_xy[1])
            floor_clearance = max(
                0.01,
                profile.floor_clearance_m
                if profile is not None
                else _env_float(
                    "MARC_GRASPGENX_FLOOR_CLEARANCE_M",
                    DEFAULT_FLOOR_CLEARANCE_M,
                ),
            )
            z_shift = float(np.clip(wrist_top_z - top_z, -0.015, 0.015))
            z = max(
                support_z + floor_clearance,
                min(wrist_top_z - 0.005, z + z_shift),
            )
            top_z = wrist_top_z
            pregrasp = np.array(
                (fwd, lat, max(z + above, top_z + overhead_clearance)),
                dtype=np.float64,
            )
            target.fwd, target.lat, target.z = fwd, lat, z
            self._emit(
                "[pick] wrist-fused contact fwd=%.3f lat=%.3f z=%.3f "
                "pregrasp=(%.3f,%.3f,%.3f)"
                % (fwd, lat, z, pregrasp[0], pregrasp[1], pregrasp[2])
            )
        if side_profile:
            target.fwd, target.lat, target.z = fwd, lat, z
            self._emit(
                "[pick] upright pad-center lateral bias "
                "object_lat=%.3f bias=%+.3f grasp_lat=%.3f"
                % (object_lat, profile.pad_lateral_bias_m, lat)
            )
            min_pregrasp_fwd = float(profile.side_pregrasp_min_fwd_m)
            if float(pregrasp[0]) < min_pregrasp_fwd:
                self._emit(
                    "[pick] upright side pregrasp inside base keepout "
                    "fwd=%.3f min=%.3f -- fail closed; rerun A from farther standoff"
                    % (pregrasp[0], min_pregrasp_fwd)
                )
                return
            self._emit(
                "[pick] upright rear-insertion plan rear=(%.3f,%.3f,%.3f) "
                "contact=(%.3f,%.3f,%.3f) forward_only=%.3fm"
                % (
                    pregrasp[0], pregrasp[1], pregrasp[2],
                    fwd, lat, z, fwd - pregrasp[0],
                )
            )
            self._emit(
                "[pick] upright deterministic handoff complete; "
                "GraspGen-X owns final forward insertion only"
            )
        elif not arm.move_grasp_line(
            pregrasp,
            yaw,
            "ggx-pregrasp",
            approach_tilt_rad=approach_tilt,
            pad_center_offset_m=pad_center_offset,
            joint_step_rad=line_joint_step,
        ):
            return
        if not arm.move_grasp_line(
            (fwd, lat, z),
            yaw,
            "ggx-contact",
            approach_tilt_rad=approach_tilt,
            pad_center_offset_m=pad_center_offset,
            cart_step_m=(0.002 if side_profile else None),
            joint_step_rad=line_joint_step,
            control_rate_hz=(20.0 if side_profile else None),
            preserve_current_orientation=side_profile,
        ):
            return
        if not arm.stabilize_grasp_pose(
            "ggx-contact-settle",
            dwell_s=(profile.contact_settle_s if profile is not None else None),
            max_drift_m=(
                profile.contact_max_drift_m if profile is not None else None
            ),
        ):
            return
        # GraspGen-X uses one immediate full close: no ramp and no sensor-based
        # early stop. Hold feedback is consulted only after the lift.
        close_cmd = min(
            DEFAULT_RETENTION_CLOSE,
            max(
                float(arm.cfg.grip_close),
                _env_float("MARC_GRASPGENX_GRIP_CLOSE", DEFAULT_GRASP_CLOSE),
            ),
        )
        self._emit("[pick] GraspGen-X immediate grip close=%.3f" % close_cmd)
        if not arm.set_grip_immediate(close_cmd):
            return
        if not arm.lift_for_grasp_verification():
            self._emit("[pick] GraspGen-X verification lift failed")
            return

        if self._confirmed_hold_after_lift():
            self._visual_grasp_verified = True
            self._emit(
                "[pick] GraspGen-X grasp verified by %d consecutive post-lift "
                "hold samples; skipping base-camera reacquisition"
                % max(
                    1,
                    _env_int(
                        "MARC_GRASPGENX_HOLD_CONFIRM_SAMPLES",
                        DEFAULT_HOLD_CONFIRM_SAMPLES,
                    ),
                )
            )
            return

        after = self._observe_target(self._visual_target_observer)
        if after is None or getattr(after, "valid", False) is not True:
            try:
                held_after_lift = self.client.is_grasping() is True
            except Exception:
                held_after_lift = False
            self._visual_grasp_verified = bool(held_after_lift)
            self._emit(
                "[pick] GraspGen-X camera occluded after lift; hold sensor %s"
                % ("verified grasp" if held_after_lift else "did not verify grasp")
            )
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
        valid = valid and rise <= _env_float(
            "MARC_GRASPGENX_VERIFY_MAX_RISE_M", DEFAULT_VERIFY_MAX_RISE_M
        )
        drift = math.hypot(
            float(after.fwd) - float(localized_before.fwd),
            float(after.lat) - float(localized_before.lat),
        )
        if valid:
            retention = min(
                DEFAULT_RETENTION_CLOSE,
                max(
                    float(arm.cfg.grip_close),
                    _env_float(
                        "MARC_GRASPGENX_RETENTION_CLOSE", DEFAULT_RETENTION_CLOSE
                    ),
                ),
            )
            if retention > float(arm.grip_cmd) and not arm.set_grip(retention):
                valid = False
        self._visual_grasp_verified = bool(valid)
        self._emit(
            "[pick] GraspGen-X visual grasp %s rise=%.3fm xy_drift=%.3fm"
            % ("verified" if valid else "rejected", rise, drift)
        )

    def poll_grasping(self):
        # The simulator's contact sensor can drop once the object is lifted and
        # settles between the pads. A geometrically verified co-moving detection
        # is the independent positive signal this path was designed to provide.
        if self._visual_grasp_verified:
            return True
        held = super().poll_grasping()
        if held is True:
            self._emit("[pick] rejecting hold=True without base-camera lift evidence")
            return False
        return held

    def _place(self) -> None:
        """Consume verified hold only after placement actually opens the jaws."""
        custom_place = self._place_fn is not None
        super()._place()
        arm = self._teleop
        released = custom_place or (arm is not None and not arm.closing)
        if released:
            self._visual_grasp_verified = False
        elif self._visual_grasp_verified:
            self._emit(
                "[pick] placement did not release; preserving verified hold "
                "for place retry"
            )

    def _seed_closed_grip(self) -> None:
        """Keep GraspGen-X's full squeeze during transport instead of live relaxation."""
        super()._seed_closed_grip()
        arm = self._teleop
        if arm is None:
            return
        retention = min(
            DEFAULT_RETENTION_CLOSE,
            max(
                float(arm.cfg.grip_close),
                _env_float(
                    "MARC_GRASPGENX_RETENTION_CLOSE", DEFAULT_RETENTION_CLOSE
                ),
            ),
        )
        arm.grip_cmd = retention
        arm.grip_target = retention

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
