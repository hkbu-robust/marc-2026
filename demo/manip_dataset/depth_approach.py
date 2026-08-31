"""User-triggered base auto-approach: YOLO centering + arm-base-fwd stop."""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .images import depth_in_bbox, image_to_depth, image_to_rgb
from .yolo_detect import YoloConfig, YoloTracker

from nearfield import (
    ARM_BASE_FWD,
    base_camera_to_car_transform,
    choose_lock_depth,
    k_from_info,
    localize_detection,
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


# Depth/RGB frames arrive far slower than the 20 Hz control tick, so speed is
# clamped to max_step_m per observed frame interval and the base holds still
# whenever the measurement goes stale.
DEFAULT_CRAWL_MPS = 0.30
DEFAULT_MAX_STEP_M = 0.04
DEFAULT_STALE_STOP_S = 0.5
DEFAULT_PERSIST_K = 3
DEFAULT_PERCENTILE = 30.0
DEFAULT_MIN_VALID = 4
DEFAULT_LOCK_FRAMES = 12
DEFAULT_SETTLE_S = 0.5
DEFAULT_TICK_S = 0.05
DEFAULT_LOG_EVERY = 1
DEFAULT_OVERRUN_M = 0.68
DEFAULT_YAW_GAIN = 1.2
DEFAULT_YAW_MAX = 0.35
DEFAULT_YAW_MIN = 0.10
DEFAULT_CENTER_TOL = 0.05
DEFAULT_REALIGN = 0.12
DEFAULT_ALIGN_HOLD = 2
DEFAULT_ALIGN_SETTLE_S = 0.15
DEFAULT_ALIGN_TIMEOUT_S = 0.0  # 0 = never drive on timeout
DEFAULT_BURST_M = 0.15
DEFAULT_LAT_MIN_M = 0.05
DEFAULT_LAT_MAX_M = 20.0
# Stop once arm-base forward (same metric as pick localize) is this close (m).
DEFAULT_STOP_M = 0.56
# Hard range fallback: never drive closer even if the box-edge signal is bad.
DEFAULT_ALREADY_NEAR_M = 0.44
# A tall/upright object fills the lower camera view sooner and needs more room
# for the deterministic overhead stage than the same object lying flat.
DEFAULT_UPRIGHT_BBOX_ASPECT = 1.20
DEFAULT_UPRIGHT_ALREADY_NEAR_M = DEFAULT_ALREADY_NEAR_M
# Preserve this fraction of the image below the YOLO box during final crawl.
DEFAULT_FRAME_BOTTOM_MARGIN = 0.10
# Any exact target class may flicker in the base-camera view. A lock is still
# acquired only after RGB-D localizes that class nearby and roughly centered.
DEFAULT_TARGET_LOCK_ACQUIRE_PAD_M = 0.25
DEFAULT_TARGET_LOCK_DRIVE_TOL = 0.10
TARGET_LOCK_MATCH_RANGE_PAD_M = 0.10
TARGET_LOCK_WORLD_RADIUS_M = 0.15
TARGET_LOCK_LAT_RADIUS_M = 0.10
TARGET_LOCK_Z_RADIUS_M = 0.12
TARGET_LOCK_MAX_LAT_M = 0.30
# Practice sim often closes ~1 cm/s even when cmd_vel is 0.14 m/s. Overrun
# therefore stops on arm-base fwd, with this rate only sizing the wall-clock cap.
OVERRUN_SLOW_MPS = 0.008
OVERRUN_MAX_S = 60.0
OVERRUN_MIN_S = 25.0
# Skip burst re-ALIGN once arm-base fwd is this close to STOP_M (m).
BURST_SKIP_PAD_M = 0.10


@dataclass
class ApproachConfig:
    crawl_mps: float = DEFAULT_CRAWL_MPS
    max_step_m: float = DEFAULT_MAX_STEP_M
    stale_stop_s: float = DEFAULT_STALE_STOP_S
    persist_k: int = DEFAULT_PERSIST_K
    lock_frames: int = DEFAULT_LOCK_FRAMES
    percentile: float = DEFAULT_PERCENTILE
    min_valid: int = DEFAULT_MIN_VALID
    settle_s: float = DEFAULT_SETTLE_S
    tick_s: float = DEFAULT_TICK_S
    log_every: int = DEFAULT_LOG_EVERY
    overrun_m: float = DEFAULT_OVERRUN_M
    yaw_gain: float = DEFAULT_YAW_GAIN
    yaw_max: float = DEFAULT_YAW_MAX
    yaw_min: float = DEFAULT_YAW_MIN
    center_tol: float = DEFAULT_CENTER_TOL
    realign: float = DEFAULT_REALIGN
    align_hold: int = DEFAULT_ALIGN_HOLD
    align_settle_s: float = DEFAULT_ALIGN_SETTLE_S
    align_timeout_s: float = DEFAULT_ALIGN_TIMEOUT_S
    burst_m: float = DEFAULT_BURST_M
    lat_min_m: float = DEFAULT_LAT_MIN_M
    lat_max_m: float = DEFAULT_LAT_MAX_M
    stop_m: float = DEFAULT_STOP_M
    already_near_m: float = DEFAULT_ALREADY_NEAR_M
    upright_bbox_aspect: float = DEFAULT_UPRIGHT_BBOX_ASPECT
    upright_already_near_m: float = DEFAULT_UPRIGHT_ALREADY_NEAR_M
    frame_bottom_margin: float = DEFAULT_FRAME_BOTTOM_MARGIN
    target_lock_acquire_pad_m: float = DEFAULT_TARGET_LOCK_ACQUIRE_PAD_M
    target_lock_drive_tol: float = DEFAULT_TARGET_LOCK_DRIVE_TOL

    @classmethod
    def from_env(cls) -> "ApproachConfig":
        return cls(
            crawl_mps=_env_float("MARC_APPROACH_CRAWL", DEFAULT_CRAWL_MPS),
            max_step_m=_env_float("MARC_APPROACH_MAX_STEP", DEFAULT_MAX_STEP_M),
            stale_stop_s=_env_float("MARC_APPROACH_STALE_STOP", DEFAULT_STALE_STOP_S),
            persist_k=_env_int("MARC_APPROACH_K", DEFAULT_PERSIST_K),
            lock_frames=_env_int("MARC_APPROACH_LOCK_FRAMES", DEFAULT_LOCK_FRAMES),
            percentile=_env_float("MARC_APPROACH_PERCENTILE", DEFAULT_PERCENTILE),
            min_valid=_env_int("MARC_APPROACH_MIN_VALID", DEFAULT_MIN_VALID),
            settle_s=_env_float("MARC_APPROACH_SETTLE", DEFAULT_SETTLE_S),
            log_every=_env_int("MARC_APPROACH_LOG_EVERY", DEFAULT_LOG_EVERY),
            overrun_m=_env_float("MARC_APPROACH_OVERRUN_M", DEFAULT_OVERRUN_M),
            yaw_gain=_env_float("MARC_APPROACH_YAW_GAIN", DEFAULT_YAW_GAIN),
            yaw_max=_env_float("MARC_APPROACH_YAW_MAX", DEFAULT_YAW_MAX),
            yaw_min=_env_float("MARC_APPROACH_YAW_MIN", DEFAULT_YAW_MIN),
            center_tol=_env_float("MARC_APPROACH_CENTER_TOL", DEFAULT_CENTER_TOL),
            realign=_env_float("MARC_APPROACH_REALIGN", DEFAULT_REALIGN),
            align_hold=_env_int("MARC_APPROACH_ALIGN_HOLD", DEFAULT_ALIGN_HOLD),
            align_settle_s=_env_float("MARC_APPROACH_ALIGN_SETTLE", DEFAULT_ALIGN_SETTLE_S),
            align_timeout_s=_env_float("MARC_APPROACH_ALIGN_TIMEOUT", DEFAULT_ALIGN_TIMEOUT_S),
            burst_m=_env_float("MARC_APPROACH_BURST_M", DEFAULT_BURST_M),
            lat_min_m=_env_float("MARC_APPROACH_LAT_MIN_M", DEFAULT_LAT_MIN_M),
            lat_max_m=_env_float("MARC_APPROACH_LAT_MAX_M", DEFAULT_LAT_MAX_M),
            stop_m=_env_float("MARC_APPROACH_STOP_M", DEFAULT_STOP_M),
            already_near_m=_env_float("MARC_APPROACH_ALREADY_NEAR", DEFAULT_ALREADY_NEAR_M),
            upright_bbox_aspect=_env_float(
                "MARC_APPROACH_UPRIGHT_BBOX_ASPECT", DEFAULT_UPRIGHT_BBOX_ASPECT
            ),
            upright_already_near_m=_env_float(
                "MARC_APPROACH_UPRIGHT_ALREADY_NEAR",
                DEFAULT_UPRIGHT_ALREADY_NEAR_M,
            ),
            frame_bottom_margin=_env_float(
                "MARC_APPROACH_FRAME_BOTTOM_MARGIN", DEFAULT_FRAME_BOTTOM_MARGIN
            ),
            target_lock_acquire_pad_m=_env_float(
                "MARC_APPROACH_LOCK_ACQUIRE_PAD_M",
                DEFAULT_TARGET_LOCK_ACQUIRE_PAD_M,
            ),
            target_lock_drive_tol=_env_float(
                "MARC_APPROACH_LOCK_DRIVE_TOL",
                DEFAULT_TARGET_LOCK_DRIVE_TOL,
            ),
        )


def _debug_dir() -> str:
    return os.environ.get(
        "MARC_YOLO_DEBUG_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "datasets", "yolo_debug"),
    )


def _dump_rgb(rgb: np.ndarray, tag: str, extra: str = ""):
    try:
        from .recorder import write_png

        root = _debug_dir()
        os.makedirs(root, exist_ok=True)
        h, w = rgb.shape[:2]
        path = os.path.join(root, f"{tag}_{h}x{w}.png")
        write_png(path, rgb)
        print(f"[approach] dumped {path}{extra}")
    except Exception as e:
        print(f"[approach] dump failed ({tag}): {e}")


def _stamp_key(msg) -> Optional[tuple]:
    try:
        st = msg.header.stamp
        return (int(st.sec), int(st.nanosec))
    except Exception:
        return None


class DepthApproachController:
    """Approach an object by alternating turn-in-place centering with straight crawls.

    Linear and angular velocity are never commanded together: the base either
    yaws (ALIGN) or drives (DRIVE). The object is the YOLO box on the base RGB
    camera; range is arm-base forward from the same localize as pick.
    """

    def __init__(self, client, cfg: Optional[ApproachConfig] = None,
                 yolo_cfg: Optional[YoloConfig] = None):
        self._client = client
        self.cfg = cfg or ApproachConfig.from_env()
        self.yolo_cfg = yolo_cfg or YoloConfig.from_env()
        self._yolo = YoloTracker(self.yolo_cfg)
        self.active = False
        self._hit_count = 0
        self._tick_i = 0
        self._t0: Optional[float] = None
        self._overrun = False
        self._overrun_t0: Optional[float] = None
        self._overrun_need_s = 0.0
        self._overrun_d0: Optional[float] = None
        self._detect_reason: str = ""
        self._last_rgb_stamp: Optional[tuple] = None
        self._last_depth_stamp: Optional[tuple] = None
        self._last_new_frame_t: Optional[float] = None
        self._frame_dt: Optional[float] = None
        self._last_good_t: Optional[float] = None
        self._last_fail: str = ""
        self._lock_rgb: Optional[tuple] = None
        self._lock_depth: Optional[tuple] = None
        self._rgb_swap = False
        self._lat_err: Optional[float] = None
        self._det: Optional[dict] = None
        self._d_obj_cached: Optional[float] = None
        self._fwd_cached: Optional[float] = None
        self._phase: str = "align"
        self._phase_t0: Optional[float] = None
        self._phase_why: str = "start"
        self._align_ok = 0
        self._drive_m = 0.0
        self._arrive_now = False
        self._skip_overrun = False
        self._timeout_logged = False
        self._target_name: Optional[str] = None
        self._near_lock_det: Optional[dict] = None
        self._near_lock_target = None

    def detector_ready(self) -> bool:
        return self._yolo.model is not None

    def set_target(self, name: Optional[str]):
        target = None if not name else str(name).strip()
        if target != self._target_name:
            self._clear_near_lock()
        self._target_name = target
        self._yolo.set_target(name)

    def _clear_near_lock(self):
        self._near_lock_det = None
        self._near_lock_target = None

    def locked_target(self, name: Optional[str]):
        """Return the selected class's maintained near-range localization."""
        target = None if not name else str(name).strip()
        if not target or target != self._target_name:
            return None
        locked = self._near_lock_target
        return locked if locked is not None and getattr(locked, "valid", False) else None

    def ensure_detector(self) -> bool:
        return self._yolo.load()

    def _note_new_frame(self, stamp, now: float):
        is_new = stamp is None or stamp != self._last_rgb_stamp
        if is_new:
            if self._last_new_frame_t is not None:
                dt = now - self._last_new_frame_t
                if 0.0 < dt < 5.0:
                    self._frame_dt = dt if self._frame_dt is None else (
                        0.4 * dt + 0.6 * self._frame_dt
                    )
            self._last_new_frame_t = now
        return is_new

    def _rgb_msg(self, shape=None):
        cam = self.yolo_cfg.camera
        if shape is not None:
            getter = getattr(self._client, "get_robot_image_hw", None)
            if getter is not None:
                return getter(cam, shape)
        return self._client.get_robot_image(cam)

    def _depth_msg(self, shape=None):
        if shape is not None:
            getter = getattr(self._client, "get_robot_depth_hw", None)
            if getter is not None:
                return getter("base", shape)
        return self._client.get_robot_depth("base")

    def _prep_rgb(self, rgb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if rgb is None:
            return None
        if self._rgb_swap:
            return np.ascontiguousarray(rgb[:, :, ::-1])
        return rgb

    def _camera_k(self, rgb_shape):
        getter = getattr(self._client, "get_robot_info", None)
        info = None
        if callable(getter):
            try:
                info = getter(self.yolo_cfg.camera)
            except Exception:
                info = None
        return k_from_info(info, rgb_shape)

    def _localize_target(self, det, depth, rgb_shape):
        """Localize one detection with the same camera transform as pick snapshot."""
        if det is None or depth is None or det.get("bbox") is None:
            return None
        shape = det.get("rgb_shape") or rgb_shape
        if shape is None:
            return None
        try:
            R, t = base_camera_to_car_transform()
            pose = None
            getter = getattr(self._client, "get_world_pose", None)
            if callable(getter):
                pose = getter()
            loc = localize_detection(
                det["bbox"],
                depth,
                shape,
                self._camera_k(shape),
                pose=pose,
                hint_xy=None,
                hint_radius=None,
                conf=det.get("conf"),
                name=det.get("name") or "",
                R=R,
                t=t,
            )
        except Exception:
            return None
        if loc.reason not in ("ok", "out_of_reach", "outside_radius"):
            return None
        return loc

    def _localize_fwd(self, det, depth, rgb_shape) -> Optional[float]:
        """Arm-base forward of the YOLO box, same helper as pick snapshot."""
        loc = self._localize_target(det, depth, rgb_shape)
        return None if loc is None else float(loc.fwd)

    def _matches_near_lock(self, candidate) -> bool:
        locked = self._near_lock_target
        if candidate is None or locked is None or not getattr(candidate, "valid", False):
            return False
        old_world = getattr(locked, "world_xy", None)
        new_world = getattr(candidate, "world_xy", None)
        if old_world is not None and new_world is not None:
            return math.dist(old_world, new_world) <= TARGET_LOCK_WORLD_RADIUS_M
        return (
            abs(float(candidate.lat) - float(locked.lat)) <= TARGET_LOCK_LAT_RADIUS_M
            and abs(float(candidate.z) - float(locked.z)) <= TARGET_LOCK_Z_RADIUS_M
            and float(candidate.fwd)
            <= float(locked.fwd) + TARGET_LOCK_MATCH_RANGE_PAD_M
        )

    def _reuse_near_lock(self, depth, rgb_shape):
        """Project the locked world coordinate into the robot's current arm frame."""
        if (
            not self._target_name
            or self._near_lock_det is None
            or str(self._near_lock_det.get("name") or "").strip()
            != self._target_name
        ):
            return None
        locked = self._near_lock_target
        if (
            locked is None
            or str(getattr(locked, "name", "") or "").strip()
            != self._target_name
        ):
            return None
        world_xy = None if locked is None else getattr(locked, "world_xy", None)
        getter = getattr(self._client, "get_world_pose", None)
        pose = getter() if callable(getter) else None
        if world_xy is None or pose is None or len(pose) < 3:
            return None
        dx = float(world_xy[0]) - float(pose[0])
        dy = float(world_xy[1]) - float(pose[1])
        yaw = float(pose[2])
        cc, ss = math.cos(yaw), math.sin(yaw)
        car_fwd = dx * cc + dy * ss
        car_lat = -dx * ss + dy * cc
        arm_fwd = car_fwd - ARM_BASE_FWD
        if arm_fwd <= 0.0 or abs(car_lat) > TARGET_LOCK_MAX_LAT_M:
            return None
        projected = replace(
            locked,
            fwd=float(arm_fwd),
            lat=float(car_lat),
            valid=True,
            reason="near_target_lock",
        )
        self._near_lock_target = projected
        det = dict(self._near_lock_det)
        det["rgb_shape"] = rgb_shape or det.get("rgb_shape")
        det["error"] = float(
            np.clip(-car_lat / max(0.10, car_fwd), -1.0, 1.0)
        )
        det["status"] = "near_target_lock"
        det["n_obj"] = 1
        det["n_raw"] = 0
        det["near_lock_reused"] = True
        return det

    def _maybe_lock_near_target(self, det, localized):
        if (
            not self._target_name
            or det is None
            or localized is None
            or not getattr(localized, "valid", False)
            or str(det.get("name") or "").strip() != self._target_name
            or str(getattr(localized, "name", "") or "").strip()
            != self._target_name
            or getattr(localized, "world_xy", None) is None
        ):
            return
        max_fwd = max(float(self.cfg.stop_m), float(self.cfg.already_near_m))
        max_fwd += max(0.0, float(self.cfg.target_lock_acquire_pad_m))
        if float(localized.fwd) > max_fwd or abs(float(localized.lat)) > self.cfg.realign:
            return
        if self._near_lock_target is not None and not self._matches_near_lock(localized):
            old = self._near_lock_target
            print(
                f"[approach] {self._target_name} lock reset: fresh detection "
                f"conflicts with locked position "
                f"old=({old.fwd:.3f},{old.lat:.3f}) "
                f"new=({localized.fwd:.3f},{localized.lat:.3f})"
            )
            self._clear_near_lock()
        first = self._near_lock_target is None
        self._near_lock_det = dict(det)
        self._near_lock_det.pop("near_lock_reused", None)
        self._near_lock_target = localized
        if first:
            print(
                f"[approach] {self._target_name} lock acquired at "
                f"fwd={localized.fwd:.3f} lat={localized.lat:.3f} "
                f"z={localized.z:.3f} conf={localized.conf:.2f}"
            )

    def _measure(self) -> Optional[dict]:
        """Read RGB + depth; YOLO on new RGB. None if neither stream is usable."""
        now = time.monotonic()
        rgb_msg = self._rgb_msg(self._lock_rgb)
        depth_msg = self._depth_msg(self._lock_depth)

        rgb = None
        rgb_shape = None
        rgb_stamp = None
        rgb_new = False
        if rgb_msg is not None:
            rgb = self._prep_rgb(image_to_rgb(rgb_msg))
            if rgb is not None:
                rgb_shape = tuple(rgb.shape[:2])
                if self._lock_rgb is not None and rgb_shape != self._lock_rgb:
                    self._last_fail = (
                        f"other rgb {rgb_shape[0]}x{rgb_shape[1]} "
                        f"(locked {self._lock_rgb[0]}x{self._lock_rgb[1]})"
                    )
                    rgb = None
                    rgb_shape = None
                else:
                    rgb_stamp = _stamp_key(rgb_msg)
                    rgb_new = self._note_new_frame(rgb_stamp, now)
                    if rgb_new:
                        self._last_rgb_stamp = rgb_stamp
            else:
                self._last_fail = (
                    f"rgb decode failed (encoding={getattr(rgb_msg, 'encoding', '?')})"
                )
        elif self._lock_rgb is not None:
            have = []
            lister = getattr(self._client, "list_robot_image_shapes", None)
            if lister is not None:
                have = [f"{h}x{w}" for h, w in lister(self.yolo_cfg.camera)]
            self._last_fail = (
                f"waiting for locked rgb {self._lock_rgb[0]}x{self._lock_rgb[1]}"
                + (f" (have {','.join(have)})" if have else "")
            )

        depth = None
        depth_shape = None
        if depth_msg is not None:
            depth = image_to_depth(depth_msg)
            if depth is not None:
                depth_shape = tuple(depth.shape[:2])
                if self._lock_depth is not None and depth_shape != self._lock_depth:
                    self._last_fail = (
                        f"other depth {depth_shape[0]}x{depth_shape[1]} "
                        f"(locked {self._lock_depth[0]}x{self._lock_depth[1]})"
                    )
                    depth = None
                    depth_shape = None
                else:
                    self._last_depth_stamp = _stamp_key(depth_msg)
            else:
                self._last_fail = (
                    f"depth decode failed (encoding={getattr(depth_msg, 'encoding', '?')})"
                )

        if rgb is None:
            n_keep = 0 if self._det is None else int(self._det.get("n_obj") or 0)
            if n_keep <= 0:
                if not self._last_fail:
                    self._last_fail = "no rgb"
                return None

        if rgb is not None and rgb_new:
            det = self._yolo.detect(rgb)
            if det is None or int(det.get("n_obj") or 0) <= 0:
                reused = self._reuse_near_lock(depth, rgb_shape)
                if reused is not None:
                    det = reused
            self._det = det
        else:
            det = self._det

        n_obj = 0 if det is None else int(det.get("n_obj") or 0)
        n_raw = 0 if det is None else int(det.get("n_raw") or 0)
        lat = None if det is None or n_obj <= 0 else det.get("error")

        d_obj = None
        if n_obj <= 0 or det is None or det.get("bbox") is None:
            self._d_obj_cached = None
        elif depth is not None:
            d_obj = depth_in_bbox(
                depth,
                det["bbox"],
                det.get("rgb_shape") or rgb_shape,
                percentile=self.cfg.percentile,
                min_m=self.cfg.lat_min_m,
                max_m=self.cfg.lat_max_m,
                min_valid=self.cfg.min_valid,
            )
            if d_obj is not None:
                self._d_obj_cached = float(d_obj)
            else:
                d_obj = self._d_obj_cached
        else:
            d_obj = self._d_obj_cached

        if n_obj > 0 and lat is not None:
            self._lat_err = float(lat)

        fwd = None
        localized = None
        if n_obj > 0 and det is not None:
            if det.get("near_lock_reused"):
                localized = self._near_lock_target
            else:
                localized = self._localize_target(
                    det, depth, rgb_shape or det.get("rgb_shape")
                )
            fwd = None if localized is None else float(localized.fwd)
            if fwd is not None:
                self._fwd_cached = float(fwd)
                self._maybe_lock_near_target(det, localized)
            else:
                fwd = self._fwd_cached
        else:
            fwd = None

        self._last_good_t = now
        self._last_fail = ""
        return {
            "lat_err": lat,
            "lat_n_obj": n_obj,
            "n_raw": n_raw,
            "d_obj": None if d_obj is None else float(d_obj),
            "fwd": None if fwd is None else float(fwd),
            "name": None if det is None else det.get("name"),
            "conf": None if det is None else det.get("conf"),
            "bbox": None if det is None else det.get("bbox"),
            "area": None if det is None else det.get("area"),
            "target_locked": bool(det is not None and det.get("near_lock_reused")),
            "target_lock_active": self._near_lock_target is not None,
            "is_new": bool(rgb_new),
            "rgb_shape": None if det is None else (det.get("rgb_shape") or rgb_shape),
            "depth_shape": depth_shape,
            "stamp": rgb_stamp,
        }

    def _speed(self) -> float:
        """Crawl speed clamped so the base moves at most max_step_m per frame."""
        v = max(0.0, float(self.cfg.crawl_mps))
        step = max(0.005, float(self.cfg.max_step_m))
        dt = self._frame_dt if self._frame_dt else self.cfg.tick_s
        dt = max(1e-3, float(self.cfg.tick_s), float(dt))
        return min(v, step / dt)

    def _yaw(self, err: float) -> float:
        """Yaw rate for a signed image error (image-right positive → turn right)."""
        ang = -float(self.cfg.yaw_gain) * float(err)
        mag = min(abs(ang), float(self.cfg.yaw_max))
        mag = max(mag, float(self.cfg.yaw_min))
        return math.copysign(mag, ang)

    def _is_centered(self, m: Optional[dict]) -> bool:
        if m is None:
            return False
        n_obj = int(m.get("lat_n_obj") or 0)
        lat = m.get("lat_err")
        return n_obj > 0 and lat is not None and abs(float(lat)) <= self.cfg.center_tol

    def _d_obj(self, m: Optional[dict]) -> Optional[float]:
        if m is None or m.get("d_obj") is None:
            return None
        return float(m["d_obj"])

    def _range_m(self, m: Optional[dict]) -> Optional[float]:
        """Arm-base forward for stop/overrun. Fall back to bbox depth if missing."""
        if m is not None and m.get("fwd") is not None:
            return float(m["fwd"])
        if self._fwd_cached is not None:
            return float(self._fwd_cached)
        return self._d_obj(m)

    def _too_near(self, m: Optional[dict]) -> bool:
        """YOLO box is already at pick range — do not yaw or keep crawling."""
        d = self._range_m(m)
        lim = self._near_limit(m)
        n_obj = 0 if m is None else int(m.get("lat_n_obj") or 0)
        return n_obj > 0 and d is not None and lim > 0.0 and d <= lim

    def _near_limit(self, m: Optional[dict]) -> float:
        """Use a farther hard stop when the current target box is upright."""
        base = float(self.cfg.already_near_m)
        if m is None:
            return base
        bbox = m.get("bbox")
        if bbox is None or len(bbox) < 4:
            return base
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width <= 1.0 or height / width < float(self.cfg.upright_bbox_aspect):
            return base
        return max(base, float(self.cfg.upright_already_near_m))

    def _frame_bottom_limit(self, m: Optional[dict]) -> bool:
        """True before forward motion would crop the object's lower edge."""
        if m is None or int(m.get("lat_n_obj") or 0) <= 0:
            return False
        bbox = m.get("bbox")
        shape = m.get("rgb_shape")
        if bbox is None or shape is None or len(bbox) < 4 or len(shape) < 2:
            return False
        height = float(shape[0])
        margin = min(0.45, max(0.0, float(self.cfg.frame_bottom_margin)))
        if height <= 0.0 or margin <= 0.0:
            return False
        return float(bbox[3]) >= height * (1.0 - margin)

    def _obj_stop(self, m: Optional[dict]) -> bool:
        """Arm-base forward is close enough to stop, and not far off-axis."""
        d = self._range_m(m)
        lim = float(self.cfg.stop_m)
        n_obj = 0 if m is None else int(m.get("lat_n_obj") or 0)
        lat = None if m is None else m.get("lat_err")
        if n_obj <= 0 or d is None or lim <= 0.0 or lat is None:
            return False
        if abs(float(lat)) > float(self.cfg.realign):
            return False
        return d < lim

    def _enter(self, phase: str, why: str):
        self._phase = phase
        self._phase_t0 = time.monotonic()
        self._align_ok = 0
        self._phase_why = why
        self._timeout_logged = False
        if self.cfg.log_every > 0:
            t_s = 0.0 if self._t0 is None else (time.monotonic() - self._t0)
            print(f"[approach] t={t_s:5.2f}s  -> {phase.upper()} ({why})")

    def _command(self, m: Optional[dict]) -> tuple:
        """Alternate between turning in place and driving straight."""
        now = time.monotonic()
        lat = None if m is None else m.get("lat_err")
        n_obj = 0 if m is None else int(m.get("lat_n_obj") or 0)
        is_new = bool(m is not None and m.get("is_new"))
        has_obj = lat is not None and n_obj > 0
        lock_active = bool(m is not None and m.get("target_lock_active"))
        if has_obj:
            self._lat_err = float(lat)
        err = self._lat_err
        speed = self._speed()
        since = now - (self._phase_t0 or now)

        if self._too_near(m) or self._frame_bottom_limit(m):
            return 0.0, 0.0
        lock_drive_tol = min(
            float(self.cfg.realign),
            max(float(self.cfg.center_tol), float(self.cfg.target_lock_drive_tol)),
        )
        if lock_active and has_obj and abs(float(lat)) <= lock_drive_tol:
            if self._phase != "drive":
                self._enter(
                    "drive",
                    f"spatial lock err={float(lat):+.2f} "
                    f"tol={lock_drive_tol:.2f}",
                )
            return speed, 0.0
        if self._is_centered(m):
            if self._phase != "drive":
                self._enter("drive", f"already centered err={err:+.2f}")
            return speed, 0.0

        if self._phase == "align":
            if since < self.cfg.align_settle_s:
                return 0.0, 0.0
            if not has_obj or err is None:
                # Wait for YOLO — never crawl without a box.
                return 0.0, 0.0
            if abs(err) <= self.cfg.center_tol:
                if is_new:
                    self._align_ok += 1
                if self._align_ok >= max(1, self.cfg.align_hold):
                    self._enter("drive", f"centered err={err:+.2f}")
                    return speed, 0.0
                return 0.0, 0.0
            self._align_ok = 0
            timeout = float(self.cfg.align_timeout_s)
            if timeout > 0.0 and since >= timeout:
                if not self._timeout_logged:
                    print(
                        f"[approach] ALIGN still off-center err={err:+.2f} "
                        f"after {timeout:.1f}s — holding yaw (not driving)"
                    )
                    self._timeout_logged = True
            return 0.0, self._yaw(err)

        self._drive_m = since * speed
        if not has_obj:
            self._enter("align", "lost detection")
            return 0.0, 0.0
        if abs(err) > self.cfg.realign:
            self._enter("align", f"off-center err={err:+.2f}")
            return 0.0, 0.0
        if self.cfg.burst_m > 0.0 and self._drive_m >= self.cfg.burst_m:
            d = self._range_m(m)
            stop = float(self.cfg.stop_m)
            if d is None or stop <= 0.0 or d > stop + BURST_SKIP_PAD_M:
                self._enter("align", f"burst {self._drive_m:.2f}m")
                return 0.0, 0.0
        return speed, 0.0

    def _lock_streams(self) -> bool:
        """Stop, sample frames, lock the taller RGB (chassis 480x640, not 360 close-up).

        Two publishers share base_left. Majority 360x640 is a gripper-like view;
        recorded plaza/base frames are 480x640. Shape-keyed cache still serves
        that rarer stream after lock even when 360 arrives last.
        """
        self._client.send_cmd_vel(0.0, 0.0)
        self._lock_rgb = None
        self._lock_depth = None
        self._rgb_swap = False
        rgb_shapes = []
        depth_shapes = []
        best_by_shape: dict = {}  # shape -> (n_raw, rgb, enc, mean, raw, swapped)
        dumped = set()
        t_start = time.monotonic()
        n = max(4, int(self.cfg.lock_frames))
        low_probed = set()
        for i in range(n):
            rgb_msg = self._client.get_robot_image(self.yolo_cfg.camera)
            depth_msg = self._client.get_robot_depth("base")
            rgb = image_to_rgb(rgb_msg) if rgb_msg is not None else None
            depth = image_to_depth(depth_msg) if depth_msg is not None else None
            enc = getattr(rgb_msg, "encoding", "?") if rgb_msg is not None else "?"
            step = int(getattr(rgb_msg, "step", 0) or 0) if rgb_msg is not None else 0
            if rgb is not None:
                shape = tuple(rgb.shape[:2])
                rgb_shapes.append(shape)
                mean = float(rgb.mean())
                det = self._yolo.detect(rgb)
                swapped = False
                n_raw = 0 if det is None else int(det.get("n_raw") or 0)
                if n_raw == 0:
                    det_b = self._yolo.detect(np.ascontiguousarray(rgb[:, :, ::-1]))
                    n_b = 0 if det_b is None else int(det_b.get("n_raw") or 0)
                    if n_b > n_raw:
                        det = det_b
                        swapped = True
                        n_raw = n_b
                if n_raw == 0 and shape not in low_probed:
                    low_probed.add(shape)
                    lo = self._yolo.detect(rgb, conf=0.05)
                    lo_n = 0 if lo is None else int(lo.get("n_raw") or 0)
                    if lo_n:
                        lo_s = ",".join(
                            f"{a}:{b:.2f}" for a, b in (lo.get("raw") or [])[:6]
                        )
                        print(
                            f"[approach] {shape[0]}x{shape[1]} has low-conf boxes "
                            f"(conf=0.05, not used): [{lo_s}]"
                        )
                n_obj = 0 if det is None else int(det.get("n_obj") or 0)
                name = "-" if det is None else (det.get("name") or "-")
                raw = [] if det is None else list(det.get("raw") or [])
                prev = best_by_shape.get(shape)
                if prev is None or n_raw > prev[0]:
                    best_by_shape[shape] = (n_raw, rgb, enc, mean, raw, swapped)
                if shape not in dumped:
                    dumped.add(shape)
                    extra = (
                        f"  enc={enc} step={step} mean={mean:.1f} "
                        f"n_raw={n_raw} swap={int(swapped)}"
                    )
                    _dump_rgb(rgb, f"lock_{shape[0]}x{shape[1]}", extra=extra)
                    if swapped:
                        _dump_rgb(
                            np.ascontiguousarray(rgb[:, :, ::-1]),
                            f"lock_{shape[0]}x{shape[1]}_bgr",
                            extra="  (channel-swapped copy YOLO preferred)",
                        )
                if self.cfg.log_every > 0:
                    ds = "--" if depth is None else f"{depth.shape[0]}x{depth.shape[1]}"
                    raw_s = ",".join(f"{a}:{b:.2f}" for a, b in raw[:4]) or "-"
                    print(
                        f"[approach] lock {i+1}/{n}  rgb={shape[0]}x{shape[1]}  "
                        f"enc={enc} mean={mean:.1f}  n_raw={n_raw} n_obj={n_obj} "
                        f"{name} raw=[{raw_s}] swap={int(swapped)}  depth={ds}"
                    )
            elif self.cfg.log_every > 0:
                ds = "--" if depth is None else f"{depth.shape[0]}x{depth.shape[1]}"
                print(f"[approach] lock {i+1}/{n}  rgb=--  depth={ds}  "
                      f"enc={enc}")
            if depth is not None:
                depth_shapes.append(tuple(depth.shape[:2]))
            time.sleep(self.cfg.tick_s)
        if not rgb_shapes:
            self._last_fail = "no rgb during lock"
            return False
        if not depth_shapes:
            self._last_fail = "no depth during lock"
            return False
        counts = Counter(rgb_shapes)
        # Taller first (480x640 chassis), not majority 360 (gripper-like close-up).
        self._lock_rgb = max(counts.keys(), key=lambda s: (s[0], s[1], counts[s]))
        winner = best_by_shape.get(self._lock_rgb)
        self._rgb_swap = bool(winner[5]) if winner is not None else False
        if self._rgb_swap:
            print("[approach] RGB channels swapped for YOLO (8UC3 was RGB, not BGR)")
        dcounts = Counter(depth_shapes)
        self._lock_depth = choose_lock_depth(list(dcounts.elements()), self._lock_rgb)
        if self._lock_depth is None:
            self._last_fail = "no depth during lock"
            return False
        print(
            f"[approach] depth lock {self._lock_depth[0]}x{self._lock_depth[1]} "
            f"candidates={sorted(dcounts.keys())}"
        )
        if len(counts) > 1:
            others = ", ".join(
                f"{s[0]}x{s[1]}(n={c},raw={best_by_shape.get(s, (0,))[0]})"
                for s, c in counts.items() if s != self._lock_rgb
            )
            print(
                f"[approach] rgb lock {self._lock_rgb[0]}x{self._lock_rgb[1]} "
                f"by taller chassis view, ignoring {others}"
            )
        elapsed = time.monotonic() - t_start
        self._frame_dt = (elapsed / max(1, n - 1)) if elapsed > 0 else None
        self._last_rgb_stamp = None
        self._last_depth_stamp = None
        self._last_new_frame_t = None
        return True

    def start(self) -> bool:
        """Activate auto-approach. Returns False if YOLO or cameras are unavailable."""
        self._clear_near_lock()
        if not self._yolo.load():
            self.active = False
            print(f"[approach] YOLO FAILED: {self._yolo._last_fail}")
            return False
        if not self._lock_streams():
            self.active = False
            print(f"[approach] lock FAILED: {self._last_fail}")
            return False
        self.active = True
        self._tick_i = 0
        self._t0 = time.monotonic()
        self._overrun = False
        self._overrun_t0 = None
        self._overrun_need_s = 0.0
        self._overrun_d0 = None
        self._detect_reason = ""
        self._lat_err = None
        self._det = None
        self._d_obj_cached = None
        self._fwd_cached = None
        self._drive_m = 0.0
        self._arrive_now = False
        self._skip_overrun = False
        self._hit_count = 0
        self._timeout_logged = False
        m0 = self._measure()
        if self._frame_bottom_limit(m0):
            self._enter("drive", "object at frame margin")
            self._arrive_now = True
            self._skip_overrun = True
            self._detect_reason = "object at frame margin"
        elif self._too_near(m0):
            self._enter("drive", "already near")
            self._arrive_now = True
            self._skip_overrun = True
            self._detect_reason = "already near"
        elif self._obj_stop(m0) and self._is_centered(m0):
            self._enter("drive", "already in range")
            self._arrive_now = True
            self._detect_reason = "already in range"
        elif self._is_centered(m0):
            self._enter("drive", "already centered")
        else:
            self._enter("align", "start")
        self._last_good_t = time.monotonic()
        fps = None if not self._frame_dt else 1.0 / self._frame_dt
        print(
            f"[approach] YOLO {self.yolo_cfg.weights}  cam={self.yolo_cfg.camera}  "
            f"rgb={self._lock_rgb[0]}x{self._lock_rgb[1]}  "
            f"depth={self._lock_depth[0]}x{self._lock_depth[1]}\n"
            f"[approach] crawl={self.cfg.crawl_mps} clamped to "
            f"{self._speed():.3f} m/s  (max_step={self.cfg.max_step_m}m, "
            f"frame≈{'?' if fps is None else f'{fps:.1f}'} Hz)  "
            f"stale_stop={self.cfg.stale_stop_s}s  "
            f"overrun until fwd≤{self.cfg.already_near_m:.2f}m "
            f"(cmd {self.cfg.overrun_m:.3f}m)\n"
            f"[approach] stop-turn-go: yaw_gain={self.cfg.yaw_gain} "
            f"yaw=[{self.cfg.yaw_min},{self.cfg.yaw_max}]  "
            f"center_tol={self.cfg.center_tol} realign={self.cfg.realign} "
            f"hold={self.cfg.align_hold}  burst={self.cfg.burst_m:.3f}m  "
            f"stop={self.cfg.stop_m:.2f}m  already_near={self.cfg.already_near_m:.2f}m  "
            f"upright_near={self.cfg.upright_already_near_m:.2f}m  "
            f"bottom_margin={self.cfg.frame_bottom_margin:.0%}  "
            f"lock≤{max(self.cfg.stop_m, self.cfg.already_near_m) + self.cfg.target_lock_acquire_pad_m:.2f}m "
            f"lock_err≤{self.cfg.target_lock_drive_tol:.2f}"
        )
        print(
            "[approach] live: err from YOLO box vs image center; "
            "fwd=arm-base forward (pick localize); d=bbox depth. "
            "obj=used/raw. A nearby centered detection establishes a "
            "class-specific world lock; flicker then follows that lock. "
            "Without a valid lock, ALIGN waits and never drives blind."
        )
        return True

    def _overrun_duration_s(self) -> float:
        m = max(0.0, float(self.cfg.overrun_m))
        if m <= 0.0:
            return 0.0
        return m / max(1e-3, self._speed())

    def _overrun_cap_s(self, d0: Optional[float]) -> float:
        """Wall-clock cap. cmd_vel/speed overestimates travel when the sim runs slow."""
        commanded = self._overrun_duration_s()
        target = max(0.0, float(self.cfg.already_near_m))
        remaining = 0.0
        if d0 is not None and target > 0.0:
            remaining = max(0.0, float(d0) - target)
        if remaining <= 0.0:
            return commanded
        slow_s = remaining / OVERRUN_SLOW_MPS
        return min(OVERRUN_MAX_S, max(commanded, slow_s, OVERRUN_MIN_S))

    def _begin_overrun(self, reason: str, d0: Optional[float]) -> None:
        self._detect_reason = reason
        self._overrun = True
        self._overrun_t0 = time.monotonic()
        self._overrun_d0 = d0
        self._overrun_need_s = self._overrun_cap_s(d0)
        d_s = "--" if d0 is None else f"{float(d0):.3f}"
        print(
            f"[approach] DETECT ({reason}) — crawl until fwd≤"
            f"{self.cfg.already_near_m:.2f}m (now {d_s})  "
            f"cap={self._overrun_need_s:.2f}s  cmd={self.cfg.overrun_m:.3f}m "
            f"@ {self._speed():.2f} m/s"
        )
        self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)

    def _stop_approach(self, reason: str, t_s: float) -> bool:
        self._client.send_cmd_vel(0.0, 0.0)
        self.active = False
        self._overrun = False
        fwd = self._fwd_cached
        extra = "" if fwd is None or "fwd=" in reason else f" fwd={fwd:.3f}"
        print(f"[approach] STOP ({reason}{extra})  t={t_s:.2f}s")
        return True

    def _tick_overrun(self, t_s: float, do_log: bool) -> bool:
        assert self._overrun_t0 is not None
        elapsed = time.monotonic() - self._overrun_t0
        need = self._overrun_need_s
        m = self._measure()
        d = self._range_m(m)
        bbox = self._d_obj(m)
        if do_log:
            d_s = "--" if d is None else f"{d:.2f}"
            b_s = "--" if bbox is None else f"{bbox:.2f}"
            print(
                f"[approach] t={t_s:5.2f}s  OVERRUN {elapsed:.2f}/{need:.2f}s "
                f"fwd={d_s} d={b_s}  pick≤{self.cfg.already_near_m:.2f}m"
            )
        if self._frame_bottom_limit(m):
            return self._stop_approach("object at frame margin", t_s)
        if self._too_near(m):
            d_s = "--" if d is None else f"{d:.3f}"
            return self._stop_approach(f"overrun pick range fwd={d_s}", t_s)
        if elapsed >= need:
            extra = "" if d is None else f" fwd={d:.3f}"
            return self._stop_approach(
                f"overrun cap ({self.cfg.overrun_m:.3f}m){extra}  "
                f"detect={self._detect_reason}",
                t_s,
            )
        self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)
        return False

    def cancel(self):
        self.active = False
        self._hit_count = 0
        self._overrun = False
        self._overrun_t0 = None
        self._overrun_need_s = 0.0
        self._overrun_d0 = None
        self._lat_err = None
        self._det = None
        self._d_obj_cached = None
        self._fwd_cached = None
        self._clear_near_lock()
        self._phase = "align"
        self._phase_t0 = None
        self._align_ok = 0
        self._drive_m = 0.0
        self._arrive_now = False
        self._skip_overrun = False
        self._client.send_cmd_vel(0.0, 0.0)

    def tick(self) -> bool:
        """One control tick. Returns True when approach finished (caller should settle)."""
        if not self.active:
            return False

        self._tick_i += 1
        t_s = 0.0 if self._t0 is None else (time.monotonic() - self._t0)
        do_log = self.cfg.log_every > 0 and (self._tick_i % self.cfg.log_every == 0)

        if self._arrive_now:
            self._arrive_now = False
            reason = self._detect_reason or "already there"
            if self._skip_overrun or self._overrun_duration_s() <= 0.0:
                return self._stop_approach(f"{reason} (no overrun)", t_s)
            self._begin_overrun(reason, self._fwd_cached)
            return False

        if self._overrun:
            return self._tick_overrun(t_s, do_log)

        m = self._measure()

        if m is None:
            age = (
                999.0 if self._last_good_t is None
                else time.monotonic() - self._last_good_t
            )
            if age > self.cfg.stale_stop_s:
                self._client.send_cmd_vel(0.0, 0.0)
                if do_log:
                    print(
                        f"[approach] t={t_s:5.2f}s  HOLD (stale {age:.2f}s > "
                        f"{self.cfg.stale_stop_s}s): {self._last_fail}"
                    )
            else:
                lin, ang = self._command(None)
                self._client.send_cmd_vel(linear_x=lin, angular_z=ang)
                if do_log:
                    print(
                        f"[approach] t={t_s:5.2f}s  no measurement "
                        f"(age={age:.2f}s): {self._last_fail}"
                    )
            return False

        lin, ang = self._command(m)
        frame_limit = self._frame_bottom_limit(m)
        near = self._obj_stop(m) or self._too_near(m) or frame_limit
        if near and m.get("is_new"):
            self._hit_count += 1
        elif not near:
            self._hit_count = 0

        if do_log:
            flags = []
            if self._too_near(m):
                flags.append("NEAR")
            if self._obj_stop(m):
                flags.append("STOPD")
            if frame_limit:
                flags.append("FRAME")
            if m.get("target_locked"):
                flags.append("TARGET_LOCK")
            fps = None if not self._frame_dt else 1.0 / self._frame_dt
            lat = m.get("lat_err")
            n_obj = int(m.get("lat_n_obj") or 0)
            d_obj = m.get("d_obj")
            fwd = m.get("fwd")
            name = m.get("name") or "-"
            conf = m.get("conf")
            n_raw = int(m.get("n_raw") or 0)
            print(
                f"[approach] t={t_s:5.2f}s  {self._phase.upper():5s} "
                f"err={'--' if lat is None else f'{lat:+.2f}'} "
                f"obj={n_obj}/{n_raw} {name}"
                f"{'' if conf is None else f' {conf:.2f}'}  "
                f"fwd={'--' if fwd is None else f'{float(fwd):.2f}'} "
                f"d={'--' if d_obj is None else f'{float(d_obj):.2f}'}  "
                f"ang={ang:+.2f}  "
                f"hit={self._hit_count}/{self.cfg.persist_k}  "
                f"v={lin:.2f} "
                f"{'' if fps is None else f'{fps:.1f}Hz '}"
                f"{','.join(flags) if flags else '-'}"
            )

        if self._hit_count >= self.cfg.persist_k:
            rng = self._range_m(m)
            if frame_limit:
                reason = "object at frame margin"
            elif self._too_near(m):
                reason = "already near"
            else:
                reason = f"fwd={rng:.3f}" if rng is not None else "in range"
            if frame_limit or self._too_near(m) or self._overrun_duration_s() <= 0.0:
                return self._stop_approach(f"{reason} (no overrun)", t_s)
            self._begin_overrun(reason, rng)
            return False

        self._client.send_cmd_vel(linear_x=lin, angular_z=ang)
        return False

    def settle(self):
        self._client.send_cmd_vel(0.0, 0.0)
        time.sleep(self.cfg.settle_s)
