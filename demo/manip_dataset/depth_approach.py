"""User-triggered base auto-approach: YOLO centering + bbox-depth stop."""

from __future__ import annotations

import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .images import depth_in_bbox, image_to_depth, image_to_rgb
from .yolo_detect import YoloConfig, YoloTracker


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
# Stop once median depth inside the YOLO box is this close (m).
DEFAULT_STOP_M = 0.55
# Closer than this: already at pick range — stop with no overrun.
DEFAULT_ALREADY_NEAR_M = 0.45


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
    camera; range is median depth inside that box.
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
        self._phase: str = "align"
        self._phase_t0: Optional[float] = None
        self._phase_why: str = "start"
        self._align_ok = 0
        self._drive_m = 0.0
        self._arrive_now = False
        self._skip_overrun = False
        self._timeout_logged = False

    def detector_ready(self) -> bool:
        return self._yolo.model is not None

    def set_target(self, name: Optional[str]):
        self._yolo.set_target(name)

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

        self._last_good_t = now
        self._last_fail = ""
        return {
            "lat_err": lat,
            "lat_n_obj": n_obj,
            "n_raw": n_raw,
            "d_obj": None if d_obj is None else float(d_obj),
            "name": None if det is None else det.get("name"),
            "conf": None if det is None else det.get("conf"),
            "bbox": None if det is None else det.get("bbox"),
            "area": None if det is None else det.get("area"),
            "is_new": bool(rgb_new),
            "rgb_shape": rgb_shape,
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

    def _too_near(self, m: Optional[dict]) -> bool:
        """YOLO box is already at pick range — do not yaw or keep crawling."""
        d = self._d_obj(m)
        lim = float(self.cfg.already_near_m)
        n_obj = 0 if m is None else int(m.get("lat_n_obj") or 0)
        return n_obj > 0 and d is not None and lim > 0.0 and d < lim

    def _obj_stop(self, m: Optional[dict]) -> bool:
        """Object depth inside the box is close enough to stop, and not far off-axis."""
        d = self._d_obj(m)
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
        if has_obj:
            self._lat_err = float(lat)
        err = self._lat_err
        speed = self._speed()
        since = now - (self._phase_t0 or now)

        if self._too_near(m):
            return 0.0, 0.0
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
        same_h = [s for s in dcounts if s[0] == self._lock_rgb[0]]
        self._lock_depth = (
            max(same_h, key=lambda s: (s[0], s[1], dcounts[s])) if same_h
            else max(dcounts.keys(), key=lambda s: (s[0], s[1], dcounts[s]))
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
        if len(dcounts) > 1:
            others = ", ".join(
                f"{s[0]}x{s[1]}({c})" for s, c in dcounts.items()
                if s != self._lock_depth
            )
            print(
                f"[approach] depth lock {self._lock_depth[0]}x{self._lock_depth[1]}, "
                f"ignoring {others}"
            )
        elapsed = time.monotonic() - t_start
        self._frame_dt = (elapsed / max(1, n - 1)) if elapsed > 0 else None
        self._last_rgb_stamp = None
        self._last_depth_stamp = None
        self._last_new_frame_t = None
        return True

    def start(self) -> bool:
        """Activate auto-approach. Returns False if YOLO or cameras are unavailable."""
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
        self._detect_reason = ""
        self._lat_err = None
        self._det = None
        self._d_obj_cached = None
        self._drive_m = 0.0
        self._arrive_now = False
        self._skip_overrun = False
        self._hit_count = 0
        self._timeout_logged = False
        m0 = self._measure()
        if self._too_near(m0):
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
            f"overrun={self.cfg.overrun_m:.3f}m\n"
            f"[approach] stop-turn-go: yaw_gain={self.cfg.yaw_gain} "
            f"yaw=[{self.cfg.yaw_min},{self.cfg.yaw_max}]  "
            f"center_tol={self.cfg.center_tol} realign={self.cfg.realign} "
            f"hold={self.cfg.align_hold}  burst={self.cfg.burst_m:.3f}m  "
            f"stop={self.cfg.stop_m:.2f}m  already_near={self.cfg.already_near_m:.2f}m"
        )
        print(
            "[approach] live: err from YOLO box vs image center; "
            "d=depth inside that box. obj=used/raw. ALIGN waits for a detection "
            "(does not drive blind). Lost box → re-ALIGN, not stop."
        )
        return True

    def _overrun_duration_s(self) -> float:
        m = max(0.0, float(self.cfg.overrun_m))
        if m <= 0.0:
            return 0.0
        return m / max(1e-3, self._speed())

    def cancel(self):
        self.active = False
        self._hit_count = 0
        self._overrun = False
        self._overrun_t0 = None
        self._lat_err = None
        self._det = None
        self._d_obj_cached = None
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
                self._client.send_cmd_vel(0.0, 0.0)
                self.active = False
                print(f"[approach] STOP ({reason}) after t={t_s:.2f}s  (no overrun)")
                return True
            self._overrun = True
            self._overrun_t0 = time.monotonic()
            self._overrun_need_s = self._overrun_duration_s()
            print(
                f"[approach] DETECT ({reason}) — overrun {self.cfg.overrun_m:.3f}m "
                f"(~{self._overrun_need_s:.2f}s at {self._speed():.2f} m/s)"
            )
            self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)
            return False

        if self._overrun:
            assert self._overrun_t0 is not None
            elapsed = time.monotonic() - self._overrun_t0
            need = self._overrun_need_s
            if do_log:
                print(
                    f"[approach] t={t_s:5.2f}s  OVERRUN {elapsed:.2f}/{need:.2f}s "
                    f"(~{self.cfg.overrun_m:.3f}m)"
                )
            if elapsed >= need:
                self._client.send_cmd_vel(0.0, 0.0)
                self.active = False
                self._overrun = False
                print(
                    f"[approach] STOP after overrun ({self.cfg.overrun_m:.3f}m)  "
                    f"detect={self._detect_reason}  t={t_s:.2f}s"
                )
                return True
            self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)
            return False

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
        near = self._obj_stop(m) or self._too_near(m)
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
            fps = None if not self._frame_dt else 1.0 / self._frame_dt
            lat = m.get("lat_err")
            n_obj = int(m.get("lat_n_obj") or 0)
            d_obj = m.get("d_obj")
            name = m.get("name") or "-"
            conf = m.get("conf")
            n_raw = int(m.get("n_raw") or 0)
            print(
                f"[approach] t={t_s:5.2f}s  {self._phase.upper():5s} "
                f"err={'--' if lat is None else f'{lat:+.2f}'} "
                f"obj={n_obj}/{n_raw} {name}"
                f"{'' if conf is None else f' {conf:.2f}'}  "
                f"d={'--' if d_obj is None else f'{float(d_obj):.2f}'}  "
                f"ang={ang:+.2f}  "
                f"hit={self._hit_count}/{self.cfg.persist_k}  "
                f"v={lin:.2f} "
                f"{'' if fps is None else f'{fps:.1f}Hz '}"
                f"{','.join(flags) if flags else '-'}"
            )

        if self._hit_count >= self.cfg.persist_k:
            if self._too_near(m):
                reason = "already near"
            else:
                d = self._d_obj(m)
                reason = f"bbox_depth={d:.3f}" if d is not None else "in range"
            self._detect_reason = reason
            skip_overrun = self._too_near(m)
            self._overrun_need_s = 0.0 if skip_overrun else self._overrun_duration_s()
            if self._overrun_need_s <= 0.0:
                self._client.send_cmd_vel(0.0, 0.0)
                self.active = False
                print(f"[approach] STOP ({reason}) after t={t_s:.2f}s  (no overrun)")
                return True
            self._overrun = True
            self._overrun_t0 = time.monotonic()
            print(
                f"[approach] DETECT ({reason}) — overrun {self.cfg.overrun_m:.3f}m "
                f"(~{self._overrun_need_s:.2f}s at {self._speed():.2f} m/s)"
            )
            self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)
            return False

        self._client.send_cmd_vel(linear_x=lin, angular_z=ang)
        return False

    def settle(self):
        self._client.send_cmd_vel(0.0, 0.0)
        time.sleep(self.cfg.settle_s)
