"""User-triggered base auto-approach using base-camera depth contrast."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .images import (
    depth_center_surround,
    depth_lateral_error,
    depth_lower_stat,
    depth_roi_stat,
    image_to_depth,
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


# Depth frames arrive far slower than the 20 Hz control tick, so speed is
# clamped to max_step_m per observed frame interval and the base holds still
# whenever the measurement goes stale.
DEFAULT_CRAWL_MPS = 0.30
DEFAULT_MAX_STEP_M = 0.04
DEFAULT_STALE_STOP_S = 0.5
DEFAULT_CONTRAST_M = 0.015
DEFAULT_TAU_M = 0.08
DEFAULT_D_NEAR_M = 0.0
DEFAULT_PERSIST_K = 3
DEFAULT_INNER_FRAC = 0.18
DEFAULT_OUTER_FRAC = 0.40
DEFAULT_PERCENTILE = 30.0
DEFAULT_MIN_VALID = 4
DEFAULT_EMA_ALPHA = 0.50
DEFAULT_BASELINE_FRAMES = 20
DEFAULT_SETTLE_S = 0.5
DEFAULT_TICK_S = 0.05
DEFAULT_LOG_EVERY = 1
DEFAULT_OVERRUN_M = 0.285
DEFAULT_YAW_GAIN = 1.2
DEFAULT_YAW_MAX = 0.35
DEFAULT_YAW_MIN = 0.10
DEFAULT_CENTER_TOL = 0.05
DEFAULT_REALIGN = 0.12
DEFAULT_ALIGN_HOLD = 2
DEFAULT_ALIGN_SETTLE_S = 0.15
DEFAULT_ALIGN_TIMEOUT_S = 4.0
DEFAULT_BURST_M = 0.15
DEFAULT_LAT_MIN_M = 0.05
DEFAULT_LAT_MAX_M = 20.0
DEFAULT_LAT_CLOSER_M = 0.04
# Inner-patch depth that counts as "close enough to stop driving" when already
# aimed at the object. Empty-floor inner is ~0.70 m on the tilted base cam;
# contrast-detect used to fire around 0.64 m.
DEFAULT_INNER_STOP_M = 0.65
# Closer than this: already at pick range — stop with no overrun and do not yaw.
DEFAULT_ALREADY_NEAR_M = 0.50


@dataclass
class ApproachConfig:
    crawl_mps: float = DEFAULT_CRAWL_MPS
    max_step_m: float = DEFAULT_MAX_STEP_M
    stale_stop_s: float = DEFAULT_STALE_STOP_S
    contrast_m: float = DEFAULT_CONTRAST_M
    tau_m: float = DEFAULT_TAU_M
    d_near_m: float = DEFAULT_D_NEAR_M
    persist_k: int = DEFAULT_PERSIST_K
    baseline_frames: int = DEFAULT_BASELINE_FRAMES
    inner_frac: float = DEFAULT_INNER_FRAC
    outer_frac: float = DEFAULT_OUTER_FRAC
    percentile: float = DEFAULT_PERCENTILE
    min_valid: int = DEFAULT_MIN_VALID
    ema_alpha: float = DEFAULT_EMA_ALPHA
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
    lat_closer_m: float = DEFAULT_LAT_CLOSER_M
    inner_stop_m: float = DEFAULT_INNER_STOP_M
    already_near_m: float = DEFAULT_ALREADY_NEAR_M

    @classmethod
    def from_env(cls) -> "ApproachConfig":
        return cls(
            crawl_mps=_env_float("MARC_APPROACH_CRAWL", DEFAULT_CRAWL_MPS),
            max_step_m=_env_float("MARC_APPROACH_MAX_STEP", DEFAULT_MAX_STEP_M),
            stale_stop_s=_env_float("MARC_APPROACH_STALE_STOP", DEFAULT_STALE_STOP_S),
            contrast_m=_env_float("MARC_APPROACH_CONTRAST", DEFAULT_CONTRAST_M),
            tau_m=_env_float("MARC_APPROACH_TAU", DEFAULT_TAU_M),
            d_near_m=_env_float("MARC_APPROACH_D_NEAR", DEFAULT_D_NEAR_M),
            persist_k=_env_int("MARC_APPROACH_K", DEFAULT_PERSIST_K),
            baseline_frames=_env_int("MARC_APPROACH_BASELINE_FRAMES", DEFAULT_BASELINE_FRAMES),
            inner_frac=_env_float("MARC_APPROACH_INNER_FRAC", DEFAULT_INNER_FRAC),
            outer_frac=_env_float("MARC_APPROACH_OUTER_FRAC", DEFAULT_OUTER_FRAC),
            percentile=_env_float("MARC_APPROACH_PERCENTILE", DEFAULT_PERCENTILE),
            min_valid=_env_int("MARC_APPROACH_MIN_VALID", DEFAULT_MIN_VALID),
            ema_alpha=_env_float("MARC_APPROACH_EMA", DEFAULT_EMA_ALPHA),
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
            lat_closer_m=_env_float("MARC_APPROACH_LAT_CLOSER_M", DEFAULT_LAT_CLOSER_M),
            inner_stop_m=_env_float("MARC_APPROACH_INNER_STOP", DEFAULT_INNER_STOP_M),
            already_near_m=_env_float("MARC_APPROACH_ALREADY_NEAR", DEFAULT_ALREADY_NEAR_M),
        )


def _stamp_key(msg) -> Optional[tuple]:
    try:
        st = msg.header.stamp
        return (int(st.sec), int(st.nanosec))
    except Exception:
        return None


class DepthApproachController:
    """Approach an object by alternating turn-in-place centering with straight crawls.

    Linear and angular velocity are never commanded together: the base either
    yaws (ALIGN) or drives (DRIVE), so the object stays centered in the base
    camera instead of being walked past.
    """

    def __init__(self, client, cfg: Optional[ApproachConfig] = None):
        self._client = client
        self.cfg = cfg or ApproachConfig.from_env()
        self.active = False
        self.d0: Optional[float] = None
        self.c0: Optional[float] = None
        self.dl0: Optional[float] = None
        self._hit_count = 0
        self._last_depth: Optional[float] = None
        self._c_filt: Optional[float] = None
        self._tick_i = 0
        self._t0: Optional[float] = None
        self._overrun = False
        self._overrun_t0: Optional[float] = None
        self._overrun_need_s = 0.0
        self._detect_reason: str = ""
        # Frame freshness / rate tracking
        self._last_stamp: Optional[tuple] = None
        self._last_new_frame_t: Optional[float] = None
        self._frame_dt: Optional[float] = None
        self._last_good_t: Optional[float] = None
        self._last_fail: str = ""
        self._lock_shape: Optional[tuple] = None
        self._lat_err: Optional[float] = None
        # Stop-turn-go phase: "align" yaws in place, "drive" goes straight.
        self._phase: str = "align"
        self._phase_t0: Optional[float] = None
        self._phase_why: str = "start"
        self._align_ok = 0
        self._drive_m = 0.0
        self._seen_obj = False
        self._arrive_now = False
        self._skip_overrun = False

    def _measure(self) -> Optional[dict]:
        """Read one depth frame; returns metrics dict or None (reason in _last_fail).

        ``is_new`` marks frames whose header stamp changed since the last read.
        """
        msg = self._client.get_robot_depth("base")
        if msg is None:
            self._last_fail = "no depth msg"
            return None
        depth = image_to_depth(msg)
        if depth is None:
            self._last_fail = f"decode failed (encoding={getattr(msg, 'encoding', '?')})"
            return None
        # Two publishers share this topic at different resolutions and different
        # geometry, so their contrast values interleave. Baseline picks one; the
        # rest are dropped here (before the rate/freshness bookkeeping, so the
        # measured frame rate reflects the stream we actually steer on).
        shape = tuple(depth.shape[:2])
        if self._lock_shape is not None and shape != self._lock_shape:
            self._last_fail = (
                f"other stream {shape[0]}x{shape[1]} "
                f"(locked {self._lock_shape[0]}x{self._lock_shape[1]})"
            )
            return None

        stamp = _stamp_key(msg)
        now = time.monotonic()
        is_new = stamp is None or stamp != self._last_stamp
        if is_new:
            if self._last_new_frame_t is not None:
                dt = now - self._last_new_frame_t
                if 0.0 < dt < 5.0:
                    self._frame_dt = dt if self._frame_dt is None else (
                        0.4 * dt + 0.6 * self._frame_dt
                    )
            self._last_new_frame_t = now
            self._last_stamp = stamp

        cs = depth_center_surround(
            depth,
            inner_frac=self.cfg.inner_frac,
            outer_frac=self.cfg.outer_frac,
            percentile=self.cfg.percentile,
            min_valid=self.cfg.min_valid,
        )
        if cs is None or cs.get("contrast") is None:
            self._last_fail = (
                f"no contrast ({cs.get('reason') if cs else 'empty depth'})"
            )
            return None
        d_med = depth_roi_stat(depth, frac=self.cfg.outer_frac, percentile=50.0)
        if d_med is None:
            d_med = cs["d_ring"]
        cs["d_med"] = float(d_med)
        cs["is_new"] = bool(is_new)
        cs["stamp"] = stamp
        lat = depth_lateral_error(
            depth,
            percentile=self.cfg.percentile,
            min_valid=self.cfg.min_valid,
            closer_m=self.cfg.lat_closer_m,
            min_m=self.cfg.lat_min_m,
            max_m=self.cfg.lat_max_m,
        )
        cs["lat_err"] = None if lat is None else lat["error"]
        cs["lat_n_obj"] = 0 if lat is None else lat["n_obj"]
        cs["lat_floor"] = None if lat is None else lat["floor"]
        d_low = depth_lower_stat(
            depth,
            percentile=self.cfg.percentile,
            min_m=self.cfg.lat_min_m,
            max_m=self.cfg.lat_max_m,
            min_valid=self.cfg.min_valid,
        )
        cs["d_lower"] = d_low
        self._last_good_t = now
        self._last_fail = ""
        return cs

    def _smooth_contrast(self, c_raw: float) -> float:
        a = float(np.clip(self.cfg.ema_alpha, 0.05, 1.0))
        if self._c_filt is None:
            self._c_filt = c_raw
        else:
            self._c_filt = a * c_raw + (1.0 - a) * self._c_filt
        return float(self._c_filt)

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
        # A tiny command does not overcome base stiction, so it would stall the
        # align phase until the timeout instead of centering.
        mag = max(mag, float(self.cfg.yaw_min))
        return math.copysign(mag, ang)

    def _is_centered(self, m: Optional[dict]) -> bool:
        if m is None:
            return False
        n_obj = int(m.get("lat_n_obj") or 0)
        lat = m.get("lat_err")
        return n_obj > 0 and lat is not None and abs(float(lat)) <= self.cfg.center_tol

    def _d_inner(self, m: Optional[dict]) -> Optional[float]:
        if m is None or m.get("d_inner") is None:
            return None
        return float(m["d_inner"])

    def _too_near(self, m: Optional[dict]) -> bool:
        """Already inside pick range — do not yaw or keep crawling.

        The inner patch stays on the floor (~0.70 m) even when the object is at
        the robot's feet: a tilted camera puts a close object in the *lower*
        rows. So this also fires when the lower-band depth has dropped vs
        baseline, or when a previously seen object has left the mid-height
        centering band (it dropped below the camera).
        """
        lim = float(self.cfg.already_near_m)
        d = self._d_inner(m)
        if d is not None and lim > 0.0 and d < lim:
            return True
        d_low = None if m is None else m.get("d_lower")
        if (
            d_low is not None
            and self.dl0 is not None
            and (self.dl0 - float(d_low)) > self.cfg.tau_m
        ):
            return True
        n_obj = 0 if m is None else int(m.get("lat_n_obj") or 0)
        return bool(self._seen_obj and n_obj <= 0)

    def _inner_stop(self, m: Optional[dict]) -> bool:
        """Inner patch is at the distance contrast-detect used to fire."""
        d = self._d_inner(m)
        lim = float(self.cfg.inner_stop_m)
        if self.cfg.d_near_m > 0.0:
            lim = min(lim, float(self.cfg.d_near_m)) if lim > 0.0 else float(self.cfg.d_near_m)
        return d is not None and lim > 0.0 and d < lim

    def _enter(self, phase: str, why: str):
        self._phase = phase
        self._phase_t0 = time.monotonic()
        self._align_ok = 0
        self._phase_why = why
        if self.cfg.log_every > 0:
            t_s = 0.0 if self._t0 is None else (time.monotonic() - self._t0)
            print(f"[approach] t={t_s:5.2f}s  -> {phase.upper()} ({why})")

    def _command(self, m: Optional[dict]) -> tuple:
        """Alternate between turning in place and driving straight.

        ALIGN yaws with zero linear velocity until the object sits within
        ``center_tol`` of the image center; DRIVE goes straight and hands control
        back to ALIGN as soon as the object drifts past ``realign`` or after
        ``burst_m`` of travel, so centering is re-checked often.
        """
        now = time.monotonic()
        lat = None if m is None else m.get("lat_err")
        n_obj = 0 if m is None else int(m.get("lat_n_obj") or 0)
        is_new = bool(m is not None and m.get("is_new"))
        # error == 0.0 with no object columns means "nothing to centre on", not "centred"
        has_obj = lat is not None and n_obj > 0
        if has_obj:
            self._lat_err = float(lat)
            self._seen_obj = True
        err = self._lat_err
        speed = self._speed()
        since = now - (self._phase_t0 or now)

        # Too close to yaw: hold still and let tick() stop the approach.
        if self._too_near(m):
            return 0.0, 0.0
        # Already on the object: drive straight until inner-stop, never re-ALIGN.
        if self._is_centered(m):
            if self._phase != "drive":
                self._enter("drive", f"already centered err={err:+.2f}")
            return speed, 0.0

        if self._phase == "align":
            if since < self.cfg.align_settle_s:
                # Stand still after braking so depth catches up with the new pose.
                return 0.0, 0.0
            if not has_obj or err is None:
                # Nothing to centre on: do not sit in ALIGN. If we already had
                # an object and it vanished, hold still (too-near / lost).
                if self._seen_obj:
                    return 0.0, 0.0
                self._enter("drive", "no centering signal")
                return speed, 0.0
            if abs(err) <= self.cfg.center_tol:
                if is_new:
                    self._align_ok += 1
                if self._align_ok >= max(1, self.cfg.align_hold):
                    self._enter("drive", f"centered err={err:+.2f}")
                    return speed, 0.0
                return 0.0, 0.0
            self._align_ok = 0
            if since >= self.cfg.align_timeout_s:
                self._enter("drive", f"align timeout err={err:+.2f}")
                return speed, 0.0
            return 0.0, self._yaw(err)

        self._drive_m = since * speed
        if has_obj and abs(err) > self.cfg.realign:
            self._enter("align", f"off-center err={err:+.2f}")
            return 0.0, 0.0
        if (
            has_obj
            and self.cfg.burst_m > 0.0
            and self._drive_m >= self.cfg.burst_m
        ):
            self._enter("align", f"burst {self._drive_m:.2f}m")
            return 0.0, 0.0
        return speed, 0.0

    def capture_baseline(self) -> Optional[float]:
        """Stop base; measure baseline ROI depth and center-surround contrast."""
        self._client.send_cmd_vel(0.0, 0.0)
        samples = []  # (shape, d_med, raw_contrast, stamp, d_lower)
        self._c_filt = None
        self._lock_shape = None  # observe every stream, then pick one below
        t_start = time.monotonic()
        for i in range(self.cfg.baseline_frames):
            m = self._measure()
            if m is not None:
                samples.append(
                    (
                        tuple(m["shape"][:2]),
                        m["d_med"],
                        m["contrast"],
                        m["stamp"],
                        m.get("d_lower"),
                    )
                )
                if self.cfg.log_every > 0:
                    print(
                        f"[approach] baseline {i+1}/{self.cfg.baseline_frames}  "
                        f"d={m['d_med']:.3f}  cin={m['d_inner']:.3f} "
                        f"ring={m['d_ring']:.3f}  contrast={m['contrast']:+.3f}  "
                        f"px(in/ring)={m['n_inner']}/{m['n_ring']}  "
                        f"img={m['shape'][0]}x{m['shape'][1]}"
                    )
            elif self.cfg.log_every > 0:
                print(
                    f"[approach] baseline {i+1}/{self.cfg.baseline_frames}  "
                    f"unusable: {self._last_fail}"
                )
            time.sleep(self.cfg.tick_s)
        if not samples:
            self.d0 = None
            self.c0 = None
            self.dl0 = None
            return None

        counts: dict = {}
        for shape, _, _, _, _ in samples:
            counts[shape] = counts.get(shape, 0) + 1
        self._lock_shape = max(counts.items(), key=lambda kv: kv[1])[0]
        if len(counts) > 1:
            others = ", ".join(
                f"{s[0]}x{s[1]}({n})" for s, n in counts.items() if s != self._lock_shape
            )
            print(f"[approach] WARNING: base depth topic has multiple streams — "
                  f"locking {self._lock_shape[0]}x{self._lock_shape[1]}, "
                  f"ignoring {others}")
        sel = [
            (d, c, st, dl)
            for shape, d, c, st, dl in samples
            if shape == self._lock_shape
        ]
        self.d0 = float(np.median(np.asarray([d for d, _, _, _ in sel], dtype=np.float64)))
        self.c0 = float(np.median(np.asarray([c for _, c, _, _ in sel], dtype=np.float64)))
        lows = [dl for _, _, _, dl in sel if dl is not None]
        self.dl0 = (
            float(np.median(np.asarray(lows, dtype=np.float64))) if lows else None
        )
        self._hit_count = 0
        self._last_depth = self.d0
        self._c_filt = self.c0
        # Rate measured across both streams overestimates the one we steer on, so
        # re-estimate from distinct stamps of the locked stream only.
        distinct = len({st for _, _, st, _ in sel if st is not None})
        elapsed = time.monotonic() - t_start
        self._frame_dt = (elapsed / (distinct - 1)) if distinct >= 2 and elapsed > 0 else None
        self._last_stamp = None
        self._last_new_frame_t = None
        return self.d0

    def start(self) -> bool:
        """Activate auto-approach. Returns False if baseline could not be measured."""
        d0 = self.capture_baseline()
        if d0 is None or self.c0 is None:
            self.active = False
            print(f"[approach] baseline FAILED: {self._last_fail}")
            return False
        self.active = True
        self._tick_i = 0
        self._t0 = time.monotonic()
        self._overrun = False
        self._overrun_t0 = None
        self._detect_reason = ""
        self._lat_err = None
        self._drive_m = 0.0
        self._seen_obj = False
        self._arrive_now = False
        self._skip_overrun = False
        m0 = self._measure()
        if self._too_near(m0):
            self._enter("drive", "already near")
            self._arrive_now = True
            self._skip_overrun = True
            self._detect_reason = "already near"
        elif self._is_centered(m0) and self._inner_stop(m0):
            self._enter("drive", "already centered")
            self._arrive_now = True
            self._detect_reason = "already centered"
        elif self._is_centered(m0):
            self._enter("drive", "already centered")
        else:
            self._enter("align", "start")
        self._last_good_t = time.monotonic()
        near_s = "off" if self.cfg.d_near_m <= 0.0 else f"{self.cfg.d_near_m:.3f}"
        fps = None if not self._frame_dt else 1.0 / self._frame_dt
        print(
            f"[approach] baseline d0={d0:.3f} c0={self.c0:+.3f}  "
            f"dl0={'--' if self.dl0 is None else f'{self.dl0:.3f}'}  "
            f"contrast_tau={self.cfg.contrast_m}  drop_tau={self.cfg.tau_m}  "
            f"d_near={near_s}  K={self.cfg.persist_k}\n"
            f"[approach] crawl={self.cfg.crawl_mps} clamped to "
            f"{self._speed():.3f} m/s  (max_step={self.cfg.max_step_m}m, "
            f"depth≈{'?' if fps is None else f'{fps:.1f}'} Hz)  "
            f"stale_stop={self.cfg.stale_stop_s}s  "
            f"overrun={self.cfg.overrun_m:.3f}m\n"
            f"[approach] stop-turn-go: yaw_gain={self.cfg.yaw_gain} "
            f"yaw=[{self.cfg.yaw_min},{self.cfg.yaw_max}]  "
            f"center_tol={self.cfg.center_tol} realign={self.cfg.realign} "
            f"hold={self.cfg.align_hold}  burst={self.cfg.burst_m:.3f}m  "
            f"align_timeout={self.cfg.align_timeout_s}s  "
            f"inner_stop={self.cfg.inner_stop_m:.2f}m  "
            f"already_near={self.cfg.already_near_m:.2f}m"
        )
        print(
            "[approach] live: rise = (ring−center) − c0. Object in the center "
            "raises it; pitch/bumps cancel. ALIGN is skipped when the object is "
            "already centered or too near; then it drives or stops. "
            "Base HOLDS when depth goes stale."
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
        self._c_filt = None
        self._lat_err = None
        self._phase = "align"
        self._phase_t0 = None
        self._align_ok = 0
        self._drive_m = 0.0
        self._seen_obj = False
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
            # Already centered when detection fired; creep straight for the last
            # few centimetres so the object cannot be yawed out of the gripper.
            self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)
            return False

        m = self._measure()

        if m is None:
            age = (
                999.0 if self._last_good_t is None
                else time.monotonic() - self._last_good_t
            )
            if age > self.cfg.stale_stop_s:
                # Never drive blind: hold position until depth comes back.
                self._client.send_cmd_vel(0.0, 0.0)
                if do_log:
                    print(
                        f"[approach] t={t_s:5.2f}s  HOLD (depth stale {age:.2f}s > "
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

        contrast = self._smooth_contrast(m["contrast"])
        d_med = m["d_med"]
        self._last_depth = d_med
        assert self.d0 is not None and self.c0 is not None
        lin, ang = self._command(m)

        c_rise = contrast - self.c0
        contrast_hit = c_rise > self.cfg.contrast_m
        drop = self.d0 - d_med
        drop_hit = drop > self.cfg.tau_m
        near = (
            (self.cfg.d_near_m > 0.0 and d_med < self.cfg.d_near_m)
            or self._too_near(m)
            or (self._is_centered(m) and self._inner_stop(m))
        )

        # Yawing in place sweeps the view across whatever is beside the object, so
        # contrast/drop there says nothing about closing in: only detect while driving.
        # Exception: already-near / lost-object must still count, or ALIGN would
        # hold forever with hit=0.
        if self._phase == "align" and not near:
            self._hit_count = 0
        elif contrast_hit or drop_hit or near:
            if m["is_new"]:
                self._hit_count += 1
        else:
            self._hit_count = 0

        if do_log:
            flags = []
            if contrast_hit:
                flags.append("CTR")
            if drop_hit:
                flags.append("DROP")
            if near:
                flags.append("NEAR")
            fps = None if not self._frame_dt else 1.0 / self._frame_dt
            lat = m.get("lat_err")
            n_obj = int(m.get("lat_n_obj") or 0)
            flr = m.get("lat_floor")
            d_low = m.get("d_lower")
            print(
                f"[approach] t={t_s:5.2f}s  {self._phase.upper():5s} "
                f"cin={m['d_inner']:.3f} "
                f"ring={m['d_ring']:.3f}  c={contrast:+.3f} c0={self.c0:+.3f}  "
                f"rise={c_rise:+.3f}  drop={drop:+.3f}  "
                f"err={'--' if lat is None else f'{lat:+.2f}'} "
                f"obj={n_obj} flr={'--' if flr is None else f'{flr:.2f}'} "
                f"low={'--' if d_low is None else f'{float(d_low):.2f}'} "
                f"ang={ang:+.2f}  "
                f"hit={self._hit_count}/{self.cfg.persist_k}  "
                f"v={lin:.2f} "
                f"{'' if fps is None else f'{fps:.1f}Hz '}"
                f"{','.join(flags) if flags else '-'}"
            )

        if self._hit_count >= self.cfg.persist_k:
            if self._too_near(m):
                n_now = int(m.get("lat_n_obj") or 0)
                reason = (
                    "object lost (too near)"
                    if self._seen_obj and n_now <= 0
                    else "already near"
                )
            elif near:
                reason = "near"
            elif contrast_hit:
                reason = f"contrast_rise={c_rise:.3f}"
            else:
                reason = f"drop={drop:.3f}"
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
