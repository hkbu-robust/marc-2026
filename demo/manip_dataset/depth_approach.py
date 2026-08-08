"""User-triggered base auto-approach using base-camera depth contrast."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .images import depth_center_surround, depth_roi_stat, image_to_depth


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
DEFAULT_OVERRUN_M = 1.75


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
        )


def _stamp_key(msg) -> Optional[tuple]:
    try:
        st = msg.header.stamp
        return (int(st.sec), int(st.nanosec))
    except Exception:
        return None


class DepthApproachController:
    """Crawl forward until center-surround depth contrast detects an object."""

    def __init__(self, client, cfg: Optional[ApproachConfig] = None):
        self._client = client
        self.cfg = cfg or ApproachConfig.from_env()
        self.active = False
        self.d0: Optional[float] = None
        self.c0: Optional[float] = None
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

    def capture_baseline(self) -> Optional[float]:
        """Stop base; measure baseline ROI depth and center-surround contrast."""
        self._client.send_cmd_vel(0.0, 0.0)
        samples = []  # (shape, d_med, raw_contrast, stamp)
        self._c_filt = None
        self._lock_shape = None  # observe every stream, then pick one below
        t_start = time.monotonic()
        for i in range(self.cfg.baseline_frames):
            m = self._measure()
            if m is not None:
                samples.append(
                    (tuple(m["shape"][:2]), m["d_med"], m["contrast"], m["stamp"])
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
            return None

        counts: dict = {}
        for shape, _, _, _ in samples:
            counts[shape] = counts.get(shape, 0) + 1
        self._lock_shape = max(counts.items(), key=lambda kv: kv[1])[0]
        if len(counts) > 1:
            others = ", ".join(
                f"{s[0]}x{s[1]}({n})" for s, n in counts.items() if s != self._lock_shape
            )
            print(f"[approach] WARNING: base depth topic has multiple streams — "
                  f"locking {self._lock_shape[0]}x{self._lock_shape[1]}, "
                  f"ignoring {others}")
        sel = [(d, c, st) for shape, d, c, st in samples if shape == self._lock_shape]
        self.d0 = float(np.median(np.asarray([d for d, _, _ in sel], dtype=np.float64)))
        self.c0 = float(np.median(np.asarray([c for _, c, _ in sel], dtype=np.float64)))
        self._hit_count = 0
        self._last_depth = self.d0
        self._c_filt = self.c0
        # Rate measured across both streams overestimates the one we steer on, so
        # re-estimate from distinct stamps of the locked stream only.
        distinct = len({st for _, _, st in sel if st is not None})
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
        self._last_good_t = time.monotonic()
        near_s = "off" if self.cfg.d_near_m <= 0.0 else f"{self.cfg.d_near_m:.3f}"
        fps = None if not self._frame_dt else 1.0 / self._frame_dt
        print(
            f"[approach] baseline d0={d0:.3f} c0={self.c0:+.3f}  "
            f"contrast_tau={self.cfg.contrast_m}  drop_tau={self.cfg.tau_m}  "
            f"d_near={near_s}  K={self.cfg.persist_k}\n"
            f"[approach] crawl={self.cfg.crawl_mps} clamped to "
            f"{self._speed():.3f} m/s  (max_step={self.cfg.max_step_m}m, "
            f"depth≈{'?' if fps is None else f'{fps:.1f}'} Hz)  "
            f"stale_stop={self.cfg.stale_stop_s}s  "
            f"overrun={self.cfg.overrun_m:.3f}m"
        )
        print(
            "[approach] live: rise = (ring−center) − c0. Object in the center "
            "raises it; pitch/bumps cancel. Base HOLDS when depth goes stale."
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
        self._client.send_cmd_vel(0.0, 0.0)

    def tick(self) -> bool:
        """One control tick. Returns True when approach finished (caller should settle)."""
        if not self.active:
            return False

        self._tick_i += 1
        t_s = 0.0 if self._t0 is None else (time.monotonic() - self._t0)
        do_log = self.cfg.log_every > 0 and (self._tick_i % self.cfg.log_every == 0)

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
                # Never drive blind: hold position until depth comes back.
                self._client.send_cmd_vel(0.0, 0.0)
                if do_log:
                    print(
                        f"[approach] t={t_s:5.2f}s  HOLD (depth stale {age:.2f}s > "
                        f"{self.cfg.stale_stop_s}s): {self._last_fail}"
                    )
            else:
                self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)
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

        c_rise = contrast - self.c0
        contrast_hit = c_rise > self.cfg.contrast_m
        drop = self.d0 - d_med
        drop_hit = drop > self.cfg.tau_m
        near = self.cfg.d_near_m > 0.0 and d_med < self.cfg.d_near_m

        # Only count persistence on genuinely new frames, so a slow depth stream
        # cannot reach K by re-reading one cached frame.
        if contrast_hit or drop_hit or near:
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
            print(
                f"[approach] t={t_s:5.2f}s  cin={m['d_inner']:.3f} "
                f"ring={m['d_ring']:.3f}  c={contrast:+.3f} c0={self.c0:+.3f}  "
                f"rise={c_rise:+.3f}  drop={drop:+.3f}  "
                f"hit={self._hit_count}/{self.cfg.persist_k}  "
                f"v={self._speed():.2f} "
                f"{'' if fps is None else f'{fps:.1f}Hz '}"
                f"{','.join(flags) if flags else '-'}"
            )

        if self._hit_count >= self.cfg.persist_k:
            if near:
                reason = "near"
            elif contrast_hit:
                reason = f"contrast_rise={c_rise:.3f}"
            else:
                reason = f"drop={drop:.3f}"
            self._detect_reason = reason
            self._overrun_need_s = self._overrun_duration_s()
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

        self._client.send_cmd_vel(linear_x=self._speed(), angular_z=0.0)
        return False

    def settle(self):
        self._client.send_cmd_vel(0.0, 0.0)
        time.sleep(self.cfg.settle_s)
