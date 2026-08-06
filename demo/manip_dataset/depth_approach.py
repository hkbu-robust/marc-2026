"""User-triggered base auto-approach using base-camera depth residual."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from .images import depth_roi_median, image_to_depth


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class ApproachConfig:
    crawl_mps: float = 0.05
    tau_m: float = 0.15
    d_near_m: float = 0.55
    persist_k: int = 5
    baseline_frames: int = 15
    roi_frac: float = 0.4
    settle_s: float = 0.5
    tick_s: float = 0.05

    @classmethod
    def from_env(cls) -> "ApproachConfig":
        return cls(
            crawl_mps=_env_float("MARC_APPROACH_CRAWL", 0.05),
            tau_m=_env_float("MARC_APPROACH_TAU", 0.15),
            d_near_m=_env_float("MARC_APPROACH_D_NEAR", 0.55),
            persist_k=_env_int("MARC_APPROACH_K", 5),
            baseline_frames=_env_int("MARC_APPROACH_BASELINE_FRAMES", 15),
            settle_s=_env_float("MARC_APPROACH_SETTLE", 0.5),
        )


class DepthApproachController:
    """Crawl forward until base depth changes vs a baseline taken at activation."""

    def __init__(self, client, cfg: Optional[ApproachConfig] = None):
        self._client = client
        self.cfg = cfg or ApproachConfig.from_env()
        self.active = False
        self.d0: Optional[float] = None
        self._hit_count = 0
        self._last_median: Optional[float] = None

    def _read_median(self) -> Optional[float]:
        msg = self._client.get_robot_depth("base")
        depth = image_to_depth(msg)
        return depth_roi_median(depth, self.cfg.roi_frac)

    def capture_baseline(self) -> Optional[float]:
        """Stop base and average several ROI medians into d0."""
        self._client.send_cmd_vel(0.0, 0.0)
        samples = []
        for _ in range(self.cfg.baseline_frames):
            m = self._read_median()
            if m is not None:
                samples.append(m)
            time.sleep(self.cfg.tick_s)
        if not samples:
            self.d0 = None
            return None
        self.d0 = float(np_median(samples))
        self._hit_count = 0
        self._last_median = self.d0
        return self.d0

    def start(self) -> bool:
        """Activate auto-approach. Returns False if baseline could not be measured."""
        d0 = self.capture_baseline()
        if d0 is None:
            self.active = False
            return False
        self.active = True
        print(f"[approach] baseline d0={d0:.3f} m  tau={self.cfg.tau_m}  "
              f"d_near={self.cfg.d_near_m}  crawl={self.cfg.crawl_mps}")
        return True

    def cancel(self):
        self.active = False
        self._hit_count = 0
        self._client.send_cmd_vel(0.0, 0.0)

    def tick(self) -> bool:
        """One control tick. Returns True when stop condition met (caller should settle)."""
        if not self.active:
            return False

        med = self._read_median()
        if med is not None:
            self._last_median = med
            assert self.d0 is not None
            residual = abs(med - self.d0)
            near = med < self.cfg.d_near_m
            if residual > self.cfg.tau_m or near:
                self._hit_count += 1
            else:
                self._hit_count = 0

            if self._hit_count >= self.cfg.persist_k:
                self._client.send_cmd_vel(0.0, 0.0)
                self.active = False
                reason = "near" if near else f"residual={residual:.3f}"
                print(f"[approach] STOP median={med:.3f} ({reason})")
                return True

        self._client.send_cmd_vel(linear_x=self.cfg.crawl_mps, angular_z=0.0)
        return False

    def settle(self):
        self._client.send_cmd_vel(0.0, 0.0)
        time.sleep(self.cfg.settle_s)


def np_median(xs):
    import numpy as np
    return float(np.median(np.asarray(xs, dtype=np.float64)))
