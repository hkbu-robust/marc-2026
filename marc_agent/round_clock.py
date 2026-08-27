"""Per-round remaining-time snapshot. Count down locally; do not poll the platform."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_MIN_REMAINING_S = 4.0
DEFAULT_SUBMIT_MARGIN_S = 2.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def min_remaining_s() -> float:
    return _env_float("MARC_QWEN_MIN_REMAINING_S", DEFAULT_MIN_REMAINING_S)


def submit_margin_s() -> float:
    return _env_float("MARC_QWEN_SUBMIT_MARGIN_S", DEFAULT_SUBMIT_MARGIN_S)


class RoundClock:
    """Mission time_limit plus optional msg 203, counted down with ``now``."""

    def __init__(self):
        self._limit: Optional[float] = None
        self._t0: Optional[float] = None
        self._platform_remaining: Optional[float] = None
        self._platform_stamp: Optional[float] = None

    def begin(self, time_limit, now) -> None:
        """Start a round. Drops any leftover platform remaining from the last round."""
        self._limit = float(time_limit or 0.0)
        self._t0 = float(now)
        self._platform_remaining = None
        self._platform_stamp = None

    def note(self, remaining, now) -> None:
        """Record a pushed remaining-time update. Ignored before ``begin``."""
        if self._t0 is None:
            return
        self._platform_remaining = float(remaining)
        self._platform_stamp = float(now)

    def remaining(self, now) -> Optional[float]:
        if self._t0 is None or self._limit is None:
            return None
        local = self._limit - (float(now) - self._t0)
        if self._platform_remaining is not None and self._platform_stamp is not None:
            platform = self._platform_remaining - (float(now) - self._platform_stamp)
            return max(0.0, min(local, platform))
        return max(0.0, local)


@dataclass(frozen=True)
class QwenBudget:
    remaining: Optional[float]
    skip: bool
    timeout_s: Optional[float]
    capped: bool


def qwen_budget(
    clock: Optional[RoundClock],
    now,
    *,
    default_timeout: Optional[float] = None,
) -> QwenBudget:
    """Decide whether to skip Qwen and how long a started call may run."""
    if clock is None:
        return QwenBudget(None, False, default_timeout, False)
    remaining = clock.remaining(now)
    if remaining is None:
        return QwenBudget(None, False, default_timeout, False)
    floor = min_remaining_s()
    margin = submit_margin_s()
    skip = floor > 0.0 and remaining < floor
    timeout = remaining - margin
    if timeout <= 0.0:
        if floor > 0.0:
            skip = True
            timeout = None
        else:
            timeout = remaining
    if skip:
        return QwenBudget(remaining, True, None, False)
    if default_timeout is not None:
        timeout = min(float(default_timeout), float(timeout))
        capped = timeout + 1e-9 < float(default_timeout)
    else:
        capped = False
    return QwenBudget(remaining, False, float(timeout), capped)
