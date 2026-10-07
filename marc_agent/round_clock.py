"""Per-round remaining-time snapshot. Count down locally; do not poll the platform."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional


DEFAULT_MIN_REMAINING_S = 4.0
DEFAULT_SUBMIT_MARGIN_S = 2.0
# The all-camera scene description is optional context: it may use at most this
# long and this share of the remaining budget, leaving the rest to the target pick.
DEFAULT_SCENE_TIMEOUT_S = 8.0
DEFAULT_SCENE_BUDGET_FRACTION = 0.4


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


def scene_timeout_s() -> float:
    return _env_float("MARC_QWEN_SCENE_TIMEOUT_S", DEFAULT_SCENE_TIMEOUT_S)


def scene_budget_fraction() -> float:
    return _env_float("MARC_QWEN_SCENE_BUDGET_FRACTION", DEFAULT_SCENE_BUDGET_FRACTION)


class RoundClock:
    """The current mission's time limit, counted down with a monotonic clock."""

    def __init__(self, monotonic: Callable[[], float] = time.monotonic):
        self._monotonic = monotonic
        self._deadline: Optional[float] = None

    def begin(self, time_limit, now) -> None:
        """Start a round independently of simulation-clock jumps."""
        self._deadline = self._monotonic() + float(time_limit or 0.0)

    def note(self, remaining, now) -> None:
        """Ignore unscoped msg 203 updates, which can arrive from the prior round."""

    def remaining(self, now) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self._monotonic())


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
