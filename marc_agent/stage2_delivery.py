"""Stage 2 deadline, retry budgeting, and owner-return policy."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Optional


DEFAULT_PLATFORM_GUARD_S = 2.0
DEFAULT_CALLBACK_LAG_GUARD_S = 5.0
DEFAULT_RETURN_PROGRESS_MPS = 0.5
DEFAULT_RETURN_MARGIN_S = 60.0
DEFAULT_OWNER_TOLERANCE_M = 0.7
DROP_STOP_REASON = "basket_drop_recovery"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class DeliveryConfig:
    platform_guard_s: float = DEFAULT_PLATFORM_GUARD_S
    callback_lag_guard_s: float = DEFAULT_CALLBACK_LAG_GUARD_S
    return_progress_mps: float = DEFAULT_RETURN_PROGRESS_MPS
    return_margin_s: float = DEFAULT_RETURN_MARGIN_S
    owner_tolerance_m: float = DEFAULT_OWNER_TOLERANCE_M

    @classmethod
    def from_env(cls):
        return cls(
            platform_guard_s=max(0.0, _env_float(
                "MARC_STAGE2_PLATFORM_GUARD_S", DEFAULT_PLATFORM_GUARD_S,
            )),
            callback_lag_guard_s=max(0.0, _env_float(
                "MARC_STAGE2_CALLBACK_LAG_GUARD_S", DEFAULT_CALLBACK_LAG_GUARD_S,
            )),
            return_progress_mps=max(1e-6, _env_float(
                "MARC_STAGE2_RETURN_PROGRESS_MPS", DEFAULT_RETURN_PROGRESS_MPS,
            )),
            return_margin_s=max(0.0, _env_float(
                "MARC_STAGE2_RETURN_MARGIN_S", DEFAULT_RETURN_MARGIN_S,
            )),
            owner_tolerance_m=max(0.0, _env_float(
                "MARC_STAGE2_OWNER_TOLERANCE_M", DEFAULT_OWNER_TOLERANCE_M,
            )),
        )


class Stage2Deadline:
    """One absolute simulation-time deadline for the active Stage 2 mission."""

    def __init__(self, config: Optional[DeliveryConfig] = None):
        self.config = config or DeliveryConfig.from_env()
        self.entry_s: Optional[float] = None
        self.deadline_s: Optional[float] = None
        self.active = False
        self.source: Optional[str] = None

    def enter(self, now_s: float) -> None:
        if not self.active:
            self.entry_s = float(now_s)
        self.active = True

    def leave(self) -> None:
        self.active = False

    def begin(self, time_limit_s: float, callback_now_s: float) -> float:
        callback_now_s = float(callback_now_s)
        if self.entry_s is not None:
            self.deadline_s = (
                self.entry_s + float(time_limit_s) - self.config.platform_guard_s
            )
            self.source = "stage2_entry"
        else:
            self.deadline_s = (
                callback_now_s + float(time_limit_s)
                - self.config.callback_lag_guard_s
            )
            self.source = "mission_fallback"
        self.active = True
        return self.deadline_s

    def note_remaining(self, remaining_s: float, now_s: float) -> None:
        if not self.active or self.deadline_s is None:
            return
        candidate = float(now_s) + max(0.0, float(remaining_s)) - self.config.platform_guard_s
        self.deadline_s = min(self.deadline_s, candidate)

    def remaining(self, now_s: float) -> float:
        if self.deadline_s is None:
            return 0.0
        return max(0.0, self.deadline_s - float(now_s))


@dataclass(frozen=True)
class RetryDecision:
    allowed: bool
    reason: str
    now_s: float
    deadline_s: float
    remaining_s: float
    route_available: bool
    route_length_m: Optional[float]
    return_budget_s: Optional[float]
    return_progress_mps: float
    return_margin_s: float


@dataclass
class DeliveryResult:
    status: str
    reason: str
    owner_xy: Optional[tuple]
    final_pose_xy: Optional[tuple]
    owner_distance_m: Optional[float]
    basket_occupied: Optional[bool]
    basket_dropped: bool
    drop_pose_xy: Optional[tuple]
    recovery_attempted: bool
    recovery_succeeded: bool
    navigation_status: Optional[str]
    navigation_reason: Optional[str]
    completion_sent: bool
    remaining_s: float
    deadline_s: Optional[float]
    deadline_source: Optional[str]
    platform_guard_s: float
    return_progress_mps: float
    return_margin_s: float
    owner_tolerance_m: float
    last_retry_decision: Optional[dict]

    def as_dict(self) -> dict:
        return asdict(self)


class Stage2Delivery:
    """Protect the owner-return reserve and complete only at the real owner."""

    def __init__(
        self,
        client,
        deadline: Stage2Deadline,
        owner_xy,
        plan_route: Callable,
        navigate: Callable,
        *,
        log=None,
    ):
        self.client = client
        self.deadline = deadline
        self.owner_xy = (
            None if owner_xy is None
            else (float(owner_xy[0]), float(owner_xy[1]))
        )
        self.plan_route = plan_route
        self.navigate = navigate
        self.config = deadline.config
        self.log = log or logging.getLogger("participant_app")
        self.last_retry_decision: Optional[RetryDecision] = None

    def _emit(self, message: str, *args) -> None:
        if hasattr(self.log, "info"):
            self.log.info(message, *args)
        elif callable(self.log):
            self.log(message % args if args else message)

    def retry_decision(self) -> RetryDecision:
        now_s = float(self.client.now_s())
        deadline_s = float(self.deadline.deadline_s or now_s)
        remaining_s = max(0.0, deadline_s - now_s)
        route = None if self.owner_xy is None else self.plan_route(self.owner_xy)
        available = bool(route is not None and getattr(route, "available", False))
        length_m = float(route.length_m) if available else None
        budget_s = (
            length_m / self.config.return_progress_mps + self.config.return_margin_s
            if length_m is not None else None
        )
        if self.owner_xy is None:
            allowed, reason = False, "missing_owner"
        elif not available:
            allowed, reason = False, "owner_route_unavailable"
        elif remaining_s <= budget_s:
            allowed, reason = False, "return_reserve"
        else:
            allowed, reason = True, "within_budget"
        decision = RetryDecision(
            allowed=allowed,
            reason=reason,
            now_s=now_s,
            deadline_s=deadline_s,
            remaining_s=remaining_s,
            route_available=available,
            route_length_m=length_m,
            return_budget_s=budget_s,
            return_progress_mps=self.config.return_progress_mps,
            return_margin_s=self.config.return_margin_s,
        )
        self.last_retry_decision = decision
        self._emit(
            "[Stage2] retry cutoff now=%.2f deadline=%.2f remaining=%.2f "
            "route=%s length=%s progress=%.2f margin=%.2f budget=%s "
            "allowed=%s reason=%s",
            now_s,
            deadline_s,
            remaining_s,
            available,
            "-" if length_m is None else "%.2f" % length_m,
            self.config.return_progress_mps,
            self.config.return_margin_s,
            "-" if budget_s is None else "%.2f" % budget_s,
            allowed,
            reason,
        )
        return decision

    def retry_allowed(self) -> bool:
        return self.retry_decision().allowed

    def _deadline_value(self) -> float:
        return float(self.deadline.deadline_s or self.client.now_s())

    def return_to_owner(
        self,
        basket_check: Callable[[], Optional[bool]],
        recover_drop: Optional[Callable[[], bool]] = None,
    ) -> DeliveryResult:
        if self.owner_xy is None:
            return self._result("missing_owner", "missing_owner", basket_check)

        last_basket = basket_check()
        dropped = False
        drop_pose = None
        recovery_attempted = False
        recovery_succeeded = False
        request_recovery = False

        def observe(pose):
            nonlocal last_basket, dropped, drop_pose, recovery_attempted, request_recovery
            current = basket_check()
            if current is None:
                return None
            confirmed_drop = last_basket is True and current is False
            last_basket = current
            if not confirmed_drop:
                return None
            dropped = True
            drop_pose = (float(pose[0]), float(pose[1]))
            if recover_drop is not None and not recovery_attempted:
                if self.retry_allowed():
                    recovery_attempted = True
                    request_recovery = True
                    return DROP_STOP_REASON
            return None

        nav = self.navigate(
            self.owner_xy,
            self._deadline_value,
            observe,
        )
        if request_recovery:
            try:
                recovery_succeeded = bool(recover_drop())
            except Exception:
                recovery_succeeded = False
                if hasattr(self.log, "exception"):
                    self.log.exception("[Stage2] basket drop recovery failed")
            request_recovery = False
            last_basket = basket_check()
            nav = self.navigate(
                self.owner_xy,
                self._deadline_value,
                observe,
            )

        pose = self.client.get_world_pose()
        pose_xy = None if pose is None else (float(pose[0]), float(pose[1]))
        owner_distance = (
            None if pose_xy is None else math.hypot(
                pose_xy[0] - self.owner_xy[0], pose_xy[1] - self.owner_xy[1],
            )
        )
        basket = basket_check()
        arrived = (
            owner_distance is not None
            and owner_distance <= self.config.owner_tolerance_m
        )
        before_deadline = self.deadline.remaining(self.client.now_s()) > 0.0
        stage_active = self.deadline.active and bool(
            getattr(self.client, "is_running", False)
        )
        can_complete = arrived and stage_active and before_deadline
        completion_sent = False
        if can_complete:
            self.client.stop()
            self.client.task_complete()
            completion_sent = True
        else:
            self.client.stop()
        status = "delivered" if arrived and basket is True else (
            "return_only" if arrived else "owner_not_reached"
        )
        if can_complete:
            reason = "owner_reached"
        elif arrived and stage_active:
            reason = "owner_reached_after_deadline"
        elif arrived:
            reason = "owner_reached_after_stage_exit"
        else:
            reason = "owner_navigation_%s" % getattr(nav, "reason", "failed")
        return DeliveryResult(
            status=status,
            reason=reason,
            owner_xy=self.owner_xy,
            final_pose_xy=pose_xy,
            owner_distance_m=owner_distance,
            basket_occupied=basket,
            basket_dropped=dropped,
            drop_pose_xy=drop_pose,
            recovery_attempted=recovery_attempted,
            recovery_succeeded=recovery_succeeded,
            navigation_status=(
                getattr(getattr(nav, "status", None), "value", None)
                or str(getattr(nav, "status", "")) or None
            ),
            navigation_reason=getattr(nav, "reason", None),
            completion_sent=completion_sent,
            remaining_s=self.deadline.remaining(self.client.now_s()),
            deadline_s=self.deadline.deadline_s,
            deadline_source=self.deadline.source,
            platform_guard_s=self.config.platform_guard_s,
            return_progress_mps=self.config.return_progress_mps,
            return_margin_s=self.config.return_margin_s,
            owner_tolerance_m=self.config.owner_tolerance_m,
            last_retry_decision=(
                None if self.last_retry_decision is None
                else asdict(self.last_retry_decision)
            ),
        )

    def _result(self, status, reason, basket_check) -> DeliveryResult:
        pose = self.client.get_world_pose()
        pose_xy = None if pose is None else (float(pose[0]), float(pose[1]))
        self.client.stop()
        return DeliveryResult(
            status=status,
            reason=reason,
            owner_xy=self.owner_xy,
            final_pose_xy=pose_xy,
            owner_distance_m=None,
            basket_occupied=basket_check(),
            basket_dropped=False,
            drop_pose_xy=None,
            recovery_attempted=False,
            recovery_succeeded=False,
            navigation_status=None,
            navigation_reason=None,
            completion_sent=False,
            remaining_s=self.deadline.remaining(self.client.now_s()),
            deadline_s=self.deadline.deadline_s,
            deadline_source=self.deadline.source,
            platform_guard_s=self.config.platform_guard_s,
            return_progress_mps=self.config.return_progress_mps,
            return_margin_s=self.config.return_margin_s,
            owner_tolerance_m=self.config.owner_tolerance_m,
            last_retry_decision=(
                None if self.last_retry_decision is None
                else asdict(self.last_retry_decision)
            ),
        )
