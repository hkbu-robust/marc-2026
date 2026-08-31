from types import SimpleNamespace

from marc_agent.navigation_runtime import NavStatus
from marc_agent.stage2_delivery import (
    DeliveryConfig,
    DROP_STOP_REASON,
    Stage2Deadline,
    Stage2Delivery,
)


class FakeClient:
    def __init__(self, now=0.0, pose=(0.0, 0.0, 0.0)):
        self.now = float(now)
        self.pose = pose
        self.is_running = True
        self.events = []

    def now_s(self):
        return self.now

    def get_world_pose(self):
        return self.pose

    def stop(self):
        self.events.append("stop")

    def task_complete(self):
        self.events.append("complete")


def _clock(deadline=500.0, *, progress=0.5, margin=60.0):
    config = DeliveryConfig(
        platform_guard_s=2.0,
        callback_lag_guard_s=5.0,
        return_progress_mps=progress,
        return_margin_s=margin,
        owner_tolerance_m=0.7,
    )
    clock = Stage2Deadline(config)
    clock.deadline_s = float(deadline)
    clock.active = True
    clock.source = "test"
    return clock


def _route(length=10.0, available=True):
    return SimpleNamespace(available=available, length_m=length)


def _nav(status=NavStatus.REACHED, reason="reached"):
    return SimpleNamespace(status=status, reason=reason)


def test_deadline_uses_stage_entry_and_only_shortens():
    clock = Stage2Deadline(DeliveryConfig(platform_guard_s=2.0))
    clock.enter(100.0)
    assert clock.begin(600.0, 104.0) == 698.0
    assert clock.source == "stage2_entry"

    clock.note_remaining(700.0, 200.0)
    assert clock.deadline_s == 698.0
    clock.note_remaining(100.0, 200.0)
    assert clock.deadline_s == 298.0
    assert clock.remaining(250.0) == 48.0


def test_deadline_fallback_reserves_callback_lag():
    clock = Stage2Deadline(DeliveryConfig(callback_lag_guard_s=5.0))
    assert clock.begin(600.0, 104.0) == 699.0
    assert clock.source == "mission_fallback"


def test_retry_cutoff_is_strict_and_route_based():
    client = FakeClient(now=340.0)
    delivery = Stage2Delivery(
        client, _clock(), (10.0, 0.0), lambda _goal: _route(50.0), lambda *_: None,
    )
    decision = delivery.retry_decision()
    assert decision.return_budget_s == 160.0
    assert decision.remaining_s == 160.0
    assert not decision.allowed
    assert decision.reason == "return_reserve"

    client.now = 339.0
    assert delivery.retry_decision().allowed


def test_unavailable_route_denies_retry():
    delivery = Stage2Delivery(
        FakeClient(), _clock(), (10.0, 0.0),
        lambda _goal: _route(available=False), lambda *_: None,
    )
    decision = delivery.retry_decision()
    assert not decision.allowed
    assert decision.reason == "owner_route_unavailable"


def test_owner_completion_uses_live_distance_not_navigation_status():
    client = FakeClient(pose=(0.0, 0.0, 0.0))

    def navigate(_goal, _deadline, _observe):
        client.events.append("navigate")
        client.pose = (9.5, 0.0, 0.0)
        return _nav()

    delivery = Stage2Delivery(
        client, _clock(), (10.0, 0.0), lambda _goal: _route(), navigate,
    )
    result = delivery.return_to_owner(lambda: True)
    assert result.status == "delivered"
    assert result.owner_distance_m == 0.5
    assert result.completion_sent
    assert client.events == ["navigate", "stop", "complete"]


def test_snapped_success_away_from_owner_does_not_complete():
    client = FakeClient(pose=(0.0, 0.0, 0.0))
    delivery = Stage2Delivery(
        client, _clock(), (10.0, 0.0), lambda _goal: _route(),
        lambda *_args: _nav(),
    )
    result = delivery.return_to_owner(lambda: True)
    assert result.status == "owner_not_reached"
    assert not result.completion_sent
    assert client.events == ["stop"]


def test_owner_arrival_after_guarded_deadline_does_not_complete():
    client = FakeClient(now=500.0, pose=(10.0, 0.0, 0.0))
    delivery = Stage2Delivery(
        client, _clock(deadline=500.0), (10.0, 0.0), lambda _goal: _route(),
        lambda *_args: _nav(),
    )
    result = delivery.return_to_owner(lambda: True)
    assert result.reason == "owner_reached_after_deadline"
    assert not result.completion_sent
    assert client.events == ["stop"]


def test_owner_arrival_after_stage_exit_does_not_complete():
    client = FakeClient(now=100.0, pose=(10.0, 0.0, 0.0))
    clock = _clock()
    clock.leave()
    delivery = Stage2Delivery(
        client, clock, (10.0, 0.0), lambda _goal: _route(),
        lambda *_args: _nav(),
    )
    result = delivery.return_to_owner(lambda: True)
    assert result.reason == "owner_reached_after_stage_exit"
    assert not result.completion_sent
    assert client.events == ["stop"]


def test_missing_owner_never_navigates_or_completes():
    client = FakeClient()
    navigations = []
    delivery = Stage2Delivery(
        client, _clock(), None, lambda _goal: _route(),
        lambda *_args: navigations.append(True),
    )
    result = delivery.return_to_owner(lambda: False)
    assert result.status == "missing_owner"
    assert not navigations
    assert "complete" not in client.events


def test_confirmed_drop_gets_one_reserve_safe_recovery_then_resumes():
    client = FakeClient(now=100.0)
    readings = iter((True, None, False, True, True, True))
    navigation_calls = []
    recovery_calls = []

    def basket():
        return next(readings)

    def navigate(_goal, _deadline, observe):
        navigation_calls.append("navigate")
        if len(navigation_calls) == 1:
            assert observe(client.pose) is None
            assert observe(client.pose) == DROP_STOP_REASON
            return _nav(NavStatus.ABORTED, DROP_STOP_REASON)
        assert observe(client.pose) is None
        client.pose = (10.0, 0.0, 0.0)
        return _nav()

    def recover():
        recovery_calls.append("recover")
        return True

    delivery = Stage2Delivery(
        client, _clock(), (10.0, 0.0), lambda _goal: _route(), navigate,
    )
    result = delivery.return_to_owner(basket, recover)
    assert navigation_calls == ["navigate", "navigate"]
    assert recovery_calls == ["recover"]
    assert result.basket_dropped
    assert result.recovery_attempted
    assert result.recovery_succeeded
    assert result.status == "delivered"


def test_drop_without_return_budget_does_not_stop_for_recovery():
    client = FakeClient(now=340.0)
    readings = iter((True, False, False))
    recovery_calls = []

    def navigate(_goal, _deadline, observe):
        assert observe(client.pose) is None
        client.pose = (10.0, 0.0, 0.0)
        return _nav()

    delivery = Stage2Delivery(
        client, _clock(), (10.0, 0.0), lambda _goal: _route(50.0), navigate,
    )
    result = delivery.return_to_owner(
        lambda: next(readings), lambda: recovery_calls.append(True) or True,
    )
    assert not recovery_calls
    assert result.basket_dropped
    assert not result.recovery_attempted
    assert result.status == "return_only"


def test_owner_navigation_failure_does_not_complete():
    client = FakeClient()
    delivery = Stage2Delivery(
        client, _clock(), (10.0, 0.0), lambda _goal: _route(),
        lambda *_args: _nav(NavStatus.NO_PATH, "no_path"),
    )
    result = delivery.return_to_owner(lambda: False)
    assert result.status == "owner_not_reached"
    assert result.navigation_status == "no_path"
    assert "complete" not in client.events
