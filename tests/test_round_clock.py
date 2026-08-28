"""Round clock snapshot and Qwen time-budget skip/cap."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from marc_agent.perception.qwen_remote import QwenRemoteTimeout, RemoteQwenSarVerifier
from marc_agent.perception.qwen_runtime import QwenRemoteConfig
from marc_agent.perception.sar_selection import CandidateBoard
from marc_agent.perception.stage1_pipeline import run_stage1_candidate_pipeline
from marc_agent.round_clock import RoundClock, min_remaining_s, qwen_budget
from marc_agent.types import (
    CctvDetectionRound,
    ObjectCandidate,
    PersonCandidate,
)


def _clock(limit: float, t0: float = 0.0) -> RoundClock:
    clock = RoundClock()
    clock.begin(limit, t0)
    return clock


class _Now:
    def __init__(self, value: float):
        self.value = value

    def now_s(self):
        return self.value


def _sar_round() -> CctvDetectionRound:
    return CctvDetectionRound(
        objects=[
            ObjectCandidate(
                "rig_a:trash_can:0",
                "trash_can",
                (300.0, 190.0, 430.0, 430.0),
                0.9,
                "rig_a",
                (640, 480),
            )
        ],
        people=[
            PersonCandidate(
                "rig_a:person_walking:0",
                (30.0, 80.0, 130.0, 400.0),
                "person_walking",
                0.8,
                "rig_a",
                (640, 480),
            ),
            PersonCandidate(
                "rig_a:person_bending:0",
                (315.0, 100.0, 405.0, 390.0),
                "person_bending",
                0.8,
                "rig_a",
                (640, 480),
            ),
        ],
        camera_ids=["rig_a"],
    )


def _object_round() -> CctvDetectionRound:
    return CctvDetectionRound(
        objects=[
            ObjectCandidate(
                "rig_a:mug:0",
                "mug",
                (280.0, 200.0, 340.0, 255.0),
                0.7,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_a:picnic_table:0",
                "picnic_table",
                (200.0, 250.0, 450.0, 400.0),
                0.8,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_a:mug:1",
                "mug",
                (20.0, 20.0, 50.0, 40.0),
                0.4,
                "rig_a",
                (640, 480),
            ),
            ObjectCandidate(
                "rig_a:picnic_table:1",
                "picnic_table",
                (10.0, 45.0, 80.0, 90.0),
                0.8,
                "rig_a",
                (640, 480),
            ),
        ],
        camera_ids=["rig_a"],
    )


def _remote_config() -> QwenRemoteConfig:
    return QwenRemoteConfig(
        base_url="http://workstation:8000/v1",
        api_key="test-secret",
        model_name="qwen3-vl-8b",
        timeout_s=5.0,
        max_new_tokens=8,
        jpeg_quality=80,
    )


class RoundClockTests(unittest.TestCase):
    def test_remaining_decreases_with_now(self):
        clock = _clock(30.0, 100.0)
        self.assertAlmostEqual(30.0, clock.remaining(100.0))
        self.assertAlmostEqual(27.0, clock.remaining(103.0))
        self.assertEqual(0.0, clock.remaining(140.0))

    def test_begin_drops_stale_platform_remaining(self):
        clock = RoundClock()
        clock.note(2.0, 50.0)
        clock.begin(30.0, 100.0)
        self.assertAlmostEqual(30.0, clock.remaining(100.0))

    def test_later_platform_update_can_only_shrink(self):
        clock = _clock(30.0, 100.0)
        clock.note(10.0, 101.0)
        self.assertAlmostEqual(8.0, clock.remaining(103.0))
        clock.note(40.0, 103.0)
        self.assertAlmostEqual(27.0, clock.remaining(103.0))

    def test_unknown_clock_does_not_skip(self):
        budget = qwen_budget(None, 0.0, default_timeout=90.0)
        self.assertFalse(budget.skip)
        self.assertEqual(90.0, budget.timeout_s)


class QwenTimeBudgetTests(unittest.TestCase):
    def test_remaining_three_seconds_skips_sar_verifier(self):
        calls = []

        def detector(_client, *, stub_path=None):
            return _sar_round()

        def verifier(_board):
            calls.append(1)
            return "rig_a:person_bending:0|emergency"

        clock = _clock(30.0, 0.0)
        client = _Now(27.0)
        with patch.dict(
            os.environ,
            {"MARC_QWEN_MIN_REMAINING_S": "4", "MARC_QWEN_SUBMIT_MARGIN_S": "2"},
            clear=False,
        ):
            result = run_stage1_candidate_pipeline(
                client,
                "Find the person bent over near the trash can who needs urgent help.",
                detector=detector,
                verifier=verifier,
                clock=clock,
            )
        self.assertEqual([], calls)
        self.assertEqual("deterministic_fallback", result.sar_selection.method)
        self.assertEqual("rig_a:person_bending:0", result.selected_candidate_id)

    def test_remaining_thirty_seconds_calls_verifier(self):
        calls = []

        def detector(_client, *, stub_path=None):
            return _sar_round()

        def verifier(_board):
            calls.append(1)
            return "rig_a:person_bending:0|emergency"

        clock = _clock(30.0, 0.0)
        result = run_stage1_candidate_pipeline(
            _Now(0.0),
            "Find the person bent over near the trash can who needs urgent help.",
            detector=detector,
            verifier=verifier,
            clock=clock,
        )
        self.assertEqual([1], calls)
        self.assertEqual("qwen", result.sar_selection.method)

    def test_remaining_three_seconds_skips_object_verifier(self):
        calls = []

        def detector(_client, *, stub_path=None):
            return _object_round()

        def verifier(_board):
            calls.append(1)
            return "rig_a:mug:1|rig_a:picnic_table:1"

        clock = _clock(30.0, 0.0)
        with patch.dict(os.environ, {"MARC_QWEN_MIN_REMAINING_S": "4"}, clear=False):
            result = run_stage1_candidate_pipeline(
                _Now(27.0),
                "Find the mug on the picnic table.",
                detector=detector,
                verifier=verifier,
                clock=clock,
            )
        self.assertEqual([], calls)
        self.assertEqual("object_relation", result.object_selection.method)
        self.assertEqual("rig_a:mug:0", result.selected_candidate_id)

    def test_zero_floor_never_skips_for_time(self):
        calls = []

        def detector(_client, *, stub_path=None):
            return _sar_round()

        def verifier(_board):
            calls.append(1)
            return "rig_a:person_bending:0|emergency"

        clock = _clock(30.0, 0.0)
        with patch.dict(os.environ, {"MARC_QWEN_MIN_REMAINING_S": "0"}, clear=False):
            self.assertEqual(0.0, min_remaining_s())
            result = run_stage1_candidate_pipeline(
                _Now(27.0),
                "Find the person bent over near the trash can who needs urgent help.",
                detector=detector,
                verifier=verifier,
                clock=clock,
            )
        self.assertEqual([1], calls)
        self.assertEqual("qwen", result.sar_selection.method)

    def test_budget_caps_remote_timeout_and_skips_retry(self):
        calls = []

        def post(_url, _body, _headers, timeout):
            calls.append(timeout)
            raise QwenRemoteTimeout("vLLM did not respond within 2.0s")

        verifier = RemoteQwenSarVerifier(_remote_config(), post=post)
        verifier.set_budget_timeout(2.0)
        board = CandidateBoard(images=(), prompt="x", evidence=())
        with self.assertLogs("marc_agent.qwen_remote", level="INFO"):
            with self.assertRaises(QwenRemoteTimeout):
                verifier(board)
        self.assertEqual([2.0], calls)


if __name__ == "__main__":
    unittest.main()
