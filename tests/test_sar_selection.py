"""Tests for query-conditioned SAR evidence and bounded selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from marc_agent.command_parser import parse_command
from marc_agent.perception.sar_selection import (
    SelectionStatus,
    build_candidate_board,
    parse_verifier_output,
    select_sar_candidate,
)
from marc_agent.perception.stage1_pipeline import (
    append_stage1_record,
    get_default_verifier,
    log_qwen_startup_status,
    reset_cached_verifier,
    run_stage1_candidate_pipeline,
)
from marc_agent.types import (
    CctvDetectionRound,
    CctvFrame,
    ObjectCandidate,
    PersonCandidate,
)


def _person(candidate_id: str, pose: str, box, confidence: float = 0.8):
    return PersonCandidate(candidate_id, box, pose, confidence, "rig_a", (640, 480))


def _round() -> CctvDetectionRound:
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
            _person("rig_a:person:0", "person_walking", (30.0, 80.0, 130.0, 400.0)),
            _person("rig_a:person:1", "person_bending", (315.0, 100.0, 405.0, 390.0)),
        ],
        frames={
            "rig_a": CctvFrame("rig_a", 640, 480, np.zeros((480, 640, 3), dtype=np.uint8))
        },
        camera_ids=["rig_a"],
    )


class SarSelectionTests(unittest.TestCase):
    def setUp(self):
        reset_cached_verifier()
        self.text = "Find the person bent over near the trash can who needs urgent help."
        self.command = parse_command(self.text)

    def tearDown(self):
        reset_cached_verifier()

    def test_board_keeps_every_person_and_requested_anchor(self):
        board = build_candidate_board(self.text, self.command, _round())
        self.assertEqual(("rig_a:person:1", "rig_a:person:0"), board.candidate_ids)
        self.assertEqual(3, len(board.images))
        self.assertIn("rig_a:trash_can:0", board.prompt)
        self.assertIn(self.text, board.prompt)

    def test_board_limits_crops_but_keeps_every_candidate(self):
        board = build_candidate_board(
            self.text,
            self.command,
            _round(),
            max_crops=1,
        )
        self.assertEqual(("rig_a:person:1", "rig_a:person:0"), board.candidate_ids)
        self.assertEqual(2, len(board.images))  # one annotated camera + one crop
        self.assertIn("rig_a:person:0", board.prompt)

    def test_valid_model_output_selects_only_supplied_id(self):
        with self.assertLogs("marc_agent.sar_selection", level="INFO") as captured:
            result = select_sar_candidate(
                self.text,
                self.command,
                _round(),
                verifier=lambda board: "rig_a:person:1|emergency",
            )
        self.assertEqual(SelectionStatus.SELECTED, result.status)
        self.assertEqual("rig_a:person:1", result.selected.candidate_id)
        self.assertEqual("qwen", result.method)
        self.assertEqual("qwen", result.verifier_metrics.backend)
        self.assertEqual(3, result.verifier_metrics.image_count)
        self.assertIn("output=rig_a:person:1|emergency", "\n".join(captured.output))

    def test_invalid_model_output_uses_deterministic_fallback(self):
        with self.assertLogs("marc_agent.sar_selection", level="INFO") as captured:
            result = select_sar_candidate(
                self.text,
                self.command,
                _round(),
                verifier=lambda board: "I choose rig_a:person:0",
            )
        self.assertEqual("rig_a:person:1", result.selected.candidate_id)
        self.assertEqual("deterministic_fallback", result.method)
        self.assertEqual("verifier_failed:ValueError", result.reason)
        self.assertIn("output=I choose rig_a:person:0", "\n".join(captured.output))

    def test_remote_adapter_method_is_recorded(self):
        class RemoteStub:
            method = "qwen_remote"

            def __call__(self, _board):
                return "rig_a:person:1|emergency"

        result = select_sar_candidate(
            self.text,
            self.command,
            _round(),
            verifier=RemoteStub(),
        )
        self.assertEqual("qwen_remote", result.method)
        self.assertEqual("qwen_remote", result.verifier_metrics.backend)

    def test_output_parser_rejects_unknown_candidates_and_extra_text(self):
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            parse_verifier_output("rig_b:person:0|emergency", ("rig_a:person:0",))
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_verifier_output("rig_a:person:0|emergency because", ("rig_a:person:0",))

    def test_no_people_is_explicit_no_match(self):
        result = select_sar_candidate(
            self.text,
            self.command,
            CctvDetectionRound(),
        )
        self.assertEqual(SelectionStatus.NO_MATCH, result.status)
        self.assertIsNone(result.selected)

    def test_fallback_selects_across_cameras_instead_of_first_person(self):
        round_ = _round()
        round_.people[0] = PersonCandidate(
            "rig_b:person:0",
            (30.0, 80.0, 130.0, 400.0),
            "person_walking",
            0.99,
            "rig_b",
            (640, 480),
        )
        round_.frames["rig_b"] = CctvFrame(
            "rig_b",
            640,
            480,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        round_.camera_ids.append("rig_b")

        result = select_sar_candidate(self.text, self.command, round_)

        self.assertEqual("rig_a:person:1", result.selected.candidate_id)
        self.assertEqual("deterministic_fallback", result.method)

    def test_sitting_on_trash_can_beats_closer_standing_person(self):
        text = "Find the person sitting on top of the trash can in the parking lot."
        command = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate(
                    "rig_a:trash_can:0",
                    "trash_can",
                    (300.0, 200.0, 430.0, 430.0),
                    0.9,
                    "rig_a",
                    (640, 480),
                ),
                ObjectCandidate(
                    "rig_b:trash_can:0",
                    "trash_can",
                    (200.0, 200.0, 320.0, 400.0),
                    0.95,
                    "rig_b",
                    (640, 480),
                ),
            ],
            people=[
                PersonCandidate(
                    "rig_a:person:0",
                    (310.0, 80.0, 410.0, 210.0),
                    "person_sitting",
                    0.55,
                    "rig_a",
                    (640, 480),
                ),
                PersonCandidate(
                    "rig_b:person:0",
                    (205.0, 210.0, 310.0, 395.0),
                    "person_walking",
                    0.99,
                    "rig_b",
                    (640, 480),
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )
        result = select_sar_candidate(text, command, round_)
        self.assertEqual("rig_a:person:0", result.selected.candidate_id)
        self.assertEqual("abnormal", result.requested_situation)
        self.assertEqual("deterministic_fallback", result.method)

    def test_round_pipeline_returns_selection_without_coordinates(self):
        def detector(client, *, stub_path=None):
            return _round()

        result = run_stage1_candidate_pipeline(
            object(),
            self.text,
            detector=detector,
            verifier=lambda board: "rig_a:person:1|emergency",
        )
        self.assertEqual("rig_a:person:1", result.selected_candidate_id)
        self.assertFalse(hasattr(result, "target_coord"))
        record = result.record(expected_candidate_id="rig_a:person:1")
        self.assertEqual(1.0, record["candidate_recall"])
        self.assertEqual("qwen", record["verifier_metrics"]["backend"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rounds.jsonl"
            append_stage1_record(result, path, expected_candidate_id="rig_a:person:1")
            self.assertIn('"candidate_recall": 1.0', path.read_text(encoding="utf-8"))

    def test_collection_failure_is_explicit(self):
        def broken_detector(client, *, stub_path=None):
            raise RuntimeError("GPU unavailable")

        result = run_stage1_candidate_pipeline(
            object(),
            self.text,
            detector=broken_detector,
        )
        self.assertEqual("candidate_collection:RuntimeError", result.failure_reason)
        self.assertEqual(
            ("candidate_collection:RuntimeError",),
            result.detections.metrics.failures,
        )

    def test_qwen_mode_without_weights_falls_back_to_same_candidates(self):
        def detector(client, *, stub_path=None):
            return _round()

        with patch.dict("os.environ", {"MARC_SAR_DETECTOR": "qwen"}):
            result = run_stage1_candidate_pipeline(
                object(),
                self.text,
                detector=detector,
            )
        self.assertEqual("rig_a:person:1", result.selected_candidate_id)
        self.assertEqual("deterministic_fallback", result.sar_selection.method)
        self.assertEqual(
            "verifier_failed:QwenModelUnavailable",
            result.sar_selection.reason,
        )

    def test_remote_mode_without_connection_config_falls_back(self):
        def detector(client, *, stub_path=None):
            return _round()

        env = {
            "MARC_SAR_DETECTOR": "qwen",
            "MARC_QWEN_BACKEND": "remote",
            "MARC_QWEN_BASE_URL": "",
            "MARC_QWEN_API_KEY": "",
        }
        with patch.dict("os.environ", env):
            result = run_stage1_candidate_pipeline(
                object(),
                self.text,
                detector=detector,
            )
        self.assertEqual("rig_a:person:1", result.selected_candidate_id)
        self.assertEqual("deterministic_fallback", result.sar_selection.method)
        self.assertEqual(
            "verifier_failed:QwenConfigurationError",
            result.sar_selection.reason,
        )

    def test_object_request_selects_without_sar(self):
        def detector(client, *, stub_path=None):
            return _round()

        with self.assertLogs("marc_agent.stage1_pipeline", level="INFO") as captured:
            result = run_stage1_candidate_pipeline(
                object(),
                "Find the mug beside the bench.",
                detector=detector,
            )
        self.assertEqual("mug", result.parsed.target_type)
        self.assertIsNone(result.sar_selection)
        self.assertIsNone(result.object_selection)
        self.assertEqual("object_selection:no_match", result.failure_reason)
        output = "\n".join(captured.output)
        self.assertIn("object request", output)
        self.assertNotIn("reason=non_sar_request", output)

    def test_narrative_stage1_command_uses_find_intent(self):
        def detector(client, *, stub_path=None, discovery_timeout_s=None):
            return CctvDetectionRound()

        result = run_stage1_candidate_pipeline(
            object(),
            "I was taking notes at an outdoor table and left my pencil case behind.",
            detector=detector,
        )
        self.assertIsNotNone(result.parsed)
        self.assertEqual("pencilcase", result.parsed.target_type)
        self.assertEqual("picnic_table", result.parsed.landmark)
        self.assertNotIn("unknown_intent", result.failure_reason or "")

    def test_remote_startup_health_is_visible_and_does_not_log_key(self):
        env = {
            "MARC_SAR_DETECTOR": "qwen",
            "MARC_QWEN_BACKEND": "remote",
            "MARC_QWEN_BASE_URL": "http://workstation:8000/v1",
            "MARC_QWEN_API_KEY": "private-test-key",
        }
        with (
            patch.dict("os.environ", env),
            patch(
                "marc_agent.perception.stage1_pipeline.check_remote_health",
                return_value=("qwen3-vl-8b",),
            ),
            self.assertLogs("marc_agent.stage1_pipeline", level="INFO") as captured,
        ):
            log_qwen_startup_status()
        output = "\n".join(captured.output)
        self.assertIn("health=ok", output)
        self.assertIn("qwen3-vl-8b", output)
        self.assertNotIn("private-test-key", output)

    def test_remote_startup_failure_is_nonfatal(self):
        env = {
            "MARC_SAR_DETECTOR": "qwen",
            "MARC_QWEN_BACKEND": "remote",
            "MARC_QWEN_BASE_URL": "http://workstation:8000/v1",
            "MARC_QWEN_API_KEY": "private-test-key",
        }
        with (
            patch.dict("os.environ", env),
            patch(
                "marc_agent.perception.stage1_pipeline.check_remote_health",
                side_effect=ConnectionError("connection refused"),
            ),
            self.assertLogs("marc_agent.stage1_pipeline", level="INFO") as captured,
        ):
            log_qwen_startup_status()
        self.assertIn("health=failed", "\n".join(captured.output))

    def test_default_verifier_is_cached(self):
        env = {
            "MARC_SAR_DETECTOR": "qwen",
            "MARC_QWEN_BACKEND": "remote",
            "MARC_QWEN_BASE_URL": "http://workstation:8000/v1",
            "MARC_QWEN_API_KEY": "private-test-key",
        }
        with patch.dict("os.environ", env, clear=False):
            first = get_default_verifier()
            second = get_default_verifier()
        self.assertIs(first, second)

    def test_local_startup_calls_warmup(self):
        env = {
            "MARC_SAR_DETECTOR": "qwen",
            "MARC_QWEN_BACKEND": "local",
        }
        with (
            patch.dict("os.environ", env, clear=False),
            patch(
                "marc_agent.perception.sar_selection.QwenSarVerifier.warmup",
            ) as warmup,
            self.assertLogs("marc_agent.stage1_pipeline", level="INFO") as captured,
        ):
            log_qwen_startup_status()
        warmup.assert_called_once()
        self.assertIn("backend=local", "\n".join(captured.output))

    def test_parse_failure_does_not_fetch_cctv(self):
        called = False

        def detector(client, *, stub_path=None):
            nonlocal called
            called = True
            return _round()

        result = run_stage1_candidate_pipeline(
            object(),
            "do something",
            detector=detector,
        )
        self.assertFalse(called)
        self.assertTrue(result.failure_reason.startswith("command_parse:"))


if __name__ == "__main__":
    unittest.main()
