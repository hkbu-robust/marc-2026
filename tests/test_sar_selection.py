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
    shortlist_for_verifier,
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


def _person(candidate_id: str, pose: str, box, confidence: float = 0.8, camera_id: str = "rig_a"):
    return PersonCandidate(candidate_id, box, pose, confidence, camera_id, (640, 480))


def _object(
    candidate_id: str,
    class_name: str,
    box,
    camera_id: str = "rig_a",
    confidence: float = 0.9,
):
    return ObjectCandidate(candidate_id, class_name, box, confidence, camera_id, (640, 480))


def _frame(camera_id: str = "rig_a") -> CctvFrame:
    return CctvFrame(camera_id, 640, 480, np.zeros((480, 640, 3), dtype=np.uint8))


def _round() -> CctvDetectionRound:
    return CctvDetectionRound(
        objects=[
            _object(
                "rig_a:trash_can:0",
                "trash_can",
                (300.0, 190.0, 430.0, 430.0),
            )
        ],
        people=[
            _person("rig_a:person_walking:0", "person_walking", (30.0, 80.0, 130.0, 400.0)),
            _person("rig_a:person_bending:0", "person_bending", (315.0, 100.0, 405.0, 390.0)),
        ],
        frames={"rig_a": _frame("rig_a")},
        camera_ids=["rig_a"],
    )


def _two_bending_round() -> CctvDetectionRound:
    """Two bending people on different cameras, both next to a trash can, plus a walker."""
    return CctvDetectionRound(
        objects=[
            _object("rig_a:trash_can:0", "trash_can", (300.0, 190.0, 430.0, 430.0), "rig_a"),
            _object("rig_b:trash_can:0", "trash_can", (300.0, 190.0, 430.0, 430.0), "rig_b"),
        ],
        people=[
            _person(
                "rig_b:person_bending:0",
                "person_bending",
                (315.0, 100.0, 405.0, 390.0),
                0.99,
                "rig_b",
            ),
            _person(
                "rig_a:person_bending:0",
                "person_bending",
                (315.0, 100.0, 405.0, 390.0),
                0.4,
                "rig_a",
            ),
            _person(
                "rig_a:person_walking:0",
                "person_walking",
                (30.0, 80.0, 130.0, 400.0),
                0.8,
                "rig_a",
            ),
        ],
        frames={"rig_a": _frame("rig_a"), "rig_b": _frame("rig_b")},
        camera_ids=["rig_a", "rig_b"],
    )


def _same_camera_two_bending() -> CctvDetectionRound:
    return CctvDetectionRound(
        objects=[
            _object("rig_a:trash_can:0", "trash_can", (300.0, 190.0, 430.0, 430.0)),
        ],
        people=[
            _person("rig_a:person_walking:0", "person_walking", (30.0, 80.0, 130.0, 400.0)),
            _person(
                "rig_a:person_bending:0",
                "person_bending",
                (315.0, 100.0, 405.0, 390.0),
                0.9,
            ),
            _person(
                "rig_a:person_bending:1",
                "person_bending",
                (50.0, 100.0, 140.0, 390.0),
                0.5,
            ),
        ],
        frames={"rig_a": _frame("rig_a")},
        camera_ids=["rig_a"],
    )


def _downward_projector(*camera_ids: str):
    from marc_agent.perception.cctv_project import (
        CameraGeometry,
        CctvProjector,
        static_lookup,
    )

    geometry = CameraGeometry(
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        rotation=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation=(0.0, 0.0, -10.0),
        ground_height=0.0,
    )
    ids = camera_ids or ("rig_a",)
    return CctvProjector(static_lookup({camera_id: geometry for camera_id in ids}))


class SarSelectionTests(unittest.TestCase):
    def setUp(self):
        reset_cached_verifier()
        self.text = "Find the person bent over near the trash can who needs urgent help."
        self.command = parse_command(self.text)

    def tearDown(self):
        reset_cached_verifier()

    def test_board_keeps_every_person_and_requested_anchor(self):
        board = build_candidate_board(self.text, self.command, _round())
        self.assertEqual(("rig_a:person_bending:0",), board.candidate_ids)
        self.assertEqual(2, len(board.images))
        self.assertIn("rig_a:trash_can:0", board.prompt)
        self.assertIn(self.text, board.prompt)
        self.assertNotIn("rig_a:person_walking:0", board.prompt)

    def test_crop_limit_never_removes_a_candidate_from_the_board(self):
        board = build_candidate_board(
            self.text,
            self.command,
            _same_camera_two_bending(),
            max_crops=1,
        )
        self.assertEqual(
            ("rig_a:person_bending:0", "rig_a:person_bending:1"),
            board.candidate_ids,
        )
        self.assertEqual(2, len(board.images))  # one annotated camera + one crop
        self.assertNotIn("rig_a:person_walking:0", board.prompt)
        self.assertIn("detail_crop=no", board.prompt)
        self.assertIn("middle token of each CANDIDATE_ID is the detector pose", board.prompt)
        self.assertIn("Do not prefer P01", board.prompt)

    def test_verifier_can_select_a_candidate_outside_the_crop_budget(self):
        with patch(
            "marc_agent.perception.sar_selection.configured_max_crops",
            return_value=1,
        ):
            result = select_sar_candidate(
                self.text,
                self.command,
                _same_camera_two_bending(),
                verifier=lambda board: "rig_a:person_bending:1|emergency",
            )
        self.assertEqual("rig_a:person_bending:1", result.selected.candidate_id)

    def test_valid_model_output_selects_only_supplied_id(self):
        with self.assertLogs("marc_agent.sar_selection", level="INFO") as captured:
            result = select_sar_candidate(
                self.text,
                self.command,
                _two_bending_round(),
                verifier=lambda board: "P01",
            )
        self.assertEqual(SelectionStatus.SELECTED, result.status)
        self.assertEqual("rig_a:person_bending:0", result.selected.candidate_id)
        self.assertIsNone(result.evidence_situation)
        self.assertEqual("qwen", result.method)
        self.assertEqual("qwen", result.verifier_metrics.backend)
        self.assertEqual(4, result.verifier_metrics.image_count)
        self.assertIn("output=P01", "\n".join(captured.output))

    def test_invalid_model_output_uses_deterministic_fallback(self):
        with self.assertLogs("marc_agent.sar_selection", level="INFO") as captured:
            result = select_sar_candidate(
                self.text,
                self.command,
                _two_bending_round(),
                verifier=lambda board: "I choose nobody",
            )
        self.assertEqual("person_bending", result.selected.pose_class)
        self.assertNotEqual("rig_a:person_walking:0", result.selected.candidate_id)
        self.assertEqual("deterministic_fallback", result.method)
        self.assertEqual("verifier_failed:ReplyParseError", result.reason)
        self.assertIn("output=I choose nobody", "\n".join(captured.output))

    def test_remote_adapter_method_is_recorded(self):
        class RemoteStub:
            method = "qwen_remote"

            def __call__(self, _board):
                return "rig_a:person_bending:0|emergency"

        result = select_sar_candidate(
            self.text,
            self.command,
            _two_bending_round(),
            verifier=RemoteStub(),
        )
        self.assertEqual("qwen_remote", result.method)
        self.assertEqual("qwen_remote", result.verifier_metrics.backend)

    def test_output_parser_rejects_unknown_candidates(self):
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            parse_verifier_output("rig_b:person_walking:0|emergency", ("rig_a:person_walking:0",))

    def test_output_parser_accepts_prose_and_thinking_with_valid_id(self):
        ids = ("rig_a:person_bending:0", "rig_a:person_walking:0")
        candidate_id, situation = parse_verifier_output(
            "rig_a:person_walking:0|emergency because",
            ids,
        )
        self.assertEqual("rig_a:person_walking:0", candidate_id)
        self.assertEqual("emergency", situation)
        candidate_id, situation = parse_verifier_output(
            "<think>\nP01 is closer to the bench\n</think>\nP02",
            ids,
        )
        self.assertEqual("rig_a:person_walking:0", candidate_id)
        self.assertIsNone(situation)

    def test_output_parser_accepts_candidate_id_without_situation(self):
        candidate_id, situation = parse_verifier_output(
            "rig_a:person_walking:0",
            ("rig_a:person_walking:0",),
        )
        self.assertEqual("rig_a:person_walking:0", candidate_id)
        self.assertIsNone(situation)

    def test_output_parser_accepts_numbered_p01(self):
        candidate_id, situation = parse_verifier_output(
            "P01",
            ("rig_a:person_bending:0", "rig_a:person_walking:0"),
        )
        self.assertEqual("rig_a:person_bending:0", candidate_id)
        self.assertIsNone(situation)
        candidate_id, situation = parse_verifier_output(
            "p02|emergency",
            ("rig_a:person_bending:0", "rig_a:person_walking:0"),
        )
        self.assertEqual("rig_a:person_walking:0", candidate_id)
        self.assertEqual("emergency", situation)

    def test_board_prefixes_numbered_aliases(self):
        board = build_candidate_board(self.text, self.command, _round())
        self.assertIn("P01 | rig_a:person_bending:0", board.prompt)
        self.assertNotIn("P02 |", board.prompt)

    def test_output_parser_accepts_split_camera_pipe(self):
        board = (
            "rig_7_a:person_sitting:3",
            "rig_2_b:person_sitting:2",
        )
        candidate_id, situation = parse_verifier_output(
            "rig_7_a|person_sitting:3|abnormal",
            board,
        )
        self.assertEqual("rig_7_a:person_sitting:3", candidate_id)
        self.assertEqual("abnormal", situation)
        legacy, _ = parse_verifier_output("rig_7_a|person:3|abnormal", board)
        self.assertEqual("rig_7_a:person_sitting:3", legacy)

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
            "rig_b:person_walking:0",
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

        self.assertEqual("rig_a:person_bending:0", result.selected.candidate_id)
        self.assertEqual("deterministic_fallback", result.method)
        self.assertEqual("unique_shortlist", result.reason)

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
                    "rig_a:person_sitting:0",
                    (310.0, 80.0, 410.0, 210.0),
                    "person_sitting",
                    0.55,
                    "rig_a",
                    (640, 480),
                ),
                PersonCandidate(
                    "rig_b:person_walking:0",
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
        self.assertEqual("rig_a:person_sitting:0", result.selected.candidate_id)
        self.assertEqual("abnormal", result.requested_situation)
        self.assertEqual("deterministic_fallback", result.method)
        self.assertEqual("unique_shortlist", result.reason)

    def test_round_pipeline_returns_selection_without_coordinates(self):
        def detector(client, *, stub_path=None):
            return _two_bending_round()

        result = run_stage1_candidate_pipeline(
            object(),
            self.text,
            detector=detector,
            verifier=lambda board: "rig_a:person_bending:0|emergency",
        )
        self.assertEqual("rig_a:person_bending:0", result.selected_candidate_id)
        self.assertFalse(hasattr(result, "target_coord"))
        record = result.record(expected_candidate_id="rig_a:person_bending:0")
        self.assertEqual(1.0, record["candidate_recall"])
        self.assertEqual("qwen", record["verifier_metrics"]["backend"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rounds.jsonl"
            append_stage1_record(result, path, expected_candidate_id="rig_a:person_bending:0")
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
            return _two_bending_round()

        with patch.dict("os.environ", {"MARC_SAR_DETECTOR": "qwen"}):
            result = run_stage1_candidate_pipeline(
                object(),
                self.text,
                detector=detector,
            )
        self.assertEqual("rig_b:person_bending:0", result.selected_candidate_id)
        self.assertEqual("deterministic_fallback", result.sar_selection.method)
        self.assertEqual(
            "verifier_failed:QwenModelUnavailable",
            result.sar_selection.reason,
        )

    def test_remote_mode_without_connection_config_falls_back(self):
        def detector(client, *, stub_path=None):
            return _two_bending_round()

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
        self.assertEqual("rig_b:person_bending:0", result.selected_candidate_id)
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

    def test_no_landmark_does_not_attach_nearest_object(self):
        text = (
            "Find the person who is completely face-down and immobile in the parking lot."
        )
        command = parse_command(text)
        round_ = _round()
        board = build_candidate_board(text, command, round_)
        self.assertTrue(all(item.anchor is None for item in board.evidence))
        self.assertIn("anchor=none", board.prompt)

    def test_camera_evidence_reranks_but_never_drops_people(self):
        from marc_agent.perception.sar_selection import candidate_evidence

        text = "Find the person sitting on top of the trash can in the parking lot."
        command = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate(
                    "rig_park:police_car:0",
                    "police_car",
                    (10.0, 10.0, 80.0, 80.0),
                    0.9,
                    "rig_park",
                    (640, 480),
                ),
                ObjectCandidate(
                    "rig_park:trash_can:0",
                    "trash_can",
                    (200.0, 200.0, 280.0, 400.0),
                    0.9,
                    "rig_park",
                    (640, 480),
                ),
                ObjectCandidate(
                    "rig_other:trash_can:0",
                    "trash_can",
                    (200.0, 200.0, 280.0, 400.0),
                    0.9,
                    "rig_other",
                    (640, 480),
                ),
            ],
            people=[
                PersonCandidate(
                    "rig_park:person_sitting:0",
                    (210.0, 120.0, 270.0, 220.0),
                    "person_sitting",
                    0.8,
                    "rig_park",
                    (640, 480),
                ),
                PersonCandidate(
                    "rig_other:person_sitting:0",
                    (210.0, 120.0, 270.0, 220.0),
                    "person_sitting",
                    0.99,
                    "rig_other",
                    (640, 480),
                ),
            ],
            camera_ids=["rig_park", "rig_other"],
        )
        evidence = candidate_evidence(command, round_, text)
        ids = [item.person.candidate_id for item in evidence]
        self.assertEqual(
            ["rig_park:person_sitting:0", "rig_other:person_sitting:0"],
            ids,
        )
        self.assertIn("scene:police_car", evidence[0].camera_evidence)

    def test_round_pipeline_preserves_people_outside_the_soft_camera_match(self):
        text = "Find the person sitting on top of the trash can in the parking lot."
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate(
                    "rig_a:trash_can:0",
                    "trash_can",
                    (200.0, 200.0, 280.0, 400.0),
                    0.9,
                    "rig_a",
                    (640, 480),
                ),
                ObjectCandidate(
                    "rig_b:trash_can:0",
                    "trash_can",
                    (200.0, 200.0, 280.0, 400.0),
                    0.9,
                    "rig_b",
                    (640, 480),
                ),
            ],
            people=[
                PersonCandidate(
                    "rig_a:person_sitting:0",
                    (210.0, 120.0, 270.0, 220.0),
                    "person_sitting",
                    0.8,
                    "rig_a",
                    (640, 480),
                ),
                PersonCandidate(
                    "rig_b:person_sitting:0",
                    (210.0, 120.0, 270.0, 220.0),
                    "person_sitting",
                    0.9,
                    "rig_b",
                    (640, 480),
                ),
            ],
            camera_ids=["rig_a", "rig_b"],
        )

        result = run_stage1_candidate_pipeline(
            object(),
            text,
            detector=lambda *_args, **_kwargs: round_,
            verifier=lambda board: "rig_b:person_sitting:0|abnormal",
        )

        self.assertEqual(2, len(result.detections.people))
        self.assertEqual("rig_b:person_sitting:0", result.selected_candidate_id)

    def test_accident_without_vehicle_language_does_not_attach_bicycle(self):
        from marc_agent.perception.sar_selection import candidate_evidence

        text = (
            "Find the person lying face-down and motionless in the parking lot."
        )
        command = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate(
                    "rig_a:police_car:0",
                    "police_car",
                    (10.0, 10.0, 80.0, 80.0),
                    0.9,
                    "rig_a",
                    (640, 480),
                ),
                ObjectCandidate(
                    "rig_a:bicycle:0",
                    "bicycle",
                    (300.0, 200.0, 400.0, 280.0),
                    0.9,
                    "rig_a",
                    (640, 480),
                ),
            ],
            people=[
                PersonCandidate(
                    "rig_a:person_lying_down:0",
                    (30.0, 80.0, 130.0, 200.0),
                    "person_lying_down",
                    0.99,
                    "rig_a",
                    (640, 480),
                ),
                PersonCandidate(
                    "rig_a:person_prone:0",
                    (280.0, 160.0, 380.0, 260.0),
                    "person_prone",
                    0.7,
                    "rig_a",
                    (640, 480),
                ),
            ],
            camera_ids=["rig_a"],
        )
        evidence = candidate_evidence(command, round_, text)
        self.assertEqual("rig_a:person_prone:0", evidence[0].person.candidate_id)
        self.assertIsNone(evidence[0].anchor)

    def test_two_camera_shortlist_is_score_neutral(self):
        board = build_candidate_board(self.text, self.command, _two_bending_round())
        self.assertEqual(
            ("rig_a:person_bending:0", "rig_b:person_bending:0"),
            board.candidate_ids,
        )
        self.assertIn("P01 | rig_a:person_bending:0", board.prompt)
        self.assertIn("P02 | rig_b:person_bending:0", board.prompt)
        self.assertNotIn("rig_a:person_walking:0", board.prompt)
        result = select_sar_candidate(
            self.text,
            self.command,
            _two_bending_round(),
            verifier=lambda _board: "P01",
        )
        self.assertEqual("rig_a:person_bending:0", result.selected.candidate_id)
        self.assertEqual("qwen", result.method)

    def test_world_near_drops_far_same_camera_landmark_match(self):
        from marc_agent.perception.sar_selection import candidate_evidence

        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:trash_can:0", "trash_can", (310.0, 200.0, 370.0, 320.0)),
            ],
            people=[
                _person(
                    "rig_a:person_bending:0",
                    "person_bending",
                    (300.0, 150.0, 380.0, 280.0),
                    0.4,
                ),
                _person(
                    "rig_a:person_bending:1",
                    "person_bending",
                    (10.0, 150.0, 90.0, 280.0),
                    0.99,
                ),
            ],
            frames={"rig_a": _frame("rig_a")},
            camera_ids=["rig_a"],
        )
        projector = _downward_projector("rig_a")
        ranked = candidate_evidence(self.command, round_, self.text)
        shortlist = shortlist_for_verifier(
            ranked, self.command, self.text, projector
        )
        self.assertEqual(("rig_a:person_bending:0",), tuple(
            item.person.candidate_id for item in shortlist
        ))
        self.assertIsNotNone(shortlist[0].world_dist_m)
        self.assertLess(shortlist[0].world_dist_m, 4.0)
        board = build_candidate_board(
            self.text, self.command, round_, projector=projector
        )
        self.assertIn("world_xy=", board.prompt)
        self.assertIn("world_dist_m=", board.prompt)
        self.assertNotIn("rig_a:person_bending:1", board.prompt)

    def test_no_pose_words_keeps_every_person(self):
        text = "Find the most critical person in the parking lot."
        command = parse_command(text)
        board = build_candidate_board(text, command, _round())
        self.assertEqual(
            ("rig_a:person_bending:0", "rig_a:person_walking:0"),
            board.candidate_ids,
        )

    def test_unique_shortlist_skips_verifier(self):
        calls = []
        result = select_sar_candidate(
            self.text,
            self.command,
            _round(),
            verifier=lambda board: calls.append(board) or "P01",
        )
        self.assertEqual([], calls)
        self.assertEqual("rig_a:person_bending:0", result.selected.candidate_id)
        self.assertEqual("unique_shortlist", result.reason)
        self.assertEqual("deterministic_fallback", result.method)

    def test_fallback_after_bad_qwen_stays_on_shortlist(self):
        with self.assertLogs("marc_agent.sar_selection", level="INFO"):
            result = select_sar_candidate(
                self.text,
                self.command,
                _two_bending_round(),
                verifier=lambda _board: "not a candidate",
            )
        self.assertEqual("person_bending", result.selected.pose_class)
        self.assertIn(
            result.selected.candidate_id,
            {"rig_a:person_bending:0", "rig_b:person_bending:0"},
        )
        self.assertEqual("verifier_failed:ReplyParseError", result.reason)

    def test_lying_narrative_then_bent_keeps_only_bending(self):
        text = (
            "People are relaxing at the park and one is even lying on the grass; "
            "find the one near the trash can who is bent over and not moving "
            "and actually needs help."
        )
        command = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:trash_can:0", "trash_can", (300.0, 190.0, 430.0, 430.0)),
                _object("rig_b:trash_can:0", "trash_can", (300.0, 190.0, 430.0, 430.0), "rig_b"),
            ],
            people=[
                _person(
                    "rig_a:person_lying_down:0",
                    "person_lying_down",
                    (200.0, 120.0, 280.0, 220.0),
                    0.99,
                ),
                _person(
                    "rig_b:person_bending:0",
                    "person_bending",
                    (315.0, 100.0, 405.0, 390.0),
                    0.4,
                    "rig_b",
                ),
            ],
            frames={"rig_a": _frame("rig_a"), "rig_b": _frame("rig_b")},
            camera_ids=["rig_a", "rig_b"],
        )
        board = build_candidate_board(text, command, round_)
        self.assertEqual(("rig_b:person_bending:0",), board.candidate_ids)

    def test_bench_face_down_keeps_lying_and_prone(self):
        text = "Find the person lying face-down on the bench near the taxi."
        command = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:bench:0", "bench", (200.0, 200.0, 400.0, 350.0), "rig_a"),
                _object("rig_b:bench:0", "bench", (200.0, 200.0, 400.0, 350.0), "rig_b"),
                _object("rig_b:taxi:0", "taxi", (10.0, 10.0, 80.0, 80.0), "rig_b"),
            ],
            people=[
                _person(
                    "rig_a:person_lying_down:0",
                    "person_lying_down",
                    (210.0, 120.0, 270.0, 220.0),
                    0.8,
                    "rig_a",
                ),
                _person(
                    "rig_b:person_prone:0",
                    "person_prone",
                    (210.0, 120.0, 270.0, 220.0),
                    0.9,
                    "rig_b",
                ),
                _person(
                    "rig_a:person_walking:0",
                    "person_walking",
                    (30.0, 80.0, 130.0, 400.0),
                    0.99,
                    "rig_a",
                ),
            ],
            frames={"rig_a": _frame("rig_a"), "rig_b": _frame("rig_b")},
            camera_ids=["rig_a", "rig_b"],
        )
        board = build_candidate_board(text, command, round_)
        self.assertEqual(
            ("rig_a:person_lying_down:0", "rig_b:person_prone:0"),
            board.candidate_ids,
        )
        calls = []
        result = select_sar_candidate(
            text,
            command,
            round_,
            verifier=lambda board: calls.append(board) or "rig_b:person_prone:0",
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("rig_b:person_prone:0", result.selected.candidate_id)
        self.assertEqual("valid_model_output", result.reason)

    def test_lying_on_bench_with_prone_does_not_skip_verifier(self):
        text = "Near the taxi, find the person lying down on top of the bench."
        command = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:bench:0", "bench", (200.0, 200.0, 400.0, 350.0), "rig_a"),
                _object("rig_a:taxi:0", "taxi", (10.0, 10.0, 80.0, 80.0), "rig_a"),
            ],
            people=[
                _person(
                    "rig_a:person_lying_down:0",
                    "person_lying_down",
                    (210.0, 120.0, 270.0, 220.0),
                    0.8,
                ),
                _person(
                    "rig_a:person_prone:1",
                    "person_prone",
                    (300.0, 140.0, 370.0, 240.0),
                    0.7,
                ),
            ],
            frames={"rig_a": _frame("rig_a")},
            camera_ids=["rig_a"],
        )
        calls = []
        result = select_sar_candidate(
            text,
            command,
            round_,
            verifier=lambda board: calls.append(board) or "rig_a:person_prone:1",
        )
        self.assertEqual(1, len(calls))
        self.assertEqual("rig_a:person_prone:1", result.selected.candidate_id)
        self.assertNotEqual("unique_shortlist", result.reason)

    def test_fallen_by_bicycle_keeps_prostrate_not_walking(self):
        text = "Find the person fallen on the ground next to the toppled bicycle."
        command = parse_command(text)
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:bicycle:0", "bicycle", (200.0, 200.0, 320.0, 280.0)),
            ],
            people=[
                _person(
                    "rig_a:person_walking:0",
                    "person_walking",
                    (30.0, 80.0, 130.0, 400.0),
                    0.99,
                ),
                _person(
                    "rig_a:person_bending:0",
                    "person_bending",
                    (210.0, 120.0, 270.0, 220.0),
                    0.4,
                ),
                _person(
                    "rig_a:person_lying_down:0",
                    "person_lying_down",
                    (180.0, 150.0, 250.0, 230.0),
                    0.5,
                ),
                _person(
                    "rig_a:person_prone:0",
                    "person_prone",
                    (300.0, 160.0, 380.0, 240.0),
                    0.6,
                ),
            ],
            frames={"rig_a": _frame("rig_a")},
            camera_ids=["rig_a"],
        )
        board = build_candidate_board(text, command, round_)
        self.assertEqual(
            {
                "rig_a:person_bending:0",
                "rig_a:person_lying_down:0",
                "rig_a:person_prone:0",
            },
            set(board.candidate_ids),
        )
        self.assertNotIn("rig_a:person_walking:0", board.candidate_ids)


if __name__ == "__main__":
    unittest.main()
