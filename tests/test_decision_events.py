"""Every non-normal Stage 1 decision path leaves a structured note (#85).

The notes are logging only: these tests also check that recording them does
not change the submitted grounding.
"""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import marc_agent.experiment_log as experiment_log
from marc_agent.command_parser import parse_command
from marc_agent.experiment_log import decision_scope, note_decision, start_experiment
from marc_agent.perception.cctv_project import CctvProjector, static_lookup
from marc_agent.perception.cctv_yolo import refine_detection_round
from marc_agent.perception.object_selection import (
    ObjectSelection,
    select_object_candidate,
)
from marc_agent.perception.sar_selection import select_sar_candidate
from marc_agent.perception.stage1_pipeline import (
    StaticSceneCache,
    _project_pair,
    _project_selection,
    compose_stage1_grounding,
    run_stage1_candidate_pipeline,
    run_stage1_round,
)
from marc_agent.round_clock import RoundClock
from marc_agent.types import CctvDetectionRound, ObjectCandidate

from test_sar_selection import _frame, _object, _person
from test_stage1_round import _geometry, _object_round, _person_round


def _notes(run, *args, **kwargs):
    """Run ``run`` inside a decision scope and return (result, notes)."""
    with decision_scope("stage1", 1) as notes:
        result = run(*args, **kwargs)
    return result, notes


def _only(notes, decision, **match):
    found = [
        note for note in notes
        if note["decision"] == decision
        and all(note.get(key) == value for key, value in match.items())
    ]
    assert len(found) == 1, (decision, match, notes)
    return found[0]


def _fixed(round_):
    return lambda client, *, stub_path=None: round_


def _object_with_frames():
    round_ = _object_round()
    round_.frames = {"rig_a": object(), "rig_b": object()}
    return round_


def _raise(_board):
    raise RuntimeError("vLLM down")


class RecorderTests(unittest.TestCase):
    def tearDown(self):
        experiment_log._current = None

    def test_scope_collects_notes_and_run_writes_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = start_experiment(tmp, environ={})
            with decision_scope("stage1", 7) as notes:
                note_decision("sentinel", reason="no_object_pairs")
            self.assertEqual(
                [{"decision": "sentinel", "reason": "no_object_pairs"}], notes
            )
            events = [
                json.loads(line)
                for line in run.events_path.read_text().splitlines()
            ]
            event = events[-1]
            self.assertEqual("decision.sentinel", event["event"])
            self.assertEqual("stage1", event["stage"])
            self.assertEqual(7, event["round"])
            self.assertEqual("no_object_pairs", event["reason"])

    def test_note_outside_scope_and_run_is_silent(self):
        self.assertIsNone(note_decision("selector", outcome="qwen_choice"))

    def test_note_never_raises(self):
        broken = Mock()
        broken.event.side_effect = RuntimeError("disk full")
        experiment_log._current = broken
        with decision_scope("stage1", 1) as notes:
            note_decision("sentinel", reason="no_object_pairs")
        self.assertEqual(1, len(notes))

    def test_call_sites_pass_a_name_and_no_reserved_fields(self):
        reserved = {"stage", "round", "status", "decision"}
        sources = [
            *Path("marc_agent").rglob("*.py"),
            Path("demo/participant_app.py"),
        ]
        calls = 0
        for path in sources:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "note_decision"
                ):
                    continue
                calls += 1
                self.assertIsInstance(node.args[0], ast.Constant, path)
                self.assertEqual(1, len(node.args), path)
                names = {keyword.arg for keyword in node.keywords}
                self.assertFalse(names & reserved, (path, node.lineno))
        self.assertGreaterEqual(calls, 20)


class ObjectSelectorTests(unittest.TestCase):
    def test_no_verifier_is_geometry_only(self):
        _, notes = _notes(
            run_stage1_round, object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()), projector=_geometry(),
        )
        _only(notes, "selector", path="object", outcome="geometry_only",
              reason="no_verifier")

    def test_valid_reply_is_qwen_choice(self):
        _, notes = _notes(
            run_stage1_round, object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()), projector=_geometry(),
            verifier=lambda board: "rig_a:mug:0|rig_a:picnic_table:0",
        )
        _only(notes, "selector", path="object", outcome="qwen_choice")

    def test_failed_call_is_a_labelled_fallback_with_the_geometry_answer(self):
        failed, notes = _notes(
            run_stage1_round, object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()), projector=_geometry(),
            verifier=_raise,
        )
        geometry = run_stage1_round(
            object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()), projector=_geometry(),
        )
        note = _only(notes, "selector", path="object", outcome="fallback")
        self.assertEqual("verifier_failed:RuntimeError", note["reason"])
        self.assertEqual("target_pick", note["failed_step"])
        self.assertEqual("deterministic_fallback", failed.method)
        self.assertEqual(geometry.grounding, failed.grounding)

    def test_malformed_anchor_repick_records_the_failed_step(self):
        # Seen live: Qwen answers the anchor re-pick with the target id only.
        text = "Find the tumbler I dropped as I got out of my car in the parking lot."
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:tumbler:0", "tumbler", (20, 100, 60, 160), "cam"),
                _object("cam:tumbler:1", "tumbler", (150, 100, 190, 160), "cam"),
                _object("cam:police_car:0", "police_car", (10, 50, 200, 200), "cam"),
                _object("cam:suv:0", "suv", (120, 60, 320, 210), "cam"),
            ],
            frames={"cam": _frame("cam")},
            camera_ids=["cam"],
        )

        class Verifier:
            method = "qwen_remote"
            last_metrics = None
            last_finish_reason = "stop"

            def __call__(self, board):
                return "cam:tumbler:1"

        selected, notes = _notes(
            select_object_candidate, parse_command(text), round_,
            command_text=text, verifier=Verifier(),
        )
        note = _only(notes, "selector", outcome="fallback")
        self.assertEqual("verifier_failed:ReplyParseError", note["reason"])
        self.assertEqual("anchor_repick", note["failed_step"])
        self.assertEqual("cam:tumbler:1", note["qwen_target"])
        self.assertEqual("deterministic_fallback", selected.method)

    def test_guard_veto_is_a_geometry_override(self):
        text = "Find the mug near the bench."
        round_ = CctvDetectionRound(
            objects=[
                _object("rig_a:mug:0", "mug", (200.0, 300.0, 230.0, 330.0)),
                _object("rig_a:bench:0", "bench", (180.0, 320.0, 400.0, 420.0)),
                ObjectCandidate(
                    "rig_a:mug:1", "mug", (380.0, 330.0, 392.0, 342.0), 0.1,
                    "rig_a", (640, 480),
                ),
            ],
            frames={"rig_a": _frame("rig_a")},
            camera_ids=["rig_a"],
        )
        selected, notes = _notes(
            select_object_candidate, parse_command(text), round_,
            command_text=text, verifier=lambda board: "rig_a:mug:1|rig_a:bench:0",
        )
        self.assertEqual("rig_a:mug:0", selected.target.candidate_id)
        note = _only(notes, "geometry_override", path="object")
        self.assertEqual("weak_override_guard", note["reason"])
        self.assertEqual("rig_a:mug:1", note["qwen_target"])
        _only(notes, "selector", outcome="geometry_override")

    def test_failed_class_screen_is_recorded(self):
        text = "Find the mug."
        round_ = CctvDetectionRound(
            objects=[
                _object("cam:mug:0", "mug", (20, 100, 60, 160), "cam"),
                _object("cam:mug:1", "mug", (80, 100, 120, 160), "cam"),
                _object("cam:cola_can:0", "cola_can", (140, 100, 180, 160), "cam"),
                _object("cam:cola_can:1", "cola_can", (200, 100, 240, 160), "cam"),
            ],
            frames={"cam": _frame("cam")},
            camera_ids=["cam"],
        )
        calls = []

        class Verifier:
            method = "qwen_remote"
            last_metrics = None
            last_finish_reason = "stop"

            def __call__(self, board):
                calls.append(board)
                if len(calls) == 1:
                    raise TimeoutError("slow")
                return "cam:mug:1"

        _, notes = _notes(
            select_object_candidate, parse_command(text), round_,
            command_text=text, verifier=Verifier(),
        )
        note = _only(notes, "class_screen_failed", class_name="mug")
        self.assertEqual("TimeoutError", note["error"])


class SarSelectorTests(unittest.TestCase):
    text = "Find the person bent over near the trash can who needs urgent help."

    def test_valid_reply_is_qwen_choice(self):
        _, notes = _notes(
            run_stage1_round, object(), self.text,
            detector=_fixed(_person_round()), projector=_geometry(),
            verifier=lambda board: "rig_a:person_bending:0|emergency",
        )
        _only(notes, "selector", path="sar", outcome="qwen_choice")

    def test_failed_call_is_a_fallback(self):
        _, notes = _notes(
            run_stage1_round, object(), self.text,
            detector=_fixed(_person_round()), projector=_geometry(),
            verifier=_raise,
        )
        note = _only(notes, "selector", path="sar", outcome="fallback")
        self.assertEqual("verifier_failed:RuntimeError", note["reason"])

    def test_no_verifier_is_geometry_only(self):
        _, notes = _notes(
            run_stage1_round, object(), self.text,
            detector=_fixed(_person_round()), projector=_geometry(),
        )
        _only(notes, "selector", path="sar", outcome="geometry_only")

    def test_pose_guard_is_a_geometry_override(self):
        text = "Find the person collapsed on the parking lot floor."
        round_ = CctvDetectionRound(
            objects=[_object("rig_b:suv:0", "suv", (20.0, 20.0, 120.0, 120.0), "rig_b")],
            people=[
                _person("rig_b:person_lying_down:0", "person_lying_down",
                        (200, 200, 320, 280), 0.8, "rig_b"),
                _person("rig_b:person_prone:0", "person_prone",
                        (330, 200, 450, 280), 0.9, "rig_b"),
            ],
            frames={"rig_b": _frame("rig_b")},
            camera_ids=["rig_b"],
        )
        _, notes = _notes(
            select_sar_candidate, text, parse_command(text), round_,
            verifier=lambda _board: "rig_b:person_prone:0",
        )
        note = _only(notes, "geometry_override", path="sar")
        self.assertEqual("pose_geometry_override", note["reason"])
        self.assertEqual(["pose_geometry_override"], note["guards"])
        self.assertEqual("rig_b:person_prone:0", note["qwen_target"])
        _only(notes, "selector", path="sar", outcome="geometry_override")


class QwenAvailabilityTests(unittest.TestCase):
    def test_time_budget_skip_is_recorded(self):
        clock = RoundClock()
        clock.begin(1.0, 0.0)
        result, notes = _notes(
            run_stage1_round, object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()), projector=_geometry(),
            verifier=lambda board: "rig_a:mug:0|rig_a:picnic_table:0",
            clock=clock,
        )
        _only(notes, "qwen_skipped", reason="time_budget")
        _only(notes, "selector", outcome="geometry_only", reason="no_verifier")
        self.assertEqual("object_relation", result.method)

    def test_scene_description_failure_is_recorded(self):
        class Verifier:
            def __call__(self, board):
                return "rig_a:mug:0|rig_a:picnic_table:0"

            def prepare_scene(self, round_, command_text=""):
                raise ConnectionError("down")

        _, notes = _notes(
            run_stage1_round, object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()), projector=_geometry(),
            verifier=Verifier(),
        )
        _only(notes, "scene_description_failed", error="ConnectionError")


class ProxyTests(unittest.TestCase):
    def test_landmark_context_proxy_is_recorded(self):
        from marc_agent.perception.cctv_yolo import CctvYoloDetector, RuntimeContract
        import numpy as np
        from marc_agent.types import CctvFrame

        detector = CctvYoloDetector.__new__(CctvYoloDetector)
        detector.contract = RuntimeContract.load()
        detector.model = object()
        detector.names = {}
        detector._predict_frame = Mock(return_value=[])
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = {
            camera_id: CctvFrame(camera_id, 640, 480, rgb.copy())
            for camera_id in ("cam_a", "cam_b")
        }
        objects = [
            ObjectCandidate(candidate_id, class_name, box, 0.9, camera_id, (640, 480))
            for candidate_id, class_name, box, camera_id in (
                ("cam_a:umbrella:0", "umbrella", (20, 20, 40, 50), "cam_a"),
                ("cam_a:trash_can:0", "trash_can", (100, 100, 180, 300), "cam_a"),
                ("cam_b:trash_can:0", "trash_can", (300, 100, 380, 300), "cam_b"),
            )
        ]
        _, notes = _notes(
            detector.refine_around_landmarks,
            frames, objects, "umbrella", frozenset({"trash_can"}),
        )
        note = _only(notes, "proxy_created", variant="landmark_context")
        self.assertEqual("cam_b:umbrella:0", note["candidate"])
        self.assertEqual("cam_b:trash_can:0", note["landmark"])

    def test_refine_additions_are_recorded(self):
        parsed = parse_command("Find the tumbler on the bench.")
        missed = ObjectCandidate(
            "rig_1:tumbler:0", "tumbler", (200.0, 200.0, 240.0, 250.0), 0.2,
            "rig_1", (640, 480),
        )
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate("rig_1:bench:0", "bench", (180.0, 180.0, 400.0, 360.0),
                                0.9, "rig_1", (640, 480)),
            ],
            camera_ids=["rig_1"],
        )
        detector = Mock()
        detector.refine_around_landmarks.return_value = [missed]
        with patch("marc_agent.perception.cctv_yolo._detector", detector):
            _, notes = _notes(refine_detection_round, parsed, round_)
        note = _only(notes, "refine_added", path="object")
        self.assertEqual(["rig_1:tumbler:0"], note["candidates"])

    def test_hidden_umbrella_proxy_is_recorded(self):
        target = ObjectCandidate(
            "rig_a:umbrella:0", "umbrella", (10, 10, 20, 20), 0.019, "rig_a", (1280, 720),
        )
        anchor = ObjectCandidate(
            "rig_a:trash_can:0", "trash_can", (100, 100, 200, 300), 0.9, "rig_a", (1280, 720),
        )
        projector = Mock()
        projector.project_contact.return_value = (0.0, 0.0, 0.0)
        _, notes = _notes(
            _project_selection, projector,
            parse_command("Find the umbrella next to the trash can."),
            ObjectSelection(target, anchor, 1.0), [target, anchor],
        )
        note = _only(notes, "projection_proxy", variant="hidden_umbrella")
        self.assertEqual("rig_a:trash_can:0", note["proxy"])

    def test_clipped_target_proxy_is_recorded(self):
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate("rig_a:pencilcase:0", "pencilcase",
                                (635.0, 300.0, 640.0, 340.0), 0.006, "rig_a", (640, 480)),
                ObjectCandidate("rig_a:picnic_table:0", "picnic_table",
                                (200.0, 220.0, 500.0, 420.0), 0.9, "rig_a", (640, 480)),
                ObjectCandidate("rig_a:trash_can:0", "trash_can",
                                (380.0, 150.0, 440.0, 260.0), 0.9, "rig_a", (640, 480)),
            ],
            camera_ids=["rig_a"],
        )
        _, notes = _notes(
            run_stage1_round, object(), "Find the pencil case near the picnic table.",
            detector=_fixed(round_), projector=_geometry(),
        )
        note = _only(notes, "projection_proxy", variant="clipped_target")
        self.assertEqual("rig_a:trash_can:0", note["proxy"])

    def test_occluded_target_proxy_is_recorded(self):
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate("rig_a:umbrella:0", "umbrella",
                                (215.0, 240.0, 226.0, 264.0), 0.9, "rig_a", (640, 480)),
                ObjectCandidate("rig_a:taxi:0", "taxi",
                                (300.0, 280.0, 600.0, 470.0), 0.9, "rig_a", (640, 480)),
                ObjectCandidate("rig_a:kick_scooter:0", "kick_scooter",
                                (200.0, 120.0, 320.0, 260.0), 0.9, "rig_a", (640, 480)),
            ],
            camera_ids=["rig_a"],
        )
        _, notes = _notes(
            run_stage1_round, object(), "Find the umbrella near the taxi.",
            detector=_fixed(round_), projector=_geometry(),
        )
        note = _only(notes, "projection_proxy", variant="occluded_target")
        self.assertEqual("rig_a:kick_scooter:0", note["proxy"])


class ProjectionTests(unittest.TestCase):
    def test_anchor_projection_failure_is_recorded(self):
        anchor_box = (100.0, 100.0, 200.0, 200.0)
        projector = Mock()
        projector.project_contact.side_effect = (
            lambda camera_id, box, **kwargs: None if box == anchor_box else (1.0, 2.0, 0.0)
        )
        projected, notes = _notes(
            _project_pair, projector, "rig_a", (0.0, 0.0, 10.0, 10.0), anchor_box,
            relation=None, support_class=None,
        )
        self.assertEqual(((1.0, 2.0, 0.0), (1.0, 2.0, 0.0)), projected)
        _only(notes, "anchor_collapsed", camera_id="rig_a")

    def test_fallthrough_to_a_lower_pair_is_recorded(self):
        candidates = run_stage1_candidate_pipeline(
            object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()), projector=_geometry(),
        )
        self.assertEqual("rig_a:mug:0", candidates.object_selection.target.candidate_id)
        projector = _geometry()
        project = projector.project_contact
        first_box = candidates.object_selection.target.bbox_xyxy
        projector.project_contact = lambda camera_id, box, **kwargs: (
            None if tuple(box) == tuple(first_box) else project(camera_id, box, **kwargs)
        )
        result, notes = _notes(compose_stage1_grounding, candidates, projector)
        note = _only(notes, "projection_fallthrough")
        self.assertEqual("rig_a:mug:0", note["failed_target"])
        self.assertEqual(result.target_id, note["used_target"])
        self.assertGreater(note["rank"], 0)


class SentinelTests(unittest.TestCase):
    def test_no_object_pairs(self):
        round_ = CctvDetectionRound(
            objects=[
                ObjectCandidate("rig_a:vending_machine:0", "vending_machine",
                                (200.0, 180.0, 420.0, 430.0), 0.92, "rig_a", (640, 480)),
            ],
            camera_ids=["rig_a"],
        )
        _, notes = _notes(
            run_stage1_round, object(), "Find the umbrella beside the vending machine.",
            detector=_fixed(round_), projector=_geometry(),
        )
        _only(notes, "sentinel", reason="no_object_pairs")

    def test_object_projection_failed(self):
        _, notes = _notes(
            run_stage1_round, object(), "Find the mug on the picnic table.",
            detector=_fixed(_object_round()),
            projector=CctvProjector(static_lookup({})),
        )
        _only(notes, "sentinel", reason="object_projection_failed")

    def test_person_projection_failed(self):
        _, notes = _notes(
            run_stage1_round, object(),
            "Find the person bent over near the trash can who needs urgent help.",
            detector=_fixed(_person_round()),
            verifier=lambda board: "rig_a:person_bending:0|emergency",
            projector=CctvProjector(static_lookup({})),
        )
        _only(notes, "sentinel", reason="person_projection_failed")

    def test_no_person_selected(self):
        round_ = CctvDetectionRound(objects=[], people=[], camera_ids=["rig_a"])
        result, notes = _notes(
            run_stage1_round, object(), "Find the person lying on the bench.",
            detector=_fixed(round_), projector=_geometry(),
        )
        _only(notes, "sentinel", reason="no_person_selected")
        self.assertEqual("ignored_class_dummy", result.method)


class StaticSceneCacheTests(unittest.TestCase):
    command = "Find the mug on the picnic table."

    def _round(self, verifier=None, detections=None):
        return run_stage1_round(
            object(), self.command,
            detector=_fixed(detections or _object_with_frames()),
            projector=_geometry(), verifier=verifier,
        )

    def test_hit_submits_the_stored_answer_and_records_the_fresh_one(self):
        cache = StaticSceneCache(enabled=True)
        first = cache.apply(self._round(), 3)
        moved = _object_with_frames()
        moved.objects = [item for item in moved.objects if item.candidate_id != "rig_a:mug:0"]
        fresh = self._round(detections=moved)
        self.assertNotEqual(first.target_id, fresh.target_id)
        with decision_scope("stage1", 9) as notes:
            reused = cache.apply(fresh, 9)
        self.assertEqual("static_scene_cache", reused.method)
        self.assertEqual(first.grounding, reused.grounding)
        note = _only(notes, "cache_hit")
        self.assertEqual(3, note["source_round"])
        self.assertEqual(first.target_id, note["cached_target"])
        self.assertEqual(fresh.target_id, note["fresh_target"])
        self.assertFalse(note["same_pair"])
        self.assertGreater(note["target_shift_m"], 0.0)

    def test_switch_off_never_reuses(self):
        cache = StaticSceneCache.from_env({"MARC_STAGE1_SCENE_CACHE": "0"})
        cache.apply(self._round(), 1)
        fresh = self._round()
        with decision_scope("stage1", 2) as notes:
            result = cache.apply(fresh, 2)
        self.assertIs(fresh, result)
        self.assertEqual([], notes)

    def test_default_is_off(self):
        self.assertFalse(StaticSceneCache.from_env({}).enabled)
        self.assertFalse(StaticSceneCache.from_env({"MARC_STAGE1_SCENE_CACHE": "0"}).enabled)
        self.assertTrue(StaticSceneCache.from_env({"MARC_STAGE1_SCENE_CACHE": "1"}).enabled)

    def test_incomplete_coverage_is_not_stored(self):
        cache = StaticSceneCache(enabled=True)
        cache.apply(self._round(detections=_object_round()), 1)
        fresh = self._round()
        self.assertIs(fresh, cache.apply(fresh, 2))


class NoBehaviourChangeTests(unittest.TestCase):
    """Recording on (scope and run active) gives the same grounding as off."""

    cases = [
        ("Find the mug on the picnic table.", _object_round, None),
        ("Find the mug on the picnic table.", _object_round,
         lambda board: "rig_a:mug:0|rig_a:picnic_table:0"),
        ("Find the mug on the picnic table.", _object_round, _raise),
        ("Find the person bent over near the trash can who needs urgent help.",
         _person_round, lambda board: "rig_a:person_bending:0|emergency"),
        ("Find the person bent over near the trash can who needs urgent help.",
         _person_round, _raise),
        ("Find the umbrella beside the vending machine.", _object_round, None),
    ]

    def tearDown(self):
        experiment_log._current = None

    def test_grounding_is_identical_with_recording_on_and_off(self):
        for text, make_round, verifier in self.cases:
            with self.subTest(text=text, verifier=verifier):
                off = run_stage1_round(
                    object(), text, detector=_fixed(make_round()),
                    projector=_geometry(), verifier=verifier,
                )
                with tempfile.TemporaryDirectory() as tmp:
                    start_experiment(tmp, environ={})
                    with decision_scope("stage1", 1):
                        on = run_stage1_round(
                            object(), text, detector=_fixed(make_round()),
                            projector=_geometry(), verifier=verifier,
                        )
                    experiment_log._current = None
                self.assertEqual(off.grounding, on.grounding)
                self.assertEqual(off.method, on.method)
                self.assertEqual(off.target_id, on.target_id)
                self.assertEqual(off.anchor_id, on.anchor_id)


if __name__ == "__main__":
    unittest.main()
