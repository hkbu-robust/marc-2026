"""Per-round dump folders: command, overlays, detections; env off writes nothing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from marc_agent.command_parser import Intent, ParsedCommand, Relation
from marc_agent.perception.object_selection import ObjectSelection
from marc_agent.perception.round_dump import dump_stage_round
from marc_agent.perception.stage1_pipeline import Stage1CandidateResult, Stage1RoundResult
from marc_agent.types import CctvDetectionRound, CctvFrame, ObjectCandidate


def _round_result():
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[6:22, 10:36] = (200, 20, 20)
    frame = CctvFrame("cam1", 48, 32, rgb)
    target = ObjectCandidate(
        "cam1:mug:0", "mug", (10.0, 6.0, 36.0, 22.0), 0.91, "cam1", (48, 32),
    )
    detections = CctvDetectionRound(
        objects=[target],
        frames={"cam1": frame},
        camera_ids=["cam1"],
    )
    parsed = ParsedCommand(
        intent=Intent.FIND,
        target_type="mug",
        landmark="bench",
        relation=Relation.NEAR,
    )
    candidates = Stage1CandidateResult(
        command_text="find the mug next to the bench",
        parsed=parsed,
        detections=detections,
        sar_selection=None,
        failure_reason=None,
        total_ms=12.0,
    )
    grounding = SimpleNamespace(
        camera_id="cam1",
        target_type="mug",
        anchor_coord=[1.0, 2.0, 0.1],
        target_coord=[1.2, 2.1, 0.1],
        landmark="bench",
        relation="near",
        situation=None,
    )
    return Stage1RoundResult(
        candidates=candidates,
        grounding=grounding,
        method="object_relation",
        failure_reason=None,
        target_id="cam1:mug:0",
        anchor_id=None,
    )


class RoundDumpTests(unittest.TestCase):
    def test_writes_command_overlay_and_detections(self):
        round_result = _round_result()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage1_003"
            with patch.dict(os.environ, {"MARC_ROUND_DUMP": "1"}, clear=False):
                path = dump_stage_round(out, round_result)
            self.assertEqual(out, path)
            self.assertTrue((out / "command.txt").is_file())
            self.assertIn("mug", (out / "command.txt").read_text(encoding="utf-8"))
            self.assertTrue((out / "cameras" / "cam1.png").is_file())
            self.assertTrue((out / "cameras" / "cam1_boxes.png").is_file())
            detections = json.loads((out / "detections.json").read_text(encoding="utf-8"))
            self.assertEqual("cam1:mug:0", detections["objects"][0]["candidate_id"])
            grounding = json.loads((out / "grounding.json").read_text(encoding="utf-8"))
            self.assertEqual([1.2, 2.1, 0.1], grounding["submitted"]["target_coord"])
            self.assertNotEqual([0, 0, 0], grounding["submitted"]["target_coord"])

    def test_disabled_writes_nothing(self):
        round_result = _round_result()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage1_003"
            with patch.dict(os.environ, {"MARC_ROUND_DUMP": "0"}, clear=False):
                path = dump_stage_round(out, round_result)
            self.assertIsNone(path)
            self.assertFalse(out.exists())

    def test_deletes_stale_qwen_transcript_when_verifier_skipped(self):
        round_result = _round_result()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage1_001"
            out.mkdir()
            leftover = out / "qwen.txt"
            leftover.write_text("stale leftover\n", encoding="utf-8")
            with patch.dict(os.environ, {"MARC_ROUND_DUMP": "1"}, clear=False):
                dump_stage_round(out, round_result)
            self.assertFalse(leftover.exists())

    def test_writes_actual_qwen_prompt_reply_and_evidence_board(self):
        result = _round_result()
        target = result.candidates.detections.objects[0]
        image = Image.fromarray(result.candidates.detections.frames["cam1"].rgb)
        selection = ObjectSelection(
            target,
            None,
            1.0,
            method="qwen_remote",
            raw_output=target.candidate_id,
            prompt="Cached all-camera scene reference:\ncam1: red mug",
            evidence_images=(image,),
        )
        result = replace(
            result,
            candidates=replace(result.candidates, object_selection=selection),
            method="qwen_remote",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "stage1_001"
            with patch.dict(os.environ, {"MARC_ROUND_DUMP": "1"}, clear=False):
                dump_stage_round(out, result)
            transcript = (out / "qwen.txt").read_text(encoding="utf-8")
            self.assertIn("Cached all-camera scene reference", transcript)
            self.assertIn(target.candidate_id, transcript)
            self.assertTrue((out / "qwen_object_01.png").is_file())


if __name__ == "__main__":
    unittest.main()
