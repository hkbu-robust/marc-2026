"""Round-level Stage 1 candidate collection and SAR selection composition."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

from marc_agent.command_parser import CommandParseError, ParsedCommand, parse_command
from marc_agent.perception.sar_selection import (
    QwenSarVerifier,
    SarSelectionResult,
    Verifier,
    select_sar_candidate,
)
from marc_agent.perception.stage1_detect import detect_stage1_round
from marc_agent.types import CctvDetectionRound, CctvRoundMetrics


Detector = Callable[..., CctvDetectionRound]


@dataclass(frozen=True)
class Stage1CandidateResult:
    """Candidate-stage output; deliberately contains no world coordinates."""

    command_text: str
    parsed: Optional[ParsedCommand]
    detections: CctvDetectionRound
    sar_selection: Optional[SarSelectionResult]
    failure_reason: Optional[str]
    total_ms: float

    @property
    def selected_candidate_id(self) -> Optional[str]:
        if self.sar_selection and self.sar_selection.selected:
            return self.sar_selection.selected.candidate_id
        return None

    def record(self, *, expected_candidate_id: Optional[str] = None) -> dict:
        retained_ids = [item.candidate_id for item in self.detections.objects]
        retained_ids.extend(item.candidate_id for item in self.detections.people)
        recall = None
        if expected_candidate_id is not None:
            recall = 1.0 if expected_candidate_id in retained_ids else 0.0
        return {
            "command": self.command_text,
            "parsed": asdict(self.parsed) if self.parsed else None,
            "camera_ids": self.detections.camera_ids,
            "frames_loaded": sorted(self.detections.frames),
            "object_candidates": len(self.detections.objects),
            "person_candidates": len(self.detections.people),
            "selected_candidate_id": self.selected_candidate_id,
            "selection_status": (
                self.sar_selection.status.value if self.sar_selection else None
            ),
            "selection_method": self.sar_selection.method if self.sar_selection else None,
            "selection_reason": self.sar_selection.reason if self.sar_selection else None,
            "failure_reason": self.failure_reason,
            "candidate_recall": recall,
            "metrics": asdict(self.detections.metrics),
            "total_ms": self.total_ms,
        }


def _default_verifier() -> Optional[Verifier]:
    mode = (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()
    return QwenSarVerifier() if mode == "qwen" else None


def run_stage1_candidate_pipeline(
    client,
    command_text: str,
    *,
    detector: Detector = detect_stage1_round,
    verifier: Optional[Verifier] = None,
    stub_path: Optional[str] = None,
) -> Stage1CandidateResult:
    """Parse one request, collect all-camera candidates, and select SAR if requested."""
    started = time.perf_counter()
    try:
        parsed = parse_command(command_text)
    except CommandParseError as exc:
        return Stage1CandidateResult(
            command_text,
            None,
            CctvDetectionRound(),
            None,
            f"command_parse:{exc.code.value}",
            (time.perf_counter() - started) * 1000.0,
        )

    try:
        detections = detector(client, stub_path=stub_path)
    except Exception as exc:
        reason = f"candidate_collection:{type(exc).__name__}"
        return Stage1CandidateResult(
            command_text,
            parsed,
            CctvDetectionRound(metrics=CctvRoundMetrics(failures=(reason,))),
            None,
            reason,
            (time.perf_counter() - started) * 1000.0,
        )
    selection = None
    failure_reason = None
    if parsed.target_type == "person":
        selected_verifier = verifier if verifier is not None else _default_verifier()
        selection = select_sar_candidate(
            command_text,
            parsed,
            detections,
            verifier=selected_verifier,
        )
        if selection.selected is None:
            failure_reason = f"sar_selection:{selection.reason}"

    return Stage1CandidateResult(
        command_text,
        parsed,
        detections,
        selection,
        failure_reason,
        (time.perf_counter() - started) * 1000.0,
    )


def append_stage1_record(
    result: Stage1CandidateResult,
    path: str | Path,
    *,
    expected_candidate_id: Optional[str] = None,
) -> None:
    """Append one structured round record for latency, recall, and failure audits."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        json.dump(
            result.record(expected_candidate_id=expected_candidate_id),
            stream,
            sort_keys=True,
        )
        stream.write("\n")
