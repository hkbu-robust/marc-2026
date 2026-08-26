"""Round-level Stage 1 candidate collection and SAR selection composition."""

from __future__ import annotations

import json
import logging
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
from marc_agent.perception.qwen_remote import (
    RemoteQwenSarVerifier,
    check_remote_health,
)
from marc_agent.perception.qwen_runtime import (
    QwenBackend,
    QwenRemoteConfig,
    configured_backend,
)
from marc_agent.perception.stage1_detect import detect_stage1_round
from marc_agent.types import CctvDetectionRound, CctvRoundMetrics


Detector = Callable[..., CctvDetectionRound]
log = logging.getLogger("marc_agent.stage1_pipeline")


def log_qwen_startup_status() -> None:
    """Log configured Qwen mode and non-fatally probe a remote vLLM server."""
    mode = (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()
    if mode not in {"qwen", "qwen-remote"}:
        log.info("[qwen-startup] disabled mode=%s", mode)
        return
    try:
        backend = QwenBackend.REMOTE if mode == "qwen-remote" else configured_backend()
        if backend is QwenBackend.LOCAL:
            verifier = QwenSarVerifier()
            log.info(
                "[qwen-startup] configured backend=local model_dir=%s",
                verifier.config.model_dir,
            )
            return
        config = QwenRemoteConfig.load()
        log.info(
            "[qwen-startup] configured backend=remote models_url=%s model=%s "
            "timeout=%.1fs",
            config.models_url,
            config.model_name,
            config.timeout_s,
        )
        started = time.perf_counter()
        model_ids = check_remote_health(config)
        log.info(
            "[qwen-startup] health=ok model=%s available=%s latency=%.0fms",
            config.model_name,
            ",".join(model_ids),
            (time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        log.warning(
            "[qwen-startup] health=failed error=%s detail=%s; "
            "SAR rounds will use deterministic fallback if the failure persists",
            type(exc).__name__,
            exc,
        )


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
            "verifier_metrics": (
                asdict(self.sar_selection.verifier_metrics)
                if self.sar_selection and self.sar_selection.verifier_metrics
                else None
            ),
            "failure_reason": self.failure_reason,
            "candidate_recall": recall,
            "metrics": asdict(self.detections.metrics),
            "total_ms": self.total_ms,
        }


def _default_verifier() -> Optional[Verifier]:
    mode = (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()
    if mode not in {"qwen", "qwen-remote"}:
        log.info("[qwen] verifier disabled mode=%s", mode)
        return None
    backend = QwenBackend.REMOTE if mode == "qwen-remote" else configured_backend()
    if backend is QwenBackend.REMOTE:
        verifier = RemoteQwenSarVerifier()
        log.info(
            "[qwen] verifier configured backend=remote endpoint=%s model=%s "
            "timeout=%.1fs",
            verifier.config.chat_completions_url,
            verifier.config.model_name,
            verifier.config.timeout_s,
        )
        return verifier
    verifier = QwenSarVerifier()
    log.info(
        "[qwen] verifier configured backend=local model_dir=%s",
        verifier.config.model_dir,
    )
    return verifier


def _failed_verifier(exc: Exception) -> Verifier:
    """Delay backend setup failures until the shared fallback can classify them."""
    def verify(_board):
        raise exc

    return verify


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

    log.info(
        "[stage1-pipeline] parsed target_type=%s situation=%s landmark=%s relation=%s",
        parsed.target_type,
        parsed.situation.value if parsed.situation else None,
        parsed.landmark,
        parsed.relation.value if parsed.relation else None,
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
        log.info(
            "[qwen] SAR request situation=%s people=%d cameras=%d",
            parsed.situation.value if parsed.situation else None,
            len(detections.people),
            len(detections.camera_ids),
        )
        if verifier is not None:
            selected_verifier = verifier
        else:
            try:
                selected_verifier = _default_verifier()
            except Exception as exc:
                selected_verifier = _failed_verifier(exc)
        selection = select_sar_candidate(
            command_text,
            parsed,
            detections,
            verifier=selected_verifier,
        )
        if selection.selected is None:
            failure_reason = f"sar_selection:{selection.reason}"
        metrics = selection.verifier_metrics
        log.info(
            "[qwen] selection method=%s status=%s selected=%s reason=%s "
            "images=%s latency_ms=%s request_bytes=%s",
            selection.method,
            selection.status.value,
            selection.selected.candidate_id if selection.selected else None,
            selection.reason,
            metrics.image_count if metrics else None,
            (
                round(metrics.latency_ms)
                if metrics and metrics.latency_ms is not None
                else None
            ),
            metrics.request_bytes if metrics else None,
        )
    else:
        qwen_mode = (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()
        qwen_backend = (os.environ.get("MARC_QWEN_BACKEND") or "local").strip().lower()
        log.info(
            "[qwen] skipped mode=%s backend=%s target_type=%s reason=non_sar_request "
            "(object selection is handled downstream)",
            qwen_mode,
            qwen_backend,
            parsed.target_type,
        )

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
