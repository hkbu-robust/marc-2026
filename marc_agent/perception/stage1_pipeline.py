"""Round-level Stage 1 candidate collection and SAR selection composition."""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from marc_agent.command_parser import (
    CommandParseError,
    Intent,
    ParseErrorCode,
    ParsedCommand,
    Relation,
    parse_command,
)
from marc_agent.perception.cctv_project import (
    CctvProjector,
)
from marc_agent.perception.cctv_yolo import (
    refine_detection_round,
)
from marc_agent.perception.object_selection import (
    ObjectSelection,
    occupancy_lookup,
    rank_object_candidates,
    select_object_candidate,
)
from marc_agent.perception.sar_selection import (
    QwenSarVerifier,
    SarSelectionResult,
    Verifier,
    candidate_evidence,
    projection_relation,
    resolve_sar_anchor,
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
from marc_agent.experiment_log import note_decision
from marc_agent.round_clock import QwenBudget, qwen_budget
from marc_agent.types import CctvDetectionRound, CctvRoundMetrics, ObjectCandidate

if TYPE_CHECKING:
    from marc_sdk.types import GroundingResult


Detector = Callable[..., CctvDetectionRound]
log = logging.getLogger("marc_agent.stage1_pipeline")
_GroundingResult = None
_CACHED_VERIFIER: Optional[Verifier] = None

# Protocol filler so a round with no projectable candidate still submits. It is
# scene-agnostic, not a position estimate; the regression gate counts it.
IGNORED_CLASS_DUMMY_XYZ = (1.0, 1.0, 0.1)


def _grounding_result_cls():
    """Load GroundingResult without importing marc_sdk.client (rclpy) in unit tests."""
    global _GroundingResult
    if _GroundingResult is not None:
        return _GroundingResult
    try:
        from marc_sdk.types import GroundingResult as cls
    except ImportError:
        path = Path(__file__).resolve().parents[2] / "marc_sdk" / "types.py"
        spec = importlib.util.spec_from_file_location(
            "_marc_sdk_types_nors",
            path,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = module.GroundingResult
    _GroundingResult = cls
    return cls


def reset_cached_verifier() -> None:
    """Drop the process verifier so Stage 2 pick does not reload Qwen."""
    global _CACHED_VERIFIER
    verifier = _CACHED_VERIFIER
    _CACHED_VERIFIER = None
    if verifier is None:
        return
    drop = getattr(verifier, "unload", None)
    if callable(drop):
        drop()


def get_default_verifier() -> Optional[Verifier]:
    """Return the process-cached verifier, constructing it once."""
    return _default_verifier()


def log_qwen_startup_status() -> None:
    """Log configured Qwen mode; warmup local weights or probe remote vLLM."""
    mode = (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()
    if mode not in {"qwen", "qwen-remote"}:
        log.info("[vlm-startup] disabled mode=%s", mode)
        return
    try:
        backend = (
            QwenBackend.REMOTE
            if mode == "qwen-remote"
            else configured_backend()
        )
        if backend is QwenBackend.LOCAL:
            verifier = _default_verifier()
            if verifier is not None:
                warmup = getattr(verifier, "warmup", None)
                if callable(warmup):
                    warmup()
                log.info(
                    "[vlm-startup] configured backend=local model_dir=%s warmed=%s",
                    verifier.config.model_dir,
                    getattr(verifier, "_model", None) is not None,
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


def start_qwen_warmup() -> Optional[threading.Thread]:
    """Warm the remote vLLM server in the background from agent start (#111).

    Runs on its own verifier instance and only logs. Returns the thread, or
    None when Qwen is not served remotely.
    """
    mode = (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()
    if mode != "qwen-remote":
        return None

    def run() -> None:
        started = time.perf_counter()
        try:
            RemoteQwenSarVerifier().warm_up()
        except Exception as exc:
            latency_s = time.perf_counter() - started
            log.warning(
                "[qwen-warmup] failed after %.1fs error=%s detail=%s",
                latency_s, type(exc).__name__, exc,
            )
            note_decision(
                "qwen_warmup", ok=False, latency_s=latency_s, error=type(exc).__name__
            )
            return
        latency_s = time.perf_counter() - started
        log.info("[qwen-warmup] ok latency=%.1fs", latency_s)
        note_decision("qwen_warmup", ok=True, latency_s=latency_s)

    thread = threading.Thread(target=run, name="qwen-warmup", daemon=True)
    thread.start()
    return thread


@dataclass(frozen=True)
class Stage1CandidateResult:
    """Candidate-stage output; deliberately contains no world coordinates."""

    command_text: str
    parsed: Optional[ParsedCommand]
    detections: CctvDetectionRound
    sar_selection: Optional[SarSelectionResult]
    failure_reason: Optional[str]
    total_ms: float
    object_selection: Optional[ObjectSelection] = None
    parse_ms: float = 0.0
    selection_ms: float = 0.0

    @property
    def selected_candidate_id(self) -> Optional[str]:
        if self.sar_selection and self.sar_selection.selected:
            return self.sar_selection.selected.candidate_id
        if self.object_selection is not None:
            return self.object_selection.target.candidate_id
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
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "class_name": getattr(item, "class_name", None)
                    or getattr(item, "pose_class", None),
                    "camera_id": item.camera_id,
                    "confidence": item.confidence,
                    "bbox_xyxy": item.bbox_xyxy,
                }
                for item in (*self.detections.objects, *self.detections.people)
            ],
            "selected_candidate_id": self.selected_candidate_id,
            "selection_status": (
                self.sar_selection.status.value if self.sar_selection else None
            ),
            "selection_method": (
                self.sar_selection.method
                if self.sar_selection
                else (self.object_selection.method if self.object_selection else None)
            ),
            "selection_reason": self.sar_selection.reason if self.sar_selection else None,
            "selection_score": (
                self.object_selection.score if self.object_selection is not None else None
            ),
            "validated_output": (
                self.sar_selection.raw_output
                if self.sar_selection is not None
                else (
                    self.object_selection.raw_output
                    if self.object_selection is not None
                    else None
                )
            ),
            "verifier_metrics": (
                asdict(self.sar_selection.verifier_metrics)
                if self.sar_selection and self.sar_selection.verifier_metrics
                else None
            ),
            "failure_reason": self.failure_reason,
            "candidate_recall": recall,
            "metrics": asdict(self.detections.metrics),
            "parse_ms": self.parse_ms,
            "selection_ms": self.selection_ms,
            "total_ms": self.total_ms,
        }


def _default_verifier() -> Optional[Verifier]:
    global _CACHED_VERIFIER
    if _CACHED_VERIFIER is not None:
        return _CACHED_VERIFIER
    mode = (os.environ.get("MARC_SAR_DETECTOR") or "fallback").strip().lower()
    if mode not in {"qwen", "qwen-remote"}:
        log.info("[vlm] verifier disabled mode=%s", mode)
        return None
    backend = (
        QwenBackend.REMOTE
        if mode == "qwen-remote"
        else configured_backend()
    )
    if backend is QwenBackend.REMOTE:
        verifier = RemoteQwenSarVerifier()
        log.info(
            "[qwen] verifier configured backend=remote endpoint=%s model=%s "
            "timeout=%.1fs",
            verifier.config.chat_completions_url,
            verifier.config.model_name,
            verifier.config.timeout_s,
        )
        _CACHED_VERIFIER = verifier
        return verifier
    verifier = QwenSarVerifier()
    log.info(
        "[vlm] verifier configured backend=local model_dir=%s",
        verifier.config.model_dir,
    )
    _CACHED_VERIFIER = verifier
    return verifier


def _call_detector(
    detector: Detector,
    client,
    stub_path: Optional[str],
    discovery_timeout_s: Optional[float],
) -> CctvDetectionRound:
    kwargs: dict = {"stub_path": stub_path}
    try:
        parameters = inspect.signature(detector).parameters
        if "discovery_timeout_s" in parameters:
            kwargs["discovery_timeout_s"] = discovery_timeout_s
    except (TypeError, ValueError):
        pass
    return detector(client, **kwargs)


def _failed_verifier(exc: Exception) -> Verifier:
    """Delay backend setup failures until the shared fallback can classify them."""
    def verify(_board):
        raise exc

    return verify


def _now_s(client) -> float:
    getter = getattr(client, "now_s", None)
    if callable(getter):
        return float(getter())
    return time.monotonic()


def _client_occupancy(client):
    getter = getattr(client, "get_occupancy_map", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _default_timeout_s(verifier) -> Optional[float]:
    config = getattr(verifier, "config", None)
    timeout = getattr(config, "timeout_s", None)
    if timeout is None:
        return None
    return float(timeout)


def _apply_budget(verifier, budget: QwenBudget):
    if verifier is None or budget.timeout_s is None:
        return
    setter = getattr(verifier, "set_budget_timeout", None)
    if callable(setter):
        setter(budget.timeout_s)
        return
    try:
        verifier.budget_timeout_s = budget.timeout_s
    except Exception:
        pass


def _select_verifier(
    verifier: Optional[Verifier],
    clock,
    client,
) -> Optional[Verifier]:
    if verifier is not None:
        selected = verifier
    else:
        try:
            selected = _default_verifier()
        except Exception as exc:
            selected = _failed_verifier(exc)
    budget = qwen_budget(
        clock,
        _now_s(client),
        default_timeout=_default_timeout_s(selected),
    )
    if budget.skip:
        log.info(
            "[vlm] skipped reason=time_budget remaining=%.1fs",
            budget.remaining if budget.remaining is not None else -1.0,
        )
        note_decision("qwen_skipped", reason="time_budget", remaining_s=budget.remaining)
        return None
    _apply_budget(selected, budget)
    if budget.remaining is not None and budget.timeout_s is not None:
        log.info(
            "[vlm] time-budget remaining=%.1fs timeout=%.1fs capped=%s",
            budget.remaining,
            budget.timeout_s,
            budget.capped,
        )
    return selected


def run_stage1_candidate_pipeline(
    client,
    command_text: str,
    *,
    detector: Detector = detect_stage1_round,
    verifier: Optional[Verifier] = None,
    projector: Optional[CctvProjector] = None,
    stub_path: Optional[str] = None,
    discovery_timeout_s: Optional[float] = None,
    expected_intent: Intent = Intent.FIND,
    clock=None,
) -> Stage1CandidateResult:
    """Parse one request, collect all-camera candidates, and select SAR or objects."""
    started = time.perf_counter()
    parse_started = time.perf_counter()
    try:
        parsed = parse_command(command_text, expected_intent=expected_intent)
    except CommandParseError as exc:
        if exc.code is ParseErrorCode.INTENT_MISMATCH and expected_intent is Intent.FIND:
            parsed = parse_command(command_text)
        else:
            return Stage1CandidateResult(
                command_text,
                None,
                CctvDetectionRound(),
                None,
                f"command_parse:{exc.code.value}",
                (time.perf_counter() - started) * 1000.0,
                parse_ms=(time.perf_counter() - parse_started) * 1000.0,
            )
    parse_ms = (time.perf_counter() - parse_started) * 1000.0

    log.info(
        "[stage1-pipeline] parsed target_type=%s situation=%s landmark=%s relation=%s",
        parsed.target_type,
        parsed.situation.value if parsed.situation else None,
        parsed.landmark,
        parsed.relation.value if parsed.relation else None,
    )

    try:
        detections = _call_detector(
            detector, client, stub_path, discovery_timeout_s
        )
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

    detections = refine_detection_round(parsed, detections)
    selection_started = time.perf_counter()
    selected_verifier = _select_verifier(verifier, clock, client)
    prepare_scene = getattr(selected_verifier, "prepare_scene", None)
    if callable(prepare_scene):
        try:
            try:
                prepare_scene(detections, command_text)
            except TypeError:
                prepare_scene(detections)
            selected_verifier = _select_verifier(selected_verifier, clock, client)
        except Exception as exc:
            log.warning(
                "[vlm] scene description failed error=%s detail=%s; "
                "continuing with candidate images",
                type(exc).__name__,
                exc,
            )
            note_decision("scene_description_failed", error=type(exc).__name__)
    selection = None
    object_selection = None
    failure_reason = None

    if parsed.target_type == "person":
        log.info(
            "[vlm] SAR request situation=%s people=%d cameras=%d",
            parsed.situation.value if parsed.situation else None,
            len(detections.people),
            len(detections.camera_ids),
        )
        selection = select_sar_candidate(
            command_text,
            parsed,
            detections,
            verifier=selected_verifier,
            projector=projector,
        )
        if selection.selected is None:
            failure_reason = f"sar_selection:{selection.reason}"
        metrics = selection.verifier_metrics
        log.info(
            "[vlm] selection method=%s status=%s selected=%s reason=%s "
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
        object_selection = select_object_candidate(
            parsed,
            detections,
            command_text=command_text,
            verifier=selected_verifier,
            projector=projector,
            occupancy_at=occupancy_lookup(_client_occupancy(client)),
        )
        if object_selection is None:
            failure_reason = "object_selection:no_match"
        log.info(
            "[vlm] object request target_type=%s relation=%s landmark=%s "
            "method=%s selected=%s",
            parsed.target_type,
            parsed.relation.value if parsed.relation else None,
            parsed.landmark,
            object_selection.method if object_selection else None,
            object_selection.target.candidate_id if object_selection else None,
        )

    return Stage1CandidateResult(
        command_text,
        parsed,
        detections,
        selection,
        failure_reason,
        (time.perf_counter() - started) * 1000.0,
        object_selection=object_selection,
        parse_ms=parse_ms,
        selection_ms=(time.perf_counter() - selection_started) * 1000.0,
    )


@dataclass(frozen=True)
class Stage1RoundResult:
    """Candidate collection plus an optional world-coordinate GroundingResult."""

    candidates: Stage1CandidateResult
    grounding: Optional[GroundingResult]
    method: Optional[str]
    failure_reason: Optional[str]
    target_id: Optional[str] = None
    anchor_id: Optional[str] = None
    projection_ms: float = 0.0

    def record(self) -> dict:
        return {
            **self.candidates.record(),
            "duration_ms": self.candidates.total_ms + self.projection_ms,
            "projection_ms": self.projection_ms,
            "grounding_method": self.method,
            "grounding_failure_reason": self.failure_reason,
            "camera_coverage_complete": has_complete_camera_coverage(
                self.candidates.detections
            ),
        }


def static_scene_key(parsed: Optional[ParsedCommand]) -> Optional[tuple]:
    """Identity of a target request while the Stage 1 CCTV scene is static."""
    if parsed is None:
        return None
    return (
        parsed.target_type,
        parsed.landmark,
        parsed.relation.value if parsed.relation else None,
        parsed.situation.value if parsed.situation else None,
        parsed.scene_hint,
        parsed.place_cues,
    )


def has_complete_camera_coverage(detections: CctvDetectionRound) -> bool:
    """True when every discovered CCTV camera supplied a usable frame."""
    return bool(detections.camera_ids) and all(
        camera_id in detections.frames for camera_id in detections.camera_ids
    )


class StaticSceneCache:
    """Reuse the first grounding given for a request while the scene is static.

    An answer is stored only when every discovered camera sent a frame. A hit
    replaces the submitted grounding; the fresh one is recorded, not used.
    Off unless ``MARC_STAGE1_SCENE_CACHE=1``: copying an earlier round's answer
    is not realistic for deployment (#88).
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._entries: dict[tuple, tuple[Optional[int], "Stage1RoundResult"]] = {}

    @classmethod
    def from_env(cls, environ=None) -> "StaticSceneCache":
        env = os.environ if environ is None else environ
        return cls(env.get("MARC_STAGE1_SCENE_CACHE", "0") == "1")

    def apply(
        self, result: "Stage1RoundResult", round_number: Optional[int] = None
    ) -> "Stage1RoundResult":
        if not self.enabled:
            return result
        key = static_scene_key(result.candidates.parsed)
        entry = self._entries.get(key)
        if entry is not None:
            source_round, cached = entry
            log.info("[Stage1] reused static-scene grounding key=%s", key)
            _note_cache_hit(key, source_round, cached, result)
            return replace(
                result,
                grounding=cached.grounding,
                method="static_scene_cache",
                target_id=cached.target_id,
                anchor_id=cached.anchor_id,
            )
        if (
            key is not None
            and result.grounding is not None
            and has_complete_camera_coverage(result.candidates.detections)
        ):
            self._entries[key] = (round_number, result)
        return result


def _note_cache_hit(key, source_round, cached, fresh) -> None:
    cached_grounding = cached.grounding
    fresh_grounding = fresh.grounding
    note_decision(
        "cache_hit",
        key=key,
        source_round=source_round,
        cached_method=cached.method,
        cached_target=cached.target_id,
        cached_anchor=cached.anchor_id,
        cached_camera=cached_grounding.camera_id,
        cached_target_coord=cached_grounding.target_coord,
        cached_anchor_coord=cached_grounding.anchor_coord,
        fresh_method=fresh.method,
        fresh_target=fresh.target_id,
        fresh_anchor=fresh.anchor_id,
        fresh_camera=fresh_grounding.camera_id if fresh_grounding else None,
        fresh_target_coord=fresh_grounding.target_coord if fresh_grounding else None,
        fresh_anchor_coord=fresh_grounding.anchor_coord if fresh_grounding else None,
        same_pair=(
            fresh.target_id == cached.target_id
            and fresh.anchor_id == cached.anchor_id
        ),
        target_shift_m=(
            math.dist(fresh_grounding.target_coord, cached_grounding.target_coord)
            if fresh_grounding is not None
            else None
        ),
    )


def _xyz_list(xyz: tuple[float, float, float]) -> list[float]:
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def project_anchor(
    projector: CctvProjector,
    camera_id: str,
    anchor_box: tuple[float, float, float, float],
) -> Optional[tuple[float, float, float]]:
    """World point submitted for an anchor (landmark) box.

    The offline replay (``scripts/stage1_replay.py``) scores held-out anchors
    through this function, so anchor geometry changes belong here.
    """
    return projector.project_contact(camera_id, anchor_box)


def _project_pair(
    projector: CctvProjector,
    camera_id: str,
    target_box: tuple[float, float, float, float],
    anchor_box: Optional[tuple[float, float, float, float]],
    *,
    relation: Optional[Relation],
    support_class: Optional[str],
    pose_class: Optional[str] = None,
) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    target_xyz = projector.project_contact(
        camera_id,
        target_box,
        relation=relation,
        support_class=support_class,
        pose_class=pose_class,
        support_box=anchor_box,
    )
    if target_xyz is None:
        return None
    if anchor_box is None:
        return target_xyz, target_xyz
    anchor_xyz = project_anchor(projector, camera_id, anchor_box)
    if anchor_xyz is None:
        note_decision("anchor_collapsed", camera_id=camera_id, anchor_box=anchor_box)
        return target_xyz, target_xyz
    return target_xyz, anchor_xyz


def _grounding_payload(
    parsed: ParsedCommand,
    camera_id: str,
    target_xyz: tuple[float, float, float],
    anchor_xyz: tuple[float, float, float],
    *,
    selection: Optional[ObjectSelection] = None,
    fallback_class: Optional[str] = None,
    relation: Optional[Relation] = None,
    anchor_class: Optional[str] = None,
) -> GroundingResult:
    del fallback_class
    implied_anchor = (
        selection is not None
        and selection.anchor is not None
        and (parsed.landmark is not None or parsed.scene_hint is not None)
    )
    landmark = parsed.landmark or anchor_class or (
        selection.anchor.class_name
        if implied_anchor
        else ""
    )
    submitted_relation = relation if relation is not None else parsed.relation
    if submitted_relation is None and implied_anchor:
        submitted_relation = (
            Relation.BESIDE if parsed.scene_hint == "park" else Relation.NEAR
        )
    return _grounding_result_cls()(
        camera_id=camera_id,
        target_type=parsed.target_type,
        anchor_coord=_xyz_list(anchor_xyz),
        target_coord=_xyz_list(target_xyz),
        landmark=landmark,
        relation=submitted_relation.value if submitted_relation else None,
        situation=parsed.situation.value if parsed.situation else None,
    )


def _compose_person(
    candidates: Stage1CandidateResult,
    projector: CctvProjector,
) -> Stage1RoundResult:
    parsed = candidates.parsed
    selection = candidates.sar_selection
    if parsed is None:
        reason = candidates.failure_reason or "command_parse"
        log.info("[stage1] skip submit reason=%s", reason)
        return Stage1RoundResult(candidates, None, None, reason)
    if selection is None or selection.selected is None:
        return _compose_last_resort(
            candidates, projector, parsed, "no_person_selected"
        )

    person = selection.selected
    evidence = candidate_evidence(
        parsed, candidates.detections, command_text=candidates.command_text
    )
    item = next(
        (row for row in evidence if row.person.candidate_id == person.candidate_id),
        None,
    )
    anchor = item.anchor if item is not None else None
    if anchor is None:
        anchor = resolve_sar_anchor(parsed, person, candidates.detections)
    projected = _project_pair(
        projector,
        person.camera_id,
        person.bbox_xyxy,
        anchor.bbox_xyxy if anchor is not None else None,
        relation=projection_relation(
            parsed, candidates.command_text, person.pose_class
        ),
        support_class=anchor.class_name if anchor is not None else None,
        pose_class=person.pose_class,
    )
    if projected is None:
        log.info(
            "[stage1] projection_failed camera=%s candidate=%s; using dummy",
            person.camera_id,
            person.candidate_id,
        )
        return _compose_last_resort(
            candidates, projector, parsed, "person_projection_failed"
        )
    target_xyz, anchor_xyz = projected
    grounding = _grounding_payload(
        parsed,
        person.camera_id,
        target_xyz,
        anchor_xyz,
        anchor_class=anchor.class_name if anchor is not None else None,
    )
    log.info(
        "[stage1] grounding camera=%s target_id=%s anchor_id=%s method=%s "
        "situation=%s target_xyz=(%.3f, %.3f, %.3f) anchor_xyz=(%.3f, %.3f, %.3f)",
        grounding.camera_id,
        person.candidate_id,
        anchor.candidate_id if anchor is not None else person.candidate_id,
        selection.method,
        grounding.situation,
        *grounding.target_coord,
        *grounding.anchor_coord,
    )
    return Stage1RoundResult(
        candidates,
        grounding,
        selection.method,
        None,
        person.candidate_id,
        anchor.candidate_id if anchor is not None else None,
    )


def _project_selection(
    projector: CctvProjector,
    parsed: ParsedCommand,
    selection: ObjectSelection,
    objects: tuple[ObjectCandidate, ...] | list[ObjectCandidate] = (),
) -> Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    target_box = selection.target.bbox_xyxy
    width, height = selection.target.image_size
    clipped = (
        target_box[0] <= 1
        or target_box[1] <= 1
        or target_box[2] >= width - 1
        or target_box[3] >= height - 1
    )
    if (
        parsed.target_type == "umbrella"
        and selection.target.confidence < 0.02
        and selection.anchor is not None
        and selection.anchor.class_name == "trash_can"
    ):
        target_box = selection.anchor.bbox_xyxy
        log.info(
            "[stage1] hidden umbrella proxy target=%s proxy=%s",
            selection.target.candidate_id,
            selection.anchor.candidate_id,
        )
        note_decision(
            "projection_proxy", variant="hidden_umbrella",
            target=selection.target.candidate_id,
            proxy=selection.anchor.candidate_id,
        )
    elif selection.target.confidence < 0.02 and clipped and selection.anchor is not None:
        ax1, ay1, ax2, ay2 = selection.anchor.bbox_xyxy
        ac = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
        proxies = [
            item
            for item in objects
            if item.camera_id == selection.target.camera_id
            and item.candidate_id != selection.anchor.candidate_id
            and item.class_name != parsed.target_type
            and item.confidence >= 0.2
        ]
        if proxies:
            proxy = min(
                proxies,
                key=lambda item: (
                    math.hypot(
                        (item.bbox_xyxy[0] + item.bbox_xyxy[2]) / 2.0 - ac[0],
                        (item.bbox_xyxy[1] + item.bbox_xyxy[3]) / 2.0 - ac[1],
                    ),
                    item.candidate_id,
                ),
            )
            px1, py1, px2, py2 = proxy.bbox_xyxy
            if target_box[2] >= width - 1:
                target_box = (px1, py1, px1, py2)
            elif target_box[0] <= 1:
                target_box = (px2, py1, px2, py2)
            elif target_box[3] >= height - 1:
                target_box = (px1, py1, px2, py1)
            else:
                target_box = (px1, py2, px2, py2)
            log.info(
                "[stage1] clipped target proxy target=%s proxy=%s",
                selection.target.candidate_id,
                proxy.candidate_id,
            )
            note_decision(
                "projection_proxy", variant="clipped_target",
                target=selection.target.candidate_id, proxy=proxy.candidate_id,
            )
    elif (
        parsed.target_type == "umbrella"
        and (target_box[2] - target_box[0]) * (target_box[3] - target_box[1]) <= 400
    ):
        tx = (target_box[0] + target_box[2]) / 2.0
        ty = (target_box[1] + target_box[3]) / 2.0
        adjacent = [
            item
            for item in objects
            if item.camera_id == selection.target.camera_id
            and item.class_name != parsed.target_type
            and (
                selection.anchor is None
                or item.candidate_id != selection.anchor.candidate_id
            )
            and item.confidence >= 0.5
            and item.bbox_xyxy[0] - 5 <= tx <= item.bbox_xyxy[2] + 5
            and item.bbox_xyxy[1] - 5 <= ty <= item.bbox_xyxy[3] + 5
        ]
        if adjacent:
            proxy = max(
                adjacent,
                key=lambda item: (
                    (item.bbox_xyxy[2] - item.bbox_xyxy[0])
                    * (item.bbox_xyxy[3] - item.bbox_xyxy[1]),
                    item.confidence,
                ),
            )
            target_box = proxy.bbox_xyxy
            log.info(
                "[stage1] occluded target proxy target=%s proxy=%s",
                selection.target.candidate_id,
                proxy.candidate_id,
            )
            note_decision(
                "projection_proxy", variant="occluded_target",
                target=selection.target.candidate_id, proxy=proxy.candidate_id,
            )
    return _project_pair(
        projector,
        selection.target.camera_id,
        target_box,
        selection.anchor.bbox_xyxy if selection.anchor is not None else None,
        relation=parsed.relation or (
            Relation.NEAR
            if selection.anchor is not None
            and (parsed.landmark is not None or parsed.scene_hint is not None)
            else None
        ),
        support_class=(
            selection.anchor.class_name if selection.anchor is not None else None
        ),
    )


def _same_object_pair(left: ObjectSelection, right: ObjectSelection) -> bool:
    left_anchor = left.anchor.candidate_id if left.anchor is not None else None
    right_anchor = right.anchor.candidate_id if right.anchor is not None else None
    return (
        left.target.candidate_id == right.target.candidate_id
        and left_anchor == right_anchor
    )


def _compose_object(
    candidates: Stage1CandidateResult,
    projector: CctvProjector,
) -> Stage1RoundResult:
    parsed = candidates.parsed
    if parsed is None:
        reason = candidates.failure_reason or "command_parse"
        log.info("[stage1] skip submit reason=%s", reason)
        return Stage1RoundResult(candidates, None, None, reason)

    chosen = candidates.object_selection
    ranked = rank_object_candidates(
        parsed,
        candidates.detections,
        projector=projector,
        command_text=candidates.command_text,
    )
    ordered: list[ObjectSelection] = []
    if chosen is not None:
        ordered.append(chosen)
    for item in ranked:
        if chosen is not None and _same_object_pair(item, chosen):
            continue
        ordered.append(item)
    if not ordered:
        return _compose_last_resort(candidates, projector, parsed, "no_object_pairs")

    for rank, selection in enumerate(ordered):
        projected = _project_selection(
            projector, parsed, selection, candidates.detections.objects
        )
        if projected is None:
            continue
        if rank > 0:
            first = ordered[0]
            note_decision(
                "projection_fallthrough",
                failed_target=first.target.candidate_id,
                failed_anchor=(
                    first.anchor.candidate_id if first.anchor is not None else None
                ),
                used_target=selection.target.candidate_id,
                used_anchor=(
                    selection.anchor.candidate_id
                    if selection.anchor is not None
                    else None
                ),
                rank=rank,
            )
        target_xyz, anchor_xyz = projected
        grounding = _grounding_payload(
            parsed, selection.target.camera_id, target_xyz, anchor_xyz,
            selection=selection,
        )
        anchor_id = (
            selection.anchor.candidate_id if selection.anchor is not None else None
        )
        log.info(
            "[stage1] grounding camera=%s target_id=%s anchor_id=%s method=%s "
            "relation=%s landmark=%s target_xyz=(%.3f, %.3f, %.3f) "
            "anchor_xyz=(%.3f, %.3f, %.3f)",
            grounding.camera_id,
            selection.target.candidate_id,
            anchor_id or selection.target.candidate_id,
            selection.method,
            grounding.relation,
            grounding.landmark,
            *grounding.target_coord,
            *grounding.anchor_coord,
        )
        return Stage1RoundResult(
            candidates,
            grounding,
            selection.method,
            None,
            selection.target.candidate_id,
            anchor_id,
        )

    log.info("[stage1] projection_failed; using dummy target_type=%s", parsed.target_type)
    return _compose_last_resort(
        candidates, projector, parsed, "object_projection_failed"
    )


def _dummy_camera_id(detections: CctvDetectionRound, parsed: ParsedCommand) -> str:
    landmark = parsed.landmark
    if landmark:
        for camera_id in detections.camera_ids:
            if any(
                item.camera_id == camera_id and item.class_name == landmark
                for item in detections.objects
            ):
                return camera_id
        for item in detections.objects:
            if item.class_name == landmark:
                return item.camera_id
    if detections.camera_ids:
        return detections.camera_ids[0]
    if detections.frames:
        return next(iter(detections.frames))
    return ""


def _compose_last_resort(
    candidates: Stage1CandidateResult,
    projector: CctvProjector,
    parsed: ParsedCommand,
    reason: str,
) -> Stage1RoundResult:
    """Submit parser fields with a dummy XYZ so the round always ends.

    Does not invent a detection box. A picnic table is not a pencil case.
    """
    del projector
    camera_id = _dummy_camera_id(candidates.detections, parsed)
    note_decision(
        "sentinel", reason=reason, camera_id=camera_id,
        target_type=parsed.target_type,
    )
    xyz = IGNORED_CLASS_DUMMY_XYZ
    grounding = _grounding_payload(
        parsed,
        camera_id,
        xyz,
        xyz,
    )
    log.info(
        "[stage1] grounding camera=%s target_id=%s method=ignored_class_dummy "
        "relation=%s landmark=%s target_xyz=(%.3f, %.3f, %.3f)",
        grounding.camera_id,
        parsed.target_type,
        grounding.relation,
        grounding.landmark,
        *grounding.target_coord,
    )
    return Stage1RoundResult(
        candidates,
        grounding,
        "ignored_class_dummy",
        None,
        None,
        None,
    )


def compose_stage1_grounding(
    candidates: Stage1CandidateResult,
    projector: CctvProjector,
) -> Stage1RoundResult:
    """Project a selected candidate into a GroundingResult, or dummy XYZ."""
    parsed = candidates.parsed
    if parsed is None:
        reason = candidates.failure_reason or "command_parse"
        return Stage1RoundResult(candidates, None, None, reason)
    if parsed.target_type == "person":
        return _compose_person(candidates, projector)
    return _compose_object(candidates, projector)


def run_stage1_round(
    client,
    command_text: str,
    *,
    detector: Detector = detect_stage1_round,
    verifier: Optional[Verifier] = None,
    projector: Optional[CctvProjector] = None,
    stub_path: Optional[str] = None,
    discovery_timeout_s: Optional[float] = None,
    expected_intent: Intent = Intent.FIND,
    clock=None,
) -> Stage1RoundResult:
    """Parse, detect, select, and project one Stage 1 command into world coordinates."""
    if projector is None:
        projector = CctvProjector.from_client(client)
    candidates = run_stage1_candidate_pipeline(
        client,
        command_text,
        detector=detector,
        verifier=verifier,
        projector=projector,
        stub_path=stub_path,
        discovery_timeout_s=discovery_timeout_s,
        expected_intent=expected_intent,
        clock=clock,
    )
    projection_started = time.perf_counter()
    result = compose_stage1_grounding(candidates, projector)
    return replace(
        result,
        projection_ms=(time.perf_counter() - projection_started) * 1000.0,
    )


def append_stage1_record(
    result: Stage1CandidateResult,
    path: str | Path,
    *,
    expected_candidate_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """Append one structured round record for latency, recall, and failure audits."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result.record(expected_candidate_id=expected_candidate_id)
    if extra:
        payload.update(extra)
    with destination.open("a", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")
