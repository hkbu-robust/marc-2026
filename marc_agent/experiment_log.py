"""Fail-open structured logging for one participant session."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


_SAFE_ENV = (
    "MARC_RUN_MODE",
    "MARC_BUILD_ID",
    "MARC_SAR_DETECTOR",
    "MARC_QWEN_BACKEND",
    "MARC_QWEN_MODEL_DIR",
    "MARC_QWEN_MODEL",
    "MARC_CCTV_YOLO_CONF",
    "MARC_CCTV_YOLO_DEVICE",
    "MARC_YOLO_WEIGHTS",
    "MARC_YOLO_DEVICE",
    "MARC_YOLO_CONF",
    "MARC_YOLO_IMGSZ",
    "MARC_PICK_POLICY",
    "MARC_GRASPGENX_WEIGHTS",
    "GRASPGENX_CHECKPOINT_DIR",
    "MARC_GRASPGENX_PLANNER",
    "MARC_GRASPGENX_GRIPPER",
    "ROS_DOMAIN_ID",
)
_FAILURE_STATUSES = {"error", "failed", "timeout", "unavailable", "not_implemented"}
_current = None


class ExperimentLog:
    """Own the files for a single run without affecting participant control flow."""

    def __init__(self, root: str | Path | None = None, *, environ=None):
        env = os.environ if environ is None else environ
        self.run_id = uuid.uuid4().hex[:8]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = Path(root or env.get("MARC_EXPERIMENTS_DIR", "experiments")) / (
            f"{stamp}_{self.run_id}"
        )
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.log_path = self.run_dir / "agent.log"
        self._started_monotonic = time.monotonic()
        self._lock = threading.Lock()
        self._disabled = False
        self._finished = False
        self._handler = None
        self._simulation_clock = None
        try:
            repository = Path(__file__).resolve().parent.parent
            checksums = {}
            for model in ("stage1-cctv-detector", "stage2-nearfield-detector"):
                manifest = repository / "models" / "runtime" / model / "SHA256SUMS"
                if manifest.is_file():
                    checksums[model] = manifest.read_text(encoding="utf-8").splitlines()
            (self.run_dir / "artifacts" / "stage1").mkdir(parents=True)
            for concern in ("grounding", "navigation", "nearfield", "manipulation", "delivery"):
                (self.run_dir / "artifacts" / "stage2" / concern).mkdir(parents=True)
            env["MARC_ROUND_DUMP_DIR"] = str(self.run_dir / "artifacts")
            self.event(
                "run.started",
                status="started",
                run_mode=env.get("MARC_RUN_MODE", "submission"),
                config={
                    **{name: env[name] for name in _SAFE_ENV if env.get(name)},
                    "stage1_cctv_weights": str(
                        repository / "models/runtime/stage1-cctv-detector/model.pt"
                    ),
                    "stage2_nearfield_weights": env.get("MARC_YOLO_WEIGHTS") or str(
                        repository / "models/runtime/stage2-nearfield-detector/model.pt"
                    ),
                    "graspgenx_checkpoint_dir": (
                        env.get("MARC_GRASPGENX_WEIGHTS")
                        or env.get("GRASPGENX_CHECKPOINT_DIR")
                        or str(repository / "models/runtime/graspgenx")
                    ),
                    "pick_policy": env.get("MARC_PICK_POLICY", "graspgenx"),
                    "existing_checksum_manifests": checksums,
                },
            )
        except OSError as exc:
            self._disable(exc)

    def _disable(self, exc: BaseException) -> None:
        if not self._disabled:
            print(f"[experiment-log] disabled: {exc}", file=sys.stderr)
        self._disabled = True

    def attach_python_logging(self) -> None:
        """Mirror normal Python logs and tracebacks to agent.log."""
        if self._disabled or self._handler is not None:
            return
        try:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            ))
            logging.getLogger().addHandler(handler)
            self._handler = handler
        except OSError as exc:
            self._disable(exc)

    def event(self, name: str, *, stage: str | None = None, status: str = "ok", **fields) -> None:
        if self._disabled:
            return
        payload = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_offset_s": time.monotonic() - self._started_monotonic,
            "run_id": self.run_id,
            "event": name,
            "stage": stage,
            "status": status,
            **fields,
        }
        if self._simulation_clock is not None:
            try:
                payload["simulation_timestamp"] = self._simulation_clock()
            except Exception:
                pass
        try:
            line = json.dumps(payload, sort_keys=True, default=str)
            with self._lock, self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except (OSError, TypeError, ValueError) as exc:
            self._disable(exc)

    def set_simulation_clock(self, clock) -> None:
        self._simulation_clock = clock

    @contextmanager
    def span(self, name: str, *, stage: str | None = None, **fields):
        started = time.perf_counter()
        self.event(f"{name}.started", stage=stage, status="started", **fields)
        try:
            yield
        except Exception as exc:
            self.event(
                f"{name}.completed",
                stage=stage,
                status="error",
                duration_ms=(time.perf_counter() - started) * 1000.0,
                error_type=type(exc).__name__,
                **fields,
            )
            raise
        else:
            self.event(
                f"{name}.completed",
                stage=stage,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                **fields,
            )

    def finish(self, outcome: str = "completed", **fields) -> None:
        if self._disabled or self._finished:
            return
        self.event(
            "run.completed", status=outcome,
            terminal=outcome in {"failed", "timeout"}, **fields,
        )
        try:
            events = []
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            durations = [
                event for event in events
                if isinstance(event.get("duration_ms"), (int, float))
            ]
            failures = [
                event for event in events
                if event.get("terminal") and event.get("status") in _FAILURE_STATUSES
            ]
            scores = [event for event in events if event.get("event") == "score.received"]
            scores_by_round = {
                event.get("round"): event
                for event in scores
                if event.get("round") is not None and not event.get("is_final")
            }
            stage1_rounds = [
                {**event, "score": scores_by_round.get(event.get("round"))}
                for event in events
                if event.get("event") == "stage1.round.completed"
            ]
            stage2_outcomes = [
                event for event in events
                if event.get("event") == "stage2.mission.completed"
            ]
            summary = {
                "run_id": self.run_id,
                "outcome": outcome,
                "started_at": events[0]["timestamp"] if events else None,
                "finished_at": events[-1]["timestamp"] if events else None,
                "duration_ms": (
                    events[-1].get("monotonic_offset_s", 0.0) * 1000.0
                    if events else None
                ),
                "event_count": len(events),
                "config": events[0].get("config", {}) if events else {},
                "scores": scores,
                "stage1_rounds": stage1_rounds,
                "stage2_outcome": stage2_outcomes[-1] if stage2_outcomes else None,
                "spans": durations,
                "slowest_span": max(durations, key=lambda item: item["duration_ms"], default=None),
                "first_terminal_failure": failures[0] if failures else None,
                **fields,
            }
            self.summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
            self._finished = True
            if self._handler is not None:
                logging.getLogger().removeHandler(self._handler)
                self._handler.close()
                self._handler = None
        except OSError as exc:
            self._disable(exc)


def start_experiment(root: str | Path | None = None, *, environ=None) -> ExperimentLog:
    global _current
    _current = ExperimentLog(root, environ=environ)
    return _current


def current_experiment() -> ExperimentLog | None:
    return _current
