"""World-pose feedback guard for stationary mobile-manipulation phases."""

from __future__ import annotations

import math
import os
import threading
from typing import Callable, Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, limit: float) -> float:
    bound = max(0.0, float(limit))
    return max(-bound, min(bound, float(value)))


def _angle_error(target: float, current: float) -> float:
    return math.atan2(math.sin(target - current), math.cos(target - current))


class StationaryBase:
    """Hold the chassis at its entry pose while a blocking pick runs.

    A zero ``cmd_vel`` alone does not prevent gravity-driven rolling on the trainer's
    sloped surfaces. The guard therefore anchors the live world pose and applies a
    small bounded longitudinal/yaw correction. If world pose is unavailable, it keeps
    the previous zero-command behaviour.
    """

    def __init__(
        self,
        client,
        *,
        interval_s: float = 0.05,
        log: Optional[Callable[[str], None]] = None,
        position_kp: Optional[float] = None,
        max_linear_mps: Optional[float] = None,
        position_deadband_m: Optional[float] = None,
        yaw_kp: Optional[float] = None,
        max_angular_rps: Optional[float] = None,
        yaw_deadband_rad: Optional[float] = None,
    ):
        self._client = client
        self._interval_s = max(0.01, float(interval_s))
        self._log = log
        self._position_kp = max(
            0.0,
            float(
                position_kp
                if position_kp is not None
                else _env_float("MARC_BASE_HOLD_POS_KP", 2.0)
            ),
        )
        self._max_linear_mps = max(
            0.0,
            float(
                max_linear_mps
                if max_linear_mps is not None
                else _env_float("MARC_BASE_HOLD_MAX_MPS", 0.08)
            ),
        )
        self._position_deadband_m = max(
            0.0,
            float(
                position_deadband_m
                if position_deadband_m is not None
                else _env_float("MARC_BASE_HOLD_DEADBAND_M", 0.002)
            ),
        )
        self._yaw_kp = max(
            0.0,
            float(
                yaw_kp
                if yaw_kp is not None
                else _env_float("MARC_BASE_HOLD_YAW_KP", 1.5)
            ),
        )
        self._max_angular_rps = max(
            0.0,
            float(
                max_angular_rps
                if max_angular_rps is not None
                else _env_float("MARC_BASE_HOLD_MAX_RADPS", 0.15)
            ),
        )
        self._yaw_deadband_rad = max(
            0.0,
            float(
                yaw_deadband_rad
                if yaw_deadband_rad is not None
                else _env_float("MARC_BASE_HOLD_YAW_DEADBAND_RAD", 0.01)
            ),
        )
        self._anchor: Optional[tuple[float, float, float]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _publish_stop(self) -> None:
        self._client.send_cmd_vel(linear_x=0.0, angular_z=0.0)

    def _read_pose(self) -> Optional[tuple[float, float, float]]:
        getter = getattr(self._client, "get_world_pose", None)
        if not callable(getter):
            return None
        pose = getter()
        if pose is None or len(pose) < 3:
            return None
        result = (float(pose[0]), float(pose[1]), float(pose[2]))
        return result if all(math.isfinite(value) for value in result) else None

    def _correction(self, pose) -> tuple[float, float]:
        if self._anchor is None or pose is None:
            return 0.0, 0.0
        anchor_x, anchor_y, anchor_yaw = self._anchor
        x, y, yaw = pose
        # Only correct displacement along the chassis heading. The platform exposes
        # forward and yaw velocity, not a lateral command, and turning to chase a tiny
        # lateral error during a grasp would disturb the camera composition.
        dx = anchor_x - float(x)
        dy = anchor_y - float(y)
        forward_error = dx * math.cos(float(yaw)) + dy * math.sin(float(yaw))
        yaw_error = _angle_error(anchor_yaw, float(yaw))
        linear = 0.0
        angular = 0.0
        if abs(forward_error) > self._position_deadband_m:
            linear = _clamp(
                self._position_kp * forward_error,
                self._max_linear_mps,
            )
        if abs(yaw_error) > self._yaw_deadband_rad:
            angular = _clamp(self._yaw_kp * yaw_error, self._max_angular_rps)
        return linear, angular

    def _publish_hold(self) -> None:
        pose = self._read_pose()
        linear, angular = self._correction(pose)
        self._client.send_cmd_vel(linear_x=linear, angular_z=angular)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._publish_hold()
            except Exception as exc:
                if self._log is not None:
                    self._log(f"[base-hold] pose correction failed: {exc}")
                return

    def start(self) -> None:
        if self.active:
            return
        self._stop_event.clear()
        self._anchor = self._read_pose()
        self._publish_hold()
        self._thread = threading.Thread(
            target=self._run,
            name="stationary-base-hold",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        thread = self._thread
        if thread is not None:
            self._stop_event.set()
            thread.join(timeout=max(0.2, 4.0 * self._interval_s))
            self._thread = None
        self._anchor = None
        self._publish_stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.stop()
        return False
