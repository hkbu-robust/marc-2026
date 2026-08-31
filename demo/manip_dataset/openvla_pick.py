"""Deterministic-to-learned OpenVLA handoff with visual grasp verification."""

from __future__ import annotations

import math
import os

import numpy as np

from marc_agent.openvla_actions import ACTION_DIM, gripper_should_close
from marc_agent.pick_runtime import PickRuntime


DEFAULT_VERIFY_MIN_RISE_M = 0.035
DEFAULT_VERIFY_MAX_XY_DRIFT_M = 0.12
OPENVLA_EE_FWD_MIN_M = 0.32


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def openvla_stage_goal(target, cfg, current_fwd=None) -> np.ndarray:
    """Pre-grasp point after guaranteeing forward clearance from the camera."""
    fwd = float(target.fwd) - cfg.openvla_stage_back_m
    if current_fwd is not None:
        fwd = max(fwd, float(current_fwd) + cfg.openvla_stage_min_forward_m)
    return np.array(
        [
            fwd,
            float(target.lat),
            float(target.z) + cfg.openvla_stage_up_m,
        ],
        dtype=np.float64,
    )


def openvla_stage_waypoints(current, target, cfg):
    """Forward clearance and lateral alignment at ready height, then descent."""
    current = np.asarray(current, dtype=np.float64).reshape(3)
    goal = openvla_stage_goal(target, cfg, current_fwd=float(current[0]))
    clear_fwd = float(goal[0])
    clear_z = max(float(current[2]), -0.08)
    return (
        ("stage-forward", np.array([clear_fwd, current[1], clear_z])),
        ("stage-align", np.array([clear_fwd, goal[1], clear_z])),
        ("stage-view", goal),
    )


def graspgenx_stage_goal(target, cfg) -> np.ndarray:
    """Object-centred overhead handoff for geometry-driven grasping."""
    return np.array(
        [
            float(target.fwd),
            float(target.lat),
            float(target.z) + cfg.openvla_stage_up_m,
        ],
        dtype=np.float64,
    )


def graspgenx_stage_waypoints(current, target, cfg):
    """Clear the base at ready height, then descend directly above the object."""
    current = np.asarray(current, dtype=np.float64).reshape(3)
    goal = graspgenx_stage_goal(target, cfg)
    clear_z = max(float(current[2]), -0.08)
    return (
        ("stage-forward", np.array([goal[0], current[1], clear_z])),
        ("stage-align", np.array([goal[0], goal[1], clear_z])),
        ("stage-view", goal),
    )


def openvla_target_is_safe(target) -> tuple[bool, str]:
    """Accept any finite localization; IK determines whether its stage is reachable."""
    try:
        xyz = np.array([target.fwd, target.lat, target.z], dtype=np.float64)
    except (AttributeError, TypeError, ValueError):
        return False, "target has no finite arm-base pose"
    if not np.all(np.isfinite(xyz)):
        return False, "target has no finite arm-base pose"
    return True, "ok"


def openvla_ee_is_safe(xyz) -> tuple[bool, str]:
    """Keep only the measured forward base/camera collision boundary."""
    p = np.asarray(xyz, dtype=np.float64).reshape(-1)
    if p.size < 3 or not np.all(np.isfinite(p[:3])):
        return False, "missing end-effector pose"
    if p[0] < OPENVLA_EE_FWD_MIN_M:
        return False, "end effector forward %.3fm breaches base/camera keep-out" % p[0]
    return True, "ok"


def target_moved_with_gripper(
    before,
    after,
    *,
    min_rise_m: float = DEFAULT_VERIFY_MIN_RISE_M,
    max_xy_drift_m: float = DEFAULT_VERIFY_MAX_XY_DRIFT_M,
) -> bool:
    """Return true only when the localized object rose without a large XY jump."""
    if after is None or getattr(after, "valid", False) is not True:
        return False
    rise = float(after.z) - float(before.z)
    xy_drift = math.hypot(
        float(after.fwd) - float(before.fwd),
        float(after.lat) - float(before.lat),
    )
    return rise >= float(min_rise_m) and xy_drift <= float(max_xy_drift_m)


class StagedOpenVLAPickRuntime(PickRuntime):
    """Keep global approach deterministic; give OpenVLA only the final grasp phase."""

    def __init__(self, *args, target_observer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_observer = target_observer
        self._visual_grasp_verified = False
        self._verify_min_rise_m = max(
            0.01,
            _env_float("MARC_OPENVLA_VERIFY_MIN_RISE_M", DEFAULT_VERIFY_MIN_RISE_M),
        )
        self._verify_max_xy_drift_m = max(
            0.02,
            _env_float(
                "MARC_OPENVLA_VERIFY_MAX_XY_DRIFT_M",
                DEFAULT_VERIFY_MAX_XY_DRIFT_M,
            ),
        )

    def _grasp(self, target) -> None:
        self._visual_grasp_verified = False
        safe, why = openvla_target_is_safe(target)
        if not safe:
            raise RuntimeError("OpenVLA target invalid: %s" % why)
        arm = self._teleop_arm()
        if not arm.stage_for_openvla(target):
            raise RuntimeError("deterministic OpenVLA staging failed")
        self._emit(
            "[openvla] handoff: staged arm is visible; "
            "policy_input=base_left; policy now controls final grasp"
        )
        super()._grasp(target)

    def _observe_target(self):
        if not callable(self._target_observer):
            return None
        try:
            return self._target_observer()
        except Exception as exc:
            self._emit("[openvla] visual grasp check failed (%s)" % exc)
            return None

    def _verify_candidate_grasp(self, before) -> bool:
        arm = self._teleop_arm()
        if not arm.lift_for_grasp_verification():
            self._emit("[openvla] grasp rejected: verification lift failed")
            return False
        after = self._observe_target()
        valid = target_moved_with_gripper(
            before,
            after,
            min_rise_m=self._verify_min_rise_m,
            max_xy_drift_m=self._verify_max_xy_drift_m,
        )
        if after is None or getattr(after, "valid", False) is not True:
            self._emit("[openvla] grasp rejected: target not visible after verification lift")
            return False
        rise = float(after.z) - float(before.z)
        drift = math.hypot(
            float(after.fwd) - float(before.fwd),
            float(after.lat) - float(before.lat),
        )
        self._emit(
            "[openvla] visual grasp %s rise=%.3fm xy_drift=%.3fm"
            % ("verified" if valid else "rejected", rise, drift)
        )
        return valid

    def _reset_after_false_hold(self, target) -> bool:
        arm = self._teleop_arm()
        observed = self._observe_target()
        restage_target = (
            observed
            if observed is not None and getattr(observed, "valid", False) is True
            else target
        )
        self._emit("[openvla] hold sensor was a false positive; opening and restaging")
        return arm.stage_for_openvla(restage_target)

    def _geom_or_openvla_grasp(self, target, policy_obj) -> None:
        """Run OpenVLA without accepting an unverified gripper contact signal."""
        arm = self._teleop_arm()
        close_cmd = float(arm.cfg.grip_close)
        for _ in range(max(1, self.policy_ticks)):
            obs = self._snapshot_obs(target)
            action = np.asarray(policy_obj.act(obs), dtype=np.float32).reshape(-1)
            if action.size < ACTION_DIM:
                action = np.pad(action, (0, ACTION_DIM - int(action.size)))
            if self.policy == "openvla_remote":
                limits = np.asarray(
                    (
                        arm.cfg.dx,
                        arm.cfg.dy,
                        arm.cfg.dz,
                        arm.cfg.d_yaw,
                        arm.cfg.d_pitch,
                        arm.cfg.d_yaw,
                    ),
                    dtype=np.float32,
                )
                action[:6] = np.clip(action[:6], -limits, limits)
            ee = getattr(obs, "ee_xyz", None)
            proposed = None if ee is None else np.asarray(ee, dtype=np.float64) + action[:3]
            safe, why = openvla_ee_is_safe(proposed)
            if not safe:
                self._emit("[openvla] SAFETY STOP: %s" % why)
                return
            arm.apply_delta(
                dx=float(action[0]),
                dy=float(action[1]),
                dz=float(action[2]),
                droll=float(action[3]),
                dpitch=float(action[4]),
                dyaw=float(action[5]),
                grip=None,
            )
            open_is_high = getattr(policy_obj, "stats", None) is not None
            if not gripper_should_close(float(action[6]), open_is_high=open_is_high):
                continue
            arm.apply_delta(grip=close_cmd)
            try:
                holding = self.client.is_grasping() is True
            except Exception:
                holding = False
            if not holding:
                continue
            if self._verify_candidate_grasp(target):
                self._visual_grasp_verified = True
                return
            if not self._reset_after_false_hold(target):
                return

    def poll_grasping(self):
        held = super().poll_grasping()
        if held is True and not self._visual_grasp_verified:
            self._emit("[openvla] rejecting hold=True without visual lift evidence")
            return False
        return held

    def _recover(self) -> None:
        """Use the known joint-space ready pose, not another unconstrained DLS path."""
        try:
            ok = self._teleop_arm().go_ready(open_grip=True)
            self._emit("[openvla] safe recovery -> ready (%s)" % ("OK" if ok else "timeout"))
        except Exception as exc:
            self._emit("[openvla] safe recovery failed: %s" % exc)
        finally:
            self._teleop = None
