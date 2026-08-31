"""Downward ready pose and EE-delta click teleop via arm_kin DLS."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

import arm_kin
from arm_pick import (
    ARM_JOINT_NAMES,
    GRIPPER_CLOSE,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN,
    _make_js,
    _URDF,
    _EE_FRAME,
    load_params,
)

from .images import image_to_rgb


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# The SDK exposes both stereo rigs and depth, but vanilla OpenVLA fine-tuning loads
# one primary RGB view only. Collection therefore touches and stores only that view.
RGB_CAMERAS = ("base_left", "base_right", "gripper_left", "gripper_right")
# Accept the old short names so existing .env files keep working.
_CAMERA_ALIASES = {"base": "base_left", "gripper": "gripper_left"}


def _env_camera(name: str, default: str) -> str:
    raw = (os.environ.get(name) or "").strip().lower()
    raw = _CAMERA_ALIASES.get(raw, raw)
    if raw in RGB_CAMERAS:
        return raw
    if raw:
        print(f"[arm] {name}={raw!r} invalid (want one of {RGB_CAMERAS}); using {default!r}")
    return default


# Single source of truth for TeleopConfig defaults (override with MARC_* env vars).
DEFAULT_TELEOP_DX = 0.01
DEFAULT_TELEOP_DY = 0.01
DEFAULT_TELEOP_DZ = 0.01
DEFAULT_TELEOP_DYAW = 0.05
DEFAULT_TELEOP_DPITCH = 0.05
DEFAULT_READY_PITCH = 0.40  # unused for ready (joint-space start); kept for env compat
DEFAULT_READY_J1 = 0.0
DEFAULT_READY_J2 = float(arm_kin.DEFAULT_ARM_Q[1])   # typically  0.262
DEFAULT_READY_J3 = float(arm_kin.DEFAULT_ARM_Q[2])   # typically -2.094
DEFAULT_READY_J4 = float(arm_kin.DEFAULT_ARM_Q[3])   # typically  0.0
DEFAULT_READY_J5 = float(arm_kin.DEFAULT_ARM_Q[4])   # typically -0.785
DEFAULT_READY_J6 = float(arm_kin.DEFAULT_ARM_Q[5])   # typically  1.571
DEFAULT_READY_FWD = 0.30   # unused for ready (joint-space); kept for env compat
DEFAULT_READY_LAT = 0.0
DEFAULT_READY_UP = 0.28
DEFAULT_RATE_HZ = 15.0
DEFAULT_POS_STEP = 0.008
DEFAULT_SETTLE_TOL_M = 0.015
DEFAULT_SETTLE_TOL_JOINT = 0.05
DEFAULT_READY_MOVE_S = 2.0
DEFAULT_MAX_TICKS_PER_CLICK = 40
DEFAULT_IK_JOINT_STEP = 0.04  # rad/tick; bounds DLS speed while avoiding branch jumps
DEFAULT_GRIP_DWELL_S = 1.0  # fingers need ~1s to travel before holding can be observed
# Closing jumps to the configured target by default. A positive MARC_GRIP_RAMP restores
# interpolation; the optional sensor-stop mode can then confirm and stop on a hold.
DEFAULT_GRIP_RAMP_S = 0.0
DEFAULT_GRIP_EXTRA = 0.04    # rad of extra bite past confirmed contact (0 = stop dead)
DEFAULT_GRIP_CONFIRM_SAMPLES = 3  # reject one-frame holding sensor spikes
DEFAULT_GRIP_STOP_ON_GRASP = False
DEFAULT_READY_OPEN_GRIP = False  # ready pose holds whatever the gripper was commanded to
# Basket sits behind the arm. Retract to a compact central pose before joint-1
# rotates so the entire yaw sweep stays inside the chassis footprint.
DEFAULT_PLACE_YAW = float(np.pi)
DEFAULT_PLACE_APEX_FWD = 0.14
DEFAULT_PLACE_APEX_UP = 0.40
DEFAULT_PLACE_SWEEP_RADIUS_M = 0.22
DEFAULT_PLACE_LINK_PAD_M = 0.06
DEFAULT_PLACE_SPEED = 0.5    # rad/s per joint; joint 1's URDF limit is 1.396
DEFAULT_PLACE_IK_GUARD = 1.2  # rad any joint may move from its IK seed before the
                              # solution is treated as a branch jump and refused
ARM_JOINT_LOWER = np.array([-6.2832, -2.24, -2.57, -6.2832, -2.09, -6.2832])
ARM_JOINT_UPPER = np.array([6.2832, 2.24, 2.57, 6.2832, 2.09, 6.2832])
PLACE_SWEEP_FRAMES = (
    "shoulder_link",
    "bicep_link",
    "forearm_link",
    "spherical_wrist_1_link",
    "spherical_wrist_2_link",
    "bracelet_link",
    "end_effector_link",
    "left_inner_finger_pad",
    "right_inner_finger_pad",
)
# The overhead fold is several seconds of ticks. Recording all of them would bury
# the part a policy has to learn (approach and grasp), so scripted joint moves log
# every Nth tick. The release dwell is never decimated.
DEFAULT_PLACE_REC_EVERY = 3
# OpenVLA pretrains on third-person views, and the base camera is static during
# ARM_TELEOP (base is parked), so only the arm and object move in frame.
DEFAULT_PRIMARY_CAM = "base_left"
# Put the open gripper above and slightly behind the localized object before a
# learned policy receives control.  This gives the base camera a repeatable
# arm/object composition and leaves the contact-rich final approach to OpenVLA.
DEFAULT_OPENVLA_STAGE_BACK_M = 0.01
DEFAULT_OPENVLA_STAGE_MIN_FORWARD_M = 0.16
DEFAULT_OPENVLA_STAGE_UP_M = 0.225
DEFAULT_OPENVLA_VERIFY_LIFT_M = 0.08


@dataclass
class TeleopConfig:
    dx: float = DEFAULT_TELEOP_DX
    dy: float = DEFAULT_TELEOP_DY
    dz: float = DEFAULT_TELEOP_DZ
    d_yaw: float = DEFAULT_TELEOP_DYAW
    d_pitch: float = DEFAULT_TELEOP_DPITCH
    ready_pitch_down: float = DEFAULT_READY_PITCH
    # Ready start pose = absolute joint angles (rad). No Cartesian IK during ready.
    ready_j1: float = DEFAULT_READY_J1
    ready_j2: float = DEFAULT_READY_J2
    ready_j3: float = DEFAULT_READY_J3
    ready_j4: float = DEFAULT_READY_J4
    ready_j5: float = DEFAULT_READY_J5
    ready_j6: float = DEFAULT_READY_J6
    ready_fwd: float = DEFAULT_READY_FWD
    ready_lat: float = DEFAULT_READY_LAT
    ready_up: float = DEFAULT_READY_UP
    rate_hz: float = DEFAULT_RATE_HZ
    pos_step: float = DEFAULT_POS_STEP
    settle_tol_m: float = DEFAULT_SETTLE_TOL_M
    settle_tol_joint: float = DEFAULT_SETTLE_TOL_JOINT
    ready_move_s: float = DEFAULT_READY_MOVE_S
    max_ticks_per_click: int = DEFAULT_MAX_TICKS_PER_CLICK
    ik_joint_step: float = DEFAULT_IK_JOINT_STEP
    grip_close: float = GRIPPER_CLOSE
    ready_open_grip: bool = DEFAULT_READY_OPEN_GRIP
    primary_camera: str = DEFAULT_PRIMARY_CAM
    grip_dwell_s: float = DEFAULT_GRIP_DWELL_S
    grip_ramp_s: float = DEFAULT_GRIP_RAMP_S
    grip_extra: float = DEFAULT_GRIP_EXTRA
    grip_confirm_samples: int = DEFAULT_GRIP_CONFIRM_SAMPLES
    grip_stop_on_grasp: bool = DEFAULT_GRIP_STOP_ON_GRASP
    place_yaw: float = DEFAULT_PLACE_YAW
    place_apex_fwd: float = DEFAULT_PLACE_APEX_FWD
    place_apex_up: float = DEFAULT_PLACE_APEX_UP
    place_sweep_radius_m: float = DEFAULT_PLACE_SWEEP_RADIUS_M
    place_speed: float = DEFAULT_PLACE_SPEED
    place_ik_guard: float = DEFAULT_PLACE_IK_GUARD
    place_rec_every: int = DEFAULT_PLACE_REC_EVERY
    openvla_stage_back_m: float = DEFAULT_OPENVLA_STAGE_BACK_M
    openvla_stage_min_forward_m: float = DEFAULT_OPENVLA_STAGE_MIN_FORWARD_M
    openvla_stage_up_m: float = DEFAULT_OPENVLA_STAGE_UP_M
    openvla_verify_lift_m: float = DEFAULT_OPENVLA_VERIFY_LIFT_M

    @classmethod
    def from_env(cls) -> "TeleopConfig":
        p = load_params()
        return cls(
            dx=_env_float("MARC_TELEOP_DX", DEFAULT_TELEOP_DX),
            dy=_env_float("MARC_TELEOP_DY", DEFAULT_TELEOP_DY),
            dz=_env_float("MARC_TELEOP_DZ", DEFAULT_TELEOP_DZ),
            d_yaw=_env_float("MARC_TELEOP_DYAW", DEFAULT_TELEOP_DYAW),
            d_pitch=_env_float("MARC_TELEOP_DPITCH", DEFAULT_TELEOP_DPITCH),
            ready_pitch_down=_env_float("MARC_READY_PITCH", DEFAULT_READY_PITCH),
            ready_j1=_env_float("MARC_READY_J1", DEFAULT_READY_J1),
            ready_j2=_env_float("MARC_READY_J2", DEFAULT_READY_J2),
            ready_j3=_env_float("MARC_READY_J3", DEFAULT_READY_J3),
            ready_j4=_env_float("MARC_READY_J4", DEFAULT_READY_J4),
            ready_j5=_env_float("MARC_READY_J5", DEFAULT_READY_J5),
            ready_j6=_env_float("MARC_READY_J6", DEFAULT_READY_J6),
            ready_fwd=_env_float("MARC_READY_FWD", DEFAULT_READY_FWD),
            ready_lat=_env_float("MARC_READY_LAT", DEFAULT_READY_LAT),
            ready_up=_env_float("MARC_READY_UP", DEFAULT_READY_UP),
            ready_move_s=_env_float("MARC_READY_MOVE_S", DEFAULT_READY_MOVE_S),
            ik_joint_step=max(
                0.01,
                _env_float("MARC_IK_JOINT_STEP", DEFAULT_IK_JOINT_STEP),
            ),
            # pnp_params tunes grip_close for the scripted pick's exact geometry; a
            # hand-placed teleop grasp is off-centre and needs a firmer close, so allow
            # an override up to the joint limit.
            grip_close=min(
                GRIPPER_CLOSE,
                max(0.0, _env_float("MARC_GRIP_CLOSE",
                                    float(p.get("grip_close", GRIPPER_CLOSE)))),
            ),
            ready_open_grip=_env_bool("MARC_READY_OPEN_GRIP", DEFAULT_READY_OPEN_GRIP),
            primary_camera=_env_camera("MARC_PRIMARY_CAM", DEFAULT_PRIMARY_CAM),
            grip_dwell_s=_env_float(
                "MARC_GRIP_DWELL", float(p.get("grip_dwell", DEFAULT_GRIP_DWELL_S))
            ),
            grip_ramp_s=_env_float("MARC_GRIP_RAMP", DEFAULT_GRIP_RAMP_S),
            grip_extra=max(0.0, _env_float("MARC_GRIP_EXTRA", DEFAULT_GRIP_EXTRA)),
            grip_confirm_samples=max(
                1,
                int(
                    _env_float(
                        "MARC_GRIP_CONFIRM_SAMPLES", DEFAULT_GRIP_CONFIRM_SAMPLES
                    )
                ),
            ),
            grip_stop_on_grasp=_env_bool(
                "MARC_GRIP_STOP_ON_GRASP", DEFAULT_GRIP_STOP_ON_GRASP
            ),
            place_yaw=_env_float("MARC_PLACE_YAW", DEFAULT_PLACE_YAW),
            place_apex_fwd=_env_float("MARC_PLACE_APEX_FWD", DEFAULT_PLACE_APEX_FWD),
            place_apex_up=_env_float("MARC_PLACE_APEX_UP", DEFAULT_PLACE_APEX_UP),
            place_sweep_radius_m=_env_float(
                "MARC_PLACE_SWEEP_RADIUS_M", DEFAULT_PLACE_SWEEP_RADIUS_M
            ),
            place_speed=_env_float(
                "MARC_PLACE_SPEED", float(p.get("carry_speed", DEFAULT_PLACE_SPEED))
            ),
            place_ik_guard=_env_float("MARC_PLACE_IK_GUARD", DEFAULT_PLACE_IK_GUARD),
            place_rec_every=max(
                1, int(_env_float("MARC_PLACE_REC_EVERY", DEFAULT_PLACE_REC_EVERY))
            ),
            openvla_stage_back_m=max(
                0.0,
                _env_float("MARC_OPENVLA_STAGE_BACK_M", DEFAULT_OPENVLA_STAGE_BACK_M),
            ),
            openvla_stage_min_forward_m=max(
                0.0,
                _env_float(
                    "MARC_OPENVLA_STAGE_MIN_FORWARD_M",
                    DEFAULT_OPENVLA_STAGE_MIN_FORWARD_M,
                ),
            ),
            openvla_stage_up_m=max(
                0.02,
                _env_float("MARC_OPENVLA_STAGE_UP_M", DEFAULT_OPENVLA_STAGE_UP_M),
            ),
            openvla_verify_lift_m=max(
                0.03,
                _env_float(
                    "MARC_OPENVLA_VERIFY_LIFT_M", DEFAULT_OPENVLA_VERIFY_LIFT_M
                ),
            ),
        )

_KIN = None


def _kin():
    global _KIN
    if _KIN is None:
        _KIN = arm_kin.PlacoArmKinematics(_URDF, _EE_FRAME, list(ARM_JOINT_NAMES))
    return _KIN


def read_arm_q(client) -> Optional[np.ndarray]:
    try:
        st = client.get_arm_state()
    except Exception:
        st = None
    if st is None or not getattr(st, "name", None):
        return None
    try:
        idx = [list(st.name).index(n) for n in ARM_JOINT_NAMES]
    except ValueError:
        return None
    return np.array([st.position[i] for i in idx], dtype=np.float64)


def read_grip_and_inner(client) -> Tuple[float, list]:
    grip = GRIPPER_OPEN
    inners = []
    try:
        st = client.get_arm_state()
    except Exception:
        return grip, inners
    if st is None or not getattr(st, "name", None):
        return grip, inners
    names = list(st.name)
    pos = list(st.position)
    for n, p in zip(names, pos):
        if n in GRIPPER_JOINT_NAMES:
            grip = float(p)
        if "inner_finger_knuckle" in n:
            inners.append(float(p))
    return grip, inners


def _shortest_joint_delta(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Element-wise shortest angular delta from q0 to q1 (rad)."""
    d = np.asarray(q1, dtype=np.float64) - np.asarray(q0, dtype=np.float64)
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def clamp_arm_q(q: np.ndarray) -> np.ndarray:
    """Clip a 6-vector to the Gen3 URDF joint limits."""
    return np.clip(np.asarray(q, dtype=np.float64), ARM_JOINT_LOWER, ARM_JOINT_UPPER)


def _cosine_ease(fraction: float) -> float:
    """Zero-velocity endpoint interpolation for carried-object joint moves."""
    t = float(np.clip(fraction, 0.0, 1.0))
    return 0.5 - 0.5 * float(np.cos(np.pi * t))


class ArmTeleop:
    def __init__(self, client, cfg: Optional[TeleopConfig] = None, log=print):
        self._client = client
        self.cfg = cfg or TeleopConfig.from_env()
        self._log = log
        self.kin = _kin()
        self.grip_cmd = GRIPPER_OPEN
        # What the gripper was *asked* to do, as opposed to what is being published.
        # Stopping the close at first contact can leave grip_cmd below the halfway
        # mark, and "am I holding something" must not depend on how early it gripped.
        self.grip_target = GRIPPER_OPEN
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self._saw_grasp_after_close = False
        self._closed_since_last_open = False
        self._place_params: dict = {}
        self._place_quat: Optional[np.ndarray] = None
        self._img_shape: dict = {}   # view key -> locked (h, w, ...) shape
        self._odd_img: dict = {}     # view key -> foreign-frame count
        self._last_view: dict = {}   # view key -> last frame at the locked shape
        overrides = [
            f"J{i}={os.environ[f'MARC_READY_J{i}']}"
            for i in range(1, 7)
            if os.environ.get(f"MARC_READY_J{i}", "").strip() != ""
        ]
        q_ready = self._ready_q()
        source = "default DEFAULT_ARM_Q" if not overrides else "env " + ",".join(overrides)
        self._log(
            "[arm] TeleopConfig (joint-space ready)  "
            + " ".join(f"j{i+1}={q_ready[i]:.4f}" for i in range(6))
            + f"  source={source}  primary_cam={self.cfg.primary_camera}"
        )
        self._log(
            f"[arm] grip_close={self.cfg.grip_close:.3f} (limit {GRIPPER_CLOSE:.3f}, "
            f"pnp_params {float(load_params().get('grip_close', GRIPPER_CLOSE)):.3f}"
            f"{', MARC_GRIP_CLOSE override' if os.environ.get('MARC_GRIP_CLOSE') else ''})"
            f"  dwell={self.cfg.grip_dwell_s:.2f}s"
        )

    @property
    def closing(self) -> bool:
        """True when the gripper was commanded shut, whatever value it settled at."""
        return self.grip_target >= 0.5 * self.cfg.grip_close

    def reset_grasp_flags(self):
        self._saw_grasp_after_close = False
        self._closed_since_last_open = False

    @property
    def saw_grasp_after_close(self) -> bool:
        return self._saw_grasp_after_close

    def _sleep(self, dt: float):
        fn = getattr(self._client, "sleep", None)
        if callable(fn):
            fn(dt)
        else:
            time.sleep(dt)

    def _publish(self, q: np.ndarray):
        self._client.send_arm_command(_make_js(q, self.grip_cmd))

    def warm_cameras(self, seconds: float = 1.0):
        """Touch only the training camera so its lazy subscription is warm."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self._client.get_robot_image(self.cfg.primary_camera)
            self._sleep(0.05)

    def _ready_q(self) -> np.ndarray:
        """Configured ready joint vector (absolute rad)."""
        return np.array(
            [
                float(self.cfg.ready_j1),
                float(self.cfg.ready_j2),
                float(self.cfg.ready_j3),
                float(self.cfg.ready_j4),
                float(self.cfg.ready_j5),
                float(self.cfg.ready_j6),
            ],
            dtype=np.float64,
        )

    def _sync_ee_targets_from_q(self, q: np.ndarray):
        """Seed IK teleop targets from FK so nudges start at the ready EE pose."""
        T = self.kin.fk(np.asarray(q, dtype=np.float64))
        self.target_pos = T[:3, 3].copy()
        self.target_quat = arm_kin.R_to_quat(T[:3, :3])

    def move_ee_to(self, xyz, label: str = "move") -> bool:
        """Move to one Cartesian waypoint while preserving the live orientation."""
        q0 = read_arm_q(self._client)
        if q0 is None:
            self._log(f"[arm] {label}: no joint_states")
            return False
        self._sync_ee_targets_from_q(q0)
        goal = np.asarray(xyz, dtype=np.float64)
        delta = goal - self.target_pos
        self.apply_delta(dx=float(delta[0]), dy=float(delta[1]), dz=float(delta[2]))
        q1 = read_arm_q(self._client)
        if q1 is None:
            self._log(f"[arm] {label}: no final joint_states")
            return False
        actual = self.kin.fk(q1)[:3, 3]
        err = float(np.linalg.norm(actual - goal))
        self._sync_ee_targets_from_q(q1)
        ok = err <= max(0.03, 2.0 * float(self.cfg.settle_tol_m))
        self._log(
            "[arm] %s %s goal=(%.3f,%.3f,%.3f) actual=(%.3f,%.3f,%.3f) err=%.3fm"
            % (
                label,
                "OK" if ok else "FAILED",
                goal[0], goal[1], goal[2],
                actual[0], actual[1], actual[2],
                err,
            )
        )
        return ok

    def move_grasp_pose(self, xyz, yaw: float, label: str = "grasp-pose") -> bool:
        """Move the finger-pad contact point to ``xyz`` at an absolute tool yaw.

        GraspGen-X poses use +X as the closing axis and +Z as the approach axis.
        The analytic arm baseline is intentionally restricted to top-down grasps,
        so only the in-plane closing-axis yaw is applied to the known-safe default
        wrist orientation.  This avoids composing a new yaw onto an accumulated
        teleop orientation on every retry.
        """
        q0 = read_arm_q(self._client)
        if q0 is None:
            self._log(f"[arm] {label}: no joint_states")
            return False
        current = self.kin.fk(q0)[:3, 3]
        base_R = self.kin.fk(arm_kin.DEFAULT_ARM_Q)[:3, :3]
        base_quat = arm_kin.R_to_quat(base_R)
        yaw_quat = arm_kin.rotvec_to_quat(
            np.array([0.0, 0.0, float(yaw)], dtype=np.float64)
        )
        target_quat = arm_kin.quat_mul(base_quat, yaw_quat)
        norm = float(np.linalg.norm(target_quat))
        if norm <= 1e-9:
            self._log(f"[arm] {label}: invalid target orientation")
            return False
        self.target_pos = current.copy()
        self.target_quat = target_quat / norm
        goal = np.asarray(xyz, dtype=np.float64).reshape(3)
        delta = goal - current
        self.apply_delta(dx=float(delta[0]), dy=float(delta[1]), dz=float(delta[2]))
        q1 = read_arm_q(self._client)
        if q1 is None:
            self._log(f"[arm] {label}: no final joint_states")
            return False
        actual = self.kin.fk(q1)[:3, 3]
        err = float(np.linalg.norm(actual - goal))
        self._sync_ee_targets_from_q(q1)
        # Preserve the requested absolute orientation for the next straight-line
        # waypoint; _sync_ee_targets_from_q is used only to refresh live position.
        self.target_quat = target_quat / norm
        ok = err <= max(0.03, 2.0 * float(self.cfg.settle_tol_m))
        self._log(
            "[arm] %s %s goal=(%.3f,%.3f,%.3f) actual=(%.3f,%.3f,%.3f) "
            "yaw=%.2f err=%.3fm"
            % (
                label,
                "OK" if ok else "FAILED",
                goal[0], goal[1], goal[2],
                actual[0], actual[1], actual[2],
                float(yaw), err,
            )
        )
        return ok

    def stage_for_openvla(self, target) -> bool:
        """Move forward clear of the base, then descend into the base-camera view."""
        if not self.set_grip(GRIPPER_OPEN):
            return False
        self.reset_grasp_flags()
        q = read_arm_q(self._client)
        if q is None:
            self._log("[openvla] stage: no joint_states")
            return False
        current = self.kin.fk(q)[:3, 3]
        target_fwd = float(target.fwd)
        target_lat = float(target.lat)
        target_z = float(target.z)
        from .openvla_pick import (
            openvla_stage_goal,
            openvla_stage_waypoints,
            openvla_target_is_safe,
        )

        safe, why = openvla_target_is_safe(target)
        if not safe:
            self._log(f"[openvla] target invalid: {why}")
            return False

        goal = openvla_stage_goal(target, self.cfg, current_fwd=float(current[0]))
        self._log(
            "[openvla] deterministic stage target=(%.3f,%.3f,%.3f) "
            "pregrasp=(%.3f,%.3f,%.3f)"
            % (
                target_fwd, target_lat, target_z,
                goal[0], goal[1], goal[2],
            )
        )
        # Clear the base/camera in +X before any descent.  At the normal ready pose
        # this is an 8–10 cm forward move, matching the demonstrated entry trajectory.
        waypoints = openvla_stage_waypoints(current, target, self.cfg)
        for label, waypoint in waypoints:
            if not self.move_ee_to(waypoint, label):
                return False
        return True

    def lift_for_grasp_verification(self) -> bool:
        """Lift a candidate grasp so vision can distinguish cargo from finger contact."""
        q = read_arm_q(self._client)
        if q is None:
            self._log("[openvla] verify-lift: no joint_states")
            return False
        current = self.kin.fk(q)[:3, 3]
        goal = current.copy()
        goal[2] = min(0.40, float(goal[2]) + self.cfg.openvla_verify_lift_m)
        return self.move_ee_to(goal, "verify-lift")

    def set_grip(self, grip: float) -> bool:
        """Command the gripper without moving the arm or recording steps.

        Uses the same configured ramp as a recorded grip click. With the default
        full-close mode, a close always reaches MARC_GRIP_CLOSE despite hold sensors.
        """
        if read_arm_q(self._client) is None:
            self._log("[arm] no joint_states — grip command skipped")
            return False
        self._grip_click(float(grip), None)
        return True

    def go_ready(self, timeout_s: float = 8.0, open_grip: Optional[bool] = None) -> bool:
        """Move to the configured joint-space ready pose (no Cartesian IK).

        The gripper command is held through the motion unless `open_grip` (or
        MARC_READY_OPEN_GRIP) asks for a release, so a grasped object survives the
        return to ready.

        After settle, EE targets are set from FK so subsequent teleop uses normal DLS IK.
        """
        if open_grip is None:
            open_grip = bool(self.cfg.ready_open_grip)
        if open_grip:
            self.grip_cmd = GRIPPER_OPEN
            self.grip_target = GRIPPER_OPEN
        self._saw_grasp_after_close = False
        self._closed_since_last_open = self.closing
        self._log(f"[arm] ready keeps grip_cmd={self.grip_cmd:.3f}"
                  f"{' (opened)' if open_grip else ' (held)'}")
        q0 = read_arm_q(self._client)
        if q0 is None:
            t0 = time.monotonic()
            while q0 is None and time.monotonic() - t0 < 3.0:
                self._sleep(0.05)
                q0 = read_arm_q(self._client)
        if q0 is None:
            self._log("[arm] no joint_states for ready pose")
            return False

        q_goal = self._ready_q()
        self._log(
            "[arm] joint-space ready -> "
            + " ".join(f"j{i+1}={q_goal[i]:.3f}" for i in range(6))
        )

        dt = 1.0 / self.cfg.rate_hz
        move_s = max(0.3, float(self.cfg.ready_move_s))
        n_steps = max(1, int(move_s * self.cfg.rate_hz))
        delta = _shortest_joint_delta(q0, q_goal)

        for i in range(1, n_steps + 1):
            a = float(i) / float(n_steps)
            cmd = q0 + a * delta
            self._publish(cmd)
            self._sleep(dt)

        # Hold at goal until feedback catches up (or timeout).
        t0 = time.monotonic()
        hold_deadline = t0 + max(0.5, timeout_s - move_s)
        while time.monotonic() < hold_deadline:
            self._publish(q_goal)
            q = read_arm_q(self._client)
            if q is not None:
                err = float(np.linalg.norm(_shortest_joint_delta(q, q_goal)))
                if err < self.cfg.settle_tol_joint:
                    self._sync_ee_targets_from_q(q_goal)
                    ee = self.target_pos
                    self._log(
                        f"[arm] ready OK (joint-space)  "
                        f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  "
                        f"joint_err={err:.4f}"
                    )
                    return True
            self._sleep(dt)

        self._publish(q_goal)
        self._sync_ee_targets_from_q(q_goal)
        self._log("[arm] ready hold timeout — continuing with commanded joint pose")
        return False

    def _locked_view(self, key: str, frame, allow_stale: bool = True):
        """Keep one resolution per view, rejecting frames from a foreign publisher.

        Some camera topics carry a second publisher at a different resolution and a
        different viewpoint. Mixing them would break the episode's image tensor and
        mislabel foreign frames as this camera. Secondary views fall back to their
        last valid frame; the primary view returns None instead, so the step is
        dropped rather than paired with a duplicated training image.
        """
        stale = self._last_view.get(key) if allow_stale else None
        if frame is None:
            return stale
        shape = frame.shape
        locked = self._img_shape.get(key)
        if locked is None:
            self._img_shape[key] = shape
        elif shape != locked:
            self._odd_img[key] = self._odd_img.get(key, 0) + 1
            n = self._odd_img[key]
            if n == 1 or n % 100 == 0:
                self._log(f"[arm] {key}: ignoring foreign frames at "
                          f"{shape[1]}x{shape[0]} (locked {locked[1]}x{locked[0]}, "
                          f"{n} so far)")
            return stale
        self._last_view[key] = frame
        return frame

    def snapshot_obs(self) -> Optional[dict]:
        """Build one observation dict; None if the primary-camera RGB is missing.

        OpenVLA's current RLDS loader requests only the primary RGB view and no depth.
        Reading the other five streams on every control tick duplicates storage and
        can also stall the manipulation loop behind irrelevant camera publishers.
        """
        primary_key = f"image_{self.cfg.primary_camera}"
        primary = self._locked_view(
            primary_key,
            image_to_rgb(self._client.get_robot_image(self.cfg.primary_camera)),
            allow_stale=False,
        )
        if primary is None:
            return None
        q = read_arm_q(self._client)
        if q is None:
            return None
        grip, inners = read_grip_and_inner(self._client)
        T = self.kin.fk(q)
        grasping = self._client.is_grasping()
        if self._closed_since_last_open and grasping is True:
            self._saw_grasp_after_close = True
        try:
            t_sim = float(self._client.now_s())
        except Exception:
            t_sim = time.time()
        return {
            primary_key: primary,
            "q_arm": q.astype(np.float32),
            "q_grip": np.float32(grip),
            "q_inner": np.asarray(inners, dtype=np.float32),
            "ee_pos": T[:3, 3].astype(np.float32),
            "ee_quat": arm_kin.R_to_quat(T[:3, :3]).astype(np.float32),
            "grasping": -1 if grasping is None else (1 if grasping else 0),
            "t_sim": np.float64(t_sim),
            "t_wall": np.float64(time.time()),
        }

    def _ensure_targets_from_fk(self):
        q = read_arm_q(self._client)
        if q is None:
            return False
        T = self.kin.fk(q)
        if self.target_pos is None:
            self.target_pos = T[:3, 3].copy()
        if self.target_quat is None:
            self.target_quat = arm_kin.R_to_quat(T[:3, :3])
        return True

    def apply_delta(
        self,
        dx=0.0, dy=0.0, dz=0.0,
        droll=0.0, dpitch=0.0, dyaw=0.0,
        grip: Optional[float] = None,
        on_step: Optional[Callable[[dict, np.ndarray], None]] = None,
    ) -> int:
        """Servo toward target+delta; record subframe steps via on_step(obs, action7).

        Returns number of recorded steps.
        """
        if not self._ensure_targets_from_fk():
            self._log("[arm] no joint_states")
            return 0

        if grip is not None:
            moving = (abs(dx) + abs(dy) + abs(dz)
                      + abs(droll) + abs(dpitch) + abs(dyaw)) > 1e-9
            if not moving:
                # Grip-only click: ramp there rather than jumping, so closing can stop
                # at first contact instead of crushing past it.
                return self._grip_click(float(grip), on_step)
            self.grip_target = float(grip)
            self.grip_cmd = float(grip)
            self._closed_since_last_open = self.closing

        goal_pos = self.target_pos + np.array([dx, dy, dz], dtype=np.float64)
        rot = np.array([droll, dpitch, dyaw], dtype=np.float64)
        if float(np.linalg.norm(rot)) > 1e-9:
            self.target_quat = arm_kin.quat_mul(
                self.target_quat, arm_kin.rotvec_to_quat(rot)
            )
            n = float(np.linalg.norm(self.target_quat))
            if n > 0:
                self.target_quat = self.target_quat / n
        self.target_pos = goal_pos.copy()

        # Full click action label (OpenVLA-style EE delta + grip cmd normalized 0/1)
        grip_bin = 1.0 if self.closing else 0.0
        full_action = np.array(
            [dx, dy, dz, droll, dpitch, dyaw, grip_bin], dtype=np.float32
        )

        dt = 1.0 / self.cfg.rate_hz
        recorded = 0
        prev_ee = None

        for _ in range(self.cfg.max_ticks_per_click):
            q = read_arm_q(self._client)
            if q is None:
                self._sleep(dt)
                continue
            ee = self.kin.fk(q)[:3, 3]
            remaining = self.target_pos - ee
            dist = float(np.linalg.norm(remaining))

            # Per-tick action: progress toward goal this tick (pos) + 0 rot after first,
            # grip held constant. First tick carries the full intended delta label.
            if recorded == 0:
                action = full_action.copy()
            else:
                step = min(self.cfg.pos_step, dist) if dist > 1e-6 else 0.0
                direction = remaining / dist if dist > 1e-6 else np.zeros(3)
                dpos = direction * step
                action = np.array(
                    [dpos[0], dpos[1], dpos[2], 0.0, 0.0, 0.0, grip_bin],
                    dtype=np.float32,
                )

            raw_cmd = arm_kin.dls_solve(
                self.kin, q, self.target_pos, self.target_quat
            )
            # DLS can jump IK branches near a singularity.  Leash every joint each
            # tick so an unstable solve cannot become a violent physical command.
            dq = _shortest_joint_delta(q, raw_cmd)
            step = np.clip(dq, -self.cfg.ik_joint_step, self.cfg.ik_joint_step)
            cmd = clamp_arm_q(q + step)
            self._publish(cmd)

            if on_step is not None:
                obs = self.snapshot_obs()
                if obs is not None:
                    on_step(obs, action)
                    recorded += 1

            if dist < self.cfg.settle_tol_m and recorded > 0:
                break
            if prev_ee is not None and dist < self.cfg.settle_tol_m:
                break
            prev_ee = ee.copy()
            self._sleep(dt)

        # Grip-only click with no motion still needs at least one frame
        if recorded == 0 and on_step is not None and abs(dx) + abs(dy) + abs(dz) < 1e-9:
            # dwell a bit for gripper
            for _ in range(max(3, int(0.3 * self.cfg.rate_hz))):
                q = read_arm_q(self._client)
                if q is not None:
                    self._publish(q)
                obs = self.snapshot_obs()
                if obs is not None:
                    on_step(obs, full_action)
                    recorded += 1
                    break
                self._sleep(dt)

        return recorded

    def _grip_click(self, target: float, on_step, cancel_event=None) -> int:
        """Ramp to `target`, optionally stopping after a sensor-confirmed bite.

        Two reasons this is not a single command plus a wait. `is_grasping()` only
        flips once the fingers have physically travelled (~grip_dwell), so one sample
        right after the command always reads False and the grasp never gets latched for
        the success filter. With grip_stop_on_grasp disabled, sensor readings are
        ignored and the command always reaches the requested target.
        """
        dt = 1.0 / self.cfg.rate_hz
        self.grip_target = float(target)
        start = float(self.grip_cmd)
        ramp_ticks = (
            1
            if self.cfg.grip_ramp_s <= 0.0
            else max(1, int(self.cfg.grip_ramp_s * self.cfg.rate_hz))
        )
        hold_ticks = max(2, int(max(0.2, self.cfg.grip_dwell_s) * self.cfg.rate_hz))
        stop_early = self.closing and bool(self.cfg.grip_stop_on_grasp)
        self._closed_since_last_open = self.closing
        q_hold = read_arm_q(self._client)
        recorded = 0
        contact_at = None
        bite_target = None
        bite_complete_at = None
        ramp_step = abs(float(target) - start) / float(ramp_ticks)
        bite_ticks = (
            max(1, int(np.ceil(self.cfg.grip_extra / max(ramp_step, 1e-9))))
            if stop_early and self.cfg.grip_extra > 0.0
            else 0
        )
        normal_end_tick = ramp_ticks + hold_ticks
        max_ticks = normal_end_tick + bite_ticks + hold_ticks + 1
        grip_bin = 1.0 if self.closing else 0.0
        # The learned action is a high-level open/close command. Deployment executes
        # the whole sensor-controlled ramp after the first close prediction, so record
        # the pre-action image once instead of attaching the same binary label to every
        # physical finger-control tick.
        if on_step is not None:
            obs = self.snapshot_obs()
            if obs is not None:
                on_step(obs, np.array([0.0] * 6 + [grip_bin], dtype=np.float32))
                recorded = 1
        for i in range(1, max_ticks + 1):
            if cancel_event is not None and cancel_event.is_set():
                break
            if contact_at is None and i <= ramp_ticks:
                frac = float(i) / float(ramp_ticks)
                self.grip_cmd = start + (float(target) - start) * frac
            elif contact_at is not None and bite_complete_at is None:
                self.grip_cmd = min(
                    float(bite_target),
                    self.grip_cmd + ramp_step,
                )
            if q_hold is not None:
                self._publish(q_hold)
            holding = stop_early and self._client.is_grasping() is True
            if holding and contact_at is None:
                confirmed = True
                for _ in range(1, self.cfg.grip_confirm_samples):
                    if cancel_event is not None and cancel_event.is_set():
                        confirmed = False
                        break
                    # Freeze both the arm and current finger command while confirming;
                    # confirmation must not itself squeeze or push the tumbler.
                    self._sleep(dt)
                    if q_hold is not None:
                        self._publish(q_hold)
                    if self._client.is_grasping() is not True:
                        confirmed = False
                        self._log(
                            f"[arm] ignored transient hold signal at "
                            f"grip={self.grip_cmd:.3f}"
                        )
                        break
                if not confirmed:
                    self._sleep(dt)
                    continue
                contact_at = i
                contact_grip = self.grip_cmd
                bite_target = min(float(target), contact_grip + self.cfg.grip_extra)
                self._saw_grasp_after_close = True
                self._log(
                    f"[arm] hold confirmed ({self.cfg.grip_confirm_samples} samples) "
                    f"at grip={contact_grip:.3f} after "
                    f"{i / self.cfg.rate_hz:.2f}s; continuing close to "
                    f"{bite_target:.3f} (full close {target:.3f})"
                )
            if (
                contact_at is not None
                and bite_complete_at is None
                and self.grip_cmd >= float(bite_target) - 1e-9
            ):
                bite_complete_at = i
                self._log(
                    f"[arm] post-contact bite complete at grip={self.grip_cmd:.3f}; "
                    "holding"
                )
            if bite_complete_at is not None and i - bite_complete_at >= hold_ticks:
                break
            if contact_at is None and i >= normal_end_tick:
                break
            self._sleep(dt)
        if stop_early and contact_at is None:
            self._log(f"[arm] no hold detected during close; at grip={self.grip_cmd:.3f}")
        return recorded

    def refresh_grasp(self, samples: int = 8, cancel_event=None):
        """Re-sample hold state so a late-settling grasp still counts at episode end."""
        closed = self.closing
        latest = None
        for _ in range(max(1, samples)):
            if cancel_event is not None and cancel_event.is_set():
                break
            g = self._client.is_grasping()
            if g is not None:
                latest = g
            if closed and g is True:
                self._closed_since_last_open = True
                self._saw_grasp_after_close = True
                return True
            self._sleep(0.05)
        return latest

    def _record_tick(self, q, wp_pos, on_step) -> int:
        """Label one scripted tick with the same EE-delta convention as manual clicks."""
        if on_step is None:
            return 0
        obs = self.snapshot_obs()
        if obs is None:
            return 0
        T = self.kin.fk(q)
        remaining = np.asarray(wp_pos, dtype=np.float64) - T[:3, 3]
        dist = float(np.linalg.norm(remaining))
        step = min(self.cfg.pos_step, dist) if dist > 1e-6 else 0.0
        dpos = (remaining / dist * step) if dist > 1e-6 else np.zeros(3)
        # A grip-only segment can be the first thing a place does, before any move has
        # set a target orientation; there is no rotation to label in that case.
        if self._place_quat is None:
            rot = np.zeros(3)
        else:
            rot = arm_kin.quat_error_axis_angle(
                self._place_quat, arm_kin.R_to_quat(T[:3, :3])
            )
            rot_n = float(np.linalg.norm(rot))
            if rot_n > self.cfg.d_yaw:
                rot = rot / rot_n * self.cfg.d_yaw
        grip_bin = 1.0 if self.closing else 0.0
        action = np.array(
            [dpos[0], dpos[1], dpos[2], rot[0], rot[1], rot[2], grip_bin],
            dtype=np.float32,
        )
        on_step(obs, action)
        return 1

    def _grip_segment(self, label, grip: float, on_step, dwell_s: float,
                      intent: Optional[float] = None, cancel_event=None) -> int:
        """Hold the arm still at grip command `grip` for a dwell.

        `intent` overrides what this counts as: holding a grasp that stopped early at
        0.3 still means "closed", even though 0.3 is below the halfway mark.
        """
        dt = 1.0 / self.cfg.rate_hz
        self.grip_cmd = float(grip)
        self.grip_target = float(grip if intent is None else intent)
        self._closed_since_last_open = self.closing
        # Lock one arm pose for the entire gripper segment. Re-commanding each noisy
        # measured joint state makes the arm chase encoder noise and shake its cargo.
        q_hold = read_arm_q(self._client)
        recorded = 0
        for _ in range(max(1, int(max(0.2, dwell_s) * self.cfg.rate_hz))):
            if cancel_event is not None and cancel_event.is_set():
                break
            q = read_arm_q(self._client)
            if q is not None:
                recorded += self._record_tick(q, self.kin.fk(q)[:3, 3], on_step)
            if q_hold is not None:
                self._publish(q_hold)
            self._sleep(dt)
        self._log(f"  [place] {label} grip={grip:.3f} steps+={recorded}")
        return recorded

    def _joint_move(self, label, q_goal, on_step, speed: float,
                    timeout_s: float = 4.0, wrap: bool = False,
                    cancel_event=None) -> Optional[int]:
        """Feedback-paced, eased joint interpolation for carrying to the basket.

        Every waypoint is a joint vector, so nothing here can flip an IK branch or
        drive the wrist through the forearm; the arm sweeps between two poses that are
        each known to be safe. `wrap` takes the shortest angular path per joint, which
        is what you want coming back from an arbitrary teleop pose. Tucked yaw legs
        use wrap off so joint 1 follows the configured direction. The
        trajectory advances only after joint feedback changes, and cosine easing
        removes the velocity jumps that can shake an object out at waypoints.
        """
        dt = 1.0 / self.cfg.rate_hz
        q0 = read_arm_q(self._client)
        if q0 is None:
            self._log(f"  [place] {label}: no joint_states")
            return None
        q_goal = np.asarray(q_goal, dtype=np.float64)
        T = self.kin.fk(q_goal)
        ee_goal = T[:3, 3]
        self._place_quat = arm_kin.R_to_quat(T[:3, :3])
        delta = _shortest_joint_delta(q0, q_goal) if wrap else q_goal - q0
        span = float(np.max(np.abs(delta)))
        # Cosine easing peaks at pi/2 times its average rate. Increase duration so
        # MARC_PLACE_SPEED remains a true maximum rather than only an average.
        peak_slope = 0.5 * np.pi
        n_ticks = max(
            1,
            int(np.ceil(peak_slope * span / max(1e-4, speed / self.cfg.rate_hz))),
        )
        every = max(1, int(self.cfg.place_rec_every))
        recorded = 0
        previous_q = q0.copy()
        last_cmd = q0.copy()
        max_stale = max(3, int(max(0.5, timeout_s) * self.cfg.rate_hz))
        follow_tol = max(
            float(self.cfg.settle_tol_joint),
            2.5 * float(speed) / float(self.cfg.rate_hz),
        )
        for i in range(1, n_ticks + 1):
            if cancel_event is not None and cancel_event.is_set():
                break
            if i > 1:
                waited = 0
                followed = False
                while waited < max_stale:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    q = read_arm_q(self._client)
                    if q is not None and not np.array_equal(q, previous_q):
                        previous_q = q.copy()
                        lag = float(np.max(np.abs(
                            _shortest_joint_delta(q, last_cmd)
                        )))
                        if lag <= follow_tol:
                            followed = True
                            break
                    self._publish(last_cmd)
                    self._sleep(dt)
                    waited += 1
                if cancel_event is not None and cancel_event.is_set():
                    break
                if not followed:
                    self._log(
                        f"  [place] {label}: joint feedback did not follow command; "
                        "stopped before advancing trajectory"
                    )
                    return None
            fraction = _cosine_ease(float(i) / float(n_ticks))
            last_cmd = q0 + fraction * delta
            self._publish(last_cmd)
            if i % every == 0:
                q = read_arm_q(self._client)
                if q is not None:
                    recorded += self._record_tick(q, ee_goal, on_step)
            self._sleep(dt)
        q_hold = q0 + delta
        hold_ticks = max(1, int(max(0.3, timeout_s) * self.cfg.rate_hz))
        settled = False
        for i in range(1, hold_ticks + 1):
            if cancel_event is not None and cancel_event.is_set():
                break
            self._publish(q_hold)
            q = read_arm_q(self._client)
            if q is not None:
                if i % every == 0:
                    recorded += self._record_tick(q, ee_goal, on_step)
                if float(np.linalg.norm(_shortest_joint_delta(q, q_hold))) < \
                        self.cfg.settle_tol_joint:
                    settled = True
                    break
            self._sleep(dt)
        if cancel_event is None or not cancel_event.is_set():
            if not settled:
                self._log(f"  [place] {label}: settle timeout; trajectory aborted")
                return None
        q_end = read_arm_q(self._client)
        ee = self.kin.fk(q_hold if q_end is None else q_end)[:3, 3]
        self._log(f"  [place] {label}: {span:.2f} rad in {n_ticks} eased ticks, "
                  f"ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) steps+={recorded}")
        return recorded

    def _place_sweep_radius(self, q) -> float:
        positions = self.kin.frame_positions(q, PLACE_SWEEP_FRAMES)
        return float(np.max(np.linalg.norm(positions[:, :2], axis=1))) + \
            DEFAULT_PLACE_LINK_PAD_M

    def _guarded_ik(self, label, target, q_seed) -> Optional[np.ndarray]:
        """IK for `target`, refused if the solver jumped to a different branch.

        A branch jump is what wrecked the old Cartesian path: the solver happily
        returns a pose that reaches the point with the elbow folded through the
        forearm, and interpolating into it drives the arm through itself. Legitimate
        moves here are local, so a large joint deviation from the seed means the
        solution is not the one we asked for.
        """
        target = np.asarray(target, dtype=np.float64)
        quat = arm_kin.R_to_quat(self.kin.fk(q_seed)[:3, :3])
        q = arm_kin.unwrap(arm_kin.ik_solve(self.kin, target, quat, q_seed), q_seed)
        q = clamp_arm_q(q)
        err = float(np.linalg.norm(target - self.kin.fk(q)[:3, 3]))
        dev = float(np.max(np.abs(q - q_seed)))
        if err < self.cfg.settle_tol_m * 2.0 and dev <= float(self.cfg.place_ik_guard):
            self._log(
                f"  [place] {label} solved: err={err:.3f}m dev={dev:.2f} rad "
                f"j5={float(q[4]):+.3f} j6={float(q[5]):+.3f}"
            )
            return q
        self._log(f"  [place] {label} REJECTED: err={err:.3f}m dev={dev:.2f} rad "
                  f"(guard {self.cfg.place_ik_guard:.2f})")
        return None

    def _place_poses(self):
        """Lift, retract inside the chassis, yaw joint 1, then extend to basket.

        Only the compact pose is rotated. If IK or the conservative swept-cylinder
        check fails, return None before moving and keep holding.
        """
        p = self._place_params
        q_home = np.asarray(arm_kin.DEFAULT_ARM_Q, dtype=np.float64)
        carry_z = float(p["carry_up"])
        ee_home = self.kin.fk(q_home)[:3, 3]
        q_lift = self._guarded_ik(
            "front-high", [float(ee_home[0]), float(ee_home[1]), carry_z], q_home
        )
        if q_lift is None:
            return None
        q_tuck = self._guarded_ik(
            "tuck-high",
            [float(self.cfg.place_apex_fwd), 0.0, float(self.cfg.place_apex_up)],
            q_lift,
        )
        if q_tuck is None:
            return None
        q_over = q_tuck.copy()
        q_over[0] += float(self.cfg.place_yaw)
        q_over = clamp_arm_q(q_over)
        sweep_radius = max(
            self._place_sweep_radius(q_tuck),
            self._place_sweep_radius(q_over),
        )
        if sweep_radius > float(self.cfg.place_sweep_radius_m):
            self._log(
                "  [place] tuck REJECTED: swept radius %.3fm exceeds %.3fm chassis envelope"
                % (sweep_radius, float(self.cfg.place_sweep_radius_m))
            )
            return None
        q_drop = self._guarded_ik(
            "in-basket",
            [float(p["place_fwd"]), float(p["place_lat"]), float(p["place_up"])],
            q_over,
        )
        if q_drop is None:
            return None
        return q_lift, q_tuck, q_over, q_drop

    def place_in_basket(
        self,
        on_step: Optional[Callable[[dict, np.ndarray], None]] = None,
        cancel_event=None,
    ) -> int:
        """Keep the squeeze, tuck inside the chassis, rotate into the basket, drop.

        Dataset collection passes `on_step=None`: place is hard-coded, so its ticks must
        not enter the training set. Pass a callback only when you want those steps logged.

        Yaw legs use wrap=False so interpolation follows the configured direction.
        """
        if read_arm_q(self._client) is None:
            self._log("[place] no joint_states")
            return 0
        self._place_params = load_params()
        q_home = np.asarray(arm_kin.DEFAULT_ARM_Q, dtype=np.float64)
        poses = self._place_poses()
        if poses is None:
            self._log("[place] cannot solve a body-contained basket path — nothing moved, "
                      "still holding")
            return 0
        q_lift, q_tuck, q_over, q_drop = poses
        speed = float(self.cfg.place_speed)
        dwell = max(float(self.cfg.grip_dwell_s), 0.4)
        self._log(
            "[place] tucked sweep radius=%.3fm j1=%.2f at %.2f rad/s"
            % (
                self._place_sweep_radius(q_tuck),
                float(q_over[0]),
                speed,
            )
        )

        n = 0

        def stop_if_requested():
            if cancel_event is None or not cancel_event.is_set():
                return False
            q_stop = read_arm_q(self._client)
            if q_stop is not None:
                self._publish(q_stop)
                self._sync_ee_targets_from_q(q_stop)
            self._log("[place] E stop received — scripted placement cancelled")
            return True

        q_now = read_arm_q(self._client)
        self._place_quat = arm_kin.R_to_quat(self.kin.fk(q_now)[:3, :3])
        if stop_if_requested():
            return n
        # Keep the latched squeeze. Publishing json/env close after contact ejects
        # the object. grip_target already means "closed" even if grip_cmd is 0.33.
        if self.closing or self._closed_since_last_open:
            n += self._grip_segment("hold", self.grip_cmd, on_step, dwell,
                                    intent=self.cfg.grip_close,
                                    cancel_event=cancel_event)
        else:
            self._log("[place] gripper is open — closing before the move")
            n += self._grip_click(self.cfg.grip_close, on_step, cancel_event)
        if stop_if_requested():
            return n
        self._log(
            f"[place] holding={self.refresh_grasp(cancel_event=cancel_event)} "
            f"at grip={self.grip_cmd:.3f}"
        )
        if stop_if_requested():
            return n
        moves = (
            ("raise to carry height", q_lift, False),
            ("tuck inside chassis", q_tuck, False),
            ("rotate tucked over basket", q_over, False),
            ("extend into basket", q_drop, False),
        )
        for label, goal, wrap in moves:
            moved = self._joint_move(
                label, goal, on_step, speed, wrap=wrap, cancel_event=cancel_event
            )
            if moved is None:
                self._log(f"[place] {label} failed — stopped while still holding")
                return n
            n += moved
            if stop_if_requested():
                return n
        n += self._grip_segment(
            "release", GRIPPER_OPEN, on_step, dwell, cancel_event=cancel_event
        )
        if stop_if_requested():
            return n
        return_moves = (
            ("retract inside chassis", q_over),
            ("rotate tucked back to front", q_tuck),
            ("fold to ready", q_home),
        )
        for label, goal in return_moves:
            moved = self._joint_move(
                label, goal, on_step, speed, cancel_event=cancel_event
            )
            if moved is None:
                self._log(f"[place] {label} failed — trajectory stopped")
                return n
            n += moved
            if stop_if_requested():
                return n

        # Resync IK targets so later manual nudges start from where the arm actually is.
        q_end = read_arm_q(self._client)
        if q_end is not None:
            self._sync_ee_targets_from_q(q_end)
        self._log(f"[place] done  steps+={n}")
        return n
