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

from .images import image_to_depth, image_to_rgb


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
DEFAULT_READY_OPEN_GRIP = False  # ready pose holds whatever the gripper was commanded to


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
    grip_close: float = GRIPPER_CLOSE
    ready_open_grip: bool = DEFAULT_READY_OPEN_GRIP

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
            grip_close=float(p.get("grip_close", GRIPPER_CLOSE)),
            ready_open_grip=_env_bool("MARC_READY_OPEN_GRIP", DEFAULT_READY_OPEN_GRIP),
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


class ArmTeleop:
    def __init__(self, client, cfg: Optional[TeleopConfig] = None, log=print):
        self._client = client
        self.cfg = cfg or TeleopConfig.from_env()
        self._log = log
        self.kin = _kin()
        self.grip_cmd = GRIPPER_OPEN
        self.target_pos: Optional[np.ndarray] = None
        self.target_quat: Optional[np.ndarray] = None
        self._saw_grasp_after_close = False
        self._closed_since_last_open = False
        self._log(
            "[arm] TeleopConfig (joint-space ready)  "
            f"j1={self.cfg.ready_j1:.4f} j2={self.cfg.ready_j2:.4f} "
            f"j3={self.cfg.ready_j3:.4f} j4={self.cfg.ready_j4:.4f} "
            f"j5={self.cfg.ready_j5:.4f} j6={self.cfg.ready_j6:.4f} "
            f"(MARC_READY_J2={os.environ.get('MARC_READY_J2')!r} "
            f"J3={os.environ.get('MARC_READY_J3')!r} "
            f"J4={os.environ.get('MARC_READY_J4')!r} "
            f"J5={os.environ.get('MARC_READY_J5')!r} "
            f"J6={os.environ.get('MARC_READY_J6')!r})"
        )

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
        """Touch camera topics so subscriptions fill."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self._client.get_robot_image("gripper_left")
            self._client.get_robot_image("base_left")
            self._client.get_robot_depth("gripper")
            self._client.get_robot_depth("base")
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

    def set_grip(self, grip: float) -> bool:
        """Command the gripper without moving the arm or recording steps."""
        q = read_arm_q(self._client)
        if q is None:
            self._log("[arm] no joint_states — grip command skipped")
            return False
        if grip >= 0.5 * self.cfg.grip_close:
            self._closed_since_last_open = True
        else:
            self._closed_since_last_open = False
        self.grip_cmd = float(grip)
        for _ in range(max(1, int(0.3 * self.cfg.rate_hz))):
            self._publish(q)
            self._sleep(1.0 / self.cfg.rate_hz)
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
        self._saw_grasp_after_close = False
        self._closed_since_last_open = self.grip_cmd >= 0.5 * self.cfg.grip_close
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

    def snapshot_obs(self) -> Optional[dict]:
        """Build one observation dict; None if gripper RGB missing."""
        g_img = image_to_rgb(self._client.get_robot_image("gripper_left"))
        if g_img is None:
            return None
        b_img = image_to_rgb(self._client.get_robot_image("base_left"))
        g_dep = image_to_depth(self._client.get_robot_depth("gripper"))
        b_dep = image_to_depth(self._client.get_robot_depth("base"))
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
            "image_gripper": g_img,
            "image_base": b_img,
            "depth_gripper": g_dep,
            "depth_base": b_dep,
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
            if grip >= 0.5 * self.cfg.grip_close:
                self._closed_since_last_open = True
            else:
                self._closed_since_last_open = False
            self.grip_cmd = float(grip)

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
        grip_bin = 1.0 if self.grip_cmd >= 0.5 * self.cfg.grip_close else 0.0
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

            cmd = arm_kin.dls_solve(self.kin, q, self.target_pos, self.target_quat)
            # leash: don't command huge jumps
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
