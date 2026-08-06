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
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class TeleopConfig:
    dx: float = 0.01
    dy: float = 0.01
    dz: float = 0.01
    d_yaw: float = 0.05
    d_pitch: float = 0.05
    ready_pitch_down: float = 0.70   # rad ~40 deg about EE Y
    ready_fwd: float = 0.30
    ready_lat: float = 0.0
    ready_up: float = 0.20
    rate_hz: float = 15.0
    pos_step: float = 0.008
    settle_tol_m: float = 0.015
    max_ticks_per_click: int = 40
    grip_close: float = GRIPPER_CLOSE

    @classmethod
    def from_env(cls) -> "TeleopConfig":
        p = load_params()
        return cls(
            dx=_env_float("MARC_TELEOP_DX", 0.01),
            dy=_env_float("MARC_TELEOP_DY", 0.01),
            dz=_env_float("MARC_TELEOP_DZ", 0.01),
            d_yaw=_env_float("MARC_TELEOP_DYAW", 0.05),
            d_pitch=_env_float("MARC_TELEOP_DPITCH", 0.05),
            ready_pitch_down=_env_float("MARC_READY_PITCH", 0.70),
            grip_close=float(p.get("grip_close", GRIPPER_CLOSE)),
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


def _pitch_down_quat(base_quat: np.ndarray, pitch_rad: float) -> np.ndarray:
    """Rotate about EE local Y (pitch down)."""
    q = arm_kin.quat_mul(base_quat, arm_kin.rotvec_to_quat(np.array([0.0, pitch_rad, 0.0])))
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else base_quat


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

    def go_ready(self, timeout_s: float = 8.0) -> bool:
        """Servo to a down-tilted ready EE pose with gripper open."""
        self.grip_cmd = GRIPPER_OPEN
        self.reset_grasp_flags()
        q = read_arm_q(self._client)
        if q is None:
            # wait briefly for joint_states
            t0 = time.monotonic()
            while q is None and time.monotonic() - t0 < 3.0:
                self._sleep(0.05)
                q = read_arm_q(self._client)
        if q is None:
            self._log("[arm] no joint_states for ready pose")
            return False

        T0 = self.kin.fk(arm_kin.DEFAULT_ARM_Q)
        base_quat = arm_kin.R_to_quat(T0[:3, :3])
        self.target_quat = _pitch_down_quat(base_quat, self.cfg.ready_pitch_down)
        self.target_pos = np.array(
            [self.cfg.ready_fwd, self.cfg.ready_lat, self.cfg.ready_up], dtype=np.float64
        )

        dt = 1.0 / self.cfg.rate_hz
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            q = read_arm_q(self._client)
            if q is None:
                self._sleep(dt)
                continue
            cmd = arm_kin.dls_solve(self.kin, q, self.target_pos, self.target_quat)
            self._publish(cmd)
            ee = self.kin.fk(q)[:3, 3]
            if float(np.linalg.norm(ee - self.target_pos)) < self.cfg.settle_tol_m:
                self._log("[arm] ready pose reached (down-tilt)")
                return True
            self._sleep(dt)
        self._log("[arm] ready pose settle timeout (continuing)")
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
