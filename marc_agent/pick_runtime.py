"""Stage 2 pick: grasp on positive feedback, scripted basket place, bounded retry.

Approach/reacquire stays with #12. This module only accepts ``is_grasping() is True``,
hands off to ``ArmTeleop.place_in_basket``, and requires ``is_basket_occupied() is True``.
``None`` from either sensor is not success.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from marc_agent.graspgenx_policy import (
    MIN_CLOUD_POINTS,
    best_grasp,
    grasp_yaw_from_matrix,
    load_graspgenx,
    points_from_target,
)
from marc_agent.openvla_actions import (
    ACTION_DIM,
    gripper_should_close,
    pick_instruction,
)
from marc_agent.openvla_policy import (
    GeometricGraspStub,
    PolicyObservation,
    RemoteOpenVLAPolicy,
    load_openvla,
)

_DEMO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo")
if _DEMO not in sys.path:
    sys.path.insert(0, _DEMO)

log = logging.getLogger("participant_app")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_GRASP_POLL = 8
DEFAULT_BASKET_POLL = 10
DEFAULT_POLL_DT = 0.05
DEFAULT_POLICY_TICKS = 80
DEFAULT_REMOTE_POLICY_TICKS = 20
DEFAULT_POLICY = "graspgenx"
DEFAULT_GRASP_FLOOR_Z = -0.28
GRASP_FWD_MIN_M = 0.15
GRASP_FWD_MAX_M = 0.50
GRASP_LAT_MAX_M = 0.45
REASON_OK = "ok"
REASON_NO_GRASP = "no_grasp"
REASON_DROP = "drop"
REASON_NO_BASKET = "no_basket"
REASON_LOST_TARGET = "lost_target"
REASON_TIMEOUT = "timeout"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def pick_policy_name() -> str:
    raw = (os.environ.get("MARC_PICK_POLICY") or DEFAULT_POLICY).strip().lower()
    if raw in (
        "graspgenx", "ik", "openvla", "openvla_remote", "openvla_geom", "wrist",
    ):
        return raw
    return DEFAULT_POLICY


def _grasp_floor_z() -> float:
    try:
        from arm_pick import load_params
        return float(load_params().get("grasp_down", DEFAULT_GRASP_FLOOR_Z))
    except Exception:
        return DEFAULT_GRASP_FLOOR_Z


def clamp_grasp_xyz(ggx_xyz, localized_xyz=None, floor_z=DEFAULT_GRASP_FLOOR_Z):
    """Keep GraspGen-X XY; never go below the pavement floor z."""
    del localized_xyz
    gx, gy, gz = (float(ggx_xyz[0]), float(ggx_xyz[1]), float(ggx_xyz[2]))
    return gx, gy, max(gz, float(floor_z))


def grasp_xyz_in_reach(xyz) -> bool:
    fwd, lat = float(xyz[0]), float(xyz[1])
    return GRASP_FWD_MIN_M <= fwd <= GRASP_FWD_MAX_M and abs(lat) <= GRASP_LAT_MAX_M


@dataclass
class PickResult:
    ok: bool = False
    reason: str = REASON_NO_GRASP
    attempts: int = 0
    grasped: bool = False
    basket_occupied: bool = False
    policy: str = DEFAULT_POLICY

    @property
    def picked(self) -> bool:
        return self.ok


class PickRuntime:
    """Grasp (OpenVLA stub or IK) then scripted basket load, with bounded retry."""

    def __init__(
        self,
        client,
        log=None,
        *,
        max_attempts: Optional[int] = None,
        policy: Optional[str] = None,
        grasp_fn: Optional[Callable] = None,
        place_fn: Optional[Callable] = None,
        recover_fn: Optional[Callable] = None,
        reacquire_fn: Optional[Callable] = None,
        grasp_generator: Optional[Callable] = None,
        openvla_policy=None,
        poll_dt: float = DEFAULT_POLL_DT,
        grasp_poll: int = DEFAULT_GRASP_POLL,
        basket_poll: int = DEFAULT_BASKET_POLL,
        policy_ticks: Optional[int] = None,
    ):
        self.client = client
        self._log = log if log is not None else logging.getLogger("participant_app")
        self.max_attempts = int(
            max_attempts if max_attempts is not None
            else _env_int("MARC_PICK_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
        )
        self.policy = (policy or pick_policy_name()).strip().lower()
        self._grasp_fn = grasp_fn
        self._place_fn = place_fn
        self._recover_fn = recover_fn
        self._reacquire_fn = reacquire_fn
        self._grasp_generator = grasp_generator
        self._openvla_policy = openvla_policy
        self._openvla_load_attempted = openvla_policy is not None
        self.poll_dt = float(poll_dt)
        self.grasp_poll = int(grasp_poll)
        self.basket_poll = int(basket_poll)
        self.policy_ticks = int(
            policy_ticks if policy_ticks is not None else (
                _env_int("MARC_OPENVLA_TICKS", DEFAULT_REMOTE_POLICY_TICKS)
                if self.policy == "openvla_remote" else DEFAULT_POLICY_TICKS
            )
        )
        self._teleop = None
        self._wrist_detector = None
        self._openvla_fallback_logged = False

    def _emit(self, msg: str) -> None:
        if callable(self._log):
            self._log(msg)
        elif hasattr(self._log, "info"):
            self._log.info("%s", msg)
        else:
            log.info("%s", msg)

    def _sleep(self, dt: float) -> None:
        fn = getattr(self.client, "sleep", None)
        if callable(fn):
            fn(dt)
        else:
            import time
            time.sleep(dt)

    def _past_deadline(self, deadline) -> bool:
        if deadline is None:
            return False
        now = getattr(self.client, "now_s", None)
        if callable(now):
            return float(now()) >= float(deadline)
        return False

    def _session_ended(self) -> bool:
        """True when a Stage 2 mission has ended. Practice clients never enter Stage 2."""
        if getattr(self.client, "_shutdown", False):
            return True
        if not getattr(self.client, "_stage2_active", False):
            return False
        return not bool(getattr(self.client, "is_running", True))

    def poll_grasping(self) -> Optional[bool]:
        """True only after a positive hold. None/False are not success."""
        latest = None
        for _ in range(max(1, self.grasp_poll)):
            try:
                g = self.client.is_grasping()
            except Exception:
                g = None
            if g is not None:
                latest = bool(g)
            if g is True:
                return True
            self._sleep(self.poll_dt)
        return latest

    def poll_basket(self) -> Optional[bool]:
        latest = None
        for _ in range(max(1, self.basket_poll)):
            try:
                b = self.client.is_basket_occupied()
            except Exception:
                b = None
            if b is not None:
                latest = bool(b)
            if b is True:
                return True
            self._sleep(self.poll_dt)
        return latest

    def _teleop_arm(self):
        if self._teleop is not None:
            return self._teleop
        from manip_dataset.arm_control import ArmTeleop, TeleopConfig
        self._teleop = ArmTeleop(
            self.client, TeleopConfig.from_env(), log=self._emit)
        return self._teleop

    def _seed_closed_grip(self) -> None:
        """Copy the live latched squeeze; keep close intent without republishing json close."""
        arm = self._teleop
        if arm is None:
            return
        live = None
        try:
            from arm_pick import GRIPPER_JOINT_NAMES
            st = self.client.get_arm_state()
            if st is not None and getattr(st, "name", None):
                for name, pos in zip(st.name, st.position):
                    if name in GRIPPER_JOINT_NAMES:
                        live = float(pos)
        except Exception:
            live = None
        if live is not None and float(live) > 1e-3:
            arm.grip_cmd = float(live)
        arm.grip_target = float(arm.cfg.grip_close)
        arm._saw_grasp_after_close = True
        arm._closed_since_last_open = True

    def _ik_grasp(self, target) -> None:
        import arm_pick
        p = arm_pick.load_params()
        p["grasp_fwd"] = float(target.fwd)
        p["grasp_lat"] = float(target.lat)
        p["target_name"] = getattr(target, "name", None) or "tumbler"
        z = getattr(target, "z", None)
        if z is not None:
            p["grasp_down"] = float(z)
        yaw = getattr(target, "grasp_yaw", None)
        if yaw is not None:
            p["grasp_yaw"] = float(yaw)
        self._emit(
            "[pick] ik fwd=%.3f lat=%.3f z=%.3f yaw=%.2f"
            % (
                float(p["grasp_fwd"]),
                float(p["grasp_lat"]),
                float(p["grasp_down"]),
                float(p.get("grasp_yaw", 0.0)),
            )
        )
        on_pre = None
        if self.policy == "wrist":
            from marc_agent.wrist_pick import refine_at_pre

            def on_pre(seq, client, kin, params):
                refine_at_pre(
                    client,
                    self._wrist_detector,
                    self._teleop_arm(),
                    seq,
                    params,
                    kin,
                    self._emit,
                )

        arm_pick.run_motion(self.client, self._log, p, "pick", on_pre=on_pre)

    def _snapshot_obs(self, target=None) -> PolicyObservation:
        rgb = None
        try:
            img = self.client.get_robot_image("base_left")
            if img is not None:
                if isinstance(img, np.ndarray):
                    rgb = img
                else:
                    from manip_dataset.images import image_to_rgb
                    rgb = image_to_rgb(img)
        except Exception:
            rgb = None
        proprio = None
        ee = None
        ee_quat = None
        try:
            from manip_dataset.arm_control import read_arm_q, read_grip_and_inner
            import arm_kin
            q = read_arm_q(self.client)
            grip, _ = read_grip_and_inner(self.client)
            if q is not None:
                proprio = np.concatenate(
                    [np.asarray(q, dtype=np.float32),
                     np.array([float(grip)], dtype=np.float32)])
                arm = self._teleop_arm()
                pose = arm.kin.fk(q)
                ee = pose[:3, 3]
                ee_quat = arm_kin.R_to_quat(pose[:3, :3])
        except Exception:
            pass
        return PolicyObservation(
            rgb=rgb, proprio=proprio, ee_xyz=ee, ee_quat=ee_quat,
            instruction=pick_instruction(target),
        )

    def _geom_or_openvla_grasp(self, target, policy_obj) -> None:
        arm = self._teleop_arm()
        # TeleopConfig owns the physical calibration and clamps MARC_GRIP_CLOSE.
        close_cmd = float(arm.cfg.grip_close)
        for _ in range(max(1, self.policy_ticks)):
            try:
                if self.client.is_grasping() is True:
                    return
            except Exception:
                pass
            obs = self._snapshot_obs(target)
            action = np.asarray(policy_obj.act(obs), dtype=np.float32).reshape(-1)
            if action.size < ACTION_DIM:
                action = np.pad(action, (0, ACTION_DIM - int(action.size)))
            if self.policy == "openvla_remote":
                limits = np.asarray(
                    (arm.cfg.dx, arm.cfg.dy, arm.cfg.dz,
                     arm.cfg.d_yaw, arm.cfg.d_pitch, arm.cfg.d_yaw),
                    dtype=np.float32,
                )
                action[:6] = np.clip(action[:6], -limits, limits)
            dx, dy, dz = float(action[0]), float(action[1]), float(action[2])
            droll, dpitch, dyaw = float(action[3]), float(action[4]), float(action[5])
            # EE delta first with grip unchanged so apply_delta composes the
            # rotvec into target_quat. Mixed motion+grip would skip the ramp.
            arm.apply_delta(
                dx=dx, dy=dy, dz=dz,
                droll=droll, dpitch=dpitch, dyaw=dyaw,
                grip=None,
            )
            try:
                if self.client.is_grasping() is True:
                    return
            except Exception:
                pass
            # Real OpenVLA uses 0=closed/1=open; the geometric test stub emits
            # the collector's 1=close command intent.
            open_is_high = getattr(policy_obj, "stats", None) is not None
            if gripper_should_close(float(action[6]), open_is_high=open_is_high):
                arm.apply_delta(grip=close_cmd)

    def _apply_graspgenx(self, target) -> bool:
        generator = self._grasp_generator
        if generator is None:
            generator = load_graspgenx()
            if generator is None:
                self._emit("[pick] GraspGen-X unavailable -- fail closed")
                return False
            self._grasp_generator = generator
        try:
            points = points_from_target(target, client=self.client)
        except Exception as exc:
            self._emit("[pick] GraspGen-X cloud failed (%s) -- fail closed" % exc)
            return False
        n_pts = 0 if points is None else int(points.shape[0])
        if n_pts < MIN_CLOUD_POINTS:
            self._emit(
                "[pick] GraspGen-X empty cloud n_pts=%d -- fail closed" % n_pts
            )
            return False
        try:
            pose = best_grasp(
                points,
                generate_fn=generator,
                near_xyz=(
                    float(target.fwd),
                    float(target.lat),
                    float(getattr(target, "z", 0.0) or 0.0),
                ),
            )
        except Exception as exc:
            self._emit("[pick] GraspGen-X failed (%s) -- fail closed" % exc)
            return False
        if pose is None:
            self._emit(
                "[pick] GraspGen-X no grasp n_pts=%d n_keep=0 -- fail closed" % n_pts
            )
            return False
        fwd, lat, z = clamp_grasp_xyz(pose.xyz, None, _grasp_floor_z())
        if not grasp_xyz_in_reach((fwd, lat, z)):
            self._emit(
                "[pick] GraspGen-X out of reach fwd=%.3f lat=%.3f z=%.3f -- fail closed"
                % (fwd, lat, z)
            )
            return False
        target.fwd, target.lat, target.z = fwd, lat, z
        yaw = grasp_yaw_from_matrix(pose.matrix)
        if yaw is not None:
            target.grasp_yaw = yaw
        dyolo = float(getattr(pose, "dyolo", 0.0) or 0.0)
        n_keep = int(getattr(pose, "n_keep", 1) or 0)
        self._emit(
            "[pick] ggx fwd=%.3f lat=%.3f z=%.3f yaw=%.2f n_pts=%d n_keep=%d dyolo=%.3f"
            % (
                fwd, lat, z,
                float(getattr(target, "grasp_yaw", 0.0) or 0.0),
                n_pts,
                n_keep,
                dyolo,
            )
        )
        return True

    def unload_generator(self) -> None:
        self._grasp_generator = None
        self._openvla_policy = None
        self._openvla_load_attempted = False

    def _grasp(self, target) -> None:
        if self._grasp_fn is not None:
            self._grasp_fn(target)
            return
        if self.policy == "wrist":
            self._ik_grasp(target)
            return
        if self.policy == "graspgenx":
            if not self._apply_graspgenx(target):
                return
        elif self.policy == "openvla_geom":
            policy_obj = GeometricGraspStub((target.fwd, target.lat, target.z))
            self._geom_or_openvla_grasp(target, policy_obj)
            return
        elif self.policy == "openvla":
            if not self._openvla_load_attempted:
                self._openvla_policy = load_openvla()
                self._openvla_load_attempted = True
            if self._openvla_policy is not None:
                self._geom_or_openvla_grasp(target, self._openvla_policy)
                return
            if not self._openvla_fallback_logged:
                self._emit("[pick] OpenVLA unavailable -- IK fallback")
                self._openvla_fallback_logged = True
        elif self.policy == "openvla_remote":
            if not self._openvla_load_attempted:
                self._openvla_policy = RemoteOpenVLAPolicy.from_env()
                self._openvla_load_attempted = True
            self._geom_or_openvla_grasp(target, self._openvla_policy)
            return
        self._ik_grasp(target)

    def _place(self) -> None:
        if self._place_fn is not None:
            self._place_fn()
            return
        arm = self._teleop_arm()
        self._seed_closed_grip()
        arm.place_in_basket(on_step=None)

    def _recover(self) -> None:
        self._teleop = None
        if self._recover_fn is not None:
            self._recover_fn()
            return
        try:
            from arm_pick import GRIPPER_OPEN, HOME, _make_js, _read_arm_q, _kin, load_params
            import arm_kin
            client = self.client
            q = _read_arm_q(client)
            if q is not None:
                kin = _kin()
                p = load_params()
                cur_ee = kin.fk(q)[:3, 3]
                ascend_z = float(p.get("carry_up", 0.246)) + 0.10
                tgt = np.array(
                    [cur_ee[0], cur_ee[1], max(float(cur_ee[2]), ascend_z)],
                    dtype=float,
                )
                hold_quat = arm_kin.R_to_quat(kin.fk(arm_kin.DEFAULT_ARM_Q)[:3, :3])
                tpos = np.array(cur_ee, dtype=float)
                pos_step = 0.006
                dt = 0.05
                for _ in range(50):
                    aq = _read_arm_q(client)
                    if aq is None:
                        self._sleep(dt)
                        continue
                    d = tgt - tpos
                    dn = float(np.linalg.norm(d))
                    tpos = tgt if dn <= pos_step else tpos + d / dn * pos_step
                    cmd = arm_kin.dls_solve(kin, aq, tpos, hold_quat)
                    client.send_arm_command(_make_js(cmd, GRIPPER_OPEN))
                    self._sleep(dt)
                    if float(np.linalg.norm(kin.fk(aq)[:3, 3] - tgt)) < 0.02:
                        break
            client.send_arm_command(_make_js(HOME, GRIPPER_OPEN))
            self._sleep(0.4)
            self._emit("[pick] recover -> HOME (gripper open)")
        except Exception as exc:
            self._emit("[pick] recover failed: %s" % exc)

    def _reacquire(self, target):
        if self._reacquire_fn is None:
            return target
        nxt = self._reacquire_fn()
        if nxt is None:
            return None
        if getattr(nxt, "valid", True) is False:
            return None
        return nxt

    def run(self, target, deadline=None, reveal=None) -> PickResult:
        """Pick ``target`` into the basket. ``ok`` only if the basket is occupied."""
        del reveal  # reserved for callers that close over reveal in reacquire_fn
        policy = self.policy
        if target is None or not getattr(target, "valid", True):
            return PickResult(
                ok=False, reason=REASON_LOST_TARGET, attempts=0,
                grasped=False, basket_occupied=False, policy=policy,
            )
        still_holding = False
        last_reason = REASON_NO_GRASP
        grasped_any = False
        attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            if self._past_deadline(deadline):
                return PickResult(
                    ok=False, reason=REASON_TIMEOUT, attempts=attempts,
                    grasped=grasped_any, basket_occupied=False, policy=policy,
                )
            if self._session_ended():
                return PickResult(
                    ok=False, reason=REASON_TIMEOUT, attempts=attempts,
                    grasped=grasped_any, basket_occupied=False, policy=policy,
                )
            self._emit("[pick] attempt %d/%d policy=%s" % (
                attempt, self.max_attempts, policy))
            if not still_holding:
                try:
                    self._grasp(target)
                except Exception as exc:
                    self._emit("[pick] grasp raised: %s" % exc)
                held = self.poll_grasping()
                if held is not True:
                    last_reason = REASON_NO_GRASP
                    self._emit("[pick] no positive is_grasping (got %s)" % held)
                    self._recover()
                    nxt = self._reacquire(target)
                    if nxt is None:
                        last_reason = REASON_LOST_TARGET
                        return PickResult(
                            ok=False, reason=last_reason, attempts=attempts,
                            grasped=False, basket_occupied=False, policy=policy,
                        )
                    target = nxt
                    continue
                grasped_any = True
                still_holding = True
            try:
                self._place()
            except Exception as exc:
                self._emit("[pick] place raised: %s" % exc)
            basket = self.poll_basket()
            if basket is True:
                self._emit("[pick] basket occupied after attempt %d" % attempt)
                return PickResult(
                    ok=True, reason=REASON_OK, attempts=attempts,
                    grasped=True, basket_occupied=True, policy=policy,
                )
            still = self.poll_grasping()
            if still is True:
                last_reason = REASON_NO_BASKET
                still_holding = True
                self._emit("[pick] still holding, basket empty -- retry place")
            else:
                last_reason = REASON_DROP if grasped_any else REASON_NO_BASKET
                still_holding = False
                self._emit("[pick] drop or miss (grasping=%s basket=%s)" % (still, basket))
                self._recover()
                nxt = self._reacquire(target)
                if nxt is None:
                    return PickResult(
                        ok=False, reason=REASON_LOST_TARGET, attempts=attempts,
                        grasped=grasped_any, basket_occupied=False, policy=policy,
                    )
                target = nxt
        return PickResult(
            ok=False, reason=last_reason, attempts=attempts,
            grasped=grasped_any, basket_occupied=False, policy=policy,
        )
