"""Keyboard pick tester against manip-trainer. No dataset recording."""

from __future__ import annotations

import enum
import os
import sys
import time

from marc_agent.gpu_handoff import (
    after_successful_pick,
    before_graspgenx,
    before_reacquire,
)
from marc_agent.pick_runtime import PickRuntime, pick_policy_name

from .base_hold import StationaryBase
from .depth_approach import ApproachConfig, DepthApproachController
from .graspgenx_pick import StagedGraspGenXPickRuntime
from .keyboard import Keyboard
from .openvla_pick import StagedOpenVLAPickRuntime, openvla_target_is_safe
from .pick_target import nearfield_for_approach, snapshot_pick_target, stable_pick_target


class Phase(enum.Enum):
    MANUAL_BASE = "MANUAL_BASE"
    AUTO_APPROACH = "AUTO_APPROACH"


# Keep in sync with demo/manip_dataset/app.py TARGET_OBJECTS.
TARGET_OBJECTS = [
    "tumbler",
    "cracker_box",
    "cola_can",
    "disposable_cup",
    "umbrella",
    "juice",
    "tissue",
    "mug",
    "sunblock",
    "pencilcase",
]


HELP_TEXT = """
=== Manip-trainer GraspGen-X pick tester (no dataset) ===
MANUAL_BASE:
  i/k     base forward / back
  j/l     base yaw left / right
  space   stop base
  A       start calibrated auto-approach (required before V if target is out of range)
  G       snapshot + GraspGen-X pick + basket drop
  H       wrist-YOLO pick + basket
  V       zero-shot OpenVLA pick + basket
ANY:
  O       cycle target object (must match the spawned class before A/G/H/V)
  C       cancel auto-approach (-> MANUAL_BASE)
  N       recover / ready pose (nothing is saved)
  ?       help
  Ctrl-C  quit
Trainer GUI: choose object, use A to enter calibrated OpenVLA range, then press V.
===========================================
"""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class PickTestApp:
    """Drive, then test one pick policy against the rear basket."""

    def __init__(
        self,
        client,
        team_id: str,
        *,
        arm=None,
        approach=None,
        nearfield=None,
        pick_runtime=None,
    ):
        self.client = client
        self.team_id = team_id
        self.phase = Phase.MANUAL_BASE
        self.approach = approach or DepthApproachController(
            client, ApproachConfig.from_env()
        )
        if arm is None:
            from .arm_control import ArmTeleop, TeleopConfig
            arm = ArmTeleop(client, TeleopConfig.from_env())
        self.arm = arm
        if nearfield is None:
            nearfield = nearfield_for_approach(client, self.approach)
        self.nearfield = nearfield
        self._pick = pick_runtime
        default_obj = os.environ.get("MARC_TARGET_OBJECT", TARGET_OBJECTS[0])
        self._obj_idx = (
            TARGET_OBJECTS.index(default_obj) if default_obj in TARGET_OBJECTS else 0
        )
        self.target_object = TARGET_OBJECTS[self._obj_idx]
        self.approach.set_target(self.target_object)
        self.base_lin = _env_float("MARC_BASE_LIN", 0.08)
        self.base_lin_back = _env_float("MARC_BASE_LIN_BACK", self.base_lin)
        self.base_ang = _env_float("MARC_BASE_ANG", 0.25)
        self._base_cmd = (0.0, 0.0)
        self._base_hold = StationaryBase(client, log=print)
        self._running = True

    def print_help(self):
        print(HELP_TEXT)
        print(
            "[state] phase=%s  target=%s  policy=%s"
            % (self.phase.value, self.target_object, pick_policy_name())
        )

    def begin(self):
        self._base_hold.stop()
        self.client.send_cmd_vel(0.0, 0.0)
        self._base_cmd = (0.0, 0.0)
        self.approach.cancel()
        print("[pick] moving arm to ready pose...")
        warm = getattr(self.arm, "warm_cameras", None)
        if callable(warm):
            warm(1.0)
        go = getattr(self.arm, "go_ready", None)
        if callable(go):
            go()
        if not self.approach.detector_ready():
            print("[yolo] loading detector (once)...")
            if not self.approach.ensure_detector():
                print("[yolo] detector not ready — press G after fixing weights / rebuild")
        self.phase = Phase.MANUAL_BASE
        print("[pick] MANUAL_BASE — Place near then G/H, or i/k/j/l + A then G/H")
        self.print_help()

    def _set_base(self, lin: float, ang: float):
        self._base_hold.stop()
        self._base_cmd = (lin, ang)
        self.client.send_cmd_vel(linear_x=lin, angular_z=ang)

    def _ensure_pick(self, policy="graspgenx"):
        current = None if self._pick is None else getattr(self._pick, "policy", None)
        if self._pick is None or (current is not None and current != policy):
            if policy.startswith("openvla"):
                self._pick = StagedOpenVLAPickRuntime(
                    self.client,
                    log=print,
                    policy=policy,
                    target_observer=lambda: self.nearfield.snapshot(self.target_object),
                )
            elif policy == "graspgenx":
                self._pick = StagedGraspGenXPickRuntime(
                    self.client,
                    log=print,
                    policy=policy,
                )
            else:
                self._pick = PickRuntime(self.client, log=print, policy=policy)
        return self._pick

    def handle_key(self, ch: str):
        if not ch:
            return
        if ch == "?":
            self.print_help()
            return
        if ch in ("\x03", "\x04"):
            self._running = False
            return
        if ch == "O":
            self._obj_idx = (self._obj_idx + 1) % len(TARGET_OBJECTS)
            self.target_object = TARGET_OBJECTS[self._obj_idx]
            self.approach.set_target(self.target_object)
            print("[target] object → %s" % self.target_object)
            return
        if ch == "N":
            self._reset()
            return
        if ch == "C":
            if self.phase == Phase.AUTO_APPROACH:
                self.approach.cancel()
                self.phase = Phase.MANUAL_BASE
                print("[approach] cancelled -> MANUAL_BASE")
            return
        if self.phase == Phase.MANUAL_BASE:
            self._keys_manual_base(ch)

    def _keys_manual_base(self, ch: str):
        lin, ang = self._base_cmd
        if ch == "i":
            self._set_base(self.base_lin, ang)
        elif ch == "k":
            self._set_base(-self.base_lin_back, ang)
        elif ch == "j":
            self._set_base(lin, self.base_ang)
        elif ch == "l":
            self._set_base(lin, -self.base_ang)
        elif ch == " ":
            self._set_base(0.0, 0.0)
        elif ch == "A":
            self._set_base(0.0, 0.0)
            print(
                "[approach] locking cameras + YOLO "
                f"target={self.target_object}..."
            )
            started = self.approach.start()
            if not started:
                reason = getattr(self.approach, "last_failure", "") or "camera lock failed"
                print(f"[approach] FAILED — {reason}; stay in MANUAL_BASE")
                return
            self.phase = Phase.AUTO_APPROACH
            print("[approach] AUTO_APPROACH crawling (C to cancel)...")
        elif ch == "G":
            self._graspgenx_pick()
        elif ch == "H":
            self._wrist_pick()
        elif ch == "V":
            self._openvla_pick()

    def _snapshot_target(self):
        return snapshot_pick_target(self.nearfield, self.target_object, log=print)

    def _print_pick_result(self, result):
        print(
            "[pick] ok=%s reason=%s attempts=%s grasping=%s basket=%s policy=%s"
            % (
                result.ok,
                result.reason,
                result.attempts,
                result.grasped,
                result.basket_occupied,
                result.policy,
            )
        )

    def _graspgenx_pick(self):
        self._set_base(0.0, 0.0)
        # Hold forward position against gravity while leaving yaw untouched so
        # arm reaction cannot rotate the carefully composed camera view.
        grasp_hold = StationaryBase(self.client, log=print, yaw_kp=0.0)
        with grasp_hold:
            locked_target = getattr(self.approach, "locked_target", None)
            posture_hint = (
                locked_target(self.target_object)
                if callable(locked_target)
                else None
            )
            target = stable_pick_target(
                self.nearfield,
                self.target_object,
                log=print,
                posture_hint=posture_hint,
            )
            if target is None:
                print("[pick] GraspGen-X found no safe high-confidence localization")
                return
            detector = getattr(self.nearfield, "detector", None)
            before_graspgenx(detector)
            runtime = self._ensure_pick("graspgenx")

            def _observe_target():
                before_reacquire(runtime, detector)
                return stable_pick_target(
                    self.nearfield,
                    self.target_object,
                    log=print,
                    require_reach=False,
                )

            if isinstance(runtime, StagedGraspGenXPickRuntime):
                runtime._target_observer = _observe_target

                def _observe_visual_target():
                    before_reacquire(runtime, detector)
                    return stable_pick_target(
                        self.nearfield,
                        self.target_object,
                        log=print,
                        visual=True,
                        require_reach=False,
                    )

                runtime._visual_target_observer = _observe_visual_target

            def _reacquire():
                before_reacquire(runtime, detector)
                nxt = stable_pick_target(
                    self.nearfield, self.target_object, log=print
                )
                if nxt is not None:
                    before_graspgenx(detector)
                return nxt

            runtime._reacquire_fn = _reacquire
            result = runtime.run(target)
            after_successful_pick(runtime, detector)
            self._print_pick_result(result)

    def _wrist_pick(self):
        self._set_base(0.0, 0.0)
        with self._base_hold:
            target = self._snapshot_target()
            if target is None:
                return
            detector = getattr(self.nearfield, "detector", None)
            runtime = self._ensure_pick("wrist")
            runtime._wrist_detector = detector
            runtime._teleop = self.arm

            def _reacquire():
                return self.nearfield.snapshot(self.target_object)

            runtime._reacquire_fn = _reacquire
            result = runtime.run(target)
            after_successful_pick(runtime, detector)
            self._print_pick_result(result)

    def _openvla_pick(self):
        self._set_base(0.0, 0.0)
        with self._base_hold:
            target = self._snapshot_target()
            if target is None:
                return
            detector = getattr(self.nearfield, "detector", None)
            runtime = self._ensure_pick("openvla_remote")
            if isinstance(runtime, StagedOpenVLAPickRuntime):
                safe, why = openvla_target_is_safe(target)
                if not safe:
                    print(f"[openvla] target invalid: {why}")
                    return
            runtime._teleop = self.arm
            runtime._reacquire_fn = lambda: self.nearfield.snapshot(self.target_object)
            runtime._target_observer = runtime._reacquire_fn
            result = runtime.run(target)
            after_successful_pick(runtime, detector)
            self._print_pick_result(result)

    def _reset(self):
        self._base_hold.stop()
        self.client.send_cmd_vel(0.0, 0.0)
        self._base_cmd = (0.0, 0.0)
        self.approach.cancel()
        recover = getattr(self._pick, "_recover", None) if self._pick is not None else None
        if callable(recover):
            try:
                recover()
            except Exception as exc:
                print("[pick] recover failed: %s" % exc)
        go = getattr(self.arm, "go_ready", None)
        if callable(go):
            go(open_grip=True)
        self.phase = Phase.MANUAL_BASE
        print("[pick] reset -> MANUAL_BASE  target=%s" % self.target_object)

    def run(self):
        self.begin()
        with Keyboard() as kb:
            try:
                while self._running:
                    ch = kb.poll()
                    if ch:
                        self.handle_key(ch)
                    if self.phase == Phase.AUTO_APPROACH:
                        if self.approach.tick():
                            self.approach.settle()
                            self.phase = Phase.MANUAL_BASE
                            print("[state] MANUAL_BASE — press G, H, or V to pick")
                        time.sleep(self.approach.cfg.tick_s)
                    else:
                        time.sleep(0.05)
            except KeyboardInterrupt:
                print("\n[quit] KeyboardInterrupt")
            finally:
                self._base_hold.stop()
                self.client.send_cmd_vel(0.0, 0.0)
                self.approach.cancel()


def main():
    from .client import ManipTrainerClient, report_camera_publishers, wait_for_robot_topics

    team_id = os.getenv("MARC_TEAM_ID", "u1").strip() or "u1"
    client = ManipTrainerClient(
        team_id=team_id, token=os.getenv("MARC_TOKEN", "dummy-token")
    )
    print("[pick] team_id=%s namespace=/marc/%s" % (team_id, team_id))
    print("[pick] GraspGen-X pick tester (no SESSION_ACK, no dataset)")
    if not client.connect():
        print("failed to connect", file=sys.stderr)
        sys.exit(1)
    if not wait_for_robot_topics(client, team_id):
        client.shutdown()
        sys.exit(1)
    report_camera_publishers(client, team_id)
    app = PickTestApp(client, team_id)
    try:
        app.run()
    finally:
        client.stop()
        client.shutdown()
        print("[quit] shutdown complete")


if __name__ == "__main__":
    main()
