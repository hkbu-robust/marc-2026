"""Keyboard pick tester against manip-trainer. G = GraspGen-X, H = wrist-YOLO. No dataset recording."""

from __future__ import annotations

import enum
import os
import sys
import time

from marc_agent.gpu_handoff import (
    after_successful_pick,
    before_graspgenx,
    before_reacquire,
    reload_nearfield_detector,
)
from marc_agent.pick_runtime import PickRuntime, pick_policy_name

from .depth_approach import ApproachConfig, DepthApproachController
from .keyboard import Keyboard


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
  A       start auto-approach (Place far)
  G       snapshot + GraspGen-X pick + basket drop
  H       wrist-YOLO pick + basket
ANY:
  O       cycle target object
  C       cancel auto-approach (-> MANUAL_BASE)
  N       recover / ready pose (nothing is saved)
  ?       help
  Ctrl-C  quit
Trainer GUI: choose object, Place near (or Place far then A), then G or H.
===========================================
"""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class PickTestApp:
    """Drive (optional), then G (GraspGen-X) or H (wrist-YOLO) pick into the rear basket."""

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
            from nearfield import NearFieldAcquire, YoloDetector
            nearfield = NearFieldAcquire(
                client, detector=YoloDetector(self.approach._yolo)
            )
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
        self._running = True

    def print_help(self):
        print(HELP_TEXT)
        print(
            "[state] phase=%s  target=%s  policy=%s"
            % (self.phase.value, self.target_object, pick_policy_name())
        )

    def begin(self):
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
        self._base_cmd = (lin, ang)
        self.client.send_cmd_vel(linear_x=lin, angular_z=ang)

    def _ensure_pick(self, policy="graspgenx"):
        current = None if self._pick is None else getattr(self._pick, "policy", None)
        if self._pick is None or (current is not None and current != policy):
            self._pick = PickRuntime(
                self.client, log=print, policy=policy,
            )
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
            print("[approach] locking cameras + YOLO...")
            if not self.approach.start():
                print("[approach] FAILED — YOLO or cameras unavailable; stay in MANUAL_BASE")
                return
            self.phase = Phase.AUTO_APPROACH
            print("[approach] AUTO_APPROACH crawling (C to cancel)...")
        elif ch == "G":
            self._graspgenx_pick()
        elif ch == "H":
            self._wrist_pick()

    def _snapshot_target(self):
        detector = getattr(self.nearfield, "detector", None)
        reload_nearfield_detector(detector)
        print("[pick] snapshot %s..." % self.target_object)
        target = self.nearfield.snapshot(self.target_object)
        if target is None or not getattr(target, "valid", False):
            reason = "invalid" if target is None else (target.reason or "invalid")
            if target is None:
                print("[pick] snapshot failed (%s)" % reason)
            else:
                print(
                    "[pick] snapshot failed (%s) fwd=%.3f lat=%.3f z=%.3f"
                    % (
                        reason,
                        float(getattr(target, "fwd", 0.0) or 0.0),
                        float(getattr(target, "lat", 0.0) or 0.0),
                        float(getattr(target, "z", 0.0) or 0.0),
                    )
                )
            return None
        print(
            "[pick] localized %s fwd=%.3f lat=%.3f z=%.3f conf=%.2f"
            % (
                getattr(target, "name", "") or self.target_object,
                float(getattr(target, "fwd", 0.0) or 0.0),
                float(getattr(target, "lat", 0.0) or 0.0),
                float(getattr(target, "z", 0.0) or 0.0),
                float(getattr(target, "conf", 0.0) or 0.0),
            )
        )
        return target

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
        target = self._snapshot_target()
        if target is None:
            return
        detector = getattr(self.nearfield, "detector", None)
        before_graspgenx(detector)
        runtime = self._ensure_pick("graspgenx")

        def _reacquire():
            before_reacquire(runtime, detector)
            nxt = self.nearfield.snapshot(self.target_object)
            if nxt is not None and getattr(nxt, "valid", False):
                before_graspgenx(detector)
            return nxt

        runtime._reacquire_fn = _reacquire
        result = runtime.run(target)
        after_successful_pick(runtime, detector)
        self._print_pick_result(result)

    def _wrist_pick(self):
        self._set_base(0.0, 0.0)
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

    def _reset(self):
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
                            print("[state] MANUAL_BASE — press G for GraspGen-X, H for wrist-YOLO")
                        time.sleep(self.approach.cfg.tick_s)
                    else:
                        time.sleep(0.05)
            except KeyboardInterrupt:
                print("\n[quit] KeyboardInterrupt")
            finally:
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
