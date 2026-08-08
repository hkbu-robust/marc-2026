"""State-machine app: manual base -> auto approach -> arm teleop -> sensor success."""

from __future__ import annotations

import enum
import os
import sys
import time

from arm_pick import load_params

from .arm_control import ArmTeleop, TeleopConfig
from .client import ManipTrainerClient, report_camera_publishers, wait_for_robot_topics
from .depth_approach import ApproachConfig, DepthApproachController
from .keyboard import HELP_TEXT, Keyboard
from .recorder import EpisodeRecorder
from .success import evaluate_success


class Phase(enum.Enum):
    MANUAL_BASE = "MANUAL_BASE"
    AUTO_APPROACH = "AUTO_APPROACH"
    ARM_TELEOP = "ARM_TELEOP"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _dataset_root() -> str:
    return os.environ.get(
        "MARC_DATASET_ROOT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "marc_pnp"),
    )


class CollectorApp:
    def __init__(self, client: ManipTrainerClient, team_id: str):
        self.client = client
        self.team_id = team_id
        self.phase = Phase.MANUAL_BASE
        self.arm = ArmTeleop(client, TeleopConfig.from_env())
        self.approach = DepthApproachController(client, ApproachConfig.from_env())
        self.recorder = EpisodeRecorder(
            _dataset_root(), team_id, primary_camera=self.arm.cfg.primary_camera
        )
        self.base_lin = _env_float("MARC_BASE_LIN", 0.08)
        # Reverse magnitude for `k` (defaults to forward speed if unset).
        self.base_lin_back = _env_float("MARC_BASE_LIN_BACK", self.base_lin)
        self.base_ang = _env_float("MARC_BASE_ANG", 0.25)
        self._base_cmd = (0.0, 0.0)
        self._running = True
        self._episode_armed = False  # True after ready; recording starts on first arm action

    def print_help(self):
        print(HELP_TEXT)
        print(f"[state] phase={self.phase.value}  steps={len(self.recorder.steps)}")

    def begin_episode_cycle(self):
        """Ready pose + enter MANUAL_BASE; recording not started until arm teleop."""
        self.client.send_cmd_vel(0.0, 0.0)
        self._base_cmd = (0.0, 0.0)
        self.approach.cancel()
        print("[episode] moving arm to down-tilt ready pose...")
        self.arm.warm_cameras(1.0)
        self.arm.go_ready()
        pose = self.client.get_world_pose()
        self.recorder.start_episode({
            "pnp_params": load_params(),
            "spawn_world_pose": list(pose) if pose else None,
        })
        # Do not record until ARM_TELEOP actions — clear buffer start marker
        self.recorder.steps = []
        self.recorder.recording = False
        self._episode_armed = True
        self.phase = Phase.MANUAL_BASE
        print("[episode] MANUAL_BASE — drive with i/k/j/l, then press A for auto-approach")
        self.print_help()

    def _ensure_recording(self):
        if not self.recorder.recording and self._episode_armed:
            pose = self.client.get_world_pose()
            if not self.recorder.meta:
                self.recorder.start_episode({"pnp_params": load_params()})
            self.recorder.meta["approach_world_pose"] = list(pose) if pose else None
            self.recorder.recording = True
            print(f"[record] started episode {self.recorder.meta.get('episode_id')}")

    def _on_step(self, obs, action):
        self._ensure_recording()
        self.recorder.add_step(obs, action)

    def _set_base(self, lin: float, ang: float):
        self._base_cmd = (lin, ang)
        self.client.send_cmd_vel(linear_x=lin, angular_z=ang)

    def handle_key(self, ch: str):
        if not ch:
            return
        if ch == "?":
            self.print_help()
            return
        if ch in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
            self._running = False
            return
        if ch == "N":
            self._abort_episode()
            return
        if ch == "C":
            if self.phase == Phase.AUTO_APPROACH:
                self.approach.cancel()
                self.phase = Phase.MANUAL_BASE
                print("[approach] cancelled -> MANUAL_BASE")
            return

        if self.phase == Phase.MANUAL_BASE:
            self._keys_manual_base(ch)
        elif self.phase == Phase.AUTO_APPROACH:
            # only C handled above; ignore arm
            pass
        elif self.phase == Phase.ARM_TELEOP:
            self._keys_arm(ch)

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
            print("[approach] capturing depth baseline...")
            if not self.approach.start():
                print("[approach] FAILED — no base depth; stay in MANUAL_BASE")
                return
            self.phase = Phase.AUTO_APPROACH
            print("[approach] AUTO_APPROACH crawling (C to cancel)...")
        elif ch == "v":
            self.arm.set_grip(self.arm.cfg.grip_close)
            print(f"[arm] GRIP CLOSE (no record)  grip={self.arm.grip_cmd:.3f}"
                  f"  grasping={self.client.is_grasping()}")
        elif ch == "b":
            self.arm.set_grip(0.0)
            print("[arm] GRIP OPEN (no record) — object released")
        elif ch == "E":
            print("[warn] end attempt only valid in ARM_TELEOP after approach")
        else:
            # allow accidental arm keys hint
            if ch in "tgfhryuo,":
                print("[warn] arm teleop unlocked after auto-approach stop")

    def _keys_arm(self, ch: str):
        cfg = self.arm.cfg
        if ch == "t":
            n = self.arm.apply_delta(dx=cfg.dx, on_step=self._on_step)
            print(f"[arm] +X  steps+={n}")
        elif ch == "g":
            n = self.arm.apply_delta(dx=-cfg.dx, on_step=self._on_step)
            print(f"[arm] -X  steps+={n}")
        elif ch == "f":
            n = self.arm.apply_delta(dy=cfg.dy, on_step=self._on_step)
            print(f"[arm] +Y  steps+={n}")
        elif ch == "h":
            n = self.arm.apply_delta(dy=-cfg.dy, on_step=self._on_step)
            print(f"[arm] -Y  steps+={n}")
        elif ch == "r":
            n = self.arm.apply_delta(dz=cfg.dz, on_step=self._on_step)
            print(f"[arm] +Z  steps+={n}")
        elif ch == "y":
            n = self.arm.apply_delta(dz=-cfg.dz, on_step=self._on_step)
            print(f"[arm] -Z  steps+={n}")
        elif ch == "u":
            n = self.arm.apply_delta(dyaw=cfg.d_yaw, on_step=self._on_step)
            print(f"[arm] +yaw  steps+={n}")
        elif ch == "o":
            n = self.arm.apply_delta(dyaw=-cfg.d_yaw, on_step=self._on_step)
            print(f"[arm] -yaw  steps+={n}")
        elif ch == ",":
            n = self.arm.apply_delta(dpitch=cfg.d_pitch, on_step=self._on_step)
            print(f"[arm] +pitch  steps+={n}")
        elif ch == ".":
            n = self.arm.apply_delta(dpitch=-cfg.d_pitch, on_step=self._on_step)
            print(f"[arm] -pitch  steps+={n}")
        elif ch == "v":
            n = self.arm.apply_delta(grip=cfg.grip_close, on_step=self._on_step)
            print(f"[arm] GRIP CLOSE  steps+={n}  grip={self.arm.grip_cmd:.3f}"
                  f"  grasping={self.client.is_grasping()}"
                  f"  grasp_latched={self.arm.saw_grasp_after_close}")
        elif ch == "b":
            n = self.arm.apply_delta(grip=0.0, on_step=self._on_step)
            print(f"[arm] GRIP OPEN  steps+={n}")
        elif ch == "P":
            self._place_in_basket()
        elif ch == "E":
            self._end_attempt()
        elif ch in "ikjl":
            print("[warn] base keys disabled in ARM_TELEOP (finish with E or N)")

    def _place_in_basket(self):
        if self.client.is_grasping() is False:
            print("[place] not holding anything — grasp with v first (running anyway)")
        # Scripted place is hard-coded and must not enter the training set — the
        # dataset is approach + grasp only. Basket occupancy is still checked at E.
        self.arm.place_in_basket(on_step=None)
        basket = self._wait_basket_occupied()
        print(f"[place] done (not recorded)  basket={basket}"
              f"{'  — press E to keep' if basket is True else '  — drop missed, retry or N'}")

    def _wait_basket_occupied(self, timeout_s: float = 2.0):
        """Poll basket occupancy; the sensor lags the physical drop by a few frames."""
        deadline = time.monotonic() + timeout_s
        occupied = self.client.is_basket_occupied()
        while occupied is not True and time.monotonic() < deadline:
            time.sleep(0.1)
            occupied = self.client.is_basket_occupied()
        return occupied

    def _abort_episode(self):
        self.client.send_cmd_vel(0.0, 0.0)
        self.approach.cancel()
        path = None
        if self.recorder.steps:
            try:
                path = self.recorder.discard(save_failed=True)
            except Exception as e:
                print(f"[record] discard write failed: {e}")
                self.recorder.steps = []
                self.recorder.recording = False
        else:
            self.recorder.recording = False
            self.recorder.steps = []
        print(f"[episode] DISCARDED (abort){f' -> {path}' if path else ''}")
        self.begin_episode_cycle()

    def _end_attempt(self):
        self.client.send_cmd_vel(0.0, 0.0)
        self.arm.refresh_grasp()
        basket = self._wait_basket_occupied()
        ok = evaluate_success(
            saw_grasp_after_close=self.arm.saw_grasp_after_close,
            basket_occupied=basket,
        )
        print(f"[eval] grasp_after_close={self.arm.saw_grasp_after_close}  "
              f"basket={basket}  keep={ok}  steps={len(self.recorder.steps)}")
        try:
            if ok and self.recorder.steps:
                path = self.recorder.keep()
                print(f"[episode] KEPT -> {path}")
            else:
                path = self.recorder.discard(save_failed=True) if self.recorder.steps else None
                print(f"[episode] DISCARDED{f' -> {path}' if path else ''}")
        except Exception as e:
            print(f"[record] write failed: {e}")
            self.recorder.steps = []
            self.recorder.recording = False
        self.begin_episode_cycle()

    def run(self):
        self.begin_episode_cycle()
        with Keyboard() as kb:
            try:
                while self._running:
                    ch = kb.poll()
                    if ch:
                        self.handle_key(ch)

                    if self.phase == Phase.AUTO_APPROACH:
                        if self.approach.tick():
                            self.approach.settle()
                            self.phase = Phase.ARM_TELEOP
                            print("[state] ARM_TELEOP — nudge arm, v/b grip, E to end")
                        time.sleep(self.approach.cfg.tick_s)
                    elif self.phase == Phase.MANUAL_BASE:
                        # re-publish last base cmd lightly so stop doesn't decay oddly
                        time.sleep(0.05)
                    else:
                        time.sleep(0.02)
            except KeyboardInterrupt:
                print("\n[quit] KeyboardInterrupt")
            finally:
                self.client.send_cmd_vel(0.0, 0.0)
                self.approach.cancel()


def main():
    team_id = os.getenv("MARC_TEAM_ID", "u1").strip() or "u1"
    if team_id != "u1":
        print(
            f"ERROR: manip-trainer publishes under /marc/u1/...; "
            f"refusing MARC_TEAM_ID={team_id}. Export MARC_TEAM_ID=u1.",
            file=sys.stderr,
        )
        sys.exit(2)

    client = ManipTrainerClient(team_id=team_id, token=os.getenv("MARC_TOKEN", "dummy-token"))
    print(f"[probe] team_id={team_id} namespace=/marc/{team_id}")
    print("[probe] teleop dataset collector (no SESSION_ACK)")

    if not client.connect():
        print("failed to connect", file=sys.stderr)
        sys.exit(1)

    if not wait_for_robot_topics(client, team_id):
        client.shutdown()
        sys.exit(1)

    report_camera_publishers(client, team_id)

    app = CollectorApp(client, team_id)
    try:
        app.run()
    finally:
        client.stop()
        client.shutdown()
        print("[quit] shutdown complete")


if __name__ == "__main__":
    main()
