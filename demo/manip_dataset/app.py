"""State-machine app: manual base -> auto approach -> arm teleop -> sensor success."""

from __future__ import annotations

import enum
import os
import sys
import threading
import time

from arm_pick import load_params

from .arm_control import ArmTeleop, TeleopConfig
from .base_hold import StationaryBase
from .client import ManipTrainerClient, report_camera_publishers, wait_for_robot_topics
from .depth_approach import ApproachConfig, DepthApproachController
from .keyboard import HELP_TEXT, Keyboard
from .pick_target import nearfield_for_approach, snapshot_pick_target
from .recorder import EpisodeRecorder
from .success import evaluate_success


class Phase(enum.Enum):
    MANUAL_BASE = "MANUAL_BASE"
    AUTO_APPROACH = "AUTO_APPROACH"
    ARM_TELEOP = "ARM_TELEOP"


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
HANDOFF_ATTEMPTS = 3
HANDOFF_RETRY_CAMERA_S = 0.5


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
        self.nearfield = nearfield_for_approach(client, self.approach)
        self.recorder = EpisodeRecorder(
            _dataset_root(), team_id, primary_camera=self.arm.cfg.primary_camera
        )
        default_obj = os.environ.get("MARC_TARGET_OBJECT", TARGET_OBJECTS[0])
        self._obj_idx = TARGET_OBJECTS.index(default_obj) if default_obj in TARGET_OBJECTS else 0
        self.target_object = TARGET_OBJECTS[self._obj_idx]
        self.approach.set_target(self.target_object)
        self.base_lin = _env_float("MARC_BASE_LIN", 0.08)
        # Reverse magnitude for `k` (defaults to forward speed if unset).
        self.base_lin_back = _env_float("MARC_BASE_LIN_BACK", self.base_lin)
        self.base_ang = _env_float("MARC_BASE_ANG", 0.25)
        self._base_cmd = (0.0, 0.0)
        self._base_hold = StationaryBase(client, log=print)
        self._running = True
        self._episode_armed = False  # True after ready; recording starts on first arm action
        self._keyboard = None

    def print_help(self):
        print(HELP_TEXT)
        print(f"[state] phase={self.phase.value}  steps={len(self.recorder.steps)}  target={self.target_object}")

    def begin_episode_cycle(self, open_grip: bool = False):
        """Ready pose + enter MANUAL_BASE; recording not started until arm teleop."""
        self._base_hold.stop()
        self.client.send_cmd_vel(0.0, 0.0)
        self._base_cmd = (0.0, 0.0)
        self.approach.cancel()
        print("[episode] moving arm to down-tilt ready pose...")
        self.arm.warm_cameras(1.0)
        self.arm.go_ready(open_grip=open_grip)
        if not self.approach.detector_ready():
            print("[yolo] loading detector (once)...")
            if not self.approach.ensure_detector():
                print("[yolo] detector not ready — press A after fixing weights / rebuild")
        pose = self.client.get_world_pose()
        self.recorder.start_episode({
            "pnp_params": load_params(),
            "spawn_world_pose": list(pose) if pose else None,
        }, object_name=self.target_object)
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
                self.recorder.start_episode({"pnp_params": load_params()}, object_name=self.target_object)
            self.recorder.meta["approach_world_pose"] = list(pose) if pose else None
            self.recorder.recording = True
            print(f"[record] started episode {self.recorder.meta.get('episode_id')}")

    def _on_step(self, obs, action):
        self._ensure_recording()
        self.recorder.add_step(obs, action)

    def _stage_manual_handoff(self) -> bool:
        """Enter the exact deterministic pose used before OpenVLA policy control."""
        target = snapshot_pick_target(self.nearfield, self.target_object)
        if target is None:
            target = self.approach.locked_target(self.target_object)
            if target is None:
                return False
            print(
                f"[handoff] fresh {self.target_object} snapshot missed — "
                "using maintained lock "
                f"fwd={target.fwd:.3f} lat={target.lat:.3f} z={target.z:.3f}"
            )
        if not self.arm.stage_for_openvla(target):
            print("[handoff] deterministic OpenVLA stage failed")
            return False
        if self.recorder.meta is not None:
            self.recorder.meta["openvla_handoff"] = {
                "target_fwd": float(target.fwd),
                "target_lat": float(target.lat),
                "target_z": float(target.z),
                "stage_back_m": float(self.arm.cfg.openvla_stage_back_m),
                "stage_min_forward_m": float(
                    self.arm.cfg.openvla_stage_min_forward_m
                ),
                "stage_up_m": float(self.arm.cfg.openvla_stage_up_m),
                "primary_camera": self.arm.cfg.primary_camera,
            }
        print(
            "[handoff] deterministic stage complete; "
            "manual demonstrator now replaces OpenVLA from the same pose"
        )
        return True

    def _stage_manual_handoff_with_retries(self) -> bool:
        """Try a fresh localization and deterministic handoff up to three times."""
        for attempt in range(1, HANDOFF_ATTEMPTS + 1):
            print(f"[handoff] attempt {attempt}/{HANDOFF_ATTEMPTS}")
            if self._stage_manual_handoff():
                return True
            if attempt == HANDOFF_ATTEMPTS:
                break
            print(
                f"[handoff] attempt {attempt} failed — returning to ready "
                "for a fresh snapshot"
            )
            self.arm.go_ready(open_grip=True)
            self.arm.warm_cameras(HANDOFF_RETRY_CAMERA_S)
        return False

    def _set_base(self, lin: float, ang: float):
        self._base_hold.stop()
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
        if ch == "O":
            self._obj_idx = (self._obj_idx + 1) % len(TARGET_OBJECTS)
            self.target_object = TARGET_OBJECTS[self._obj_idx]
            self.approach.set_target(self.target_object)
            print(f"[target] object → {self.target_object}")
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
            print("[approach] locking cameras + YOLO...")
            if not self.approach.start():
                print("[approach] FAILED — YOLO or cameras unavailable; stay in MANUAL_BASE")
                return
            self.phase = Phase.AUTO_APPROACH
            print("[approach] AUTO_APPROACH crawling (C to cancel)...")
        elif ch == "v":
            self._set_base(0.0, 0.0)
            with self._base_hold:
                self.arm.set_grip(self.arm.cfg.grip_close)
            print(f"[arm] GRIP CLOSE (no record)  grip={self.arm.grip_cmd:.3f}"
                  f"  grasping={self.client.is_grasping()}")
        elif ch == "b":
            self._set_base(0.0, 0.0)
            with self._base_hold:
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
        # dataset is approach + grasp only. A later E uses the basket sensor; an E
        # during this motion is explicit human approval and cancels the trajectory.
        end_requested = threading.Event()
        stop_watcher = threading.Event()
        watcher = None
        if self._keyboard is not None:
            watcher = threading.Thread(
                target=self._watch_place_keys,
                args=(stop_watcher, end_requested),
                name="place-key-watcher",
                daemon=True,
            )
            watcher.start()
        try:
            self.arm.place_in_basket(on_step=None, cancel_event=end_requested)
        finally:
            stop_watcher.set()
            if watcher is not None:
                watcher.join(timeout=0.25)

        if end_requested.is_set():
            print("[place] scripted placement stopped — keeping episode and resetting")
            self._end_attempt(manual_accept=True, open_grip_on_reset=True)
            return

        basket = self._wait_basket_occupied()
        print(f"[place] done (not recorded)  basket={basket}"
              f"{'  — press E to keep' if basket is True else '  — drop missed, retry or N'}")

    def _watch_place_keys(self, stop_watcher, end_requested):
        """Turn E into a placement cancel while the scripted motion blocks run()."""
        while not stop_watcher.wait(0.02):
            ch = self._keyboard.poll() if self._keyboard is not None else None
            if ch == "E":
                end_requested.set()
                print("[place] E accepted — stopping placement now")
                return
            if ch in ("\x03", "\x04"):
                self._running = False
                return

    def _wait_basket_occupied(self, timeout_s: float = 2.0):
        """Poll basket occupancy; the sensor lags the physical drop by a few frames."""
        deadline = time.monotonic() + timeout_s
        occupied = self.client.is_basket_occupied()
        while occupied is not True and time.monotonic() < deadline:
            time.sleep(0.1)
            occupied = self.client.is_basket_occupied()
        return occupied

    def _abort_episode(self):
        self._base_hold.stop()
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

    def _end_attempt(self, manual_accept: bool = False, open_grip_on_reset: bool = False):
        self._base_hold.stop()
        self.client.send_cmd_vel(0.0, 0.0)
        if manual_accept:
            basket = self.client.is_basket_occupied()
            ok = True
            print(f"[eval] manual E approval during placement  basket={basket}  "
                  f"keep=True  steps={len(self.recorder.steps)}")
        else:
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
        self.begin_episode_cycle(open_grip=open_grip_on_reset)

    def run(self):
        self.begin_episode_cycle()
        with Keyboard() as kb:
            self._keyboard = kb
            try:
                while self._running:
                    ch = kb.poll()
                    if ch:
                        self.handle_key(ch)

                    if self.phase == Phase.AUTO_APPROACH:
                        if self.approach.tick():
                            self._base_hold.start()
                            self.approach.settle()
                            if self._stage_manual_handoff_with_retries():
                                self.phase = Phase.ARM_TELEOP
                                print(
                                    "[state] ARM_TELEOP — base-hold ACTIVE; "
                                    "OpenVLA-equivalent handoff pose; nudge arm, "
                                    "v/b grip, E to end"
                                )
                            else:
                                print(
                                    "[handoff] all 3 attempts failed — returning arm to ready; "
                                    "stay in MANUAL_BASE and press A to retry"
                                )
                                self.arm.go_ready(open_grip=True)
                                self._base_hold.stop()
                                self.phase = Phase.MANUAL_BASE
                        time.sleep(self.approach.cfg.tick_s)
                    elif self.phase == Phase.MANUAL_BASE:
                        # re-publish last base cmd lightly so stop doesn't decay oddly
                        time.sleep(0.05)
                    else:
                        time.sleep(0.02)
            except KeyboardInterrupt:
                print("\n[quit] KeyboardInterrupt")
            finally:
                self._keyboard = None
                self._base_hold.stop()
                self.client.send_cmd_vel(0.0, 0.0)
                self.approach.cancel()


def main():
    # Must match MARC_TEAM_ID on the manip-trainer side, which now forwards it as the
    # robot namespace; a mismatch surfaces as the wait_for_robot_topics timeout below.
    team_id = os.getenv("MARC_TEAM_ID", "u1").strip() or "u1"

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
