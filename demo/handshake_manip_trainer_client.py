#!/usr/bin/env python3
"""Use MARCClient-style methods against a manip-trainer ROS 2 instance.

The stock MARCClient.connect() waits for SESSION_ACK from the competition platform.
The manipulation trainer profile does not implement that handshake, so this script
uses the same SDK client class but overrides the registration step to start the
node and publishers/subscribers directly.

After DDS discovery sees the trainer robot topics, it runs the closed-loop
pick-and-place sequence from arm_pick.py (placo FK/IK via arm_kin.py).

Usage:
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=0
    export MARC_TEAM_ID=u1
    python3 demo/handshake_manip_trainer_client.py
"""

import logging
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from marc_sdk.client import MARCClient, _ParticipantNode

from arm_pick import load_params, run_pick_sequence

log = logging.getLogger("manip_trainer_client")


class ManipTrainerClient(MARCClient):
    """MARCClient subclass that skips the platform SESSION_ACK handshake."""

    def connect(self, timeout: float = 1.0, register_period: float = 0.2) -> bool:
        if not rclpy.ok():
            rclpy.init()

        self._registered = False
        self._register_event.clear()
        self._node = _ParticipantNode(self, self._node_name)
        self._setup_publishers()
        self._setup_subscribers()

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self._registered = True
        self._register_event.set()
        self._log("[MANIP-TRAINER] using direct ROS 2 connection (handshake bypassed)")
        return True


def wait_for_robot_topics(client: MARCClient, team_id: str, timeout_sec: float = 60.0) -> bool:
    """Poll the ROS graph until trainer joint_states appear under this team namespace."""
    joint_topic = f"/marc/{team_id}/robot/arm/joint_states"
    started = time.monotonic()
    last_count = -1

    while time.monotonic() - started < timeout_sec:
        names_and_types = client._node.get_topic_names_and_types()
        all_topics = sorted(name for name, _ in names_and_types)
        if len(all_topics) != last_count:
            print(f"[probe] topics ({len(all_topics)}): {all_topics}")
            last_count = len(all_topics)

        if joint_topic in all_topics:
            print(f"[probe] found {joint_topic}")
            return True

        remaining = timeout_sec - (time.monotonic() - started)
        print(f"[probe] waiting for {joint_topic} ({remaining:.0f}s left)")
        time.sleep(2.0)

    print(f"[probe] timed out waiting for {joint_topic}", file=sys.stderr)
    return False


def _grasp_xy_from_params():
    """Default grasp xy in arm_base frame from pnp_params / arm_pick defaults."""
    p = load_params()
    return (float(p["grasp_fwd"]), float(p["grasp_lat"]))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    team_id = os.getenv("MARC_TEAM_ID", "u1").strip() or "u1"
    client = ManipTrainerClient(team_id=team_id, token="dummy-token")

    print(f"[probe] team_id={team_id} namespace=/marc/{team_id}")
    print(f"[probe] expected robot topics under /marc/{team_id}/robot/*")
    print("[probe] note: this manip-trainer profile is a practice environment, not the full MARC platform runtime")
    print("[probe] it does not implement the participant registration handshake or SESSION_ACK")
    if team_id != "u1":
        print(f"[probe] WARN: manip-trainer publishes under /marc/u1/...; "
              f"MARC_TEAM_ID={team_id} will miss joint_states / sensors",
              file=sys.stderr)

    if not client.connect():
        print("failed to connect", file=sys.stderr)
        sys.exit(1)

    print(f"Connected as {team_id}; waiting for robot topics...")
    if not wait_for_robot_topics(client, team_id):
        client.shutdown()
        sys.exit(1)

    # Hold chassis still before arm motion (residual cmd_vel can push the base).
    for _ in range(5):
        client.send_cmd_vel(0.0, 0.0)
        time.sleep(0.1)

    grasp_xy = _grasp_xy_from_params()
    # Optional override: MARC_GRASP_XY=fwd,lat
    raw = os.getenv("MARC_GRASP_XY", "").strip()
    if raw:
        parts = [float(x) for x in raw.split(",")]
        if len(parts) >= 2:
            grasp_xy = (parts[0], parts[1])

    print(f"[pick] starting arm pick-and-place (grasp_xy arm_base={grasp_xy})")
    try:
        held = run_pick_sequence(client, log, grasp_xy)
        print(f"[pick] sequence finished (holding={held})")
    except Exception as e:
        log.exception("pick sequence failed: %s", e)
        client.stop()
        client.shutdown()
        sys.exit(1)

    client.stop()
    client.shutdown()
    print("[pick] done")


if __name__ == "__main__":
    main()
