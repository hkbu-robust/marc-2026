"""MARC client that skips SESSION_ACK for the manip-trainer practice environment."""

from __future__ import annotations

import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from marc_sdk.client import MARCClient, _ParticipantNode


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
