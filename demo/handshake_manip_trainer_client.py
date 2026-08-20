#!/usr/bin/env python3
"""Manip-trainer teleop dataset collector entry point.

Connects to the manip-trainer ROS 2 instance (handshake bypassed), then runs the
keyboard teleop collector: manual base pose -> user-triggered YOLO auto-approach
-> human arm EE teleop -> sensor-only success -> HDF5 episodes.

Usage:
    export ROS_DOMAIN_ID=0
    export MARC_TEAM_ID=u1
    python3 demo/handshake_manip_trainer_client.py

Docker:
    MARC_TEAM_ID=u1 docker compose up --build agent-manip-probe
"""

from manip_dataset.app import main

if __name__ == "__main__":
    main()
