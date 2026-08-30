#!/usr/bin/env python3
"""Manip-trainer GraspGen-X pick tester entry point.

Connects to the manip-trainer ROS 2 instance (handshake bypassed). Press G to
localize the current object and run GraspGen-X pick + basket drop. No HDF5.

Usage:
    export ROS_DOMAIN_ID=0
    export MARC_TEAM_ID=u1
    python3 demo/handshake_manip_pick_client.py

Docker:
    cd demo
    docker compose --profile manip-pick up --build agent-manip-pick
"""

from manip_dataset.pick_app import main

if __name__ == "__main__":
    main()
