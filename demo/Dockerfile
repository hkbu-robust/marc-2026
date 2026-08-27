# syntax=docker/dockerfile:1
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - Participant demo application (Docker submission template)
# ---------------------------------------------------------------------------
# The participant application must be developed/submitted as a Docker image (required).
# The standard entry point is `docker compose up`, and the team ID/token are
# entered in docker-compose.yml.
#
# Build context = the starter kit root (copies marc_sdk + demo together).
#   -> see compose build.context=.. , dockerfile=demo/Dockerfile.
# Build-time network bakes Qwen, GraspGen-X, and YOLO11x. Runtime is offline.
# ---------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 + ROS2 Humble + Python 3.10 (provides rclpy / *_msgs, default RMW=FastDDS)
FROM ros:humble-ros-base AS agent

ENV DEBIAN_FRONTEND=noninteractive

# Participant algorithm dependencies. numpy + pyyaml (stdlib/ROS2 for the rest), plus placo
# for the closed-loop arm IK pick (Pinocchio-based, pip-installable on Python 3.10).
# The network is used at build time only -- the judging runtime environment has no internet (no runtime dependency).
# Install h5py via pip (not apt python3-h5py): placo pulls a newer numpy that is
# ABI-incompatible with Debian's h5py ("numpy.dtype size changed").
# GraspGen-X is installed --no-deps so it cannot pin huggingface-hub/numpy against Qwen.
# BuildKit dropped `docker build --dns 8.8.8.8 --dns 1.1.1.1`; use host network instead.
# extras must not upgrade huggingface_hub to 1.x (transformers 4.57.1 requires <1.0).
# urdfpy is --no-deps so it cannot downgrade networkx under torch.
# packaging>=26.2 + no-build-isolation: GraspGen-X setuptools hits Ubuntu's packaging 21.3.
# setuptools>=64: Debian 59.6 ignores PEP 621 and bakes UNKNOWN-0.0.0 (~5 KB, no graspgenx).
RUN --network=host --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-yaml python3-pip git libgl1 libglib2.0-0 \
        ros-humble-tf2-ros ros-humble-tf2-ros-py \
    && pip3 install placo h5py \
    && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install "ultralytics-opencv-headless==8.4.118" \
    && pip3 install \
        "transformers==4.57.1" "accelerate==1.10.1" "safetensors==0.6.2" \
        "huggingface_hub>=0.30.0,<1.0" "qwen-vl-utils" \
    && rm -rf /var/lib/apt/lists/*

RUN --network=host --mount=type=cache,target=/root/.cache/pip \
    pip3 install \
        omegaconf hydra-core trimesh scipy yourdfpy tqdm addict timm tensordict einops \
        matplotlib scikit-learn webdataset PyOpenGL imageio pyrender \
        torch-geometric diffusers \
        "huggingface_hub>=0.30.0,<1.0" "safetensors==0.6.2" \
    && pip3 install --no-deps urdfpy \
    && pip3 install --upgrade "packaging>=26.2" "setuptools>=64,<78" wheel setuptools-scm \
    && python3 -c "import setuptools; v=tuple(map(int, setuptools.__version__.split('.')[:2])); assert v >= (64, 0), setuptools.__version__" \
    && git init /tmp/GraspGenX \
    && git -C /tmp/GraspGenX remote add origin https://github.com/NVlabs/GraspGenX.git \
    && git -C /tmp/GraspGenX fetch --depth 1 origin b9429097728cb1c430dd78b92edf17ba318aad03 \
    && git -C /tmp/GraspGenX checkout --detach FETCH_HEAD \
    && pip3 install --no-deps --no-build-isolation /tmp/GraspGenX \
    && python3 -c "from graspgenx.grasp_server import GraspGenXSampler; print('graspgenx import ok')" \
    && rm -rf /tmp/GraspGenX

WORKDIR /agent

# context = starter kit root -> copy marc_sdk, participant intelligence, demo adapter, models
COPY marc_sdk /agent/marc_sdk
COPY marc_agent /agent/marc_agent
COPY demo     /agent/demo
COPY models/runtime /agent/models/runtime

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ENV PYTHONPATH=/agent:/agent/demo
ENV MARC_QWEN_MODEL_DIR=/agent/models/runtime/stage1-sar-qwen/model
ENV MARC_GRASPGENX_WEIGHTS=/agent/models/runtime/graspgenx
ENV GRASPGENX_CHECKPOINT_DIR=/agent/models/runtime/graspgenx
ENV GRASPGENX_GRIPPER_CFG_DIR=/agent/models/runtime/graspgenx/grippers
ENV MARC_GRASPGENX_GRIPPER=robotiq_2f_140
ENV YOLO_CONFIG_DIR=/agent/demo/weights

# Optional HF_TOKEN (BuildKit secret) so Hub rate limits can use a logged-in account.
# Not required: organizers can rebuild anonymously. Never ENV the token into the image.
# HF_HUB_DISABLE_XET: Hub Xet can abort mid-shard with "error decoding response body".
RUN --network=host \
    --mount=type=cache,target=/root/.cache/huggingface \
    --mount=type=secret,id=HF_TOKEN,env=HF_TOKEN,required=false \
    HF_HUB_DISABLE_XET=1 python3 /agent/models/runtime/fetch_runtime_weights.py

# Offline after weights are in the image. Do not download at scoring runtime.
ENV YOLO_OFFLINE=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /agent/demo

# The ros:humble ENTRYPOINT (/ros_entrypoint.sh) sources /opt/ros/humble/setup.bash
# and then runs the CMD below. team_id/token are injected via the compose env (from_env).
CMD ["python3", "participant_app.py"]

# Probe-only extra: YOLO11x for auto-approach. The graded `agent` stage already
# has torch, ultralytics, and baked yolo11x.pt.
FROM agent AS manip-probe
ENV YOLO_CONFIG_DIR=/agent/demo/weights
RUN mkdir -p /agent/demo/weights
