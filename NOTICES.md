# Third-Party Notices & Component Licenses

Original source code authored by **Team roBUst (Hong Kong Baptist University)** in this repository is licensed under the **MIT License** (see [LICENSE](LICENSE)).

Third-party components, organizer starter-kit files, model checkpoints, and paper materials retain their own licenses and copyright notices as listed below.

---

## Component Provenance & License Table

| Component / Path | Origin / Author | License / Terms |
|---|---|---|
| `marc_agent/`, `scripts/`, `training/`, `tests/`, `docs/` | Team roBUst (HKBU) | MIT ([LICENSE](LICENSE)) |
| `marc_sdk/` | MARC 2026 Organizers (IoTCOSS) | MIT |
| `simulation-platform/` Dockerfiles, `Makefile`, `docker-compose*.yml`, `marc.sh`, `.env.example`, `problems.yaml.example`, and organizer patches (`patches/boot.py`, `patches/patch_lidar_frameid.py`, `patches/patch_object_rotation.py`, `patches/patch_object_xformops.py`) | MARC 2026 Starter Kit — Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS) | Starter-kit Dockerfile/script notice; NVIDIA `IsaacSim-dockerfiles` base is Apache-2.0. No NVIDIA binaries or USD assets are redistributed. |
| `simulation-platform/patches/` team extensions (`ego_dataset_gen.py`, `ego_dataset_plan.py`, `patch_gt_world_xyz.py`, `patch_runtime_auto_start_delay.py`, `patch_trainer_auto_scenes.py`, `patch_trainer_instances.py`, `patch_trainer_on_landmark.py`, `patch_trainer_people.py`, `run_qwen_dataset_gen.py`, `sar_recipes.yaml`) | Team roBUst (HKBU) | MIT |
| Stage 1 & Stage 2 YOLO26x checkpoints (`models/runtime/stage1-cctv-detector/`, `models/runtime/stage2-nearfield-detector/`) | Fine-tuned by Team roBUst using Ultralytics YOLO26x | **GNU AGPL-3.0** (see bundled `LICENSE` in each detector directory). Running this MIT source with Ultralytics and the bundled YOLO weights forms a combined AGPL-3.0 work. |
| Fine-tuned Qwen3-VL-8B selector (`roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned`, `models/runtime/stage1-sar-qwen/`) | Fine-tuned by Team roBUst from `Qwen/Qwen3-VL-8B-Instruct` | Apache-2.0 |
| `NVlabs/GraspGenX` & `adithyamurali/GraspGenXModel` (`models/runtime/graspgenx/`) | NVIDIA / GraspGenX authors | Code: Apache-2.0; Checkpoints: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/); Gripper descriptions: MIT |
| ROS 2 Humble (`rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`) | Open Robotics | Apache-2.0 |
| NVIDIA Isaac Sim (`nvcr.io/nvidia/isaac-sim:5.1.0`) | NVIDIA (pulled and built locally by user; not redistributed) | NVIDIA Isaac Sim Additional Software and Materials License |
| `demo/urdf/gen3_6dof_vision_2f140.urdf` | Derived from Kinova Gen3 + Robotiq 2F-140 ROS descriptions (MARC Starter Kit) | BSD-3-Clause (upstream Kinova/Robotiq descriptions) |
| Manuscript PDF (`paper/MARC2026_Grounding_VLMs.pdf`) | Authors / IEEE MetaCom 2026 (Author Accepted Manuscript) | © 2026 IEEE (see IEEE author-posting notice in [README.md](README.md)) |

---

## NVIDIA Isaac Sim Distribution Notice

The simulation platform containers under `simulation-platform/` are distributed as **Dockerfiles only** (no prebuilt NVIDIA binaries or USD scenes are stored in this repository). Users pull `nvcr.io/nvidia/isaac-sim:5.1.0` directly from NGC under NVIDIA's EULA and build locally. The public practice playground content image (`ghcr.io/marc-challenge/marc-platform-content:2026.R03`) is provided by the MARC 2026 organizers.

## Simulator 3D Asset Attributions (from MARC 2026 Starter Kit)

The runtime 3D assets are delivered inside the organizers' platform container image (not stored in this repository):

- **People characters:** CC0 1.0 (Public Domain).
- **Vehicles** (`landmark/car/{police_car, normal_car_1, normal_car_2, sports_car, sports_car_2, suv, taxi}`): LowPoly Models by **@Quaternius** under **CC0 1.0 Universal** (`https://www.patreon.com/quaternius`).
- **Objects (CC BY — Creative Commons Attribution, Sketchfab / Fab):**
  - `closedlongumbrella`: <https://sketchfab.com/3d-models/umbrella-not-open-b5835433a6414388a9846568cc110da>
  - `pencilcase` (modified): <https://sketchfab.com/3d-models/pencil-case-pencil-sharpener-eraser-and-ruler-62d50825c5de4d1ba574356fbe9b2985>
  - `sunblock` (modified): <https://sketchfab.com/3d-models/kolagra-sunblock-tube-d3b9b069af3a4d1ba574356fbe9b2985>
