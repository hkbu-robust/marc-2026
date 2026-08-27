# GraspGen-X runtime snapshot (not in git)

Weights and gripper assets are downloaded at **Docker build**, not stored in git.
The loader is offline-only: it never clones Hugging Face and never downloads on
import. A missing package or unloadable snapshot falls back to analytic IK.

## Pins

| Artifact | Source | Revision |
|---|---|---|
| Package | `NVlabs/GraspGenX` | `b9429097728cb1c430dd78b92edf17ba318aad03` |
| Checkpoints | `adithyamurali/GraspGenXModel` | `7c834043c11a11417e31d6d5ea9355801e40a2c1` |
| Grippers | `adithyamurali/gripper_descriptions` (dataset) | `19a03c00d19aeaf052d0f6801f0041982d676e8a` |

Checkpoint license: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
Code license: Apache-2.0. Gripper descriptions: MIT.

HF layout is `release/gen` and `release/dis`. Fetch copies those to `gen/` and
`dis/` here so `load_model_cfg` can open them. Gripper checkout lands in `grippers/`.
The Python package is cloned at that pin and `pip install`ed from the checkout
(`demo/Dockerfile`) after upgrading setuptools past 61. Debian 59.6 ignores
PEP 621 and bakes an empty `UNKNOWN-0.0.0` wheel.

## Required files after fetch

- `gen/config.yaml` and a generator `.pth`
- `dis/config.yaml` and a discriminator `.pth`
- `grippers/` containing `x_grippers/` (possibly nested under `gripper_descriptions/assets/`)

Default gripper is `robotiq_2f_140` (Kinova Gen3 + Robotiq 2F-140).

## Env

| Variable | Meaning |
|---|---|
| `MARC_GRASPGENX_WEIGHTS` | Path to this directory (`gen/` + `dis/`) |
| `GRASPGENX_CHECKPOINT_DIR` | Same path; stops NVIDIA's import-time clone |
| `GRASPGENX_GRIPPER_CFG_DIR` | Path to the gripper snapshot |
| `MARC_GRASPGENX_GRIPPER` | Gripper name, default `robotiq_2f_140` |
| `MARC_PICK_POLICY` | `graspgenx` (default) / `ik` |

## Action convention

The pick seam crops the ego depth cloud inside the near-field bbox, asks
GraspGen-X for a 6-DoF grasp, uses the translation in **arm_base**
`(fwd, lat, z)` as the IK target, and maps the 4x4 XY closing axis to
``grasp_yaw`` for the pre/descend/grasp/lift waypoints. Basket place keeps
the default EE quaternion.
