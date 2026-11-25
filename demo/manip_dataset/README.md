# Stage 2 Manipulation Runtime Helpers (`demo/manip_dataset/`)

> **Scope note:** Stage 2 mobile manipulation is included for competition-submission completeness and is **outside the scope of the IEEE MetaCom 2026 Stage 1 paper**.

This package provides Stage 2 close-range approach, stationary-base holding, GraspGen-X grasp staging, class-specific pick profiles, and basket placement helpers used by `demo/participant_app.py` and `demo/handshake_manip_pick_client.py`.

## Modules

- `depth_approach.py` — Stop-turn-go RGB-D visual servoing toward a near-field YOLO detection
- `base_hold.py` — Active chassis pose holding during arm manipulation
- `grasp_profiles.py` — Upright and lying grasp profiles per object class
- `openvla_pick.py` — Deterministic pre-grasp staging and grasp verification guards
- `arm_control.py`, `pick_target.py`, `basket_place.py` — IK control, target snapshots, and collision-checked rear-basket placement
- `app.py`, `recorder.py` — Interactive manipulation trainer client utilities
