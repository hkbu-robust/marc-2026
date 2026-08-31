#!/usr/bin/env python3
"""Checks for the dataset-gen on-landmark / above-mode trainer patch."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "simulation-platform" / "patches" / "patch_trainer_on_landmark.py"
SNAPSHOT = Path("/tmp/container_trainer.py")


def _load_patch():
    spec = importlib.util.spec_from_file_location(
        "patch_trainer_on_landmark", PATCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class OnLandmarkPatchTests(unittest.TestCase):
    def test_helper_and_positions_compile(self):
        patch = _load_patch()
        ast.parse(patch.ABOVE_HELPER)
        ast.parse(patch.BESIDE_HELPER)
        ast.parse(patch.SEAT_HELPER)
        ast.parse("def outer():\n    def _deploy_into():\n" + patch.NEW_POSITIONS)
        ast.parse("def outer():\n    def _settle():\n" + patch.NEW_SETTLE)
        ast.parse("def outer():\n    def _settle():\n" + patch.NEW_SNAP)
        ast.parse("def outer():\n    def _capture():\n" + patch.NEW_SAVE)
        ast.parse("def outer():\n    def _capture():\n" + patch.NEW_RENDER_STEP)
        self.assertIn("_marc_above_placed_n", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_aabb_stack", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_freeze", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_always_on_top", patch.ABOVE_HELPER)
        self.assertIn("CreateKinematicEnabledAttr", patch.ABOVE_HELPER)
        self.assertIn("RemoveAPI", patch.ABOVE_HELPER)
        self.assertIn("_marc_freeze_person", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_ray_roof", patch.ABOVE_HELPER)
        self.assertIn("skip insane upward", patch.ABOVE_HELPER)
        self.assertIn("_seat_at", patch.ABOVE_HELPER)
        self.assertIn("_top_candidates", patch.ABOVE_HELPER)
        self.assertIn("posed-joint roof contact", patch.ABOVE_HELPER)
        self.assertNotIn("max(support[2], top_z)", patch.ABOVE_HELPER)
        self.assertNotIn("_visible_stack", patch.NEW_SAVE)
        self.assertNotIn("_top_m", patch.NEW_SAVE)
        self.assertIn("GetPointsAttr", patch.ABOVE_HELPER)
        self.assertIn("if not placement.get(\"above\")", patch.NEW_PLACE_POSE)
        self.assertIn("not placement.get(\"recipe_id\")", patch.NEW_PLACE_POSE)
        self.assertIn("not placement.get(\"recipe_id\")", patch.NEW_REPOSITION_PL)
        self.assertIn("_marc_place_beside_people", patch.NEW_SNAP)
        self.assertIn("def _marc_place_beside_people", patch.BESIDE_HELPER)
        self.assertIn("_marc_beside_face_yaw", patch.BESIDE_HELPER)
        self.assertIn("_marc_yaw_keep_root", patch.BESIDE_HELPER)
        self.assertIn("_marc_recipe_seat_yaw", patch.SEAT_HELPER)
        self.assertIn("_marc_recipe_visual_center", patch.SEAT_HELPER)
        self.assertIn("_seat_ok", patch.NEW_SAVE)
        self.assertIn("_marc_yaw_keep_root", patch.SEAT_HELPER)
        self.assertIn("def _long_axis_yaw", patch.SEAT_HELPER)
        self.assertIn("_marc_above_trash_sit_yaw", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_yaw_no_double", patch.ABOVE_HELPER)
        self.assertIn("_marc_yaw_keep_root", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_sit_v80", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_trash_weight_lid", patch.ABOVE_HELPER)
        self.assertIn("_marc_trash_sit_rim_xy", patch.ABOVE_HELPER)
        self.assertIn("_marc_trash_butt_z", patch.ABOVE_HELPER)
        self.assertIn("_marc_vend_cam_lip", patch.ABOVE_HELPER)
        self.assertIn("_marc_vend_visual_roof", patch.ABOVE_HELPER)
        self.assertIn("_marc_sit_frame_head", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_TRASH_SIT_YAW", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_TRASH_RIM", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_TRASH_DROP", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_SIT_SINK_TRASH_WEIGHT", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_SIT_TORSO", patch.ABOVE_HELPER)
        self.assertIn("def visible_above_stack", patch.ABOVE_HELPER)
        self.assertIn("_marc_recipe_climb_visible", patch.NEW_SAVE)
        self.assertIn("visible_above_stack", patch.NEW_SAVE)
        self.assertIn('in ("on", "above")', patch.NEW_SAVE)
        self.assertNotIn(
            'if _tgt is None and _pose:',
            patch.NEW_SAVE)
        self.assertIn("def _marc_above_scene", patch.ABOVE_HELPER)
        self.assertIn("def _marc_recipe_scene", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_scene()", patch.NEW_SAVE)
        self.assertIn("_marc_recipe_scene()", patch.NEW_SAVE)
        self.assertIn("_marc_above_scene()", patch.NEW_SETTLE)
        self.assertIn("_marc_recipe_scene()", patch.NEW_SETTLE)
        self.assertIn("_marc_reseat_recipe", patch.NEW_SETTLE)
        self.assertIn("_marc_above_scene()", patch.NEW_POSITIONS)
        self.assertIn("_marc_recipe_scene()", patch.NEW_POSITIONS)
        self.assertIn("beside_landmark_type", patch.NEW_POSITIONS)
        self.assertIn("_near_types", patch.NEW_POSITIONS)
        self.assertIn("_marc_above_scene()", patch.NEW_CAPTURE_RETURN)
        self.assertIn("_marc_recipe_scene()", patch.NEW_CAPTURE_RETURN)
        self.assertIn("Recipe scene", patch.NEW_SAVE)
        self.assertIn("_marc_recipe_scene()", patch.NEW_RENDER_STEP)
        self.assertIn("_obj_pl", patch.NEW_PLACE_POSE)
        self.assertIn("TRAINER_PEOPLE_SIT_SINK", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_SIT_SINK_SHORT", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_sit_sink", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_sit_sink_70", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_sit_sink_short", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_drape", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_drape_lip", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_trash_raise", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_hydrant_crouch", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_trash_crouch", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_CROUCH_SINK_HYDRANT", patch.ABOVE_HELPER)
        self.assertIn("xy_from = \"centroid\"", patch.ABOVE_HELPER)
        self.assertIn("xy_from = \"butt\"", patch.ABOVE_HELPER)
        self.assertIn("(\"hydrant\", \"trash_can\")", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_BUTT_SINK", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_SIT_SINK_TRASH", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_trash_sit_rim", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_trash_visual_seat", patch.ABOVE_HELPER)
        self.assertIn("def _align_visual_on_trash_lid", patch.ABOVE_HELPER)
        self.assertIn("character=None", patch.ABOVE_HELPER)
        self.assertIn("float(top_z) - 0.12", patch.ABOVE_HELPER)
        self.assertNotIn("body_z >= float(top_z) - 0.20", patch.ABOVE_HELPER)
        self.assertIn("def _marc_place_seat_people", patch.SEAT_HELPER)
        self.assertIn("_marc_place_seat_people", patch.NEW_SETTLE)
        self.assertIn("_marc_place_seat_people", patch.NEW_SNAP)
        self.assertIn("TRAINER_SAR_ON_XY_MAX", patch.NEW_SAVE)
        self.assertIn("if _rel is True", patch.NEW_SAVE)
        self.assertIn('_rel = "on"', patch.NEW_SAVE)
        self.assertIn('_rel in ("on", "above")', patch.NEW_SAVE)
        self.assertNotIn(
            'str(rec.get("relation") or "") in ("on", "above")',
            patch.NEW_SAVE,
        )
        self.assertIn('"0.42") or "0.42"', patch.ABOVE_HELPER)
        self.assertIn("_drape_xy", patch.ABOVE_HELPER)
        self.assertIn("rim * half * dx / n", patch.ABOVE_HELPER)
        self.assertIn("hip = _mesh_hip_xy(", patch.ABOVE_HELPER)
        self.assertIn("plane_z) + _torso", patch.ABOVE_HELPER)
        _ensure_src = inspect.getsource(patch._ensure_above_helper)
        self.assertIn("_marc_trash_sit_rim_xy", _ensure_src)
        self.assertIn("_marc_trash_butt_z", _ensure_src)
        self.assertIn("_marc_vend_cam_lip", _ensure_src)
        self.assertIn("_marc_vend_visual_roof", _ensure_src)
        self.assertIn("_marc_sit_frame_head", _ensure_src)
        self.assertIn("def _vending_front_xy", patch.ABOVE_HELPER)
        self.assertIn("def _butt_from_joints", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_LIP", patch.ABOVE_HELPER)
        self.assertIn("def _yaw_person", patch.ABOVE_HELPER)
        self.assertIn("TransformDir", patch.ABOVE_HELPER)
        self.assertIn("q_delta", patch.ABOVE_HELPER)
        self.assertIn("def _yaw_legs_along", patch.ABOVE_HELPER)
        self.assertIn("def _vending_roof_xy", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_sit_yaw", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_abs_frame", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_leg_yaw", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_glass_drape", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_sit_fix", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_sit_nudge", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_sit_unraise", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_lip_fwd", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_lip_fwd2", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_lip_fwd3", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_lip_back", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof2", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof3", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof4", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof5", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof6", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female2", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female3", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female4", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female5", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female6", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female7", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_male", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_male2", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_male3", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_male4", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_male5", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_male6", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_male7", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight2", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight3", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight4", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight5", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight6", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_no_closeup", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_mid_field", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_roof_in_view", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_mesh_seat", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_mid_dist", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_on_roof", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof7", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female8", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_PROF_SINK", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_PROF_LIP", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_FEMALE_LIP", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_FEMALE_SINK", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_MALE_LIP", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_MALE_SINK", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_MALE_YAW", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_MALE_FWD", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_WEIGHT_LIP", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_WEIGHT_SINK", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_roof_visible", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_joint_z", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight_yaw", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female_fwd", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_stand", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof10", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_PROF_YAW", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_PROF_FWD", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_WEIGHT_FWD", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_sit_hip", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight7", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight8", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight9", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_headroom", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_visual_seat", patch.ABOVE_HELPER)
        self.assertIn("def _align_visual_on_roof", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_mesh_contact", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_crouch_authored", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_mesh_contact2", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_mesh_nudge", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_mesh_snap", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_lock_contact", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_joint_only", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof11", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_PROF_CAM_AWAY", patch.ABOVE_HELPER)
        self.assertIn('"0.08"', patch.ABOVE_HELPER)
        self.assertIn('"0.04"', patch.ABOVE_HELPER)
        self.assertIn("CreateVelocityAttr", patch.ABOVE_HELPER)
        self.assertNotIn("setStaticCollider", patch.ABOVE_HELPER)
        self.assertIn("_od_lock._marc_above_lock = {}", patch.NEW_SETTLE)
        self.assertIn("TRAINER_PEOPLE_VEND_MESH_CONTACT", patch.ABOVE_HELPER)
        self.assertIn("def _mesh_nudge_xy", patch.ABOVE_HELPER)
        self.assertIn("def _mesh_snap_z", patch.ABOVE_HELPER)
        self.assertIn(
            'if typ == "vending_machine" and "sitting" in pose:',
            patch.ABOVE_HELPER)
        self.assertNotIn(
            'or "standing" in pose or "crouching" in pose',
            patch.ABOVE_HELPER)
        self.assertNotIn("0.22 * span", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_WEIGHT_YAW", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_FEMALE_FWD", patch.ABOVE_HELPER)
        self.assertIn("person_walking", patch.ABOVE_HELPER)
        self.assertIn("def _mesh_hip_xy", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_FEMALE_YAW", patch.ABOVE_HELPER)
        self.assertIn('_yaw_key, "0") or "0"', patch.ABOVE_HELPER)
        self.assertNotIn('_yaw_key, "180"', patch.ABOVE_HELPER)
        self.assertIn(
            'elif typ == "vending_machine" and "reaching" in pose:',
            patch.ABOVE_HELPER)
        self.assertIn("def _mesh_seat_z", patch.ABOVE_HELPER)
        self.assertIn("half_front - margin", patch.ABOVE_HELPER)
        self.assertNotIn("half + 0.04", patch.ABOVE_HELPER)
        self.assertIn("def _vend_slot_xy", patch.ABOVE_HELPER)
        self.assertIn("def _vending_roof_clamp", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_slot", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_female9", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_prof12", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_SLOT", patch.ABOVE_HELPER)
        self.assertIn(
            'or "male_student" in _ch or "weight" in _ch',
            patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_weight9", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_MALE_YAW", patch.ABOVE_HELPER)
        self.assertIn('"0.0727"', patch.ABOVE_HELPER)
        self.assertIn('"-0.01"', patch.ABOVE_HELPER)
        self.assertIn('"0.16"', patch.ABOVE_HELPER)
        self.assertIn('"0.23"', patch.ABOVE_HELPER)
        self.assertIn('"0.07"', patch.ABOVE_HELPER)
        self.assertIn('"0.05"', patch.ABOVE_HELPER)
        self.assertNotIn("if d_want < 7.0 * 7.0", patch.ABOVE_HELPER)
        self.assertNotIn("if d_want < d_now", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_farther3", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_farther4", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_not_too_far", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_closer", patch.ABOVE_HELPER)
        self.assertNotIn("TRAINER_PEOPLE_VEND_DRAPE_BACK", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_headroom", patch.ABOVE_HELPER)
        self.assertIn("TRAINER_PEOPLE_VEND_FRAME_V", patch.ABOVE_HELPER)
        self.assertIn('"0.55"', patch.ABOVE_HELPER)
        self.assertNotIn('else "0.80")', patch.ABOVE_HELPER)
        self.assertNotIn('else "0.66")', patch.ABOVE_HELPER)
        self.assertIn("_marc_above_vend_visual_seat", patch.ABOVE_HELPER)
        self.assertNotIn(
            'v = 0.50 if typ == "vending_machine" else 0.55',
            patch.ABOVE_HELPER)
        self.assertIn("0.22 + random.random() * 0.16", patch.NEW_POSITIONS)
        self.assertIn("0.20 + random.random() * 0.12", patch.NEW_POSITIONS)
        self.assertIn("def _marc_nudge_above_nearer", patch.ABOVE_HELPER)
        self.assertIn("Never fake Z", patch.ABOVE_HELPER)
        self.assertIn("_marc_above_near_xy", patch.NEW_POSITIONS)
        self.assertIn("_n_placed > 0", patch.NEW_SAVE)
        self.assertIn("_stack_ok", patch.NEW_SAVE)
        self.assertNotIn("_n_placed > 0 and _n_stack > 0", patch.NEW_SAVE)
        self.assertIn("_marc_above_target_in_view", patch.NEW_SAVE)
        self.assertIn("not in CCTV detections", patch.NEW_SAVE)
        self.assertIn("top-clipped", patch.NEW_SAVE)
        self.assertIn("_headroom", patch.NEW_SAVE)
        self.assertIn("constructed stack invariant", patch.NEW_SAVE)
        self.assertNotIn("raise RuntimeError", patch.NEW_SAVE)
        self.assertIn("skip save", patch.NEW_SAVE)
        self.assertNotIn("_marc_nudge_above_nearer", patch.NEW_SAVE)
        self.assertIn("0.30 + random.random() * 0.40", patch.NEW_POSITIONS)
        self.assertIn("0.16 + random.random() * 0.18", patch.NEW_POSITIONS)
        self.assertIn("0.82 + random.random() * 0.10", patch.NEW_POSITIONS)
        self.assertIn("_extra_k", patch.NEW_POSITIONS)
        self.assertNotIn("for _ in range(10):", patch.NEW_POSITIONS)
        self.assertIn("return bool(gt.last_render_ok)", patch.NEW_CAPTURE_RETURN)
        self.assertIn("_marc_above_target_in_view", patch.NEW_CAPTURE_RETURN)
        self.assertNotIn("return _active_valid", patch.NEW_CAPTURE_RETURN)
        self.assertIn("above-mode: target=", patch.ABOVE_HELPER)
        self.assertIn("target={_tgt}", patch.NEW_GT_LOG)
        self.assertIn("target={_tgt}", patch.NEW_AUTO_FAILURE)
        self.assertIn("_marc_above_target", patch.ABOVE_HELPER)
        self.assertIn("vending_machine", patch.NEW_DEPLOY_LANDMARKS)
        self.assertIn("contact_clearance", patch.NEW_REPOSITION_PL)
        self.assertIn("_marc_gt_step", patch.NEW_RENDER_STEP)
        self.assertIn("_marc_reseat", patch.NEW_RENDER_STEP)
        self.assertIn("_marc_place_above_people", patch.NEW_RENDER_STEP)
        self.assertIn("simulation_context.pause()", patch.NEW_SETTLE)
        self.assertIn("_marc_place_above_people", patch.NEW_SETTLE)
        self.assertIn("for _ in range(_SETTLE_FRAMES):", patch.NEW_SETTLE)
        self.assertIn("simulation_context.pause()", patch.NEW_SNAP)
        self.assertIn("_marc_place_above_people", patch.NEW_SNAP)
        self.assertNotIn("_marc_run_above_place", patch.NEW_SNAP)

    def test_contact_ok_requires_on_top_and_touching(self):
        patch = _load_patch()
        lo, hi = (-0.4, -0.4, 16.0), (0.4, 0.4, 18.0)
        top_z = 18.0
        self.assertTrue(
            patch.above_contact_ok((0.0, 0.0, 18.02), lo, hi, top_z))
        self.assertFalse(
            patch.above_contact_ok((0.0, 0.0, 16.05), lo, hi, top_z),
            "standing beside at ground height must not count as on top")
        self.assertFalse(
            patch.above_contact_ok((0.0, 0.0, 18.30), lo, hi, top_z),
            "30cm hover above the top is not contact")
        self.assertFalse(
            patch.above_contact_ok((2.0, 0.0, 18.02), lo, hi, top_z),
            "off the footprint is not on the landmark")

    def test_mesh_top_is_the_roof_not_the_side_aabb(self):
        patch = _load_patch()
        points = [
            (0.0, 0.0, 18.0), (0.1, 0.0, 18.0), (0.0, 0.1, 18.0),
            (0.1, 0.1, 18.0),
            (0.4, 0.0, 16.2), (-0.4, 0.0, 16.2),
        ]
        top, roof_lo, roof_hi, patch_pts = patch.mesh_top_patch(points)
        self.assertAlmostEqual(top[2], 18.0)
        self.assertLess(roof_hi[0] - roof_lo[0], 0.25)
        body_lo, body_hi = (-0.4, -0.4, 16.0), (0.4, 0.4, 18.0)
        self.assertTrue(
            patch.above_roof_contact_ok(
                (0.05, 0.05, 18.02), roof_lo, roof_hi, body_lo, body_hi, 18.0))
        self.assertFalse(
            patch.above_roof_contact_ok(
                (0.35, 0.0, 18.02), roof_lo, roof_hi, body_lo, body_hi, 18.0),
            "standing against the cabinet side is not on the roof")

    def test_vending_glass_sign_locks_to_detailed_face(self):
        patch = _load_patch()
        plus = [(-1.0, 0.0, float(z)) for z in range(3)]
        plus += [(1.0, 0.0, float(z)) for z in range(12)]
        self.assertEqual(patch.vending_glass_sign(plus, 0), 1)
        minus = [(1.0, 0.0, float(z)) for z in range(3)]
        minus += [(-1.0, 0.0, float(z)) for z in range(12)]
        self.assertEqual(patch.vending_glass_sign(minus, 0), -1)
        self.assertEqual(patch.vending_glass_sign([], 0), 1)

    def test_visible_stack_rejects_crop_and_beside(self):
        patch = _load_patch()
        on_vending = {
            "class": "person_sitting",
            "bbox": [600, 200, 700, 420],
        }
        vending = {
            "class": "vending_machine",
            "bbox": [580, 280, 720, 500],
        }
        self.assertTrue(patch.visible_above_stack(on_vending, vending, 720))
        clipped = {
            "class": "vending_machine",
            "bbox": [580, 0, 720, 400],
        }
        clipped_person = {
            "class": "person_sitting",
            "bbox": [600, 0, 700, 180],
        }
        self.assertFalse(
            patch.visible_above_stack(clipped_person, clipped, 720),
            "top-clipped vending roof must not count as stacked")
        beside = {
            "class": "person_walking",
            "bbox": [590, 240, 710, 500],
        }
        self.assertFalse(patch.visible_above_stack(beside, vending, 720))
        hydrant_side = {
            "class": "person_reaching",
            "bbox": [120, 200, 220, 520],
        }
        hydrant = {
            "class": "hydrant",
            "bbox": [400, 360, 460, 520],
        }
        self.assertFalse(patch.visible_above_stack(hydrant_side, hydrant, 720))

    def test_colocates_climber_with_target_cell(self):
        patch = _load_patch()
        source = (
            patch.ABOVE_SCENE_FN
            + "def generate_cell_positions(area, n):\n"
            "    return [[float(i), 0.0, 0.0] for i in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = ["
            "{'name': 'vending_machine_0'}, {'name': 'bench_0'}]\n"
            "        objects = [\n"
            "            {'type': 'mug', 'count': 1},\n"
            "            {'type': 'person_reaching', 'count': 1,\n"
            "             'placement': {'above': True,\n"
            "                           'on_landmark_type': 'vending_machine'}},\n"
            "            {'type': 'person_walking', 'count': 1},\n"
            "        ]\n"
            "        n_lm = len(landmarks)\n"
            "        n_obj = sum(int(o.get('count', 1) or 1) for o in objects)\n"
            "        area = None\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos, obj_pos\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os}
        exec(compile(source, "<above-positions>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            lm_pos, obj_pos = namespace["outer"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(obj_pos[1], lm_pos[0])
        self.assertNotEqual(obj_pos[0], lm_pos[0])
        self.assertNotEqual(obj_pos[2], lm_pos[0])

    def test_headroom_puts_vending_mid_near_not_on_far_edge(self):
        patch = _load_patch()
        source = (
            patch.ABOVE_SCENE_FN
            + "def generate_cell_positions(area, n):\n"
            "    return [[100.0, 100.0, 16.0] for _ in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = [{'name': 'vending_machine_0'}]\n"
            "        objects = [\n"
            "            {'type': 'person_sitting', 'count': 1,\n"
            "             'placement': {'above': True,\n"
            "                           'on_landmark_type': 'vending_machine'}},\n"
            "        ]\n"
            "        n_lm = 1\n"
            "        n_obj = 1\n"
            "        area = {\n"
            "            'top_left': [0.0, 20.0, 16.0],\n"
            "            'top_right': [10.0, 20.0, 16.0],\n"
            "            'bottom_right': [10.0, 0.0, 16.0],\n"
            "            'bottom_left': [0.0, 0.0, 16.0],\n"
            "        }\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos, obj_pos\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os, "random": __import__("random")}
        exec(compile(source, "<above-headroom>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            namespace["random"].seed(0)
            lm_pos, obj_pos = namespace["outer"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(lm_pos[0], obj_pos[0])
        # Far edge is y=20, near is y=0. Near-mid so the roof stays
        # below the image top on these low CCTVs.
        self.assertGreater(lm_pos[0][1], 3.0)
        self.assertLess(lm_pos[0][1], 7.5)
        self.assertGreater(lm_pos[0][0], 2.0)
        self.assertLess(lm_pos[0][0], 8.0)

    def test_fov_pack_keeps_climb_stack_in_frame_and_scatters_rest(self):
        patch = _load_patch()
        source = (
            patch.ABOVE_SCENE_FN
            + "def generate_cell_positions(area, n):\n"
            "    return [[100.0, 100.0, 16.0] for _ in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = ["
            "{'name': 'vending_machine_0'}, {'name': 'bench_0'}]\n"
            "        objects = [\n"
            "            {'type': 'mug', 'count': 1},\n"
            "            {'type': 'person_sitting', 'count': 1,\n"
            "             'placement': {'above': True,\n"
            "                           'on_landmark_type': 'vending_machine'}},\n"
            "            {'type': 'person_walking', 'count': 1},\n"
            "        ]\n"
            "        n_lm = 2\n"
            "        n_obj = 3\n"
            "        area = {\n"
            "            'top_left': [0.0, 20.0, 16.0],\n"
            "            'top_right': [10.0, 20.0, 16.0],\n"
            "            'bottom_right': [10.0, 0.0, 16.0],\n"
            "            'bottom_left': [0.0, 0.0, 16.0],\n"
            "        }\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos, obj_pos\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os, "random": __import__("random")}
        exec(compile(source, "<above-fov-pack>", "exec"), namespace)
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            namespace["random"].seed(1)
            lm_pos, obj_pos = namespace["outer"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(lm_pos[0], obj_pos[1])
        climb = lm_pos[0]
        self.assertGreater(climb[0], 2.0)
        self.assertLess(climb[0], 8.0)
        self.assertGreater(climb[1], 3.0)
        self.assertLess(climb[1], 7.5)
        for pos in list(lm_pos) + list(obj_pos):
            self.assertNotEqual(pos[:2], [100.0, 100.0])
        bench = lm_pos[1]
        self.assertGreater(
            (bench[0] - climb[0]) ** 2 + (bench[1] - climb[1]) ** 2,
            2.0,
        )
        extra_xs = [obj_pos[0][0], obj_pos[2][0]]
        self.assertLess(min(extra_xs), 4.0)
        self.assertGreater(max(extra_xs), 6.0)

    def test_climb_slot_jitters_left_and_right(self):
        patch = _load_patch()
        source = (
            patch.ABOVE_SCENE_FN
            + "def generate_cell_positions(area, n):\n"
            "    return [[5.0, 5.0, 16.0] for _ in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = [{'name': 'vending_machine_0'}]\n"
            "        objects = [\n"
            "            {'type': 'person_sitting', 'count': 1,\n"
            "             'placement': {'above': True,\n"
            "                           'on_landmark_type': 'vending_machine'}},\n"
            "        ]\n"
            "        n_lm = 1\n"
            "        n_obj = 1\n"
            "        area = {\n"
            "            'top_left': [0.0, 20.0, 16.0],\n"
            "            'top_right': [10.0, 20.0, 16.0],\n"
            "            'bottom_right': [10.0, 0.0, 16.0],\n"
            "            'bottom_left': [0.0, 0.0, 16.0],\n"
            "        }\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos[0][0]\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os, "random": __import__("random")}
        exec(compile(source, "<above-jitter>", "exec"), namespace)
        xs = []
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            for seed in range(24):
                namespace["random"].seed(seed)
                xs.append(namespace["outer"]())
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertGreater(max(xs) - min(xs), 2.0)
        self.assertLess(min(xs), 4.5)
        self.assertGreater(max(xs), 5.5)
        for x in xs:
            self.assertGreater(x, 2.0)
            self.assertLess(x, 8.0)

    def test_replaces_stale_above_helper(self):
        patch = _load_patch()
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as fh:
            fh.write(
                "def other():\n    return 1\n\n"
                "def _marc_place_above_people(mission_id, objects, configurations=None):\n"
                "    globals()['_marc_above_placed_n'] = 0\n"
                "    return None\n"
            )
            path = fh.name
        try:
            self.assertTrue(patch._ensure_above_helper(path))
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("_marc_above_aabb_stack", text)
            self.assertIn("_marc_above_freeze", text)
            self.assertIn("_marc_above_always_on_top", text)
            self.assertIn("_marc_above_sit_sink", text)
            self.assertIn("_marc_above_sit_sink_70", text)
            self.assertIn("_marc_above_sit_sink_short", text)
            self.assertIn("_marc_above_vend_drape", text)
            self.assertIn("_marc_above_vend_drape_lip", text)
            self.assertIn("_marc_above_trash_raise", text)
            self.assertIn("_marc_above_hydrant_crouch", text)
            self.assertIn("_marc_above_trash_crouch", text)
            self.assertIn("_marc_above_vend_farther", text)
            self.assertIn("_marc_above_vend_farther2", text)
            self.assertIn("_marc_above_vend_farther3", text)
            self.assertIn("_marc_above_vend_farther4", text)
            self.assertIn("_marc_above_vend_not_too_far", text)
            self.assertIn("_marc_above_vend_closer", text)
            self.assertIn("_marc_above_vend_sit_yaw", text)
            self.assertIn("_marc_above_vend_abs_frame", text)
            self.assertIn("_marc_above_vend_leg_yaw", text)
            self.assertIn("_marc_above_vend_glass_drape", text)
            self.assertIn("_marc_above_vend_sit_fix", text)
            self.assertIn("_marc_above_vend_sit_nudge", text)
            self.assertIn("_marc_above_vend_sit_unraise", text)
            self.assertIn("_marc_above_vend_lip_fwd", text)
            self.assertIn("_marc_above_vend_lip_fwd2", text)
            self.assertIn("_marc_above_vend_lip_fwd3", text)
            self.assertIn("_marc_above_vend_lip_back", text)
            self.assertIn("_marc_above_vend_prof", text)
            self.assertIn("_marc_above_vend_prof2", text)
            self.assertIn("_marc_above_vend_prof3", text)
            self.assertIn("_marc_above_vend_prof4", text)
            self.assertIn("_marc_above_vend_prof5", text)
            self.assertIn("_marc_above_vend_prof6", text)
            self.assertIn("_marc_above_vend_female", text)
            self.assertIn("_marc_above_vend_female2", text)
            self.assertIn("_marc_above_vend_female3", text)
            self.assertIn("_marc_above_vend_female4", text)
            self.assertIn("_marc_above_vend_female5", text)
            self.assertIn("_marc_above_vend_female6", text)
            self.assertIn("_marc_above_vend_female7", text)
            self.assertIn("_marc_above_vend_male", text)
            self.assertIn("_marc_above_vend_male2", text)
            self.assertIn("_marc_above_vend_male3", text)
            self.assertIn("_marc_above_vend_male4", text)
            self.assertIn("_marc_above_vend_male5", text)
            self.assertIn("_marc_above_vend_male6", text)
            self.assertIn("_marc_above_vend_male7", text)
            self.assertIn("_marc_above_vend_weight", text)
            self.assertIn("_marc_above_vend_weight2", text)
            self.assertIn("_marc_above_vend_weight3", text)
            self.assertIn("_marc_above_vend_weight4", text)
            self.assertIn("_marc_above_vend_weight5", text)
            self.assertIn("_marc_above_vend_weight6", text)
            self.assertIn("_marc_above_no_closeup", text)
            self.assertIn("_marc_above_mid_field", text)
            self.assertIn("_marc_above_roof_in_view", text)
            self.assertIn("_marc_above_vend_roof_visible", text)
            self.assertIn("_marc_above_vend_joint_z", text)
            self.assertIn("_marc_above_vend_prof8", text)
            self.assertIn("_marc_above_vend_mesh_xy", text)
            self.assertIn("_marc_above_vend_female_yaw", text)
            self.assertIn("_marc_above_vend_weight_yaw", text)
            self.assertIn("_marc_above_vend_female_fwd", text)
            self.assertIn("_marc_above_vend_stand", text)
            self.assertIn("_marc_above_vend_prof10", text)
            self.assertIn("TRAINER_PEOPLE_VEND_PROF_YAW", text)
            self.assertIn("TRAINER_PEOPLE_VEND_PROF_FWD", text)
            self.assertIn("TRAINER_PEOPLE_VEND_WEIGHT_FWD", text)
            self.assertIn("_marc_above_vend_sit_hip", text)
            self.assertIn("_marc_above_vend_weight7", text)
            self.assertIn("_marc_above_vend_weight8", text)
            self.assertIn("_marc_above_vend_weight9", text)
            self.assertIn("_marc_above_headroom", text)
            self.assertIn("TRAINER_PEOPLE_VEND_FRAME_V", text)
            self.assertIn("_marc_above_vend_visual_seat", text)
            self.assertIn("_marc_above_trash_sit_yaw", text)
            self.assertIn("_marc_above_vend_yaw_no_double", text)
            self.assertIn("_marc_yaw_keep_root", text)
            self.assertIn("_marc_above_vend_sit_v80", text)
            self.assertIn("_marc_above_trash_weight_lid", text)
            self.assertIn("_marc_trash_sit_rim_xy", text)
            self.assertIn("_marc_trash_butt_z", text)
            self.assertIn("_marc_vend_cam_lip", text)
            self.assertIn("_marc_vend_visual_roof", text)
            self.assertIn("_marc_sit_frame_head", text)
            self.assertIn("TRAINER_PEOPLE_TRASH_SIT_YAW", text)
            self.assertIn("TRAINER_PEOPLE_TRASH_RIM", text)
            self.assertIn("TRAINER_PEOPLE_TRASH_DROP", text)
            self.assertIn("TRAINER_PEOPLE_SIT_SINK_TRASH_WEIGHT", text)
            self.assertIn("TRAINER_PEOPLE_VEND_SIT_TORSO", text)
            self.assertIn("def visible_above_stack", text)
            self.assertIn('"0.55"', text)
            self.assertIn("q_delta", text)
            self.assertIn("TransformDir", text)
            self.assertIn("float(top_z) - 0.12", text)
            self.assertNotIn("body_z >= float(top_z) - 0.20", text)
            self.assertNotIn(
                "seated = True\n        if not seated:",
                text)
            self.assertIn("def _align_visual_on_roof", text)
            self.assertIn("_marc_above_vend_mesh_contact", text)
            self.assertIn("_marc_above_crouch_authored", text)
            self.assertIn("_marc_above_vend_mesh_contact2", text)
            self.assertIn("_marc_above_vend_mesh_nudge", text)
            self.assertIn("_marc_above_vend_mesh_snap", text)
            self.assertIn("_marc_above_lock_contact", text)
            self.assertIn("_marc_above_joint_only", text)
            self.assertIn("_marc_above_vend_prof11", text)
            self.assertIn("_marc_above_vend_slot", text)
            self.assertIn("_marc_above_vend_female9", text)
            self.assertIn("_marc_above_vend_prof12", text)
            self.assertIn("TRAINER_PEOPLE_VEND_SLOT", text)
            self.assertIn("TRAINER_PEOPLE_VEND_PROF_CAM_AWAY", text)
            self.assertIn("TRAINER_PEOPLE_VEND_MESH_CONTACT", text)
            self.assertIn("TRAINER_PEOPLE_VEND_PROF_SINK", text)
            self.assertIn("TRAINER_PEOPLE_VEND_PROF_LIP", text)
            self.assertIn("TRAINER_PEOPLE_VEND_FEMALE_LIP", text)
            self.assertIn("TRAINER_PEOPLE_VEND_FEMALE_SINK", text)
            self.assertIn("TRAINER_PEOPLE_VEND_MALE_LIP", text)
            self.assertIn("TRAINER_PEOPLE_VEND_MALE_SINK", text)
            self.assertIn("TRAINER_PEOPLE_VEND_MALE_YAW", text)
            self.assertIn("TRAINER_PEOPLE_VEND_MALE_FWD", text)
            self.assertIn("TRAINER_PEOPLE_VEND_WEIGHT_LIP", text)
            self.assertIn("TRAINER_PEOPLE_VEND_WEIGHT_SINK", text)
            self.assertIn("TRAINER_PEOPLE_VEND_LIP", text)
            self.assertIn("TRAINER_PEOPLE_VEND_BUTT_SINK", text)
            self.assertIn("def _vending_front_xy", text)
            self.assertIn("def _butt_from_joints", text)
            self.assertIn("xy_from = \"butt\"", text)
            self.assertNotIn("TRAINER_PEOPLE_VEND_DRAPE_BACK", text)
            self.assertIn("def _yaw_person", text)
            self.assertIn("def _yaw_legs_along", text)
            self.assertIn("def _vending_roof_xy", text)
            self.assertIn("TRAINER_PEOPLE_CROUCH_SINK_HYDRANT", text)
            self.assertIn("TRAINER_PEOPLE_SIT_SINK", text)
            self.assertIn("TRAINER_PEOPLE_SIT_SINK_SHORT", text)
            self.assertIn("TRAINER_PEOPLE_SIT_SINK_TRASH", text)
            self.assertIn("_drape_xy", text)
            self.assertIn("rim * half * dx / n", text)
            self.assertIn("hip = _mesh_hip_xy(", text)
            self.assertIn("plane_z) + _torso", text)
            self.assertIn("CreateKinematicEnabledAttr", text)
            self.assertIn("RemoveAPI", text)
            self.assertIn("GetPointsAttr", text)
            self.assertIn("def _marc_nudge_above_nearer", text)
            self.assertIn("raycast_closest", text)
            self.assertEqual(text.count("def _marc_place_above_people"), 1)
            self.assertFalse(patch._ensure_above_helper(path))
        finally:
            os.unlink(path)

    def test_replaces_stale_beside_helper_without_dropping_seat(self):
        patch = _load_patch()
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as fh:
            fh.write(
                "def other():\n    return 1\n\n"
                "def _marc_place_beside_people(mission_id, objects, configurations=None):\n"
                "    return None\n\n"
                "def _marc_place_seat_people(mission_id, objects, configurations=None):\n"
                "    # _marc_recipe_seat_yaw\n"
                "    return None\n"
            )
            path = fh.name
        try:
            self.assertTrue(patch._ensure_beside_helper(path))
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("_marc_beside_face_yaw", text)
            self.assertIn("_marc_yaw_keep_root", text)
            self.assertIn("q_delta", text)
            self.assertEqual(text.count("def _marc_place_beside_people"), 1)
            self.assertEqual(text.count("def _marc_place_seat_people"), 1)
            self.assertIn("_marc_recipe_seat_yaw", text)
            self.assertFalse(patch._ensure_beside_helper(path))
        finally:
            os.unlink(path)

    def test_replaces_stale_seat_helper(self):
        patch = _load_patch()
        with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as fh:
            fh.write(
                "def other():\n    return 1\n\n"
                "def _marc_place_seat_people(mission_id, objects, configurations=None):\n"
                "    return None\n"
            )
            path = fh.name
        try:
            self.assertTrue(patch._ensure_seat_helper(path))
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("_marc_recipe_seat_yaw", text)
            self.assertIn("_marc_yaw_keep_root", text)
            self.assertIn("def _long_axis_yaw", text)
            self.assertIn("q_delta", text)
            self.assertEqual(text.count("def _marc_place_seat_people"), 1)
            self.assertFalse(patch._ensure_seat_helper(path))
        finally:
            os.unlink(path)

    def test_snapshot_anchors_still_match(self):
        if not SNAPSHOT.is_file():
            self.skipTest("stock trainer snapshot not at %s" % SNAPSHOT)
        patch = _load_patch()
        src = SNAPSHOT.read_text(encoding="utf-8")
        for name in (
                "OLD_POSITIONS",
                "OLD_DEPLOY_LANDMARKS",
                "OLD_SAVE",
                "OLD_GT_LOG",
                "OLD_SNAP",
                "OLD_SETTLE",
                "OLD_RENDER_STEP",
                "OLD_CAPTURE_RETURN",
                "OLD_INITIAL_WARN",
                "OLD_AUTO_REMAINING",
                "OLD_AUTO_ATTEMPT",
                "OLD_AUTO_FAILURE",
        ):
            self.assertIn(getattr(patch, name), src, name)
        patched = src.replace(patch.OLD_POSITIONS, patch.NEW_POSITIONS, 1)
        patched = patched.replace(patch.OLD_DEPLOY_LANDMARKS,
                                  patch.NEW_DEPLOY_LANDMARKS, 1)
        patched = patched.replace(patch.OLD_SETTLE, patch.NEW_SETTLE, 1)
        patched = patched.replace(patch.OLD_RENDER_STEP, patch.NEW_RENDER_STEP, 1)
        patched = patched.replace(patch.OLD_SAVE, patch.NEW_SAVE, 1)
        patched = patched.replace(patch.OLD_GT_LOG, patch.NEW_GT_LOG, 1)
        ast.parse(patched)

    def test_vending_sit_slots_stay_on_roof(self):
        patch = _load_patch()
        start = patch.ABOVE_HELPER.find("    def _vend_slot_xy(")
        self.assertGreater(start, 0)
        end = patch.ABOVE_HELPER.find("\n    def ", start + 1)
        ns = {}
        exec(compile(textwrap.dedent(patch.ABOVE_HELPER[start:end]),
                     "<vend-slot>", "exec"), ns)
        ox, oy, fx, fy = 0.0, 0.0, 1.0, 0.0
        hf, hs, margin = 0.35, 0.45, 0.12
        center = ns["_vend_slot_xy"](ox, oy, fx, fy, hf, hs, "center", margin)
        left = ns["_vend_slot_xy"](ox, oy, fx, fy, hf, hs, "left", margin)
        right = ns["_vend_slot_xy"](ox, oy, fx, fy, hf, hs, "right", margin)
        depth = hf - margin
        side = hs - margin
        self.assertAlmostEqual(center[0], depth)
        self.assertAlmostEqual(center[1], 0.0)
        self.assertAlmostEqual(left[0], depth)
        self.assertAlmostEqual(left[1], side)
        self.assertAlmostEqual(right[0], depth)
        self.assertAlmostEqual(right[1], -side)
        for pt in (center, left, right):
            self.assertLessEqual(abs(pt[0]), depth + 1e-9)
            self.assertLessEqual(abs(pt[1]), side + 1e-9)
        self.assertNotAlmostEqual(left[1], right[1])
        self.assertNotAlmostEqual(left[1], center[1])

    def test_recipe_above_rewrites_env_climb_gates(self):
        patch = _load_patch()
        env_on = (
            '        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0"))'
            '.lower() not in (\n'
            '                "0", "false", "no", "off", ""):\n'
            '            skip = True\n'
        )
        env_off = (
            '    if not render_ok and str(\n'
            '            os.environ.get("TRAINER_PEOPLE_ABOVE", "0")\n'
            '    ).lower() in ("0", "false", "no", "off", ""):\n'
            '        warn()\n'
        )
        self.assertIn("_marc_above_scene()", patch._rewrite_above_gates(env_on))
        self.assertIn("not _marc_above_scene()", patch._rewrite_above_gates(env_off))
        ns = {}
        exec(compile(patch.ABOVE_SCENE_FN, "<above-scene>", "exec"), ns)
        import json
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            with open("/tmp/marc_sar_recipe.json", "w", encoding="utf-8") as f:
                json.dump({"above": True}, f)
            self.assertTrue(ns["_marc_above_scene"]())
            with open("/tmp/marc_sar_recipe.json", "w", encoding="utf-8") as f:
                json.dump({"above": False}, f)
            self.assertFalse(ns["_marc_above_scene"]())
            os.environ["TRAINER_PEOPLE_ABOVE"] = "1"
            self.assertTrue(ns["_marc_above_scene"]())
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_recipe_scene_requires_env_and_recipe_id(self):
        patch = _load_patch()
        ns = {}
        exec(compile(patch.ABOVE_SCENE_FN, "<recipe-scene>", "exec"), ns)
        import json
        old = dict(os.environ)
        dump = "/tmp/marc_sar_recipe.json"
        try:
            os.environ.pop("TRAINER_SAR_RECIPES", None)
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            with open(dump, "w", encoding="utf-8") as f:
                json.dump({"recipe_id": "sar03_bench_lie", "above": False}, f)
            self.assertFalse(ns["_marc_recipe_scene"]())
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            self.assertTrue(ns["_marc_recipe_scene"]())
            self.assertFalse(ns["_marc_above_scene"]())
            with open(dump, "w", encoding="utf-8") as f:
                json.dump({"above": True}, f)
            self.assertFalse(ns["_marc_recipe_scene"]())
            self.assertTrue(ns["_marc_above_scene"]())
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_colocates_recipe_beside_with_anchor(self):
        patch = _load_patch()
        source = (
            patch.ABOVE_SCENE_FN
            + "def generate_cell_positions(area, n):\n"
            "    return [[float(i), 0.0, 0.0] for i in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = ["
            "{'name': 'taxi_0'}, {'name': 'bench_0'}]\n"
            "        objects = [\n"
            "            {'type': 'mug', 'count': 1},\n"
            "            {'type': 'person_lying_down', 'count': 1,\n"
            "             'placement': {'recipe_id': 'sar06_parking_collapsed',\n"
            "                           'beside_landmark_type': 'taxi'}},\n"
            "            {'type': 'person_walking', 'count': 1},\n"
            "        ]\n"
            "        n_lm = len(landmarks)\n"
            "        n_obj = sum(int(o.get('count', 1) or 1) for o in objects)\n"
            "        area = None\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos, obj_pos\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os}
        exec(compile(source, "<recipe-positions>", "exec"), namespace)
        import json
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            with open("/tmp/marc_sar_recipe.json", "w", encoding="utf-8") as f:
                json.dump({
                    "recipe_id": "sar06_parking_collapsed",
                    "above": False,
                    "surface_anchor": "taxi",
                }, f)
            lm_pos, obj_pos = namespace["outer"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(obj_pos[1], lm_pos[0])
        self.assertNotEqual(obj_pos[0], lm_pos[0])

    def test_recipe_pack_keeps_taxi_near_bench(self):
        patch = _load_patch()
        source = (
            patch.ABOVE_SCENE_FN
            + "def generate_cell_positions(area, n):\n"
            "    return [[100.0, 100.0, 16.0] for _ in range(n)]\n"
            "def outer():\n"
            "    def _deploy_into():\n"
            "        landmarks = ["
            "{'name': 'bench_0'}, {'name': 'taxi_0'}]\n"
            "        objects = [\n"
            "            {'type': 'person_lying_down', 'count': 1,\n"
            "             'placement': {'recipe_id': 'sar03_bench_lie',\n"
            "                           'on_landmark_type': 'bench'}},\n"
            "        ]\n"
            "        n_lm = 2\n"
            "        n_obj = 1\n"
            "        area = {\n"
            "            'top_left': [0.0, 20.0, 16.0],\n"
            "            'top_right': [10.0, 20.0, 16.0],\n"
            "            'bottom_right': [10.0, 0.0, 16.0],\n"
            "            'bottom_left': [0.0, 0.0, 16.0],\n"
            "        }\n"
            + patch.NEW_POSITIONS
            + "        return lm_pos, obj_pos\n"
            "    return _deploy_into()\n"
        )
        namespace = {"os": os, "random": __import__("random")}
        exec(compile(source, "<recipe-near-pack>", "exec"), namespace)
        import json
        old = dict(os.environ)
        try:
            os.environ["TRAINER_PEOPLE_ABOVE"] = "0"
            os.environ["TRAINER_SAR_RECIPES"] = "1"
            with open("/tmp/marc_sar_recipe.json", "w", encoding="utf-8") as f:
                json.dump({
                    "recipe_id": "sar03_bench_lie",
                    "above": False,
                    "extra_landmarks": ["taxi"],
                }, f)
            namespace["random"].seed(2)
            lm_pos, obj_pos = namespace["outer"]()
        finally:
            os.environ.clear()
            os.environ.update(old)
        bench, taxi = lm_pos[0], lm_pos[1]
        self.assertEqual(bench, obj_pos[0])
        dist2 = (taxi[0] - bench[0]) ** 2 + (taxi[1] - bench[1]) ** 2
        self.assertLess(dist2, 16.0)
        self.assertGreater(dist2, 0.04)


if __name__ == "__main__":
    unittest.main()
