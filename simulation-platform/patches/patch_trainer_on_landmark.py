#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - source patch: put some people on landmark tops in dataset-gen
# ---------------------------------------------------------------------------
# Stock trainer.py calls deploy_mission_objects(..., place_on_landmark=False),
# so every person stands on a ground cell. Competition SAR commands include
# "climbed on top of the drink machine" / "sitting on the trash can"; those
# layouts never appear unless we turn placement back on.
#
# object_deployer.py reads placement.on_landmark_type from the pose YAML, not
# from the object spec. This patch:
#   1. sets place_on_landmark=True in trainer.py
#   2. lets an object spec's placement overlay the pose YAML
#   3. re-seats those people after landmark ground-snap (trash cans tip)
#   4. if placement.above is set, raycasts the landmark's real top surface
#      and drops the posed skeleton onto that patch (in contact AND on top,
#      not AABB-hovering beside it). Save only when that contact verifies
#      AND the camera sees the person stacked on the unclipped landmark top.
#
# boot.py runs this before dataset-gen. Idempotent via MARKER. Pair with
# patch_trainer_people.py, which actually stamps placement onto people specs.
import os
import re
import sys

TRAINER = "/metacom2026/simulation_app/marc2026/trainer.py"
DEPLOYER = "/metacom2026/simulation_app/marc2026/world/object_deployer.py"
MARKER = "_marc_obj_placement"

OLD_PLACE_POSE = '''    placement = (pose or {}).get("placement") or {}
    # 트레이너 데이터셋 모드는 사람을 지면(cell)에 세운다 → 랜드마크 위 배치 비활성.
    on_landmark_type = placement.get("on_landmark_type") if place_on_landmark else None
    z_offset_from_top = float(placement.get("z_offset_from_top", 0.0) or 0.0)
'''

NEW_PLACE_POSE = '''    placement = dict((pose or {}).get("placement") or {})
    _obj_pl = obj_group.get("placement") or {}
    if _obj_pl:
        placement.update(_obj_pl)
    # [marc-patch] _marc_obj_placement: object-spec overlay beats pose YAML.
    on_landmark_type = placement.get("on_landmark_type") if place_on_landmark else None
    z_offset_from_top = float(placement.get("z_offset_from_top", 0.0) or 0.0)
'''

OLD_REPOSITION_PL = '''        placement = (pose or {}).get("placement") or {}
        on_lm = placement.get("on_landmark_type")
        if not on_lm:
            continue
        z_off = float(placement.get("z_offset_from_top", 0.0) or 0.0)
'''

NEW_REPOSITION_PL = '''        placement = dict((pose or {}).get("placement") or {})
        _obj_pl = obj_group.get("placement") or {}
        if _obj_pl:
            placement.update(_obj_pl)
        on_lm = placement.get("on_landmark_type")
        if not on_lm or placement.get("above"):
            continue
        z_off = float(placement.get("z_offset_from_top", 0.0) or 0.0)
'''

OLD_SNAP = "        _snap_landmarks_to_ground([mission])\n"

NEW_SNAP_V2 = '''        _snap_landmarks_to_ground([mission])
        try:
            from world.object_deployer import reposition_on_landmark_people
            reposition_on_landmark_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"reposition on-landmark people failed (non-fatal): {_e}")
        try:
            from world.object_deployer import _marc_place_above_people
            _marc_place_above_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"place-above people failed (non-fatal): {_e}")
'''

NEW_SNAP_V3 = '''        _snap_landmarks_to_ground([mission])
        try:
            from world.object_deployer import reposition_on_landmark_people
            reposition_on_landmark_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"reposition on-landmark people failed (non-fatal): {_e}")
        try:
            from world.object_deployer import _marc_place_above_people
            _marc_place_above_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"place-above people failed (non-fatal): {_e}")
        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            try:
                _warm = max(1, int(os.environ.get(
                    "TRAINER_RENDER_WARMUP_FRAMES", "12") or "12"))
            except (TypeError, ValueError):
                _warm = 12
            for _ in range(_warm):
                simulation_context.step(render=True)
'''

NEW_SNAP = '''        _snap_landmarks_to_ground([mission])
        try:
            from world.object_deployer import reposition_on_landmark_people
            reposition_on_landmark_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"reposition on-landmark people failed (non-fatal): {_e}")
        def _marc_run_above_place():
            from world.object_deployer import _marc_place_above_people
            _marc_place_above_people(
                mission["id"], mission.get("objects") or [], configurations)
        try:
            _marc_run_above_place()
        except Exception as _e:
            log.warning(f"place-above people failed (non-fatal): {_e}")
        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            try:
                _warm = max(1, int(os.environ.get(
                    "TRAINER_RENDER_WARMUP_FRAMES", "12") or "12"))
            except (TypeError, ValueError):
                _warm = 12
            for _ in range(_warm):
                simulation_context.step(render=True)
            try:
                _marc_run_above_place()
            except Exception as _e:
                log.warning(f"place-above people failed (non-fatal): {_e}")
'''

NEW_SNAP_V1 = '''        _snap_landmarks_to_ground([mission])
        try:
            from world.object_deployer import reposition_on_landmark_people
            reposition_on_landmark_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"reposition on-landmark people failed (non-fatal): {_e}")
'''

OLD_SAVE = '''        if save_offline_enabled:
            # 선택된 카메라만 저장 (배치 콘텐츠가 있는 active 미션).
            active_payload = gt_by_mission.get(active_mission_id)
            if active_payload is not None:
                try:
                    gt.save_offline(scene_id, {active_mission_id: active_payload})
                except Exception as e:
                    log.warning(f"offline GT save failed: {e}")
'''

NEW_SAVE_V1 = '''        active_payload = gt_by_mission.get(active_mission_id)
        _active_valid = True
        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            _dets = list((active_payload or {}).get("detections") or [])
            _want = {
                x.strip() for x in
                (os.environ.get("TRAINER_PEOPLE_ABOVE_TYPES", "") or "").split(",")
                if x.strip()
            }
            _people = [
                d for d in _dets
                if str(d.get("class") or "").startswith("person_")
            ]
            _landmarks = [
                d for d in _dets
                if (
                    str(d.get("class") or "") in _want
                    if _want else d.get("kind") == "landmark"
                )
            ]
            try:
                from world import object_deployer as _od_above
                _n_placed = int(getattr(_od_above, "_marc_above_placed_n", 0) or 0)
            except Exception:
                _n_placed = 0
            # Posed-skeleton / raycast contact verified (on top AND touching).
            _active_valid = _n_placed > 0 and bool(_people) and bool(_landmarks)
            if not _active_valid:
                log.warning(
                    f"Above scene {scene_id} skipped: placed={_n_placed} "
                    f"people={len(_people)} targets={len(_landmarks)}")
        if save_offline_enabled and _active_valid:
            # 선택된 카메라만 저장 (배치 콘텐츠가 있는 active 미션).
            if active_payload is not None:
                try:
                    gt.save_offline(scene_id, {active_mission_id: active_payload})
                except Exception as e:
                    log.warning(f"offline GT save failed: {e}")
'''

NEW_SAVE = '''        active_payload = gt_by_mission.get(active_mission_id)
        _active_valid = True
        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            _dets = list((active_payload or {}).get("detections") or [])
            _img = (active_payload or {}).get("image") or {}
            try:
                _ih = float(_img.get("height") or 0.0)
            except (TypeError, ValueError):
                _ih = 0.0
            _top_m = max(8.0, 0.03 * _ih) if _ih else 8.0
            try:
                from world import object_deployer as _od_above
                _n_placed = int(getattr(_od_above, "_marc_above_placed_n", 0) or 0)
                _target = str(getattr(_od_above, "_marc_above_target", "") or "")
            except Exception:
                _n_placed = 0
                _target = ""
            _want = {
                x.strip() for x in
                (os.environ.get("TRAINER_PEOPLE_ABOVE_TYPES", "") or "").split(",")
                if x.strip()
            }
            if _target:
                _want = {_target}
            def _bbox4(_det):
                _b = _det.get("bbox") or []
                if len(_b) != 4:
                    return None
                _x0, _y0, _x1, _y1 = (
                    float(_b[0]), float(_b[1]), float(_b[2]), float(_b[3]))
                if _x1 < _x0:
                    _x0, _x1 = _x1, _x0
                if _y1 < _y0:
                    _y0, _y1 = _y1, _y0
                if _x1 - _x0 < 1.0 or _y1 - _y0 < 1.0:
                    return None
                return _x0, _y0, _x1, _y1
            def _visible_stack(_person, _lm):
                _pb, _lb = _bbox4(_person), _bbox4(_lm)
                if _pb is None or _lb is None:
                    return False
                _ox0, _oy0, _ox1, _oy1 = _pb
                _lx0, _ly0, _lx1, _ly1 = _lb
                # Roof / climber cut off at the top of the CCTV frame.
                if _oy0 <= _top_m or _ly0 <= _top_m:
                    return False
                _ovx = min(_ox1, _lx1) - max(_ox0, _lx0)
                _minw = min(_ox1 - _ox0, _lx1 - _lx0)
                if _minw <= 1.0 or _ovx < 0.4 * _minw:
                    return False
                if 0.5 * (_oy0 + _oy1) >= 0.5 * (_ly0 + _ly1):
                    return False
                _lmh = _ly1 - _ly0
                if _lmh < 8.0:
                    return False
                if (_ly1 - _oy1) < 0.22 * _lmh:
                    return False
                if _oy0 > _ly0:
                    return False
                return True
            _people = [
                d for d in _dets
                if str(d.get("class") or "").startswith("person_")
            ]
            _landmarks = [
                d for d in _dets
                if (
                    str(d.get("class") or "") in _want
                    if _want else d.get("kind") == "landmark"
                )
            ]
            _n_stack = sum(
                1 for _p in _people for _lm in _landmarks
                if _visible_stack(_p, _lm)
            )
            _active_valid = _n_placed > 0 and _n_stack > 0
            log.info(
                f"Above scene {scene_id}: target={_target or '-'} "
                f"{'saved' if _active_valid else 'skipped'} "
                f"placed={_n_placed} stacked={_n_stack} "
                f"people={len(_people)} landmarks={len(_landmarks)}")
        if save_offline_enabled and _active_valid:
            # 선택된 카메라만 저장 (배치 콘텐츠가 있는 active 미션).
            if active_payload is not None:
                try:
                    gt.save_offline(scene_id, {active_mission_id: active_payload})
                except Exception as e:
                    log.warning(f"offline GT save failed: {e}")
'''

OLD_CAPTURE_RETURN = "        return gt.last_render_ok\n"
NEW_CAPTURE_RETURN = '''        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            return _active_valid
        return gt.last_render_ok
'''

OLD_GT_LOG = '''        log.info(f"GT scene {scene_id}: {total} detections / {len(gt_by_mission)} cameras "
                 f"(saved: {active_mission_id})")
'''
NEW_GT_LOG = '''        _tgt = "-"
        try:
            from world import object_deployer as _od_gt
            _tgt = str(getattr(_od_gt, "_marc_above_target", "") or "") or "-"
        except Exception:
            _tgt = "-"
        log.info(f"GT scene {scene_id}: {total} detections / {len(gt_by_mission)} cameras "
                 f"(saved: {active_mission_id}; target={_tgt})")
'''

OLD_INITIAL_WARN = '''    if not render_ok and gt is not None:
        gt.warn_black_frame(trainer_state.scene_id)
'''

NEW_INITIAL_WARN = '''    if not render_ok and gt is not None and str(
            os.environ.get("TRAINER_PEOPLE_ABOVE", "0")
    ).lower() in ("0", "false", "no", "off", ""):
        gt.warn_black_frame(trainer_state.scene_id)
'''

OLD_AUTO_REMAINING = '''    auto_remaining = max(0, auto_scenes - 1)   # scene 0 은 위에서 이미 생성
    if auto_scenes:
'''

NEW_AUTO_REMAINING = '''    # Count only captures that were actually saved (person on a landmark).
    auto_remaining = max(0, auto_scenes - (1 if render_ok else 0))
    _auto_failed_scenes = 0
    _auto_failure_limit = max(20, auto_scenes * 8)
    if auto_scenes:
'''

OLD_AUTO_ATTEMPT = '''            auto_remaining -= 1
            active = _pick_random_active(exclude_id=active["id"])
'''

NEW_AUTO_ATTEMPT = '''            active = _pick_random_active(exclude_id=active["id"])
'''

OLD_AUTO_FAILURE = '''                if not ok and gt is not None:
                    gt.warn_black_frame(trainer_state.scene_id)
            except Exception as e:
'''

NEW_AUTO_FAILURE = '''                if ok:
                    auto_remaining -= 1
                    _auto_failed_scenes = 0
                else:
                    _auto_failed_scenes += 1
                    if gt is not None and str(
                            os.environ.get("TRAINER_PEOPLE_ABOVE", "0")
                    ).lower() in ("0", "false", "no", "off", ""):
                        gt.warn_black_frame(trainer_state.scene_id)
                    _tgt = "-"
                    try:
                        from world import object_deployer as _od_fail
                        _tgt = str(getattr(
                            _od_fail, "_marc_above_target", "") or "") or "-"
                    except Exception:
                        _tgt = "-"
                    log.warning(
                        f"Scene {trainer_state.scene_id} not saved "
                        f"(target={_tgt}, "
                        f"{auto_remaining} on-landmark scenes remain)")
                    if _auto_failed_scenes >= _auto_failure_limit:
                        log.error(
                            f"Stopping after {_auto_failed_scenes} consecutive "
                            "scenes without a visible person-on-landmark pair")
                        break
            except Exception as e:
'''

OLD_POSITIONS = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
'''

NEW_POSITIONS_V1 = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            import re as _re_pos
            _pl = ""
            _climber_i = None
            _obj_i = 0
            for _spec in objects or []:
                _n = max(1, int(_spec.get("count", 1) or 1))
                _opl = (_spec.get("placement") or {})
                if _climber_i is None and _opl.get("above"):
                    _climber_i = _obj_i
                    _pl = str(_opl.get("on_landmark_type") or "")
                _obj_i += _n
            _lm_i = None
            if _pl:
                for _i, _spec in enumerate(landmarks):
                    _t = _re_pos.sub(r"_[0-9]+$", "", str(
                        (_spec or {}).get("name") or ""))
                    if _t == _pl:
                        _lm_i = _i
                        break
            if _lm_i is None:
                for _i, _spec in enumerate(landmarks):
                    _t = _re_pos.sub(r"_[0-9]+$", "", str(
                        (_spec or {}).get("name") or ""))
                    if _t not in ("bench", "picnic_table"):
                        _lm_i = _i
                        break
            if (_lm_i is not None and _climber_i is not None
                    and _lm_i < len(lm_pos) and _climber_i < len(obj_pos)):
                obj_pos[_climber_i] = list(lm_pos[_lm_i])
'''

NEW_POSITIONS = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            import re as _re_pos
            _pl = ""
            _climber_i = None
            _obj_i = 0
            for _spec in objects or []:
                _n = max(1, int(_spec.get("count", 1) or 1))
                _opl = (_spec.get("placement") or {})
                if _climber_i is None and _opl.get("above"):
                    _climber_i = _obj_i
                    _pl = str(_opl.get("on_landmark_type") or "")
                _obj_i += _n
            _lm_i = None
            if _pl:
                for _i, _spec in enumerate(landmarks):
                    _t = _re_pos.sub(r"_[0-9]+$", "", str(
                        (_spec or {}).get("name") or ""))
                    if _t == _pl:
                        _lm_i = _i
                        break
            if _lm_i is None:
                for _i, _spec in enumerate(landmarks):
                    _t = _re_pos.sub(r"_[0-9]+$", "", str(
                        (_spec or {}).get("name") or ""))
                    if _t not in ("bench", "picnic_table"):
                        _lm_i = _i
                        break
            if (_lm_i is not None and _climber_i is not None
                    and _lm_i < len(lm_pos) and _climber_i < len(obj_pos)):
                _spot = None
                if isinstance(area, dict):
                    try:
                        _tl = list(area["top_left"])
                        _tr = list(area["top_right"])
                        _br = list(area["bottom_right"])
                        _bl = list(area["bottom_left"])
                        _u = 0.42 + random.random() * 0.16
                        _v = (0.28 + random.random() * 0.14
                              if _pl in ("vending_machine", "postbox")
                              else 0.38 + random.random() * 0.16)
                        _bot = [(1.0 - _u) * _bl[_k] + _u * _br[_k]
                                for _k in range(3)]
                        _top = [(1.0 - _u) * _tl[_k] + _u * _tr[_k]
                                for _k in range(3)]
                        _spot = [(1.0 - _v) * _bot[_k] + _v * _top[_k]
                                 for _k in range(3)]
                    except Exception:
                        _spot = None
                if _spot is not None:
                    lm_pos[_lm_i] = _spot
                obj_pos[_climber_i] = list(lm_pos[_lm_i])
'''

OLD_DEPLOY_LANDMARKS = '''        deploy_landmarks(mission_id, landmarks, area, configurations,
                         randomize_rotation=True,
                         positions=lm_pos, free_fall_types=_TIP_OVER_LANDMARKS)
'''

NEW_DEPLOY_LANDMARKS = '''        _free_fall = _TIP_OVER_LANDMARKS
        if str(os.environ.get("TRAINER_PEOPLE_ABOVE", "0")).lower() not in (
                "0", "false", "no", "off", ""):
            # Climb targets stay upright so bounds-contact still meets the top.
            _keep_upright = {
                x.strip() for x in
                (os.environ.get("TRAINER_PEOPLE_ABOVE_TYPES", "") or "").split(",")
                if x.strip()
            } or {
                "trash_can", "bicycle", "kick_scooter",
                "vending_machine", "hydrant", "postbox",
            }
            _free_fall = _TIP_OVER_LANDMARKS - _keep_upright
        deploy_landmarks(mission_id, landmarks, area, configurations,
                         randomize_rotation=True,
                         positions=lm_pos, free_fall_types=_free_fall)
'''

def visible_above_stack(person, lm, image_height=0.0, top_margin=8.0):
    """True when a person is stacked on a landmark in-frame, roof not clipped."""
    def _bbox(det):
        b = (det or {}).get("bbox") or []
        if len(b) != 4:
            return None
        x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 - x0 < 1.0 or y1 - y0 < 1.0:
            return None
        return x0, y0, x1, y1

    pb, lb = _bbox(person), _bbox(lm)
    if pb is None or lb is None:
        return False
    ox0, oy0, ox1, oy1 = pb
    lx0, ly0, lx1, ly1 = lb
    try:
        ih = float(image_height or 0.0)
    except (TypeError, ValueError):
        ih = 0.0
    margin = max(float(top_margin), 0.03 * ih) if ih else float(top_margin)
    if oy0 <= margin or ly0 <= margin:
        return False
    overlap_x = min(ox1, lx1) - max(ox0, lx0)
    min_w = min(ox1 - ox0, lx1 - lx0)
    if min_w <= 1.0 or overlap_x < 0.4 * min_w:
        return False
    if 0.5 * (oy0 + oy1) >= 0.5 * (ly0 + ly1):
        return False
    lm_h = ly1 - ly0
    if lm_h < 8.0:
        return False
    if (ly1 - oy1) < 0.22 * lm_h:
        return False
    if oy0 > ly0:
        return False
    return True


def above_contact_ok(foot_xyz, lm_lo, lm_hi, top_z, max_gap=0.06):
    """True when support is on the landmark top and touching it, not beside/hovering."""
    fx, fy, fz = (float(foot_xyz[0]), float(foot_xyz[1]), float(foot_xyz[2]))
    x0, y0, z0 = (float(lm_lo[0]), float(lm_lo[1]), float(lm_lo[2]))
    x1, y1, z1 = (float(lm_hi[0]), float(lm_hi[1]), float(lm_hi[2]))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    pad_x = max(0.02, 0.05 * (x1 - x0))
    pad_y = max(0.02, 0.05 * (y1 - y0))
    if not (x0 - pad_x <= fx <= x1 + pad_x and y0 - pad_y <= fy <= y1 + pad_y):
        return False
    height = z1 - z0
    if height < 0.12:
        return False
    if (fz - z0) < 0.50 * height:
        return False
    gap = fz - float(top_z)
    return -0.02 <= gap <= float(max_gap)


ABOVE_HELPER = r'''

def _marc_place_above_people(mission_id, objects, configurations=None):
    """[marc-patch] _marc_above_surface_contact: posed joints on raycast top."""
    jobs = []
    for spec in objects or []:
        pl = spec.get("placement") or {}
        if not pl.get("above"):
            continue
        try:
            clearance = float(pl.get("contact_clearance", 0.03) or 0.03)
        except (TypeError, ValueError):
            clearance = 0.03
        jobs.append((
            spec, str(pl.get("on_landmark_type") or ""), clearance))
    globals()["_marc_above_placed_n"] = 0
    globals()["_marc_above_target"] = jobs[0][1] if jobs else ""
    if not jobs:
        return
    try:
        import carb
        import omni.usd
        from pxr import Usd, UsdGeom, UsdSkel, Gf
    except Exception:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    try:
        from omni.physx import get_physx_scene_query_interface
        sq = get_physx_scene_query_interface()
    except Exception:
        sq = None
    try:
        from world.people_poser import drop_to_ground as _drop_to_support
    except Exception:
        _drop_to_support = None

    def _world_xyz(prim):
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return None
        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )

    def _world_range(prim):
        try:
            bounds.Clear()
            r = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            if r.IsEmpty():
                return None
            lo, hi = r.GetMin(), r.GetMax()
            return (
                [float(lo[0]), float(lo[1]), float(lo[2])],
                [float(hi[0]), float(hi[1]), float(hi[2])],
            )
        except Exception:
            return None

    def _set_xyz(prim, xyz):
        path = str(prim.GetPath())
        setter = globals().get("_set_prim_translate")
        if callable(setter):
            try:
                setter(path, xyz)
                return
            except Exception:
                pass
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        op = ops.get("xformOp:translate")
        if op is None:
            op = xf.AddTranslateOp()
        op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def _posed_joints(prim):
        root = prim
        skel_root = None
        skeleton = None
        for child in Usd.PrimRange(root):
            if child.IsA(UsdSkel.Root) and skel_root is None:
                skel_root = child
            if child.IsA(UsdSkel.Skeleton) and skeleton is None:
                skeleton = child
        if skel_root is None or skeleton is None:
            return []
        try:
            cache = UsdSkel.Cache()
            cache.Populate(UsdSkel.Root(skel_root), Usd.PrimDefaultPredicate)
            query = cache.GetSkelQuery(UsdSkel.Skeleton(skeleton))
            xfs = query.ComputeJointWorldTransforms(UsdGeom.XformCache())
            pts = []
            for m in xfs:
                t = m.ExtractTranslation()
                pts.append([float(t[0]), float(t[1]), float(t[2])])
            return pts
        except Exception:
            return []

    def _support_from_joints(pts):
        if not pts:
            return None
        min_z = min(p[2] for p in pts)
        lowest = [p for p in pts if p[2] <= min_z + 0.08]
        if not lowest:
            lowest = pts
        cx = sum(p[0] for p in lowest) / float(len(lowest))
        cy = sum(p[1] for p in lowest) / float(len(lowest))
        return [cx, cy, min_z]

    def _ray_landmark_z(lm_path, x, y, start_z):
        if sq is None:
            return None
        origin_z = float(start_z)
        for _ in range(12):
            hit = sq.raycast_closest(
                carb.Float3(float(x), float(y), origin_z),
                carb.Float3(0.0, 0.0, -1.0),
                8.0,
            )
            if not hit or not hit.get("hit"):
                return None
            path = str(hit.get("rigidBody") or hit.get("collision") or "")
            hz = float(hit["position"][2])
            if path.startswith(lm_path) or (lm_path in path):
                return hz
            origin_z = hz - 0.01
        return None

    def _top_patch(lm):
        rng = _world_range(lm)
        if rng is None:
            return None
        lo, hi = rng
        path = str(lm.GetPath())
        xs = [0.2, 0.35, 0.5, 0.65, 0.8]
        ys = [0.2, 0.35, 0.5, 0.65, 0.8]
        hits = []
        start_z = hi[2] + 0.8
        for u in xs:
            for v in ys:
                x = lo[0] + u * (hi[0] - lo[0])
                y = lo[1] + v * (hi[1] - lo[1])
                hz = _ray_landmark_z(path, x, y, start_z)
                if hz is not None:
                    hits.append((x, y, hz))
        if hits:
            top_z = max(h[2] for h in hits)
            patch = [h for h in hits if h[2] >= top_z - 0.04]
            cx = sum(h[0] for h in patch) / float(len(patch))
            cy = sum(h[1] for h in patch) / float(len(patch))
            return [cx, cy, top_z], lo, hi
        return [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), hi[2]], lo, hi

    def _contact_ok(foot, lo, hi, top_z):
        fx, fy, fz = foot
        pad_x = max(0.02, 0.05 * (hi[0] - lo[0]))
        pad_y = max(0.02, 0.05 * (hi[1] - lo[1]))
        if not (lo[0] - pad_x <= fx <= hi[0] + pad_x
                and lo[1] - pad_y <= fy <= hi[1] + pad_y):
            return False
        height = hi[2] - lo[2]
        if height < 0.12:
            return False
        if (fz - lo[2]) < 0.50 * height:
            return False
        gap = fz - top_z
        return -0.02 <= gap <= 0.06

    def _landmark_prims(typ):
        group = stage.GetPrimAtPath("/world/landmarks/%s" % mission_id)
        hits = []
        if group and group.IsValid():
            for child in group.GetChildren():
                name = str(child.GetName() or "")
                blob = _norm(name) + _norm(str(child.GetPath()))
                if (not typ) or name.startswith(typ) or _norm(typ) in blob:
                    hits.append(child)
        if hits:
            return [p for p in hits if str(p.GetPath()) not in used_lms]
        root = stage.GetPrimAtPath("/world/landmarks")
        if not root or not root.IsValid():
            return []
        fuzzy = []
        token = _norm(typ)
        for prim in Usd.PrimRange(root):
            if prim == root or not prim.IsValid() or not prim.IsActive():
                continue
            blob = _norm(prim.GetName()) + _norm(str(prim.GetPath()))
            if (not token) or token in blob:
                fuzzy.append(prim)
        fuzzy.sort(key=lambda p: (len(str(p.GetPath())), str(p.GetPath())))
        roots, seen = [], []
        for prim in fuzzy:
            path = str(prim.GetPath())
            if path in used_lms:
                continue
            if any(path.startswith(parent + "/") for parent in seen):
                continue
            roots.append(prim)
            seen.append(path)
        return roots

    def _person_prims(spec):
        pose = str(spec.get("type") or "")
        ch = str(spec.get("character") or "")
        found = []
        try:
            n = max(1, int(spec.get("count", 1) or 1))
        except (TypeError, ValueError):
            n = 1
        for seq in range(n):
            pp = "/world/people/%s_%s_%s_%s" % (mission_id, pose, ch, seq)
            prim = stage.GetPrimAtPath(pp)
            if prim and prim.IsValid() and str(prim.GetPath()) not in used_people:
                found.append(prim)
        if found:
            return found
        root = stage.GetPrimAtPath("/world/people")
        if not root or not root.IsValid():
            return []
        tokens = tuple(_norm(t) for t in (pose, ch) if str(t).strip())
        hits = []
        for prim in Usd.PrimRange(root):
            if prim == root or not prim.IsValid() or not prim.IsActive():
                continue
            blob = _norm(prim.GetName()) + _norm(str(prim.GetPath()))
            if (not tokens) or all(token in blob for token in tokens):
                hits.append(prim)
        hits.sort(key=lambda p: (len(str(p.GetPath())), str(p.GetPath())))
        roots, seen = [], []
        for prim in hits:
            path = str(prim.GetPath())
            if path in used_people:
                continue
            if any(path.startswith(parent + "/") for parent in seen):
                continue
            roots.append(prim)
            seen.append(path)
        return roots

    used_people, used_lms = set(), set()
    n = 0
    for spec, typ, clearance in jobs:
        lms = _landmark_prims(typ)
        people = _person_prims(spec)
        if not lms or not people:
            continue
        lm, person = lms[0], people[0]
        top = _top_patch(lm)
        if top is None:
            continue
        (cx, cy, top_z), lo, hi = top
        person_xyz = _world_xyz(person)
        if person_xyz is None:
            continue
        joints = _posed_joints(person)
        support = _support_from_joints(joints)
        if support is None:
            rng = _world_range(person)
            if rng is None:
                continue
            p_lo, p_hi = rng
            support = [
                0.5 * (p_lo[0] + p_hi[0]),
                0.5 * (p_lo[1] + p_hi[1]),
                p_lo[2],
            ]
        _set_xyz(person, [
            person_xyz[0] + (cx - support[0]),
            person_xyz[1] + (cy - support[1]),
            person_xyz[2],
        ])
        if callable(_drop_to_support):
            try:
                _drop_to_support(
                    stage, str(person.GetPath()),
                    support_z=top_z, clearance=clearance)
            except Exception:
                person_xyz = _world_xyz(person)
                joints = _posed_joints(person)
                support = _support_from_joints(joints)
                if person_xyz is None or support is None:
                    continue
                _set_xyz(person, [
                    person_xyz[0], person_xyz[1],
                    person_xyz[2] + (top_z + clearance - support[2]),
                ])
        else:
            person_xyz = _world_xyz(person)
            joints = _posed_joints(person)
            support = _support_from_joints(joints)
            if person_xyz is None or support is None:
                continue
            _set_xyz(person, [
                person_xyz[0], person_xyz[1],
                person_xyz[2] + (top_z + clearance - support[2]),
            ])
        joints = _posed_joints(person)
        support = _support_from_joints(joints)
        if support is None:
            rng = _world_range(person)
            if rng is None:
                continue
            support = [
                0.5 * (rng[0][0] + rng[1][0]),
                0.5 * (rng[0][1] + rng[1][1]),
                rng[0][2],
            ]
        if not _contact_ok(support, lo, hi, top_z):
            print(
                "[on-landmark-patch] above-mode: rejected %s on %s "
                "(support=%.3f,%.3f,%.3f top_z=%.3f)"
                % (str(person.GetPath()), str(lm.GetPath()),
                   support[0], support[1], support[2], top_z)
            )
            continue
        used_lms.add(str(lm.GetPath()))
        used_people.add(str(person.GetPath()))
        n += 1
    globals()["_marc_above_placed_n"] = n
    print(
        "[on-landmark-patch] above-mode: target=%s placed %d/%d person(s) "
        "with surface contact" % ((jobs[0][1] if jobs else "-"), n, len(jobs))
    )
'''


def _ensure_above_helper(path):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    have_fn = "def _marc_place_above_people" in src
    have_contact = (
        "_marc_above_surface_contact" in src and "_marc_above_target" in src
        and "above-mode: target=" in src)
    if have_fn and have_contact:
        return False
    if have_fn:
        start = src.find("\ndef _marc_place_above_people")
        if start < 0:
            start = src.find("def _marc_place_above_people")
        if start >= 0:
            src = src[:start]
        action = "replaced"
    else:
        action = "appended"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src.rstrip() + ABOVE_HELPER)
    except OSError as e:
        print(f"[on-landmark-patch] above-helper write failed: {e}")
        return False
    print("[on-landmark-patch] deployer: above-mode helper %s" % action)
    return True



def _patch_file(path, replacers, label):
    if not os.path.exists(path):
        print(f"[on-landmark-patch] {label} not found, skipping: {path}")
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if MARKER in src and label == "deployer":
        print(f"[on-landmark-patch] {label} already applied, skipping")
        return True
    orig = src
    for old, new in replacers:
        if old in src:
            src = src.replace(old, new, 1)
    if src == orig:
        print(f"[on-landmark-patch] {label} anchors not found, skipping")
        return False
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    except OSError as e:
        print(f"[on-landmark-patch] {label} write failed, skipping: {e}")
        return False
    print(f"[on-landmark-patch] {label} patched")
    return True


def _enable_place_flag(src: str) -> str:
    """Turn every place_on_landmark=False into True (kwarg and comments)."""
    return src.replace("place_on_landmark=False", "place_on_landmark=True")


def main():
    n = 0
    if os.path.exists(TRAINER):
        with open(TRAINER, "r", encoding="utf-8") as f:
            tsrc = f.read()
        tnew = _enable_place_flag(tsrc)
        if NEW_SNAP not in tnew:
            if NEW_SNAP_V3 in tnew:
                tnew = tnew.replace(NEW_SNAP_V3, NEW_SNAP, 1)
            elif (NEW_SNAP_V2 in tnew and
                    "TRAINER_RENDER_WARMUP_FRAMES" not in tnew):
                tnew = tnew.replace(NEW_SNAP_V2, NEW_SNAP, 1)
            elif NEW_SNAP_V1 in tnew and "_marc_place_above_people" not in tnew:
                tnew = tnew.replace(NEW_SNAP_V1, NEW_SNAP, 1)
            elif OLD_SNAP in tnew and "reposition_on_landmark_people" not in tnew:
                tnew = tnew.replace(OLD_SNAP, NEW_SNAP, 1)
        if NEW_SAVE not in tnew:
            if NEW_SAVE_V1 in tnew:
                tnew = tnew.replace(NEW_SAVE_V1, NEW_SAVE, 1)
            elif OLD_SAVE in tnew:
                tnew = tnew.replace(OLD_SAVE, NEW_SAVE, 1)
        _old_above_skip = (
            "            if not _active_valid:\n"
            "                log.warning(\n"
            '                    f"Above scene {scene_id} skipped: placed={_n_placed} "\n'
            '                    f"stacked={_n_stack} people={len(_people)} "\n'
            '                    f"targets={len(_landmarks)} type={_target or \'-\'}")\n'
        )
        _new_above_skip = (
            "            log.info(\n"
            '                f"Above scene {scene_id}: target={_target or \'-\'} "\n'
            '                f"{\'saved\' if _active_valid else \'skipped\'} "\n'
            '                f"placed={_n_placed} stacked={_n_stack} "\n'
            '                f"people={len(_people)} landmarks={len(_landmarks)}")\n'
        )
        if _old_above_skip in tnew:
            tnew = tnew.replace(_old_above_skip, _new_above_skip, 1)
        if OLD_GT_LOG in tnew:
            tnew = tnew.replace(OLD_GT_LOG, NEW_GT_LOG, 1)
        _old_fail_msg = (
            "                    log.warning(\n"
            '                        f"Scene {trainer_state.scene_id} not saved "\n'
            '                        f"({auto_remaining} on-landmark scenes remain)")\n'
        )
        if _old_fail_msg in tnew:
            tnew = tnew.replace(
                _old_fail_msg,
                "                    _tgt = \"-\"\n"
                "                    try:\n"
                "                        from world import object_deployer as _od_fail\n"
                "                        _tgt = str(getattr(\n"
                "                            _od_fail, \"_marc_above_target\", \"\")"
                " or \"\") or \"-\"\n"
                "                    except Exception:\n"
                "                        _tgt = \"-\"\n"
                "                    log.warning(\n"
                '                        f"Scene {trainer_state.scene_id} not saved "\n'
                '                        f"(target={_tgt}, "\n'
                '                        f"{auto_remaining} on-landmark scenes remain)")\n',
                1,
            )
        if (OLD_CAPTURE_RETURN in tnew
                and "return _active_valid" not in tnew
                and "_active_valid = True" in tnew):
            tnew = tnew.replace(
                OLD_CAPTURE_RETURN, NEW_CAPTURE_RETURN, 1)
        if OLD_INITIAL_WARN in tnew:
            tnew = tnew.replace(OLD_INITIAL_WARN, NEW_INITIAL_WARN, 1)
        if NEW_POSITIONS not in tnew:
            if NEW_POSITIONS_V1 in tnew:
                tnew = tnew.replace(NEW_POSITIONS_V1, NEW_POSITIONS, 1)
            elif OLD_POSITIONS in tnew:
                tnew = tnew.replace(OLD_POSITIONS, NEW_POSITIONS, 1)
        if OLD_DEPLOY_LANDMARKS in tnew:
            tnew = tnew.replace(
                OLD_DEPLOY_LANDMARKS, NEW_DEPLOY_LANDMARKS, 1)
        if OLD_AUTO_REMAINING in tnew:
            tnew = tnew.replace(
                OLD_AUTO_REMAINING, NEW_AUTO_REMAINING, 1)
        if OLD_AUTO_ATTEMPT in tnew:
            tnew = tnew.replace(
                OLD_AUTO_ATTEMPT, NEW_AUTO_ATTEMPT, 1)
        if OLD_AUTO_FAILURE in tnew:
            tnew = tnew.replace(
                OLD_AUTO_FAILURE, NEW_AUTO_FAILURE, 1)
        if tnew != tsrc:
            try:
                with open(TRAINER, "w", encoding="utf-8") as f:
                    f.write(tnew)
                n += 1
                print("[on-landmark-patch] trainer.py: place_on_landmark=True "
                      "(+ surface contact, warmup; save/count on-top only)")
            except OSError as e:
                print(f"[on-landmark-patch] trainer write failed, skipping: {e}")
        else:
            print("[on-landmark-patch] trainer.py already has on-landmark placement")
    else:
        print(f"[on-landmark-patch] trainer not found, skipping: {TRAINER}")

    if _patch_file(
        DEPLOYER,
        [(OLD_PLACE_POSE, NEW_PLACE_POSE), (OLD_REPOSITION_PL, NEW_REPOSITION_PL)],
        "deployer",
    ):
        n += 1
    else:
        # Comment text may differ across content images; fall back to a regex.
        if os.path.exists(DEPLOYER):
            with open(DEPLOYER, "r", encoding="utf-8") as f:
                dsrc = f.read()
            if MARKER not in dsrc:
                dsrc2, k = re.subn(
                    r'placement = \(pose or \{\}\)\.get\("placement"\) or \{\}\n'
                    r'(?:[^\n]*\n)?'
                    r'( *)on_landmark_type = placement\.get\("on_landmark_type"\) '
                    r'if place_on_landmark else None',
                    NEW_PLACE_POSE.rstrip() + "\n",
                    dsrc,
                    count=1,
                )
                dsrc2, k2 = re.subn(
                    r'( *)placement = \(pose or \{\}\)\.get\("placement"\) or \{\}\n'
                    r'\1on_lm = placement\.get\("on_landmark_type"\)\n',
                    NEW_REPOSITION_PL,
                    dsrc2,
                    count=1,
                )
                if k or k2:
                    try:
                        with open(DEPLOYER, "w", encoding="utf-8") as f:
                            f.write(dsrc2)
                        print("[on-landmark-patch] deployer patched via regex fallback")
                        n += 1
                    except OSError as e:
                        print(f"[on-landmark-patch] deployer write failed: {e}")

    if os.path.exists(DEPLOYER):
        with open(DEPLOYER, "r", encoding="utf-8") as f:
            dsrc = f.read()
        _old_repo_lm = (
            '        on_lm = placement.get("on_landmark_type")\n'
            '        if not on_lm:\n'
            '            continue\n'
        )
        _new_repo_lm = (
            '        on_lm = placement.get("on_landmark_type")\n'
            '        if not on_lm or placement.get("above"):\n'
            '            continue\n'
        )
        if _old_repo_lm in dsrc and _new_repo_lm not in dsrc:
            try:
                with open(DEPLOYER, "w", encoding="utf-8") as f:
                    f.write(dsrc.replace(_old_repo_lm, _new_repo_lm, 1))
                print("[on-landmark-patch] deployer: skip origin-reseat for climbers")
                n += 1
            except OSError as e:
                print(f"[on-landmark-patch] deployer climber-skip failed: {e}")

    if _ensure_above_helper(DEPLOYER):
        n += 1

    if n:
        print("[on-landmark-patch] people with spec.placement land on that landmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
