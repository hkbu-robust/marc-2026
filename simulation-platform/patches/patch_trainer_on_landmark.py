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
#   4. if placement.above is set, construct the stack: person AABB sits on
#      the landmark AABB top (XY on the roof). For tall cabinets the
#      posed sitting mesh hangs below the bind-pose AABB, so Z is the
#      lowest posed joint on the mesh roof. Strip rigid bodies so 90
#      settle frames and GT playback cannot drop them to the ground.
#      Climb landmark+person stay in-frame with headroom; extras may go
#      off-frame.
#
# boot.py runs this before dataset-gen. Idempotent via MARKER. Pair with
# patch_trainer_people.py, which actually stamps placement onto people specs.
import inspect
import os
import re
import sys

TRAINER = "/metacom2026/simulation_app/marc2026/trainer.py"
DEPLOYER = "/metacom2026/simulation_app/marc2026/world/object_deployer.py"
MARKER = "_marc_obj_placement"

RECIPE_SCENE_FN = '''
def _marc_recipe_scene():
    """True when this scene is a TRAINER_SAR_RECIPES archetype."""
    if str(__import__("os").environ.get("TRAINER_SAR_RECIPES", "0")).strip().lower() in (
            "0", "false", "no", "off", ""):
        return False
    try:
        import json as _json_rs
        with open("/tmp/marc_sar_recipe.json", encoding="utf-8") as _rf:
            rec = _json_rs.load(_rf) or {}
        return bool(rec.get("recipe_id"))
    except Exception:
        return False
'''

ABOVE_SCENE_FN = RECIPE_SCENE_FN + '''
def _marc_above_scene():
    """True when this scene should frame a climb stack (env or recipe.above)."""
    if str(__import__("os").environ.get("TRAINER_PEOPLE_ABOVE", "0")).strip().lower() not in (
            "0", "false", "no", "off", ""):
        return True
    try:
        import json as _json_ab
        with open("/tmp/marc_sar_recipe.json", encoding="utf-8") as _rf:
            return bool(_json_ab.load(_rf).get("above"))
    except Exception:
        return False
'''

_ABOVE_ON_RE = re.compile(
    r"str\((?:os|__import__\(\"os\"\))\.environ\.get\(\"TRAINER_PEOPLE_ABOVE\", \"0\"\)\)"
    r"\.lower\(\) not in \(\s*\"0\", \"false\", \"no\", \"off\", \"\"\)",
    re.S,
)
_ABOVE_OFF_RE = re.compile(
    r"str\(\s*(?:os|__import__\(\"os\"\))\.environ\.get\(\"TRAINER_PEOPLE_ABOVE\", \"0\"\)\s*\)"
    r"\s*\.lower\(\) in \(\s*\"0\", \"false\", \"no\", \"off\", \"\"\)",
    re.S,
)


def _rewrite_above_gates(src: str) -> str:
    """Env-only climb framing also runs when the current recipe has above=true."""
    src = _ABOVE_ON_RE.sub("_marc_above_scene()", src)
    src = _ABOVE_OFF_RE.sub("(not _marc_above_scene())", src)
    return src


def _ensure_module_above_scene(src: str) -> str:
    if "def _marc_recipe_scene(" not in src:
        if "def _marc_above_scene(" in src:
            src = src.replace(
                "def _marc_above_scene(",
                RECIPE_SCENE_FN.strip() + "\n\ndef _marc_above_scene(",
                1,
            )
        else:
            idx = src.find("\ndef ")
            if idx < 0:
                return ABOVE_SCENE_FN + "\n" + src
            src = src[:idx] + "\n" + ABOVE_SCENE_FN + src[idx:]
    elif "def _marc_above_scene(" not in src:
        idx = src.find("\ndef ")
        if idx < 0:
            return ABOVE_SCENE_FN + "\n" + src
        src = src[:idx] + "\n" + ABOVE_SCENE_FN + src[idx:]
    return src


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
    if _marc_above_scene():
        # Only the climber (placement.above) may sit on a landmark. Pose YAML
        # would otherwise drop extras (lying/prone) through the same vending.
        if not placement.get("above") and not placement.get("recipe_id"):
            on_landmark_type = None
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
        if not on_lm:
            continue
        if _marc_above_scene():
            if not placement.get("above") and not placement.get("recipe_id"):
                continue
        z_off = float(placement.get("z_offset_from_top",
                                    placement.get("contact_clearance", 0.0)) or 0.0)
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
        if _marc_above_scene():
            try:
                _warm = max(1, int(os.environ.get(
                    "TRAINER_RENDER_WARMUP_FRAMES", "12") or "12"))
            except (TypeError, ValueError):
                _warm = 12
            for _ in range(_warm):
                simulation_context.step(render=True)
'''

NEW_SNAP_V4 = '''        _snap_landmarks_to_ground([mission])
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
        if _marc_above_scene():
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

NEW_SNAP = '''        _snap_landmarks_to_ground([mission])
        try:
            from world.object_deployer import reposition_on_landmark_people
            reposition_on_landmark_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"reposition on-landmark people failed (non-fatal): {_e}")
        try:
            simulation_context.pause()
        except Exception:
            pass
        try:
            from world.object_deployer import _marc_place_above_people
            _marc_place_above_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"place-above people failed (non-fatal): {_e}")
        try:
            from world.object_deployer import _marc_place_beside_people
            _marc_place_beside_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"place-beside people failed (non-fatal): {_e}")
        try:
            from world.object_deployer import _marc_place_seat_people
            _marc_place_seat_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"place-seat people failed (non-fatal): {_e}")
'''

OLD_SETTLE = '''        simulation_context.play()
        for _ in range(_SETTLE_FRAMES):
            simulation_context.step(render=False)
        _snap_landmarks_to_ground([mission])
'''

NEW_SETTLE = '''        if _marc_above_scene():
            # Construct the stack first, freeze the climber, then let lost
            # objects fall. Re-seat every settle tick so PhysX cannot restore
            # the ground spawn pose. Clear the contact lock once per scene.
            try:
                simulation_context.pause()
            except Exception:
                pass
            _snap_landmarks_to_ground([mission])
            try:
                from world import object_deployer as _od_lock
                _od_lock._marc_above_lock = {}
            except Exception:
                pass
            try:
                from world.object_deployer import (
                    reposition_on_landmark_people,
                    _marc_place_above_people,
                )
                reposition_on_landmark_people(
                    mission["id"], mission.get("objects") or [],
                    configurations)
                _marc_place_above_people(
                    mission["id"], mission.get("objects") or [],
                    configurations)
            except Exception as _e:
                log.warning(f"place-above after snap failed (non-fatal): {_e}")
            simulation_context.play()
            for _ in range(_SETTLE_FRAMES):
                try:
                    from world.object_deployer import _marc_place_above_people
                    _marc_place_above_people(
                        mission["id"], mission.get("objects") or [],
                        configurations)
                except Exception:
                    pass
                simulation_context.step(render=False)
            try:
                simulation_context.pause()
            except Exception:
                pass
            try:
                from world.object_deployer import _marc_place_above_people
                _marc_place_above_people(
                    mission["id"], mission.get("objects") or [],
                    configurations)
            except Exception as _e:
                log.warning(f"place-above after settle failed (non-fatal): {_e}")
        elif _marc_recipe_scene():
            # Seat-lie / beside never freeze as a climb stack. Re-place the
            # recipe target every settle tick so PhysX cannot throw them
            # 10 m from the support the camera actually sees.
            try:
                simulation_context.pause()
            except Exception:
                pass
            _snap_landmarks_to_ground([mission])
            def _marc_reseat_recipe():
                try:
                    from world.object_deployer import (
                        reposition_on_landmark_people,
                        _marc_place_beside_people,
                        _marc_place_seat_people,
                    )
                    reposition_on_landmark_people(
                        mission["id"], mission.get("objects") or [],
                        configurations)
                    _marc_place_seat_people(
                        mission["id"], mission.get("objects") or [],
                        configurations)
                    _marc_place_beside_people(
                        mission["id"], mission.get("objects") or [],
                        configurations)
                except Exception as _e:
                    log.warning(f"recipe reseat failed (non-fatal): {_e}")
            _marc_reseat_recipe()
            simulation_context.play()
            for _ in range(_SETTLE_FRAMES):
                try:
                    _marc_reseat_recipe()
                except Exception:
                    pass
                simulation_context.step(render=False)
            try:
                simulation_context.pause()
            except Exception:
                pass
            _marc_reseat_recipe()
        else:
            simulation_context.play()
            for _ in range(_SETTLE_FRAMES):
                simulation_context.step(render=False)
            _snap_landmarks_to_ground([mission])
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
        if _marc_above_scene():
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

NEW_SAVE_V2 = '''        active_payload = gt_by_mission.get(active_mission_id)
        _active_valid = True
        if _marc_above_scene():
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

NEW_SAVE_V3 = '''        active_payload = gt_by_mission.get(active_mission_id)
        _active_valid = True
        if _marc_above_scene():
            _img = (active_payload or {}).get("image") or {}
            try:
                _ih = float(_img.get("height") or 0.0)
            except (TypeError, ValueError):
                _ih = 0.0
            _top_m = max(8.0, 0.03 * _ih) if _ih else 8.0
            _want = {
                x.strip() for x in
                (os.environ.get("TRAINER_PEOPLE_ABOVE_TYPES", "") or "").split(",")
                if x.strip()
            }
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
            def _marc_eval_above(_payload):
                try:
                    from world import object_deployer as _od_above
                    _n_placed = int(getattr(
                        _od_above, "_marc_above_placed_n", 0) or 0)
                    _target = str(getattr(
                        _od_above, "_marc_above_target", "") or "")
                except Exception:
                    _n_placed = 0
                    _target = ""
                _want_now = {_target} if _target else _want
                _dets = list((_payload or {}).get("detections") or [])
                _people = [
                    d for d in _dets
                    if str(d.get("class") or "").startswith("person_")
                ]
                _landmarks = [
                    d for d in _dets
                    if (
                        str(d.get("class") or "") in _want_now
                        if _want_now else d.get("kind") == "landmark"
                    )
                ]
                _n_stack = sum(
                    1 for _p in _people for _lm in _landmarks
                    if _visible_stack(_p, _lm)
                )
                return _n_placed, _target, _n_stack, len(_people), len(_landmarks)
            _n_placed, _target, _n_stack, _n_peep, _n_lm = _marc_eval_above(
                active_payload)
            _active_valid = _n_placed > 0 and _n_stack > 0
            _nudge_i = 0
            while (not _active_valid) and _n_placed > 0 and _nudge_i < 3:
                _nudge_i += 1
                try:
                    from world.object_deployer import (
                        _marc_nudge_above_nearer as _nudge_fn,
                        _marc_place_above_people as _place_fn,
                    )
                    _miss = next(
                        m for m in missions if m.get("id") == active_mission_id)
                    if not _nudge_fn(
                            _miss["id"], _miss.get("objects") or [],
                            configurations):
                        break
                    _place_fn(
                        _miss["id"], _miss.get("objects") or [],
                        configurations)
                    for _ in range(6):
                        simulation_context.step(render=True)
                    gt_by_mission = gt.capture(
                        scene_id, simulation_app,
                        stamp=float(getattr(
                            simulation_context, "current_time", 0.0) or 0.0),
                        render_step=lambda: simulation_context.step(
                            render=True))
                    total = 0
                    for mid, payload in gt_by_mission.items():
                        ros2.publish_groundtruth(mid, payload)
                        total += len(payload.get("detections", []))
                    active_payload = gt_by_mission.get(active_mission_id)
                    _n_placed, _target, _n_stack, _n_peep, _n_lm = (
                        _marc_eval_above(active_payload))
                    _active_valid = _n_placed > 0 and _n_stack > 0
                    log.info(
                        f"Above scene {scene_id}: nudge {_nudge_i} "
                        f"target={_target or '-'} "
                        f"{'aligned' if _active_valid else 'retry'} "
                        f"placed={_n_placed} stacked={_n_stack}")
                except Exception as _ne:
                    log.warning(
                        f"above-mode nudge failed (non-fatal): {_ne}")
                    break
            log.info(
                f"Above scene {scene_id}: target={_target or '-'} "
                f"{'saved' if _active_valid else 'skipped'} "
                f"placed={_n_placed} stacked={_n_stack} "
                f"people={_n_peep} landmarks={_n_lm}")
        if save_offline_enabled and _active_valid:
            # 선택된 카메라만 저장 (배치 콘텐츠가 있는 active 미션).
            if active_payload is not None:
                try:
                    gt.save_offline(scene_id, {active_mission_id: active_payload})
                except Exception as e:
                    log.warning(f"offline GT save failed: {e}")
'''

NEW_SAVE_V4 = '''        active_payload = gt_by_mission.get(active_mission_id)
        _active_valid = True
        if _marc_above_scene():
            try:
                from world import object_deployer as _od_above
                _n_placed = int(getattr(
                    _od_above, "_marc_above_placed_n", 0) or 0)
                _target = str(getattr(
                    _od_above, "_marc_above_target", "") or "")
            except Exception:
                _n_placed = 0
                _target = ""
            # 3D raycast+joint contact only. 2D boxes and AABBs disagree
            # with a pitched CCTV view and are not a placement oracle.
            _active_valid = _n_placed > 0
            if not _active_valid:
                raise RuntimeError(
                    f"Above scene {scene_id}: constructed stack invariant "
                    f"failed (target={_target or '-'} placed={_n_placed})")
            log.info(
                f"Above scene {scene_id}: target={_target or '-'} "
                f"saved placed={_n_placed}")
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
        if _marc_above_scene():
            try:
                from world import object_deployer as _od_above
                _n_placed = int(getattr(
                    _od_above, "_marc_above_placed_n", 0) or 0)
                _target = str(getattr(
                    _od_above, "_marc_above_target", "") or "")
            except Exception:
                _n_placed = 0
                _target = ""
            _dets = list((active_payload or {}).get("detections") or [])
            _target_in_view = True
            if _target:
                _t = str(_target)
                _target_in_view = any(
                    str(d.get("class") or "") == _t
                    or str(d.get("class") or "").startswith(_t)
                    for d in _dets)
            _img = (active_payload or {}).get("image") or {}
            try:
                _ih = float(_img.get("height") or 0.0)
            except (TypeError, ValueError):
                _ih = 0.0
            _margin = max(8.0, 0.03 * _ih) if _ih else 8.0
            _headroom = True
            for _d in _dets:
                _b = _d.get("bbox") or []
                if len(_b) != 4:
                    continue
                try:
                    _y0 = min(float(_b[1]), float(_b[3]))
                except (TypeError, ValueError):
                    continue
                _cls = str(_d.get("class") or "")
                if _d.get("recipe_id") and _y0 <= _margin:
                    _headroom = False
                    break
                if _target and (
                        _cls == _target or _cls.startswith(_target)
                        ) and _y0 <= _margin:
                    _headroom = False
                    break
            try:
                from world import object_deployer as _od_view
                _od_view._marc_above_target_in_view = (
                    _target_in_view and _headroom)
            except Exception:
                pass
            _stack_ok = True
            _lm_d = None
            if _target:
                _t = str(_target)
                for _d in _dets:
                    _cls = str(_d.get("class") or "")
                    if _cls == _t or _cls.startswith(_t):
                        _lm_d = _d
                        break
            # _marc_recipe_climb_visible: landmark-in-view saved empty
            # vending close-ups; require the person stacked on it.
            try:
                from world.object_deployer import visible_above_stack
                _people_d = [
                    _d for _d in _dets
                    if str(_d.get("class") or "").startswith("person_")]
                _rid_people = [p for p in _people_d if p.get("recipe_id")]
                _try = _rid_people or _people_d
                if _lm_d is None or not _try:
                    _stack_ok = False
                else:
                    _stack_ok = any(
                        visible_above_stack(p, _lm_d, _ih) for p in _try)
            except Exception:
                _stack_ok = any(
                    str(d.get("class") or "").startswith("person_")
                    for d in _dets)
            # A seated body can overlap the cabinet in projection instead of
            # appearing strictly above it; the helper already verified 3D contact.
            _seat_ok = False
            try:
                import json as _json_seat
                _seat_rec = _json_seat.load(
                    open("/tmp/marc_sar_recipe.json")) or {}
                _seat_ok = "sitting" in str(
                    _seat_rec.get("target_pose") or "")
            except Exception:
                pass
            # Skip save if the climb landmark is off camera, or if the
            # roof / climber is cropped at the top of the CCTV frame.
            _active_valid = (
                _n_placed > 0 and _target_in_view and _headroom
                and (_stack_ok or _seat_ok))
            if _n_placed <= 0:
                log.warning(
                    f"Above scene {scene_id}: constructed stack invariant "
                    f"failed (target={_target or '-'} placed={_n_placed}); "
                    f"skip save")
            elif not _target_in_view:
                log.warning(
                    f"Above scene {scene_id}: target={_target or '-'} "
                    f"not in CCTV detections; skip save")
            elif not _headroom:
                log.warning(
                    f"Above scene {scene_id}: target={_target or '-'} "
                    f"roof/climber top-clipped; skip save")
            elif not (_stack_ok or _seat_ok):
                log.warning(
                    f"Above scene {scene_id}: target={_target or '-'} "
                    f"climber not visible on support; skip save")
            else:
                log.info(
                    f"Above scene {scene_id}: target={_target or '-'} "
                    f"saved placed={_n_placed}")
        elif _marc_recipe_scene():
            _dets = list((active_payload or {}).get("detections") or [])
            rec = {}
            try:
                import json as _json_sv
                rec = _json_sv.load(open("/tmp/marc_sar_recipe.json")) or {}
            except Exception:
                rec = {}
            _rid = str(rec.get("recipe_id") or "")
            _pose = str(rec.get("target_pose") or "")
            _sup = str(
                rec.get("support_class")
                or rec.get("support_landmark")
                or rec.get("surface_anchor")
                or "")
            _rel = rec.get("relation")
            if _rel is True:
                _rel = "on"
            _rel = str(_rel or "").strip().lower()
            _tgt = None
            for _d in _dets:
                if str(_d.get("class") or "").startswith("person_"):
                    if _rid and str(_d.get("recipe_id") or "") == _rid:
                        _tgt = _d
                        break
            _sup_d = None
            if _sup:
                _best = None
                _pxyz = (_tgt or {}).get("world_xyz")
                for _d in _dets:
                    if str(_d.get("class") or "") != _sup:
                        continue
                    _w = _d.get("world_xyz")
                    if (_pxyz and _w and len(_pxyz) >= 2 and len(_w) >= 2):
                        _dd = ((_pxyz[0] - _w[0]) ** 2
                               + (_pxyz[1] - _w[1]) ** 2) ** 0.5
                        if _best is None or _dd < _best[0]:
                            _best = (_dd, _d)
                    elif _sup_d is None:
                        _sup_d = _d
                if _best is not None:
                    _sup_d = _best[1]
            _pair_ok = True
            if _tgt is not None and _sup_d is not None:
                _pxyz = _tgt.get("world_xyz")
                _w = _sup_d.get("world_xyz")
                if _pxyz and _w and len(_pxyz) >= 2 and len(_w) >= 2:
                    _dxy = ((_pxyz[0] - _w[0]) ** 2
                            + (_pxyz[1] - _w[1]) ** 2) ** 0.5
                    try:
                        _max = float(__import__("os").environ.get(
                            "TRAINER_SAR_PAIR_XY_MAX", "2.8") or "2.8")
                    except (TypeError, ValueError):
                        _max = 2.8
                    if _rel in ("on", "above"):
                        try:
                            _max = min(_max, float(__import__("os").environ.get(
                                "TRAINER_SAR_ON_XY_MAX", "1.4") or "1.4"))
                        except (TypeError, ValueError):
                            _max = min(_max, 1.4)
                    if _dxy > _max:
                        _pair_ok = False
            if (bool(rec.get("above")) and _rel in ("on", "above")
                    and _tgt is not None and _sup_d is not None):
                # _marc_recipe_climb_visible: no same-pose distractor
                # fallback; person must be stacked on the support in-frame.
                try:
                    from world.object_deployer import visible_above_stack
                    _imgh = 0.0
                    try:
                        _imgh = float(
                            ((active_payload or {}).get("image") or {}).get(
                                "height") or 0.0)
                    except (TypeError, ValueError):
                        _imgh = 0.0
                    if not visible_above_stack(_tgt, _sup_d, _imgh):
                        _pair_ok = False
                except Exception:
                    pass
            _active_valid = (
                _tgt is not None
                and ((not _sup) or _sup_d is not None)
                and _pair_ok)
            try:
                from world import object_deployer as _od_view
                _od_view._marc_above_target_in_view = bool(_active_valid)
            except Exception:
                pass
            if not _active_valid:
                log.warning(
                    f"Recipe scene {scene_id}: "
                    f"id={_rid or '-'} pose={_pose or '-'} "
                    f"support={_sup or '-'} pair={_pair_ok}; skip save")
            else:
                log.info(
                    f"Recipe scene {scene_id}: "
                    f"id={_rid or '-'} pose={_pose or '-'} saved")
        if save_offline_enabled and _active_valid:
            # 선택된 카메라만 저장 (배치 콘텐츠가 있는 active 미션).
            if active_payload is not None:
                try:
                    gt.save_offline(scene_id, {active_mission_id: active_payload})
                except Exception as e:
                    log.warning(f"offline GT save failed: {e}")
'''

OLD_RENDER_STEP = (
    "        simulation_context.play()\n"
    "        stamp = float(getattr(simulation_context, \"current_time\", 0.0) or 0.0)\n"
    "        gt_by_mission = gt.capture(\n"
    "            scene_id, simulation_app, stamp=stamp,\n"
    "            render_step=lambda: simulation_context.step(render=True))\n"
)
NEW_RENDER_STEP = '''        if _marc_above_scene() or _marc_recipe_scene():
            try:
                simulation_context.pause()
            except Exception:
                pass
            try:
                _miss = next(
                    m for m in missions if m.get("id") == active_mission_id)
                if _marc_above_scene():
                    from world.object_deployer import _marc_place_above_people
                    _marc_place_above_people(
                        _miss["id"], _miss.get("objects") or [],
                        configurations)
                elif _marc_recipe_scene():
                    from world.object_deployer import (
                        reposition_on_landmark_people,
                        _marc_place_beside_people,
                        _marc_place_seat_people,
                    )
                    reposition_on_landmark_people(
                        _miss["id"], _miss.get("objects") or [],
                        configurations)
                    _marc_place_seat_people(
                        _miss["id"], _miss.get("objects") or [],
                        configurations)
                    _marc_place_beside_people(
                        _miss["id"], _miss.get("objects") or [],
                        configurations)
            except Exception as _pe:
                log.warning(
                    f"recipe reseat before GT capture failed (non-fatal): {_pe}")
        simulation_context.play()
        def _marc_gt_step():
            # Playback ticks restore spawn poses. Freeze/re-seat the climber
            # on the landmark top before AND after each annotator frame.
            def _marc_reseat():
                if _marc_above_scene():
                    try:
                        from world.object_deployer import _marc_place_above_people
                        _miss = next(
                            m for m in missions if m.get("id") == active_mission_id)
                        _marc_place_above_people(
                            _miss["id"], _miss.get("objects") or [],
                            configurations)
                    except Exception as _pe:
                        log.warning(
                            f"place-above GT step failed (non-fatal): {_pe}")
                elif _marc_recipe_scene():
                    try:
                        from world.object_deployer import (
                            reposition_on_landmark_people,
                            _marc_place_beside_people,
                            _marc_place_seat_people,
                        )
                        _miss = next(
                            m for m in missions if m.get("id") == active_mission_id)
                        reposition_on_landmark_people(
                            _miss["id"], _miss.get("objects") or [],
                            configurations)
                        _marc_place_seat_people(
                            _miss["id"], _miss.get("objects") or [],
                            configurations)
                        _marc_place_beside_people(
                            _miss["id"], _miss.get("objects") or [],
                            configurations)
                    except Exception as _pe:
                        log.warning(
                            f"recipe reseat GT step failed (non-fatal): {_pe}")
            _marc_reseat()
            simulation_context.step(render=True)
            _marc_reseat()
        stamp = float(getattr(simulation_context, "current_time", 0.0) or 0.0)
        gt_by_mission = gt.capture(
            scene_id, simulation_app, stamp=stamp,
            render_step=_marc_gt_step)
'''

NEW_RENDER_STEP_V1 = '''        simulation_context.play()
        def _marc_gt_step():
            # Playback ticks restore spawn poses. Re-seat the climber on
            # the landmark top before each annotator/render frame.
            if _marc_above_scene():
                try:
                    from world.object_deployer import _marc_place_above_people
                    _miss = next(
                        m for m in missions if m.get("id") == active_mission_id)
                    _marc_place_above_people(
                        _miss["id"], _miss.get("objects") or [],
                        configurations)
                except Exception as _pe:
                    log.warning(
                        f"place-above before GT step failed (non-fatal): {_pe}")
            simulation_context.step(render=True)
        stamp = float(getattr(simulation_context, "current_time", 0.0) or 0.0)
        gt_by_mission = gt.capture(
            scene_id, simulation_app, stamp=stamp,
            render_step=_marc_gt_step)
'''

OLD_CAPTURE_RETURN = "        return gt.last_render_ok\n"
NEW_CAPTURE_RETURN = '''        if _marc_above_scene():
            # Scene 0 often has a valid RGB but the climb landmark is off
            # camera. Return False so the stock black-frame redeploy retries
            # instead of keeping a cabinet-less first frame.
            try:
                from world import object_deployer as _od_cap
                _n = int(getattr(_od_cap, "_marc_above_placed_n", 0) or 0)
                _in = bool(getattr(
                    _od_cap, "_marc_above_target_in_view", True))
            except Exception:
                _n, _in = 0, True
            return bool(gt.last_render_ok) and _n > 0 and _in
        if _marc_recipe_scene():
            try:
                _pld = (gt_by_mission or {}).get(active_mission_id) or {}
                _dets = list(_pld.get("detections") or [])
                import json as _json_cap
                rec = _json_cap.load(open("/tmp/marc_sar_recipe.json")) or {}
                _rid = str(rec.get("recipe_id") or "")
                _pose = str(rec.get("target_pose") or "")
                _sup = str(
                    rec.get("support_class")
                    or rec.get("support_landmark")
                    or rec.get("surface_anchor")
                    or "")
                _people_ok = any(
                    (_rid and str(d.get("recipe_id") or "") == _rid)
                    or ((not _rid) and _pose
                        and str(d.get("class") or "") == _pose)
                    for d in _dets
                    if str(d.get("class") or "").startswith("person_"))
                _sup_ok = (not _sup) or any(
                    str(d.get("class") or "") == _sup for d in _dets)
                return bool(gt.last_render_ok) and _people_ok and _sup_ok
            except Exception:
                return bool(gt.last_render_ok)
        return gt.last_render_ok
'''

NEW_CAPTURE_RETURN_V1 = '''        if _marc_above_scene():
            # Scene 0 often has a valid RGB but the climb landmark is off
            # camera. Return False so the stock black-frame redeploy retries
            # instead of keeping a cabinet-less first frame.
            try:
                from world import object_deployer as _od_cap
                _n = int(getattr(_od_cap, "_marc_above_placed_n", 0) or 0)
                _in = bool(getattr(
                    _od_cap, "_marc_above_target_in_view", True))
            except Exception:
                _n, _in = 0, True
            return bool(gt.last_render_ok) and _n > 0 and _in
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

NEW_INITIAL_WARN = '''    if not render_ok and gt is not None and (not _marc_above_scene()):
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
                    if gt is not None and (not _marc_above_scene()):
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
        if _marc_above_scene():
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

NEW_POSITIONS_V2 = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if _marc_above_scene():
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

NEW_POSITIONS_V3 = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if _marc_above_scene():
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
            if _lm_i is not None and _lm_i < len(lm_pos):
                _spot = None
                if isinstance(area, dict):
                    try:
                        _tl = list(area["top_left"])
                        _tr = list(area["top_right"])
                        _br = list(area["bottom_right"])
                        _bl = list(area["bottom_left"])
                        _u = 0.42 + random.random() * 0.16
                        _v = (0.20 + random.random() * 0.12
                              if _pl in ("vending_machine", "postbox")
                              else 0.30 + random.random() * 0.12)
                        _bot = [(1.0 - _u) * _bl[_k] + _u * _br[_k]
                                for _k in range(3)]
                        _top = [(1.0 - _u) * _tl[_k] + _u * _tr[_k]
                                for _k in range(3)]
                        _spot = [(1.0 - _v) * _bot[_k] + _v * _top[_k]
                                 for _k in range(3)]
                        try:
                            from world import object_deployer as _od_near
                            _od_near._marc_above_near_xy = (
                                _bot[0] - _top[0], _bot[1] - _top[1])
                        except Exception:
                            pass
                    except Exception:
                        _spot = None
                if _spot is not None:
                    lm_pos[_lm_i] = _spot
                if (_climber_i is not None
                        and _climber_i < len(obj_pos)):
                    obj_pos[_climber_i] = list(lm_pos[_lm_i])
                _lx, _ly = lm_pos[_lm_i][0], lm_pos[_lm_i][1]
                _ndx = _ndy = None
                if _spot is not None:
                    try:
                        _ndx = _bot[0] - _top[0]
                        _ndy = _bot[1] - _top[1]
                    except Exception:
                        _ndx = _ndy = None
                _mag = None
                if _ndx is not None:
                    _mag = (_ndx * _ndx + _ndy * _ndy) ** 0.5
                _oi = 0
                for _spec in objects or []:
                    _n = max(1, int(_spec.get("count", 1) or 1))
                    _is_climber = bool((_spec.get("placement") or {}).get("above"))
                    _is_person = bool(_spec.get("character")) or str(
                        _spec.get("type") or "").startswith("person_")
                    for _ in range(_n):
                        if (_is_person and (not _is_climber)
                                and _oi < len(obj_pos)):
                            _dx = obj_pos[_oi][0] - _lx
                            _dy = obj_pos[_oi][1] - _ly
                            if (_dx * _dx + _dy * _dy) < 4.0:
                                if _mag and _mag > 1e-4:
                                    obj_pos[_oi] = [
                                        _lx + _ndx / _mag * 2.4,
                                        _ly + _ndy / _mag * 2.4,
                                        obj_pos[_oi][2],
                                    ]
                        _oi += 1
'''

NEW_POSITIONS_V5 = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if _marc_above_scene():
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
            if isinstance(area, dict):
                try:
                    _tl = list(area["top_left"])
                    _tr = list(area["top_right"])
                    _br = list(area["bottom_right"])
                    _bl = list(area["bottom_left"])
                    def _marc_quad(_u, _v):
                        _bot = [(1.0 - _u) * _bl[_k] + _u * _br[_k]
                                for _k in range(3)]
                        _top = [(1.0 - _u) * _tl[_k] + _u * _tr[_k]
                                for _k in range(3)]
                        return [(1.0 - _v) * _bot[_k] + _v * _top[_k]
                                for _k in range(3)]
                    _climb_u = 0.28 + random.random() * 0.44
                    _climb_v = (0.18 + random.random() * 0.14
                                if _pl in ("vending_machine", "postbox")
                                else 0.22 + random.random() * 0.14)
                    _climb = _marc_quad(_climb_u, _climb_v)
                    try:
                        from world import object_deployer as _od_near
                        _od_near._marc_above_near_xy = (
                            _bl[0] - _tl[0], _bl[1] - _tl[1])
                    except Exception:
                        pass
                    if _lm_i is not None and _lm_i < len(lm_pos):
                        lm_pos[_lm_i] = list(_climb)
                    if (_climber_i is not None
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
                    for _i in range(len(lm_pos)):
                        if _i == _lm_i:
                            continue
                        _p = list(_climb)
                        for _ in range(8):
                            _u = 0.30 + random.random() * 0.40
                            _v = 0.16 + random.random() * 0.22
                            _cand = _marc_quad(_u, _v)
                            _dx = _cand[0] - _climb[0]
                            _dy = _cand[1] - _climb[1]
                            if (_dx * _dx + _dy * _dy) >= 4.0:
                                _p = _cand
                                break
                        lm_pos[_i] = _p
                    _oi = 0
                    for _spec in objects or []:
                        _n = max(1, int(_spec.get("count", 1) or 1))
                        for _ in range(_n):
                            if (_oi < len(obj_pos)
                                    and _oi != _climber_i):
                                _p = list(obj_pos[_oi])
                                for _ in range(8):
                                    _u = 0.30 + random.random() * 0.40
                                    _v = 0.16 + random.random() * 0.22
                                    _cand = _marc_quad(_u, _v)
                                    _dx = _cand[0] - _climb[0]
                                    _dy = _cand[1] - _climb[1]
                                    if (_dx * _dx + _dy * _dy) >= 4.0:
                                        _p = _cand
                                        break
                                obj_pos[_oi] = _p
                            _oi += 1
                except Exception:
                    if (_lm_i is not None and _climber_i is not None
                            and _lm_i < len(lm_pos)
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
            elif (_lm_i is not None and _climber_i is not None
                    and _lm_i < len(lm_pos)
                    and _climber_i < len(obj_pos)):
                obj_pos[_climber_i] = list(lm_pos[_lm_i])
'''

NEW_POSITIONS_V4 = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if _marc_above_scene():
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
            if isinstance(area, dict):
                try:
                    _tl = list(area["top_left"])
                    _tr = list(area["top_right"])
                    _br = list(area["bottom_right"])
                    _bl = list(area["bottom_left"])
                    def _marc_quad(_u, _v):
                        _bot = [(1.0 - _u) * _bl[_k] + _u * _br[_k]
                                for _k in range(3)]
                        _top = [(1.0 - _u) * _tl[_k] + _u * _tr[_k]
                                for _k in range(3)]
                        return [(1.0 - _v) * _bot[_k] + _v * _top[_k]
                                for _k in range(3)]
                    _climb_u = 0.24 + random.random() * 0.52
                    _climb_v = (0.08 + random.random() * 0.16
                                if _pl in ("vending_machine", "postbox")
                                else 0.12 + random.random() * 0.18)
                    _climb = _marc_quad(_climb_u, _climb_v)
                    try:
                        from world import object_deployer as _od_near
                        _od_near._marc_above_near_xy = (
                            _bl[0] - _tl[0], _bl[1] - _tl[1])
                    except Exception:
                        pass
                    if _lm_i is not None and _lm_i < len(lm_pos):
                        lm_pos[_lm_i] = list(_climb)
                    if (_climber_i is not None
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
                    for _i in range(len(lm_pos)):
                        if _i == _lm_i:
                            continue
                        _p = list(_climb)
                        for _ in range(10):
                            _u = random.random() * 1.05 - 0.025
                            _v = 0.04 + random.random() * 0.66
                            _cand = _marc_quad(_u, _v)
                            _dx = _cand[0] - _climb[0]
                            _dy = _cand[1] - _climb[1]
                            if (_dx * _dx + _dy * _dy) >= 9.0:
                                _p = _cand
                                break
                        lm_pos[_i] = _p
                    _oi = 0
                    for _spec in objects or []:
                        _n = max(1, int(_spec.get("count", 1) or 1))
                        for _ in range(_n):
                            if (_oi < len(obj_pos)
                                    and _oi != _climber_i):
                                _p = list(obj_pos[_oi])
                                for _ in range(10):
                                    _u = random.random() * 1.20 - 0.10
                                    _v = random.random() * 0.90
                                    _cand = _marc_quad(_u, _v)
                                    _dx = _cand[0] - _climb[0]
                                    _dy = _cand[1] - _climb[1]
                                    if (_dx * _dx + _dy * _dy) >= 2.25:
                                        _p = _cand
                                        break
                                obj_pos[_oi] = _p
                            _oi += 1
                except Exception:
                    if (_lm_i is not None and _climber_i is not None
                            and _lm_i < len(lm_pos)
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
            elif (_lm_i is not None and _climber_i is not None
                    and _lm_i < len(lm_pos)
                    and _climber_i < len(obj_pos)):
                obj_pos[_climber_i] = list(lm_pos[_lm_i])
'''

NEW_POSITIONS_V6 = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if _marc_above_scene():
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
            if isinstance(area, dict):
                try:
                    _tl = list(area["top_left"])
                    _tr = list(area["top_right"])
                    _br = list(area["bottom_right"])
                    _bl = list(area["bottom_left"])
                    def _marc_quad(_u, _v):
                        _bot = [(1.0 - _u) * _bl[_k] + _u * _br[_k]
                                for _k in range(3)]
                        _top = [(1.0 - _u) * _tl[_k] + _u * _tr[_k]
                                for _k in range(3)]
                        return [(1.0 - _v) * _bot[_k] + _v * _top[_k]
                                for _k in range(3)]
                    # Tall climb targets: nearer-mid so the roof stays below
                    # the image top. Helper also frames the roof in camera.
                    _climb_u = 0.30 + random.random() * 0.40
                    if _pl == "vending_machine":
                        # Nearer fallback if camera framing misses. Helper
                        # then parks the roof on a mid-image downward ray.
                        _climb_v = 0.20 + random.random() * 0.12
                    elif _pl == "postbox":
                        _climb_v = 0.22 + random.random() * 0.16
                    else:
                        _climb_v = 0.16 + random.random() * 0.18
                    _climb = _marc_quad(_climb_u, _climb_v)
                    try:
                        from world import object_deployer as _od_near
                        _od_near._marc_above_near_xy = (
                            _bl[0] - _tl[0], _bl[1] - _tl[1])
                    except Exception:
                        pass
                    if _lm_i is not None and _lm_i < len(lm_pos):
                        lm_pos[_lm_i] = list(_climb)
                    if (_climber_i is not None
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
                    _extra_k = 0
                    for _i in range(len(lm_pos)):
                        if _i == _lm_i:
                            continue
                        _u = (0.08 + random.random() * 0.10
                              if _extra_k % 2 == 0
                              else 0.82 + random.random() * 0.10)
                        _v = 0.10 + random.random() * 0.55
                        lm_pos[_i] = _marc_quad(_u, _v)
                        _extra_k += 1
                    _oi = 0
                    _extra_k = 0
                    for _spec in objects or []:
                        _n = max(1, int(_spec.get("count", 1) or 1))
                        for _ in range(_n):
                            if (_oi < len(obj_pos)
                                    and _oi != _climber_i):
                                _u = (0.06 + 0.16 * ((_extra_k // 2) % 3)
                                      + random.random() * 0.08
                                      if _extra_k % 2 == 0
                                      else 0.70 + 0.16 * ((_extra_k // 2) % 3)
                                      + random.random() * 0.08)
                                _v = 0.06 + random.random() * 0.70
                                obj_pos[_oi] = _marc_quad(_u, _v)
                                _extra_k += 1
                            _oi += 1
                except Exception:
                    if (_lm_i is not None and _climber_i is not None
                            and _lm_i < len(lm_pos)
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
            elif (_lm_i is not None and _climber_i is not None
                    and _lm_i < len(lm_pos)
                    and _climber_i < len(obj_pos)):
                obj_pos[_climber_i] = list(lm_pos[_lm_i])
'''

NEW_POSITIONS = '''        positions = generate_cell_positions(area, n_lm + n_obj)
        lm_pos = positions[:n_lm]
        obj_pos = positions[n_lm:]
        if _marc_above_scene() or _marc_recipe_scene():
            import re as _re_pos
            _pl = ""
            _climber_i = None
            _obj_i = 0
            for _spec in objects or []:
                _n = max(1, int(_spec.get("count", 1) or 1))
                _opl = (_spec.get("placement") or {})
                if _climber_i is None and (
                        _opl.get("above") or _opl.get("recipe_id")):
                    _climber_i = _obj_i
                    _pl = str(_opl.get("on_landmark_type")
                              or _opl.get("beside_landmark_type") or "")
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
            if isinstance(area, dict):
                try:
                    _tl = list(area["top_left"])
                    _tr = list(area["top_right"])
                    _br = list(area["bottom_right"])
                    _bl = list(area["bottom_left"])
                    def _marc_quad(_u, _v):
                        _bot = [(1.0 - _u) * _bl[_k] + _u * _br[_k]
                                for _k in range(3)]
                        _top = [(1.0 - _u) * _tl[_k] + _u * _tr[_k]
                                for _k in range(3)]
                        return [(1.0 - _v) * _bot[_k] + _v * _top[_k]
                                for _k in range(3)]
                    # Tall climb targets: nearer-mid so the roof stays below
                    # the image top. Helper also frames the roof in camera.
                    _climb_u = 0.30 + random.random() * 0.40
                    if _pl == "vending_machine":
                        # Nearer fallback if camera framing misses. Helper
                        # then parks the roof on a mid-image downward ray.
                        _climb_v = 0.20 + random.random() * 0.12
                    elif _pl == "postbox":
                        _climb_v = 0.22 + random.random() * 0.16
                    else:
                        _climb_v = 0.16 + random.random() * 0.18
                    _climb = _marc_quad(_climb_u, _climb_v)
                    try:
                        from world import object_deployer as _od_near
                        _od_near._marc_above_near_xy = (
                            _bl[0] - _tl[0], _bl[1] - _tl[1])
                    except Exception:
                        pass
                    if _lm_i is not None and _lm_i < len(lm_pos):
                        lm_pos[_lm_i] = list(_climb)
                    if (_climber_i is not None
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
                    _near_types = set()
                    if _marc_recipe_scene():
                        try:
                            import json as _json_near
                            with open(
                                    "/tmp/marc_sar_recipe.json",
                                    encoding="utf-8") as _nf:
                                _recn = _json_near.load(_nf) or {}
                            for _nk in (
                                    _recn.get("surface_anchor"),
                                    _recn.get("tip_landmark"),
                            ):
                                if _nk:
                                    _near_types.add(str(_nk))
                            for _nk in _recn.get("extra_landmarks") or []:
                                if _nk:
                                    _near_types.add(str(_nk))
                        except Exception:
                            _near_types = set()
                    _near_types.discard(_pl)
                    _extra_k = 0
                    for _i in range(len(lm_pos)):
                        if _i == _lm_i:
                            continue
                        _t = ""
                        if _i < len(landmarks):
                            _t = _re_pos.sub(r"_[0-9]+$", "", str(
                                (landmarks[_i] or {}).get("name") or ""))
                        if _t in _near_types:
                            _du = (0.10 + random.random() * 0.08) * (
                                1.0 if _extra_k % 2 == 0 else -1.0)
                            _dv = 0.02 + random.random() * 0.08
                            lm_pos[_i] = _marc_quad(
                                min(0.82, max(0.18, _climb_u + _du)),
                                min(0.50, max(0.12, _climb_v + _dv)))
                        else:
                            _u = (0.08 + random.random() * 0.10
                                  if _extra_k % 2 == 0
                                  else 0.82 + random.random() * 0.10)
                            _v = 0.10 + random.random() * 0.55
                            lm_pos[_i] = _marc_quad(_u, _v)
                        _extra_k += 1
                    _oi = 0
                    _extra_k = 0
                    for _spec in objects or []:
                        _n = max(1, int(_spec.get("count", 1) or 1))
                        for _ in range(_n):
                            if (_oi < len(obj_pos)
                                    and _oi != _climber_i):
                                _u = (0.06 + 0.16 * ((_extra_k // 2) % 3)
                                      + random.random() * 0.08
                                      if _extra_k % 2 == 0
                                      else 0.70 + 0.16 * ((_extra_k // 2) % 3)
                                      + random.random() * 0.08)
                                _v = 0.06 + random.random() * 0.70
                                obj_pos[_oi] = _marc_quad(_u, _v)
                                _extra_k += 1
                            _oi += 1
                except Exception:
                    if (_lm_i is not None and _climber_i is not None
                            and _lm_i < len(lm_pos)
                            and _climber_i < len(obj_pos)):
                        obj_pos[_climber_i] = list(lm_pos[_lm_i])
            elif (_lm_i is not None and _climber_i is not None
                    and _lm_i < len(lm_pos)
                    and _climber_i < len(obj_pos)):
                obj_pos[_climber_i] = list(lm_pos[_lm_i])
'''

OLD_DEPLOY_LANDMARKS = '''        deploy_landmarks(mission_id, landmarks, area, configurations,
                         randomize_rotation=True,
                         positions=lm_pos, free_fall_types=_TIP_OVER_LANDMARKS)
'''

NEW_DEPLOY_LANDMARKS = '''        _free_fall = _TIP_OVER_LANDMARKS
        if _marc_above_scene():
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


def vending_glass_sign(local_pts, axis):
    """+1 if the detailed face (glass/doors) is toward local +axis, else -1.

    Independent of camera and of world yaw: the door panel has more mesh
    than the flat back, so vertex count on each end of the short local
    axis is a stable glass lock.
    """
    if not local_pts:
        return 1
    ax = 0 if int(axis) <= 0 else 1
    vals = [float(p[ax]) for p in local_pts]
    mid = 0.5 * (min(vals) + max(vals))
    pos = sum(1 for v in vals if v >= mid)
    neg = len(vals) - pos
    return 1 if pos >= neg else -1


def mesh_top_patch(points, thickness=0.04):
    """Centroid and XY bounds of the highest mesh vertices (the real top face)."""
    if not points:
        return None
    top_z = max(float(p[2]) for p in points)
    patch = [p for p in points if float(p[2]) >= top_z - float(thickness)]
    if not patch:
        patch = list(points)
    xs = [float(p[0]) for p in patch]
    ys = [float(p[1]) for p in patch]
    cx = sum(xs) / float(len(xs))
    cy = sum(ys) / float(len(ys))
    return (cx, cy, top_z), (min(xs), min(ys)), (max(xs), max(ys)), patch


def above_roof_contact_ok(foot_xyz, roof_lo, roof_hi, body_lo, body_hi, top_z,
                          max_gap=0.06):
    """True when support sits on the roof patch, not the landmark's side AABB."""
    fx, fy, fz = (float(foot_xyz[0]), float(foot_xyz[1]), float(foot_xyz[2]))
    x0, y0 = float(roof_lo[0]), float(roof_lo[1])
    x1, y1 = float(roof_hi[0]), float(roof_hi[1])
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    pad_x = max(0.02, 0.12 * (x1 - x0))
    pad_y = max(0.02, 0.12 * (y1 - y0))
    if not (x0 - pad_x <= fx <= x1 + pad_x and y0 - pad_y <= fy <= y1 + pad_y):
        return False
    z0, z1 = float(body_lo[2]), float(body_hi[2])
    height = z1 - z0
    if height < 0.12:
        return False
    if (fz - z0) < 0.50 * height:
        return False
    gap = fz - float(top_z)
    return -0.02 <= gap <= float(max_gap)


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
''' + ABOVE_SCENE_FN + inspect.getsource(visible_above_stack) + r'''
def _marc_place_above_people(mission_id, objects, configurations=None):
    """[marc-patch] _marc_above_aabb_stack: person on roof (_marc_above_freeze, _marc_above_always_on_top, _marc_above_joint_seat, _marc_above_ray_roof, _marc_above_sit_sink, _marc_above_sit_sink_70, _marc_above_sit_sink_short, _marc_above_vend_drape, _marc_above_vend_drape_lip, _marc_above_trash_raise, _marc_above_hydrant_crouch, _marc_above_trash_crouch, _marc_above_vend_farther, _marc_above_vend_farther2, _marc_above_vend_farther3, _marc_above_vend_farther4, _marc_above_vend_not_too_far, _marc_above_vend_closer, _marc_above_vend_sit_yaw, _marc_above_vend_glass_drape, _marc_above_vend_sit_fix, _marc_above_vend_sit_nudge, _marc_above_vend_sit_unraise, _marc_above_vend_lip_fwd, _marc_above_vend_lip_fwd2, _marc_above_vend_lip_fwd3, _marc_above_vend_lip_back, _marc_above_vend_prof, _marc_above_vend_prof2, _marc_above_vend_prof3, _marc_above_vend_prof4, _marc_above_vend_prof5, _marc_above_vend_prof6, _marc_above_vend_female, _marc_above_vend_female2, _marc_above_vend_female3, _marc_above_vend_female4, _marc_above_vend_female5, _marc_above_vend_female6, _marc_above_vend_female7, _marc_above_vend_male, _marc_above_vend_male2, _marc_above_vend_male3, _marc_above_vend_male4, _marc_above_vend_male5, _marc_above_vend_weight, _marc_above_vend_weight2, _marc_above_vend_weight3, _marc_above_vend_weight4, _marc_above_vend_weight5, _marc_above_no_closeup, _marc_above_mid_field, _marc_above_roof_in_view, _marc_above_vend_abs_frame, _marc_above_vend_leg_yaw, _marc_above_camera_frame, _marc_above_vend_mesh_seat, _marc_above_vend_mid_dist, _marc_above_vend_on_roof, _marc_above_vend_male6, _marc_above_vend_prof7, _marc_above_vend_female8, _marc_above_vend_weight6, _marc_above_vend_roof_visible, _marc_above_vend_joint_z, _marc_above_vend_prof8, _marc_above_vend_mesh_xy, _marc_above_vend_female_yaw, _marc_above_vend_prof9, _marc_above_vend_weight_yaw, _marc_above_vend_female_fwd, _marc_above_vend_stand, _marc_above_vend_prof10, _marc_above_vend_prof_yaw, _marc_above_vend_sit_hip, _marc_above_vend_weight7, _marc_above_vend_weight8, _marc_above_vend_mesh_contact, _marc_above_crouch_authored, _marc_above_vend_mesh_contact2, _marc_above_vend_mesh_nudge, _marc_above_vend_mesh_snap, _marc_above_lock_contact, _marc_above_joint_only, _marc_above_vend_prof11, _marc_above_vend_slot, _marc_above_vend_female9, _marc_above_vend_prof12, _marc_above_vend_male7, _marc_above_vend_weight9, _marc_above_headroom, _marc_above_vend_visual_seat, _marc_recipe_scene, _marc_above_trash_sit_rim)."""
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
        print(
            "[on-landmark-patch] above-mode ERROR: no climber jobs "
            "(placement.above missing) TRAINER_PEOPLE_ABOVE=%s"
            % __import__("os").environ.get("TRAINER_PEOPLE_ABOVE", "0")
        )
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

    def _camera_prim():
        mid = str(mission_id or "")
        nmid = _norm(mid)
        parts = mid.split("_")
        rig_tok = "_".join(parts[:2]) if len(parts) >= 2 else mid
        letter = parts[-1].upper() if len(parts) >= 3 else ""
        fallback = None
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Camera):
                continue
            try:
                override = prim.GetAttribute("isaac:nameOverride").Get()
            except Exception:
                override = None
            path = str(prim.GetPath())
            blob = _norm(path)
            if str(override or "") == mid or (nmid and nmid in blob):
                return prim
            if rig_tok and ("/%s/" % rig_tok) in path:
                if (not letter) or (("Camera_%s" % letter) in path) or path.endswith("/Camera"):
                    fallback = prim
        return fallback

    def _camera_xyz():
        """World position of the CCTV camera for this mission."""
        prim = _camera_prim()
        if prim is not None:
            try:
                t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()).ExtractTranslation()
                return [float(t[0]), float(t[1]), float(t[2])]
            except Exception:
                pass
        return None

    def _frame_landmark(lm, roof_z=None, typ="", pose=""):
        """Put the landmark ROOF on a camera ray so it stays in view."""
        rng = _world_range(lm)
        cam = _camera_prim()
        xyz = _world_xyz(lm)
        if rng is None or cam is None or xyz is None:
            return rng
        lo, hi = rng
        try:
            usd_cam = UsdGeom.Camera(cam)
            focal = float(usd_cam.GetFocalLengthAttr().Get() or 0.0)
            h_ap = float(usd_cam.GetHorizontalApertureAttr().Get() or 0.0)
            v_ap = float(usd_cam.GetVerticalApertureAttr().Get() or 0.0)
            cam_xf = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            cam_t = cam_xf.ExtractTranslation()
            if focal <= 0.0 or h_ap <= 0.0 or v_ap <= 0.0:
                return rng
            parity = sum(ord(ch) for ch in str(mission_id)) % 2
            u = 0.40 if parity == 0 else 0.60
            # _marc_above_headroom: roof at v=0.50 crops the sitter off
            # the top of these low CCTVs. Park the roof in the lower
            # midfield so the person on it stays in frame.
            _pose = str(pose or "").lower()
            if typ == "vending_machine":
                _v_key = "TRAINER_PEOPLE_VEND_FRAME_V"
                # _marc_sit_frame_head / _marc_above_vend_sit_v80: park
                # torso (roof+0.55 m) on the ray at v≈0.55 so the sitter
                # occupies the upper third, not an empty roof close-up.
                _v_def = ("0.74" if (
                    "reaching" in _pose or "walking" in _pose
                ) else "0.55")
            elif typ == "postbox":
                _v_key = "TRAINER_PEOPLE_POST_FRAME_V"
                _v_def = "0.62"
            else:
                _v_key = "TRAINER_PEOPLE_CLIMB_FRAME_V"
                _v_def = "0.58"
            try:
                v = float(__import__("os").environ.get(
                    _v_key, _v_def) or _v_def)
            except (TypeError, ValueError):
                v = float(_v_def)
            ray_local = Gf.Vec3d(
                (2.0 * u - 1.0) * 0.5 * h_ap,
                (1.0 - 2.0 * v) * 0.5 * v_ap,
                -focal,
            )
            ray = cam_xf.TransformDir(ray_local)
            if abs(float(ray[2])) < 1e-6:
                return rng
            plane_z = float(roof_z) if roof_z is not None else float(hi[2])
            if typ == "vending_machine" and "sitting" in _pose:
                try:
                    _torso = float(__import__("os").environ.get(
                        "TRAINER_PEOPLE_VEND_SIT_TORSO", "0.55") or "0.55")
                except (TypeError, ValueError):
                    _torso = 0.55
                plane_z = float(plane_z) + _torso
            t = (plane_z - float(cam_t[2])) / float(ray[2])
            if t <= 0.0:
                return rng
            want_x = float(cam_t[0]) + t * float(ray[0])
            want_y = float(cam_t[1]) + t * float(ray[1])
            center_x = 0.5 * (lo[0] + hi[0])
            center_y = 0.5 * (lo[1] + hi[1])
            _set_xyz(lm, [
                xyz[0] + (want_x - center_x),
                xyz[1] + (want_y - center_y),
                xyz[2],
            ])
            return _world_range(lm)
        except Exception:
            return rng

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

    def _marc_freeze_person(prim):
        """Kinematic hold. Do not convex-hull the posed mesh: that hull is
        mostly empty space and PhysX bounces climbers off the cabinet.
        _marc_above_lock_contact
        """
        try:
            from pxr import UsdPhysics, PhysxSchema, Gf
            for p in Usd.PrimRange(prim):
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        rb = UsdPhysics.RigidBodyAPI(p)
                        rb.CreateKinematicEnabledAttr(True)
                        rb.CreateRigidBodyEnabledAttr(True)
                        try:
                            rb.CreateVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
                            rb.CreateAngularVelocityAttr().Set(
                                Gf.Vec3f(0, 0, 0))
                        except Exception:
                            pass
                    except Exception:
                        pass
                if p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    try:
                        PhysxSchema.PhysxRigidBodyAPI(p).CreateDisableGravityAttr(True)
                    except Exception:
                        pass
                    try:
                        p.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
                    except Exception:
                        pass
        except Exception:
            pass

    def _set_xyz(prim, xyz):
        path = str(prim.GetPath())
        try:
            from pxr import UsdPhysics, PhysxSchema, Usd
            for p in Usd.PrimRange(prim):
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    try:
                        UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(True)
                    except Exception:
                        pass
                    try:
                        UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(True)
                    except Exception:
                        pass
                if p.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    try:
                        PhysxSchema.PhysxRigidBodyAPI(p).CreateDisableGravityAttr(True)
                    except Exception:
                        pass
        except Exception:
            pass
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
            if "/people/" in path or path.startswith("/world/people"):
                origin_z = hz - 0.01
                continue
            if path.startswith(lm_path) or (lm_path in path):
                return hz
            origin_z = hz - 0.01
        return None

    def _mesh_world_points(root):
        pts = []
        xfc = UsdGeom.XformCache()
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            local = mesh.GetPointsAttr().Get()
            if not local:
                continue
            xf = xfc.GetLocalToWorldTransform(prim)
            for p in local:
                wp = xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                pts.append((float(wp[0]), float(wp[1]), float(wp[2])))
        return pts

    def _top_candidates(lm):
        # _marc_above_ray_roof: roof Z is a PhysX hit on THIS landmark.
        # Mesh-vertex max Z launched sitting people into the sky; AABB /
        # 2D overlap is not a placement oracle (perspective mismatch).
        rng = _world_range(lm)
        xyz = _world_xyz(lm)
        if rng is None and xyz is None:
            return None
        if rng is None:
            body_lo = [xyz[0] - 0.4, xyz[1] - 0.4, xyz[2]]
            body_hi = [xyz[0] + 0.4, xyz[1] + 0.4, xyz[2] + 2.0]
        else:
            body_lo, body_hi = rng
        cx = 0.5 * (body_lo[0] + body_hi[0])
        cy = 0.5 * (body_lo[1] + body_hi[1])
        start_z = float(body_hi[2]) + 1.2
        hits = []
        for u in (0.30, 0.40, 0.50, 0.60, 0.70):
            for v in (0.30, 0.40, 0.50, 0.60, 0.70):
                x = body_lo[0] + u * (body_hi[0] - body_lo[0])
                y = body_lo[1] + v * (body_hi[1] - body_lo[1])
                hz = _ray_landmark_z(str(lm.GetPath()), x, y, start_z)
                if hz is not None:
                    hits.append((x, y, hz))
        if hits:
            top_z = max(h[2] for h in hits)
            patch = [h for h in hits if h[2] >= top_z - 0.08]
            cx = sum(h[0] for h in patch) / float(len(patch))
            cy = sum(h[1] for h in patch) / float(len(patch))
            roof_lo = (min(h[0] for h in patch), min(h[1] for h in patch))
            roof_hi = (max(h[0] for h in patch), max(h[1] for h in patch))
            return [(cx, cy, top_z)], roof_lo, roof_hi, body_lo, body_hi
        mid = (cx, cy, float(body_hi[2]))
        roof_lo = (body_lo[0], body_lo[1])
        roof_hi = (body_hi[0], body_hi[1])
        return [mid], roof_lo, roof_hi, body_lo, body_hi

    def _mesh_seat_z(person, bx, by, joint_z):
        # _marc_above_vend_mesh_seat: visual butt, not joint cluster.
        # Male mesh hangs below the pelvis joints; professor sits high.
        pts = _mesh_world_points(person)
        if not pts:
            return None
        rad2 = 0.18 * 0.18
        near = [p[2] for p in pts
                if (p[0] - bx) ** 2 + (p[1] - by) ** 2 <= rad2]
        if len(near) < 12:
            rad2 = 0.28 * 0.28
            near = [p[2] for p in pts
                    if (p[0] - bx) ** 2 + (p[1] - by) ** 2 <= rad2]
        if not near:
            return None
        lo = float(joint_z) - 0.18
        hi = float(joint_z) + 0.50
        band = [z for z in near if lo <= z <= hi]
        if not band:
            return None
        return min(band)

    def _mesh_hip_xy(person, bx, by, bz):
        # _marc_above_vend_mesh_xy: visual hips, not joint-butt. Professor
        # and female clips park the mesh a metre beside the skeleton.
        pts = _mesh_world_points(person)
        if not pts:
            return None
        hip = []
        rad2 = 0.32 * 0.32
        lo = float(bz) - 0.12
        hi = float(bz) + 0.50
        for p in pts:
            if (p[0] - bx) ** 2 + (p[1] - by) ** 2 > rad2:
                continue
            if lo <= p[2] <= hi:
                hip.append(p)
        if len(hip) < 16:
            # _marc_above_vend_prof10 / _marc_above_vend_sit_hip:
            # professor visual is ~1m from the joints. Widen XY at the
            # same hip Z. Do not average the whole sitting mesh:
            # student_weight hanging legs pull the centroid off the roof.
            rad2 = 1.20 * 1.20
            hip = []
            for p in pts:
                if (p[0] - bx) ** 2 + (p[1] - by) ** 2 > rad2:
                    continue
                if lo <= p[2] <= hi:
                    hip.append(p)
        if not hip:
            return None
        n = float(len(hip))
        return (sum(p[0] for p in hip) / n, sum(p[1] for p in hip) / n)

    def _align_visual_on_roof(person, tx, ty, top_z):
        # _marc_above_vend_visual_seat: joint seat leaves professor/male
        # meshes hovering ~1m beside the cabinet. Slide the prim so the
        # visual hips meet the roof. Never raise (that sky-launched).
        pxyz = _world_xyz(person)
        joints = _posed_joints(person)
        if pxyz is None or not joints:
            return None
        support = _butt_from_joints(joints) or _support_from_joints(joints)
        if support is None:
            return None
        hip = _mesh_hip_xy(
            person, float(support[0]), float(support[1]), float(support[2]))
        vis_z = _mesh_seat_z(
            person,
            float(hip[0]) if hip else float(support[0]),
            float(hip[1]) if hip else float(support[1]),
            float(support[2]))
        dx = dy = 0.0
        if hip is not None:
            dx = float(tx) - float(hip[0])
            dy = float(ty) - float(hip[1])
            mag = (dx * dx + dy * dy) ** 0.5
            if mag > 1.4:
                s = 1.4 / mag
                dx, dy = dx * s, dy * s
        dz = 0.0
        if vis_z is not None:
            dz = float(top_z) - float(vis_z)
            if dz > 0.04:
                dz = 0.04
            if dz < -0.85:
                dz = -0.85
        if abs(dx) < 1e-4 and abs(dy) < 1e-4 and abs(dz) < 1e-4:
            return [
                float(hip[0]) if hip else float(support[0]),
                float(hip[1]) if hip else float(support[1]),
                float(vis_z) if vis_z is not None else float(support[2]),
            ]
        _set_xyz(person, [
            float(pxyz[0]) + dx,
            float(pxyz[1]) + dy,
            float(pxyz[2]) + dz,
        ])
        _marc_freeze_person(person)
        hip2 = _mesh_hip_xy(person, float(tx), float(ty), float(top_z))
        vis2 = _mesh_seat_z(
            person,
            float(hip2[0]) if hip2 else float(tx),
            float(hip2[1]) if hip2 else float(ty),
            float(top_z))
        if hip2 is not None:
            return [
                float(hip2[0]), float(hip2[1]),
                float(vis2) if vis2 is not None else float(top_z),
            ]
        joints = _posed_joints(person)
        return _butt_from_joints(joints) or support

    def _align_visual_on_trash_lid(person, tx, ty, top_z, character=None):
        # _marc_trash_butt_z / _marc_above_trash_weight_lid: pelvis on
        # the lid, not mesh min in the hole. Hanging calves used to
        # pull vis_z into the cavity.
        del character
        pxyz = _world_xyz(person)
        joints = _posed_joints(person)
        if pxyz is None:
            return None
        butt = _butt_from_joints(joints) if joints else None
        if butt is None:
            return None
        vis_z = float(butt[2])
        dz = float(top_z) - vis_z
        if dz > 0.03:
            dz = 0.03
        try:
            drop_cap = -abs(float(__import__("os").environ.get(
                "TRAINER_PEOPLE_TRASH_DROP", "0.25") or "0.25"))
        except (TypeError, ValueError):
            drop_cap = -0.25
        if dz < drop_cap:
            dz = drop_cap
        dx = float(tx) - float(butt[0])
        dy = float(ty) - float(butt[1])
        mag = (dx * dx + dy * dy) ** 0.5
        if mag > 0.45:
            s = 0.45 / mag
            dx, dy = dx * s, dy * s
        if abs(dx) < 1e-4 and abs(dy) < 1e-4 and abs(dz) < 1e-4:
            if vis_z < float(top_z) - 0.12:
                return None
            return [float(butt[0]), float(butt[1]), vis_z]
        _set_xyz(person, [
            float(pxyz[0]) + dx,
            float(pxyz[1]) + dy,
            float(pxyz[2]) + dz,
        ])
        _marc_freeze_person(person)
        joints = _posed_joints(person)
        butt2 = _butt_from_joints(joints) if joints else None
        if butt2 is None:
            return None
        if float(butt2[2]) < float(top_z) - 0.12:
            return None
        return [float(butt2[0]), float(butt2[1]), float(butt2[2])]

    def _vend_mesh_contact_on():
        # TRAINER_PEOPLE_VEND_MESH_CONTACT is ignored.
        # _marc_above_joint_only: mesh-vert snap launched people into the
        # sky. Joint seat + per-character yaw/sink/lip only.
        return False

    def _mesh_on_roof(person, cx, cy, roof_lo, roof_hi):
        pts = _mesh_world_points(person)
        if not pts:
            return [], []
        pad = 0.06
        x0 = min(float(roof_lo[0]), float(roof_hi[0])) - pad
        x1 = max(float(roof_lo[0]), float(roof_hi[0])) + pad
        y0 = min(float(roof_lo[1]), float(roof_hi[1])) - pad
        y1 = max(float(roof_lo[1]), float(roof_hi[1])) + pad
        if (x1 - x0) < 0.25:
            x0, x1 = float(cx) - 0.32, float(cx) + 0.32
        if (y1 - y0) < 0.25:
            y0, y1 = float(cy) - 0.32, float(cy) + 0.32
        over = [p for p in pts
                if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
        if len(over) < 8:
            rad2 = 0.38 * 0.38
            over = [p for p in pts
                    if (p[0] - float(cx)) ** 2 + (p[1] - float(cy)) ** 2 <= rad2]
        return over, pts

    def _mesh_nudge_xy(verts):
        # _marc_above_vend_mesh_nudge: mid-height body over the roof.
        # A dangling foot must not pull XY.
        if not verts:
            return None
        zs = [p[2] for p in verts]
        zlo, zhi = min(zs), max(zs)
        span = max(zhi - zlo, 0.01)
        lo_b = zlo + 0.28 * span
        hi_b = zlo + 0.62 * span
        band = [p for p in verts if lo_b <= p[2] <= hi_b] or list(verts)
        n = float(len(band))
        return (sum(p[0] for p in band) / n, sum(p[1] for p in band) / n)

    def _mesh_snap_z(verts, all_pts, mode="seat"):
        # _marc_above_vend_mesh_snap: drop until hips/butt meet the roof.
        # Lowest-on-roof is often a grazing foot; that parked people
        # "seated" on one toe.
        if mode == "feet":
            src = verts or all_pts
            if not src:
                return None
            zmin = min(p[2] for p in src)
            cluster = [p for p in src if p[2] <= zmin + 0.05]
            return sum(p[2] for p in cluster) / float(len(cluster))
        src = verts or []
        span = 0.0
        if src:
            zs = [p[2] for p in src]
            span = max(zs) - min(zs)
        if (not src) or span < 0.35:
            # Only a foot/hand is on the roof footprint. Use the visual
            # hip band of the whole posed mesh so we keep lowering.
            src = all_pts or src
            if not src:
                return None
            zs = [p[2] for p in src]
            zlo, zhi = min(zs), max(zs)
            sp = max(zhi - zlo, 0.01)
            hip = [p for p in src
                   if (zlo + 0.28 * sp) <= p[2] <= (zlo + 0.55 * sp)]
            src = hip or src
        zs = [p[2] for p in src]
        zlo, zhi = min(zs), max(zs)
        cut = zlo + 0.35 * max(zhi - zlo, 0.01)
        body = [p for p in src if p[2] >= cut]
        if len(body) < 8:
            body = list(src)
        bz = min(p[2] for p in body)
        layer = [p for p in body if p[2] <= bz + 0.08]
        return sum(p[2] for p in layer) / float(len(layer))

    def _mesh_contact(person, cx, cy, top_z, roof_lo, roof_hi, mode="seat"):
        # _marc_above_vend_mesh_contact / _marc_above_vend_mesh_contact2:
        # posed mesh vs roof, not bind AABB. Body seat, not first foot.
        verts, all_pts = _mesh_on_roof(person, cx, cy, roof_lo, roof_hi)
        nxy = _mesh_nudge_xy(verts) if verts else None
        mz = _mesh_snap_z(verts, all_pts, mode)
        if mz is None:
            return None
        if nxy is None:
            nxy = (float(cx), float(cy))
        return [float(nxy[0]), float(nxy[1]), float(mz)]

    def _seat_at(person, cx, cy, top_z, clearance, xy_from="support",
                 roof_lo=None, roof_hi=None, mesh_z=False, mesh_snap="seat"):
        person_xyz = _world_xyz(person)
        if person_xyz is None:
            return None
        joints = _posed_joints(person)
        support = _support_from_joints(joints)
        if support is None:
            return None
        ax, ay = float(support[0]), float(support[1])
        if xy_from == "butt" and joints:
            support = _butt_from_joints(joints) or support
            ax, ay = float(support[0]), float(support[1])
        elif xy_from == "centroid" and joints:
            ax = sum(p[0] for p in joints) / float(len(joints))
            ay = sum(p[1] for p in joints) / float(len(joints))
        _set_xyz(person, [
            person_xyz[0] + (cx - ax),
            person_xyz[1] + (cy - ay),
            person_xyz[2],
        ])
        person_xyz = _world_xyz(person)
        joints = _posed_joints(person)
        support = _support_from_joints(joints)
        if xy_from == "butt" and joints:
            support = _butt_from_joints(joints) or support
        if person_xyz is None or support is None:
            return None
        # _marc_above_joint_only / _marc_above_vend_joint_z: posed joints
        # only. Mesh min-Z, hip XY, and roof-vert contact all launched
        # people into the sky or off the cabinet. Per-character sit
        # extras are yaw / lip / sink, not vertex samples.
        delta = (float(top_z) + float(clearance)) - float(support[2])
        # Spawn already puts the root on the roof. A bogus roof (mesh max Z)
        # used to add metres and park sitting people in the sky. Bring them
        # down always; never climb more than a cabinet is tall.
        if delta > 3.0:
            print(
                "[on-landmark-patch] above-mode: skip insane upward "
                "dz=%.2f roof=%.3f support=%.3f"
                % (delta, float(top_z), float(support[2]))
            )
        else:
            _set_xyz(person, [
                person_xyz[0], person_xyz[1],
                person_xyz[2] + delta,
            ])
        joints = _posed_joints(person)
        support = (
            _butt_from_joints(joints) if xy_from == "butt"
            else _support_from_joints(joints))
        return support

    def _butt_from_joints(pts):
        # Sitting clips park the feet on the roof or hang them; either way
        # the lowest joints are not the seat. Skip the hanging/foot layer
        # and sit on the remaining lowest cluster (pelvis / thighs).
        if not pts:
            return None
        min_z = min(p[2] for p in pts)
        max_z = max(p[2] for p in pts)
        skip = 0.40 * max(max_z - min_z, 0.01)
        body = [p for p in pts if p[2] >= min_z + skip]
        if len(body) < 3:
            body = list(pts)
        bz = min(p[2] for p in body)
        layer = [p for p in body if p[2] <= bz + 0.10]
        if not layer:
            layer = body
        return [
            sum(p[0] for p in layer) / float(len(layer)),
            sum(p[1] for p in layer) / float(len(layer)),
            bz,
        ]

    def _yaw_person(prim, deg):
        """Compose world-Z yaw onto pose-root pitch/roll (_marc_yaw_keep_root).

        people_poser uses q = R_z(yaw) * R_pose_root. Replacing with a
        pure R_z stands lying / crouch / prone people up.
        """
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        orient = ops.get("xformOp:orient")
        if orient is None:
            orient = xf.AddOrientOp()
        _math = __import__("math")
        q_new = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(deg)).GetQuat()
        try:
            m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            xdir = m.TransformDir(Gf.Vec3d(1, 0, 0))
            ydir = m.TransformDir(Gf.Vec3d(0, 1, 0))
            xn = float(xdir[0]) ** 2 + float(xdir[1]) ** 2
            yn = float(ydir[0]) ** 2 + float(ydir[1]) ** 2
            if xn >= yn:
                cur = _math.degrees(_math.atan2(float(xdir[1]), float(xdir[0])))
            else:
                cur = _math.degrees(_math.atan2(float(ydir[1]), float(ydir[0])))
            delta = (float(deg) - cur + 180.0) % 360.0 - 180.0
            q_old = orient.Get()
            if q_old is None:
                q_old = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            q_delta = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(delta)).GetQuat()
            try:
                q_new = Gf.Quatf(q_delta) * Gf.Quatf(q_old)
            except Exception:
                q_new = Gf.Quatd(q_delta) * Gf.Quatd(q_old)
        except Exception:
            pass
        try:
            orient.Set(Gf.Quatf(q_new))
        except Exception:
            orient.Set(q_new)
        ordered = []
        if "xformOp:translate" in ops:
            ordered.append(ops["xformOp:translate"])
        ordered.append(orient)
        if "xformOp:scale" in ops:
            ordered.append(ops["xformOp:scale"])
        if ordered:
            xf.SetXformOpOrder(ordered)

    def _feet_from_joints(pts):
        if not pts:
            return None
        zs = [p[2] for p in pts]
        span = max(max(zs) - min(zs), 0.01)
        cut = min(zs) + 0.28 * span
        feet = [p for p in pts if p[2] <= cut]
        if len(feet) < 2:
            feet = list(pts)
        n = float(len(feet))
        return [sum(p[0] for p in feet) / n, sum(p[1] for p in feet) / n]

    def _yaw_legs_along(person, fx, fy, extra=0.0):
        # Sitting clips do not share a bind-forward. Rotate so the posed
        # feet hang toward glass, not toward a camera or a +90 guess.
        _math = __import__("math")
        want = _math.atan2(float(fy), float(fx))
        best, best_dot = None, -1e9
        for off in (0.0, 90.0, -90.0, 180.0):
            deg = _math.degrees(want) + off
            _yaw_person(person, deg)
            joints = _posed_joints(person)
            butt = _butt_from_joints(joints)
            feet = _feet_from_joints(joints)
            if not butt or not feet:
                continue
            dx = float(feet[0]) - float(butt[0])
            dy = float(feet[1]) - float(butt[1])
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-4:
                continue
            d = (dx / n) * float(fx) + (dy / n) * float(fy)
            if d > best_dot:
                best_dot, best = d, deg
        if best is not None:
            _yaw_person(person, best + float(extra))

    def _vending_local_frame(lm, body_lo, body_hi):
        xf = UsdGeom.Xformable(lm)
        if not xf:
            return None
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xdir = m.TransformDir(Gf.Vec3d(1, 0, 0))
        ydir = m.TransformDir(Gf.Vec3d(0, 1, 0))
        ldx, ldy = 1.0, 1.0
        lo = hi = None
        try:
            bound = UsdGeom.Imageable(lm).ComputeLocalBound(
                Usd.TimeCode.Default(), "default")
            rng = bound.GetRange()
            lo, hi = rng.GetMin(), rng.GetMax()
            ldx = abs(float(hi[0]) - float(lo[0]))
            ldy = abs(float(hi[1]) - float(lo[1]))
        except Exception:
            lo = hi = None
        if lo is None and body_lo is not None and body_hi is not None:
            try:
                inv = m.GetInverse()
                xs, ys = [], []
                for x in (body_lo[0], body_hi[0]):
                    for y in (body_lo[1], body_hi[1]):
                        for z in (body_lo[2], body_hi[2]):
                            lp = inv.Transform(Gf.Vec3d(
                                float(x), float(y), float(z)))
                            xs.append(lp[0])
                            ys.append(lp[1])
                ldx = max(xs) - min(xs)
                ldy = max(ys) - min(ys)
                lo = [min(xs), min(ys), 0.0]
                hi = [max(xs), max(ys), 0.0]
            except Exception:
                ldx = abs(float(body_hi[0]) - float(body_lo[0]))
                ldy = abs(float(body_hi[1]) - float(body_lo[1]))
        return m, xdir, ydir, ldx, ldy, lo, hi

    def _vending_roof_xy(lm, body_lo, body_hi):
        """Cabinet-local roof centre in world XY (not a world-AABB patch)."""
        fr = _vending_local_frame(lm, body_lo, body_hi)
        if fr is None:
            return None
        m, _xdir, _ydir, _ldx, _ldy, lo, hi = fr
        if lo is None or hi is None:
            return None
        mid = m.Transform(Gf.Vec3d(
            0.5 * (float(lo[0]) + float(hi[0])),
            0.5 * (float(lo[1]) + float(hi[1])),
            0.5 * (float(lo[2]) + float(hi[2])),
        ))
        return float(mid[0]), float(mid[1])

    def _vend_slot_xy(ox, oy, fx, fy, half_front, half_side, slot, margin=0.10):
        # _marc_above_vend_slot: glass-lip center or left/right corner.
        # Depth stays on the roof (never past the cabinet face).
        depth = max(0.06, float(half_front) - float(margin))
        side = max(0.0, float(half_side) - float(margin))
        lat = 0.0
        sl = str(slot or "center").lower()
        if sl == "left":
            lat = side
        elif sl == "right":
            lat = -side
        lx, ly = -float(fy), float(fx)
        return (
            float(ox) + depth * float(fx) + lat * lx,
            float(oy) + depth * float(fy) + lat * ly,
        )

    def _vending_roof_clamp(lm, body_lo, body_hi, wx, wy, margin=0.10):
        # Pull world XY onto the cabinet-local roof, inset so the butt
        # cannot sit past a face. Female/professor clips hang off if we
        # only clamp after a glass-forward that is already past the lip.
        fr = _vending_local_frame(lm, body_lo, body_hi) if lm is not None else None
        if fr is None:
            return float(wx), float(wy)
        m, _xd, _yd, _ldx, _ldy, lo, hi = fr
        if lo is None or hi is None:
            return float(wx), float(wy)
        try:
            inv = m.GetInverse()
            zmid = 0.5 * (float(lo[2]) + float(hi[2]))
            lp = inv.Transform(Gf.Vec3d(float(wx), float(wy), zmid))
            dx = abs(float(hi[0]) - float(lo[0]))
            dy = abs(float(hi[1]) - float(lo[1]))
            mx = max(0.02, min(float(margin), max(0.02, 0.5 * dx - 0.02)))
            my = max(0.02, min(float(margin), max(0.02, 0.5 * dy - 0.02)))
            x = min(max(float(lp[0]), float(lo[0]) + mx), float(hi[0]) - mx)
            y = min(max(float(lp[1]), float(lo[1]) + my), float(hi[1]) - my)
            wp = m.Transform(Gf.Vec3d(x, y, float(lp[2])))
            return float(wp[0]), float(wp[1])
        except Exception:
            return float(wx), float(wy)

    def _on_vending_roof(lm, body_lo, body_hi, wx, wy, margin=0.06):
        nx, ny = _vending_roof_clamp(
            lm, body_lo, body_hi, wx, wy, margin)
        return ((float(nx) - float(wx)) ** 2
                + (float(ny) - float(wy)) ** 2) <= 0.03 * 0.03

    def _vending_front_xy(lm, body_lo, body_hi):
        """Glass-door outward WORLD XY, locked to the cabinet mesh.

        Not the camera-facing panel, not a world AABB, not a random +X
        sign that flips with landmark yaw.
        """
        def nrm2(dx, dy):
            n = (float(dx) * float(dx) + float(dy) * float(dy)) ** 0.5
            if n < 1e-4:
                return None
            return (float(dx) / n, float(dy) / n)

        origin = _world_xyz(lm)
        for p in Usd.PrimRange(lm):
            blob = _norm(str(p.GetName() or ""))
            if not any(t in blob for t in (
                    "glass", "door", "display", "window", "front")):
                continue
            c = _world_xyz(p)
            if origin is None or c is None:
                continue
            hit = nrm2(c[0] - origin[0], c[1] - origin[1])
            if hit is not None:
                pick = hit
                break
        else:
            pick = None
        fr = _vending_local_frame(lm, body_lo, body_hi)
        if pick is None and fr is not None:
            m, xdir, ydir, ldx, ldy, lo, hi = fr
            local_pts = []
            try:
                inv = m.GetInverse()
                for wp in _mesh_world_points(lm) or []:
                    lp = inv.Transform(Gf.Vec3d(
                        float(wp[0]), float(wp[1]), float(wp[2])))
                    local_pts.append(
                        (float(lp[0]), float(lp[1]), float(lp[2])))
            except Exception:
                local_pts = []
            axis = 0 if ldx <= ldy else 1
            base = nrm2(xdir[0], xdir[1]) if axis == 0 else nrm2(
                ydir[0], ydir[1])
            if base is None:
                base = nrm2(xdir[0], xdir[1]) or nrm2(
                    ydir[0], ydir[1]) or (1.0, 0.0)
            if not local_pts:
                sign = 1
            else:
                vals = [p[axis] for p in local_pts]
                midv = 0.5 * (min(vals) + max(vals))
                pos = sum(1 for v in vals if v >= midv)
                sign = 1 if pos >= (len(vals) - pos) else -1
            pick = (base[0] * sign, base[1] * sign)
        if pick is None:
            pick = (1.0, 0.0)
        try:
            flip = str(__import__("os").environ.get(
                "TRAINER_PEOPLE_VEND_FRONT_FLIP", "") or "").strip().lower()
        except Exception:
            flip = ""
        if flip in ("1", "true", "yes", "on"):
            pick = (-pick[0], -pick[1])
        return pick

    def _drape_xy(cx, cy, typ, pose, body_lo, body_hi, lm=None, ch="",
                 slot="lip"):
        # Sitting on a vending: hips on the glass-side lip in *cabinet*
        # local space so every CCTV sees the same physical pose.
        if typ == "trash_can" and "sitting" in str(pose or ""):
            # _marc_trash_sit_rim_xy: hips on the near rim, not the
            # hollow opening. Legs already yaw toward the CCTV.
            cam = _camera_xyz()
            hx = hy = 0.35
            if body_lo is not None and body_hi is not None:
                hx = 0.5 * abs(float(body_hi[0]) - float(body_lo[0]))
                hy = 0.5 * abs(float(body_hi[1]) - float(body_lo[1]))
            half = min(hx, hy) if min(hx, hy) > 0.05 else max(hx, hy)
            half = max(0.12, float(half))
            try:
                rim = float(__import__("os").environ.get(
                    "TRAINER_PEOPLE_TRASH_RIM", "0.72") or "0.72")
            except (TypeError, ValueError):
                rim = 0.72
            if cam is None:
                return float(cx) + rim * half, float(cy)
            dx = float(cam[0]) - float(cx)
            dy = float(cam[1]) - float(cy)
            n = (dx * dx + dy * dy) ** 0.5
            if n < 1e-4:
                return float(cx), float(cy)
            return (
                float(cx) + rim * half * dx / n,
                float(cy) + rim * half * dy / n,
            )
        if typ != "vending_machine" or "sitting" not in str(pose or ""):
            return float(cx), float(cy)
        fx, fy = _vending_front_xy(lm, body_lo, body_hi) if lm is not None else (1.0, 0.0)
        try:
            lip = float(__import__("os").environ.get(
                "TRAINER_PEOPLE_VEND_LIP", "0.50") or "0.50")
        except (TypeError, ValueError):
            lip = 0.50
        if "professor" in str(ch or "").lower():
            try:
                lip += float(__import__("os").environ.get(
                    "TRAINER_PEOPLE_VEND_PROF_LIP", "0.0727") or "0.0727")
            except (TypeError, ValueError):
                lip += 0.0727
        elif "female" in str(ch or "").lower():
            try:
                lip += float(__import__("os").environ.get(
                    "TRAINER_PEOPLE_VEND_FEMALE_LIP", "0.0727") or "0.0727")
            except (TypeError, ValueError):
                lip += 0.0727
        elif "weight" in str(ch or "").lower():
            try:
                lip += float(__import__("os").environ.get(
                    "TRAINER_PEOPLE_VEND_WEIGHT_LIP", "0.0727") or "0.0727")
            except (TypeError, ValueError):
                lip += 0.0727
        elif "male_student" in str(ch or "").lower():
            try:
                lip += float(__import__("os").environ.get(
                    "TRAINER_PEOPLE_VEND_MALE_LIP", "0.0727") or "0.0727")
            except (TypeError, ValueError):
                lip += 0.0727
        origin = _vending_roof_xy(lm, body_lo, body_hi) if lm is not None else None
        ox, oy = origin if origin is not None else (float(cx), float(cy))
        # _marc_vend_cam_lip: glass can be the far face. Camera wins
        # so the sit is on the CCTV-facing roof, not behind the cabinet.
        cam = _camera_xyz()
        if cam is not None:
            cdx = float(cam[0]) - float(ox)
            cdy = float(cam[1]) - float(oy)
            cn = (cdx * cdx + cdy * cdy) ** 0.5
            if cn > 1e-4:
                cdx, cdy = cdx / cn, cdy / cn
                if cdx * float(fx) + cdy * float(fy) < 0.15:
                    fx, fy = cdx, cdy
        # _marc_above_vend_on_roof: keep the butt on the cabinet. Female
        # / professor used to get half+0.04 past the face and sit in air.
        half_front = 0.35
        half_side = 0.35
        if lm is not None:
            fr = _vending_local_frame(lm, body_lo, body_hi)
            if fr is not None:
                _m, _xd, _yd, ldx, ldy, _lo, _hi = fr
                half_front = 0.5 * min(float(ldx), float(ldy))
                half_side = 0.5 * max(float(ldx), float(ldy))
        _ch = str(ch or "").lower()
        margin = 0.12 if (
            "female" in _ch or "professor" in _ch or "male_student" in _ch
            or "weight" in _ch
        ) else 0.08
        sl = str(slot or "lip").lower()
        if sl in ("center", "left", "right"):
            # _marc_above_vend_female9 / _marc_above_vend_prof12: glass
            # lip, center or a corner, then clamp onto the roof.
            wx, wy = _vend_slot_xy(
                ox, oy, fx, fy, half_front, half_side, sl, margin)
            return _vending_roof_clamp(
                lm, body_lo, body_hi, wx, wy, margin)
        lip = min(float(lip), max(0.06, half_front - margin))
        if ("female" in _ch or "professor" in _ch or "weight" in _ch
                or "male_student" in _ch):
            # Female sit is the good close-up: hips on the glass lip.
            # student_weight without this extra sits on roof centre and
            # drops out of tight CCTV crops. _marc_above_vend_weight7
            # _marc_above_vend_male7: male sitting clip hung ~1.2 m off
            # the cabinet without the same glass-forward drape.
            if "female" in _ch:
                _fwd_key = "TRAINER_PEOPLE_VEND_FEMALE_FWD"
                _fwd_def = "0.16"
            elif "professor" in _ch:
                _fwd_key = "TRAINER_PEOPLE_VEND_PROF_FWD"
                _fwd_def = "0.16"
            elif "weight" in _ch:
                # _marc_above_vend_weight8: +7 cm glass-forward vs female.
                _fwd_key = "TRAINER_PEOPLE_VEND_WEIGHT_FWD"
                _fwd_def = "0.23"
            else:
                _fwd_key = "TRAINER_PEOPLE_VEND_MALE_FWD"
                _fwd_def = "0.16"
            try:
                fwd = float(__import__("os").environ.get(
                    _fwd_key, _fwd_def) or _fwd_def)
            except (TypeError, ValueError):
                fwd = float(_fwd_def)
            lip = min(half_front - margin, float(lip) + fwd)
        wx = float(ox) + lip * fx
        wy = float(oy) + lip * fy
        return _vending_roof_clamp(lm, body_lo, body_hi, wx, wy, margin)

    def _contact_ok(foot, cx, cy, top_z, min_gap=-0.05, max_xy=0.55):
        # Never fake Z: posed support vs raycast roof only.
        fx, fy, fz = float(foot[0]), float(foot[1]), float(foot[2])
        dist_xy = ((fx - float(cx)) ** 2 + (fy - float(cy)) ** 2) ** 0.5
        if dist_xy > float(max_xy):
            return False
        gap = fz - float(top_z)
        return float(min_gap) <= gap <= 0.15

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
            print(
                "[on-landmark-patch] above-mode: missing prims type=%s "
                "landmarks=%d people=%d spec=%s/%s"
                % (typ, len(lms), len(people),
                   str(spec.get("type") or ""),
                   str(spec.get("character") or ""))
            )
            continue
        lm, person = lms[0], people[0]
        _lock = dict(globals().get("_marc_above_lock") or {})
        _lk = str(person.GetPath())
        _hold = _lock.get(_lk)
        if _hold and _hold.get("person") and _hold.get("lm"):
            # _marc_above_lock_contact: already seated this scene. Restore
            # and freeze; do not re-snap (PhysX bounce + mesh min-Z).
            _set_xyz(lm, _hold["lm"])
            _set_xyz(person, _hold["person"])
            _marc_freeze_person(person)
            used_lms.add(str(lm.GetPath()))
            used_people.add(_lk)
            n += 1
            continue
        _marc_freeze_person(person)
        tops = _top_candidates(lm)
        if tops is None:
            print(
                "[on-landmark-patch] above-mode: no roof for %s"
                % str(getattr(lm, "GetPath", lambda: "?")())
            )
            continue
        ordered, roof_lo, roof_hi, body_lo, body_hi = tops
        cx, cy, top_z = ordered[0]
        pose = str(spec.get("type") or "")
        _frame_landmark(lm, top_z, typ, pose)
        tops = _top_candidates(lm)
        if tops is None:
            continue
        ordered, roof_lo, roof_hi, body_lo, body_hi = tops
        cx, cy, top_z = ordered[0]
        sink = 0.0
        if "sitting" in pose:
            if typ == "vending_machine":
                try:
                    sink = float(__import__("os").environ.get(
                        "TRAINER_PEOPLE_VEND_BUTT_SINK", "0.12") or "0.12")
                except (TypeError, ValueError):
                    sink = 0.12
                if "professor" in str(spec.get("character") or "").lower():
                    try:
                        sink += float(__import__("os").environ.get(
                            "TRAINER_PEOPLE_VEND_PROF_SINK", "0.08") or "0.08")
                    except (TypeError, ValueError):
                        sink += 0.08
                elif "female" in str(spec.get("character") or "").lower():
                    try:
                        sink += float(__import__("os").environ.get(
                            "TRAINER_PEOPLE_VEND_FEMALE_SINK", "0.05") or "0.05")
                    except (TypeError, ValueError):
                        sink += 0.05
                elif "weight" in str(spec.get("character") or "").lower():
                    try:
                        sink += float(__import__("os").environ.get(
                            "TRAINER_PEOPLE_VEND_WEIGHT_SINK", "0.07") or "0.07")
                    except (TypeError, ValueError):
                        sink += 0.07
                elif "male_student" in str(
                        spec.get("character") or "").lower():
                    try:
                        sink += float(__import__("os").environ.get(
                            "TRAINER_PEOPLE_VEND_MALE_SINK", "-0.01") or "-0.01")
                    except (TypeError, ValueError):
                        sink += -0.01
            else:
                try:
                    sink = float(__import__("os").environ.get(
                        "TRAINER_PEOPLE_SIT_SINK", "0.70") or "0.70")
                except (TypeError, ValueError):
                    sink = 0.70
                if typ == "trash_can":
                    # _marc_above_trash_sit_rim / _marc_above_trash_visual_seat:
                    # joint start near the lid; visual hips drop onto the rim.
                    try:
                        sink = float(__import__("os").environ.get(
                            "TRAINER_PEOPLE_SIT_SINK_TRASH", "0.42") or "0.42")
                    except (TypeError, ValueError):
                        sink = 0.42
                    if "weight" in str(spec.get("character") or "").lower():
                        try:
                            sink = float(__import__("os").environ.get(
                                "TRAINER_PEOPLE_SIT_SINK_TRASH_WEIGHT",
                                "0.28") or "0.28")
                        except (TypeError, ValueError):
                            sink = 0.28
                elif typ in ("postbox", "hydrant"):
                    try:
                        sink += float(__import__("os").environ.get(
                            "TRAINER_PEOPLE_SIT_SINK_SHORT", "0.35") or "0.35")
                    except (TypeError, ValueError):
                        sink += 0.35
        elif "crouching" in pose and typ == "hydrant":
            try:
                sink = float(__import__("os").environ.get(
                    "TRAINER_PEOPLE_CROUCH_SINK_HYDRANT", "0.18") or "0.18")
            except (TypeError, ValueError):
                sink = 0.18
        if typ == "postbox" or (
                typ == "vending_machine" and "sitting" not in pose):
            clearance = max(float(clearance), 0.05)
        if typ == "vending_machine" and "sitting" in pose:
            # _marc_above_crouch_authored: sitting clips need the glass
            # bind yaw. Crouch / person_walking / prone keep authored facing.
            fx, fy = _vending_front_xy(lm, body_lo, body_hi)
            extra = 0.0
            _ch = str(spec.get("character") or "").lower()
            if ("female" in _ch or "weight" in _ch or "professor" in _ch
                    or "male_student" in _ch):
                # Per-clip extra defaults to 0; _yaw_legs_along already
                # searches bind-forward. Env can still force a flip.
                if "female" in _ch:
                    _yaw_key = "TRAINER_PEOPLE_VEND_FEMALE_YAW"
                elif "professor" in _ch:
                    _yaw_key = "TRAINER_PEOPLE_VEND_PROF_YAW"
                elif "weight" in _ch:
                    _yaw_key = "TRAINER_PEOPLE_VEND_WEIGHT_YAW"
                else:
                    _yaw_key = "TRAINER_PEOPLE_VEND_MALE_YAW"
                try:
                    extra = float(__import__("os").environ.get(
                        _yaw_key, "0") or "0")
                except (TypeError, ValueError):
                    extra = 0.0
            # _marc_above_vend_yaw_no_double: _yaw_legs_along already
            # searches 0/90/180 for bind-forward. Extra 180 used to
            # flip every clip off the glass.
            _yaw_legs_along(person, fx, fy, extra)
            _marc_freeze_person(person)
        elif typ == "vending_machine" and "reaching" in pose:
            fx, fy = _vending_front_xy(lm, body_lo, body_hi)
            _yaw_legs_along(person, fx, fy, 0.0)
            _marc_freeze_person(person)
        elif typ == "trash_can" and "sitting" in pose:
            # _marc_above_trash_sit_yaw: hang legs toward the CCTV so
            # they drape off the visible rim, not into the cavity.
            # Yaw before seat; rotating after visual-align unseats.
            cam = _camera_xyz()
            if cam is not None:
                try:
                    _tyaw = float(__import__("os").environ.get(
                        "TRAINER_PEOPLE_TRASH_SIT_YAW", "0") or "0")
                except (TypeError, ValueError):
                    _tyaw = 0.0
                _yaw_legs_along(
                    person,
                    float(cam[0]) - float(cx),
                    float(cam[1]) - float(cy),
                    _tyaw)
                _marc_freeze_person(person)
        _ch = str(spec.get("character") or "").lower()
        _slots = ["lip"]
        _sit_margin = 0.08
        if typ == "vending_machine" and "sitting" in pose:
            if ("female" in _ch or "professor" in _ch
                    or "male_student" in _ch or "weight" in _ch):
                # _marc_above_vend_weight9: same glass-lip slots as the
                # other sitting clips, not roof-centre lip-only.
                _sit_margin = 0.12
                try:
                    _forced = str(__import__("os").environ.get(
                        "TRAINER_PEOPLE_VEND_SLOT", "auto") or "auto"
                    ).strip().lower()
                except Exception:
                    _forced = "auto"
                if _forced in ("center", "left", "right"):
                    _slots = [_forced]
                else:
                    _slots = ["center", "left", "right"]
                    __import__("random").shuffle(_slots)
        xy_from = "support"
        if typ in ("hydrant", "trash_can") and "crouching" in pose and body_lo is not None:
            xy_from = "centroid"
        elif typ == "trash_can" and "sitting" in pose:
            xy_from = "butt"
        elif typ == "vending_machine" and "sitting" in pose:
            xy_from = "butt"
        seated = False
        gap = 999.0
        pcx = pcy = 0.0
        support = None
        sx, sy = float(cx), float(cy)
        for _slot in _slots:
            if typ in ("hydrant", "trash_can") and "crouching" in pose and body_lo is not None:
                sx = 0.5 * (float(body_lo[0]) + float(body_hi[0]))
                sy = 0.5 * (float(body_lo[1]) + float(body_hi[1]))
            else:
                sx, sy = _drape_xy(
                    cx, cy, typ, pose, body_lo, body_hi, lm,
                    spec.get("character"), slot=_slot)
            support = _seat_at(
                person, sx, sy, float(top_z), float(clearance) - sink,
                xy_from=xy_from)
            _marc_freeze_person(person)
            if support is None:
                support = _support_from_joints(_posed_joints(person))
            if (typ == "vending_machine" and "sitting" in pose
                    and support is not None):
                nx, ny = _vending_roof_clamp(
                    lm, body_lo, body_hi,
                    float(support[0]), float(support[1]), _sit_margin)
                pxyz = _world_xyz(person)
                if pxyz is not None:
                    _set_xyz(person, [
                        float(pxyz[0]) + (nx - float(support[0])),
                        float(pxyz[1]) + (ny - float(support[1])),
                        float(pxyz[2]),
                    ])
                    _marc_freeze_person(person)
                    joints = _posed_joints(person)
                    support = (
                        _butt_from_joints(joints) if xy_from == "butt"
                        else _support_from_joints(joints)) or support
            if "professor" in _ch:
                # _marc_above_vend_prof11: a few cm away from the CCTV,
                # then clamp so cam-away cannot walk them off the roof.
                pxyz = _world_xyz(person)
                cam = _camera_xyz()
                if pxyz is not None:
                    try:
                        away = float(__import__("os").environ.get(
                            "TRAINER_PEOPLE_VEND_PROF_CAM_AWAY", "0.04") or "0.04")
                    except (TypeError, ValueError):
                        away = 0.04
                    nx = ny = 0.0
                    if cam is not None and abs(away) > 1e-6:
                        dx = float(pxyz[0]) - float(cam[0])
                        dy = float(pxyz[1]) - float(cam[1])
                        nrm = (dx * dx + dy * dy) ** 0.5
                        if nrm > 1e-4:
                            nx, ny = away * dx / nrm, away * dy / nrm
                    _set_xyz(person, [
                        float(pxyz[0]) + nx,
                        float(pxyz[1]) + ny,
                        float(pxyz[2]),
                    ])
                    _marc_freeze_person(person)
                    if typ == "vending_machine" and "sitting" in pose:
                        joints = _posed_joints(person)
                        support = (
                            _butt_from_joints(joints) if xy_from == "butt"
                            else _support_from_joints(joints)) or support
                        if support is not None:
                            cx2, cy2 = _vending_roof_clamp(
                                lm, body_lo, body_hi,
                                float(support[0]), float(support[1]),
                                _sit_margin)
                            pxyz = _world_xyz(person)
                            if pxyz is not None:
                                _set_xyz(person, [
                                    float(pxyz[0]) + (cx2 - float(support[0])),
                                    float(pxyz[1]) + (cy2 - float(support[1])),
                                    float(pxyz[2]),
                                ])
                                _marc_freeze_person(person)
                            joints = _posed_joints(person)
                            support = (
                                _butt_from_joints(joints) if xy_from == "butt"
                                else _support_from_joints(joints)
                            ) or support
            if (typ == "vending_machine" and "sitting" in pose):
                vis = _align_visual_on_roof(
                    person, float(sx), float(sy), float(top_z))
                if vis is not None:
                    support = vis
            if typ == "trash_can" and "sitting" in pose:
                vis = _align_visual_on_trash_lid(
                    person, float(sx), float(sy), float(top_z),
                    spec.get("character"))
                if vis is not None:
                    support = vis
                else:
                    support = None
            if support is not None:
                pcx, pcy = float(support[0]), float(support[1])
                gap = float(support[2]) - float(top_z)
                max_xy = 0.55
                seated = bool(_contact_ok(
                    support, sx, sy, float(top_z),
                    min_gap=(-(abs(sink) + 0.15) if sink else -0.05),
                    max_xy=max_xy))
                if (seated and typ == "vending_machine"
                        and "sitting" in pose):
                    # _marc_vend_visual_roof: visual hips, not joint butt.
                    hip = _mesh_hip_xy(
                        person, pcx, pcy, float(top_z))
                    hx = float(hip[0]) if hip else pcx
                    hy = float(hip[1]) if hip else pcy
                    seated = bool(_on_vending_roof(
                        lm, body_lo, body_hi, hx, hy, 0.04))
                if (seated and typ == "trash_can"
                        and "sitting" in pose):
                    seated = bool(
                        float(support[2]) >= float(top_z) - 0.12)
            if seated:
                break
        if (not seated) and typ == "vending_machine" and "sitting" in pose:
            origin = _vending_roof_xy(lm, body_lo, body_hi)
            if origin is not None:
                sx, sy = _vending_roof_clamp(
                    lm, body_lo, body_hi,
                    float(origin[0]), float(origin[1]), _sit_margin)
                support = _seat_at(
                    person, sx, sy, float(top_z),
                    float(clearance) - sink, xy_from="butt")
                _marc_freeze_person(person)
                if support is None:
                    support = _support_from_joints(_posed_joints(person))
                if support is not None:
                    nx, ny = _vending_roof_clamp(
                        lm, body_lo, body_hi,
                        float(support[0]), float(support[1]), _sit_margin)
                    pxyz = _world_xyz(person)
                    if pxyz is not None:
                        _set_xyz(person, [
                            float(pxyz[0]) + (nx - float(support[0])),
                            float(pxyz[1]) + (ny - float(support[1])),
                            float(pxyz[2]),
                        ])
                        _marc_freeze_person(person)
                        joints = _posed_joints(person)
                        support = _butt_from_joints(joints) or support
                    vis = _align_visual_on_roof(
                        person, float(sx), float(sy), float(top_z))
                    if vis is not None:
                        support = vis
                    pcx, pcy = float(support[0]), float(support[1])
                    gap = float(support[2]) - float(top_z)
                    hip = _mesh_hip_xy(
                        person, pcx, pcy, float(top_z))
                    hx = float(hip[0]) if hip else pcx
                    hy = float(hip[1]) if hip else pcy
                    seated = bool(_contact_ok(
                        support, sx, sy, float(top_z),
                        min_gap=(-(abs(sink) + 0.15) if sink else -0.05),
                        max_xy=0.55) and _on_vending_roof(
                            lm, body_lo, body_hi, hx, hy, 0.04))
        if (not seated) and typ == "trash_can" and "sitting" in pose:
            vis = _align_visual_on_trash_lid(
                person, float(sx), float(sy), float(top_z),
                spec.get("character"))
            if vis is not None:
                support = vis
                pcx, pcy = float(support[0]), float(support[1])
                gap = float(support[2]) - float(top_z)
                seated = bool(_contact_ok(
                    support, sx, sy, float(top_z),
                    min_gap=(-(abs(sink) + 0.15) if sink else -0.05),
                    max_xy=0.55) and float(support[2]) >= float(top_z) - 0.12)
        if not seated:
            _rej = (
                str(person.GetPath()), str(lm.GetPath()),
                round(gap, 3), round(pcx, 2), round(pcy, 2),
            )
            if globals().get("_marc_above_last_rej") != _rej:
                globals()["_marc_above_last_rej"] = _rej
                print(
                    "[on-landmark-patch] above-mode: rejected %s on %s "
                    "(gap=%.3f xy=%.2f,%.2f top_z=%.3f)"
                    % (str(person.GetPath()), str(lm.GetPath()),
                       gap, pcx, pcy, top_z)
                )
            continue
        _ok = (
            str(person.GetPath()), str(lm.GetPath()),
            round(gap, 3), round(pcx - cx, 2), round(pcy - cy, 2),
        )
        if globals().get("_marc_above_last_ok") != _ok:
            globals()["_marc_above_last_ok"] = _ok
            print(
                "[on-landmark-patch] above-mode: seated %s on %s "
                "dx=%.2f dy=%.2f gap=%.3f"
                % (str(person.GetPath()), str(lm.GetPath()),
                   pcx - cx, pcy - cy, gap)
            )
        used_lms.add(str(lm.GetPath()))
        used_people.add(str(person.GetPath()))
        _pxyz = _world_xyz(person)
        _lxyz = _world_xyz(lm)
        if _pxyz is not None and _lxyz is not None:
            _lock = dict(globals().get("_marc_above_lock") or {})
            _lock[str(person.GetPath())] = {
                "person": list(_pxyz), "lm": list(_lxyz)}
            globals()["_marc_above_lock"] = _lock
        _marc_freeze_person(person)
        n += 1
    globals()["_marc_above_placed_n"] = n
    _sum = ((jobs[0][1] if jobs else "-"), n, len(jobs))
    if globals().get("_marc_above_last_sum") != _sum:
        globals()["_marc_above_last_sum"] = _sum
        print(
            "[on-landmark-patch] above-mode: target=%s placed %d/%d person(s) "
            "with posed-joint roof contact" % _sum
        )


def _marc_nudge_above_nearer(mission_id, objects, configurations=None, step=0.18):
    """[marc-patch] Slide climb target toward camera-near so the roof stays in frame."""
    near = globals().get("_marc_above_near_xy")
    if not near:
        return False
    try:
        nx, ny = float(near[0]), float(near[1])
    except (TypeError, ValueError, IndexError):
        return False
    mag = (nx * nx + ny * ny) ** 0.5
    if mag < 1e-4:
        return False
    nx, ny = nx / mag * float(step), ny / mag * float(step)
    jobs = []
    for spec in objects or []:
        pl = spec.get("placement") or {}
        if pl.get("above"):
            jobs.append((spec, str(pl.get("on_landmark_type") or "")))
    if not jobs:
        return False
    try:
        import omni.usd
        from pxr import UsdGeom, Gf, Usd
    except Exception:
        return False
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return False

    def _world_xyz(prim):
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return None
        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

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

    spec, typ = jobs[0]
    group = stage.GetPrimAtPath("/world/landmarks/%s" % mission_id)
    lm = None
    if group and group.IsValid():
        for child in group.GetChildren():
            name = str(child.GetName() or "")
            blob = _norm(name) + _norm(str(child.GetPath()))
            if (not typ) or name.startswith(typ) or _norm(typ) in blob:
                lm = child
                break
    pose = str(spec.get("type") or "")
    ch = str(spec.get("character") or "")
    person = None
    try:
        n = max(1, int(spec.get("count", 1) or 1))
    except (TypeError, ValueError):
        n = 1
    for seq in range(n):
        pp = "/world/people/%s_%s_%s_%s" % (mission_id, pose, ch, seq)
        prim = stage.GetPrimAtPath(pp)
        if prim and prim.IsValid():
            person = prim
            break
    if lm is None or person is None:
        return False
    lm_xyz = _world_xyz(lm)
    p_xyz = _world_xyz(person)
    if lm_xyz is None or p_xyz is None:
        return False
    _set_xyz(lm, [lm_xyz[0] + nx, lm_xyz[1] + ny, lm_xyz[2]])
    _set_xyz(person, [p_xyz[0] + nx, p_xyz[1] + ny, p_xyz[2]])
    print(
        "[on-landmark-patch] above-mode: nudged %s toward camera-near "
        "(dx=%.3f dy=%.3f)" % (typ or "-", nx, ny)
    )
    return True
'''


BESIDE_HELPER = r'''

def _marc_place_beside_people(mission_id, objects, configurations=None):
    """[marc-patch] Ground-place recipe targets beside a landmark (not on the roof)."""
    try:
        import json as _json_tip
        import math
        import omni.usd
        from pxr import Usd, UsdGeom, Gf
    except Exception:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def _world_xyz(prim):
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return None
        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

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

    def _yaw_person(prim, deg):
        # _marc_yaw_keep_root: compose world-Z onto pose-root pitch/roll.
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        orient = ops.get("xformOp:orient")
        if orient is None:
            orient = xf.AddOrientOp()
        _math = __import__("math")
        q_new = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(deg)).GetQuat()
        try:
            m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            xdir = m.TransformDir(Gf.Vec3d(1, 0, 0))
            ydir = m.TransformDir(Gf.Vec3d(0, 1, 0))
            xn = float(xdir[0]) ** 2 + float(xdir[1]) ** 2
            yn = float(ydir[0]) ** 2 + float(ydir[1]) ** 2
            if xn >= yn:
                cur = _math.degrees(_math.atan2(float(xdir[1]), float(xdir[0])))
            else:
                cur = _math.degrees(_math.atan2(float(ydir[1]), float(ydir[0])))
            delta = (float(deg) - cur + 180.0) % 360.0 - 180.0
            q_old = orient.Get()
            if q_old is None:
                q_old = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            q_delta = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(delta)).GetQuat()
            try:
                q_new = Gf.Quatf(q_delta) * Gf.Quatf(q_old)
            except Exception:
                q_new = Gf.Quatd(q_delta) * Gf.Quatd(q_old)
        except Exception:
            pass
        try:
            orient.Set(Gf.Quatf(q_new))
        except Exception:
            orient.Set(q_new)
        ordered = []
        if "xformOp:translate" in ops:
            ordered.append(ops["xformOp:translate"])
        ordered.append(orient)
        if "xformOp:scale" in ops:
            ordered.append(ops["xformOp:scale"])
        if ordered:
            xf.SetXformOpOrder(ordered)

    def _find_lm(typ):
        group = stage.GetPrimAtPath("/world/landmarks/%s" % mission_id)
        if not (group and group.IsValid()):
            return None
        for child in group.GetChildren():
            name = str(child.GetName() or "")
            blob = _norm(name) + _norm(str(child.GetPath()))
            if (not typ) or name.startswith(typ) or _norm(typ) in blob:
                return child
        return None

    def _find_person(spec):
        pose = str(spec.get("type") or "")
        ch = str(spec.get("character") or "")
        try:
            n = max(1, int(spec.get("count", 1) or 1))
        except (TypeError, ValueError):
            n = 1
        for seq in range(n):
            pp = "/world/people/%s_%s_%s_%s" % (mission_id, pose, ch, seq)
            prim = stage.GetPrimAtPath(pp)
            if prim and prim.IsValid():
                return prim
        return None

    rec = {}
    try:
        rec = _json_tip.load(open("/tmp/marc_sar_recipe.json")) or {}
    except Exception:
        rec = {}
    tip = str(rec.get("tip_landmark") or "")
    if tip:
        lm = _find_lm(tip)
        if lm is not None:
            tip_fn = globals().get("_set_fallen_rotation")
            if callable(tip_fn):
                try:
                    tip_fn(str(lm.GetPath()), True)
                except Exception:
                    try:
                        tip_fn(str(lm.GetPath()))
                    except Exception:
                        pass

    n = 0
    for spec in objects or []:
        pl = spec.get("placement") or {}
        typ = str(pl.get("beside_landmark_type") or "")
        if not typ:
            continue
        lm = _find_lm(typ)
        person = _find_person(spec)
        if lm is None or person is None:
            continue
        lxyz = _world_xyz(lm)
        pxyz = _world_xyz(person)
        if lxyz is None:
            continue
        dx, dy = 1.6, 0.4
        if pxyz is not None:
            vx, vy = pxyz[0] - lxyz[0], pxyz[1] - lxyz[1]
            mag = math.hypot(vx, vy)
            if mag > 0.15:
                dx, dy = vx / mag * 1.6, vy / mag * 1.6
        want = [lxyz[0] + dx, lxyz[1] + dy, lxyz[2]]
        drop = globals().get("drop_to_ground")
        if callable(drop):
            try:
                dropped = drop(want[0], want[1], want[2])
                if dropped is not None and len(dropped) >= 3:
                    want = [float(dropped[0]), float(dropped[1]), float(dropped[2])]
            except Exception:
                pass
        else:
            try:
                from world.people_poser import drop_to_ground as _drop
                dropped = _drop(want[0], want[1], want[2])
                if dropped is not None and len(dropped) >= 3:
                    want = [float(dropped[0]), float(dropped[1]), float(dropped[2])]
            except Exception:
                pass
        _set_xyz(person, want)
        pose = str(spec.get("type") or "")
        # _marc_beside_face_yaw: leftover walk-spawn yaw made bending
        # miss the can and lying people axis-align to the world, not
        # the taxi/bike.
        vx, vy = want[0] - lxyz[0], want[1] - lxyz[1]
        if "lying" in pose or "prone" in pose:
            deg = math.degrees(math.atan2(vy, vx)) + 90.0
        else:
            deg = math.degrees(math.atan2(
                lxyz[1] - want[1], lxyz[0] - want[0]))
        _yaw_person(person, deg)
        n += 1
        print(
            "[on-landmark-patch] beside: %s next to %s at %.2f,%.2f"
            % (str(person.GetPath()), typ, want[0], want[1])
        )
    if n:
        print("[on-landmark-patch] beside-mode: placed %d person(s)" % n)
'''

SEAT_HELPER = r'''

def _marc_place_seat_people(mission_id, objects, configurations=None):
    """[marc-patch] Seat-lie recipe targets on the bench/table top (_marc_recipe_seat)."""
    try:
        import math
        import omni.usd
        from pxr import Usd, UsdGeom, Gf, UsdPhysics
    except Exception:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    def _norm(s):
        return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def _world_xyz(prim):
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return None
        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

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

    def _yaw_person(prim, deg):
        # _marc_yaw_keep_root: compose world-Z onto pose-root pitch/roll.
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return
        ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
        orient = ops.get("xformOp:orient")
        if orient is None:
            orient = xf.AddOrientOp()
        _math = __import__("math")
        q_new = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(deg)).GetQuat()
        try:
            m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            xdir = m.TransformDir(Gf.Vec3d(1, 0, 0))
            ydir = m.TransformDir(Gf.Vec3d(0, 1, 0))
            xn = float(xdir[0]) ** 2 + float(xdir[1]) ** 2
            yn = float(ydir[0]) ** 2 + float(ydir[1]) ** 2
            if xn >= yn:
                cur = _math.degrees(_math.atan2(float(xdir[1]), float(xdir[0])))
            else:
                cur = _math.degrees(_math.atan2(float(ydir[1]), float(ydir[0])))
            delta = (float(deg) - cur + 180.0) % 360.0 - 180.0
            q_old = orient.Get()
            if q_old is None:
                q_old = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
            q_delta = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(delta)).GetQuat()
            try:
                q_new = Gf.Quatf(q_delta) * Gf.Quatf(q_old)
            except Exception:
                q_new = Gf.Quatd(q_delta) * Gf.Quatd(q_old)
        except Exception:
            pass
        try:
            orient.Set(Gf.Quatf(q_new))
        except Exception:
            orient.Set(q_new)
        ordered = []
        if "xformOp:translate" in ops:
            ordered.append(ops["xformOp:translate"])
        ordered.append(orient)
        if "xformOp:scale" in ops:
            ordered.append(ops["xformOp:scale"])
        if ordered:
            xf.SetXformOpOrder(ordered)

    def _long_axis_yaw(lm):
        xf = UsdGeom.Xformable(lm)
        if not xf:
            return 0.0
        m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xdir = m.TransformDir(Gf.Vec3d(1, 0, 0))
        ydir = m.TransformDir(Gf.Vec3d(0, 1, 0))
        ax, ay = float(xdir[0]), float(xdir[1])
        try:
            bound = UsdGeom.Imageable(lm).ComputeLocalBound(
                Usd.TimeCode.Default(), "default")
            rng = bound.GetRange()
            lo, hi = rng.GetMin(), rng.GetMax()
            ex = abs(float(hi[0]) - float(lo[0]))
            ey = abs(float(hi[1]) - float(lo[1]))
            if ey > ex:
                ax, ay = float(ydir[0]), float(ydir[1])
        except Exception:
            pass
        return math.degrees(math.atan2(ay, ax))

    def _top_xyz(prim):
        try:
            boundable = UsdGeom.Boundable(prim)
            rng = boundable.ComputeWorldBound(
                Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
            box = rng.ComputeAlignedBox()
            mn, mx = box.GetMin(), box.GetMax()
            return [
                0.5 * (float(mn[0]) + float(mx[0])),
                0.5 * (float(mn[1]) + float(mx[1])),
                float(mx[2]),
            ]
        except Exception:
            return _world_xyz(prim)

    def _freeze(prim):
        try:
            rb = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath())
            if rb:
                attr = rb.GetKinematicEnabledAttr()
                if not attr:
                    attr = rb.CreateKinematicEnabledAttr()
                attr.Set(True)
        except Exception:
            pass

    def _find_lm(typ):
        group = stage.GetPrimAtPath("/world/landmarks/%s" % mission_id)
        if not (group and group.IsValid()):
            return None
        for child in group.GetChildren():
            name = str(child.GetName() or "")
            blob = _norm(name) + _norm(str(child.GetPath()))
            if (not typ) or name.startswith(typ) or _norm(typ) in blob:
                return child
        return None

    def _find_person(spec):
        pose = str(spec.get("type") or "")
        ch = str(spec.get("character") or "")
        try:
            n = max(1, int(spec.get("count", 1) or 1))
        except (TypeError, ValueError):
            n = 1
        for seq in range(n):
            pp = "/world/people/%s_%s_%s_%s" % (mission_id, pose, ch, seq)
            prim = stage.GetPrimAtPath(pp)
            if prim and prim.IsValid():
                return prim
        return None

    n = 0
    for spec in objects or []:
        pl = spec.get("placement") or {}
        if pl.get("above") or pl.get("beside_landmark_type"):
            continue
        if not pl.get("recipe_id"):
            continue
        typ = str(pl.get("on_landmark_type") or "")
        if not typ:
            continue
        lm = _find_lm(typ)
        person = _find_person(spec)
        if lm is None or person is None:
            continue
        _freeze(lm)
        top = _top_xyz(lm)
        if top is None:
            continue
        _set_xyz(person, [top[0], top[1], top[2] + 0.06])
        pose = str(spec.get("type") or "")
        # _marc_recipe_seat_yaw: lie along the bench/table long axis,
        # not leftover walk-spawn heading.
        deg = _long_axis_yaw(lm)
        if "sitting" in pose:
            deg = deg + 90.0
        _yaw_person(person, deg)
        # _marc_recipe_visual_center: animation meshes are offset from their
        # USD roots; center the rendered body over the support after yaw.
        try:
            box = UsdGeom.Imageable(person).ComputeWorldBound(
                Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedBox()
            lo, hi = box.GetMin(), box.GetMax()
            pxyz = _world_xyz(person)
            _set_xyz(person, [
                pxyz[0] + top[0] - 0.5 * (float(lo[0]) + float(hi[0])),
                pxyz[1] + top[1] - 0.5 * (float(lo[1]) + float(hi[1])),
                pxyz[2],
            ])
        except Exception:
            pass
        _freeze(person)
        n += 1
        print(
            "[on-landmark-patch] seat: %s on %s at %.2f,%.2f,%.2f"
            % (str(person.GetPath()), typ, top[0], top[1], top[2] + 0.06)
        )
    if n:
        print("[on-landmark-patch] seat-mode: placed %d person(s)" % n)
'''


def _ensure_above_helper(path):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    have_fn = "def _marc_place_above_people" in src
    have_contact = (
        "_marc_above_aabb_stack" in src and "_marc_above_target" in src
        and "_marc_above_freeze" in src
        and "_marc_above_always_on_top" in src
        and "_marc_above_joint_seat" in src
        and "_marc_above_ray_roof" in src
        and "_marc_above_sit_sink" in src
        and "_marc_above_sit_sink_70" in src
        and "_marc_above_sit_sink_short" in src
        and "_marc_above_vend_drape" in src
        and "_marc_above_vend_drape_lip" in src
        and "_marc_above_trash_raise" in src
        and "_marc_above_hydrant_crouch" in src
        and "_marc_above_trash_crouch" in src
        and "_marc_above_vend_farther" in src
        and "_marc_above_vend_farther2" in src
        and "_marc_above_vend_farther3" in src
        and "_marc_above_vend_farther4" in src
        and "_marc_above_vend_not_too_far" in src
        and "_marc_above_vend_closer" in src
        and "_marc_above_vend_sit_yaw" in src
        and "_marc_above_vend_glass_drape" in src
        and "_marc_above_vend_sit_fix" in src
        and "_marc_above_vend_sit_nudge" in src
        and "_marc_above_vend_sit_unraise" in src
        and "_marc_above_vend_lip_fwd" in src
        and "_marc_above_vend_lip_fwd2" in src
        and "_marc_above_vend_lip_fwd3" in src
        and "_marc_above_vend_lip_back" in src
        and "_marc_above_vend_prof" in src
        and "_marc_above_vend_prof2" in src
        and "_marc_above_vend_prof3" in src
        and "_marc_above_vend_prof4" in src
        and "_marc_above_vend_prof5" in src
        and "_marc_above_vend_prof6" in src
        and "_marc_above_vend_female" in src
        and "_marc_above_vend_female2" in src
        and "_marc_above_vend_female3" in src
        and "_marc_above_vend_female4" in src
        and "_marc_above_vend_female5" in src
        and "_marc_above_vend_female6" in src
        and "_marc_above_vend_female7" in src
        and "_marc_above_vend_male" in src
        and "_marc_above_vend_male2" in src
        and "_marc_above_vend_male3" in src
        and "_marc_above_vend_male4" in src
        and "_marc_above_vend_male5" in src
        and "_marc_above_vend_weight" in src
        and "_marc_above_vend_weight2" in src
        and "_marc_above_vend_weight3" in src
        and "_marc_above_vend_weight4" in src
        and "_marc_above_vend_weight5" in src
        and "_marc_above_no_closeup" in src
        and "_marc_above_mid_field" in src
        and "_marc_above_roof_in_view" in src
        and "_marc_above_vend_abs_frame" in src
        and "_marc_above_vend_leg_yaw" in src
        and "_marc_above_vend_mesh_seat" in src
        and "_marc_above_vend_on_roof" in src
        and "_marc_above_vend_male6" in src
        and "_marc_above_vend_male7" in src
        and "_marc_above_vend_prof7" in src
        and "_marc_above_vend_female8" in src
        and "_marc_above_vend_weight6" in src
        and "_marc_above_vend_roof_visible" in src
        and "_marc_above_vend_joint_z" in src
        and "_marc_above_vend_prof8" in src
        and "_marc_above_vend_mesh_xy" in src
        and "_marc_above_vend_female_yaw" in src
        and "_marc_above_vend_prof9" in src
        and "_marc_above_vend_weight_yaw" in src
        and "_marc_above_vend_female_fwd" in src
        and "_marc_above_vend_stand" in src
        and "_marc_above_vend_prof10" in src
        and "_marc_above_vend_prof_yaw" in src
        and "_marc_above_vend_sit_hip" in src
        and "_marc_above_vend_weight7" in src
        and "_marc_above_vend_weight8" in src
        and "_marc_above_vend_weight9" in src
        and "_marc_above_vend_mesh_contact" in src
        and "_marc_above_crouch_authored" in src
        and "_marc_above_vend_mesh_contact2" in src
        and "_marc_above_vend_mesh_nudge" in src
        and "_marc_above_vend_mesh_snap" in src
        and "_marc_above_lock_contact" in src
        and "_marc_above_joint_only" in src
        and "_marc_above_vend_prof11" in src
        and "_marc_above_vend_slot" in src
        and "_marc_above_vend_female9" in src
        and "_marc_above_vend_prof12" in src
        and "_marc_above_camera_frame" in src
        and "_marc_above_headroom" in src
        and "_marc_above_vend_visual_seat" in src
        and "_marc_recipe_scene" in src
        and "_marc_above_trash_sit_rim" in src
        and "_marc_above_trash_visual_seat" in src
        and "_marc_above_trash_sit_yaw" in src
        and "_marc_above_vend_yaw_no_double" in src
        and "_marc_yaw_keep_root" in src
        and "_marc_above_vend_sit_v80" in src
        and "_marc_above_trash_weight_lid" in src
        and "_marc_trash_sit_rim_xy" in src
        and "_marc_trash_butt_z" in src
        and "_marc_vend_cam_lip" in src
        and "_marc_vend_visual_roof" in src
        and "_marc_sit_frame_head" in src
        and "def visible_above_stack" in src
        and "RemoveAPI" in src
        and "GetPointsAttr" in src
        and "def _marc_nudge_above_nearer" in src)
    if have_fn and have_contact:
        if "def _marc_above_scene(" not in src:
            src = src.replace(
                "\ndef _marc_place_above_people(",
                "\n" + ABOVE_SCENE_FN + "def _marc_place_above_people(",
                1,
            )
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(src)
            except OSError as e:
                print(f"[on-landmark-patch] above-scene helper write failed: {e}")
                return False
            print("[on-landmark-patch] deployer: recipe-above framing helper inserted")
            return True
        return False
    if have_fn:
        starts = []
        for name in (
                "\ndef _marc_place_above_people",
                "\ndef _marc_nudge_above_nearer",
                "def _marc_place_above_people",
        ):
            i = src.find(name)
            if i >= 0:
                starts.append(i)
        if starts:
            src = src[:min(starts)]
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


def _ensure_beside_helper(path):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    have = "def _marc_place_beside_people" in src
    if (have and "_marc_beside_face_yaw" in src
            and "_marc_yaw_keep_root" in src):
        return False
    if have:
        i = src.find("\ndef _marc_place_beside_people")
        if i < 0:
            i = src.find("def _marc_place_beside_people")
        j = src.find("\ndef _marc_place_seat_people")
        if i >= 0 and j > i:
            src = src[:i] + BESIDE_HELPER + src[j:]
        elif i >= 0:
            src = src[:i].rstrip() + BESIDE_HELPER
        else:
            src = src.rstrip() + BESIDE_HELPER
        action = "replaced"
    else:
        src = src.rstrip() + BESIDE_HELPER
        action = "appended"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    except OSError as e:
        print(f"[on-landmark-patch] beside-helper write failed: {e}")
        return False
    print("[on-landmark-patch] deployer: beside-mode helper %s" % action)
    return True


def _ensure_seat_helper(path):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    have = "def _marc_place_seat_people" in src
    if (have and "_marc_recipe_seat_yaw" in src
            and "_marc_yaw_keep_root" in src):
        return False
    if have:
        i = src.find("\ndef _marc_place_seat_people")
        if i < 0:
            i = src.find("def _marc_place_seat_people")
        if i >= 0:
            src = src[:i].rstrip() + SEAT_HELPER
        else:
            src = src.rstrip() + SEAT_HELPER
        action = "replaced"
    else:
        src = src.rstrip() + SEAT_HELPER
        action = "appended"
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    except OSError as e:
        print(f"[on-landmark-patch] seat-helper write failed: {e}")
        return False
    print("[on-landmark-patch] deployer: seat-mode helper %s" % action)
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
        if NEW_SETTLE not in tnew:
            _play_loop = (
                "        simulation_context.play()\n"
                "        for _ in range(_SETTLE_FRAMES):\n"
                "            simulation_context.step(render=False)\n"
            )
            if OLD_SETTLE in tnew:
                tnew = tnew.replace(OLD_SETTLE, NEW_SETTLE, 1)
            else:
                _settle_done = False
                for _snap in (NEW_SNAP, NEW_SNAP_V4, NEW_SNAP_V3, NEW_SNAP_V2,
                              NEW_SNAP_V1):
                    _blob = _play_loop + _snap
                    if _blob in tnew:
                        tnew = tnew.replace(_blob, NEW_SETTLE, 1)
                        _settle_done = True
                        break
                if (not _settle_done) and NEW_SNAP not in tnew:
                    if NEW_SNAP_V4 in tnew:
                        tnew = tnew.replace(NEW_SNAP_V4, NEW_SNAP, 1)
                    elif NEW_SNAP_V3 in tnew:
                        tnew = tnew.replace(NEW_SNAP_V3, NEW_SNAP, 1)
                    elif (NEW_SNAP_V2 in tnew and
                            "TRAINER_RENDER_WARMUP_FRAMES" not in tnew):
                        tnew = tnew.replace(NEW_SNAP_V2, NEW_SNAP, 1)
                    elif (NEW_SNAP_V1 in tnew
                            and "_marc_place_above_people" not in tnew):
                        tnew = tnew.replace(NEW_SNAP_V1, NEW_SNAP, 1)
                    elif (OLD_SNAP in tnew
                            and "reposition_on_landmark_people" not in tnew):
                        tnew = tnew.replace(OLD_SNAP, NEW_SNAP, 1)
        if "_marc_reseat_recipe" not in tnew:
            _old_settle_else = (
                "            except Exception as _e:\n"
                '                log.warning(f"place-above after settle failed '
                '(non-fatal): {_e}")\n'
                "        else:\n"
                "            simulation_context.play()\n"
                "            for _ in range(_SETTLE_FRAMES):\n"
                "                simulation_context.step(render=False)\n"
                "            _snap_landmarks_to_ground([mission])\n"
            )
            _new_settle_else = NEW_SETTLE[
                NEW_SETTLE.find("            except Exception as _e:\n"
                                '                log.warning(f"place-above after settle failed'):
            ]
            if _old_settle_else in tnew and _new_settle_else:
                tnew = tnew.replace(_old_settle_else, _new_settle_else, 1)
        if NEW_SAVE not in tnew:
            if NEW_SAVE_V4 in tnew:
                tnew = tnew.replace(NEW_SAVE_V4, NEW_SAVE, 1)
            elif NEW_SAVE_V3 in tnew:
                tnew = tnew.replace(NEW_SAVE_V3, NEW_SAVE, 1)
            elif NEW_SAVE_V2 in tnew:
                tnew = tnew.replace(NEW_SAVE_V2, NEW_SAVE, 1)
            elif NEW_SAVE_V1 in tnew:
                tnew = tnew.replace(NEW_SAVE_V1, NEW_SAVE, 1)
            elif OLD_SAVE in tnew:
                tnew = tnew.replace(OLD_SAVE, NEW_SAVE, 1)
        if "Recipe scene" not in tnew:
            _old_save_tail = (
                "            else:\n"
                "                log.info(\n"
                '                    f"Above scene {scene_id}: target={_target or \'-\'} "\n'
                '                    f"saved placed={_n_placed}")\n'
                "        if save_offline_enabled and _active_valid:\n"
            )
            _elif_a = NEW_SAVE.find("        elif _marc_recipe_scene():")
            _elif_b = NEW_SAVE.find("        if save_offline_enabled and _active_valid:")
            if _old_save_tail in tnew and _elif_a >= 0 and _elif_b > _elif_a:
                tnew = tnew.replace(
                    _old_save_tail,
                    NEW_SAVE[_elif_a:_elif_b]
                    + "        if save_offline_enabled and _active_valid:\n",
                    1,
                )
        if ("_marc_recipe_scene()" in tnew
                and "_marc_recipe_climb_visible" not in tnew):
            # Stale NEW_SAVE (recipe gate without stack visibility).
            _a = tnew.find("        if _marc_above_scene():")
            _b = tnew.find("        if save_offline_enabled and _active_valid:")
            _na = NEW_SAVE.find("        if _marc_above_scene():")
            _nb = NEW_SAVE.find("        if save_offline_enabled and _active_valid:")
            if (_a >= 0 and _b > _a and _na >= 0 and _nb > _na
                    and "active_payload = gt_by_mission.get(active_mission_id)"
                    in tnew[_a - 200:_a]):
                tnew = tnew[:_a] + NEW_SAVE[_na:_nb] + tnew[_b:]
                print("[on-landmark-patch] trainer.py: recipe climb-visible "
                      "skip-save upgraded")
        if ('str(rec.get("relation") or "") in ("on", "above")' in tnew
                and "if _rel is True" not in tnew):
            tnew = tnew.replace(
                "            _sup = str(\n"
                "                rec.get(\"support_class\")\n"
                "                or rec.get(\"support_landmark\")\n"
                "                or rec.get(\"surface_anchor\")\n"
                "                or \"\")\n"
                "            _tgt = None\n",
                "            _sup = str(\n"
                "                rec.get(\"support_class\")\n"
                "                or rec.get(\"support_landmark\")\n"
                "                or rec.get(\"surface_anchor\")\n"
                "                or \"\")\n"
                "            _rel = rec.get(\"relation\")\n"
                "            if _rel is True:\n"
                "                _rel = \"on\"\n"
                "            _rel = str(_rel or \"\").strip().lower()\n"
                "            _tgt = None\n",
                1,
            )
            tnew = tnew.replace(
                'str(rec.get("relation") or "") in ("on", "above")',
                '_rel in ("on", "above")',
            )
            print("[on-landmark-patch] trainer.py: YAML on-boolean "
                  "skip-save upgraded")
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
        if NEW_RENDER_STEP not in tnew:
            _old_render_above = (
                "        if _marc_above_scene():\n"
                "            try:\n"
                "                simulation_context.pause()\n"
                "            except Exception:\n"
                "                pass\n"
                "            try:\n"
                "                from world.object_deployer import "
                "_marc_place_above_people\n"
                "                _miss = next(\n"
                "                    m for m in missions if m.get(\"id\") "
                "== active_mission_id)\n"
                "                _marc_place_above_people(\n"
                "                    _miss[\"id\"], _miss.get(\"objects\") or [],\n"
                "                    configurations)\n"
                "            except Exception as _pe:\n"
                "                log.warning(\n"
                '                    f"place-above before GT capture failed '
                '(non-fatal): {_pe}")\n'
            )
            if _old_render_above in tnew:
                tnew = tnew.replace(_old_render_above, NEW_RENDER_STEP[
                    :NEW_RENDER_STEP.find("        simulation_context.play()\n")
                ], 1)
            _old_reseat = (
                "                if _marc_above_scene():\n"
                "                    try:\n"
                "                        from world.object_deployer import "
                "_marc_place_above_people\n"
                "                        _miss = next(\n"
                "                            m for m in missions if m.get(\"id\") "
                "== active_mission_id)\n"
                "                        _marc_place_above_people(\n"
                "                            _miss[\"id\"], _miss.get(\"objects\") "
                "or [],\n"
                "                            configurations)\n"
                "                    except Exception as _pe:\n"
                "                        log.warning(\n"
                '                            f"place-above GT step failed '
                '(non-fatal): {_pe}")\n'
            )
            _new_reseat_a = NEW_RENDER_STEP.find(
                "                if _marc_above_scene():\n")
            _new_reseat_b = NEW_RENDER_STEP.find("            _marc_reseat()\n")
            if (_old_reseat in tnew and _new_reseat_a >= 0
                    and _new_reseat_b > _new_reseat_a):
                tnew = tnew.replace(
                    _old_reseat, NEW_RENDER_STEP[_new_reseat_a:_new_reseat_b], 1)
            elif NEW_RENDER_STEP_V1 in tnew:
                tnew = tnew.replace(NEW_RENDER_STEP_V1, NEW_RENDER_STEP, 1)
            elif OLD_RENDER_STEP in tnew:
                tnew = tnew.replace(OLD_RENDER_STEP, NEW_RENDER_STEP, 1)
        tnew = tnew.replace(
            "            _active_valid = _n_placed > 0 and _n_stack > 0\n",
            "            _active_valid = _n_placed > 0\n",
            1,
        )
        tnew = tnew.replace(
            "                    _climb_v = (0.01 + random.random() * 0.06\n"
            "                                if _pl in (\"vending_machine\", \"postbox\")\n"
            "                                else 0.04 + random.random() * 0.08)\n",
            "                    _climb_v = 0.16 + random.random() * 0.18\n",
            1,
        )
        tnew = tnew.replace(
            "                        _climb_v = 0.66 + random.random() * 0.12\n",
            "                        _climb_v = 0.20 + random.random() * 0.12\n",
            1,
        )
        tnew = tnew.replace(
            "                        _climb_v = 0.78 + random.random() * 0.10\n",
            "                        _climb_v = 0.20 + random.random() * 0.12\n",
            1,
        )
        tnew = tnew.replace(
            "                        _climb_v = 0.48 + random.random() * 0.12\n",
            "                        _climb_v = 0.20 + random.random() * 0.12\n",
            1,
        )
        tnew = tnew.replace(
            "                        _climb_v = 0.55 + random.random() * 0.10\n",
            "                        _climb_v = 0.20 + random.random() * 0.12\n",
            1,
        )
        tnew = tnew.replace(
            "                        _climb_v = 0.42 + random.random() * 0.10\n",
            "                        _climb_v = 0.20 + random.random() * 0.12\n",
            1,
        )
        _cap_lottery = (
            "        if str(os.environ.get(\"TRAINER_PEOPLE_ABOVE\", \"0\"))"
            ".lower() not in (\n"
            "                \"0\", \"false\", \"no\", \"off\", \"\"):\n"
            "            return _active_valid\n"
            "        return gt.last_render_ok\n"
        )
        if NEW_CAPTURE_RETURN not in tnew:
            if NEW_CAPTURE_RETURN_V1 in tnew:
                tnew = tnew.replace(NEW_CAPTURE_RETURN_V1, NEW_CAPTURE_RETURN, 1)
            elif _cap_lottery in tnew:
                tnew = tnew.replace(_cap_lottery, NEW_CAPTURE_RETURN, 1)
            elif OLD_CAPTURE_RETURN in tnew:
                tnew = tnew.replace(OLD_CAPTURE_RETURN, NEW_CAPTURE_RETURN, 1)
        if OLD_INITIAL_WARN in tnew:
            tnew = tnew.replace(OLD_INITIAL_WARN, NEW_INITIAL_WARN, 1)
        if NEW_POSITIONS not in tnew:
            if NEW_POSITIONS_V6 in tnew:
                tnew = tnew.replace(NEW_POSITIONS_V6, NEW_POSITIONS, 1)
            elif NEW_POSITIONS_V5 in tnew:
                tnew = tnew.replace(NEW_POSITIONS_V5, NEW_POSITIONS, 1)
            elif NEW_POSITIONS_V4 in tnew:
                tnew = tnew.replace(NEW_POSITIONS_V4, NEW_POSITIONS, 1)
            elif NEW_POSITIONS_V3 in tnew:
                tnew = tnew.replace(NEW_POSITIONS_V3, NEW_POSITIONS, 1)
            elif NEW_POSITIONS_V2 in tnew:
                tnew = tnew.replace(NEW_POSITIONS_V2, NEW_POSITIONS, 1)
            elif NEW_POSITIONS_V1 in tnew:
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
        tnew = _ensure_module_above_scene(tnew)
        tnew = _rewrite_above_gates(tnew)
        if "_marc_place_seat_people" not in tnew and "_marc_place_beside_people" in tnew:
            tnew = tnew.replace(
                "                        _marc_place_beside_people,\n"
                "                    )\n",
                "                        _marc_place_beside_people,\n"
                "                        _marc_place_seat_people,\n"
                "                    )\n",
            )
            tnew = tnew.replace(
                "                    _marc_place_beside_people(\n"
                '                        mission["id"], mission.get("objects") or [],\n'
                "                        configurations)\n",
                "                    _marc_place_seat_people(\n"
                '                        mission["id"], mission.get("objects") or [],\n'
                "                        configurations)\n"
                "                    _marc_place_beside_people(\n"
                '                        mission["id"], mission.get("objects") or [],\n'
                "                        configurations)\n",
            )
            tnew = tnew.replace(
                "            from world.object_deployer import "
                "_marc_place_beside_people\n"
                "            _marc_place_beside_people(\n"
                '                mission["id"], mission.get("objects") or [], '
                "configurations)\n"
                "        except Exception as _e:\n"
                '            log.warning(f"place-beside people failed '
                '(non-fatal): {_e}")\n',
                "            from world.object_deployer import "
                "_marc_place_beside_people\n"
                "            _marc_place_beside_people(\n"
                '                mission["id"], mission.get("objects") or [], '
                "configurations)\n"
                "        except Exception as _e:\n"
                '            log.warning(f"place-beside people failed '
                '(non-fatal): {_e}")\n'
                "        try:\n"
                "            from world.object_deployer import "
                "_marc_place_seat_people\n"
                "            _marc_place_seat_people(\n"
                '                mission["id"], mission.get("objects") or [], '
                "configurations)\n"
                "        except Exception as _e:\n"
                '            log.warning(f"place-seat people failed '
                '(non-fatal): {_e}")\n',
                1,
            )
        if tnew != tsrc:
            try:
                with open(TRAINER, "w", encoding="utf-8") as f:
                    f.write(tnew)
                n += 1
                print("[on-landmark-patch] trainer.py: place_on_landmark=True "
                      "(+ freeze climber on landmark top through settle/GT)")
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
        _no_extra_lm = (
            '    if _marc_above_scene():\n'
            '        if not placement.get("above") and not placement.get("recipe_id"):\n'
            '            on_landmark_type = None\n'
        )
        _no_extra_lm_old = (
            '    if str(__import__("os").environ.get("TRAINER_PEOPLE_ABOVE", "0"))'
            '.lower() not in (\n'
            '            "0", "false", "no", "off", ""):\n'
            '        if not placement.get("above"):\n'
            '            on_landmark_type = None\n'
        )
        if _no_extra_lm_old in dsrc and _no_extra_lm not in dsrc:
            try:
                with open(DEPLOYER, "w", encoding="utf-8") as f:
                    f.write(dsrc.replace(_no_extra_lm_old, _no_extra_lm, 1))
                print("[on-landmark-patch] deployer: recipe seat-lie keeps on_landmark")
                n += 1
            except OSError as e:
                print(f"[on-landmark-patch] deployer recipe-seat failed: {e}")
            with open(DEPLOYER, "r", encoding="utf-8") as f:
                dsrc = f.read()
        if _no_extra_lm not in dsrc:
            _anchor_lm = (
                '    on_landmark_type = placement.get("on_landmark_type") '
                'if place_on_landmark else None\n'
                '    z_offset_from_top = float(placement.get("z_offset_from_top", 0.0) or 0.0)\n'
            )
            _with_gate = (
                '    on_landmark_type = placement.get("on_landmark_type") '
                'if place_on_landmark else None\n'
                + _no_extra_lm +
                '    z_offset_from_top = float(placement.get("z_offset_from_top", 0.0) or 0.0)\n'
            )
            if _anchor_lm in dsrc:
                try:
                    with open(DEPLOYER, "w", encoding="utf-8") as f:
                        f.write(dsrc.replace(_anchor_lm, _with_gate, 1))
                    print("[on-landmark-patch] deployer: extras never inherit pose YAML landmark")
                    n += 1
                except OSError as e:
                    print(f"[on-landmark-patch] deployer extra-gate failed: {e}")
                with open(DEPLOYER, "r", encoding="utf-8") as f:
                    dsrc = f.read()
        _old_repo_lm = (
            '        on_lm = placement.get("on_landmark_type")\n'
            '        if not on_lm or placement.get("above"):\n'
            '            continue\n'
        )
        _new_repo_lm = (
            '        on_lm = placement.get("on_landmark_type")\n'
            '        if not on_lm:\n'
            '            continue\n'
        )
        if _old_repo_lm in dsrc:
            try:
                with open(DEPLOYER, "w", encoding="utf-8") as f:
                    f.write(dsrc.replace(_old_repo_lm, _new_repo_lm, 1))
                print("[on-landmark-patch] deployer: reseat climbers on landmark top")
                n += 1
            except OSError as e:
                print(f"[on-landmark-patch] deployer climber-reseat failed: {e}")
            with open(DEPLOYER, "r", encoding="utf-8") as f:
                dsrc = f.read()
        _ground_spawn = (
            '    on_landmark_type = placement.get("on_landmark_type") '
            'if place_on_landmark else None\n'
            '    if placement.get("above"):\n'
            '        on_landmark_type = None\n'
            '    z_offset_from_top = float(placement.get("z_offset_from_top", 0.0) or 0.0)\n'
        )
        _stock_spawn = (
            '    on_landmark_type = placement.get("on_landmark_type") '
            'if place_on_landmark else None\n'
            '    z_offset_from_top = float(placement.get("z_offset_from_top", 0.0) or 0.0)\n'
        )
        if _ground_spawn in dsrc:
            try:
                with open(DEPLOYER, "w", encoding="utf-8") as f:
                    f.write(dsrc.replace(_ground_spawn, _stock_spawn, 1))
                print("[on-landmark-patch] deployer: spawn climbers on landmark top")
                n += 1
            except OSError as e:
                print(f"[on-landmark-patch] deployer spawn-restore failed: {e}")

    if _ensure_above_helper(DEPLOYER):
        n += 1
    if _ensure_beside_helper(DEPLOYER):
        n += 1
    if _ensure_seat_helper(DEPLOYER):
        n += 1

    if os.path.exists(TRAINER):
        with open(TRAINER, "r", encoding="utf-8") as f:
            tsrc = f.read()
        if ("_marc_place_above_people" in tsrc
                and "_marc_place_beside_people" not in tsrc):
            _old_above_call = (
                "            from world.object_deployer import "
                "_marc_place_above_people\n"
                "            _marc_place_above_people(\n"
                '                mission["id"], mission.get("objects") or [], '
                "configurations)\n"
                "        except Exception as _e:\n"
                '            log.warning(f"place-above people failed '
                '(non-fatal): {_e}")\n'
            )
            _new_above_call = (
                _old_above_call
                + "        try:\n"
                "            from world.object_deployer import "
                "_marc_place_beside_people\n"
                "            _marc_place_beside_people(\n"
                '                mission["id"], mission.get("objects") or [], '
                "configurations)\n"
                "        except Exception as _e:\n"
                '            log.warning(f"place-beside people failed '
                '(non-fatal): {_e}")\n'
            )
            if _old_above_call in tsrc:
                try:
                    with open(TRAINER, "w", encoding="utf-8") as f:
                        f.write(tsrc.replace(_old_above_call, _new_above_call, 1))
                    print("[on-landmark-patch] trainer: beside place after above")
                    n += 1
                except OSError as e:
                    print(f"[on-landmark-patch] trainer beside-call failed: {e}")

    if os.path.exists(DEPLOYER):
        with open(DEPLOYER, "r", encoding="utf-8") as f:
            dsrc = f.read()
        dnew = _ensure_module_above_scene(dsrc)
        dnew = _rewrite_above_gates(dnew)
        if dnew != dsrc:
            try:
                with open(DEPLOYER, "w", encoding="utf-8") as f:
                    f.write(dnew)
                print("[on-landmark-patch] deployer: climb framing follows recipe.above")
                n += 1
            except OSError as e:
                print(f"[on-landmark-patch] deployer above-scene rewrite failed: {e}")

    if n:
        print("[on-landmark-patch] people with spec.placement land on that landmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
