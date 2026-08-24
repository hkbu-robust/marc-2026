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
#   4. if placement.above is set, aligns world bounding boxes after ground
#      snap: person XY center to landmark XY center, and person bottom to
#      landmark top plus a small clearance. This is pose/asset-height aware.
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
        if not on_lm:
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

NEW_SNAP = '''        _snap_landmarks_to_ground([mission])
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

NEW_SAVE = '''        active_payload = gt_by_mission.get(active_mission_id)
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
                and isinstance(d.get("bbox"), (list, tuple))
                and len(d.get("bbox") or []) >= 4
            ]
            _landmarks = [
                d for d in _dets
                if isinstance(d.get("bbox"), (list, tuple))
                and len(d.get("bbox") or []) >= 4
                and (
                    str(d.get("class") or "") in _want
                    if _want else d.get("kind") == "landmark"
                )
            ]

            def _bbox4(det):
                b = det.get("bbox") or []
                x0, y0, x1, y1 = (
                    float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
                return x0, y0, x1, y1

            def _stacked(person, lm):
                px0, py0, px1, py1 = _bbox4(person)
                lx0, ly0, lx1, ly1 = _bbox4(lm)
                overlap_x = min(px1, lx1) - max(px0, lx0)
                min_w = min(px1 - px0, lx1 - lx0)
                if min_w <= 1.0 or overlap_x < 0.4 * min_w:
                    return False
                if 0.5 * (py0 + py1) >= 0.5 * (ly0 + ly1):
                    return False
                # Climb targets are non-seat: feet must sit on the body, not
                # share the ground line beside the landmark.
                return (ly1 - py1) >= 0.22 * (ly1 - ly0)

            _active_valid = any(
                _stacked(p, lm) for p in _people for lm in _landmarks
            )
            if not _active_valid:
                log.warning(
                    f"Above scene {scene_id} skipped: no visible "
                    "person-on-landmark pair")
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
                    log.warning(
                        f"Scene {trainer_state.scene_id} not saved "
                        f"({auto_remaining} on-landmark scenes remain)")
                    if _auto_failed_scenes >= _auto_failure_limit:
                        log.error(
                            f"Stopping after {_auto_failed_scenes} consecutive "
                            "scenes without a visible person-on-landmark pair")
                        break
            except Exception as e:
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

ABOVE_HELPER = r'''

def _marc_place_above_people(mission_id, objects, configurations=None):
    """[marc-patch] Pose-aware world-bounds contact for abnormal climbers."""
    jobs = []
    for spec in objects or []:
        pl = spec.get("placement") or {}
        if not pl.get("above"):
            continue
        try:
            clearance = float(pl.get("contact_clearance", 0.01) or 0.01)
        except (TypeError, ValueError):
            clearance = 0.01
        jobs.append((
            spec, str(pl.get("on_landmark_type") or ""), clearance))
    if not jobs:
        return
    try:
        import omni.usd
        from pxr import Usd, UsdGeom, Gf
    except Exception:
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    def _world_xyz(prim):
        xf = UsdGeom.Xformable(prim)
        if not xf:
            return None
        t = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        return [float(t[0]), float(t[1]), float(t[2])]

    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_],
        useExtentsHint=False,
    )

    def _world_range(prim):
        try:
            r = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
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

    def _roots_matching(scope, *tokens):
        tokens = tuple(str(t).strip() for t in tokens if str(t).strip())
        if not tokens:
            return []
        root = stage.GetPrimAtPath(scope)
        if not root or not root.IsValid():
            return []
        hits = []
        for prim in Usd.PrimRange(root):
            if prim == root or not prim.IsValid() or not prim.IsActive():
                continue
            if all(token in prim.GetName() for token in tokens):
                hits.append(prim)
        hits.sort(key=lambda p: (len(str(p.GetPath())), str(p.GetPath())))
        roots, seen = [], []
        for prim in hits:
            path = str(prim.GetPath())
            if any(path.startswith(parent + "/") for parent in seen):
                continue
            roots.append(prim)
            seen.append(path)
        return roots

    used_people, used_lms = set(), set()
    n = 0
    for spec, typ, clearance in jobs:
        lms = [
            p for p in _roots_matching("/world/landmarks", typ)
            if str(p.GetPath()) not in used_lms
        ]
        pose = str(spec.get("type") or "")
        ch = str(spec.get("character") or "")
        people = [
            p for p in _roots_matching("/world/people", pose, ch)
            if str(p.GetPath()) not in used_people
        ]
        if not lms or not people:
            continue
        lm, person = lms[0], people[0]
        person_range = _world_range(person)
        landmark_range = _world_range(lm)
        person_xyz = _world_xyz(person)
        if person_range is None or landmark_range is None or person_xyz is None:
            continue
        p_lo, p_hi = person_range
        l_lo, l_hi = landmark_range
        p_cx = 0.5 * (p_lo[0] + p_hi[0])
        p_cy = 0.5 * (p_lo[1] + p_hi[1])
        l_cx = 0.5 * (l_lo[0] + l_hi[0])
        l_cy = 0.5 * (l_lo[1] + l_hi[1])
        desired = [
            person_xyz[0] + (l_cx - p_cx),
            person_xyz[1] + (l_cy - p_cy),
            person_xyz[2] + (l_hi[2] + clearance - p_lo[2]),
        ]
        _set_xyz(person, desired)
        used_lms.add(str(lm.GetPath()))
        used_people.add(str(person.GetPath()))
        n += 1
    print(
        "[on-landmark-patch] above-mode: placed %d/%d person(s) "
        "with bounds contact" % (n, len(jobs))
    )
'''


def _ensure_above_helper(path):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if "_marc_place_above_people" in src:
        return False
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src.rstrip() + ABOVE_HELPER)
    except OSError as e:
        print(f"[on-landmark-patch] above-helper write failed: {e}")
        return False
    print("[on-landmark-patch] deployer: above-mode helper appended")
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
        if (NEW_SNAP_V2 in tnew and
                "TRAINER_RENDER_WARMUP_FRAMES" not in tnew):
            tnew = tnew.replace(NEW_SNAP_V2, NEW_SNAP, 1)
        elif NEW_SNAP_V1 in tnew and "_marc_place_above_people" not in tnew:
            tnew = tnew.replace(NEW_SNAP_V1, NEW_SNAP, 1)
        elif OLD_SNAP in tnew and "reposition_on_landmark_people" not in tnew:
            tnew = tnew.replace(OLD_SNAP, NEW_SNAP, 1)
        if OLD_SAVE in tnew:
            tnew = tnew.replace(OLD_SAVE, NEW_SAVE, 1)
        if (OLD_CAPTURE_RETURN in tnew
                and "return _active_valid" not in tnew
                and "_active_valid = True" in tnew):
            tnew = tnew.replace(
                OLD_CAPTURE_RETURN, NEW_CAPTURE_RETURN, 1)
        if OLD_INITIAL_WARN in tnew:
            tnew = tnew.replace(OLD_INITIAL_WARN, NEW_INITIAL_WARN, 1)
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
                      "(+ bounds contact, warmup; save/count on-landmark only)")
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

    if _ensure_above_helper(DEPLOYER):
        n += 1

    if n:
        print("[on-landmark-patch] people with spec.placement land on that landmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
