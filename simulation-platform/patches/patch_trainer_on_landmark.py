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

NEW_SNAP = '''        _snap_landmarks_to_ground([mission])
        try:
            from world.object_deployer import reposition_on_landmark_people
            reposition_on_landmark_people(
                mission["id"], mission.get("objects") or [], configurations)
        except Exception as _e:
            log.warning(f"reposition on-landmark people failed (non-fatal): {_e}")
'''


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
        if OLD_SNAP in tnew and "reposition_on_landmark_people" not in tnew:
            tnew = tnew.replace(OLD_SNAP, NEW_SNAP, 1)
        if tnew != tsrc:
            try:
                with open(TRAINER, "w", encoding="utf-8") as f:
                    f.write(tnew)
                n += 1
                print("[on-landmark-patch] trainer.py: place_on_landmark=True "
                      "(+ re-seat after ground snap)")
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

    if n:
        print("[on-landmark-patch] people with spec.placement land on that landmark")
    return 0


if __name__ == "__main__":
    sys.exit(main())
