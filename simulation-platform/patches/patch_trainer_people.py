#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - source patch: all four person identities per dataset-gen scene
# ---------------------------------------------------------------------------
# Stock _build_deploy_lists person-mode is one radio character x every pose
# (defaults to male_student_01). Female / professor / student_weight never
# appear. This replaces the combo loop with one of each catalog identity and
# a random pose (4 people/scene). TRAINER_PEOPLE_ALL_CHARS=0 restores stock.
#
# Also stamps placement.on_landmark_type onto a fraction of people so the
# on-landmark deployer patch can seat them on benches / vending machines /
# trash cans (SAR "climbed on the drink machine"). TRAINER_PEOPLE_ON_LANDMARK_PROB
# (default 0.45) is per person; TRAINER_PEOPLE_WRONG_PLACE_PROB (default 0.55)
# of those prefer a non-seat landmark. TRAINER_FORCE_CLIMB_LANDMARK_PROB
# (default 0.35) injects vending_machine or trash_can when the scene has none.
#
# boot.py runs this before dataset-gen (trainer.py). Idempotent via MARKER.
import os
import sys

TARGET = "/metacom2026/simulation_app/marc2026/trainer.py"
MARKER = "_marc_people_on_landmark"

OLD_COMBOS = '''    combos = []
    if not people_state.get("enabled", True):
        # "Add people" 해제 → 사람 미배치.
        return landmarks, objects
    if people_state.get("mode") == "pose":
        pose = pose_by_key.get(people_state.get("pose"))
        if pose:
            combos = [(pose, ch) for ch in all_chars]
    else:  # person
        ch = char_by_key.get(people_state.get("person"))
        if ch:
            combos = [(pose, ch) for pose in all_poses]
    for pose, ch in combos:
        objects.append({"type": pose, "character": ch, "count": 1})
'''

V1_COMBOS = '''    combos = []
    if not people_state.get("enabled", True):
        # "Add people" 해제 → 사람 미배치.
        return landmarks, objects
    def _marc_people_all_chars():
        """[marc-patch] One of each identity, random pose (covers female/professor/weight)."""
        raw = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
        return str(raw).strip().lower() not in ("0", "false", "no", "off")
    if _marc_people_all_chars() and all_chars and all_poses:
        for ch in all_chars:
            combos.append((random.choice(all_poses), ch))
    elif people_state.get("mode") == "pose":
        pose = pose_by_key.get(people_state.get("pose"))
        if pose:
            combos = [(pose, ch) for ch in all_chars]
    else:
        ch = char_by_key.get(people_state.get("person"))
        if ch:
            combos = [(pose, ch) for pose in all_poses]
    for pose, ch in combos:
        objects.append({"type": pose, "character": ch, "count": 1})
'''

NEW_COMBOS = '''    combos = []
    if not people_state.get("enabled", True):
        # "Add people" 해제 → 사람 미배치.
        return landmarks, objects
    def _marc_people_all_chars():
        """[marc-patch] One of each identity, random pose (covers female/professor/weight)."""
        raw = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
        return str(raw).strip().lower() not in ("0", "false", "no", "off")
    def _marc_frac(name, default):
        """[marc-patch] Clamp an env probability into [0, 1]."""
        try:
            v = float(os.environ.get(name, str(default)) or str(default))
        except (TypeError, ValueError):
            v = float(default)
        return max(0.0, min(1.0, v))
    def _marc_people_on_landmark():
        """[marc-patch] Marker: people may sit/climb on landmark tops."""
        return True
    def _marc_lm_typ(spec):
        import re as _re_lm
        return _re_lm.sub(r"_[0-9]+$", "", str((spec or {}).get("name") or ""))
    if _marc_people_all_chars() and all_chars and all_poses:
        for ch in all_chars:
            combos.append((random.choice(all_poses), ch))
    elif people_state.get("mode") == "pose":
        pose = pose_by_key.get(people_state.get("pose"))
        if pose:
            combos = [(pose, ch) for ch in all_chars]
    else:
        ch = char_by_key.get(people_state.get("person"))
        if ch:
            combos = [(pose, ch) for pose in all_poses]
    _marc_people_on_landmark()
    _SEAT = ("bench", "picnic_table")
    _CLIMB = ("vending_machine", "trash_can", "hydrant", "postbox")
    _VEH = ("police_car", "normal_car_1", "normal_car_2",
            "sports_car", "sports_car_2", "suv", "taxi")
    _WRONG = _CLIMB + _VEH
    _ON_WRONG_POSES = ("person_sitting", "person_crouching", "person_reaching")
    _ON_SEAT_POSES = ("person_sitting", "person_reclining", "person_lying_down")
    if random.random() < _marc_frac("TRAINER_FORCE_CLIMB_LANDMARK_PROB", 0.35):
        _have = {_marc_lm_typ(s) for s in landmarks}
        if not (_have & set(_CLIMB)):
            landmarks.append({"name": "%s_0" % random.choice(_CLIMB)})
    _lm_types = []
    for _s in landmarks:
        _t = _marc_lm_typ(_s)
        if _t and _t not in _lm_types:
            _lm_types.append(_t)
    _seats = [t for t in _lm_types if t in _SEAT]
    _wrongs = [t for t in _lm_types if t in _WRONG]
    _pose_set = set(all_poses)
    for pose, ch in combos:
        spec = {"type": pose, "character": ch, "count": 1}
        if random.random() < _marc_frac("TRAINER_PEOPLE_ON_LANDMARK_PROB", 0.45):
            want_wrong = random.random() < _marc_frac(
                "TRAINER_PEOPLE_WRONG_PLACE_PROB", 0.55)
            lm_typ = None
            if want_wrong and _wrongs:
                lm_typ = random.choice(_wrongs)
            elif _seats:
                lm_typ = random.choice(_seats)
            elif _wrongs:
                lm_typ = random.choice(_wrongs)
            if lm_typ:
                spec["placement"] = {
                    "on_landmark_type": lm_typ,
                    "z_offset_from_top": 0.02,
                }
                pool = _ON_WRONG_POSES if lm_typ in _WRONG else _ON_SEAT_POSES
                fit = [p for p in pool if p in _pose_set]
                if fit:
                    spec["type"] = random.choice(fit)
        objects.append(spec)
'''


def _write(src):
    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    except OSError as e:
        print(f"[people-patch] write failed, skipping: {e}")
        return False


def main():
    if not os.path.exists(TARGET):
        print(f"[people-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[people-patch] already applied, skipping")
        return 0

    if V1_COMBOS in src:
        src = src.replace(V1_COMBOS, NEW_COMBOS, 1)
        if not _write(src):
            return 0
        print("[people-patch] upgraded: identities + on-landmark seating")
        return 0

    if OLD_COMBOS not in src:
        print("[people-patch] people-combo anchor not found, skipping - "
              "trainer.py may have changed upstream")
        return 0

    src = src.replace(OLD_COMBOS, NEW_COMBOS, 1)
    if not _write(src):
        return 0

    mode = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
    on_p = os.environ.get("TRAINER_PEOPLE_ON_LANDMARK_PROB", "0.45")
    print(f"[people-patch] all identities per scene "
          f"(TRAINER_PEOPLE_ALL_CHARS={mode}); on-landmark p={on_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
