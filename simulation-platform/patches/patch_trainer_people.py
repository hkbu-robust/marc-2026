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
# boot.py runs this before dataset-gen (trainer.py). Idempotent via MARKER.
import os
import sys

TARGET = "/metacom2026/simulation_app/marc2026/trainer.py"
MARKER = "_marc_people_all_chars"

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

NEW_COMBOS = '''    combos = []
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


def main():
    if not os.path.exists(TARGET):
        print(f"[people-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[people-patch] already applied, skipping")
        return 0

    if OLD_COMBOS not in src:
        print("[people-patch] people-combo anchor not found, skipping - "
              "trainer.py may have changed upstream")
        return 0

    src = src.replace(OLD_COMBOS, NEW_COMBOS, 1)
    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
    except OSError as e:
        print(f"[people-patch] write failed, skipping: {e}")
        return 0

    mode = os.environ.get("TRAINER_PEOPLE_ALL_CHARS", "1")
    print(f"[people-patch] all identities per scene (TRAINER_PEOPLE_ALL_CHARS={mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
