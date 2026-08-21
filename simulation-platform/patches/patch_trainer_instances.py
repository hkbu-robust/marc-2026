#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - source patch: 4-6 instances of each selected class per dataset-gen scene
# ---------------------------------------------------------------------------
# trainer.py _build_catalog() hardcodes lost-object count:1 and one landmark
# named <type>_0. deploy_mission_objects() already loops on count, and
# deploy_landmarks() already looks up assets by name prefix (bench_3 → bench).
# This patch rewrites _build_deploy_lists() so each checked object type gets
# count in [TRAINER_MIN_INSTANCES, TRAINER_MAX_INSTANCES] (default 4-6) and each
# checked landmark type is expanded to <type>_0 .. <type>_{n-1}. People stay at
# one copy. Set both env vars to 1 to restore the stock catalogue.
#
# boot.py runs this before dataset-gen (trainer.py). Idempotent via MARKER.
import os
import sys

TARGET = "/metacom2026/simulation_app/marc2026/trainer.py"
MARKER = "_marc_instance_count"

OLD_LISTS = '''    objects = [dict(it["spec"]) for it in catalog["objects"]
               if it["key"] in selected_keys]
    landmarks = [dict(it["spec"]) for it in catalog["landmarks"]
                 if it["key"] in selected_keys]
'''

NEW_LISTS = '''    objects = []
    for it in catalog["objects"]:
        if it["key"] not in selected_keys:
            continue
        spec = dict(it["spec"])
        spec["count"] = _marc_instance_count()
        objects.append(spec)
    landmarks = []
    import re as _re_inst
    for it in catalog["landmarks"]:
        if it["key"] not in selected_keys:
            continue
        raw = (it.get("spec") or {}).get("name") or it.get("label") or "lm"
        typ = _re_inst.sub(r"_\d+$", "", str(raw))
        n = _marc_instance_count()
        for i in range(n):
            landmarks.append({"name": f"{typ}_{i}"})
'''

HELPER = '''
def _marc_instance_count():
    """[marc-patch] Copies of each selected lost-object / landmark class per scene."""
    try:
        lo = int(os.environ.get("TRAINER_MIN_INSTANCES", "4") or "4")
    except (TypeError, ValueError):
        lo = 4
    try:
        hi = int(os.environ.get("TRAINER_MAX_INSTANCES", "6") or "6")
    except (TypeError, ValueError):
        hi = 6
    lo = max(1, lo)
    hi = max(lo, hi)
    return random.randint(lo, hi)

'''


def main():
    if not os.path.exists(TARGET):
        print(f"[instances-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[instances-patch] already applied, skipping")
        return 0

    if OLD_LISTS not in src:
        print("[instances-patch] deploy-list anchor not found, skipping - "
              "trainer.py may have changed upstream")
        return 0

    insert_at = src.find("\ndef _build_deploy_lists(")
    if insert_at < 0:
        print("[instances-patch] _build_deploy_lists not found, skipping")
        return 0

    src = src[:insert_at] + HELPER + src[insert_at:]
    src = src.replace(OLD_LISTS, NEW_LISTS, 1)
    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
    except OSError as e:
        print(f"[instances-patch] write failed, skipping: {e}")
        return 0

    lo = os.environ.get("TRAINER_MIN_INSTANCES", "4")
    hi = os.environ.get("TRAINER_MAX_INSTANCES", "6")
    print(f"[instances-patch] each selected class spawns {lo}-{hi} instances "
          f"per scene (objects via count, landmarks via <type>_N)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
