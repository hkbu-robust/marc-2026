#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - source patch: honor TRAINER_AUTO_SCENES without the stock 200 cap
# ---------------------------------------------------------------------------
# trainer.py _resolve_auto_scenes() does min(n, _AUTO_SCENES_MAX) with
# _AUTO_SCENES_MAX = 200, so TRAINER_AUTO_SCENES=4000 generates 200 and exits.
# This replaces the constant so the requested count is used. Optional
# TRAINER_AUTO_SCENES_MAX>0 restores a safety ceiling (old behaviour: 200).
#
# boot.py runs this before dataset-gen (trainer.py). Idempotent via MARKER.
import os
import sys

TARGET = "/metacom2026/simulation_app/marc2026/trainer.py"
MARKER = "_marc_auto_scenes_max"

OLD_CONST = "_AUTO_SCENES_MAX = 200\n"

NEW_CONST = '''def _marc_auto_scenes_max():
    """[marc-patch] Optional per-run cap. 0 (default) = honor TRAINER_AUTO_SCENES."""
    try:
        return max(0, int(os.environ.get("TRAINER_AUTO_SCENES_MAX", "0") or "0"))
    except ValueError:
        return 0

_AUTO_SCENES_MAX = _marc_auto_scenes_max() or 10 ** 9
'''

OLD_RESOLVE = "    return min(n, _AUTO_SCENES_MAX), n\n"

NEW_RESOLVE = '''    cap = _marc_auto_scenes_max()
    if cap > 0:
        return min(n, cap), n
    return n, n
'''


def main():
    if not os.path.exists(TARGET):
        print(f"[auto-scenes-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[auto-scenes-patch] already applied, skipping")
        return 0

    if OLD_CONST not in src or OLD_RESOLVE not in src:
        print("[auto-scenes-patch] anchors not found, skipping - trainer.py may have changed")
        return 0

    src = src.replace(OLD_CONST, NEW_CONST, 1)
    src = src.replace(OLD_RESOLVE, NEW_RESOLVE, 1)
    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
    except OSError as e:
        print(f"[auto-scenes-patch] write failed, skipping: {e}")
        return 0

    cap = os.environ.get("TRAINER_AUTO_SCENES_MAX", "0")
    req = os.environ.get("TRAINER_AUTO_SCENES", "0")
    print(f"[auto-scenes-patch] TRAINER_AUTO_SCENES={req} (cap={cap or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
