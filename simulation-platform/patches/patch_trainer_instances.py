#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# MARC 2026 - source patch: per-scene instance counts for dataset-gen
# ---------------------------------------------------------------------------
# trainer.py _build_catalog() hardcodes lost-object count:1 and one landmark
# named <type>_0. deploy_mission_objects() already loops on count, and
# deploy_landmarks() already looks up assets by name prefix (bench_3 → bench).
#
# This patch rewrites _build_deploy_lists() so:
#   - each checked object type gets count in
#     [TRAINER_MIN_INSTANCES, TRAINER_MAX_INSTANCES] (default 4-6)
#   - each kept non-vehicle landmark type is expanded to <type>_0 .. <type>_{n-1}
#     with n in [TRAINER_LANDMARK_MIN_INSTANCES, TRAINER_LANDMARK_MAX_INSTANCES]
#     (default 1-2)
#   - cars / SUVs / taxis use a tighter cap: at most
#     TRAINER_VEHICLE_MAX_TYPES types per scene (default 1), each with
#     [TRAINER_VEHICLE_MIN_INSTANCES, TRAINER_VEHICLE_MAX_INSTANCES] copies
#     (default 1-1), kept with TRAINER_VEHICLE_INCLUSION_PROBABILITY (default 0.15)
#   - TRAINER_CLASS_INCLUSION_PROBABILITY (default 0.35) independently drops
#     non-vehicle landmark types so each scene is a subset. If every type is
#     dropped, one checked non-vehicle (else a vehicle) is kept as an anchor.
#   - TRAINER_SKIP_CLASSES (default umbrella,long_umbrella) drops those lost-object
#     types entirely, including any catalogue name containing "umbrella".
# People stay at one copy.
#
# boot.py runs this before dataset-gen (trainer.py). Idempotent via MARKER.
# Replaces older injected helpers if those are already in trainer.py.
import os
import sys

TARGET = "/metacom2026/simulation_app/marc2026/trainer.py"
MARKER = "_marc_skip_object"

OLD_LISTS = '''    objects = [dict(it["spec"]) for it in catalog["objects"]
               if it["key"] in selected_keys]
    landmarks = [dict(it["spec"]) for it in catalog["landmarks"]
                 if it["key"] in selected_keys]
'''

V1_HELPER = '''
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

V1_LISTS = '''    objects = []
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
        typ = _re_inst.sub(r"_\\d+$", "", str(raw))
        n = _marc_instance_count()
        for i in range(n):
            landmarks.append({"name": f"{typ}_{i}"})
'''

V2_HELPER = '''
def _marc_instance_count(lo_name, hi_name, lo_default, hi_default):
    """[marc-patch] Inclusive random copy count from a pair of env bounds."""
    try:
        lo = int(os.environ.get(lo_name, str(lo_default)) or str(lo_default))
    except (TypeError, ValueError):
        lo = lo_default
    try:
        hi = int(os.environ.get(hi_name, str(hi_default)) or str(hi_default))
    except (TypeError, ValueError):
        hi = hi_default
    lo = max(1, lo)
    hi = max(lo, hi)
    return random.randint(lo, hi)


def _marc_include_landmark():
    """[marc-patch] Keep this landmark type in the scene with the configured probability."""
    try:
        p = float(os.environ.get("TRAINER_CLASS_INCLUSION_PROBABILITY", "1") or "1")
    except (TypeError, ValueError):
        p = 1.0
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    return random.random() < p

'''

V2_LISTS = '''    objects = []
    for it in catalog["objects"]:
        if it["key"] not in selected_keys:
            continue
        spec = dict(it["spec"])
        spec["count"] = _marc_instance_count(
            "TRAINER_MIN_INSTANCES", "TRAINER_MAX_INSTANCES", 4, 6)
        objects.append(spec)
    landmarks = []
    import re as _re_inst
    _lm_candidates = []
    for it in catalog["landmarks"]:
        if it["key"] not in selected_keys:
            continue
        _lm_candidates.append(it)
    _lm_kept = [it for it in _lm_candidates if _marc_include_landmark()]
    if _lm_candidates and not _lm_kept:
        _lm_kept = [random.choice(_lm_candidates)]
    for it in _lm_kept:
        raw = (it.get("spec") or {}).get("name") or it.get("label") or "lm"
        typ = _re_inst.sub(r"_\\d+$", "", str(raw))
        n = _marc_instance_count(
            "TRAINER_LANDMARK_MIN_INSTANCES", "TRAINER_LANDMARK_MAX_INSTANCES", 1, 2)
        for i in range(n):
            landmarks.append({"name": f"{typ}_{i}"})
'''

V3_LISTS = '''    objects = []
    for it in catalog["objects"]:
        if it["key"] not in selected_keys:
            continue
        spec = dict(it["spec"])
        spec["count"] = _marc_instance_count(
            "TRAINER_MIN_INSTANCES", "TRAINER_MAX_INSTANCES", 4, 6)
        objects.append(spec)
    landmarks = []
    _lm_other, _lm_vehicles = [], []
    for it in catalog["landmarks"]:
        if it["key"] not in selected_keys:
            continue
        typ = _marc_landmark_typ(it)
        if _marc_is_vehicle(typ):
            _lm_vehicles.append((it, typ))
        else:
            _lm_other.append((it, typ))
    _kept = []
    for it, typ in _lm_other:
        if _marc_include_landmark():
            _kept.append((it, typ, False))
    _veh_kept = [(it, typ) for it, typ in _lm_vehicles if _marc_include_vehicle()]
    _vmax = _marc_vehicle_max_types()
    if _vmax <= 0:
        _veh_kept = []
    elif len(_veh_kept) > _vmax:
        _veh_kept = random.sample(_veh_kept, _vmax)
    for it, typ in _veh_kept:
        _kept.append((it, typ, True))
    if not _kept:
        _pool = _lm_other or _lm_vehicles
        if _pool:
            it, typ = random.choice(_pool)
            _kept.append((it, typ, _marc_is_vehicle(typ)))
    for it, typ, is_veh in _kept:
        if is_veh:
            n = _marc_instance_count(
                "TRAINER_VEHICLE_MIN_INSTANCES",
                "TRAINER_VEHICLE_MAX_INSTANCES", 1, 1)
        else:
            n = _marc_instance_count(
                "TRAINER_LANDMARK_MIN_INSTANCES",
                "TRAINER_LANDMARK_MAX_INSTANCES", 1, 2)
        for i in range(n):
            landmarks.append({"name": f"{typ}_{i}"})
'''

NEW_LISTS = '''    objects = []
    for it in catalog["objects"]:
        if it["key"] not in selected_keys:
            continue
        if _marc_skip_object(it):
            continue
        spec = dict(it["spec"])
        spec["count"] = _marc_instance_count(
            "TRAINER_MIN_INSTANCES", "TRAINER_MAX_INSTANCES", 4, 6)
        objects.append(spec)
    landmarks = []
    _lm_other, _lm_vehicles = [], []
    for it in catalog["landmarks"]:
        if it["key"] not in selected_keys:
            continue
        typ = _marc_landmark_typ(it)
        if _marc_is_vehicle(typ):
            _lm_vehicles.append((it, typ))
        else:
            _lm_other.append((it, typ))
    _kept = []
    for it, typ in _lm_other:
        if _marc_include_landmark():
            _kept.append((it, typ, False))
    _veh_kept = [(it, typ) for it, typ in _lm_vehicles if _marc_include_vehicle()]
    _vmax = _marc_vehicle_max_types()
    if _vmax <= 0:
        _veh_kept = []
    elif len(_veh_kept) > _vmax:
        _veh_kept = random.sample(_veh_kept, _vmax)
    for it, typ in _veh_kept:
        _kept.append((it, typ, True))
    if not _kept:
        _pool = _lm_other or _lm_vehicles
        if _pool:
            it, typ = random.choice(_pool)
            _kept.append((it, typ, _marc_is_vehicle(typ)))
    for it, typ, is_veh in _kept:
        if is_veh:
            n = _marc_instance_count(
                "TRAINER_VEHICLE_MIN_INSTANCES",
                "TRAINER_VEHICLE_MAX_INSTANCES", 1, 1)
        else:
            n = _marc_instance_count(
                "TRAINER_LANDMARK_MIN_INSTANCES",
                "TRAINER_LANDMARK_MAX_INSTANCES", 1, 2)
        for i in range(n):
            landmarks.append({"name": f"{typ}_{i}"})
'''

V3_HELPER = '''
_MARC_VEHICLE_TYPES = frozenset({
    "police_car", "normal_car_1", "normal_car_2",
    "sports_car", "sports_car_2", "suv", "taxi",
})


def _marc_instance_count(lo_name, hi_name, lo_default, hi_default):
    """[marc-patch] Inclusive random copy count from a pair of env bounds."""
    try:
        lo = int(os.environ.get(lo_name, str(lo_default)) or str(lo_default))
    except (TypeError, ValueError):
        lo = lo_default
    try:
        hi = int(os.environ.get(hi_name, str(hi_default)) or str(hi_default))
    except (TypeError, ValueError):
        hi = hi_default
    lo = max(1, lo)
    hi = max(lo, hi)
    return random.randint(lo, hi)


def _marc_include_prob(name, default):
    try:
        p = float(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        p = float(default)
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    return random.random() < p


def _marc_include_landmark():
    """[marc-patch] Keep this non-vehicle landmark type this scene."""
    return _marc_include_prob("TRAINER_CLASS_INCLUSION_PROBABILITY", 0.35)


def _marc_include_vehicle():
    """[marc-patch] Cars/SUVs/taxis are opted in rarely; they fill the frustum."""
    return _marc_include_prob("TRAINER_VEHICLE_INCLUSION_PROBABILITY", 0.15)


def _marc_vehicle_max_types():
    try:
        n = int(os.environ.get("TRAINER_VEHICLE_MAX_TYPES", "1") or "1")
    except (TypeError, ValueError):
        n = 1
    return max(0, n)


def _marc_landmark_typ(it):
    import re as _re_inst
    raw = (it.get("spec") or {}).get("name") or it.get("label") or "lm"
    return _re_inst.sub(r"_\\d+$", "", str(raw))


def _marc_is_vehicle(typ):
    """[marc-patch] Huge drivable landmarks that bury small objects."""
    t = str(typ).lower()
    if t in _MARC_VEHICLE_TYPES:
        return True
    return ("car" in t) or t in ("suv", "taxi")

'''

HELPER = '''
_MARC_VEHICLE_TYPES = frozenset({
    "police_car", "normal_car_1", "normal_car_2",
    "sports_car", "sports_car_2", "suv", "taxi",
})


def _marc_instance_count(lo_name, hi_name, lo_default, hi_default):
    """[marc-patch] Inclusive random copy count from a pair of env bounds."""
    try:
        lo = int(os.environ.get(lo_name, str(lo_default)) or str(lo_default))
    except (TypeError, ValueError):
        lo = lo_default
    try:
        hi = int(os.environ.get(hi_name, str(hi_default)) or str(hi_default))
    except (TypeError, ValueError):
        hi = hi_default
    lo = max(1, lo)
    hi = max(lo, hi)
    return random.randint(lo, hi)


def _marc_include_prob(name, default):
    try:
        p = float(os.environ.get(name, str(default)) or str(default))
    except (TypeError, ValueError):
        p = float(default)
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    return random.random() < p


def _marc_include_landmark():
    """[marc-patch] Keep this non-vehicle landmark type this scene."""
    return _marc_include_prob("TRAINER_CLASS_INCLUSION_PROBABILITY", 0.35)


def _marc_include_vehicle():
    """[marc-patch] Cars/SUVs/taxis are opted in rarely; they fill the frustum."""
    return _marc_include_prob("TRAINER_VEHICLE_INCLUSION_PROBABILITY", 0.15)


def _marc_vehicle_max_types():
    try:
        n = int(os.environ.get("TRAINER_VEHICLE_MAX_TYPES", "1") or "1")
    except (TypeError, ValueError):
        n = 1
    return max(0, n)


def _marc_landmark_typ(it):
    import re as _re_inst
    raw = (it.get("spec") or {}).get("name") or it.get("label") or "lm"
    return _re_inst.sub(r"_\\d+$", "", str(raw))


def _marc_is_vehicle(typ):
    """[marc-patch] Huge drivable landmarks that bury small objects."""
    t = str(typ).lower()
    if t in _MARC_VEHICLE_TYPES:
        return True
    return ("car" in t) or t in ("suv", "taxi")


def _marc_skip_object(it):
    """[marc-patch] Drop umbrella / long_umbrella (and any *umbrella* name)."""
    raw = os.environ.get("TRAINER_SKIP_CLASSES", "umbrella,long_umbrella")
    skip = {n.strip().lower() for n in raw.split(",") if n.strip()}
    spec = it.get("spec") or {}
    name = str(spec.get("name") or spec.get("type") or it.get("label") or it.get("key") or "").lower()
    key = str(it.get("key") or "").lower()
    blob = name + " " + key
    if any(s == name or s == key for s in skip):
        return True
    if "umbrella" in blob and any("umbrella" in s for s in skip):
        return True
    return False

'''


def _log_bounds():
    olo = os.environ.get("TRAINER_MIN_INSTANCES", "4")
    ohi = os.environ.get("TRAINER_MAX_INSTANCES", "6")
    llo = os.environ.get("TRAINER_LANDMARK_MIN_INSTANCES", "1")
    lhi = os.environ.get("TRAINER_LANDMARK_MAX_INSTANCES", "2")
    p = os.environ.get("TRAINER_CLASS_INCLUSION_PROBABILITY", "0.35")
    vlo = os.environ.get("TRAINER_VEHICLE_MIN_INSTANCES", "1")
    vhi = os.environ.get("TRAINER_VEHICLE_MAX_INSTANCES", "1")
    vp = os.environ.get("TRAINER_VEHICLE_INCLUSION_PROBABILITY", "0.15")
    vt = os.environ.get("TRAINER_VEHICLE_MAX_TYPES", "1")
    skip = os.environ.get("TRAINER_SKIP_CLASSES", "umbrella,long_umbrella")
    print(f"[instances-patch] objects {olo}-{ohi}; landmarks {llo}-{lhi} p={p}; "
          f"vehicles {vlo}-{vhi} p={vp} max_types={vt}; skip={skip}")


def _write(src):
    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    except OSError as e:
        print(f"[instances-patch] write failed, skipping: {e}")
        return False


def main():
    if not os.path.exists(TARGET):
        print(f"[instances-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        print("[instances-patch] already applied, skipping")
        return 0

    if V3_HELPER in src and V3_LISTS in src:
        src = src.replace(V3_HELPER, HELPER, 1).replace(V3_LISTS, NEW_LISTS, 1)
        if not _write(src):
            return 0
        print("[instances-patch] upgraded: umbrellas dropped from objects")
        _log_bounds()
        return 0

    if V2_HELPER in src and V2_LISTS in src:
        src = src.replace(V2_HELPER, HELPER, 1).replace(V2_LISTS, NEW_LISTS, 1)
        if not _write(src):
            return 0
        print("[instances-patch] upgraded: vehicles capped separately")
        _log_bounds()
        return 0

    if V1_HELPER in src and V1_LISTS in src:
        src = src.replace(V1_HELPER, HELPER, 1).replace(V1_LISTS, NEW_LISTS, 1)
        if not _write(src):
            return 0
        print("[instances-patch] upgraded from single-range instance counts")
        _log_bounds()
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
    if not _write(src):
        return 0

    _log_bounds()
    return 0


if __name__ == "__main__":
    sys.exit(main())
