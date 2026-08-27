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
#   - TRAINER_SKIP_CLASSES (default long_umbrella,master_chef_can,wood_block)
#     drops those lost-object types by exact catalog name.
#     closedlongumbrella is treated as the same long mesh. Ordinary umbrella
#     is kept. TRAINER_YOLO_MODE=1 also injects every non-skipped catalog
#     object the GUI did not check, then force-adds TRAINER_YOLO_FORCE_CLASSES
#     (default umbrella) from mission_object_assets when those keys are absent.
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
    if _marc_yolo_inject():
        _have = set()
        for s in objects:
            _have.add(str(s.get("name") or "").lower())
            _have.add(str(s.get("type") or "").lower())
            _have.add(str(s.get("key") or "").lower())
        _have.discard("")
        for it in catalog["objects"]:
            if _marc_skip_object(it):
                continue
            spec0 = it.get("spec") or {}
            name = str(
                spec0.get("name") or spec0.get("type") or it.get("key") or ""
            ).lower()
            key = str(it.get("key") or "").lower()
            if name in _have or key in _have:
                continue
            spec = dict(spec0)
            spec["count"] = _marc_instance_count(
                "TRAINER_MIN_INSTANCES", "TRAINER_MAX_INSTANCES", 4, 6)
            objects.append(spec)
            _have.add(name)
            _have.add(key)
        for _want in _marc_yolo_force_types():
            if _want in _have:
                continue
            _fake = {"key": _want, "spec": {"name": _want, "type": _want}}
            if _marc_skip_object(_fake):
                continue
            _spec = _marc_yolo_force_spec(_want, catalog)
            if _spec is None:
                continue
            _spec["count"] = _marc_instance_count(
                "TRAINER_MIN_INSTANCES", "TRAINER_MAX_INSTANCES", 4, 6)
            objects.append(_spec)
            _have.add(_want)
            _have.add(str(_spec.get("type") or "").lower())
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
    """[marc-patch] Drop TRAINER_SKIP_CLASSES by exact name (organizer exclusions)."""
    raw = os.environ.get(
        "TRAINER_SKIP_CLASSES",
        "long_umbrella,master_chef_can,wood_block")
    skip = {n.strip().lower() for n in raw.split(",") if n.strip()}
    if "long_umbrella" in skip or "closedlongumbrella" in skip:
        skip.update(("long_umbrella", "closedlongumbrella"))
    if "wood_block" in skip or "wooden_block" in skip or "woodblock" in skip:
        skip.update(("wood_block", "wooden_block", "woodblock"))
    if "master_chef_can" in skip or "chef_can" in skip or "masterchefcan" in skip:
        skip.update(("master_chef_can", "chef_can", "masterchefcan"))
    spec = it.get("spec") or {}
    tokens = {
        str(spec.get("name") or "").lower(),
        str(spec.get("type") or "").lower(),
        str(it.get("label") or "").lower(),
        str(it.get("key") or "").lower(),
    }
    tokens.discard("")
    return any(token in skip for token in tokens)


def _marc_yolo_inject():
    """[marc-patch] Inject unchecked catalog objects when TRAINER_YOLO_MODE=1.

    Named apart from the people patch's nested _marc_yolo_mode() inside
    _build_deploy_lists; calling that name here is an UnboundLocalError.
    """
    raw = os.environ.get("TRAINER_YOLO_MODE", "0")
    return str(raw).strip().lower() not in ("0", "false", "no", "off", "")


_MARC_YOLO_FORCE_DEFAULT = ("umbrella",)
_MARC_YOLO_FORCE_ALIASES = {
    "wooden_block": "wood_block",
    "woodblock": "wood_block",
    "chef_can": "master_chef_can",
    "masterchefcan": "master_chef_can",
}
_MARC_SUPPORTED_CACHE = None


def _marc_yolo_force_canonical(name):
    n = str(name or "").strip().lower()
    return _MARC_YOLO_FORCE_ALIASES.get(n, n)


def _marc_yolo_force_types():
    """[marc-patch] Mission objects to spawn even if the GUI catalog omitted them."""
    raw = os.environ.get(
        "TRAINER_YOLO_FORCE_CLASSES", ",".join(_MARC_YOLO_FORCE_DEFAULT))
    names = []
    for n in raw.split(","):
        n = _marc_yolo_force_canonical(n)
        if n and n not in names:
            names.append(n)
    return names


def _marc_supported_object_types():
    """Spawnable mission-object names from config, cached for the process."""
    global _MARC_SUPPORTED_CACHE
    if _MARC_SUPPORTED_CACHE is not None:
        return _MARC_SUPPORTED_CACHE
    found = set()
    try:
        from marc_config.configurations import Configurations
        for t in Configurations().get_supported_object_types() or []:
            if t:
                found.add(str(t).lower())
    except Exception:
        pass
    try:
        import yaml as _marc_yaml
        with open("/metacom2026/scenarios/config.yaml", "r", encoding="utf-8") as _marc_f:
            _assets = (_marc_yaml.safe_load(_marc_f) or {}).get(
                "mission_object_assets") or {}
        for _name in _assets:
            if _name:
                found.add(str(_name).lower())
    except Exception:
        pass
    _MARC_SUPPORTED_CACHE = found
    return found


def _marc_yolo_force_spec(want, catalog):
    """Build a deploy spec for a type the GUI catalog omitted."""
    want = _marc_yolo_force_canonical(want)
    for it in catalog.get("objects") or []:
        spec0 = it.get("spec") or {}
        tokens = (
            spec0.get("name"), spec0.get("type"),
            it.get("key"), it.get("label"),
        )
        if any(_marc_yolo_force_canonical(t) == want for t in tokens if t):
            return dict(spec0)
    supported = _marc_supported_object_types()
    actual = want
    if supported:
        match = [t for t in supported if _marc_yolo_force_canonical(t) == want]
        if not match:
            if not globals().get("_marc_yolo_force_missing"):
                globals()["_marc_yolo_force_missing"] = set()
            missing = globals()["_marc_yolo_force_missing"]
            if want not in missing:
                missing.add(want)
                print("[instances-patch] YOLO force type not in assets:", want)
            return None
        actual = match[0]
    return {"name": want, "type": actual}

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
    skip = os.environ.get(
        "TRAINER_SKIP_CLASSES", "long_umbrella,master_chef_can,wood_block")
    force = os.environ.get("TRAINER_YOLO_FORCE_CLASSES", "umbrella")
    print(f"[instances-patch] objects {olo}-{ohi}; landmarks {llo}-{lhi} p={p}; "
          f"vehicles {vlo}-{vhi} p={vp} max_types={vt}; skip={skip}; force={force}")


def _write(src):
    try:
        with open(TARGET, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    except OSError as e:
        print(f"[instances-patch] write failed, skipping: {e}")
        return False


# Exact skip helper as previously injected (substring "umbrella" also dropped
# ordinary umbrella). Used to upgrade an already-patched trainer.py.
OLD_SKIP = '''def _marc_skip_object(it):
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


def _new_skip() -> str:
    start = HELPER.find("def _marc_skip_object(it):")
    end = HELPER.find("\ndef _marc_yolo_inject():")
    if start < 0 or end < 0:
        raise RuntimeError("HELPER missing skip/yolo-inject helpers")
    return HELPER[start:end]


def _yolo_inject_fn() -> str:
    start = HELPER.find("\ndef _marc_yolo_inject():")
    if start < 0:
        raise RuntimeError("HELPER missing _marc_yolo_inject")
    return HELPER[start:]


def _yolo_inject_block() -> str:
    inject = NEW_LISTS.split("    landmarks = []", 1)[0]
    return inject[inject.find("    if _marc_yolo_inject():"):]


def _yolo_force_fn() -> str:
    start = HELPER.find("\n_MARC_YOLO_FORCE_DEFAULT = ")
    if start < 0:
        raise RuntimeError("HELPER missing YOLO force types")
    return HELPER[start:]


def _yolo_force_loop() -> str:
    inject = NEW_LISTS.split("    landmarks = []", 1)[0]
    start = inject.find("        for _want in _marc_yolo_force_types():")
    if start < 0:
        raise RuntimeError("NEW_LISTS missing YOLO force loop")
    return inject[start:]


def _refresh_patched(src: str) -> tuple[str, str]:
    """Upgrade an already-patched trainer.py skip/inject without a full rewrite."""
    changed = []
    if OLD_SKIP in src:
        src = src.replace(OLD_SKIP, _new_skip(), 1)
        changed.append("exact-name long_umbrella skip")
    old_inject = (
        '            match = next(\n'
        '                (t for t in ("master_chef_can", "wood_block") if t in name),\n'
        '                None)\n'
        '            if match is None or match in _have:\n'
        '                continue\n'
    )
    new_inject = (
        '            key = str(it.get("key") or "").lower()\n'
        '            if name in _have or key in _have:\n'
        '                continue\n'
    )
    if old_inject in src:
        src = src.replace(old_inject, new_inject, 1)
        src = src.replace(
            '        _have = " ".join(\n'
            '            str(s.get("name") or s.get("type") or "") for s in objects\n'
            '        ).lower()\n',
            '        _have = set()\n'
            '        for s in objects:\n'
            '            _have.add(str(s.get("name") or "").lower())\n'
            '            _have.add(str(s.get("type") or "").lower())\n'
            '            _have.add(str(s.get("key") or "").lower())\n'
            '        _have.discard("")\n',
            1,
        )
        src = src.replace(
            '            _have += " " + match\n',
            '            _have.add(name)\n            _have.add(key)\n',
            1,
        )
        changed.append("YOLO_MODE inject all non-skipped objects")
    old_force = '_MARC_YOLO_FORCE_DEFAULT = ("master_chef_can", "wood_block")'
    new_force = '_MARC_YOLO_FORCE_DEFAULT = ("umbrella",)'
    if old_force in src:
        src = src.replace(old_force, new_force, 1)
        changed.append("YOLO force umbrella only")
    old_skip_def = 'os.environ.get("TRAINER_SKIP_CLASSES", "long_umbrella")'
    new_skip_def = (
        'os.environ.get(\n'
        '        "TRAINER_SKIP_CLASSES",\n'
        '        "long_umbrella,master_chef_can,wood_block")'
    )
    if old_skip_def in src:
        src = src.replace(old_skip_def, new_skip_def, 1)
        changed.append("skip chef/wood")
    old_skip_one_line = (
        '    raw = os.environ.get("TRAINER_SKIP_CLASSES", "long_umbrella")\n'
    )
    new_skip_one_line = (
        '    raw = os.environ.get(\n'
        '        "TRAINER_SKIP_CLASSES",\n'
        '        "long_umbrella,master_chef_can,wood_block")\n'
    )
    if old_skip_one_line in src:
        src = src.replace(old_skip_one_line, new_skip_one_line, 1)
        changed.append("skip chef/wood raw")
    return src, ", ".join(changed)


def upgrade_existing(src: str) -> tuple[str, str]:
    """Bring an already-patched trainer.py to exact-name skip + YOLO inject.

    Do not call the people patch's nested `_marc_yolo_mode` from the object
    list: that name is local to `_build_deploy_lists` and the nested def sits
    after this inject, which raises UnboundLocalError.
    """
    notes = []
    src, note = _refresh_patched(src)
    if note:
        notes.append(note)
    shadowed = "    if _marc_yolo_mode():\n        _have = set()\n"
    renamed = "    if _marc_yolo_inject():\n        _have = set()\n"
    if shadowed in src:
        src = src.replace(shadowed, renamed, 1)
        notes.append("unshadow yolo inject call")
    if "def _marc_yolo_inject" not in src:
        insert_at = src.find("\ndef _build_deploy_lists(")
        fn = _yolo_inject_fn()
        if insert_at >= 0:
            src = src[:insert_at] + fn + src[insert_at:]
            notes.append("yolo inject helper")
        elif fn.strip() not in src:
            needle = "    return any(token in skip for token in tokens)\n"
            if needle in src:
                src = src.replace(needle, needle + fn, 1)
                notes.append("yolo inject helper")
    if "if _marc_yolo_inject():" not in src:
        old_lists = "        objects.append(spec)\n    landmarks = []\n"
        new_lists = (
            "        objects.append(spec)\n"
            + _yolo_inject_block()
            + "    landmarks = []\n"
        )
        if old_lists in src:
            src = src.replace(old_lists, new_lists, 1)
            notes.append("yolo inject lists")
    if "def _marc_yolo_force_types" not in src:
        insert_at = src.find("\ndef _build_deploy_lists(")
        fn = _yolo_force_fn()
        if insert_at >= 0:
            src = src[:insert_at] + fn + src[insert_at:]
            notes.append("yolo force-type helpers")
    if "for _want in _marc_yolo_force_types():" not in src:
        old_tail = (
            "            _have.add(name)\n"
            "            _have.add(key)\n"
            "    landmarks = []\n"
        )
        new_tail = (
            "            _have.add(name)\n"
            "            _have.add(key)\n"
            + _yolo_force_loop()
            + "    landmarks = []\n"
        )
        if old_tail in src:
            src = src.replace(old_tail, new_tail, 1)
            notes.append("yolo force-type spawn")
    return src, ", ".join(notes)


def main():
    if not os.path.exists(TARGET):
        print(f"[instances-patch] source not found, skipping: {TARGET}")
        return 0

    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if MARKER in src:
        src, note = upgrade_existing(src)
        if note:
            if not _write(src):
                return 0
            print("[instances-patch] upgraded: " + note)
            _log_bounds()
            return 0
        print("[instances-patch] already applied, skipping")
        return 0

    if V3_HELPER in src and V3_LISTS in src:
        src = src.replace(V3_HELPER, HELPER, 1).replace(V3_LISTS, NEW_LISTS, 1)
        if not _write(src):
            return 0
        print("[instances-patch] upgraded: exact-name skip + YOLO inject")
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
