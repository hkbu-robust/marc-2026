#!/usr/bin/env python3
# Copyright (c) 2026, IoT Convergence & Open Sharing System (IoTCOSS)
#
# Host-testable spawn / label planning for ego-dataset-gen. No Isaac imports.
# ---------------------------------------------------------------------------
# Label ids stay frozen on CANONICAL_NAMES so appending scenes cannot shift
# tumbler=0 … wood_block=10. New umbrellas are always class 11.
import math
import os
import re

# Keep class ids stable and aligned with the existing ego_yolo dump.
CANONICAL_NAMES = [
    "tumbler", "cracker_box", "cola_can", "disposable_cup",
    "juice", "tissue", "mug", "sunblock", "pencilcase",
    "master_chef_can", "wood_block", "umbrella",
]
DEFAULT_SKIP = ("long_umbrella", "master_chef_can", "wood_block")
DEFAULT_FORCE = ("umbrella",)
DEFAULT_FORCE_FRAC = 0.85
DEFAULT_EMPTY_FRAC = 0.05

_CLASS_ALIASES = {
    "closedlongumbrella": "long_umbrella",
    "wooden_block": "wood_block",
    "woodblock": "wood_block",
    "chef_can": "master_chef_can",
    "masterchefcan": "master_chef_can",
}
_SCENARIO_DUMP = re.compile(r"^marc2026_[a-z0-9_]+$", re.I)


def canonical_class(name):
    n = str(name or "").strip().lower()
    return _CLASS_ALIASES.get(n, n)


def _csv_names(raw, default):
    if raw is None:
        items = list(default)
    else:
        items = [n.strip() for n in str(raw).split(",")]
    out = []
    for n in items:
        n = canonical_class(n)
        if n and n not in out:
            out.append(n)
    return out


def skip_classes(env=None):
    env = os.environ if env is None else env
    skip = set(_csv_names(env.get("EGO_SKIP_CLASSES"), DEFAULT_SKIP))
    expanded = set(skip)
    for alias, canon in _CLASS_ALIASES.items():
        if canon in skip:
            expanded.add(alias)
            expanded.add(canon)
    return expanded


def is_skipped(name, skip=None):
    skip = skip_classes() if skip is None else skip
    return canonical_class(name) in skip or str(name or "").strip().lower() in skip


def force_classes(env=None):
    """Spawn-bias classes. MARC_MANIP_OBJECT is an alias; it never replaces labels."""
    env = os.environ if env is None else env
    raw = env.get("EGO_FORCE_CLASSES")
    if raw is None:
        names = list(DEFAULT_FORCE)
    elif str(raw).strip().lower() in ("", "0", "none", "off", "-"):
        names = []
    else:
        names = _csv_names(raw, ())
    extra = canonical_class(env.get("MARC_MANIP_OBJECT"))
    if extra and extra not in names:
        names.append(extra)
    return [n for n in names if n]


def force_frac(env=None):
    env = os.environ if env is None else env
    raw = env.get("EGO_FORCE_FRAC")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_FORCE_FRAC
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return DEFAULT_FORCE_FRAC


def empty_frac(env=None):
    """Force runs default to no empty scenes unless EGO_EMPTY_FRAC is set."""
    env = os.environ if env is None else env
    raw = env.get("EGO_EMPTY_FRAC")
    force = force_classes(env)
    if raw is None or str(raw).strip() == "":
        return 0.0 if force else DEFAULT_EMPTY_FRAC
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 0.0 if force else DEFAULT_EMPTY_FRAC


def label_names():
    """Frozen YOLO name list. Spawn skip must not drop chef/wood from this."""
    return list(CANONICAL_NAMES)


def spawn_types(catalog_types, skip=None, force=None):
    skip = skip_classes() if skip is None else skip
    force = force_classes() if force is None else force
    out = []
    for t in catalog_types or []:
        n = canonical_class(t)
        if n and not is_skipped(n, skip) and n not in out:
            out.append(n)
    for n in force:
        n = canonical_class(n)
        if n and not is_skipped(n, skip) and n not in out:
            out.append(n)
    return out


def resolve_out_dir(env=None):
    env = os.environ if env is None else env
    raw = env.get("EGO_OUT")
    if raw:
        return raw
    root = env.get("MARC_TRAINER_DIR") or "/metacom2026/trainer_output"
    return os.path.join(root, "ego_yolo")


def assert_ego_out(out_dir):
    """Refuse CCTV scenario dumps (marc2026_chungmu and siblings)."""
    path = os.path.abspath(out_dir)
    for part in path.replace("\\", "/").split("/"):
        if _SCENARIO_DUMP.match(part):
            raise SystemExit(
                f"[ego-gen] EGO_OUT refuses scenario dump {part!r}: {path}. "
                "CCTV GT lives under trainer_output/marc2026_*; "
                "ego YOLO must use trainer_output/ego_yolo"
            )
    return path


def plan_scene_classes(rng, spawn_names, n_scenes, obj_min, obj_max,
                       empty_frac_val, force=None, frac=None):
    """Per-scene class lists. Force: every content scene has a forced class."""
    spawn_names = [canonical_class(n) for n in spawn_names if canonical_class(n)]
    force = [canonical_class(n) for n in (force or []) if canonical_class(n)]
    force = [n for n in force if n in spawn_names] or [
        n for n in force if n
    ]
    n_scenes = max(0, int(n_scenes))
    obj_min = max(0, int(obj_min))
    obj_max = max(obj_min, int(obj_max))
    if force:
        return _plan_forced(
            rng, spawn_names, n_scenes, obj_min, obj_max,
            empty_frac_val, force, DEFAULT_FORCE_FRAC if frac is None else frac,
        )
    return _plan_coverage(
        rng, spawn_names, n_scenes, obj_min, obj_max, empty_frac_val)


def _plan_coverage(rng, names, n_scenes, obj_min, obj_max, empty_frac_val):
    n_cls = len(names)
    n_empty = int(round(n_scenes * empty_frac_val)) if n_scenes else 0
    if n_cls:
        n_empty = min(n_empty, max(0, n_scenes - math.ceil(n_cls / max(1, obj_max))))
    empty_idx = set(rng.sample(range(n_scenes), n_empty)) if n_empty else set()
    content = [i for i in range(n_scenes) if i not in empty_idx]
    if not content:
        return [[] for _ in range(n_scenes)]
    cap = max(1, obj_max)
    counts = [max(0, obj_min)] * len(content)
    need = max(0, n_cls - sum(counts))
    for i in range(len(content)):
        while need > 0 and counts[i] < cap:
            counts[i] += 1
            need -= 1
    i = 0
    while need > 0:
        counts[i % len(content)] += 1
        need -= 1
        i += 1
    for i in range(len(content)):
        room = cap - counts[i]
        if room > 0:
            counts[i] += rng.randint(0, room)
    n_slots = sum(counts)
    bag = []
    if names:
        must = list(names)
        rng.shuffle(must)
        bag.extend(must)
        while len(bag) < n_slots:
            extra = list(names)
            rng.shuffle(extra)
            bag.extend(extra)
        bag = bag[:n_slots]
        rng.shuffle(bag)
    plan = [[] for _ in range(n_scenes)]
    it = iter(bag)
    for scene_i, n_obj in zip(content, counts):
        plan[scene_i] = [next(it) for _ in range(n_obj)]
    return plan


def _plan_forced(rng, spawn_names, n_scenes, obj_min, obj_max,
                 empty_frac_val, force, frac):
    frac = min(1.0, max(0.0, float(frac)))
    n_empty = int(round(n_scenes * empty_frac_val)) if n_scenes else 0
    n_empty = min(n_empty, max(0, n_scenes - 1)) if n_scenes else 0
    empty_idx = set(rng.sample(range(n_scenes), n_empty)) if n_empty else set()
    distractors = [n for n in spawn_names if n not in force] or list(spawn_names)
    pool_force = [n for n in force if n in spawn_names] or list(force)
    if not pool_force:
        pool_force = list(spawn_names)
    plan = [[] for _ in range(n_scenes)]
    for i in range(n_scenes):
        if i in empty_idx:
            continue
        n_obj = rng.randint(obj_min, obj_max) if obj_max > 0 else 0
        n_obj = max(1, n_obj) if pool_force else n_obj
        classes = []
        if n_obj:
            classes.append(rng.choice(pool_force))
        for _ in range(n_obj - 1):
            if rng.random() < frac and pool_force:
                classes.append(rng.choice(pool_force))
            elif distractors:
                classes.append(rng.choice(distractors))
            elif pool_force:
                classes.append(rng.choice(pool_force))
        plan[i] = classes
    return plan
