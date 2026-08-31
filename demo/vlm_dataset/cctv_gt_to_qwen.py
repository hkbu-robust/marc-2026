#!/usr/bin/env python3
"""Convert CCTV dataset-gen GT (PNG + JSON) into Qwen3-VL instruction rows.

Does not run Isaac. Reads trainer_output/<scenario>/scene_XXXXX/{camera}.png|.json
and writes LLaMA-Factory JSONL (combined ``train.jsonl`` / ``val.jsonl`` plus
separate SAR and object files for inspection).

SAR rows use the R03 pose×support matrix (marc_agent.perception.situation_rules)
and the runtime board-answer grammar ``Pxx`` with no coordinates or prose.
The SAR user turn matches ``format_sar_verifier_prompt`` in
``marc_agent.perception.sar_selection``. Object rows keep eight-relation
referring expressions without grounding xyz.

Usage (from repo root or demo/):

  python3 demo/vlm_dataset/cctv_gt_to_qwen.py \\
    --in trainer_output/marc2026_chungmu \\
    --out trainer_output/qwen2vl_cctv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from marc_agent.command_parser import (  # noqa: E402
    CommandParseError,
    LANDMARK_TYPES,
    MISSION_OBJECT_TYPES,
    parse_command,
)
from marc_agent.perception.object_selection import (  # noqa: E402
    build_object_board,
    rank_object_candidates,
)
from marc_agent.perception.candidate_ids import person_candidate_id  # noqa: E402
from marc_agent.perception.pose_to_situation import situation_for_pose  # noqa: E402
from marc_agent.perception.sar_selection import (  # noqa: E402
    _assemble_board,
    candidate_evidence,
    format_sar_verifier_prompt,
    shortlist_for_verifier,
)
from marc_agent.perception.situation_rules import (  # noqa: E402
    RULES_VERSION,
    support_class as r03_support_class,
)
from marc_agent.types import (  # noqa: E402
    CctvDetectionRound,
    CctvFrame,
    ObjectCandidate,
    PersonCandidate,
)

_RECIPES_PATH = os.path.join(_ROOT, "simulation-platform", "patches", "sar_recipes.yaml")
_RECIPE_BY_ID: Dict[str, dict] = {}
_SKIPPED_OBJECTS = frozenset({"long_umbrella", "master_chef_can", "wood_block"})


def _load_recipes() -> Dict[str, dict]:
    global _RECIPE_BY_ID
    if _RECIPE_BY_ID:
        return _RECIPE_BY_ID
    try:
        import yaml  # type: ignore
        with open(_RECIPES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for rec in data.get("recipes") or []:
            rid = str(rec.get("id") or "")
            if rid:
                _RECIPE_BY_ID[rid] = rec
    except Exception:
        _RECIPE_BY_ID = {}
    return _RECIPE_BY_ID

SCENE_RE = re.compile(r"^scene_(\d+)$")

# Spoken names in MARC voice_command text.
DISPLAY = {
    "cracker_box": "cracker box",
    "cola_can": "cola can",
    "disposable_cup": "disposable cup",
    "pencilcase": "pencil case",
    "picnic_table": "picnic table",
    "bench": "bench",
    "hydrant": "hydrant",
    "postbox": "postbox",
    "taxi": "taxi",
    "vending_machine": "vending machine",
    "trash_can": "trash can",
    "kick_scooter": "kick scooter",
    "police_car": "police car",
    "normal_car_1": "normal car one",
    "normal_car_2": "normal car two",
    "sports_car": "sports car",
    "sports_car_2": "sports car two",
}

TEMPLATES = {
    "beside": (
        "Find the {obj} beside the {lm}.",
        "Find the {obj} next to the {lm}.",
        "Locate the {obj} beside the {lm}.",
    ),
    "near": (
        "Find the {obj} near the {lm}.",
        "Find the {obj} close to the {lm}.",
        "Locate the {obj} near the {lm}.",
    ),
    "on": (
        "Find the {obj} on the {lm}.",
        "Find the {obj} on top of the {lm}.",
        "Locate the {obj} on the {lm}.",
    ),
    "above": (
        "Find the {obj} above the {lm}.",
        "Locate the {obj} above the {lm}.",
    ),
    "left": (
        "Find the {obj} to the left of the {lm}.",
        "Locate the {obj} on the left of the {lm}.",
        "Find the {obj} left of the {lm}.",
    ),
    "right": (
        "Find the {obj} to the right of the {lm}.",
        "Locate the {obj} on the right of the {lm}.",
        "Find the {obj} right of the {lm}.",
    ),
    "front": (
        "Find the {obj} in front of the {lm}.",
        "Locate the {obj} in front of the {lm}.",
    ),
    "behind": (
        "Find the {obj} behind the {lm}.",
        "Locate the {obj} behind the {lm}.",
    ),
}

# Skip people / pose types in the lost-object family — SAR is a second row type.
_PERSON_PREFIXES = ("person_", "male_", "female_", "professor_", "student_")

# Spoken pose fragments for SAR prompts (GT class is the pose, not identity).
_SAR_POSE_SPEAK = {
    "person_prone": "lying face-down",
    "person_lying_down": "lying down",
    "person_bending": "bent over",
    "person_crouching": "crouched",
    "person_reclining": "reclining",
    "person_sitting": "sitting",
    "person_reaching": "reaching",
    "person_walking": "walking",
    "person_walking_away": "walking away",
}

# Pose→situation lives in marc_agent.perception.situation_rules (R03 2026.1).

_SEAT_LANDMARKS = frozenset({"bench", "picnic_table"})
_CLIMB_LANDMARKS = frozenset({
    "vending_machine", "trash_can", "hydrant", "postbox",
    "bicycle", "kick_scooter",
})
_VEHICLE_LANDMARKS = frozenset({
    "police_car", "normal_car_1", "normal_car_2",
    "sports_car", "sports_car_2", "suv", "taxi",
})
_WRONG_PLACE_LANDMARKS = _CLIMB_LANDMARKS | _VEHICLE_LANDMARKS

# Prim origins are not contact points (character root vs asset base), so a
# loose XY/Z window flags people standing *beside* a hydrant as "on" it.
# "On/above" prefers 2D stacking (box on the landmark, not sharing the
# ground line). dataset-gen sitters often hang legs or fill the frame, so
# a world-XYZ rescue (roof-height Z + tight XY) still counts as on/above.
_ABOVE_DZ_M = 0.7
_STACK_X_OVERLAP = 0.4
_STACK_FEET_ABOVE_BASE = 0.22
# Min person-minus-landmark Z (m) to count as on a climb landmark when 2D
# feet fail. Standing in front of a vending is ~0.85 m (pelvis vs base).
_CLIMB_DZ_M = {
    "vending_machine": 1.15,
    "trash_can": 0.40,
    "hydrant": 0.28,
    "postbox": 0.40,
    "bicycle": 0.35,
    "kick_scooter": 0.35,
}
_CLIMB_DXY_M = {
    "vending_machine": 0.95,
    "trash_can": 0.60,
    "hydrant": 0.45,
    "postbox": 0.55,
    "bicycle": 0.50,
    "kick_scooter": 0.50,
}

_LYING_POSES = ("person_prone", "person_lying_down")
_SITTING_POSES = ("person_sitting", "person_reclining")
_DISTRESS_POSES = ("person_prone", "person_lying_down", "person_bending", "person_crouching")

SAR_TEMPLATES = (
    "Find the person {pose} near the {lm}.",
    "Find the person {pose} beside the {lm}.",
    "Locate the person {pose} near the {lm}.",
)


def _display(name: str) -> str:
    if name in DISPLAY:
        return DISPLAY[name]
    return str(name).replace("_", " ")


def _is_person(cls: str) -> bool:
    c = (cls or "").lower()
    return c == "person" or c.startswith("person_") or any(c.startswith(p) for p in _PERSON_PREFIXES)


def _is_long_umbrella(cls: str) -> bool:
    c = (cls or "").lower().replace("-", "_")
    return c in ("long_umbrella", "closedlongumbrella")


def _sar_pose_speak(cls: str) -> str:
    c = (cls or "").lower()
    if c in _SAR_POSE_SPEAK:
        return _SAR_POSE_SPEAK[c]
    if c.startswith("person_"):
        return c.replace("person_", "").replace("_", " ")
    return "in distress"


def _sar_pose_family(cls: str) -> str:
    c = (cls or "").lower()
    if c.startswith("person_sitting"):
        return "sitting"
    if c.startswith("person_reclining"):
        return "reclining"
    if c.startswith("person_prone"):
        return "prone"
    if c.startswith("person_lying"):
        return "lying"
    if c.startswith("person_bending"):
        return "bending"
    if c.startswith("person_crouching"):
        return "crouching"
    return "other"


def _sar_ground_situation(cls: str) -> str:
    """Pose-only situation when the person is not on a landmark top."""
    return situation_for_pose(cls)


def _xyz(det: dict) -> Optional[List[float]]:
    xyz = det.get("world_xyz")
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    return None


def _bbox4(det: dict) -> Optional[Tuple[float, float, float, float]]:
    b = det.get("bbox")
    if not b or len(b) < 4:
        return None
    x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return x0, y0, x1, y1


def _stacked_on_top(obj: dict, lm: dict) -> bool:
    """True when the object's 2D box sits on the landmark, not on the ground beside it."""
    ob = _bbox4(obj)
    lb = _bbox4(lm)
    if ob is None or lb is None:
        return False
    ox0, oy0, ox1, oy1 = ob
    lx0, ly0, lx1, ly1 = lb
    overlap_x = min(ox1, lx1) - max(ox0, lx0)
    min_w = min(ox1 - ox0, lx1 - lx0)
    if min_w <= 1.0 or overlap_x < _STACK_X_OVERLAP * min_w:
        return False
    obj_cy = 0.5 * (oy0 + oy1)
    lm_cy = 0.5 * (ly0 + ly1)
    if obj_cy >= lm_cy:
        return False
    if str(lm.get("class") or "").lower() in _SEAT_LANDMARKS:
        return True
    # Non-seat: contact must be up on the body, not sharing the ground line.
    lm_h = ly1 - ly0
    if (ly1 - oy1) < _STACK_FEET_ABOVE_BASE * lm_h:
        return False
    return True


def _climb_from_world(person: dict, lm: dict) -> bool:
    """True when world XYZ says the person is on a climb landmark roof.

    2D stacking misses hanging-leg sits, corner sits, and full-frame crops
    where the person box shares the landmark's ground line.
    """
    typ = str(lm.get("class") or "").lower()
    if typ not in _CLIMB_DZ_M:
        return False
    p = _xyz(person)
    l = _xyz(lm)
    if p is None or l is None:
        return False
    dxy = math.hypot(p[0] - l[0], p[1] - l[1])
    dz = p[2] - l[2]
    if dxy > float(_CLIMB_DXY_M.get(typ, 0.5)):
        return False
    if dz < float(_CLIMB_DZ_M.get(typ, 0.8)):
        return False
    ob = _bbox4(person)
    lb = _bbox4(lm)
    if ob is None or lb is None:
        return True
    ox0, oy0, ox1, oy1 = ob
    lx0, ly0, lx1, ly1 = lb
    overlap_x = min(ox1, lx1) - max(ox0, lx0)
    min_w = min(ox1 - ox0, lx1 - lx0)
    weak_x = min_w > 1.0 and overlap_x > 0.08 * min_w
    obj_cy = 0.5 * (oy0 + oy1)
    lm_cy = 0.5 * (ly0 + ly1)
    if not weak_x and obj_cy >= lm_cy:
        return False
    return True


def _support_rel(person: dict, lm: dict) -> Optional[str]:
    """Return on/above when the person is stacked on the landmark in-frame."""
    if not (_stacked_on_top(person, lm) or _climb_from_world(person, lm)):
        return None
    p = _xyz(person)
    l = _xyz(lm)
    if p is not None and l is not None and (p[2] - l[2]) >= _ABOVE_DZ_M:
        return "above"
    return "on"


def _is_wrong_place(lm_cls: str) -> bool:
    c = (lm_cls or "").lower()
    if c in _WRONG_PLACE_LANDMARKS:
        return True
    return "car" in c or c in ("suv", "taxi")


def _is_seat(lm_cls: str) -> bool:
    return (lm_cls or "").lower() in _SEAT_LANDMARKS


_FALLEN_VEHICLE = frozenset({"bicycle", "kick_scooter"})
_BESIDE_XY_M = 2.5


def _is_fallen_vehicle(lm_cls: Optional[str]) -> bool:
    return (lm_cls or "").lower() in _FALLEN_VEHICLE


def _beside_fallen_vehicle(person: dict, lm: dict) -> bool:
    if not lm or not _is_fallen_vehicle(lm.get("class")):
        return False
    p, l = _xyz(person), _xyz(lm)
    if p is not None and l is not None:
        return math.hypot(p[0] - l[0], p[1] - l[1]) <= _BESIDE_XY_M
    pc = _bbox_center(person.get("bbox"))
    lc = _bbox_center(lm.get("bbox"))
    if pc is None or lc is None:
        return False
    return math.hypot(pc[0] - lc[0], pc[1] - lc[1]) < 180.0


def _is_unlabeled_lying(person: dict, rel: Optional[str]) -> bool:
    """Ground lying_down with no support/surface is dropped, not forced to accident."""
    pose = str(person.get("class") or "").lower()
    if pose.startswith("person_prone"):
        return False
    if not pose.startswith("person_lying"):
        return False
    if person.get("support_class") or person.get("surface_class"):
        return False
    return rel not in ("on", "above", "beside")


def _det_sort_key(det: dict) -> Tuple[tuple, str]:
    b = det.get("bbox") or (0.0, 0.0, 0.0, 0.0)
    box = tuple(float(x) for x in list(b)[:4])
    return (box, str(det.get("class") or ""))


def person_candidate_ids(people: List[dict], camera_id: str) -> Dict[int, str]:
    """Runtime-matching IDs: ``{camera_id}:{pose_class}:{n}`` in bbox-then-class order."""
    ordered = sorted(people, key=_det_sort_key)
    counts: dict[str, int] = defaultdict(int)
    mapping: Dict[int, str] = {}
    for person in ordered:
        pose = str(person.get("class") or "person")
        idx = counts[pose]
        counts[pose] += 1
        mapping[id(person)] = person_candidate_id(camera_id, pose, idx)
    return mapping


def person_candidate_id_list(people: List[dict], camera_id: str) -> Tuple[str, ...]:
    ordered = sorted(people, key=_det_sort_key)
    id_map = person_candidate_ids(people, camera_id)
    return tuple(id_map[id(person)] for person in ordered)


def _numbered_candidate(candidate_id: str, candidate_ids: Iterable[str]) -> str:
    """Runtime boards ask Qwen to answer with the stable Pxx row alias."""
    ids = tuple(candidate_ids)
    try:
        return f"P{ids.index(candidate_id) + 1:02d}"
    except ValueError as exc:
        raise ValueError(f"candidate is not listed: {candidate_id}") from exc


def _speak_landmark(cls: str, rng: random.Random, drink: bool = False) -> str:
    c = (cls or "").lower()
    if c == "vending_machine":
        if drink:
            return rng.choice(("drink machine", "drink machine", "vending machine"))
        return rng.choice(("vending machine", "drink machine"))
    if c == "trash_can":
        return rng.choice(("trash can", "trash can", "bin"))
    return _display(cls)


def _xy_m(a: dict, b: dict) -> Optional[float]:
    pa, pb = _xyz(a), _xyz(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _gt_support_ok(person: dict, landmarks: List[dict]) -> bool:
    gt_support = person.get("support_class")
    if not gt_support:
        return True
    kind = r03_support_class(str(gt_support))
    if kind not in ("seating", "non_mount", "fallen_vehicle"):
        return True
    for lm in landmarks:
        if str(lm.get("class") or "") != str(gt_support):
            continue
        d = _xy_m(person, lm)
        if d is not None and d <= 2.5:
            return True
        pc = _bbox_center(person.get("bbox"))
        lc = _bbox_center(lm.get("bbox"))
        if pc is not None and lc is not None and math.hypot(
                pc[0] - lc[0], pc[1] - lc[1]) < 180.0:
            return True
    return False


def _sar_situation_for(person: dict, lm: dict, rel: Optional[str],
                       rng: random.Random,
                       landmarks: Optional[List[dict]] = None) -> str:
    del rng  # labels are deterministic
    pose = str(person.get("class") or "")
    gt_support = person.get("support_class")
    gt_surface = person.get("surface_class")
    if (gt_support or gt_surface) and _gt_support_ok(person, landmarks or []):
        return situation_for_pose(pose, support=gt_support, surface=gt_surface)
    support = None
    if rel in ("on", "above") and lm:
        support = str(lm.get("class") or "") or None
    elif rel == "beside" and lm and _is_fallen_vehicle(lm.get("class")):
        support = str(lm.get("class") or "") or None
    return situation_for_pose(pose, support=support)


def _sar_instruction(person: dict, lm: dict, rel: Optional[str], situation: str,
                     rng: random.Random) -> str:
    pose = str(person.get("class") or "")
    lm_cls = str((lm or {}).get("class") or "")
    family = _sar_pose_family(pose)
    speak = _sar_pose_speak(pose)
    on_top = rel in ("on", "above")
    sit = (situation or "").strip().lower()
    if on_top and lm_cls == "vending_machine":
        drink = _speak_landmark(lm_cls, rng, drink=True)
        generic = [
            f"Someone is up where people should not be; find the person who climbed on top of the {drink}.",
            f"Find the person who climbed on top of the {drink}.",
            f"Find the person who climbed up onto the {drink}.",
            f"Find the person in an {sit or 'abnormal'} situation on the {drink}.",
        ]
        pose_bits = []
        if family == "sitting":
            pose_bits.extend((
                f"Find the person sitting on top of the {drink}.",
                "Find the person sitting on top of the vending machine.",
            ))
        elif family == "crouching":
            pose_bits.append(
                f"Find the person crouched on top of the {drink}.")
        elif str(pose).lower() == "person_reaching":
            pose_bits.extend((
                f"Find the person reaching on top of the {drink}.",
                f"Find the person reaching out on the {drink}.",
            ))
        elif str(pose).lower() in ("person_walking", "person_walking_away"):
            pose_bits.append(
                f"Find the person standing on top of the {drink}.")
        return rng.choice(generic + pose_bits)
    if on_top and lm_cls == "trash_can":
        can = _speak_landmark(lm_cls, rng)
        generic = [
            f"Find the person who climbed up onto the {can}.",
            f"Find the person in an {sit or 'abnormal'} situation on the {can}.",
        ]
        if family == "sitting":
            generic.extend((
                f"In the parking lot, find the person who climbed up and is sitting on top of the {can}.",
                f"Find the person sitting on top of the {can}.",
                f"Find the person sitting on top of the trash can in the parking lot.",
            ))
        else:
            generic.append(f"Find the person {speak} on top of the {can}.")
        return rng.choice(generic)
    if on_top and _is_wrong_place(lm_cls):
        name = _speak_landmark(lm_cls, rng)
        return rng.choice((
            f"Someone is up where people should not be; find the person on top of the {name}.",
            f"Find the person who climbed on top of the {name}.",
            f"Find the person sitting on top of the {name}.",
        ))
    if on_top and lm_cls in ("picnic_table",) and family in ("lying", "prone"):
        return rng.choice((
            "Find the person lying on top of the picnic table.",
            "Find the person lying on the picnic table.",
        ))
    if on_top and lm_cls == "bench" and family in ("lying", "prone"):
        return rng.choice((
            "Find the person lying down on top of the bench.",
            "Find the person lying face-down on the bench.",
            "Near the taxi, find the person lying down on top of the bench.",
        ))
    if on_top and family in ("sitting", "reclining") and _is_seat(lm_cls):
        name = _speak_landmark(lm_cls, rng)
        return rng.choice((
            f"Find the person sitting on the {name}.",
            f"Find the person sitting on top of the {name}.",
        ))
    if sit == "accident":
        return rng.choice((
            "Find the person lying face-down and motionless.",
            "Several people are in view; one is completely face-down and immobile; find the most critical one.",
        ))
    if sit == "emergency" and family in ("lying", "prone"):
        if lm_cls:
            name = _speak_landmark(lm_cls, rng)
            return rng.choice((
                f"Find the person collapsed on the floor near the {name}.",
                f"Find the person lying face-down and motionless near the {name}.",
            ))
        return "Find the person collapsed on the parking floor."
    if sit == "emergency" and family in ("bending", "crouching"):
        if lm_cls:
            name = _speak_landmark(lm_cls, rng)
            return rng.choice((
                f"Find the person bent over and motionless near the {name}.",
                f"Find the person {speak} near the {name}.",
            ))
        return f"Find the person {speak} who needs help."
    if lm_cls:
        name = _speak_landmark(lm_cls, rng)
        tmpl = rng.choice(SAR_TEMPLATES)
        return tmpl.format(pose=speak, lm=name)
    return f"Find the person {speak}."


def _pick_support_landmark(person: dict, landmarks: List[dict], payload: dict):
    pc = _bbox_center(person.get("bbox"))
    best = None
    best_key = None
    for lm in landmarks:
        rel = _support_rel(person, lm)
        if rel is not None:
            pc = _bbox_center(person.get("bbox"))
            lc = _bbox_center(lm.get("bbox"))
            dpx = 0.0
            if pc is not None and lc is not None:
                dpx = abs(pc[0] - lc[0])
            key = (0, dpx)
        else:
            lc = _bbox_center(lm.get("bbox"))
            if pc is None or lc is None:
                continue
            key = (1, math.hypot(pc[0] - lc[0], pc[1] - lc[1]))
        if best_key is None or key < best_key:
            best_key, best = key, lm
    return best


def _world_or_pixel_relation(obj: dict, lm: dict, pixel_rel: str,
                             payload: Optional[dict] = None) -> str:
    """On/above when stacked; else camera-relative left/right/front/behind."""
    if _stacked_on_top(obj, lm):
        p = _xyz(obj)
        l = _xyz(lm)
        if p is not None and l is not None and (p[2] - l[2]) >= _ABOVE_DZ_M:
            return "above"
        return "on"
    specific = _camera_relative_rel(obj, lm, payload)
    if specific and pixel_rel == "beside":
        p = _xyz(obj)
        l = _xyz(lm)
        if p is not None and l is not None and math.hypot(
                p[0] - l[0], p[1] - l[1]) <= 0.75:
            return "beside"
        return specific
    return pixel_rel


def _camera_xy(payload: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not payload:
        return None
    pos = (payload.get("camera") or {}).get("position")
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return None


def _camera_relative_rel(obj: dict, lm: dict,
                         payload: Optional[dict]) -> Optional[str]:
    """left/right/front/behind from camera→landmark vs landmark→object."""
    p = _xyz(obj)
    l = _xyz(lm)
    cam = _camera_xy(payload)
    if p is not None and l is not None and cam is not None:
        fx = l[0] - cam[0]
        fy = l[1] - cam[1]
        n = math.hypot(fx, fy)
        if n < 1e-3:
            return None
        fx, fy = fx / n, fy / n
        leftx, lefty = -fy, fx
        ox, oy = p[0] - l[0], p[1] - l[1]
        if math.hypot(ox, oy) < 0.18:
            return None
        along = ox * fx + oy * fy
        lat = ox * leftx + oy * lefty
        if abs(lat) > abs(along) * 1.15:
            return "left" if lat > 0.0 else "right"
        if abs(along) > abs(lat) * 1.15:
            return "behind" if along > 0.0 else "front"
        return None
    oc = _bbox_center(obj.get("bbox"))
    lc = _bbox_center(lm.get("bbox"))
    if oc is None or lc is None:
        return None
    dx = oc[0] - lc[0]
    dy = oc[1] - lc[1]
    if abs(dx) < 36.0 and abs(dy) < 36.0:
        return None
    if abs(dx) > abs(dy) * 1.25:
        return "left" if dx < 0.0 else "right"
    return None


def _sar_situation(cls: str) -> str:
    """Back-compat pose-only label. Prefer _sar_situation_for()."""
    return _sar_ground_situation(cls)


def _bbox_center(bbox) -> Optional[Tuple[float, float]]:
    if not bbox or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _matmul(a: List[List[float]], v: List[float]) -> List[float]:
    return [
        a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
        a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
        a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
    ]


def _euler_zyx_matrix(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """Match camera_projection: R.from_euler('zyx', [yaw, pitch, roll])."""
    r, p, y = (math.radians(roll), math.radians(pitch), math.radians(yaw))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    # Rz @ Ry
    zy = [
        [rz[i][0] * ry[0][j] + rz[i][1] * ry[1][j] + rz[i][2] * ry[2][j] for j in range(3)]
        for i in range(3)
    ]
    return [
        [zy[i][0] * rx[0][j] + zy[i][1] * rx[1][j] + zy[i][2] * rx[2][j] for j in range(3)]
        for i in range(3)
    ]


def _ground_xyz(payload: dict, u: float, v: float) -> Optional[List[float]]:
    """Back-project a pixel onto the ground plane using GT camera extrinsics."""
    cam = payload.get("camera") or {}
    img = payload.get("image") or {}
    pos = cam.get("position")
    euler = cam.get("orientation_euler_deg") or {}
    if not pos or len(pos) < 3:
        return None
    width = float(img.get("width") or 0.0)
    height = float(img.get("height") or 0.0)
    if width <= 1 or height <= 1:
        return None
    f = float(cam.get("focal_length_mm") or 0.0)
    ha = float(cam.get("horizontal_aperture_mm") or 0.0)
    va = float(cam.get("vertical_aperture_mm") or 0.0)
    if f <= 0 or ha <= 0:
        return None
    if va <= 0:
        va = ha * height / width
    gz = cam.get("ground_height")
    ground_z = float(gz if gz is not None else pos[2] - 8.0)
    # USD cam: +X right, +Y up, -Z forward. Image y is down.
    x = (u / width - 0.5) * ha
    y = (0.5 - v / height) * va
    z = -f
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    dir_cam = [x / n, y / n, z / n]
    rot = _euler_zyx_matrix(
        float(euler.get("roll") or 0.0),
        float(euler.get("pitch") or 0.0),
        float(euler.get("yaw") or 0.0),
    )
    d = _matmul(rot, dir_cam)
    if abs(d[2]) < 1e-6:
        return None
    t = (ground_z - float(pos[2])) / d[2]
    if t <= 0.05 or t > 80.0:
        return None
    return [
        round(float(pos[0]) + t * d[0], 4),
        round(float(pos[1]) + t * d[1], 4),
        round(ground_z, 4),
    ]


def _det_xyz(det: dict, payload: dict) -> Optional[List[float]]:
    xyz = det.get("world_xyz")
    if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
        return [round(float(xyz[0]), 4), round(float(xyz[1]), 4), round(float(xyz[2]), 4)]
    c = _bbox_center(det.get("bbox"))
    if c is None:
        return None
    return _ground_xyz(payload, c[0], c[1])


def _relation(obj_c: Tuple[float, float], lm_c: Tuple[float, float], width: float) -> str:
    dist = math.hypot(obj_c[0] - lm_c[0], obj_c[1] - lm_c[1])
    return "beside" if dist < 0.22 * max(width, 1.0) else "near"


def _coord3(c: Optional[List[float]]) -> List[float]:
    vals = [float(v) for v in (c or [])]
    while len(vals) < 3:
        vals.append(0.0)
    return [round(v, 4) for v in vals[:3]]


def _iter_scenes(root: str) -> Iterable[Tuple[int, str]]:
    try:
        names = os.listdir(root)
    except OSError:
        return
    found = []
    for name in names:
        m = SCENE_RE.match(name)
        path = os.path.join(root, name)
        if m and os.path.isdir(path):
            found.append((int(m.group(1)), path))
    for item in sorted(found):
        yield item


def _load_pair(scene_dir: str, stem: str) -> Optional[Tuple[str, dict]]:
    json_path = os.path.join(scene_dir, f"{stem}.json")
    png_path = os.path.join(scene_dir, f"{stem}.png")
    if not os.path.isfile(json_path) or not os.path.isfile(png_path):
        return None
    if stem.endswith("_overlay"):
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return png_path, payload


def _nearest_object_landmark(obj: dict, landmarks: List[dict], payload: dict,
                             width: float):
    oc = _bbox_center(obj["bbox"])
    if oc is None:
        return None, None
    best = None
    best_d = None
    for lm in landmarks:
        if lm is obj:
            continue
        if lm.get("class") == obj.get("class"):
            continue
        lc = _bbox_center(lm["bbox"])
        if lc is None:
            continue
        d = math.hypot(oc[0] - lc[0], oc[1] - lc[1])
        if best_d is None or d < best_d:
            best_d, best = d, lm
    if best is None:
        return None, None
    rel = _world_or_pixel_relation(
        obj, best, _relation(oc, _bbox_center(best["bbox"]), width), payload)
    return best, rel


def _object_relation_unique(obj: dict, lm: dict, rel: str, objects: List[dict],
                            landmarks: List[dict], payload: dict,
                            width: float) -> bool:
    """Keep a row only when the referring expression uniquely IDs this instance."""
    same = [
        other for other in objects
        if other.get("class") == obj.get("class") and other is not obj
    ]
    if not same:
        return True
    if rel in ("near", "beside"):
        return False
    lm_cls = lm.get("class")
    for other in same:
        other_lm, other_rel = _nearest_object_landmark(
            other, landmarks, payload, width)
        if other_lm is None:
            continue
        if other_rel == rel and other_lm.get("class") == lm_cls:
            return False
    return True


def _pairs_for_payload(payload: dict, rng: random.Random, max_per_image: int):
    dets = list(payload.get("detections") or [])
    objects = [
        d for d in dets
        if (d.get("kind") or "object") != "landmark"
        and not _is_person(str(d.get("class") or ""))
        and not _is_long_umbrella(str(d.get("class") or ""))
        and d.get("class")
        and _bbox_center(d.get("bbox"))
    ]
    landmarks = _payload_landmarks(dets, objects)
    if not objects or not landmarks:
        return []

    counts: Dict[str, int] = {}
    for o in objects:
        counts[o["class"]] = counts.get(o["class"], 0) + 1

    width = float((payload.get("image") or {}).get("width") or 1280)
    scored = []
    for obj in objects:
        best, rel = _nearest_object_landmark(obj, landmarks, payload, width)
        if best is None:
            continue
        if not _object_relation_unique(
                obj, best, rel, objects, landmarks, payload, width):
            continue
        hard = counts.get(obj["class"], 1) > 1
        scored.append((hard, rng.random(), obj, best, rel))

    scored.sort(key=lambda t: (t[4] != "beside", not t[0], t[1]))
    picked = scored[: max(1, max_per_image)]
    rows = []
    for hard, _, obj, lm, rel in picked:
        tmpl = rng.choice(TEMPLATES[rel])
        instruction = tmpl.format(obj=_display(obj["class"]), lm=_display(lm["class"]))
        rows.append((instruction, obj, lm, rel, hard))
    return rows


def _payload_landmarks(dets, objects):
    object_classes = {o.get("class") for o in objects}
    landmarks = [
        d for d in dets
        if str(d.get("class") or "") in LANDMARK_TYPES
        and not _is_long_umbrella(str(d.get("class") or ""))
        and not _is_person(str(d.get("class") or ""))
        and _bbox_center(d.get("bbox"))
        and (d.get("kind") == "landmark" or str(d.get("class")) not in object_classes)
    ]
    if not landmarks:
        landmarks = [
            d for d in dets
            if d not in objects
            and str(d.get("class") or "") in LANDMARK_TYPES
            and _bbox_center(d.get("bbox"))
            and not _is_person(str(d.get("class") or ""))
            and not _is_long_umbrella(str(d.get("class") or ""))
        ]
    return landmarks


def _sar_pairs_for_payload(payload: dict, rng: random.Random, max_per_image: int,
                          require_recipe_target: bool = False):
    """Person rows: situation from pose × support. Recipe frames emit the target only."""
    dets = list(payload.get("detections") or [])
    people = [
        d for d in dets
        if _is_person(str(d.get("class") or ""))
        and d.get("class")
        and _bbox_center(d.get("bbox"))
    ]
    if not people:
        return []
    objects = [
        d for d in dets
        if (d.get("kind") or "object") != "landmark"
        and not _is_person(str(d.get("class") or ""))
        and not _is_long_umbrella(str(d.get("class") or ""))
        and d.get("class")
    ]
    landmarks = _payload_landmarks(dets, objects)
    recipe_targets = [p for p in people if p.get("recipe_id")]
    if recipe_targets:
        people_iter = recipe_targets
        max_per_image = 1
    elif require_recipe_target:
        return []
    else:
        people_iter = people

    n_people = len(people)
    scored = []
    for person in people_iter:
        best = _pick_support_landmark(person, landmarks, payload) if landmarks else None
        rel = _support_rel(person, best) if best is not None else None
        if rel is None and best is not None and _beside_fallen_vehicle(person, best):
            rel = "beside"
        if _is_unlabeled_lying(person, rel):
            continue
        sit = _sar_situation_for(person, best, rel, rng, landmarks=landmarks)
        lm_cls = str((best or {}).get("class") or "")
        on_wrong = rel in ("on", "above") and _is_wrong_place(lm_cls)
        distress = sit in ("accident", "emergency")
        hard = n_people > 1
        rec = _load_recipes().get(str(person.get("recipe_id") or ""), {})
        command = str(person.get("command") or rec.get("command") or "")
        scored.append((
            0 if on_wrong else 1 if distress else 2 if sit == "abnormal" else 3,
            0 if hard else 1,
            rng.random(),
            person, best, sit, hard, rel, command,
        ))

    scored.sort()
    picked = scored[: max(1, max_per_image)]
    rows = []
    for _, _, _, person, lm, sit, hard, rel, command in picked:
        instruction = command or _sar_instruction(person, lm, rel, sit, rng)
        rows.append((instruction, person, lm, sit, hard, rel))
    return rows


def _box_overlap(a: dict, b: Optional[dict]) -> float:
    if not b:
        return 0.0
    ab, bb = _bbox4(a), _bbox4(b)
    if ab is None or bb is None:
        return 0.0
    overlap = max(0.0, min(ab[2], bb[2]) - max(ab[0], bb[0]))
    width = max(1.0, min(ab[2] - ab[0], bb[2] - bb[0]))
    return min(1.0, overlap / width)


def _gt_candidate_rows(people: List[dict], camera_id: str, payload: dict) -> List[str]:
    dets = list(payload.get("detections") or [])
    objects = [
        d for d in dets
        if (d.get("kind") or "object") != "landmark"
        and not _is_person(str(d.get("class") or ""))
        and not _is_long_umbrella(str(d.get("class") or ""))
        and d.get("class")
    ]
    landmarks = _payload_landmarks(dets, objects)
    id_map = person_candidate_ids(people, camera_id)
    ordered = sorted(people, key=_det_sort_key)
    rows = []
    for person in ordered:
        cid = id_map.get(id(person), person_candidate_id(camera_id, str(person.get("class") or "person"), 0))
        best = _pick_support_landmark(person, landmarks, payload) if landmarks else None
        dist = _xy_m(person, best) if best else None
        overlap = _box_overlap(person, best)
        rows.append(
            " | ".join(
                (
                    cid,
                    f"camera={camera_id}",
                    f"pose={person.get('class')}",
                    "confidence=1.000",
                    f"anchor={best.get('class') if best else 'none'}",
                    "anchor_id=none",
                    (
                        f"distance={dist:.4f}"
                        if dist is not None
                        else "distance=none"
                    ),
                    f"overlap={overlap:.4f}",
                )
            )
        )
    return rows


def _sar_user_prompt(instruction: str, candidate_ids: Iterable[str],
                     camera_id: str = "cam",
                     requested: Optional[str] = None,
                     landmark: Optional[str] = None,
                     scene_hint: Optional[str] = None,
                     people: Optional[List[dict]] = None,
                     payload: Optional[dict] = None) -> str:
    if people and payload is not None:
        rows = _gt_candidate_rows(people, camera_id, payload)
    else:
        rows = [
            f"{cid} | camera={camera_id} | pose=unknown | confidence=1.000 | "
            f"anchor=none | anchor_id=none | distance=none | overlap=0.0000"
            for cid in candidate_ids
        ]
    return format_sar_verifier_prompt(
        instruction, requested, landmark, scene_hint, rows)


def _sample(png_path: str, payload: dict, camera_id: str, instruction: str,
            obj: dict, lm: dict, fmt: str, rel: Optional[str] = None,
            situation: Optional[str] = None, target_type: Optional[str] = None,
            candidate_id: Optional[str] = None,
            candidate_ids: Optional[Iterable[str]] = None,
            people: Optional[List[dict]] = None,
            requested: Optional[str] = None,
            scene_hint: Optional[str] = None) -> dict:
    image_rel = os.path.abspath(png_path)
    if situation is not None:
        ids = tuple(candidate_ids or ())
        lm_name = str((lm or {}).get("class") or "") or None
        rec = _load_recipes().get(str((obj or {}).get("recipe_id") or ""), {})
        req = requested or obj.get("situation") or rec.get("situation") or situation
        hint = scene_hint if scene_hint is not None else (
            obj.get("scene_hint") or rec.get("scene_hint"))
        prompt = "<image>\n" + _sar_user_prompt(
            instruction, ids, camera_id=camera_id, requested=req,
            landmark=lm_name or rec.get("support_landmark"),
            scene_hint=hint, people=people, payload=payload)
        text = _numbered_candidate(str(candidate_id), ids)
    else:
        interp = {
            "target_type": target_type or obj["class"],
            "landmark": lm["class"] if lm else "",
        }
        if rel is not None:
            interp["relation"] = rel
        answer = {
            "camera_id": camera_id,
            "interpretation": interp,
        }
        text = json.dumps(answer, ensure_ascii=False)
        prompt = f"<image>\n{instruction}"
    if fmt == "sharegpt":
        return {
            "images": [image_rel],
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": text},
            ],
        }
    return {
        "images": [image_rel],
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ],
    }


def _runtime_round(png_path: str, payload: dict, camera_id: str):
    """Turn GT into the same finite candidates and pixels used by runtime boards."""
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(png_path).convert("RGB"))
    height, width = rgb.shape[:2]
    objects: List[ObjectCandidate] = []
    people: List[PersonCandidate] = []
    object_by_det: Dict[int, ObjectCandidate] = {}
    person_by_det: Dict[int, PersonCandidate] = {}
    class_counts: Counter = Counter()
    people_gt = [
        det for det in payload.get("detections") or []
        if _is_person(str(det.get("class") or "")) and _bbox4(det)
    ]
    person_ids = person_candidate_ids(people_gt, camera_id)
    for det in sorted(payload.get("detections") or [], key=_det_sort_key):
        box = _bbox4(det)
        class_name = str(det.get("class") or "")
        if not box or not class_name or _is_long_umbrella(class_name):
            continue
        if _is_person(class_name):
            candidate = PersonCandidate(
                person_ids[id(det)], box, class_name, 1.0,
                camera_id, (width, height),
            )
            people.append(candidate)
            person_by_det[id(det)] = candidate
            continue
        index = class_counts[class_name]
        class_counts[class_name] += 1
        candidate = ObjectCandidate(
            f"{camera_id}:{class_name}:{index}", class_name, box, 1.0,
            camera_id, (width, height),
        )
        objects.append(candidate)
        object_by_det[id(det)] = candidate
    round_ = CctvDetectionRound(
        objects=objects,
        people=people,
        frames={camera_id: CctvFrame(camera_id, width, height, rgb)},
        camera_ids=[camera_id],
    )
    return round_, object_by_det, person_by_det


def _runtime_scene_round(cameras):
    """Combine camera GT into the all-camera board used in production."""
    objects = []
    people = []
    frames = {}
    object_by_det = {}
    person_by_det = {}
    for png_path, payload, camera_id in cameras:
        round_, camera_objects, camera_people = _runtime_round(
            png_path, payload, camera_id
        )
        objects.extend(round_.objects)
        people.extend(round_.people)
        frames.update(round_.frames)
        object_by_det.update(camera_objects)
        person_by_det.update(camera_people)
    return (
        CctvDetectionRound(
            objects=objects,
            people=people,
            frames=frames,
            camera_ids=sorted(frames),
        ),
        object_by_det,
        person_by_det,
    )


def _answer_at(items, answer_id: str, slot: int, get_id):
    """Move the labelled answer without changing the candidate set."""
    ordered = list(items)
    index = next((i for i, item in enumerate(ordered) if get_id(item) == answer_id), None)
    if index is None:
        return tuple(ordered)
    answer = ordered.pop(index)
    ordered.insert(slot % (len(ordered) + 1), answer)
    return tuple(ordered)


def _runtime_sample(
        png_path: str, payload: dict, camera_id: str, instruction: str,
        target: dict, fmt: str, out_dir: str, asset_key: str,
        *, situation: Optional[str] = None, answer_slot: Optional[int] = None,
        runtime=None) -> Optional[dict]:
    """Build and persist the production Qwen board, prompt, and finite answer."""
    round_, objects, people = runtime or _runtime_round(png_path, payload, camera_id)
    if answer_slot is None:
        answer_slot = int(hashlib.sha1(asset_key.encode("utf-8")).hexdigest()[:8], 16)
    try:
        command = parse_command(instruction)
    except CommandParseError:
        return None
    if situation is not None:
        chosen = people.get(id(target))
        ranked = candidate_evidence(command, round_, instruction)
        evidence = shortlist_for_verifier(ranked, command, instruction)
        if chosen is None or chosen.candidate_id not in {
                item.person.candidate_id for item in evidence}:
            return None
        evidence = _answer_at(
            evidence, chosen.candidate_id, answer_slot,
            lambda item: item.person.candidate_id,
        )
        board = _assemble_board(
            instruction, command, round_, evidence, len(round_.people),
            pack_cameras=True,
        )
        ids = board.candidate_ids
        answer = f"P{ids.index(chosen.candidate_id) + 1:02d}"
    else:
        chosen = objects.get(id(target))
        ranked = rank_object_candidates(
            command, round_, command_text=instruction,
            visual_proposals=True,
        )
        if chosen is None or chosen.candidate_id not in {
                item.target.candidate_id for item in ranked}:
            return None
        ranked = _answer_at(
            ranked, chosen.candidate_id, answer_slot,
            lambda item: item.target.candidate_id,
        )
        board = build_object_board(
            instruction, command, round_, ranked,
            max_crops=len(ranked), pack_cameras=True,
        )
        answer = chosen.candidate_id

    digest = hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", asset_key)
    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    image_paths = []
    for index, image in enumerate(board.images, start=1):
        name = f"{stem}_{digest}_{index}.png"
        image.save(os.path.join(image_dir, name))
        image_paths.append(f"images/{name}")
    prompt = "\n".join("<image>" for _ in image_paths) + "\n" + board.prompt
    if fmt == "sharegpt":
        return {
            "images": image_paths,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
        }
    return {
        "images": image_paths,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
    }


def _coverage_missing(
        object_counts: Counter, landmark_counts: Counter,
        relation_counts: Counter, situation_counts: Counter,
        recipe_counts: Counter, camera_counts: Counter,
        min_cameras: int) -> List[str]:
    recipes = _load_recipes()
    required_situations = {
        str(recipe.get("situation")) for recipe in recipes.values()
        if recipe.get("situation")
    }
    checks = (
        ("object", set(MISSION_OBJECT_TYPES) - set(_SKIPPED_OBJECTS), object_counts),
        ("landmark", set(LANDMARK_TYPES), landmark_counts),
        ("relation", set(TEMPLATES), relation_counts),
        ("situation", required_situations, situation_counts),
        ("recipe", set(recipes), recipe_counts),
    )
    missing = [
        f"{label}:{name}"
        for label, required, observed in checks
        for name in sorted(required - set(observed))
    ]
    if len(camera_counts) < min_cameras:
        missing.append(f"cameras:{len(camera_counts)}/{min_cameras}")
    return missing


def _camera_groups(scenes):
    """Keep real multi-camera scenes; bundle one-camera captures by camera ID."""
    loaded = []
    skipped = 0
    for scene_id, scene_dir in scenes:
        cameras = []
        for name in sorted(os.listdir(scene_dir)):
            if not name.endswith(".json") or name.endswith("_overlay.json"):
                continue
            stem = os.path.splitext(name)[0]
            pair = _load_pair(scene_dir, stem)
            if pair is None:
                skipped += 1
                continue
            png_path, payload = pair
            cameras.append((
                png_path, payload, str(payload.get("frame_id") or stem)
            ))
        if cameras:
            loaded.append((scene_id, cameras))
    if not loaded or any(len(cameras) != 1 for _, cameras in loaded):
        return loaded, skipped

    by_camera = defaultdict(list)
    for scene_id, cameras in loaded:
        by_camera[cameras[0][2]].append((scene_id, cameras[0]))
    if len(by_camera) < 2:
        return loaded, skipped
    bundle_count = min(len(items) for items in by_camera.values())
    groups = []
    camera_ids = sorted(by_camera)
    for index in range(bundle_count):
        groups.append((
            f"bundle_{index:05d}",
            [by_camera[camera_id][index][1] for camera_id in camera_ids],
        ))
    skipped += len(loaded) - bundle_count * len(camera_ids)
    return groups, skipped


def convert(in_dir: str, out_dir: str, val_frac: float, max_per_image: int,
            seed: int, fmt: str, max_sar_per_image: int = 2,
            require_complete: bool = False, min_cameras: int = 6,
            max_groups: int = 0) -> None:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    scenes = list(_iter_scenes(in_dir))
    if not scenes:
        # Allow passing trainer_output/ (scenario subdirs) or a single scene root.
        nested = []
        try:
            for name in sorted(os.listdir(in_dir)):
                path = os.path.join(in_dir, name)
                if os.path.isdir(path):
                    nested.extend(_iter_scenes(path))
        except OSError:
            pass
        scenes = nested
    if not scenes:
        raise SystemExit(f"no scene_* directories under {in_dir}")

    train_sar: List[dict] = []
    val_sar: List[dict] = []
    train_obj: List[dict] = []
    val_obj: List[dict] = []
    n_hard = 0
    n_skip = 0
    n_sar = 0
    sit_counts: Counter = Counter()
    lm_counts: Counter = Counter()
    obj_counts: Counter = Counter()
    rel_counts: Counter = Counter()
    recipe_counts: Counter = Counter()
    camera_counts: Counter = Counter()
    scene_groups, load_skips = _camera_groups(scenes)
    if max_groups > 0:
        scene_groups = scene_groups[:max_groups]
    n_skip += load_skips
    n_cut = (
        max(1, int(round(len(scene_groups) * (1.0 - val_frac))))
        if len(scene_groups) > 1 else len(scene_groups)
    )

    for idx, (scene_id, cameras) in enumerate(scene_groups):
        recipe_scene = any(
            det.get("recipe_id")
            for _, payload, _ in cameras
            for det in (payload.get("detections") or [])
        )
        scene_runtime = _runtime_scene_round(cameras)
        obj_bucket = train_obj if idx < n_cut else val_obj
        sar_bucket = train_sar if idx < n_cut else val_sar
        for png_path, payload, camera_id in cameras:
            camera_counts[camera_id] += 1
            obj_rows = _pairs_for_payload(payload, rng, max_per_image)
            sar_rows = _sar_pairs_for_payload(
                payload, rng, max_sar_per_image,
                require_recipe_target=recipe_scene)
            if not obj_rows and not sar_rows:
                n_skip += 1
                continue
            for instruction, obj, lm, rel, hard in obj_rows:
                box_key = "_".join(str(round(value)) for value in obj["bbox"])
                row = _runtime_sample(
                    png_path, payload, camera_id, instruction, obj, fmt,
                    out_dir, f"s{scene_id}_{camera_id}_obj_{box_key}",
                    runtime=scene_runtime,
                )
                if row is None:
                    n_skip += 1
                    continue
                obj_bucket.append(row)
                lm_counts[str(lm.get("class") or "?")] += 1
                obj_counts[str(obj.get("class") or "?")] += 1
                rel_counts[str(rel)] += 1
                if hard:
                    n_hard += 1
            for instruction, person, lm, sit, hard, rel in sar_rows:
                box_key = "_".join(str(round(value)) for value in person["bbox"])
                row = _runtime_sample(
                    png_path, payload, camera_id, instruction, person, fmt,
                    out_dir, f"s{scene_id}_{camera_id}_sar_{box_key}",
                    situation=sit, runtime=scene_runtime,
                )
                if row is None:
                    n_skip += 1
                    continue
                sar_bucket.append(row)
                lm_counts[str(lm.get("class") or "none") if lm else "none"] += 1
                sit_counts[sit] += 1
                recipe_id = str(person.get("recipe_id") or "")
                if recipe_id:
                    recipe_counts[recipe_id] += 1
                n_sar += 1
                if hard:
                    n_hard += 1

    def _write(name: str, rows: List[dict]) -> str:
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    train_sar_path = _write("train_sar.jsonl", train_sar)
    val_sar_path = _write("val_sar.jsonl", val_sar)
    train_obj_path = _write("train_obj.jsonl", train_obj)
    val_obj_path = _write("val_obj.jsonl", val_obj)
    train_all = train_sar + train_obj
    val_all = val_sar + val_obj
    random.Random(seed).shuffle(train_all)
    random.Random(seed).shuffle(val_all)
    train_path = _write("train.jsonl", train_all)
    val_path = _write("val.jsonl", val_all)
    n_total = (len(train_sar) + len(val_sar) + len(train_obj) + len(val_obj))
    missing = _coverage_missing(
        obj_counts, lm_counts, rel_counts, sit_counts, recipe_counts,
        camera_counts, min_cameras,
    )
    info = {
        "format": fmt,
        "source": os.path.abspath(in_dir),
        "num_train_sar": len(train_sar),
        "num_val_sar": len(val_sar),
        "num_train_obj": len(train_obj),
        "num_val_obj": len(val_obj),
        "num_hard_same_class": n_hard,
        "num_sar": n_sar,
        "hard_frac": round(n_hard / n_total, 4) if n_total else 0.0,
        "skipped_cameras": n_skip,
        "max_per_image": max_per_image,
        "max_sar_per_image": max_sar_per_image,
        "landmark_class_counts": dict(lm_counts.most_common()),
        "object_class_counts": dict(obj_counts.most_common()),
        "object_relation_counts": dict(rel_counts.most_common()),
        "sar_situation_counts": dict(sit_counts.most_common()),
        "sar_recipe_counts": dict(recipe_counts.most_common()),
        "camera_counts": dict(camera_counts.most_common()),
        "coverage_complete": not missing,
        "coverage_missing": missing,
        "assistant": "SAR: Pxx; objects: TARGET_ID (production finite-board grammars)",
        "situation_rules": RULES_VERSION,
        "notes": (
            "Train on both train_sar.jsonl and train_obj.jsonl. Validation files "
            "remain scene-disjoint. Prompts, packed annotated images, detector "
            "metadata, and finite answers match the production Qwen boards."
        ),
    }
    info_path = os.path.join(out_dir, "coverage.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    if fmt == "sharegpt":
        columns = {"messages": "conversations", "images": "images"}
        tags = {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        }
    else:
        columns = {"messages": "messages", "images": "images"}
        tags = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        }
    dataset_info = {
        name: {
            "file_name": filename,
            "formatting": "sharegpt",
            "columns": columns,
            "tags": tags,
        }
        for name, filename in (
            ("marc_qwen_train", "train.jsonl"),
            ("marc_qwen_val", "val.jsonl"),
        )
    }
    registry_path = os.path.join(out_dir, "dataset_info.json")
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)
    print(f"[qwen] sar_train={len(train_sar)} sar_val={len(val_sar)} "
          f"obj_train={len(train_obj)} obj_val={len(val_obj)} "
          f"hard={n_hard} hard_frac={info['hard_frac']} skipped_cam={n_skip}")
    if lm_counts:
        lm_txt = ", ".join(f"{k}={v}" for k, v in lm_counts.most_common())
        print(f"[qwen] landmarks: {lm_txt}")
    if obj_counts:
        obj_txt = ", ".join(f"{k}={v}" for k, v in obj_counts.most_common())
        print(f"[qwen] objects: {obj_txt}")
    if sit_counts:
        sit_txt = ", ".join(f"{k}={v}" for k, v in sit_counts.most_common())
        print(f"[qwen] sar situations: {sit_txt}")
    print(
        "[qwen] coverage: "
        + ("complete" if not missing else "missing " + ", ".join(missing))
    )
    print(f"[qwen] {train_sar_path}")
    print(f"[qwen] {val_sar_path}")
    print(f"[qwen] {train_obj_path}")
    print(f"[qwen] {val_obj_path}")
    print(f"[qwen] {train_path}")
    print(f"[qwen] {val_path}")
    print(f"[qwen] {info_path}")
    print(f"[qwen] {registry_path}")
    if require_complete and missing:
        raise SystemExit(
            "dataset coverage incomplete; generate more scenes and rerun export"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="dataset-gen output (scenario dir or trainer_output/)")
    ap.add_argument("--out", dest="out_dir", required=True,
                    help="JSONL output directory")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-per-image", type=int, default=4)
    ap.add_argument("--max-sar-per-image", type=int, default=2,
                    help="max SAR (person) rows per camera image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=("llama-factory", "sharegpt"),
                    default="llama-factory",
                    help="llama-factory = messages[]; sharegpt = conversations[]")
    ap.add_argument("--require-complete", action="store_true",
                    help="fail after export if a task category has no examples")
    ap.add_argument("--min-cameras", type=int, default=6,
                    help="minimum distinct viewpoints required by --require-complete")
    ap.add_argument("--max-groups", type=int, default=0,
                    help="optional cap after one-camera captures are bundled")
    args = ap.parse_args()
    if not os.path.isdir(args.in_dir):
        raise SystemExit(f"input dir not found: {args.in_dir}")
    convert(args.in_dir, args.out_dir, args.val_frac, args.max_per_image,
            args.seed, args.format, max_sar_per_image=args.max_sar_per_image,
            require_complete=args.require_complete,
            min_cameras=max(1, args.min_cameras),
            max_groups=max(0, args.max_groups))


if __name__ == "__main__":
    sys.exit(main())
