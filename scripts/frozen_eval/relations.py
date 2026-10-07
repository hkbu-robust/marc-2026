"""Our relation rule for held-out commands and gold relations (#87).

This rule is ours, not the organisers'. The organisers' scoring code is not
available; their public problems only show how loosely they use the words
(object origin to landmark origin, planar):

- ``beside``: 1.43-1.85 m; ``near``: 1.15-5.59 m; ``on``: up to 0.31 m apart and
  0.37-1.09 m higher (``mug on picnic table`` is ``on`` at +1.09 m).

Relations are predicates, not one label per pair, so one target can satisfy
several (a mug on a table is also beside and near it):

- ``on``: the converter's 2D stacking test (``_stacked_on_top``) or its world
  climb test (``_climb_from_world``).
- ``above``: ``on`` and at least ``_ABOVE_DZ_M`` (0.7 m) higher, as in training.
- ``beside``: planar origin distance <= ``BESIDE_M``; ``near``: <= ``NEAR_M``.
  Both radii cover every public case.
- ``left``/``right``/``front``/``behind``: the converter's camera-relative test
  (``_camera_relative_rel``), only when not stacked and within ``NEAR_M``.

A command picks out its target when the target is the only instance of its
class that satisfies the relation with some instance of the landmark class.
Unlike the training converter (``_object_relation_unique``), ``near``/``beside``
cases with same-class distractors are kept whenever this check passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from demo.vlm_dataset.cctv_gt_to_qwen import (
    DISPLAY,
    _ABOVE_DZ_M,
    _camera_relative_rel,
    _climb_from_world,
    _stacked_on_top,
)

RULE_VERSION = "heldout-relations-v1"
RELATIONS = ("beside", "near", "on", "above", "left", "right", "front", "behind")
BESIDE_M = 2.0
NEAR_M = 6.0
# SAR "next to"/"near" recipes: the public sarH2 person lies 3.22 m from the bicycle.
SAR_BESIDE_M = 3.5
# SAR on/above: the generator seats the person on the support's spawn pose, then
# physics can roll a trash can away (same +0.41 m height, 0.2-1.2 m off). Public
# on/above people are 0.26-1.04 m from the support origin; a trash can is ~0.3 m wide.
SAR_ON_MAX_M = {"trash_can": 0.5}
SAR_ON_DEFAULT_MAX_M = 1.5
# A person cut at the top edge cannot be checked as on top of anything.
TOP_CLIP_PX = 8.0

PHRASE = {
    "beside": "beside", "near": "near", "on": "on", "above": "above",
    "left": "to the left of", "right": "to the right of",
    "front": "in front of", "behind": "behind",
}


def is_person(det: dict) -> bool:
    return str(det.get("class") or "").startswith("person_")


def planar_m(a: dict, b: dict) -> float:
    pa, pb = a["world_xyz"], b["world_xyz"]
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def is_stacked(obj: dict, lm: dict) -> bool:
    return _stacked_on_top(obj, lm) or _climb_from_world(obj, lm)


def relations_between(obj: dict, lm: dict, payload: dict) -> set[str]:
    """Every relation word our rule allows for ``obj`` w.r.t. landmark ``lm``."""
    if not obj.get("world_xyz") or not lm.get("world_xyz"):
        return set()
    out: set[str] = set()
    distance = planar_m(obj, lm)
    stacked = is_stacked(obj, lm)
    if stacked:
        out.add("on")
        if obj["world_xyz"][2] - lm["world_xyz"][2] >= _ABOVE_DZ_M:
            out.add("above")
    if distance <= BESIDE_M:
        out.add("beside")
    if distance <= NEAR_M:
        out.add("near")
        if not stacked:
            direction = _camera_relative_rel(obj, lm, payload)
            if direction:
                out.add(direction)
    return out


def object_command(target_class: str, relation: str, landmark_class: str) -> str:
    obj = DISPLAY.get(target_class, target_class.replace("_", " "))
    lm = DISPLAY.get(landmark_class, landmark_class.replace("_", " "))
    return f"Find the {obj} {PHRASE[relation]} the {lm}."


@dataclass(frozen=True)
class ObjectChoice:
    """One uniquely described object target."""

    target: dict
    relation: str
    landmark_class: str
    anchors: tuple[dict, ...]  # landmark instances of that class satisfying the relation
    distractor_present: bool


def object_choices(payload: dict, objects: list[dict], landmarks: list[dict]) -> dict[str, list]:
    """For each object instance: every (relation, landmark class) that picks it out uniquely."""
    by_class: dict[str, list[dict]] = {}
    for obj in objects:
        by_class.setdefault(obj["class"], []).append(obj)
    rel_cache = {
        (o["instance_id"], lm["instance_id"]): relations_between(o, lm, payload)
        for o in objects for lm in landmarks
    }
    choices: dict[str, list[ObjectChoice]] = {}
    for obj in objects:
        same_class = by_class[obj["class"]]
        options = []
        for lm_class in sorted({lm["class"] for lm in landmarks}):
            lms = [lm for lm in landmarks if lm["class"] == lm_class]
            for relation in RELATIONS:
                anchors = tuple(
                    lm for lm in lms if relation in rel_cache[(obj["instance_id"], lm["instance_id"])])
                if not anchors:
                    continue
                satisfiers = [
                    o for o in same_class
                    if any(relation in rel_cache[(o["instance_id"], lm["instance_id"])] for lm in lms)
                ]
                if len(satisfiers) != 1:
                    continue
                anchors = tuple(sorted(anchors, key=lambda lm: planar_m(obj, lm)))
                options.append(ObjectChoice(obj, relation, lm_class, anchors, len(same_class) > 1))
        choices[obj["instance_id"]] = options
    return choices


@dataclass(frozen=True)
class SarCheck:
    """Whether a recipe target matches its recipe command, and its anchors."""

    ok: bool
    reason: str
    anchors: tuple[dict, ...] = ()
    distractor_present: bool = False


def sar_check(person: dict, recipe: dict, people: list[dict], landmarks: list[dict]) -> SarCheck:
    """Check the recipe target against the recipe relation; reject ambiguous scenes.

    - pose must equal the recipe pose;
    - ``on``/``above`` recipes: stacked on a support-class landmark, within
      ``SAR_ON_MAX_M`` of it, and not cut at the top of the frame;
    - ``beside`` recipes with a support class: within ``SAR_BESIDE_M`` of one;
    - recipes without a support class: anchor = nearest landmark (any class);
    - ambiguous if another person with the same pose also satisfies the check.
    """
    if not person.get("world_xyz"):
        return SarCheck(False, "no_world_xyz")
    if person["class"] != recipe["pose"]:
        return SarCheck(False, "pose_mismatch")
    support = recipe.get("support_class") or recipe.get("support_landmark")
    relation = recipe.get("relation")
    others = [p for p in people if p is not person]
    distractor_present = bool(others)

    def satisfying(candidate: dict) -> tuple[dict, ...]:
        lms = [lm for lm in landmarks if lm["class"] == support and lm.get("world_xyz")]
        if relation in ("on", "above"):
            limit = SAR_ON_MAX_M.get(support, SAR_ON_DEFAULT_MAX_M)
            return tuple(lm for lm in lms
                         if is_stacked(candidate, lm) and planar_m(candidate, lm) <= limit)
        return tuple(lm for lm in lms if planar_m(candidate, lm) <= SAR_BESIDE_M)

    if relation in ("on", "above") and float(person["bbox"][1]) <= TOP_CLIP_PX:
        return SarCheck(False, "cut_at_top", (), distractor_present)
    if support:
        anchors = satisfying(person)
        if not anchors:
            return SarCheck(False, f"relation_{relation}_not_satisfied", (), distractor_present)
        rivals = [p for p in others if p["class"] == person["class"] and satisfying(p)]
    else:
        placed = [lm for lm in landmarks if lm.get("world_xyz")]
        if not placed:
            return SarCheck(False, "no_landmark_for_anchor", (), distractor_present)
        anchors = (min(placed, key=lambda lm: planar_m(person, lm)),)
        rivals = [p for p in others if p["class"] == person["class"]]
    if rivals:
        return SarCheck(False, "ambiguous_same_pose", (), distractor_present)
    anchors = tuple(sorted(anchors, key=lambda lm: planar_m(person, lm)))
    return SarCheck(True, "ok", anchors, distractor_present)
