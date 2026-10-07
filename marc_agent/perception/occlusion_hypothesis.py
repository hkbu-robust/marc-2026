"""An explicit hypothesis for a target hidden by a person or vehicle at its landmark (#117).

When the command's relation is satisfied by **no** detected candidate of the target
class (to any detected instance of the named or implied landmark class, in world
metres), the target may be hidden: under or behind a large object that covers or
abuts the landmark. This module then adds, per landmark instance L with such an
occluder O in L's camera, a hypothesis candidate: O's box under the target's class,
``source="occlusion_hypothesis"``, confidence 0. Selection prefers it (it is the only
candidate consistent with the command) and reports it as its own method; it never
fires while any detection satisfies the relation.

Relation radii are the #87 rule (``scripts/frozen_eval/relations.py``): planar origin
distance <= 2.0 m for beside, <= 6.0 m for near and the camera-relative relations.
On/above cannot be tested from projected points, so they use the beside radius with
the target projected on the support's raised plane, as the runtime projects it.
The occluder rule (person or vehicle, confidence >= 0.5, box overlapping L's box or
ground point within 1.5 m of L) is the ticket's, not tuned.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import replace
from typing import Optional

from marc_agent.command_parser import (
    VEHICLE_LANDMARKS,
    ParsedCommand,
    Relation,
    implicit_anchor_classes,
)
from marc_agent.experiment_log import note_decision
from marc_agent.perception.candidate_ids import object_candidate_id
from marc_agent.types import CctvDetectionRound, ObjectCandidate

log = logging.getLogger("marc_agent.occlusion_hypothesis")

SOURCE = "occlusion_hypothesis"
BESIDE_M = 2.0
NEAR_M = 6.0
OCCLUDER_MIN_CONF = 0.5
OCCLUDER_REACH_M = 1.5
_RADIUS_M = {
    Relation.BESIDE: BESIDE_M,
    Relation.ON: BESIDE_M,
    Relation.ABOVE: BESIDE_M,
}


def enabled() -> bool:
    return os.environ.get("MARC_OCCLUSION_HYPOTHESIS", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def is_hypothesis(item) -> bool:
    return getattr(item, "source", None) == SOURCE


def add_hypotheses(
    parsed: ParsedCommand,
    round_: CctvDetectionRound,
    projector,
    command_text: str = "",
) -> CctvDetectionRound:
    """``round_`` plus hypothesis candidates, when no detection satisfies the relation."""
    if not enabled() or projector is None or parsed.target_type == "person":
        return round_
    classes = implicit_anchor_classes(parsed, command_text)
    if not classes:
        return round_
    landmarks = [item for item in round_.objects if item.class_name in classes]
    if not landmarks:
        return round_
    relation = parsed.relation or Relation.NEAR
    radius = _RADIUS_M.get(relation, NEAR_M)
    landmark_xy = {item.candidate_id: _xy(projector.project_landmark(
        item.camera_id, item.bbox_xyxy, item.class_name, item.image_size)) for item in landmarks}
    targets = [item for item in round_.objects if item.class_name == parsed.target_type]
    for target in targets:
        for landmark in landmarks:
            target_xy = _xy(projector.project_contact(
                target.camera_id, target.bbox_xyxy, relation=relation,
                support_class=landmark.class_name))
            # A visible candidate satisfies the command, or might (not projectable).
            if target_xy is None or _within(target_xy, landmark_xy[landmark.candidate_id], radius):
                return round_
    added: list[ObjectCandidate] = []
    counts = {}
    for item in round_.objects:
        if item.class_name == parsed.target_type:
            counts[item.camera_id] = counts.get(item.camera_id, 0) + 1
    for landmark in sorted(landmarks, key=lambda item: (-item.confidence, item.candidate_id)):
        occluder, dist_m = _occluder(landmark, landmark_xy[landmark.candidate_id], round_, projector)
        if occluder is None:
            continue
        index = counts.get(landmark.camera_id, 0)
        counts[landmark.camera_id] = index + 1
        hypothesis = ObjectCandidate(
            candidate_id=object_candidate_id(landmark.camera_id, parsed.target_type, index),
            class_name=parsed.target_type,
            bbox_xyxy=occluder.bbox_xyxy,
            confidence=0.0,
            camera_id=landmark.camera_id,
            image_size=occluder.image_size,
            source=SOURCE,
        )
        added.append(hypothesis)
        log.info("[occlusion] hypothesis=%s landmark=%s occluder=%s dist_m=%s",
                 hypothesis.candidate_id, landmark.candidate_id, occluder.candidate_id, dist_m)
        note_decision("occlusion_hypothesis", candidate=hypothesis.candidate_id,
                      landmark=landmark.candidate_id, occluder=occluder.candidate_id,
                      dist_m=dist_m)
    if not added:
        note_decision("occlusion_hypothesis_none", landmarks=len(landmarks),
                      reason="no_occluder")
        return round_
    return replace(round_, objects=list(round_.objects) + added)


def choose(parsed: ParsedCommand, round_: CctvDetectionRound, projector, command_text: str = ""):
    """The hypothesis to submit and its landmark: the most confident landmark with one."""
    hypotheses = [item for item in round_.objects if is_hypothesis(item)]
    if not hypotheses or projector is None:
        return None
    classes = implicit_anchor_classes(parsed, command_text)
    landmarks = sorted((item for item in round_.objects if item.class_name in classes),
                       key=lambda item: (-item.confidence, item.candidate_id))
    for landmark in landmarks:
        landmark_xy = _xy(projector.project_landmark(
            landmark.camera_id, landmark.bbox_xyxy, landmark.class_name, landmark.image_size))
        occluder, _ = _occluder(landmark, landmark_xy, round_, projector)
        for hypothesis in hypotheses:
            if (occluder is not None and hypothesis.camera_id == landmark.camera_id
                    and hypothesis.bbox_xyxy == occluder.bbox_xyxy):
                return hypothesis, landmark
    return None


def _occluder(landmark: ObjectCandidate, landmark_xy, round_: CctvDetectionRound, projector):
    """The most confident person or vehicle covering or abutting ``landmark``."""
    best, best_dist = None, None
    pool = [
        *(p for p in round_.people if p.camera_id == landmark.camera_id),
        *(o for o in round_.objects if o.camera_id == landmark.camera_id
          and o.class_name in VEHICLE_LANDMARKS and o.candidate_id != landmark.candidate_id),
    ]
    for item in pool:
        if item.confidence < OCCLUDER_MIN_CONF:
            continue
        if hasattr(item, "pose_class"):
            point = projector.project_contact(item.camera_id, item.bbox_xyxy,
                                              pose_class=item.pose_class)
        else:
            point = projector.project_landmark(item.camera_id, item.bbox_xyxy,
                                               item.class_name, item.image_size)
        dist = _planar(_xy(point), landmark_xy)
        if not (_overlaps(item.bbox_xyxy, landmark.bbox_xyxy)
                or (dist is not None and dist <= OCCLUDER_REACH_M)):
            continue
        if best is None or (item.confidence, item.candidate_id) > (best.confidence, best.candidate_id):
            best, best_dist = item, (round(dist, 2) if dist is not None else None)
    return best, best_dist


def _overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _xy(point) -> Optional[tuple[float, float]]:
    return None if point is None else (float(point[0]), float(point[1]))


def _planar(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _within(a, b, radius: float) -> bool:
    dist = _planar(a, b)
    return dist is not None and dist <= radius
