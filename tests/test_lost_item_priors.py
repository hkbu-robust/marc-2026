"""Unattended-foreground priors for lost-item distractors."""

from __future__ import annotations

import math
import unittest

from marc_agent.perception.lost_item_priors import (
    PriorEvidence,
    bbox_area_frac,
    box_gap,
    distinct_target_count,
    evidence_for_candidate,
    evidence_nearfield,
    parked_vehicle_count,
    prior_key,
    speaker_left_item,
    stall_is_empty,
)
from marc_agent.types import ObjectCandidate, PersonCandidate


def _object(candidate_id, box, confidence=0.8, camera_id="rig_a"):
    return ObjectCandidate(
        candidate_id, "tumbler", box, confidence, camera_id, (640, 480)
    )


def _person(candidate_id, box, pose="person_sitting", camera_id="rig_a"):
    return PersonCandidate(candidate_id, box, pose, 0.9, camera_id, (640, 480))


class _Sel:
    def __init__(self, candidate_id):
        self.target = _object(candidate_id, (0.0, 0.0, 10.0, 10.0))


class LostItemPriorsTests(unittest.TestCase):
    def test_overlapping_boxes_have_zero_gap(self):
        self.assertEqual(0.0, box_gap((10, 10, 40, 40), (20, 20, 50, 50)))
        self.assertGreater(box_gap((10, 10, 20, 20), (80, 80, 90, 90)), 50.0)

    def test_larger_box_has_higher_area_frac(self):
        small = bbox_area_frac((10, 10, 20, 20), (640, 480))
        large = bbox_area_frac((100, 100, 300, 300), (640, 480))
        self.assertGreater(large, small)

    def test_unique_count(self):
        self.assertEqual(1, distinct_target_count([_Sel("a:tumbler:0"), _Sel("a:tumbler:0")]))
        self.assertEqual(2, distinct_target_count([_Sel("a:tumbler:0"), _Sel("a:tumbler:1")]))

    def test_person_overlap_marks_owned(self):
        target = _object("rig_a:tumbler:0", (80.0, 260.0, 140.0, 340.0))
        person = _person("rig_a:person_sitting:0", (40.0, 180.0, 200.0, 400.0))
        evidence = evidence_for_candidate(target, [person])
        self.assertTrue(evidence.owned)

    def test_far_from_person_is_not_owned(self):
        target = _object("rig_a:tumbler:0", (400.0, 260.0, 470.0, 360.0))
        person = _person("rig_a:person_sitting:0", (10.0, 10.0, 80.0, 80.0))
        evidence = evidence_for_candidate(target, [person])
        self.assertFalse(evidence.owned)

    def test_prior_key_demotes_owned_when_asked(self):
        owned = PriorEvidence(True, 0.9, 10.0)
        free = PriorEvidence(False, 0.1, 10.0)
        self.assertLess(
            prior_key(free, demote_owned=True),
            prior_key(owned, demote_owned=True),
        )
        self.assertGreater(
            prior_key(free, demote_owned=False)[0],
            -1,
        )

    def test_nearfield_closer_depth_ranks_higher(self):
        near = evidence_nearfield((1.0, 0.0), 0.4, people_xy=())
        far = evidence_nearfield((3.0, 0.0), 2.0, people_xy=())
        self.assertGreater(near.camera_prox, far.camera_prox)
        self.assertFalse(near.owned)

    def test_nearfield_person_xy_marks_owned(self):
        evidence = evidence_nearfield((0.0, 0.0), 0.5, people_xy=((0.2, 0.0),))
        self.assertTrue(evidence.owned)
        far = evidence_nearfield((0.0, 0.0), 0.5, people_xy=((5.0, 0.0),))
        self.assertFalse(far.owned)
        self.assertTrue(math.isfinite(far.person_gap))

    def test_speaker_left_item_keeps_first_person_leave(self):
        self.assertTrue(
            speaker_left_item(
                "I stopped to throw something in the bin and left my umbrella "
                "leaning against the trash can."
            )
        )
        self.assertTrue(speaker_left_item("Find the sunblock I left next to the bench."))
        self.assertFalse(speaker_left_item("Find the umbrella next to the trash can."))

    def test_stall_empty_unless_person_or_overlapping_vehicle(self):
        vacant = _object("rig_a:tumbler:1", (500.0, 400.0, 530.0, 440.0))
        occupied = _object("rig_a:tumbler:0", (100.0, 200.0, 130.0, 240.0))
        car = ObjectCandidate(
            "rig_a:police_car:0",
            "police_car",
            (90.0, 180.0, 400.0, 420.0),
            0.9,
            "rig_a",
            (640, 480),
        )
        adjacent = ObjectCandidate(
            "rig_a:normal_car_1:0",
            "normal_car_1",
            (10.0, 10.0, 80.0, 80.0),
            0.9,
            "rig_a",
            (640, 480),
        )
        person = _person("rig_a:person_lying_down:0", (70.0, 160.0, 220.0, 300.0), "person_lying_down")
        self.assertFalse(stall_is_empty(occupied, [person], [car, adjacent]))
        self.assertTrue(stall_is_empty(vacant, [person], [car, adjacent]))
        self.assertTrue(stall_is_empty(vacant, [], [adjacent]))

    def test_parked_vehicle_count_counts_every_vehicle_class(self):
        cars = [
            ObjectCandidate(
                "cam:suv:0", "suv", (10.0, 10.0, 80.0, 80.0), 0.9, "cam", (640, 480)
            ),
            ObjectCandidate(
                "cam:taxi:0", "taxi", (90.0, 90.0, 160.0, 160.0), 0.9, "cam", (640, 480)
            ),
            ObjectCandidate(
                "other:suv:0", "suv", (10.0, 10.0, 80.0, 80.0), 0.9, "other", (640, 480)
            ),
        ]
        self.assertEqual(2, parked_vehicle_count("cam", cars))  # a taxi can be parked too
        self.assertEqual(0, parked_vehicle_count("empty", cars))


if __name__ == "__main__":
    unittest.main()
