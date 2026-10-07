"""The occlusion hypothesis for hidden targets (#117)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from marc_agent.command_parser import parse_command
from marc_agent.experiment_log import decision_scope
from marc_agent.perception import occlusion_hypothesis
from marc_agent.perception.object_selection import rank_object_candidates, select_object_candidate
from marc_agent.types import CctvDetectionRound, ObjectCandidate, PersonCandidate

SIZE = (1280, 720)
TABLE = (400.0, 300.0, 600.0, 420.0)
PERSON = (380.0, 250.0, 620.0, 330.0)  # lying across the table
FAR_MUG = (100.0, 600.0, 110.0, 610.0)


class Projector:
    """World points by box: the geometry is spelled out, not derived from a camera."""

    def __init__(self, points):
        self.points = points

    def project_contact(self, _camera, box, **_kwargs):
        return self.points.get(tuple(box))

    def project_landmark(self, _camera, box, *_args, **_kwargs):
        return self.points.get(tuple(box))

    def ground_status(self, _camera):
        return "test"


def _round(*, mug_at, people=(PERSON,), extra=()):
    objects = [ObjectCandidate("cam:picnic_table:0", "picnic_table", TABLE, 0.9, "cam", SIZE)]
    if mug_at is not None:
        objects.append(ObjectCandidate("cam:mug:0", "mug", FAR_MUG, 0.8, "cam", SIZE))
    objects.extend(extra)
    persons = [PersonCandidate(f"cam:person_lying_down:{i}", box, "person_lying_down", 0.9, "cam", SIZE)
               for i, box in enumerate(people)]
    return CctvDetectionRound(objects=objects, people=persons, camera_ids=["cam"])


def _projector(mug_m):
    return Projector({TABLE: (0.0, 0.0, 0.0), PERSON: (0.6, 0.0, 0.0), FAR_MUG: (mug_m, 0.0, 0.0)})


@mock.patch.dict(os.environ, {"MARC_OCCLUSION_HYPOTHESIS": "1"})
class HypothesisTests(unittest.TestCase):
    def _add(self, command, round_, projector):
        parsed = parse_command(command)
        with decision_scope("stage1", 1) as notes:
            out = occlusion_hypothesis.add_hypotheses(parsed, round_, projector, command)
        return parsed, out, notes

    def test_fires_when_no_detection_satisfies_the_relation(self):
        command = "Find the mug near the picnic table."
        parsed, out, notes = self._add(command, _round(mug_at=10.0), _projector(10.0))
        hypotheses = [o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)]
        self.assertEqual(1, len(hypotheses))
        self.assertEqual(("cam:mug:1", PERSON, 0.0), (hypotheses[0].candidate_id,
                         hypotheses[0].bbox_xyxy, hypotheses[0].confidence))
        note = next(n for n in notes if n["decision"] == "occlusion_hypothesis")
        self.assertEqual("cam:picnic_table:0", note["landmark"])
        self.assertEqual("cam:person_lying_down:0", note["occluder"])

        selected = select_object_candidate(parsed, out, command_text=command, projector=_projector(10.0))
        self.assertEqual("occlusion_hypothesis", selected.method)
        self.assertEqual("cam:mug:1", selected.target.candidate_id)
        self.assertEqual("cam:picnic_table:0", selected.anchor.candidate_id)

    def test_never_chosen_when_a_detection_satisfies_the_relation(self):
        command = "Find the mug near the picnic table."
        parsed, out, _ = self._add(command, _round(mug_at=3.0), _projector(3.0))
        self.assertFalse([o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)])
        selected = select_object_candidate(parsed, out, command_text=command, projector=_projector(3.0))
        self.assertEqual("cam:mug:0", selected.target.candidate_id)

    def test_relation_radius_follows_the_87_rule(self):
        # 3 m satisfies "near" (6 m) but not "beside" (2 m).
        _, out, _ = self._add("Find the mug beside the picnic table.", _round(mug_at=3.0), _projector(3.0))
        self.assertTrue([o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)])

    def test_a_target_stacked_on_its_support_satisfies_on(self):
        # Projected 4 m away (raised object, ground plane), but its box sits on the table.
        on_table = (450.0, 280.0, 470.0, 300.0)
        mug = ObjectCandidate("cam:mug:0", "mug", on_table, 0.8, "cam", SIZE)
        projector = Projector({TABLE: (0.0, 0.0, 0.0), PERSON: (0.6, 0.0, 0.0), on_table: (4.0, 0.0, 0.0)})
        _, out, _ = self._add("Find the mug on the picnic table.", _round(mug_at=None, extra=[mug]), projector)
        self.assertFalse([o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)])

    def test_no_occluder_no_hypothesis(self):
        _, out, notes = self._add("Find the mug near the picnic table.",
                                  _round(mug_at=10.0, people=()), _projector(10.0))
        self.assertFalse([o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)])
        self.assertTrue([n for n in notes if n["decision"] == "occlusion_hypothesis_none"])

    def test_an_unprojectable_target_might_satisfy_so_nothing_fires(self):
        projector = Projector({TABLE: (0.0, 0.0, 0.0), PERSON: (0.6, 0.0, 0.0)})
        _, out, _ = self._add("Find the mug near the picnic table.", _round(mug_at=10.0), projector)
        self.assertFalse([o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)])

    def test_choose_rechecks_the_relation_whoever_added_the_hypothesis(self):
        command = "Find the mug near the picnic table."
        parsed, out, _ = self._add(command, _round(mug_at=10.0), _projector(10.0))
        # A satisfying detection appears next to the hypothesis (e.g. a replayed round).
        near_mug = ObjectCandidate("cam:mug:9", "mug", (500.0, 330.0, 510.0, 340.0), 0.7, "cam", SIZE)
        projector = Projector({**_projector(10.0).points, near_mug.bbox_xyxy: (1.0, 0.0, 0.0)})
        out.objects.append(near_mug)
        self.assertIsNone(occlusion_hypothesis.choose(parsed, out, projector, command))
        selected = select_object_candidate(parsed, out, command_text=command, projector=projector)
        self.assertNotEqual("occlusion_hypothesis", selected.method)

    def test_an_unprojectable_landmark_might_be_satisfied_so_nothing_fires(self):
        projector = Projector({PERSON: (0.6, 0.0, 0.0), FAR_MUG: (10.0, 0.0, 0.0)})
        _, out, _ = self._add("Find the mug near the picnic table.", _round(mug_at=10.0), projector)
        self.assertFalse([o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)])

    def test_one_hypothesis_per_occluder_with_an_unused_id(self):
        second = ObjectCandidate("cam:picnic_table:1", "picnic_table", (420.0, 310.0, 580.0, 400.0),
                                 0.8, "cam", SIZE)
        gap = ObjectCandidate("cam:mug:4", "mug", (100.0, 600.0, 110.0, 610.0), 0.8, "cam", SIZE)
        projector = Projector({TABLE: (0.0, 0.0, 0.0), second.bbox_xyxy: (0.2, 0.0, 0.0),
                               PERSON: (0.6, 0.0, 0.0), FAR_MUG: (10.0, 0.0, 0.0)})
        _, out, _ = self._add("Find the mug near the picnic table.",
                              _round(mug_at=None, extra=[second, gap]), projector)
        hypotheses = [o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)]
        self.assertEqual(["cam:mug:5"], [h.candidate_id for h in hypotheses])

    def test_a_round_submits_the_hypothesis_with_its_landmark(self):
        from marc_agent.perception.stage1_pipeline import run_stage1_round

        round_ = _round(mug_at=10.0)
        with decision_scope("stage1", 1) as notes:
            result = run_stage1_round(None, "Find the mug near the picnic table.",
                                      detector=lambda *a, **k: round_, projector=_projector(10.0))
        self.assertEqual("cam:mug:1", result.target_id)
        self.assertEqual("cam:picnic_table:0", result.anchor_id)
        self.assertEqual("picnic_table", result.grounding.landmark)
        self.assertTrue([n for n in notes if n["decision"] == "selector"
                         and n.get("outcome") == "occlusion_hypothesis"])
        record = result.record()
        self.assertIn("cam:mug:1", [c["candidate_id"] for c in record["candidates"]])

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {}):
            os.environ.pop("MARC_OCCLUSION_HYPOTHESIS", None)
            _, out, _ = self._add("Find the mug near the picnic table.", _round(mug_at=10.0),
                                  _projector(10.0))
        self.assertFalse([o for o in out.objects if occlusion_hypothesis.is_hypothesis(o)])

    def test_hypotheses_are_never_ranked_against_detections(self):
        command = "Find the mug near the picnic table."
        parsed, out, _ = self._add(command, _round(mug_at=10.0), _projector(10.0))
        ranked = rank_object_candidates(parsed, out, projector=_projector(10.0), command_text=command)
        self.assertTrue(ranked)
        self.assertFalse([r for r in ranked if occlusion_hypothesis.is_hypothesis(r.target)])


if __name__ == "__main__":
    unittest.main()
