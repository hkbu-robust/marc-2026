"""The offline replay reproduces logged rounds through the runtime projector (#110)."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marc_agent.perception.stage1_pipeline import project_anchor
from scripts.stage1_replay import (
    main,
    projector_for,
    replay_cases,
    replay_run,
    replay_targets,
)
from scripts.stage1_runs import LoggedRound, read_run, score, target_credit

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "stage1_replay" / "run-01"
FROZEN_FLAGS = {"MARC_GROUND_CALIBRATION": "0", "MARC_LANDMARK_FOOTPRINT": "0"}


class ReplayTests(unittest.TestCase):
    """Two #88 cache-off rounds: lf06 (logged through the proxy chain #116 deleted) and
    sar03 (lying person)."""

    def setUp(self):
        self.run = read_run(FIXTURE)
        # With every new mechanism's flag off, the replay is the frozen system.
        with mock.patch.dict(os.environ, FROZEN_FLAGS):
            self.replayed = {
                item.round.problem_id: item for item in replay_run(self.run, projector_for())
            }

    def test_logged_rounds_reproduce_to_the_millimetre(self):
        self.assertEqual(set(self.replayed), {"lf06_umbrella_fallenbike", "sar03_bench_lie"})
        for item in self.replayed.values():
            logged = item.round.submission
            self.assertLess(item.target_moved_m, 1e-3)
            self.assertLess(item.anchor_moved_m, 1e-3)
            self.assertEqual(item.target_id, logged["target_id"])
            self.assertEqual(item.anchor_id, logged["anchor_id"])
            for name in ("camera_id", "landmark", "relation", "situation"):
                self.assertEqual(item.submission[name], logged[name])
            self.assertAlmostEqual(item.replayed.total, item.round.detail["total"], places=9)

    def test_replay_records_the_runtime_decisions(self):
        lf06 = self.replayed["lf06_umbrella_fallenbike"]
        decisions = {note["decision"] for note in lf06.decisions}
        self.assertIn("ground_model", decisions)
        # The swap is gone (#116). lf06's logged target was the fabricated landmark-
        # context candidate, whose box is the trash can's, and the replay keeps logged
        # candidates fixed, so the point itself only changes in a live run.
        self.assertNotIn("projection_proxy", decisions)

    def test_kernel_matches_every_logged_component(self):
        for round_ in self.run.rounds:
            scored = score(round_, round_.detail["submission"])
            for name, value in scored.components.items():
                self.assertAlmostEqual(value, round_.detail[name], places=9, msg=name)
            self.assertAlmostEqual(scored.total, round_.detail["total"], places=9)

    def test_a_moved_target_is_rescored_by_distance(self):
        round_ = self.run.rounds[0]
        x, y, z = round_.expected["target_coord"]
        # Moved in -y, away from lf06's other objects, so attribution still holds.
        near = {**round_.detail["submission"], "target_coord": [x, y - 1.0, z]}
        far = {**round_.detail["submission"], "target_coord": [x, y - 3.0, z]}
        self.assertEqual(score(round_, near).components["target"], 1.0)
        self.assertTrue(score(round_, near).target_uncertain)
        # 3 m is inside the miss radius and outside the acquire radius: linear partial credit.
        self.assertAlmostEqual(score(round_, far).components["target"], 2.0 / 3.0, places=6)
        self.assertFalse(math.isclose(score(round_, far).total, round_.detail["total"]))

    def test_a_point_nearer_a_distractor_than_the_target_scores_zero(self):
        gt = (0.0, 0.0, 0.0)
        distractor = (1.0, 0.0, 0.0)
        self.assertEqual(target_credit((0.2, 0.0, 0.0), gt, [distractor]), 1.0)
        self.assertEqual(target_credit((0.6, 0.0, 0.0), gt, [distractor]), 0.0)
        near_distractor = (0.9, 0.0, 0.0)
        self.assertEqual(target_credit((0.4, 0.0, 0.0), gt, [near_distractor]), 1.0)
        self.assertEqual(
            target_credit((0.4, 0.0, 0.0), gt, [near_distractor], margin=0.15), 0.0
        )

    def test_score_attributes_against_the_public_scenario_distractors(self):
        # #128 run-08 lf03: the platform's expected block has no distractors, the point is
        # 1.284 m from the GT tumbler but 0.909 m from another object, and it logged 0.
        gt = [-36.07784, 128.70259, 16.84848]
        point = [-34.79452935730926, 128.71649787863436, 16.795886928727008]
        round_ = LoggedRound(
            number=1,
            problem_id="lf03_tumbler_hydrant",
            detail={
                "expected": {
                    "camera_id": "rig_7_a", "target_kind": "object", "target_type": "tumbler",
                    "landmark_prefix": "hydrant", "relation": "near",
                    "anchor_coord": [-33.8053, 130.52874, 16.85787], "target_coord": gt,
                },
                "submission": {"target_coord": gt},  # logged elsewhere, so the point is rescored
                "weights": {name: 1.0 for name in ("camera", "object_type", "landmark",
                                                   "relation", "anchor", "target")},
                "target": 1.0,
            },
            pipeline=None, submission=None, sim_s=None, time_limit_s=None,
        )
        submission = {"anchor_coord": [-33.8053, 130.52874, 16.85787], "target_coord": point}
        self.assertLess(math.dist(point, gt), 2.0)
        self.assertEqual(score(round_, submission).components["target"], 0.0)
        unknown = dataclasses.replace(round_, problem_id="not_a_public_problem")
        with self.assertRaises(KeyError):
            score(unknown, submission)

    def test_a_round_without_a_parse_keeps_its_logged_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-01"
            shutil.copytree(FIXTURE, run_dir)
            events = run_dir / "experiments" / "fixture" / "events.jsonl"
            lines = [json.loads(line) for line in events.read_text().splitlines()]
            for line in lines:
                if line["event"] == "stage1.pipeline.completed" and line["round"] == 6:
                    line["parsed"] = None
            events.write_text("\n".join(json.dumps(line) for line in lines))
            replayed = replay_run(read_run(run_dir), projector_for())
        carried = next(item for item in replayed if item.round.number == 6)
        self.assertFalse(carried.recomposed)
        self.assertEqual(carried.replayed.total, carried.round.detail["total"])
        self.assertTrue(next(item for item in replayed if item.round.number == 22).recomposed)

    def test_cli_writes_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--runs", str(FIXTURE), "--out", tmp])
            for name in ("rounds.csv", "runs.csv", "summary.md"):
                self.assertTrue((Path(tmp) / name).exists(), name)

    def test_case_anchor_error_uses_the_runtime_anchor_projection(self):
        box = [500.0, 300.0, 560.0, 380.0]
        gt = project_anchor(projector_for(), "rig_1_a", tuple(box))
        case = {
            "case_id": "c1", "kind": "object",
            "candidates": [
                {"candidate_id": "rig_1_a:bench:0", "camera_id": "rig_1_a",
                 "class_name": "bench", "confidence": 0.9, "bbox_xyxy": box},
            ],
            "gold": {"anchor_ids": ["rig_1_a:bench:0"], "anchor_xyz": [gt[0] + 1.0, gt[1], gt[2]]},
        }
        missing = {**case, "case_id": "c2", "gold": {"anchor_ids": [], "anchor_xyz": list(gt)}}
        with tempfile.TemporaryDirectory() as tmp:
            for item in (case, missing):
                folder = Path(tmp) / "cases" / item["case_id"]
                folder.mkdir(parents=True)
                (folder / "case.json").write_text(json.dumps(item))
            rows, skipped = replay_cases(Path(tmp), projector_for())
        self.assertEqual(skipped, 1)
        self.assertAlmostEqual(rows[0].error_m, 1.0, places=6)
        self.assertAlmostEqual(rows[0].credit, 0.8, places=6)

    def test_cases_read_the_final_candidate_version_and_score_targets(self):
        anchor_box, target_box = [500.0, 300.0, 560.0, 380.0], [700.0, 400.0, 720.0, 430.0]
        projector = projector_for()
        anchor_gt = project_anchor(projector, "rig_1_a", tuple(anchor_box))
        target_gt = projector.project_contact("rig_1_a", tuple(target_box))
        final = [
            {"candidate_id": "rig_1_a:bench:0", "camera_id": "rig_1_a",
             "class_name": "bench", "confidence": 0.9, "bbox_xyxy": anchor_box},
            {"candidate_id": "rig_1_a:mug:0", "camera_id": "rig_1_a",
             "class_name": "mug", "confidence": 0.8, "bbox_xyxy": target_box},
        ]
        case = {
            "case_id": "c1", "kind": "object", "candidates": [],
            "gold": {"anchor_ids": [], "target_ids": [], "relation": "near",
                     "anchor_xyz": list(anchor_gt),
                     "target_xyz": [target_gt[0] + 0.5, target_gt[1], target_gt[2]]},
            "candidates_final": final,
            "gold_final": {"anchor_ids": ["rig_1_a:bench:0"], "target_ids": ["rig_1_a:mug:0"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "cases" / "c1"
            folder.mkdir(parents=True)
            (folder / "case.json").write_text(json.dumps(case))
            rows, skipped = replay_cases(Path(tmp), projector)
            targets = replay_targets(Path(tmp), projector)
        self.assertEqual((len(rows), skipped), (1, 0))
        self.assertAlmostEqual(rows[0].error_m, 0.0, places=6)
        self.assertAlmostEqual(targets[0].error_m, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
