"""Score statistics over repeated runs read what the platform scored (#88)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage1_scores import cluster_bootstrap, component_means, read_run, round_table

OBJECT_WEIGHTS = {"camera": 0.1, "object_type": 0.15, "landmark": 0.1, "relation": 0.1,
                  "anchor": 0.15, "target": 0.4}
SAR_WEIGHTS = {**OBJECT_WEIGHTS, "landmark": 0.0, "relation": 0.0}


def _detail(problem, total, weights, target=1.0, coord=(1.0, 2.0, 3.0)):
    return {
        "problem_id": problem, "total": total, "weights": weights,
        "camera": 1.0, "object_type": 1.0, "landmark": 1.0,
        "relation": 1.0 if weights["relation"] else 0.0,  # SAR rounds score 0 at weight 0
        "anchor": 0.8, "target": target,
        "expected": {"camera_id": "rig_1_a", "target_type": problem, "relation": None,
                     "landmark_name": "bench_01", "situation": None,
                     "target_coord": list(coord)},
    }


def _write_run(root: Path, name: str, details: list[dict], answers: list[tuple]) -> Path:
    run_dir = root / name
    events_dir = run_dir / "experiments" / "20261006T000000Z_x"
    events_dir.mkdir(parents=True)
    average = sum(detail["total"] for detail in details) / len(details)
    (run_dir / "robust_1.json").write_text(json.dumps({
        "scores": {"stage1_average": average}, "round_details": details,
    }))
    lines = []
    for number, (detail, answer) in enumerate(zip(details, answers), start=1):
        lines += [
            {"event": "stage1.round.received", "round": number,
             "simulation_timestamp": 10.0 * number, "time_limit_s": 30.0},
            {"event": "stage1.submission.completed", "round": number,
             "simulation_timestamp": 10.0 * number + 5.0,
             "camera_id": answer[0], "target_id": answer[1], "anchor_id": answer[2]},
            {"event": "score.received", "round": number,
             "scores": {"problem_id": detail["problem_id"]}},
        ]
    (events_dir / "events.jsonl").write_text("\n".join(json.dumps(line) for line in lines))
    return run_dir


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        sar = _detail("sar01", 90.0, SAR_WEIGHTS)
        first = [_detail("lf01", 100.0, OBJECT_WEIGHTS), sar,
                 _detail("hf01", 100.0, OBJECT_WEIGHTS)]
        second = [_detail("lf01", 60.0, OBJECT_WEIGHTS, target=0.0), sar,
                  _detail("hf01", 100.0, OBJECT_WEIGHTS)]
        stable = ("rig_1_a", "rig_1_a:mug:0", "rig_1_a:bench:0")
        self.runs = [
            read_run(_write_run(root, "run-01", first, [stable] * 3)),
            read_run(_write_run(root, "run-02", second,
                                [("rig_1_a", "rig_1_a:mug:1", "rig_1_a:bench:0"), stable, stable])),
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def test_zero_weight_components_are_left_out(self):
        means = component_means(self.runs[0])
        self.assertEqual(means["relation"], 1.0)  # the SAR round's 0.0 is not averaged in
        self.assertIsNone(self.runs[0].rounds[1].components["relation"])
        self.assertIsNotNone(self.runs[0].rounds[1].components["anchor"])

    def test_answer_changes_and_lost_points(self):
        rows = {row["problem_id"]: row for row in round_table(self.runs)}
        self.assertEqual(rows["lf01"]["answer_changes"], 1)
        self.assertEqual(rows["lf01"]["lost_components"], "target")
        self.assertEqual(rows["lf01"]["runs_below_95"], 1)
        self.assertEqual(rows["sar01"]["answer_changes"], 0)
        self.assertEqual(rows["sar01"]["lost_components"], "")

    def test_round_time_is_simulation_time(self):
        self.assertEqual(self.runs[0].rounds[0].sim_s, 5.0)
        self.assertEqual(self.runs[0].rounds[0].time_limit_s, 30.0)

    def test_repeated_configurations_form_one_cluster(self):
        # lf01 and hf01 share every expected field except target_type, so three clusters.
        self.assertEqual(cluster_bootstrap(self.runs, draws=200)["clusters"], 3)


if __name__ == "__main__":
    unittest.main()
