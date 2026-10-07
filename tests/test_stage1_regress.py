"""The regression gate passes a run inside the baseline bands and names a broken round (#110)."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage1_regress import compare

CAMERAS = ["rig_1_a", "rig_2_b"]
OBJECT_WEIGHTS = {"camera": 0.1, "object_type": 0.15, "landmark": 0.1, "relation": 0.1,
                  "anchor": 0.15, "target": 0.4}
SAR_WEIGHTS = {**OBJECT_WEIGHTS, "landmark": 0.0, "relation": 0.0}


def _object_round(problem, command, target_type, landmark, relation, *, target=1.0, anchor=0.8):
    return {
        "problem_id": problem, "command": command, "weights": OBJECT_WEIGHTS,
        "camera": 1.0, "object_type": 1.0, "landmark": 1.0, "relation": 1.0,
        "anchor": anchor, "target": target,
        "expected": {"camera_id": "rig_1_a", "target_kind": "object", "target_type": target_type,
                     "landmark_prefix": landmark, "relation": relation, "situation": None,
                     "target_coord": [0.0, 0.0, 0.0], "anchor_coord": [1.0, 0.0, 0.0]},
        "submission": {"camera_id": "rig_1_a", "target_type": target_type, "landmark": landmark,
                       "relation": relation, "situation": None,
                       "target_coord": [0.2, 0.0, 0.0], "anchor_coord": [2.0, 0.0, 0.0]},
    }


def _sar_round(problem, command):
    detail = _object_round(problem, command, "person_sitting", "trash_can", None)
    detail["weights"] = SAR_WEIGHTS
    detail["relation"] = 0.0
    detail["expected"].update(target_kind="person", situation="abnormal")
    detail["submission"].update(target_type="person", relation=None, situation="abnormal")
    return detail


ROUNDS = [
    _object_round("lf01_mug_table", "Find the mug on the picnic table.", "mug", "picnic_table", "on"),
    _object_round("lf02_tumbler_bench", "Find the tumbler near the bench.", "tumbler", "bench", "near"),
    _sar_round("sar01_trash_sit", "Find the person sitting on top of the trash can in the parking lot."),
]


def _total(detail):
    weights = detail["weights"]
    return 100 * sum(weights[k] * detail[k] for k in weights) / sum(weights.values())


def _write_run(root: Path, name: str, details: list[dict], decisions=None) -> Path:
    run_dir = root / name
    events_dir = run_dir / "experiments" / "x"
    events_dir.mkdir(parents=True)
    for detail in details:
        detail["total"] = _total(detail)
    (run_dir / "robust_1.json").write_text(json.dumps({
        "scores": {"stage1_average": sum(d["total"] for d in details) / len(details)},
        "round_details": [{k: v for k, v in d.items() if k != "command"} for d in details],
    }))
    lines = []
    for number, detail in enumerate(details, start=1):
        sub = detail["submission"]
        lines += [
            {"event": "stage1.round.received", "round": number, "total_rounds": len(details),
             "simulation_timestamp": 100.0 * number, "time_limit_s": 30.0},
            {"event": "stage1.pipeline.completed", "round": number, "command": detail["command"],
             "camera_ids": CAMERAS, "frames_loaded": CAMERAS,
             "decisions": (decisions or {}).get(detail["problem_id"], [])},
            {"event": "stage1.submission.completed", "round": number,
             "simulation_timestamp": 100.0 * number + 10.0,
             "target_coord": sub["target_coord"], "anchor_coord": sub["anchor_coord"]},
            {"event": "score.received", "round": number,
             "scores": {"problem_id": detail["problem_id"]}},
        ]
    (events_dir / "events.jsonl").write_text("\n".join(json.dumps(line) for line in lines))
    return run_dir


class GateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        flipped = copy.deepcopy(ROUNDS)
        flipped[1]["target"] = 0.0  # lf02 flips between baseline runs
        self.baseline = [
            _write_run(self.root, "run-01", copy.deepcopy(ROUNDS)),
            _write_run(self.root, "run-02", flipped),
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def _gate(self, details, allow=(), decisions=None):
        candidate = _write_run(self.root, "candidate", details, decisions)
        return compare(self.baseline, candidate, set(allow), run_tests=False)

    def test_a_run_inside_the_bands_passes(self):
        details = copy.deepcopy(ROUNDS)
        details[1]["target"] = 0.0  # the low end of lf02's band
        text, failed = self._gate(details)
        self.assertFalse(failed, text)

    def test_a_broken_round_fails_and_is_named(self):
        details = copy.deepcopy(ROUNDS)
        details[0]["target"] = 0.0
        details[0]["submission"]["target_coord"] = [3.0, 0.0, 0.0]
        text, failed = self._gate(details)
        self.assertTrue(failed)
        self.assertIn("Failing rounds: lf01_mug_table", text)
        self.assertIn("target point 2.80 m from every baseline answer", text)

    def test_an_allowed_round_may_change(self):
        details = copy.deepcopy(ROUNDS)
        details[0]["target"] = 0.0
        text, failed = self._gate(details, allow={"lf01_mug_table"})
        self.assertFalse(failed, text)
        self.assertIn("allowed change", text)

    def test_an_empty_relation_on_an_object_round_fails(self):
        details = copy.deepcopy(ROUNDS)
        details[1]["submission"]["relation"] = None
        details[1]["relation"] = 0.0
        text, failed = self._gate(details, allow={"lf02_tumbler_bench"})
        self.assertTrue(failed)
        self.assertIn("| landmark and relation on object rounds | FAIL | lf02_tumbler_bench |", text)

    def test_a_sentinel_round_fails(self):
        details = copy.deepcopy(ROUNDS)
        text, failed = self._gate(
            details, decisions={"sar01_trash_sit": [{"decision": "sentinel", "reason": "x"}]}
        )
        self.assertTrue(failed)
        self.assertIn("| no sentinel or missing coordinates | FAIL | sar01_trash_sit |", text)

    def test_new_decision_paths_are_listed_without_failing(self):
        text, failed = self._gate(
            copy.deepcopy(ROUNDS),
            decisions={"lf01_mug_table": [{"decision": "projection_proxy", "variant": "x"}]},
        )
        self.assertFalse(failed, text)
        self.assertIn("new decisions: projection_proxy:x", text)


if __name__ == "__main__":
    unittest.main()
