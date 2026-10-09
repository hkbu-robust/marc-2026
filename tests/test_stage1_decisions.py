"""The decision summary counts what the pipeline recorded (#85)."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import marc_agent.experiment_log as experiment_log
from marc_agent.experiment_log import decision_scope, start_experiment
from marc_agent.perception.cctv_project import CctvProjector, static_lookup
from marc_agent.perception.stage1_pipeline import StaticSceneCache, run_stage1_round
from scripts.stage1_decisions import path_table, read_run, run_totals, write

from test_sar_selection import _frame
from test_stage1_round import _geometry, _object_round


def _round_with_frames():
    round_ = _object_round()
    round_.frames = {"rig_a": _frame("rig_a"), "rig_b": _frame("rig_b")}
    return round_


def _raise(_board):
    raise RuntimeError("vLLM down")


class SummaryTests(unittest.TestCase):
    def tearDown(self):
        experiment_log._current = None

    def _play(self, tmp):
        """Four rounds through the participant's logging path."""
        run = start_experiment(tmp, environ={})
        cache = StaticSceneCache(enabled=True)
        plays = [
            ("Find the mug on the picnic table.", _geometry(), _raise, 40.0),
            ("Find the mug on the picnic table.", _geometry(), None, 95.0),
            ("Find the umbrella beside the vending machine.", _geometry(), None, 35.0),
            ("Find the mug near the picnic table.", CctvProjector(static_lookup({})), None, 15.0),
        ]
        for number, (text, projector, verifier, total) in enumerate(plays, start=1):
            run.event("stage1.round.received", stage="stage1", round=number)
            with decision_scope("stage1", number) as decisions:
                result = run_stage1_round(
                    object(), text,
                    detector=lambda client, *, stub_path=None: _round_with_frames(),
                    projector=projector, verifier=verifier,
                )
                result = cache.apply(result, number)
            run.event(
                "stage1.pipeline.completed", stage="stage1", round=number,
                decisions=decisions, **result.record(),
            )
            run.event("stage1.submission.completed", stage="stage1", round=number)
            run.event(
                "score.received", round=number, is_final=False,
                scores={"problem_id": f"p{number}", "total": total},
            )
        return run.run_dir

    def test_counts_each_path_once_per_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = read_run(self._play(tmp))
        self.assertEqual(4, len(run.rounds))
        self.assertEqual(
            ["Qwen fallback", "cache reuse", "sentinel coordinate", "sentinel coordinate"],
            [item.path for item in run.rounds],
        )
        self.assertEqual("verifier_failed:RuntimeError", run.rounds[0].selector_reason)
        self.assertEqual(1, run.rounds[1].cache_source_round)
        self.assertTrue(run.rounds[1].cache_same_pair)
        self.assertEqual("no_object_pairs", run.rounds[2].sentinel)
        self.assertEqual("object_projection_failed", run.rounds[3].sentinel)
        self.assertTrue(all(item.coverage_complete for item in run.rounds))
        self.assertIsNotNone(run.rounds[0].round_time_s)

        totals = run_totals(run)
        self.assertEqual(1, totals["qwen_fallback"])
        self.assertEqual(1, totals["cache_hit"])
        self.assertEqual(2, totals["sentinel"])
        self.assertAlmostEqual(46.25, totals["mean_score"])

        table = {row["path"]: row for row in path_table([run])}
        self.assertEqual(2, table["sentinel coordinate"]["rounds"])
        self.assertAlmostEqual(25.0, table["sentinel coordinate"]["mean_score"])

    def test_writes_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = read_run(self._play(tmp))
            out = Path(tmp) / "summary"
            write([run], out)
            with (out / "rounds.csv").open() as stream:
                self.assertEqual(4, len(list(csv.DictReader(stream))))
            self.assertIn("| sentinel coordinate | 2 |", (out / "summary.md").read_text())


# Decision notes as logged in rounds 8 (lf08) and 20 (sar01) of the #126 smoke run of
# final-system-v1, plus an OOM skip and the always-submit sentinel.
_FINAL_ROUNDS = {
    8: [
        {"decision": "command_parse", "source": "qwen", "latency_s": 0.9, "fallback": None},
        {"decision": "ground_model", "camera_id": "rig_7_a", "ground": "applied"},
        {"decision": "tiled_refine", "added": 1, "candidates": ["rig_7_a:umbrella:0"],
         "cameras": ["rig_7_a"], "ms": 2600, "target": "umbrella"},
        {"decision": "implied_anchor", "chosen": "rig_7_a:police_car:0"},
        {"decision": "selector", "outcome": "geometry_only", "path": "object",
         "reason": "parking_geometry", "selected": "rig_7_a:umbrella:0"},
        {"decision": "field_source", "landmark": "inferred", "relation": "default",
         "target_type": "stated"},
    ],
    20: [
        {"decision": "command_parse", "source": "rules", "latency_s": 6.0},
        {"decision": "tiled_refine", "added": 0, "candidates": [], "oom_skipped": ["rig_7_a", "rig_1_a"]},
        {"decision": "situation", "source": "r03", "value": "abnormal"},
        {"decision": "field_source", "landmark": "stated", "relation": "stated",
         "situation": "r03", "target_type": "stated"},
        {"decision": "fallback_anchor", "source": "landmark", "candidate": "rig_7_a:trash_can:0"},
        {"decision": "sentinel", "reason": "fallback_grounding", "camera_id": "rig_7_a"},
    ],
}


class FinalSystemColumnTests(unittest.TestCase):
    def _run(self, tmp):
        import json

        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        lines = [
            json.dumps({"event": "stage1.pipeline.completed", "round": number, "decisions": notes})
            for number, notes in _FINAL_ROUNDS.items()
        ]
        (run_dir / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return read_run(run_dir)

    def test_reads_the_final_system_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lf08, sar01 = self._run(tmp).rounds
        self.assertEqual(("qwen", 0.9), (lf08.parse_source, lf08.parse_latency_s))
        self.assertEqual(("stated", "inferred", "default", None),
                         (lf08.field_target_type, lf08.field_landmark, lf08.field_relation,
                          lf08.field_situation))
        self.assertEqual("applied", lf08.ground_model)
        self.assertEqual((1, "rig_7_a:umbrella:0", 0),
                         (lf08.tiled_added, lf08.tiled_candidates, lf08.tiled_oom))
        self.assertEqual("rig_7_a:police_car:0", lf08.implied_anchor)
        self.assertEqual("geometry only (no Qwen)", lf08.path)
        self.assertEqual(("rules", "r03", 2, True, "fallback_grounding"),
                         (sar01.parse_source, sar01.field_situation, sar01.tiled_oom,
                          sar01.fallback_anchor, sar01.sentinel))
        self.assertEqual("sentinel coordinate", sar01.path)

    def test_columns_reach_rounds_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary"
            write([self._run(tmp)], out)
            with (out / "rounds.csv").open() as stream:
                rows = list(csv.DictReader(stream))
        for column in ("parse_source", "parse_latency_s", "field_target_type", "field_landmark",
                       "field_relation", "field_situation", "ground_model", "tiled_added",
                       "tiled_candidates", "tiled_oom", "implied_anchor", "fallback_anchor"):
            self.assertIn(column, rows[0])
        self.assertEqual("fallback_grounding", rows[1]["sentinel"])


if __name__ == "__main__":
    unittest.main()
