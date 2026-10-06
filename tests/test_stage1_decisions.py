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
        cache = StaticSceneCache()
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


if __name__ == "__main__":
    unittest.main()
