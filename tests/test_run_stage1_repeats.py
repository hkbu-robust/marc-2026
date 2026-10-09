"""The harness rejects runs that are missing evidence (#85)."""

import unittest

from pathlib import Path

from scripts.run_stage1_repeats import _problems, own_score_files


def _record(**overrides):
    record = {
        "finished": True,
        "score_files": ["robust_x.json"],
        "rounds_logged": 33,
        "round_dumps": 33,
    }
    record.update(overrides)
    return record


class ProblemsTests(unittest.TestCase):
    def test_complete_run_has_no_problems(self):
        self.assertEqual([], _problems(_record()))

    def test_missing_round_dumps_fail_the_run(self):
        self.assertEqual(["0 round dumps for 33 rounds"], _problems(_record(round_dumps=0)))

    def test_unfinished_run_without_scores_or_rounds(self):
        record = {"finished": False, "score_files": []}
        self.assertEqual(
            ["no FINISHED state", "no platform score file", "no logged rounds"],
            _problems(record),
        )



class OwnScoreFileTests(unittest.TestCase):
    # The scoreboard line from #128 run-02, whose parallel series B wrote 234407 meanwhile.
    LOG = ("platform-1  | [INFO] runtime.scoreboard: Scoreboard.record: team=robust, total=87.5431, "
           "file=/metacom2026/results/marc2026_chungmu/robust_20261009_235646.json, "
           "summary=/metacom2026/results/marc2026_chungmu/_summary.json\n")

    def test_keeps_only_the_file_this_platform_recorded(self):
        new = [Path("results/robust_20261009_234407.json"), Path("results/robust_20261009_235646.json")]
        self.assertEqual([new[1]], own_score_files(new, self.LOG))

    def test_keeps_everything_when_the_log_names_no_file(self):
        new = [Path("results/robust_a.json")]
        self.assertEqual(new, own_score_files(new, "no scoreboard line"))


if __name__ == "__main__":
    unittest.main()
