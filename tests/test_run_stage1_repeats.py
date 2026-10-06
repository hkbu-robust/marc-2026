"""The harness rejects runs that are missing evidence (#85)."""

import unittest

from scripts.run_stage1_repeats import _problems


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


if __name__ == "__main__":
    unittest.main()
