"""Paper-number macros come from the experiment outputs, and gaps show as ?? (#130)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts import paper_numbers as pn
from scripts.stage1_scores import RoundScore, RunScores, describe, write

MAPPING = textwrap.dedent(r"""
    macros:
    - name: HeadMean
      population: "public, 2 runs"
      source: scores/runs.csv
      csv: {column: stage1_average, agg: mean}
      format: {decimals: 2}
    - name: HeadSd
      population: "public, 2 runs"
      source: scores/runs.csv
      csv: {column: stage1_average, agg: sd}
    - name: Kept
      population: "public, rounds"
      source: decisions/rounds.csv
      csv: {agg: count, where: {path: Qwen choice accepted}}
    - name: KeptPct
      population: "public, rounds"
      source: decisions/rounds.csv
      csv: {agg: share, where: {path: Qwen choice accepted}}
      format: {percent: true, decimals: 1}
    - name: Recall
      population: "held-out"
      source: recall.json
      json: overall.all.2
      format: {percent: true, decimals: 1}
    - name: CiLow
      population: "public"
      source: scores/summary.md
      regex: 'CI: ([\d.]+) \[([\d.]+), ([\d.]+)\]'
      group: 2
    - name: Absent
      population: "not run yet"
      source: 999-later/numbers.json
      json: a.b
    - name: Gap
      population: "derived"
      expr: 100 - HeadMean
    - name: GapOfAbsent
      population: "derived from a missing input"
      expr: Absent - 1
""")


class FormatTests(unittest.TestCase):
    def test_numbers(self):
        self.assertEqual(pn.format_value(94.9155, {"decimals": 2}), "94.92")
        self.assertEqual(pn.format_value(11.45, {"decimals": 1}), "11.5")  # half up
        self.assertEqual(pn.format_value(330, {}), "330")
        self.assertEqual(pn.format_value(0.875, {}), "0.88")
        self.assertEqual(pn.format_value("0.4975", {"percent": True, "decimals": 1}), r"49.8\%")
        self.assertEqual(pn.format_value(1.13, {"sign": True}), "+1.13")
        self.assertEqual(pn.format_value(-0.5, {"sign": True, "decimals": 1}),
                         r"\ensuremath{-}0.5")
        self.assertEqual(pn.format_value(-0.0001, {"sign": True}), "0.00")  # no -0.00
        self.assertEqual(pn.format_value(3, {"decimals": "1", "scale": 2}), "6.0")

    def test_text(self):
        digest = "sha256:f54a9dedb4a4824daa0224ced0dd1940c709f5f0614616f6049cbdf7f0c20972"
        self.assertEqual(pn.format_value(digest, {"text": True, "strip": "sha256:", "chars": 12}),
                         "f54a9dedb4a4")
        self.assertEqual(pn.format_value("final_system 5%", {"text": True}),
                         r"final\_system 5\%")

    def test_not_a_number_is_missing(self):
        for value in ("abc", float("nan")):
            with self.assertRaises(pn.Missing):
                pn.format_value(value, {"decimals": 2})


class ExtractTests(unittest.TestCase):
    ROWS = [
        {"run": "run-01", "path": "Qwen choice accepted", "x": "1.0", "sentinel": ""},
        {"run": "run-02", "path": "geometry override", "x": "2.0", "sentinel": "fallback_grounding"},
        {"run": "run-03", "path": "Qwen choice accepted", "x": "", "sentinel": ""},
        {"run": "run-04", "path": "geometry only (no Qwen)", "x": "4.0", "sentinel": ""},
    ]

    def test_csv_aggregates_and_filters(self):
        rows = self.ROWS
        self.assertEqual(pn.extract_csv(rows, {"agg": "count",
                                               "where": {"path": "Qwen choice accepted"}}), 2)
        self.assertEqual(pn.extract_csv(rows, {"agg": "share", "where": {"sentinel": "*"}}), 0.25)
        self.assertEqual(pn.extract_csv(rows, {"agg": "count",
                                               "where": {"path": {"not": "geometry override"}}}), 3)
        self.assertEqual(pn.extract_csv(rows, {"column": "x", "agg": "mean",
                                               "where": {"run": ["run-01", "run-02"]}}), 1.5)
        self.assertEqual(pn.extract_csv(rows, {"column": "x", "agg": "sum"}), 7.0)  # empty skipped
        self.assertEqual(pn.extract_csv(rows, {"column": "x", "where": {"run": "run-04"}}), 4.0)

    def test_quantiles_match_stage1_scores(self):
        values = [92.929, 93.785, 94.136, 95.955, 95.962]
        rows = [{"v": str(value)} for value in values]
        stats = describe(values)
        for agg, key in (("q1", "q1"), ("q3", "q3"), ("median", "median"), ("sd", "sd")):
            self.assertAlmostEqual(pn.extract_csv(rows, {"column": "v", "agg": agg}), stats[key])

    def test_csv_gaps_are_missing(self):
        for spec in (
            {"column": "nope", "agg": "mean"},
            {"column": "x", "where": {"path": "Qwen choice accepted"}},  # two rows, wanted one
            {"column": "x", "agg": "mean", "where": {"run": "run-03"}},  # only an empty cell
            {"agg": "count", "where": {"nope": 1}},
            {"column": "path", "agg": "mean"},
        ):
            with self.subTest(spec=spec), self.assertRaises(pn.Missing):
                pn.extract_csv(self.ROWS, spec)

    def test_json_paths(self):
        data = {"overall": {"all": [464, 530, 0.875, [0.099, 0.155]]}, "flag": True, "n": None}
        self.assertEqual(pn.extract_json(data, "overall.all.2"), 0.875)
        self.assertEqual(pn.extract_json(data, "overall.all.3.1"), 0.155)
        self.assertEqual(pn.extract_json(data, ["overall", "all", 0]), 464)
        self.assertEqual(pn.extract_json(data, "flag"), 1)
        for path in ("overall.none", "overall.all.9", "n", "overall", "overall.all.2.x"):
            with self.subTest(path=path), self.assertRaises(pn.Missing):
                pn.extract_json(data, path)

    def test_regex(self):
        text = "| reader | no submission |\n| qwen | 0/60 | 55/56 |\nCI: 94.916 [91.603, 97.229].\n"
        self.assertEqual(pn.extract_regex(text, r"\| qwen \| (\d+/\d+) \| (\d+/\d+)", 2), "55/56")
        self.assertAlmostEqual(
            pn.extract_regex(text, r"\| qwen \| \d+/\d+ \| (\d+)/(\d+)", [1, 2]), 55 / 56)
        self.assertEqual(pn.extract_regex(text, r"\[([\d.]+),", None), 91.603)
        with self.assertRaises(pn.Missing):
            pn.extract_regex(text, r"\| lenient rules \| (\d+/\d+)", 1)

    def test_expr(self):
        self.assertAlmostEqual(pn.evaluate("100 - A * 2", {"A": "2.5"}), 95.0)
        self.assertAlmostEqual(pn.evaluate("-(A - B) / 2", {"A": 1, "B": 3}), 1.0)
        with self.assertRaises(pn.Missing):
            pn.evaluate("A - B", {"A": 1})
        with self.assertRaises(ValueError):
            pn.evaluate("__import__('os')", {})


class MappingTests(unittest.TestCase):
    def test_each_expands_products(self):
        macros = pn.parse_macros([{
            "each": [[{"S": "Held", "s": "heldout"}, {"S": "Pub", "s": "public"}],
                     [{"A": "B", "g": "2"}, {"A": "C", "g": "3"}]],
            "name": "Sel${S}${A}", "population": "$s arm $A", "source": "x/$s.md",
            "regex": "(a)(b)(c)", "group": "$g", "format": {"decimals": "$g"},
        }])
        self.assertEqual([m.name for m in macros], ["SelHeldB", "SelHeldC", "SelPubB", "SelPubC"])
        self.assertEqual(macros[3].source, "x/public.md")
        self.assertEqual(macros[3].group, 3)
        self.assertEqual(macros[0].population, "heldout arm B")

    def test_bad_names_and_duplicates(self):
        for entries in (
            [{"name": "Stage1Mean", "source": "a", "json": "x"}],  # digits are not LaTeX names
            [{"name": "A", "source": "a", "json": "x"}, {"name": "A", "source": "a", "json": "y"}],
            [{"name": "A", "json": "x"}],  # no source
            [{"name": "A", "source": "a", "json": "x", "regex": "y"}],
        ):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                pn.parse_macros(entries)

    def test_shipped_mapping_reads_only_final_experiments(self):
        macros = pn.load_mapping(pn.DEFAULT_MAPPING)
        self.assertGreater(len(macros), 100)
        roots = {"127-provenance", "128-repeated-runs", "87-eval-sets", "89-selector-arms",
                 "90-e2e-selector-arms", "129-ablations", "136-loss-ledger"}
        for macro in macros:
            if macro.source:
                self.assertNotIn("..", macro.source, macro.name)
                self.assertIn(macro.source.split("/")[0], roots, macro.name)
            self.assertTrue(macro.population, macro.name)
            self.assertNotIn("$", macro.name + str(macro.spec) + (macro.source or ""), macro.name)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.root = self.dir / "final-experiments"
        (self.root / "scores").mkdir(parents=True)
        (self.root / "decisions").mkdir()
        (self.root / "scores/runs.csv").write_text("run,stage1_average\nrun-01,94.9\nrun-02,95.0\n")
        (self.root / "scores/summary.md").write_text("CI: 94.916 [91.603, 97.229].\n")
        (self.root / "decisions/rounds.csv").write_text(
            "run,path\nr,Qwen choice accepted\nr,geometry override\nr,Qwen choice accepted\n")
        (self.root / "recall.json").write_text(json.dumps({"overall": {"all": [4, 5, 0.8]}}))
        self.mapping = self.dir / "map.yaml"
        self.mapping.write_text(MAPPING)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, *extra) -> tuple[int, str, str]:
        out = self.dir / "paper" / "numbers.tex"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = pn.main(["--root", str(self.root), "--mapping", str(self.mapping),
                            "--out", str(out), *extra])
        return code, out.read_text(), stderr.getvalue()

    def test_values_and_placeholders(self):
        code, tex, err = self._run()
        self.assertEqual(code, 0)
        self.assertIn(r"\newcommand{\HeadMean}{94.95}", tex)
        self.assertIn(r"\newcommand{\HeadSd}{0.07}", tex)
        self.assertIn(r"\newcommand{\Kept}{2}", tex)
        self.assertIn(r"\newcommand{\KeptPct}{66.7\%}", tex)
        self.assertIn(r"\newcommand{\Recall}{80.0\%}", tex)
        self.assertIn(r"\newcommand{\CiLow}{91.60}", tex)  # floats default to 2 decimals
        self.assertIn(r"\newcommand{\Gap}{5.05}", tex)
        self.assertIn(r"\newcommand{\Absent}{\textbf{??}}", tex)
        self.assertIn(r"\newcommand{\GapOfAbsent}{\textbf{??}}", tex)
        self.assertIn("% public, 2 runs | scores/runs.csv : mean stage1_average", tex)
        self.assertIn(r"?? \Absent: no file 999-later/numbers.json", err)
        self.assertIn(r"?? \GapOfAbsent: needs Absent", err)
        self.assertIn("7 of 9 macros resolved", tex)

    def test_empty_root_never_crashes(self):
        for path in sorted(self.root.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        code, tex, err = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(tex.count(r"}{\textbf{??}}"), 9)
        self.assertEqual(self._run("--strict")[0], 1)

    def test_shipped_mapping_reads_stage1_scores_output(self):
        def run(name, average, totals):
            return RunScores(name, average, [
                RoundScore(problem, total, {c: 1.0 for c in ("camera", "object_type", "landmark",
                                                           "relation", "anchor", "target")},
                           cluster=(problem,), answer=("rig_1_a", (0.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
                           sim_s=4.0 + index, time_limit_s=30.0)
                for index, (problem, total) in enumerate(totals.items())
            ])
        runs = [run("run-01", 95.0, {"lf01": 98.0, "lf02": 92.0}),
                run("run-02", 93.0, {"lf01": 96.0, "lf02": 90.0})]
        with contextlib.redirect_stdout(io.StringIO()):
            write(runs, self.root / "128-repeated-runs" / "analysis", "test")
        self.mapping.write_text(pn.DEFAULT_MAPPING.read_text())
        _, tex, _ = self._run()
        for line in (r"\newcommand{\StageOneMean}{94.00}", r"\newcommand{\StageOneSd}{1.41}",
                     r"\newcommand{\StageOneRuns}{2}", r"\newcommand{\StageOneCiClusters}{2}",
                     r"\newcommand{\StageOneSimMax}{5.00}", r"\newcommand{\StageOneTimeLimit}{30}",
                     r"\newcommand{\StageOneStableRounds}{2}",
                     r"\newcommand{\StageOneAnchorCredit}{1.000}"):
            self.assertIn(line, tex)
        self.assertNotIn(r"\newcommand{\StageOneCiLow}{\textbf{??}}", tex)


if __name__ == "__main__":
    unittest.main()
