"""Structured experiment logs remain useful, safe, and fail-open."""

import ast
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from marc_agent.experiment_log import ExperimentLog
from marc_agent.perception.round_dump import stage1_dir, stage2_dir


class ExperimentLogTests(unittest.TestCase):
    def test_stage1_callback_does_not_use_worker_result(self):
        tree = ast.parse(Path("demo/participant_app.py").read_text(encoding="utf-8"))
        callback = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_on_mission"
        )
        self.assertNotIn(
            "round_result",
            {node.id for node in ast.walk(callback) if isinstance(node, ast.Name)},
        )

    def test_run_files_summary_artifacts_and_safe_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "MARC_RUN_MODE": "practice",
                "MARC_QWEN_MODEL_DIR": "/models/qwen",
                "MARC_TOKEN": "must-not-be-recorded",
            }
            run = ExperimentLog(tmp, environ=env)
            run.attach_python_logging()
            logging.getLogger("experiment-test").warning("visible diagnostic")
            run.event(
                "stage1.pipeline.completed", stage="stage1", round=1,
                duration_ms=12.0,
            )
            with run.span("stage1.selection", stage="stage1", round=1):
                pass
            run.event("stage1.round.completed", stage="stage1", round=1)
            run.event(
                "stage2.manipulation.grasp_feedback", stage="stage2",
                status="failed", attempt=1,
            )
            run.event(
                "stage2.navigation.completed", stage="stage2", status="failed",
                terminal=True, duration_ms=30.0, reason="no_path",
            )
            run.finish("failed")

            self.assertTrue(run.events_path.is_file())
            self.assertTrue(run.summary_path.is_file())
            self.assertTrue(run.log_path.is_file())
            self.assertTrue((run.run_dir / "artifacts/stage1").is_dir())
            self.assertTrue((run.run_dir / "artifacts/stage2/navigation").is_dir())
            with patch.dict("os.environ", env, clear=False):
                self.assertEqual(
                    run.run_dir / "artifacts/stage1/round-002", stage1_dir(2)
                )
                self.assertEqual(
                    run.run_dir / "artifacts/stage2/nearfield", stage2_dir("nearfield")
                )
            self.assertEqual(
                str(run.run_dir / "artifacts"), env["MARC_ROUND_DUMP_DIR"]
            )
            self.assertIn("visible diagnostic", run.log_path.read_text(encoding="utf-8"))
            self.assertNotIn("must-not-be-recorded", run.events_path.read_text(encoding="utf-8"))
            summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
            self.assertEqual("stage2.navigation.completed", summary["slowest_span"]["event"])
            self.assertEqual(
                "stage2.navigation.completed", summary["first_terminal_failure"]["event"]
            )
            self.assertEqual(1, summary["stage1_rounds"][0]["round"])
            self.assertEqual("/models/qwen", summary["config"]["MARC_QWEN_MODEL_DIR"])

    def test_completed_lines_survive_truncated_final_jsonl_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = ExperimentLog(tmp, environ={})
            run.event("stage1.round.completed", stage="stage1", round=1)
            with run.events_path.open("a", encoding="utf-8") as stream:
                stream.write('{"event":"partial"')
            lines = run.events_path.read_text(encoding="utf-8").splitlines()
            complete = [json.loads(line) for line in lines[:-1]]
            self.assertEqual("stage1.round.completed", complete[-1]["event"])

    def test_round_and_attempt_correlation_survives_interleaving(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = ExperimentLog(tmp, environ={})
            run.event("work", stage="stage1", round=1)
            run.event("work", stage="stage1", round=2)
            run.event("feedback", stage="stage2", attempt=1)
            run.event("work", stage="stage1", round=1)
            events = [
                json.loads(line)
                for line in run.events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([1, 2, 1], [e["round"] for e in events if "round" in e])
            self.assertEqual(1, next(e["attempt"] for e in events if "attempt" in e))

    def test_unwritable_destination_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_file = Path(tmp) / "not-a-directory"
            root_file.write_text("occupied", encoding="utf-8")
            run = ExperimentLog(root_file, environ={})
            run.event("still.safe")
            pipeline_action_completed = True
            run.finish()
            self.assertTrue(pipeline_action_completed)
            self.assertFalse(run.events_path.exists())


if __name__ == "__main__":
    unittest.main()
