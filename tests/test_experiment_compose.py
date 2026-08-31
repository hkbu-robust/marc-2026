"""The official image always writes one portable host-mounted experiment copy."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class ExperimentComposeTests(unittest.TestCase):
    def test_submission_mount_has_portable_default(self):
        compose = Path("docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn('MARC_EXPERIMENTS_DIR: "/agent/experiments"', compose)
        self.assertIn(
            '"${MARC_EXPERIMENTS_HOST_DIR:-./experiments}:/agent/experiments"',
            compose,
        )

    @unittest.skipUnless(shutil.which("docker"), "docker compose is unavailable")
    def test_host_override_changes_only_bind_source(self):
        with tempfile.NamedTemporaryFile() as empty_env:
            command = ["docker", "compose", "--env-file", empty_env.name, "config"]
            portable = subprocess.run(
                command, check=True, text=True, capture_output=True,
            ).stdout
            overridden = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
                env={**os.environ, "MARC_EXPERIMENTS_HOST_DIR": "/tmp/marc-experiments"},
            ).stdout
        self.assertIn(str(Path.cwd() / "experiments"), portable)
        self.assertIn("/tmp/marc-experiments", overridden)
        self.assertIn("target: /agent/experiments", portable)
        self.assertIn("target: /agent/experiments", overridden)


if __name__ == "__main__":
    unittest.main()
