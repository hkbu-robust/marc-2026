"""Small hardware-profile checks for the vLLM startup script."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "demo" / "vllm-entrypoint.sh"


def profile(memory_mib: str, compute_caps: str = "8.0") -> str:
    env = os.environ.copy()
    env.update(
        {
            "VLLM_GPU_MEMORY_MIBS": memory_mib,
            "VLLM_GPU_COMPUTE_CAPS": compute_caps,
            "VLLM_PRINT_CONFIG": "1",
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout


class VllmEntrypointTests(unittest.TestCase):
    def test_profiles_single_large_and_multiple_small_gpus(self):
        large = profile("49152")
        self.assertIn("tensor_parallel=1", large)
        self.assertIn("utilization=0.75", large)
        self.assertIn("max_model_len=16384", large)

        split = profile("16384,16384")
        self.assertIn("visible=0,1", split)
        self.assertIn("tensor_parallel=2", split)
        self.assertIn("cpu_offload_gb=0", split)

        older = profile("24576", "7.5")
        self.assertIn("dtype=float16", older)


if __name__ == "__main__":
    unittest.main()
