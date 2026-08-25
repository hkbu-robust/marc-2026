"""Fixed paths to bundled Stage 1 model artifacts."""

from __future__ import annotations

from pathlib import Path

# marc_agent/perception -> marc_agent -> repo root (/agent in Docker)
_REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE1_CCTV_BUNDLE_DIR = _REPO_ROOT / "models" / "runtime" / "stage1-cctv-detector"
STAGE1_CCTV_WEIGHTS = STAGE1_CCTV_BUNDLE_DIR / "model.pt"
STAGE1_CCTV_CONTRACT = STAGE1_CCTV_BUNDLE_DIR / "runtime_contract.json"

STAGE1_QWEN_BUNDLE_DIR = _REPO_ROOT / "models" / "runtime" / "stage1-sar-qwen"
STAGE1_QWEN_MODEL_DIR = STAGE1_QWEN_BUNDLE_DIR / "model"
STAGE1_QWEN_CONTRACT = STAGE1_QWEN_BUNDLE_DIR / "runtime_contract.json"
