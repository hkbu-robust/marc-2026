#!/usr/bin/env python3
"""Merge the validation-best OpenVLA LoRA adapter into a vLLM/HF-ready directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForVision2Seq, AutoProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = AutoModelForVision2Seq.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(
        args.adapter, trust_remote_code=True, local_files_only=True,
    ).save_pretrained(out)
    stats = Path(args.adapter) / "dataset_statistics.json"
    if stats.is_file():
        shutil.copy2(stats, out / stats.name)
    print(out)


if __name__ == "__main__":
    main()
