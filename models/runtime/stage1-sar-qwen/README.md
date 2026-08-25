# Stage 1 Qwen SAR verifier bundle

This directory defines the offline runtime contract before the large model snapshot is
materialized. Qwen is intentionally not enabled by default while `model/` is incomplete.

Place the complete `Qwen/Qwen3-VL-8B-Instruct` snapshot at revision
`60595ebc30ec8e3b1d3b9e65d4943ca011c0006a` under `model/`. The directory must include
the processor/tokenizer configuration, `config.json`, and every safetensors shard declared
by `model.safetensors.index.json`.

Check the bundle without loading it into GPU memory:

```bash
python -m marc_agent.perception.qwen_runtime
```

After the check succeeds, rebuild the participant image. Runtime loading is local-only; it
never falls back to the Hub.

Do not set `MARC_SAR_DETECTOR=qwen` yet. Candidate-board construction and the Qwen
selection call are the next implementation step; until they are wired, keep the detector
mode on `stub` or `yolo`.
