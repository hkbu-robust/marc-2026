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

After the check passes, set `MARC_SAR_DETECTOR=qwen` and
`MARC_QWEN_BACKEND=local`. The Stage 1 candidate pipeline builds one bounded multi-image
board, runs one local generation, validates the returned candidate ID and situation token,
and uses deterministic evidence if loading or generation fails.

For practice with Qwen on another LAN workstation, set `MARC_QWEN_BACKEND=remote`,
`MARC_QWEN_BASE_URL`, `MARC_QWEN_API_KEY`, and `MARC_QWEN_MODEL`. Remote mode uses the same
board and strict output grammar through vLLM's OpenAI-compatible chat endpoint. It does not
require this local snapshot, but it is not the self-contained offline submission mode.
