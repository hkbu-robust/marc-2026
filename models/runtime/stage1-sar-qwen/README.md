# Stage 1 Qwen SAR verifier bundle

The large snapshot is **not** in git. `docker compose up --build` downloads
`Qwen/Qwen3-VL-8B-Instruct` at revision
`60595ebc30ec8e3b1d3b9e65d4943ca011c0006a` into `model/`.

Host fetch (optional):

```bash
PYTHONPATH=. python3 models/runtime/fetch_runtime_weights.py
python -m marc_agent.perception.qwen_runtime
```

The directory must include the processor/tokenizer configuration, `config.json`,
and every safetensors shard declared by `model.safetensors.index.json`.

Root compose defaults to `MARC_SAR_DETECTOR=qwen` and `MARC_QWEN_BACKEND=local`.
Runtime loading is local-only; it never falls back to the Hub. Load or generate
failures still use deterministic evidence.

For practice with Qwen on another LAN workstation, set `MARC_QWEN_BACKEND=remote`,
`MARC_QWEN_BASE_URL`, `MARC_QWEN_API_KEY`, and `MARC_QWEN_MODEL`. Remote mode uses the same
board and strict output grammar through vLLM's OpenAI-compatible chat endpoint. It does not
require this local snapshot, but it is not the self-contained offline submission mode.
