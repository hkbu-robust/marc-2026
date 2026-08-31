# Stage 1 Qwen SAR verifier bundle

The large snapshot is **not** in git. The default `docker compose up --build`
downloads `roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned` at revision
`b2360c86972018edd3df1c7ac21e28c129317928` into the dedicated vLLM image.

Host fetch (optional):

```bash
PYTHONPATH=. python3 models/runtime/fetch_runtime_weights.py --only qwen
python -m marc_agent.perception.qwen_runtime
```

The directory must include the processor/tokenizer configuration, `config.json`,
and every safetensors shard declared by `model.safetensors.index.json`.

Root compose starts the baked vLLM service on participant-host loopback and points
the agent's existing remote adapter at it. Runtime is offline-only; it never falls
back to the Hub. Load or generation failures still use deterministic evidence.

For practice with Qwen on another LAN workstation, set `MARC_QWEN_BACKEND=remote`,
`MARC_QWEN_BASE_URL`, `MARC_QWEN_API_KEY`, and `MARC_QWEN_MODEL`. Remote mode uses the same
board and strict output grammar through vLLM's OpenAI-compatible chat endpoint. It does not
require this local snapshot, but it is not the self-contained offline submission mode.
