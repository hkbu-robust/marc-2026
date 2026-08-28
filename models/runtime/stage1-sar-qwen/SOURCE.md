# Qwen3-VL-8B-Instruct runtime snapshot (not in git)

## Selected upstream

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Immutable revision: `60595ebc30ec8e3b1d3b9e65d4943ca011c0006a`
- Upstream model card: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- Upstream implementation: <https://github.com/QwenLM/Qwen3-VL>
- License declared upstream: Apache-2.0
- Size: about 17.5 GB BF16 safetensors. Do not commit `model/`.

## How the snapshot gets into the agent

`docker compose up --build` with ``MARC_QWEN_BACKEND=local`` (the submission
default) runs [`../fetch_runtime_weights.py`](../fetch_runtime_weights.py)
inside [`demo/Dockerfile`](../../../demo/Dockerfile) **before** `HF_HUB_OFFLINE=1`.
That downloads this revision into `/agent/models/runtime/stage1-sar-qwen/model`.
Remote/vLLM practice builds (`demo/docker-compose.qwen-remote.yml`) skip this
Hub pull. Scoring runtime never talks to the Hub.

On a developer host (optional, same pins):

```bash
PYTHONPATH=. python3 models/runtime/fetch_runtime_weights.py
```

Skip-if-complete: a ready `bundle_status` is left alone. Use `--force` to replace it.

The 4.57.1 loader is `transformers.AutoModelForImageTextToText` (maps `qwen3_vl`
to `Qwen3VLForConditionalGeneration`). `AutoModelForMultimodalLM` is transformers
v5 and is not present in the pinned image.

No Hub model ID is accepted by the runtime loader. `local_files_only=True`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` prevent a runtime download fallback.

After a GPU load of the baked image succeeds, set `offline_bundle_load_verified`
in `runtime_contract.json`.
