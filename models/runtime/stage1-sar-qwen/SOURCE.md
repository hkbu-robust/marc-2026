# Fine-tuned Qwen3-VL-8B runtime snapshot (not in git)

## Selected upstream

- Model: `roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned`
- Immutable revision: `b2360c86972018edd3df1c7ac21e28c129317928`
- Model repository: <https://huggingface.co/roBUst-marc-2026/qwen3-vl-8b-stage1-finetuned>
- Upstream model card: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- Upstream implementation: <https://github.com/QwenLM/Qwen3-VL>
- License declared upstream: Apache-2.0
- Size: about 17.5 GB BF16 safetensors. Do not commit `model/`.

## How the snapshot gets into Docker

The submission-default [`demo/Dockerfile.vllm`](../../../demo/Dockerfile.vllm)
runs [`../fetch_runtime_weights.py`](../fetch_runtime_weights.py) before setting
offline mode. The model is served only on participant-host loopback. The agent
image separately bakes both YOLO checkpoints and GraspGen-X. Scoring runtime
never talks to the Hub.

On a developer host (optional, same pins):

```bash
PYTHONPATH=. python3 models/runtime/fetch_runtime_weights.py --only qwen
```

Skip-if-complete: a ready `bundle_status` is left alone. Use `--force` to replace it.

The production server is vLLM 0.22.0. The retained local-development loader is
`transformers.AutoModelForImageTextToText` (maps `qwen3_vl` to
`Qwen3VLForConditionalGeneration`).

No Hub model ID is accepted by the runtime loader. `local_files_only=True`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` prevent a runtime download fallback.

After a GPU load of the baked image succeeds, set `offline_bundle_load_verified`
in `runtime_contract.json`.
