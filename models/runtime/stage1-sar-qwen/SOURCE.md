# Qwen3-VL upstream and pending artifact record

## Selected upstream

- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Immutable revision: `60595ebc30ec8e3b1d3b9e65d4943ca011c0006a`
- Upstream model card: <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>
- Upstream implementation: <https://github.com/QwenLM/Qwen3-VL>
- License declared upstream: Apache-2.0

The 8B Instruct model is selected as the local visual verifier described in
`docs/research/stage1-sar-recognizer.md`. It will consume bounded candidate-board imagery
and emit only a candidate ID plus one of the four situation tokens. It is not a second
detector and must not emit submission coordinates.

## Artifact state

The large snapshot has not been materialized in this repository yet. The runtime contract,
offline loader, Docker dependencies, and target directory are configured so the exact
snapshot can be added later without changing callers.

When the snapshot is added:

1. copy every file from the immutable revision into `model/`;
2. ensure Git LFS materializes every `*.safetensors` file;
3. copy the upstream license into this bundle if it is not included in the snapshot;
4. add and verify SHA-256 checksums for every runtime file;
5. run `python -m marc_agent.perception.qwen_runtime`;
6. load and warm the model with network access disabled; and
7. change `offline_bundle_load_verified` in `runtime_contract.json` only after that test.

No Hub model ID is accepted by the runtime loader. `local_files_only=True`,
`HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` prevent a runtime download fallback.
