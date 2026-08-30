# OpenVLA runtime snapshot (not in git)

Place a **merged** HuggingFace OpenVLA tree here (or set `MARC_OPENVLA_WEIGHTS`
to another local directory). The loader is offline-only:
`local_files_only=True`, `HF_HUB_OFFLINE=1`. A missing or unloadable snapshot
falls back to analytic IK.

Do not drop the multi-GB `openvla/openvla-7b` base into git. Copy a complete
local snapshot when you have one.

## Required files

A full HF snapshot, not a LoRA adapter:

- `config.json`
- processor / tokenizer files
- modeling `.py` if the checkpoint uses `trust_remote_code`
- weights: `model.safetensors` (or sharded `model.safetensors.index.json`),
  `pytorch_model.bin`, or `model.pt`
- `dataset_statistics.json` with nested `marc_pnp.action.{q01,q99,mask}`

A directory that only has `adapter_config.json` + `adapter_model.safetensors`
is rejected. Merge the LoRA into the base model before drop-in.

## Action convention

7-DoF end-effector delta in **arm_base**:

`dx, dy, dz, droll, dpitch, dyaw, gripper`

- Translation in metres, rotation as a rotvec (applied as
  `target_quat = quat_mul(target_quat, rotvec_to_quat([droll, dpitch, dyaw]))`).
- Gripper is the last dim, `mask=false`, values in `[-1, 1]` after decode.
- Unnormalize key: `marc_pnp` (`MARC_OPENVLA_UNNORM_KEY`). Do not use
  `bridge_orig` on MARC — pretrained OpenVLA-7B is the wrong frame and scale.

`dataset_statistics.json` is produced by `python3 -m manip_dataset.to_rlds`.
Copy that file next to the weights (or into this directory) so inference
unnormalizes with the same numbers a later LoRA train would use.

The pick-only app's **V** key is an explicit zero-shot experiment: it streams
to the remote base checkpoint, decodes `bridge_orig`, and clips each motion to
the local teleop step limits. It does not claim MARC frame compatibility.
The included `chat_template.jinja` supplies the prompt format expected by vLLM.

## Env

| Variable | Meaning |
|---|---|
| `MARC_OPENVLA_WEIGHTS` | Path to the HF snapshot (file or directory) |
| `MARC_OPENVLA_UNNORM_KEY` | Stats key, default `marc_pnp` |
| `MARC_PICK_POLICY` | `graspgenx` (default) / `ik` (OpenVLA is not called at pick) |
