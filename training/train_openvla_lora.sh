#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
openvla_dir="${OPENVLA_DIR:?set OPENVLA_DIR to an official openvla checkout}"
data_root="${1:?usage: training/train_openvla_lora.sh /path/to/tfds-root}"
base_model="${OPENVLA_MODEL:-/home/marc2026/marc-2026-work/pretrained/openvla-7b}"
work_dir="${OPENVLA_WORK_DIR:-/tmp/marc-openvla}"
best_dir="${OPENVLA_BEST_ADAPTER_DIR:-$work_dir/best-adapter}"
final_dir="${OPENVLA_FINAL_MODEL_DIR:-$work_dir/final-model}"
patched="$work_dir/finetune_marc.py"
max_steps="${OPENVLA_MAX_STEPS:-20000}"
save_steps="${OPENVLA_SAVE_STEPS:-$((max_steps + 1))}"

test -f "$openvla_dir/vla-scripts/finetune.py"
test -f "$base_model/config.json"
test -d "$data_root/marc_pnp"
work_parent="$(dirname "$work_dir")"
test -d "$work_parent" && test -w "$work_parent" || {
  echo "ERROR: work directory parent is read-only: $work_parent" >&2
  exit 1
}
mkdir -p "$work_dir" "$best_dir" "$final_dir"
test -w "$work_dir" || { echo "ERROR: work directory is read-only: $work_dir" >&2; exit 1; }
free_kib="$(df -Pk "$work_dir" | awk 'NR==2 {print $4}')"
test "$free_kib" -ge "${OPENVLA_MIN_FREE_KIB:-20971520}" || {
  echo "ERROR: OpenVLA needs at least 20 GiB free under $work_dir" >&2
  exit 1
}

python3 "$repo_dir/training/patch_openvla_finetune.py" \
  "$openvla_dir/vla-scripts/finetune.py" "$patched"

export PYTHONPATH="$openvla_dir${PYTHONPATH:+:$PYTHONPATH}"
export MARC_OPENVLA_BEST_ADAPTER_DIR="$best_dir"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

cd "$openvla_dir"
torchrun --standalone --nnodes 1 --nproc-per-node "${OPENVLA_GPUS:-1}" "$patched" \
  --vla_path "$base_model" \
  --data_root_dir "$data_root" \
  --dataset_name marc_pnp \
  --run_root_dir "$work_dir/runs" \
  --adapter_tmp_dir "$work_dir/adapters" \
  --batch_size "${OPENVLA_BATCH_SIZE:-2}" \
  --grad_accumulation_steps "${OPENVLA_GRAD_ACCUM:-8}" \
  --learning_rate "${OPENVLA_LR:-5e-4}" \
  --max_steps "$max_steps" \
  --save_steps "$save_steps" \
  --shuffle_buffer_size "${OPENVLA_SHUFFLE_BUFFER:-10000}" \
  --image_aug True \
  --use_val_set True \
  --val_steps "${OPENVLA_VAL_STEPS:-500}" \
  --val_batches "${OPENVLA_VAL_BATCHES:-32}" \
  --early_stopping_patience "${OPENVLA_EARLY_STOP_PATIENCE:-5}" \
  --early_stopping_min_delta "${OPENVLA_EARLY_STOP_MIN_DELTA:-1e-4}" \
  --use_lora True \
  --lora_rank "${OPENVLA_LORA_RANK:-32}" \
  --use_quantization False

cd "$repo_dir"
python3 training/merge_openvla_lora.py \
  --base "$base_model" --adapter "$best_dir" --out "$final_dir"
printf 'Merged model: %s\n' "$final_dir"
