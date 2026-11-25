#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_dir="${1:-$repo_dir/trainer_output/qwen3vl_lora}"
model_dir="${MARC_QWEN_MODEL_DIR:-${MARC_PRETRAINED_ROOT:-./pretrained}/Qwen3-VL-8B-Instruct}"
output_dir="${QWEN_LORA_OUTPUT_DIR:-$repo_dir/saves/qwen3-vl-8b/lora/marc}"
config="$repo_dir/training/configs/qwen3vl-8b-lora-sft.yaml"

for required in train.jsonl val.jsonl dataset_info.json; do
  [[ -s "$dataset_dir/$required" ]] || {
    echo "ERROR: missing $dataset_dir/$required; run qwen-dataset-gen first" >&2
    exit 1
  }
done
[[ -d "$model_dir" ]] || { echo "ERROR: Qwen model not found: $model_dir" >&2; exit 1; }
command -v llamafactory-cli >/dev/null || {
  echo "ERROR: activate an environment containing LLaMA-Factory (llamafactory-cli)" >&2
  exit 1
}

exec llamafactory-cli train "$config" \
  model_name_or_path="$model_dir" \
  dataset_dir="$dataset_dir" \
  output_dir="$output_dir"
