#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${QWEN_PIPELINE_WORK_DIR:-/tmp/marc-qwen-800}"
model_dir="${MARC_QWEN_MODEL_DIR:-${MARC_PRETRAINED_ROOT:-./pretrained}/Qwen3-VL-8B-Instruct}"
merged_dir="${QWEN_MERGED_MODEL_DIR:-$work_dir/merged-model}"
config="$repo_dir/training/configs/qwen3vl-8b-lora-merge.yaml"
poll_seconds="${QWEN_MERGE_POLL_SECONDS:-30}"
stage="wait-for-training"

mkdir -p "$work_dir/logs" "$(dirname "$merged_dir")"
exec > >(tee -a "$work_dir/logs/merge.log") 2>&1
trap 'code=$?; if (( code != 0 )); then printf "FAILED stage=%s exit=%s at %s\n" "$stage" "$code" "$(date -Is)" | tee "$work_dir/MERGE_FAILED"; fi' EXIT

echo "QUEUED $(date -Is) waiting_for=$work_dir/FINAL_ADAPTER_PATH"
while [[ ! -s "$work_dir/FINAL_ADAPTER_PATH" ]]; do
  if [[ -s "$work_dir/FAILED" ]]; then
    echo "ERROR: Qwen dataset/training pipeline failed:" >&2
    cat "$work_dir/FAILED" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done

stage="validate-inputs"
adapter_dir="$(head -n 1 "$work_dir/FINAL_ADAPTER_PATH")"
[[ -d "$model_dir" ]] || { echo "ERROR: base model not found: $model_dir" >&2; exit 1; }
[[ -s "$adapter_dir/adapter_config.json" ]] || { echo "ERROR: adapter config not found: $adapter_dir/adapter_config.json" >&2; exit 1; }
find "$adapter_dir" -maxdepth 1 -type f -name 'adapter_model*.safetensors' -print -quit | grep -q . || {
  echo "ERROR: adapter weights not found in $adapter_dir" >&2
  exit 1
}
export PATH="$work_dir/env/bin:$PATH"
command -v llamafactory-cli >/dev/null || {
  echo "ERROR: llamafactory-cli not found in $work_dir/env/bin" >&2
  exit 1
}
[[ ! -e "$merged_dir" && ! -e "$merged_dir.partial" ]] || {
  echo "ERROR: merge destination already exists: $merged_dir or $merged_dir.partial" >&2
  exit 1
}

required_bytes=$(( $(du -sb "$model_dir" | awk '{print $1}') + 2147483648 ))
available_bytes="$(df -PB1 "$(dirname "$merged_dir")" | awk 'NR==2 {print $4}')"
echo "Storage preflight: required=${required_bytes}B available=${available_bytes}B destination=$merged_dir"
(( available_bytes >= required_bytes )) || {
  echo "ERROR: insufficient writable storage for merged model" >&2
  exit 1
}

stage="merge-lora"
echo "MERGE START $(date -Is) adapter=$adapter_dir model=$model_dir device=cpu"
llamafactory-cli export "$config" \
  model_name_or_path="$model_dir" \
  adapter_name_or_path="$adapter_dir" \
  export_dir="$merged_dir.partial"

stage="validate-export"
[[ -s "$merged_dir.partial/config.json" ]] || { echo "ERROR: merged config.json missing" >&2; exit 1; }
find "$merged_dir.partial" -maxdepth 1 -type f -name '*.safetensors' -print -quit | grep -q . || {
  echo "ERROR: merged model weights missing" >&2
  exit 1
}
mv "$merged_dir.partial" "$merged_dir"
printf '%s\n' "$merged_dir" > "$work_dir/MERGED_MODEL_PATH"
echo "COMPLETE $(date -Is) merged_model=$merged_dir"
