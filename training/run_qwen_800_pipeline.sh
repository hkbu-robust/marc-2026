#!/usr/bin/env bash
set -Eeuo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="${QWEN_PIPELINE_WORK_DIR:-/tmp/marc-qwen-800}"
final_dir="${QWEN_FINAL_ADAPTER_DIR:-$work_dir/final-adapter}"
target_scenes="${QWEN_DATASET_SCENES:-2800}"
dataset_dir="$work_dir/dataset"
train_dir="$work_dir/saves"
stage="startup"

mkdir -p "$work_dir"/{gt,dataset,saves,logs}
rm -f "$work_dir/FAILED" "$work_dir/FINAL_ADAPTER_PATH"
exec > >(tee -a "$work_dir/logs/pipeline.log") 2>&1
trap 'code=$?; printf "FAILED stage=%s exit=%s at %s\n" "$stage" "$code" "$(date -Is)" | tee "$work_dir/FAILED"; exit "$code"' ERR

cd "$repo_dir"
echo "START $(date -Is)"

stage="stop-conflicting-services"
docker compose -f demo/docker-compose.yml down
bash simulation-platform/marc.sh down

stage="dataset-generation"
scenario_dir="$work_dir/gt/${ENV_MARC_SCENARIO:-marc2026_chungmu}"
mkdir -p "$scenario_dir"
existing_scenes="$(find "$scenario_dir" -mindepth 1 -maxdepth 1 -type d -name 'scene_*' 2>/dev/null | wc -l)"
remaining_scenes=$((target_scenes - existing_scenes))
if (( remaining_scenes <= 0 )); then
  echo "ERROR: already have $existing_scenes scenes; target $target_scenes leaves nothing to generate/export" >&2
  exit 1
fi
echo "Dataset target: $target_scenes total ($existing_scenes existing, $remaining_scenes remaining)"
QWEN_DATASET_SCENES="$remaining_scenes" \
QWEN_GT_DIR="$work_dir/gt" \
QWEN_LORA_DIR="$dataset_dir" \
  bash simulation-platform/marc.sh qwen-dataset-gen

stage="dataset-validation"
for required in train.jsonl val.jsonl dataset_info.json coverage.json; do
  test -s "$dataset_dir/$required"
done
wc -l "$dataset_dir/train.jsonl" "$dataset_dir/val.jsonl"

stage="release-vllm-vram"
vllm_pids="$(fuser 8000/tcp 2>/dev/null || true)"
if [[ -n "$vllm_pids" ]]; then
  echo "Stopping port-8000 vLLM process(es): $vllm_pids"
  kill -TERM $vllm_pids
  for _ in $(seq 1 60); do
    fuser 8000/tcp >/dev/null 2>&1 || break
    sleep 1
  done
  ! fuser 8000/tcp >/dev/null 2>&1
fi

stage="vram-preflight"
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
echo "Free VRAM before training: ${free_mib} MiB"
test "$free_mib" -ge 70000

stage="qwen-lora-training"
PATH="$work_dir/env/bin:$PATH" \
QWEN_LORA_OUTPUT_DIR="$train_dir" \
  training/train_qwen_lora.sh "$dataset_dir"

stage="persist-final-adapter"
mkdir -p "$final_dir"
find "$train_dir" -maxdepth 1 -type f -exec cp -a {} "$final_dir/" \;
printf '%s\n' "$final_dir" > "$work_dir/FINAL_ADAPTER_PATH"
echo "COMPLETE $(date -Is) adapter=$final_dir"
