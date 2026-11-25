#!/usr/bin/env bash
set -euo pipefail

build_mode="${MARC_BUILD_MODE:-baked}"
case "$build_mode" in
  baked) model_dir="${VLLM_MODEL_DIR:-/models/qwen3-vl}" ;;
  development) model_dir="${VLLM_MODEL_DIR:-/models-dev/qwen}" ;;
  *) echo "MARC_BUILD_MODE must be baked or development" >&2; exit 2 ;;
esac

gpu_ids=()
gpu_mib=()
gpu_caps=()
if [[ -n "${VLLM_GPU_MEMORY_MIBS:-}" ]]; then
  IFS=',' read -r -a gpu_mib <<< "$VLLM_GPU_MEMORY_MIBS"
  IFS=',' read -r -a gpu_caps <<< "${VLLM_GPU_COMPUTE_CAPS:-8.0}"
  for index in "${!gpu_mib[@]}"; do gpu_ids+=("$index"); done
else
  gpu_rows="$(nvidia-smi --query-gpu=index,memory.total,compute_cap --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "$gpu_rows" ]]; then
    while IFS=',' read -r index memory capability; do
      gpu_ids+=("${index//[[:space:]]/}")
      gpu_mib+=("${memory//[[:space:]]/}")
      gpu_caps+=("${capability//[[:space:]]/}")
    done <<< "$gpu_rows"
  else
    gpu_rows="$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
    while IFS=',' read -r index memory; do
      [[ -n "$index" ]] || continue
      gpu_ids+=("${index//[[:space:]]/}")
      gpu_mib+=("${memory//[[:space:]]/}")
      gpu_caps+=(unknown)
    done <<< "$gpu_rows"
  fi
fi

gpu_count="${#gpu_ids[@]}"
if (( gpu_count == 0 )); then
  echo "vLLM requires at least one NVIDIA GPU visible to Docker" >&2
  exit 1
fi

single_min_mib="${VLLM_SINGLE_GPU_MIN_MIB:-24576}"
selected_ids=()
selected_mib=()
selected_caps=()
for ((index=gpu_count-1; index>=0; index--)); do
  if (( gpu_mib[index] >= single_min_mib )); then
    selected_ids+=("${gpu_ids[index]}")
    selected_mib+=("${gpu_mib[index]}")
    selected_caps+=("${gpu_caps[index]:-${gpu_caps[0]:-8.0}}")
    break
  fi
done

if (( ${#selected_ids[@]} == 0 )); then
  tensor_parallel=1
  for candidate in 8 4 2; do
    if (( gpu_count >= candidate )); then tensor_parallel="$candidate"; break; fi
  done
  start=$((gpu_count - tensor_parallel))
  for ((index=start; index<gpu_count; index++)); do
    selected_ids+=("${gpu_ids[index]}")
    selected_mib+=("${gpu_mib[index]}")
    selected_caps+=("${gpu_caps[index]:-${gpu_caps[0]:-8.0}}")
  done
else
  tensor_parallel=1
fi

min_mib="${selected_mib[0]}"
total_mib=0
for memory in "${selected_mib[@]}"; do
  (( memory < min_mib )) && min_mib="$memory"
  total_mib=$((total_mib + memory))
done

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  utilization="$VLLM_GPU_MEMORY_UTILIZATION"
elif (( tensor_parallel > 1 )); then
  utilization=0.88
elif (( min_mib >= 40960 )); then
  utilization=0.75
elif (( min_mib >= 24576 )); then
  utilization=0.82
else
  utilization=0.90
fi

if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  max_model_len="$VLLM_MAX_MODEL_LEN"
elif (( min_mib >= 40960 )); then
  max_model_len=16384
else
  max_model_len=8192
fi

cpu_offload_gb="${VLLM_CPU_OFFLOAD_GB:-0}"
if [[ -z "${VLLM_CPU_OFFLOAD_GB:-}" ]] && (( total_mib < 24576 )); then
  cpu_offload_gb=$(((24576 - total_mib + 1023) / 1024 + 2))
fi

dtype="${VLLM_DTYPE:-auto}"
if [[ -z "${VLLM_DTYPE:-}" ]]; then
  for capability in "${selected_caps[@]}"; do
    major="${capability%%.*}"
    if [[ "$major" =~ ^[0-9]+$ ]] && (( major < 8 )); then
      dtype=float16
      break
    fi
  done
fi

visible_devices="$(IFS=,; echo "${selected_ids[*]}")"
export CUDA_VISIBLE_DEVICES="$visible_devices"

echo "[vllm-config] visible=$visible_devices memory_mib=$(IFS=,; echo "${selected_mib[*]}") tensor_parallel=$tensor_parallel dtype=$dtype utilization=$utilization max_model_len=$max_model_len cpu_offload_gb=$cpu_offload_gb"
if [[ "${VLLM_PRINT_CONFIG:-0}" = 1 ]]; then exit 0; fi
if [[ ! -f "$model_dir/config.json" ]]; then
  echo "Qwen snapshot is missing at $model_dir" >&2
  exit 1
fi

args=(
  serve "$model_dir"
  --host "${VLLM_HOST:-127.0.0.1}"
  --port "${VLLM_PORT:-8004}"
  --served-model-name "${VLLM_MODEL_NAME:-qwen3-vl-8b}"
  --api-key "${VLLM_API_KEY:-local-vllm-key}"
  --dtype "$dtype"
  --tensor-parallel-size "$tensor_parallel"
  --gpu-memory-utilization "$utilization"
  --max-model-len "$max_model_len"
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-4}"
  --limit-mm-per-prompt '{"image":24,"video":0}'
  --mm-processor-cache-gb 0
)
if (( cpu_offload_gb > 0 )); then args+=(--cpu-offload-gb "$cpu_offload_gb"); fi
if [[ "${VLLM_ENFORCE_EAGER:-1}" = 1 ]]; then args+=(--enforce-eager); fi

exec vllm "${args[@]}" "$@"
