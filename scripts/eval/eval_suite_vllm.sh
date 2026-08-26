#!/usr/bin/env bash
# vLLM-backed accuracy suite for the Ornith release candidate.
#
# Sibling to eval_suite.sh (the `hf`-backend version) - built separately rather than
# adding a backend toggle to that script, because the flag set and even the Python
# env are substantially different: vLLM needs mamba_cache_mode/mamba_block_size/
# block_size/enable_prefix_caching/max_cudagraph_capture_size/gpu_memory_utilization,
# none of which apply to the hf backend, and vllm itself only exists in vllm-env
# (torch 2.11.0, editable install pointed at ~/vllm-src), not quant-env.
#
# WHY THIS NEEDED THREE SEPARATE, CONFIRMED FIXES BEFORE IT COULD WORK AT ALL
#   1. vLLM's model registry had no entry for Qwen3_5MoeForCausalLM (only the
#      multimodal Qwen3_5MoeForConditionalGeneration) - confirmed by reading
#      vllm-src/vllm/model_executor/models/registry.py directly. The class itself
#      was already fully implemented, just unregistered. Fixed by adding the
#      registry entry (source-tree edit, vllm-src is an editable install - takes
#      effect immediately, not persisted anywhere except that local source tree).
#   2. Qwen3_5ForCausalLMBase was missing get_mamba_state_dtype_from_config /
#      get_mamba_state_shape_from_config / get_mamba_state_copy_func - confirmed via
#      __mro__ inspection that neither the class nor QwenNextMixtureOfExperts (the
#      mixin it inherits) defines them. The real, working implementations exist on
#      Qwen3_5ForConditionalGeneration - confirmed all three are fully generic (no
#      self.language_model dependency, only vllm_config + generic MambaState*
#      Calculator helpers) before copying them onto Qwen3_5ForCausalLMBase verbatim.
#      Without this: AttributeError in _get_mamba_bufs() on the first real request,
#      after model load and KV cache init both succeed - easy to mistake for a
#      config problem since it doesn't show up until generation actually starts.
#   3. Checkpoint config carries mrope_section/mrope_interleaved (inherited from the
#      original multimodal model's rope_parameters) but Qwen3_5MoeForCausalLM doesn't
#      implement SupportsMRoPE - confirmed the check (vllm/transformers_utils/
#      config.py::_uses_mrope) is purely `"mrope_section" in config.rope_parameters`,
#      unrelated to whether multimodal content is actually used. Since this
#      checkpoint serves text only, mrope's 4-axis position scheme is mathematically
#      a no-op anyway. Fixed with a config-only checkpoint variant (model.safetensors
#      hardlinked, zero extra disk cost) rather than touching the original.
#
# Plus two real launch-config requirements, confirmed via source/error message
# before guessing:
#   - Qwen3_5ForCausalLMBase.__init__ raises NotImplementedError on the default
#     mamba_cache_mode="all" - needs "align" explicitly.
#   - With mamba_block_size set (528, the value already documented in
#     docs/serving_notes.md from the 2026-08-23 research pass), the standard
#     attention KV-cache block_size must also be a multiple of it (assertion:
#     block_sizes=[528,528,528,16] must be divisible by hash_block_size=528, the 16
#     was vLLM's unrelated default) - set block_size=528 to match.
#   - max_num_seqs is the real lever for KV-cache headroom on this 16GB card, NOT
#     max_model_len or cudagraph capture size - Mamba/GDN state cache scales with
#     concurrent-sequence budget (state-space models keep O(1) memory per sequence
#     regardless of context length), confirmed by reading MambaBase.get_kv_cache_spec.
#
# Verified working end to end (2026-08-25) with these settings: model loads
# (12.49GiB), KV cache allocates (10,649 tokens at max_model_len=2048,
# max_num_seqs=2), and batched 2-request generation returns coherent output for
# both requests.

set -uo pipefail

export HF_ALLOW_CODE_EVAL=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LM="${HOME}/vllm-env/bin/lm_eval"
MODEL="${MODEL:-${HOME}/models/ornith-nvfp4a16-gptq-text-only-norope}"
TASKS_DIR="${REPO}/tasks"
OUT="${OUT:-${HOME}/eval-suite-ornith-vllm}"
LIMIT="${1:-}"

MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True"
MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${EVAL_GPU_UTIL:-0.90}"
MODEL_ARGS="${MODEL_ARGS},max_model_len=${EVAL_MAX_MODEL_LEN:-2048}"
MODEL_ARGS="${MODEL_ARGS},max_num_seqs=${EVAL_MAX_NUM_SEQS:-8}"
MODEL_ARGS="${MODEL_ARGS},max_cudagraph_capture_size=8"
MODEL_ARGS="${MODEL_ARGS},mamba_cache_mode=align"
MODEL_ARGS="${MODEL_ARGS},enable_prefix_caching=True"
MODEL_ARGS="${MODEL_ARGS},mamba_block_size=528"
MODEL_ARGS="${MODEL_ARGS},block_size=528"

mkdir -p "${OUT}"

HE_DIR="$("${HOME}/vllm-env/bin/python" -c 'import lm_eval, os; print(os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "humaneval"))')"
cp "${TASKS_DIR}/humaneval_plus_instruct.yaml" "${HE_DIR}/humaneval_plus_instruct.yaml"
echo "installed humaneval_plus_instruct -> ${HE_DIR}"

run_task() {
  local task="$1"
  local dir="${OUT}/${task}"
  if [ -n "$(find "${dir}" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
    echo "  skip ${task}: results already present"
    return 0
  fi
  mkdir -p "${dir}"

  local extra=()
  [ -n "${LIMIT}" ] && extra=(--limit "${LIMIT}")

  echo
  echo "--- ${task} @ $(date -Iseconds) ---"
  local start
  start=$(date +%s)

  local attempt rc
  for attempt in 1 2; do
    "${LM}" run \
      --model vllm \
      --model_args "${MODEL_ARGS}" \
      --tasks "${task}" \
      --include_path "${TASKS_DIR}" \
      --batch_size auto \
      "${extra[@]}" \
      --log_samples \
      --output_path "${dir}" \
      --apply_chat_template \
      --confirm_run_unsafe_code \
      --seed 1234 \
      > "${dir}/eval.log" 2>&1
    rc=$?
    if [ ${rc} -eq 0 ]; then
      break
    fi
    # Transient WSL GPU-memory-residency race at process-launch boundaries -
    # confirmed real (2026-08-25 pilot run): "No available memory for the cache
    # blocks" on the previous engine's process not having fully released VRAM
    # yet, resolved cleanly on a bare retry with no other change. One retry
    # after a short pause, not more - a real failure should still surface.
    if [ ${attempt} -eq 1 ]; then
      echo "    task failed (rc=${rc}), pausing 15s and retrying once (known transient VRAM-release race)" >&2
      sleep 15
    fi
  done

  local elapsed=$(( $(date +%s) - start ))
  local res
  res=$(find "${dir}" -name 'results_*.json' 2>/dev/null | head -1)

  if [ -n "${res}" ]; then
    local score
    score=$(grep -oE '"pass(@|_at_)1,[a-z_]*": *[0-9.]+' "${res}" | head -1 | grep -oE '[0-9.]+$')
    [ -z "${score}" ] && score="!! score not found in ${res##*/}"
    local n
    n=$(find "${dir}" -name 'samples_*.jsonl' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    printf '    pass@1 = %s   (n=%s, %ds, rc=%d)\n' "${score}" "${n}" "${elapsed}" "${rc}"
  else
    echo "    !! NO RESULTS (rc=${rc}, ${elapsed}s)"
    grep -aoE "ValueError: [^\"]{0,140}|Error: [^\"]{0,120}|AssertionError: [^\"]{0,140}" "${dir}/eval.log" \
      | grep -v "Engine core init" | head -3
  fi
}

echo "=== accuracy suite (vLLM backend), $(date -Iseconds) ==="
echo "    model: ${MODEL}"
echo "    model_args: ${MODEL_ARGS}"
[ -n "${LIMIT}" ] && echo "    LIMIT=${LIMIT} (pilot mode, not a real score)"

run_task humaneval_instruct
run_task humaneval_plus_instruct
run_task mbpp_plus_instruct

echo
echo "=== summary ==="
for t in humaneval_instruct humaneval_plus_instruct mbpp_plus_instruct; do
  r=$(find "${OUT}/${t}" -name 'results_*.json' 2>/dev/null | head -1)
  if [ -n "${r}" ]; then
    s=$(grep -oE '"pass(@|_at_)1,[a-z_]*": *[0-9.]+' "${r}" | head -1 | grep -oE '[0-9.]+$')
    printf '  %-26s %s\n' "${t}" "${s}"
  else
    printf '  %-26s (no result)\n' "${t}"
  fi
done
echo "=== done $(date -Iseconds) ==="
