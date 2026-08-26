#!/usr/bin/env bash
# The accuracy suite for the Ornith release candidate.
#
# Adapted from ~/kat-coder-16gb/scripts/eval/eval_suite.sh. One real difference:
# this uses the `hf` (transformers) backend, not `vllm`. vLLM's model registry
# (checked directly in vllm-src/vllm/model_executor/models/registry.py) only
# knows "Qwen3_5MoeForConditionalGeneration" - it has no entry for
# "Qwen3_5MoeForCausalLM", the class our vision-stripped checkpoint declares
# (see docs/vision_tower_decision.md and ROADMAP.md step 6). vLLM's
# `language_model_only` flag (used in KAT-Coder's script) only zeroes multimodal
# INPUT limits at request time - checked directly in
# vllm-src/vllm/config/multimodal.py - it does not skip vision-tower weight
# allocation, so it would not have helped here even if the class were
# registered. Serving this checkpoint via vLLM is a real open gap, tracked in
# docs/serving_notes.md, separate from this accuracy check. The `hf` backend
# has no such registry gap - it loads via plain transformers
# AutoModelForCausalLM, the same path scripts/prune/strip_vision.py's
# reload_verify already proved works.
#
# UPDATE 2026-08-25 (again): `--batch_size auto` is not safe on this box. Its
# probing tries progressively larger batches, up to attempts to allocate 6-8GB
# chunks with the card already full - on native CUDA that's a clean, catchable
# OutOfMemoryError the auto-batch-sizer backs off from, but on this WSL2 GPU
# passthrough setup some of those failures surface instead as a more opaque
# `torch.AcceleratorError: CUDA error: device not ready`, which is NOT caught by
# lm_eval's OOM backoff logic and crashes the whole task. Confirmed directly: two
# of three LIMIT=20 pilot tasks died this way at ~148s (right after model load,
# first forward pass), the third succeeded only after ~13 minutes of repeated
# failed 6-8GB allocation attempts logged to its own eval.log. Fixed batch size
# below avoids the probing entirely - this checkpoint leaves ~400MB of headroom
# on a 16GB card, there is no real room for auto-detection to explore safely.
#
# UPDATE 2026-08-25: the hf backend alone was not enough. llm-compressor 0.13.0 has
# a real, confirmed gap for this architecture - qwen3_5_moe/qwen3_5_moe_text are
# registered for GPTQ calibration but have no entry in ARCH_TO_2D_MAPPINGS, the
# table that reassembles split per-expert quantized weights back into the model's
# real fused gate_up_proj/down_proj parameters on load. Without it, from_pretrained
# silently drops every quantized expert weight as unexpected and the model's MoE
# layers are left at random init - correct shapes, garbage output. Confirmed via
# vllm-project/llm-compressor PR #3080 (open, unmerged): it names
# ornith-ai/Ornith-1.0-35B directly as a real checkpoint hitting this. Fixed locally
# via scripts/patches/patch_qwen3_5_moe_2d_load.py, applied automatically by
# scripts/eval/run_lm_eval_patched.py (used below instead of the raw lm_eval
# binary, since lm_eval's hf backend loads the model internally with no hook point
# to apply this before the raw `lm_eval` CLI would otherwise call it).
#
# Three tasks, all instruct-framed because this is a chat model:
#   humaneval_instruct       164 problems, original tests. Contaminated and
#                            saturated, so reported for comparability only.
#   humaneval_plus_instruct  164 problems, EvalPlus's much stricter tests. Harder to
#                            pass by memorisation, so this is the meaningful number.
#   mbpp_plus_instruct       378 problems, EvalPlus MBPP.
#
# Every score is pass@1 with greedy decoding (do_sample: false, repeats: 1), so these
# are deterministic and re-runnable rather than sampled estimates.
#
# HumanEval executes model-generated code. Both gates are required: the
# --confirm_run_unsafe_code flag AND HF_ALLOW_CODE_EVAL=1.

set -uo pipefail

export HF_ALLOW_CODE_EVAL=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${HOME}/quant-env/bin/python3"
LM=("${PY}" "${REPO}/scripts/eval/run_lm_eval_patched.py")
MODEL="${MODEL:-${HOME}/models/ornith-nvfp4a16-gptq-text-only}"
TASKS_DIR="${REPO}/tasks"
OUT="${OUT:-${HOME}/eval-suite-ornith}"
LIMIT="${1:-}"

mkdir -p "${OUT}"

# lm-eval ships humaneval_instruct (instruct framing, original tests) and
# humaneval_plus (EvalPlus tests, completion framing) but not the combination,
# which is the number this model is reported on. tasks/humaneval_plus_instruct.yaml
# composes the two. Its `include:` and `!function utils....` references resolve
# relative to the yaml's own directory, so it has to be installed beside lm-eval's
# humaneval tasks; --include_path is not enough.
HE_DIR="$("${PY}" -c 'import lm_eval, os; print(os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "humaneval"))')"
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

  "${LM[@]}" run \
    --model hf \
    --model_args "pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True" \
    --tasks "${task}" \
    --include_path "${TASKS_DIR}" \
    --device cuda:0 \
    --batch_size "${EVAL_BATCH_SIZE:-1}" \
    "${extra[@]}" \
    --log_samples \
    --output_path "${dir}" \
    --apply_chat_template \
    --confirm_run_unsafe_code \
    --seed 1234 \
    > "${dir}/eval.log" 2>&1

  local rc=$? elapsed=$(( $(date +%s) - start ))
  local res
  res=$(find "${dir}" -name 'results_*.json' 2>/dev/null | head -1)

  if [ -n "${res}" ]; then
    # lm-eval spells this metric two ways depending on the task's filter:
    # humaneval reports "pass@1,create_test", mbpp reports "pass_at_1,extract_code".
    # Matching only the first silently produced an empty score for MBPP+ with rc=0
    # on the KAT-Coder run - same fix carried over here.
    local score
    score=$(grep -oE '"pass(@|_at_)1,[a-z_]*": *[0-9.]+' "${res}" | head -1 | grep -oE '[0-9.]+$')
    [ -z "${score}" ] && score="!! score not found in ${res##*/}"
    local n
    n=$(find "${dir}" -name 'samples_*.jsonl' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    printf '    pass@1 = %s   (n=%s, %ds, rc=%d)\n' "${score}" "${n}" "${elapsed}" "${rc}"
  else
    echo "    !! NO RESULTS (rc=${rc}, ${elapsed}s)"
    grep -aoE "ValueError: [^\"]{0,140}|Error: [^\"]{0,120}" "${dir}/eval.log" \
      | grep -v "Engine core init" | head -2
  fi
}

echo "=== accuracy suite, $(date -Iseconds) ==="
echo "    model: ${MODEL}"
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
