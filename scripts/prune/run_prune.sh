#!/usr/bin/env bash
# The real Ornith-1.5-35B-A3B REAP prune, 256 -> 128 experts (50%), single
# seed (42) - unlike KAT-Coder's prune_both_seeds.sh, this project has not
# run the stability (multi-seed calibration disagreement) experiment on
# Ornith, so there is no established need for a second seed here. If that
# stability question matters later, add a second --seed run reusing this
# same --artifacts-dir the way prune_both_seeds.sh does.
#
# RESIDENCY: layerwise, not cpu_full, DELIBERATELY DIFFERENT FROM
# KAT-CODER'S VALIDATED PATH. See docs/ram_headroom_check.md for the full
# reasoning: cpu_full pins the full bf16 model in host RAM, and Ornith's
# ~71.9GB of bf16 weights (35.95B params) leaves only ~2-6GB of headroom in
# this box's 80GB WSL memory cap - real OOM risk, not a comfortable margin.
# `reap prune layerwise --help` documents `layerwise` as block-wise observe
# + disk offload (not a full CPU pin) - the safer choice here. TRADEOFF,
# NOT FREE: KAT-Coder's scripts (see stability_run.sh, seqlen_test.sh)
# chose cpu_full specifically because it's "the validated deterministic
# path" - layerwise's determinism/reproducibility has NOT been verified by
# this project. If bit-for-bit reproducibility across reruns turns out to
# matter later (e.g. a stability re-run), re-verify that property under
# layerwise before relying on it, don't assume it transfers from cpu_full's
# validation.
#
# MTP HEAD: not specially handled here. reap-cuda's prune loop never
# touches `mtp.*` weights (confirmed by reading model_adapters.py - see
# docs/mtp_pruning_decision.md), so this command produces a checkpoint
# whose MTP head is untouched-but-config-mismatched. DO NOT treat this
# script's output as done or reload it directly - scripts/prune/strip_mtp.py
# must run on the output directory first (see that script's docstring for
# why: the shared text_config.num_experts field would otherwise produce a
# checkpoint that fails to reload with a router shape mismatch).

set -uo pipefail

BIN="${HOME}/reap-cuda-env/bin/reap"
MODEL="${HOME}/models/Ornith-1.5-35B-A3B"
ROOT="${HOME}/reap-stability"
RUNDIR="${ROOT}/ornith_n64_s42"
RATIO=0.50
SEED=42

mkdir -p "${RUNDIR}"
log="${RUNDIR}/prune.log"

if [ ! -d "${MODEL}" ] || [ -z "$(find "${MODEL}" -maxdepth 1 -name '*.safetensors' 2>/dev/null | head -1)" ]; then
  echo "!! model not found at ${MODEL} - checkpoint download must complete first" >&2
  exit 1
fi

existing=$(find "${RUNDIR}" -maxdepth 1 -type d -name 'reap-*' 2>/dev/null | head -1)
if [ -n "${existing}" ] && [ -n "$(find "${existing}" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
  echo "skip: pruned checkpoint already exists at ${existing}"
  echo "(run scripts/prune/strip_mtp.py on it next, if not already done)"
  exit 0
fi

echo "=== pruning Ornith-1.5-35B-A3B seed ${SEED} @ $(date -Iseconds) ==="
echo "    residency=layerwise (see script header - deliberate deviation from cpu_full)"

"${BIN}" prune layerwise \
  --model "${MODEL}" \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --compression-ratio "${RATIO}" \
  --prune-method reap \
  --observe-backend bmm \
  --residency layerwise \
  --batch-size 1 \
  --batches-per-category 64 \
  --model-max-length 2048 \
  --seed "${SEED}" \
  --artifacts-dir "${RUNDIR}" \
  --no-eval \
  > "${log}" 2>&1

rc=$?

if grep -q "Aggregate cache hit" "${log}"; then
  echo "    observation cache REUSED (no recalibration)"
fi

out=$(find "${RUNDIR}" -maxdepth 1 -type d -name 'reap-*' 2>/dev/null | head -1)
if [ -n "${out}" ] && [ -n "$(find "${out}" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
  echo "    rc=${rc} checkpoint OK: ${out}"
  du -sh "${out}"
  python3 - "${out}" <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
tc = cfg.get("text_config", cfg)
print(f"    text_config.num_experts = {tc.get('num_experts')} (expect 128)")
print(f"    mtp_num_hidden_layers   = {tc.get('mtp_num_hidden_layers')} (expect 1 - NOT stripped yet, run strip_mtp.py next)")
PY
  echo ""
  echo "NEXT: python3 scripts/prune/strip_mtp.py --checkpoint ${out} --output-dir ${out}-mtp-stripped"
else
  echo "    rc=${rc} !! NO CHECKPOINT - see ${log}"
  exit 1
fi

echo "=== prune complete $(date -Iseconds) ==="
