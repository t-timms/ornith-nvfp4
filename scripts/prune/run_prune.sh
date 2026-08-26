#!/usr/bin/env bash
# The real Ornith-1.5-35B-A3B REAP prune, 256 -> 128 experts (50%), single
# seed (42) - unlike KAT-Coder's prune_both_seeds.sh, this project has not
# run the stability (multi-seed calibration disagreement) experiment on
# Ornith, so there is no established need for a second seed here. If that
# stability question matters later, add a second --seed run reusing this
# same --artifacts-dir the way prune_both_seeds.sh does.
#
# RESIDENCY: cpu_full, REVERSED FROM THE ORIGINAL layerwise CHOICE - see
# docs/layerwise_prune_run_2026-08-24.md for the full incident writeup.
# layerwise was tried first (block-wise observe + disk offload) specifically
# to avoid cpu_full's RAM risk (see docs/ram_headroom_check.md's original
# ~71.9GB-weights/~2-6GB-headroom estimate). It uncovered three real
# reap-cuda bugs on this hybrid GDN+attention model, all bypassing
# accelerate's forward-hook-based on-demand weight materialization for
# disk-offloaded blocks: (1) attention_mask captured once from block 0 and
# wrongly replayed into every block regardless of layer_type - FIXED, see
# layerwise_observer.py's _capture_extra_type_masks; (2) MoE expert weights
# read directly after accelerate had already re-offloaded a disk-backed
# block back to a meta placeholder - FIXED, see weight_cache.py's
# _materialize_offloaded/_release_offloaded, which bracket the read with
# accelerate's own AlignDevicesHook.pre_forward/post_forward; both (1) and
# (2) verified end-to-end across a full 40-block pass. (3) the same class
# of bug one step further into the pipeline, in the pruning/mutation step
# (model_adapters.py's slice_experts mutating .data directly) - found, NOT
# fixed. Rather than keep finding and fixing bugs in a code path this
# project doesn't need, switched to cpu_full: it never creates meta-tensor
# placeholders in the first place (no disk offload), sidestepping this
# entire bug class by construction. Empirically validated before the
# switch: real loaded weight footprint was ~65.4GiB (not ~71.9GB as
# estimated), leaving a real ~11-12GB margin against this box's 77GB free
# RAM - confirmed comfortable at both reduced scale and the real 64-batch
# scale (stayed ~75GB available throughout). cpu_full is also KAT-Coder's
# original "validated deterministic path" (see stability_run.sh,
# seqlen_test.sh) - reproducibility is a solved question here, unlike
# layerwise's never-verified status. Residency mode is a memory-staging
# mechanism only - it has no effect on the REAP algorithm, calibration
# data, or resulting model quality.
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

skip_search() {
  # reap-cuda nests the published checkpoint several levels down
  # (model_<hash>/dataset_<hash>/pruned_models/reap-<name>), not directly
  # under --artifacts-dir - maxdepth 1 here would never match a real output
  # (confirmed against an actual published checkpoint from tonight's
  # reduced-scale validation run). Search deep enough to find it regardless
  # of minor internal reap-cuda layout changes.
  find "${RUNDIR}" -maxdepth 6 -type d -name 'reap-*' 2>/dev/null | head -1
}

existing=$(skip_search)
if [ -n "${existing}" ] && [ -n "$(find "${existing}" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
  echo "skip: pruned checkpoint already exists at ${existing}"
  echo "(run scripts/prune/strip_mtp.py on it next, if not already done)"
  exit 0
fi

echo "=== pruning Ornith-1.5-35B-A3B seed ${SEED} @ $(date -Iseconds) ==="
echo "    residency=cpu_full (see script header - switched from layerwise after finding an unfixed reap-cuda bug)"

"${BIN}" prune layerwise \
  --model "${MODEL}" \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --compression-ratio "${RATIO}" \
  --prune-method reap \
  --observe-backend bmm \
  --residency cpu_full \
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

out=$(skip_search)
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
