#!/usr/bin/env bash
# Ornith SWE-bench overnight ladder, v2 (2026-08-26, after v1's 02:14 stop).
#
# CHANGES FROM v1, each evidence-driven:
# 1. MAXLEN 32768 -> 49152 (adaptive fallback): the same instance ran twice
#    under an identical config - draw 1 Submitted (68 msgs), draw 2
#    ContextWindowExceeded at 106 msgs while making real progress (its own
#    tests were passing). The window, not capability, is the binding
#    constraint - the prior project's exact published finding. Measured KV on
#    this box supports 49,152 (60,787-token ceiling, 1.24x at workers=1).
# 2. Gates redefined: v1 gated on non-empty patches, which conflated sampling
#    variance with infrastructure health and stopped the chain on a healthy
#    system. Gates now check MECHANICS only (rollout completed, preds.json
#    present with the expected instance count). Patch rates print as data.
# 3. Per-phase exit-status summaries printed for post-hoc scoring review.
set -uo pipefail

PORT=8000
MODEL=~/models/ornith-nvfp4a16-gptq-text-only-norope
ENVBIN=~/swebench-env/bin
CFGDIR=~/ornith-nvfp4/scripts/swebench
CONFIG=ornith_overrides_ladder.yaml
STOCK=~/swebench-env/lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml
LOG=~/ornith_serve.log
BASE_OUT="$HOME/swebench_ornith_night"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)

cleanup () {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
  fi
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 5
  echo "=== GPU after teardown: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null) ==="
}
trap cleanup EXIT INT TERM

SERVER_PID=""
MAXLEN=""
for ML in 49152 40960 32768; do
  echo "=== starting vLLM ($RUN_STAMP): trying MAXLEN=$ML ==="
  ~/vllm-env/bin/vllm serve "$MODEL" \
    --served-model-name ornith-16gb --port "$PORT" \
    --max-model-len "$ML" --max-num-seqs 2 \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --enable-prefix-caching --max-num-batched-tokens 4096 \
    --compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}' \
    --mamba-cache-mode align --mamba-block-size 528 --block-size 528 > "$LOG" 2>&1 &
  SERVER_PID=$!
  READY=0
  for i in $(seq 1 150); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200 && { READY=1; break; }
    sleep 4
  done
  if [ "$READY" = 1 ]; then MAXLEN="$ML"; break; fi
  echo "--- MAXLEN=$ML did not come up; falling back ---"
  tail -5 "$LOG" | grep -aE "Error|error|KV" | head -3
  kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; SERVER_PID=""
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null; sleep 5
done
[ -n "$MAXLEN" ] || { echo "!! no MAXLEN worked"; tail -20 "$LOG"; exit 1; }
grep -a "GPU KV cache size\|Maximum concurrency\|Available KV cache" "$LOG" | tail -3 | sed 's/^/   /'

"$ENVBIN/python" ~/preflight_ornith.py || { echo "!! preflight failed"; exit 1; }

rollout () { # $1=outdir $2=slice-args
  mkdir -p "$1"
  cd "$CFGDIR" || return 1
  LITELLM_MODEL_REGISTRY_PATH="$CFGDIR/registry_ornith.json" \
  "$ENVBIN/mini-extra" swebench \
    --subset verified --split test --shuffle $2 --workers 1 \
    -o "$1" -c "$STOCK" -c "$CFGDIR/$CONFIG" 2>&1 | tail -15
}

summarize () { # $1=outdir $2=expected-count
  local d="$1" want="$2"
  if ! [ -f "$d/preds.json" ]; then echo "   !! no preds.json in $d"; return 1; fi
  "$ENVBIN/python" - "$d/preds.json" "$want" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
want = int(sys.argv[2])
ne = sum(1 for v in d.values() if (v.get("model_patch") or "").strip())
print(f"   instances: {len(d)}/{want} | nonempty patches: {ne}/{len(d)}")
sys.exit(0 if len(d) >= want else 1)
PY
}

exit_statuses () { # $1=outdir
  cat "$1"/exit_statuses_*.yaml 2>/dev/null | sed 's/^/   /' | head -20
}

echo "=== PHASE A: single-instance submission check ==="
A_OUT="$BASE_OUT/${RUN_STAMP}_a1"
rollout "$A_OUT" "--slice 0:1"
summarize "$A_OUT" 1 || { echo "!! GATE A FAILED (mechanics)"; exit 1; }
exit_statuses "$A_OUT"
echo "GATE A PASSED $(date)"

echo "=== PHASE B: 10-instance bounded sample ==="
B_OUT="$BASE_OUT/${RUN_STAMP}_b10"
rollout "$B_OUT" "--slice 0:10"
summarize "$B_OUT" 10 || { echo "!! GATE B FAILED (mechanics)"; exit 1; }
exit_statuses "$B_OUT"
echo "GATE B PASSED $(date)"

echo "=== PHASE C: full 50-instance pilot ==="
C_OUT="$BASE_OUT/${RUN_STAMP}_c50"
rollout "$C_OUT" "--slice 0:50"
summarize "$C_OUT" 50 || { echo "!! GATE C FAILED (mechanics)"; exit 1; }
exit_statuses "$C_OUT"
echo "PILOT COMPLETE $(date)"

curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E "vllm:prefix_cache_(queries|hits)_total\{" | sed 's/^/   /'
