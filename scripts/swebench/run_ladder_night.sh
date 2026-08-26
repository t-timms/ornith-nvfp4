#!/usr/bin/env bash
# Ornith SWE-bench overnight ladder (2026-08-26 night).
#
# Chain: 1-instance submission check -> 10-instance bounded sample -> 50-instance
# pilot, each phase gated on INFRASTRUCTURE health, not score:
#   Gate A: the single instance produces a non-empty patch (submission mechanics work)
#   Gate B: >=8/10 bounded-sample instances produce non-empty patches
# Gates are deliberately about mechanics (server stability, tool parsing, patch
# submission) - they make NO capability claim; scoring/grading happens after,
# by a human reviewing artifacts. This honors both the roadmap's no-skipped-rungs
# discipline and the user's instruction to run SWE-bench overnight unattended.
#
# Config (see ornith_overrides_ladder.yaml header for derivations):
#   MAXLEN=32768, workers=1 (serialized - KV ceiling 40,366 tokens = 1.23x at 32K),
#   step_limit=65, sampling from the checkpoint's own generation_config.
set -uo pipefail

PORT=8000
MODEL=~/models/ornith-nvfp4a16-gptq-text-only-norope
ENVBIN=~/swebench-env/bin
CFGDIR=~/ornith-nvfp4/scripts/swebench
CONFIG=ornith_overrides_ladder.yaml
MAXLEN=32768
STOCK=~/swebench-env/lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml
LOG=~/ornith_serve.log
BASE_OUT="$HOME/swebench_ornith_night"
RUN_STAMP=$(date +%Y%m%d_%H%M%S)

nonempty_patches () {
  local out="$1" min="$2"
  "$ENVBIN/python" - "$out/preds.json" "$min" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
ne = sum(1 for v in d.values() if (v.get("model_patch") or "").strip())
print(f"   nonempty patches: {ne}/{len(d)}")
sys.exit(0 if ne >= int(sys.argv[2]) else 1)
PY
}

cleanup () {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
  fi
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 5
  echo "=== GPU after teardown: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null) ==="
}
trap cleanup EXIT INT TERM

echo "=== starting vLLM ($RUN_STAMP): MAXLEN=$MAXLEN workers=1 ==="
~/vllm-env/bin/vllm serve "$MODEL" \
  --served-model-name ornith-16gb --port "$PORT" \
  --max-model-len "$MAXLEN" --max-num-seqs 2 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --enable-prefix-caching --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}' \
  --mamba-cache-mode align --mamba-block-size 528 --block-size 528 > "$LOG" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 150); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "!! server died:"; grep -aE "AssertionError|ValueError|RuntimeError" "$LOG" | head -5; exit 1; }
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200 && { READY=1; break; }
  sleep 4
done
[ "$READY" = 1 ] || { echo "!! server timeout"; tail -20 "$LOG"; exit 1; }
echo "server ready $(date)"
grep -a "GPU KV cache size\|Maximum concurrency\|Available KV cache" "$LOG" | tail -3 | sed 's/^/   /'

"$ENVBIN/python" ~/preflight_ornith.py || { echo "!! preflight failed"; exit 1; }

rollout () { # $1=n $2=outdir $3=extra-flags
  mkdir -p "$2"
  cd "$CFGDIR" || return 1
  LITELLM_MODEL_REGISTRY_PATH="$CFGDIR/registry_ornith.json" \
  "$ENVBIN/mini-extra" swebench \
    --subset verified --split test --shuffle $3 --workers 1 \
    -o "$2" -c "$STOCK" -c "$CFGDIR/$CONFIG" 2>&1 | tail -15
}

echo "=== PHASE A: single-instance submission check ==="
A_OUT="$BASE_OUT/${RUN_STAMP}_a1"
rollout 1 "$A_OUT" "--slice 0:1"
[ -f "$A_OUT/preds.json" ] || { echo "!! no preds.json in A"; exit 1; }
nonempty_patches "$A_OUT" 1 || { echo "!! GATE A FAILED: no non-empty patch on the single instance"; exit 1; }
echo "GATE A PASSED $(date)"

echo "=== PHASE B: 10-instance bounded sample ==="
B_OUT="$BASE_OUT/${RUN_STAMP}_b10"
rollout 10 "$B_OUT" "--slice 0:10"
[ -f "$B_OUT/preds.json" ] || { echo "!! no preds.json in B"; exit 1; }
nonempty_patches "$B_OUT" 8 || { echo "!! GATE B FAILED: <8/10 instances produced a patch"; exit 1; }
echo "GATE B PASSED $(date)"

echo "=== PHASE C: full 50-instance pilot ==="
C_OUT="$BASE_OUT/${RUN_STAMP}_c50"
rollout 50 "$C_OUT" "--slice 0:50"
[ -f "$C_OUT/preds.json" ] && nonempty_patches "$C_OUT" 0
echo "PILOT COMPLETE $(date)"

curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -E "vllm:prefix_cache_(queries|hits)_total\{" | sed 's/^/   /'
