#!/usr/bin/env bash
# Ornith SWE-bench ladder, rung 1: single-instance smoke.
# Serve + rollout + teardown in ONE process - same architecture as the prior
# project's run_pilot_all.sh, for the same documented reasons (separate-invocation
# servers die with their parent; litellm-layer tool-call breakage that a curl
# preflight misses; silent multi-minute retry backoff).
#
# Ornith-specific serving flags come from the evaluated configuration
# (scripts/eval/eval_suite_vllm.sh), not from guesses:
#   --mamba-block-size 528 --block-size 528   (block alignment required)
#   --mamba-cache-mode align                  (experimental upstream; prefix
#                                              counters are printed post-run,
#                                              not trusted - serving_notes.md)
# cudagraph sizes [1,2] PIECEWISE also satisfies the documented mamba-cache
# assertion constraint (< default 512) from docs/serving_notes.md.
# NOT carried from KAT: --kv-cache-dtype fp8 and --language-model-only
# (vision is physically stripped here; fp8 KV unmeasured on Ornith's hybrid cache).
set -uo pipefail

N="${1:-1}"
OUT="${2:-$HOME/swebench_ornith_smoke}"
PORT=8000
MODEL=~/models/ornith-nvfp4a16-gptq-text-only-norope
ENVBIN=~/swebench-env/bin
CFGDIR=~/ornith-nvfp4/scripts/swebench
KAT_CONFIG="${ORNI_CONFIG:-ornith_overrides_smoke.yaml}"
MAXLEN="${MAXLEN:-32768}"
MAXSEQS="${MAXSEQS:-2}"
STOCK=~/swebench-env/lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml
LOG=~/ornith_serve.log

cleanup () {
  if [ -n "${SERVER_PID:-}" ]; then
    echo "=== stopping server ($SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 5
  local held
  held=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
  echo "=== GPU after teardown: ${held:-unknown} ==="
}
trap cleanup EXIT INT TERM

echo "=== starting vLLM (full log -> $LOG) ==="
echo "    max_model_len=$MAXLEN  max_num_seqs=$MAXSEQS  config=$KAT_CONFIG"
~/vllm-env/bin/vllm serve "$MODEL" \
  --served-model-name ornith-16gb --port "$PORT" \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --enable-prefix-caching --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}' \
  --mamba-cache-mode align --mamba-block-size 528 --block-size 528 > "$LOG" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 150); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "!! server died:"; grep -aE "AssertionError|ValueError|RuntimeError|Error" "$LOG" | head -5; exit 1; }
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200 && { READY=1; break; }
  sleep 4
done
[ "$READY" = 1 ] || { echo "!! timed out"; tail -20 "$LOG"; exit 1; }
echo "   ready"
grep -a "GPU KV cache size\|Maximum concurrency\|Available KV cache" "$LOG" | tail -3 | sed 's/^/   /'

echo
echo "=== PREFLIGHT: the agent's exact model path, through litellm ==="
"$ENVBIN/python" ~/preflight_ornith.py || { echo "!! aborting before spending containers or hours"; exit 1; }

echo
echo "=== PREFLIGHT: docker ==="
docker run --rm alpine:latest echo docker_ok 2>/dev/null | grep -q docker_ok \
  || { echo "!! docker run failed"; exit 1; }
echo "   ok"

echo
echo "=== ROLLOUT: $N instance(s) -> $OUT (config: $KAT_CONFIG) ==="
mkdir -p "$OUT"
cd "$CFGDIR" || exit 1
LITELLM_MODEL_REGISTRY_PATH="$CFGDIR/registry_ornith.json" \
"$ENVBIN/mini-extra" swebench \
  --subset verified --split test --shuffle --slice "0:$N" --workers 2 \
  -o "$OUT" -c "$STOCK" -c "$CFGDIR/$KAT_CONFIG" 2>&1 | tail -25

echo
echo "=== prefix cache counters during the rollout ==="
curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
  | grep -E "vllm:prefix_cache_(queries|hits)_total\{" | sed 's/^/   /'

echo
echo "=== ARTIFACTS (this stack returns 0 on failure routinely) ==="
if [ -f "$OUT/preds.json" ]; then
  "$ENVBIN/python" - "$OUT/preds.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("   instances with predictions:", len(d))
for k, v in d.items():
    p = (v.get("model_patch") or "").strip()
    print(f"     {k:<34} patch {len(p):>6} chars")
PY
else
  echo "   !! no preds.json"; ls -la "$OUT" | head
fi
