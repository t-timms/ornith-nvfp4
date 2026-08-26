#!/usr/bin/env bash
# Grade the SWE-bench pilot's nonempty patches with the official harness.
# Adapted from kat-coder-16gb/scripts/swebench/grade_pilot.sh.
set -uo pipefail

OUT="${1:-$HOME/swebench_ornith_night/20260826_061122_c50}"
RUN_ID="ornith_c50_$(date +%H%M%S)"
ENVBIN=~/swebench-env/bin

echo "=== converting preds.json -> preds.jsonl ==="
"$ENVBIN/python" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
d = json.loads((out / "preds.json").read_text())
rows, empty = [], 0
for iid, v in d.items():
    patch = (v.get("model_patch") or "").strip()
    if not patch:
        empty += 1
        continue
    rows.append({
        "instance_id": iid,
        "model_patch": v["model_patch"],
        "model_name_or_path": v.get("model_name_or_path") or "ornith-16gb",
    })
p = out / "preds.jsonl"
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"   {len(rows)} gradable, {empty} empty (empty ones cannot resolve and are")
print("   correctly counted as unresolved in the denominator later)")
PY

echo
echo "=== running the official harness ==="
cd "$OUT" || exit 1
"$ENVBIN/python" -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --predictions_path "$OUT/preds.jsonl" \
  --max_workers 2 \
  --run_id "$RUN_ID"

echo
echo "=== ARTIFACT: the report file, not the exit code ==="
report=$(ls -1t "$OUT"/*.json 2>/dev/null | grep -i "$RUN_ID" | head -1)
if [ -z "$report" ]; then
  report=$(ls -1t ./*."$RUN_ID".json 2>/dev/null | head -1)
fi
if [ -n "$report" ] && [ -f "$report" ]; then
  echo "   $report"
  "$ENVBIN/python" - "$report" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("total_instances", "submitted_instances", "completed_instances",
          "resolved_instances", "unresolved_instances", "empty_patch_instances",
          "error_instances"):
    if k in d:
        print(f"   {k:<24}: {d[k]}")
resolved = d.get("resolved_instances") or []
if resolved:
    print("   resolved:")
    for r in resolved:
        print(f"     - {r}")
PY
else
  echo "   !! no report json found; listing what was written:"
  ls -la "$OUT" | tail -10
fi
