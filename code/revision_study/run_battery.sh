#!/bin/bash
# Full re-recording of the revision-study candidate pools (deepest
# reproduction tier). Four sequential full 100-case evaluation runs:
# three gap-aware instrumented runs (A-C) and one end-to-end raw-baseline
# run (E). Requires the FloorSet dataset (FLOORSET_ROOT) and a GPU;
# roughly 70 minutes per run on an RTX 3090.
#
# Pools are appended as JSON lines via the solver's RERANK_DUMP switch;
# RERANK_MODE=raw activates the raw-magnitude final selection for run E.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL="$HERE/../project_code/gap_aware_reranking_floorplanning/evaluation/quick_eval.py"
OUT="$HERE/pools_rerun"
mkdir -p "$OUT"

run_one () {
  local name="$1"; shift
  echo "=== $name === $(date)"
  RERANK_DUMP="$OUT/$name.jsonl" "$@" python "$EVAL" --full --verbose \
    > "$OUT/$name.log" 2>&1
  echo "$name exit: $? $(date)"
}

run_one runA_gap env
run_one runE_raw env RERANK_MODE=raw
run_one runB_gap env
run_one runC_gap env
echo "=== BATTERY DONE === $(date)"
echo "Re-analyze with:"
echo "  python $HERE/analyze_pools.py $OUT/runA_gap.jsonl $OUT/runA_gap.log --out $HERE/analysis/analysis_runA_gap"
