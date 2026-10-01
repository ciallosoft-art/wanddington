#!/usr/bin/env bash
# Experiment 15 — alternative ML engines, ML only (no LLM calls). docs/ml_engines_plan.md, steps 2-3.
#
# Every arm is the C-arm's own ML engine (MLEngineArm: same metadata DepMap feature policy, same anchor
# drop, same leave-one-screen-out rows, same x10 online weighting) with only the model swapped:
#   ml_engine_<model>  retrained on each round's revealed genes
#   ml_static_<model>  the round-0 cross-experiment prior, never retrained
# Deterministic engines run two replicates, which must come out identical (checked in step 5); seeded
# engines run five seeds via --seed-mode local (replicate i uses seed i).
#
#   experiments/15_ml_engines.sh tabular   # lightgbm logreg lambdarank | rf mlp
#   experiments/15_ml_engines.sh graph     # rwr | gnn
set -euo pipefail
cd "$(dirname "$0")/.."

unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
unset WADDINGTON_REASON_FROM_STRUCTURE PYTHONOPTIMIZE
export WADDINGTON_PARALLEL=1  # the engines already use every core; repeat-level threads would only contend
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # reproducible cuBLAS for the GNN (dense neighbour mean on the GPU)

OUT=workspace/results/ml_engines
mkdir -p "$OUT"

GROUP="${1:-tabular}"
case "$GROUP" in
  tabular) DET="lightgbm logreg lambdarank"; SEEDED="rf mlp" ;;
  graph)   DET="rwr";                        SEEDED="gnn" ;;
  *) echo "usage: $0 [tabular|graph]" >&2; exit 2 ;;
esac

arms() { for m in $1; do printf 'ml_engine_%s ml_static_%s ' "$m" "$m"; done; }

run() {  # $1 = arm list, $2 = replicates, $3 = output file
  conda run --no-capture-output -n waddington-bio python3 -m waddington_select.run_sequential \
    --arms $1 --replicates "$2" --seed-mode local --out "$3" --resume
}

echo "== $GROUP: deterministic engines ($DET), 2 replicates"
run "$(arms "$DET")" 2 "$OUT/${GROUP}_deterministic.json"
echo "== $GROUP: seeded engines ($SEEDED), 5 seeds"
run "$(arms "$SEEDED")" 5 "$OUT/${GROUP}_seeded.json"
echo "done -> $OUT"
