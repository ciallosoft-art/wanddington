#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# GeneDisco's acquisition functions on BDA's other two difficulty tiers.
#
# The headline tier's GeneDisco rows live in workspace/results/sequential/genedisco_ported.json
# (three replicates, five acquisitions). This runs the same five acquisitions, the same three
# replicates, on the full-pool and combinatorial tables so all three tiers carry the same external
# active-learning baseline. No LLM is involved.
#
# Usage: bash experiments/10_tiers_genedisco.sh [tier_a|tier_b|both]
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:-both}"

ARMS=(genedisco_topuncertain genedisco_softuncertain genedisco_marginsample genedisco_coreset genedisco_kmeans_data)

unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE \
      WADDINGTON_FEATURE_POLICY WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY \
      WADDINGTON_REASON_FROM_STRUCTURE WADDINGTON_EXCLUDE_STUDY PYTHONOPTIMIZE
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Sequential, deliberately. WADDINGTON_PARALLEL runs repeats in a ThreadPoolExecutor, and sklearn's
# KMeans spawns its own threads for the n_init loop (8s wall vs 26s user on one IFNG repeat), so the
# kmeans_data acquisition deadlocks when two repeats run concurrently: 33 threads, all sleeping, no
# progress. Parallelism buys nothing here anyway -- there is no LLM latency to hide and a repeat costs
# seconds. The LLM-backed experiments (04/05/08/09) do not use this arm and are unaffected.
export WADDINGTON_PARALLEL=1

# run_tier <label> <v1 csv> <v3 csv> <outdir> <datasets...> [-- EXTRA_ENV=...]
run_tier () {
  local name="$1" v1="$2" v3="$3" outdir="$4"; shift 4
  local datasets=() extra=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do datasets+=("$1"); shift; done
  [[ "${1:-}" == "--" ]] && shift && extra=("$@")
  echo "=== GeneDisco on $name ==="
  mkdir -p "$outdir"
  env WADDINGTON_TRAINING_CSV="$PWD/$v1" WADDINGTON_TRAINING_CSV_V3="$PWD/$v3" "${extra[@]}" \
    conda run -n waddington-bio python3 -m waddington_select \
      --arms "${ARMS[@]}" --datasets "${datasets[@]}" \
      --replicates 3 --seed-mode local --rounds 5 --resume \
      --out "$outdir/genedisco_3rep.json"
}

if [[ "$WHICH" == "tier_a" || "$WHICH" == "both" ]]; then
  run_tier "1-gene full pool" \
    workspace/evaluation/lgbm_training_data_tier_a.csv \
    workspace/evaluation/lgbm_training_data_tier_a_v3.csv \
    workspace/results/gpt/tier_a_full_pool_20rep \
    IFNG IL2 Sanchez21 Sanchez21_down Carnevale22 Scharenberg22 Steinhart \
    Replogle_K562_essential Replogle_K562_gwps
fi

if [[ "$WHICH" == "tier_b" || "$WHICH" == "both" ]]; then
  run_tier "2-gene combinations" \
    workspace/evaluation/lgbm_training_data_horlbeck_pairs.csv \
    workspace/evaluation/lgbm_training_data_horlbeck_pairs_v3.csv \
    workspace/results/gpt/tier_b_pairs_20rep \
    Horlbeck -- WADDINGTON_EXTRA_DATASETS=Horlbeck:128
fi
echo "Done."
