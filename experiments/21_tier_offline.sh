#!/usr/bin/env bash
# Experiment 21 — the offline analyses of the reported C-arm on BDA's other two tiers
# (docs/tier_extension_plan.md, part 乙; no LLM calls).
#
#   full_pool  source attribution and ML-adjusted endorsement, replayed from the reported full-pool run
#              (experiment 08); recurrent vs novel hits on that tier
#   pairs      source attribution and ML-adjusted endorsement, replayed from experiment 19's paired run of
#              the reported C-arm on the pair-mean table (the September run of experiment 09 predates the
#              selection dumps the replay needs)
#
# Each tier's tables are set before Python starts, as the oracle and the arms read them at import.
#
#   TIER=full_pool|pairs bash experiments/21_tier_offline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

TIER="${TIER:?set TIER=full_pool or pairs}"
OUT_DIR="workspace/results/tier_offline/$TIER"
mkdir -p "$OUT_DIR"
for v in $(env | grep -o '^WADDINGTON_[A-Z0-9_]*'); do unset "$v"; done
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "$TIER" in
  full_pool)
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv"
    RUN=workspace/results/gpt/tier_a_full_pool_v2_20rep/main_table_20rep.json
    MANIFEST=workspace/results/gpt/tier_a_full_pool_v2_20rep/waddington_c_20rep.manifest.json
    python3 -m waddington_select.analysis.novel_hit_analysis \
      --csv "$WADDINGTON_TRAINING_CSV_V3" --run "$RUN" --out "$OUT_DIR/novel_hit_analysis.json" ;;
  pairs)
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_mean.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_mean_v3.csv"
    export WADDINGTON_EXTRA_DATASETS="Horlbeck:128"
    RUN=workspace/results/tier_ablations/pairs/ablations_20rep.json
    MANIFEST=workspace/results/tier_ablations/pairs/ablations_20rep.manifest.json ;;
  *) echo "TIER must be full_pool or pairs" >&2; exit 2 ;;
esac

python3 -m waddington_select.analysis.replay_attribution \
  --run "$RUN" --manifest "$MANIFEST" --out "$OUT_DIR/attribution_replay.json"
echo "### TIER $TIER OFFLINE DONE $(date '+%m-%d %H:%M')"
