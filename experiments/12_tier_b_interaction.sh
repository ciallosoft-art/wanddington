#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# Combinatorial tier WITH genuine interaction features x(g_a, g_b).
#
# The pair table built by build_horlbeck_pairs.py describes a pair only by symmetric functions of its
# two genes (mean, absolute difference), which a genetic interaction is orthogonal to by construction:
# GI is the deviation from the additive expectation, so the additive part is exactly what has been
# subtracted out. Measured consequence: the leave-one-phenotype-out prior scores AUC 0.4959 and its top
# 640 pairs hold fewer hits than random.
#
# build_pair_interaction_features.py adds seven features that are properties of the PAIR -- DepMap
# co-essentiality, KEGG/Reactome co-membership, the STRING a--b edge, ARCHS4 co-expression. Ranking the
# pool by an equal-weight rank-sum of them alone puts 182 hits in the top 640 (random 63), with no LLM.
#
# Deterministic arms already confirm the mechanism: static_ranker is unchanged at 58 hits (these
# features are constant-0 in every training row, because a single-gene screen contains no pair
# interactions), while online_adaptive rises 93 -> 117 by learning them from pairs revealed in-screen.
# This script settles the LLM-backed arm, which a five-replicate pilot left inconclusive
# (131.8 +- 38.7 against a 112.6 baseline -- the interval contains it).
set -euo pipefail
cd "$(dirname "$0")/.."

RESULT_DIR="workspace/results/gpt/tier_b_pairs_interaction_20rep"
REPS="${1:-20}"

unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE \
      WADDINGTON_FEATURE_POLICY WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY \
      WADDINGTON_REASON_FROM_STRUCTURE WADDINGTON_EXCLUDE_STUDY PYTHONOPTIMIZE

export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs.csv"
export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_v3.csv"
export WADDINGTON_EXTRA_DATASETS="Horlbeck:128"
# The cross-experiment memory is the nine full-pool single-gene screens, as in experiment 09.
export WADDINGTON_MEMORY_PATH="$PWD/workspace/results/gpt/tier_a_full_pool_20rep/experience_memory.json"
export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_CHAT_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export WADDINGTON_FULL_SELECTED_CONTEXT=1
export WADDINGTON_LLM_NAMED_ONLY=1
export WADDINGTON_DUMP_SELECTIONS=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-4}"
export WADDINGTON_GPT_RESULT_DIR="$RESULT_DIR"

[[ -f workspace/evaluation/pair_interaction_features.csv ]] || {
  echo "ERROR: pair_interaction_features.csv missing — run build_pair_interaction_features.py" >&2; exit 1; }

mkdir -p "$RESULT_DIR"
echo "=== Horlbeck with interaction features, $REPS replicates ==="

# Deterministic arms: identical every replicate, so one pass is the whole answer.
conda run -n waddington-bio python3 -m waddington_select \
  --arms random coreset static_ranker online_adaptive \
  --datasets Horlbeck --replicates 1 --seed-mode local --rounds 5 --resume \
  --out "$RESULT_DIR/deterministic_1rep.json"

conda run -n waddington-bio python3 -m waddington_select \
  --arms waddington_c \
  --datasets Horlbeck --replicates "$REPS" --seed-mode local --rounds 5 --resume \
  --out "$RESULT_DIR/waddington_c_${REPS}rep.json"

echo "Done. $RESULT_DIR"
