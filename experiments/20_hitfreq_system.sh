#!/usr/bin/env bash
# Experiment 20 — the full reported C-arm restricted to the single cross-screen hit-frequency feature
# (WADDINGTON_HITFREQ_ONLY=1), the recurrence baseline of "What the prior learns" and claim 3.
#
# The first run of this (workspace/results/gpt/router/hitfreq_prior_full.json, 2026-08-25, three replicates,
# before the task-prompt correction) broke the feature's massive ties by stored row order, which puts some
# screens' hits first; it scored 0.255 and 0.688 on K562-Essential, above the full system. The arms now order
# the candidates label-free in this mode (eaa46bf). This re-runs it with the reported headline settings
# (corrected prompts, the replogle_transcriptomics_v2_20rep memory, twenty replicates), so it pairs with the
# reported twenty-replicate C-arm.
#
#   [TIER=full_pool] bash experiments/20_hitfreq_system.sh [replicates]   (default 20; resumable)
export WADDINGTON_LEGACY_TASK_PROMPTS=0
set -euo pipefail
cd "$(dirname "$0")/.."

TIER="${TIER:-headline}"  # headline, or full_pool (2026-10-16); not defined for pairs
case "$TIER" in headline|full_pool) ;; *) echo "TIER must be headline or full_pool" >&2; exit 2 ;; esac
REPS="${1:-20}"
MODEL="openai-codex/gpt-5.6-luna"
ARMS="waddington_c"
OUT_DIR="workspace/results/hitfreq_system"
[ "$TIER" = full_pool ] && OUT_DIR="workspace/results/hitfreq_system/full_pool"

unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE
unset WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV WADDINGTON_TRAINING_CSV_V3 WADDINGTON_EXTRA_DATASETS
unset PYTHONOPTIMIZE

DATASETS=()
case "$TIER" in
  headline)
    MEMORY_PATH="$PWD/workspace/results/gpt/replogle_transcriptomics_v2_20rep/experience_memory.json"
    export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-6}" ;;
  full_pool)
    MEMORY_PATH="$PWD/workspace/results/gpt/tier_a_full_pool_v2_20rep/experience_memory.json"
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv"
    export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-6}" ;;
  pairs|pairs_interaction)
    MEMORY_PATH="$PWD/workspace/results/gpt/tier_a_full_pool_20rep/experience_memory.json"
    export WADDINGTON_LEGACY_TASK_PROMPTS=1
    # pairs: the pair-mean table of experiment 09 (build_horlbeck_pairs_mean.py); pairs_interaction: the same
    # table with the eight interaction columns that experiment 12 later added to it
    PT="lgbm_training_data_horlbeck_pairs_mean"; [ "${TIER:-}" = pairs_interaction ] && PT="lgbm_training_data_horlbeck_pairs"
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/${PT}.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/${PT}_v3.csv"
    export WADDINGTON_EXTRA_DATASETS="Horlbeck:128"
    export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-4}"  # each arm holds its own copy of a 46 MB table
    DATASETS=(--datasets Horlbeck) ;;
  *) echo "TIER must be headline, full_pool, pairs or pairs_interaction" >&2; exit 2 ;;
esac

export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="$MODEL"
export WADDINGTON_CHAT_MODEL="$MODEL"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export WADDINGTON_LLM_SLEEP="${WADDINGTON_LLM_SLEEP:-0}"
export WADDINGTON_FULL_SELECTED_CONTEXT=1
export WADDINGTON_LLM_NAMED_ONLY=1
export WADDINGTON_DUMP_SELECTIONS=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WADDINGTON_GPT_RESULT_DIR="$OUT_DIR"
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"
export WADDINGTON_HITFREQ_ONLY=1

[ -s "$MEMORY_PATH" ] || { echo "ERROR: memory $MEMORY_PATH missing" >&2; exit 1; }
mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/.experiment.lock"
if ! flock -n 9; then
  echo "ERROR: another experiment already holds $OUT_DIR/.experiment.lock" >&2
  exit 1
fi

node --input-type=module -e '
  import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
  const target = process.env.WADDINGTON_PI_MODEL;
  const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
  if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
  console.log(`Authorized model: ${target}`);
'

echo "=== Experiment 20: C-arm on the hit-frequency feature only, $REPS replicates, memory $MEMORY_PATH ==="
conda run --no-capture-output -n waddington-bio python3 -m waddington_select \
  --arms $ARMS ${DATASETS[@]+"${DATASETS[@]}"} \
  --replicates "$REPS" \
  --seed-mode local \
  --rounds 5 \
  --resume \
  --out "$OUT_DIR/hitfreq_c_${REPS}rep.json"
echo "### HITFREQ DONE $(date '+%m-%d %H:%M')"
