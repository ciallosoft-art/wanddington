#!/usr/bin/env bash
# Experiment 19 — component ablation of the REPORTED C-arm on each of BDA's three tiers
# (docs/tier_extension_plan.md, part 丙1).
#
# The paper's older ablation table runs the legacy target-aware router; these arms subclass the reported,
# leakage-free C-arm and change one thing each (arms/waddington_c_ablation_arms.py):
#   waddington_c             the reported system, re-run here so every variant is paired with it
#   waddington_c_nomem       no cross-experiment memory
#   waddington_c_static_ml   no in-screen retraining of the ML branch (round-0 prior throughout)
#   waddington_c_anon_names  the LLM sees anonymous identifiers, no features
# The fourth ablation, no LLM, is the C-arm's own ML engine (ml_engine_lightgbm, experiment 15; no GPT).
#
# Each tier copies the settings of the run the paper reports for it:
#   headline   experiment 05 / 17: corrected task prompts, replogle_transcriptomics_v2_20rep memory
#   full_pool  experiment 08: the tier_a tables and that run's own memory
#   pairs      experiment 09 at 128 pairs per round (640 picks), pair-mean tables, the single-gene memory
#              of tier_a_full_pool_20rep. That memory predates the task-prompt protocol field, so the legacy
#              switch is on (for pair screens it changes nothing else), as in experiments 13 and 18.
#
#   pairs_interaction  the same on the table with the interaction features (experiment 12's table)
#
#   TIER=headline|full_pool|pairs|pairs_interaction bash experiments/19_tier_ablations.sh [replicates]   (default 20; resumable)
export WADDINGTON_LEGACY_TASK_PROMPTS=0
set -euo pipefail
cd "$(dirname "$0")/.."

TIER="${TIER:?set TIER=headline, full_pool or pairs}"
REPS="${1:-20}"
MODEL="openai-codex/gpt-5.6-luna"
ARMS="${ARMS:-waddington_c waddington_c_nomem waddington_c_static_ml waddington_c_anon_names}"
OUT_DIR="workspace/results/tier_ablations/$TIER"
OUT_NAME="${OUT_NAME:-ablations}"
# Part 丁 of docs/tier_extension_plan.md reuses this harness: ARMS (the ladder's A0.5 / A1, or waddington_c__rf),
# OUT_NAME (a separate result file), EXCLUDE_STUDY=1 (the leave-one-study-out prior), SCREENS (a subset).
# Part 戊 (why the ladder reversed) adds KEEP_ANCHORS=1 (the C-arm keeps the anchor features, in the ML branch
# and in the LLM's feature view) and OLD_MEMORY=1 (the first GPT run's memory and the uncorrected task prompts,
# the conditions of the earlier ladder; headline only). Part 己 adds CRISPRA=1: the CRISPRa validation table (the
# nine screens plus Schmidt et al.'s IL2_crispra / IFNG_crispra; workspace/evaluation/build_crispra_validation.py),
# for the modality probe on the reported system (headline only; SCREENS="IL2_crispra IFNG_crispra").

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
[ "${EXCLUDE_STUDY:-0}" = 1 ] && export WADDINGTON_EXCLUDE_STUDY=1
[ "${KEEP_ANCHORS:-0}" = 1 ] && export WADDINGTON_DROP_ANCHOR_FEATS=0
if [ "${OLD_MEMORY:-0}" = 1 ]; then
  [ "$TIER" = headline ] || { echo "OLD_MEMORY applies to the headline tier only" >&2; exit 2; }
  MEMORY_PATH="$PWD/workspace/results/gpt/gpt_5_6_luna_none_20rep/experience_memory.json"
  export WADDINGTON_MEMORY_PATH="$MEMORY_PATH" WADDINGTON_LEGACY_TASK_PROMPTS=1
fi
if [ "${CRISPRA:-0}" = 1 ]; then
  [ "$TIER" = headline ] || { echo "CRISPRA applies to the headline tier only" >&2; exit 2; }
  export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_crispra_validation.csv"
fi
[ -n "${SCREENS:-}" ] && DATASETS=(--datasets $SCREENS)

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

echo "=== Experiment 19: ablations of the reported C-arm, tier $TIER, $REPS replicates, memory $MEMORY_PATH ==="
conda run --no-capture-output -n waddington-bio python3 -m waddington_select \
  --arms $ARMS ${DATASETS[@]+"${DATASETS[@]}"} \
  --replicates "$REPS" \
  --seed-mode local \
  --rounds 5 \
  --resume \
  --out "$OUT_DIR/${OUT_NAME}_${REPS}rep.json"
echo "### TIER $TIER DONE $(date '+%m-%d %H:%M')"
