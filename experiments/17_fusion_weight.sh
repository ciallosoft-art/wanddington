#!/usr/bin/env bash
# Experiment 17 — sensitivity of the full GPT C-arm to its ML-LLM fusion weight (docs/fusion_weight_plan.md).
#
# Everything is held at the reported protocol except w_llm: the same GPT-5.6 Luna settings, task prompts,
# named-only vote and verified experience memory as the corrected-prompt 20-replicate run
# (replogle_transcriptomics_v2_20rep, w_llm = 0.2, C = 0.2603), which is the grid's 0.2 point. The grid's
# 0.0 point is the C-arm's own ML engine (identical selections; no GPT needed). One process per weight,
# because WADDINGTON_FORCE_WLLM is read when the arm is built.
#
#   experiments/17_fusion_weight.sh               # 0.1 0.3 0.4 0.5, five replicates each
#   experiments/17_fusion_weight.sh 0.4 0.5       # a subset (resumable: --resume per weight)
export WADDINGTON_LEGACY_TASK_PROMPTS=0
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="openai-codex/gpt-5.6-luna"
REF_DIR="workspace/results/gpt/replogle_transcriptomics_v2_20rep"
MEMORY_PATH="$REF_DIR/experience_memory.json"
OUT_DIR="workspace/results/fusion_weight"
REPS="${REPS:-5}"
WEIGHTS="${*:-0.1 0.3 0.4 0.5}"

unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE
unset WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV WADDINGTON_TRAINING_CSV_V3 WADDINGTON_EXTRA_DATASETS
unset PYTHONOPTIMIZE

# TIER=full_pool|pairs runs the same grid on BDA's other two tiers (docs/tier_extension_plan.md, part 丙2),
# with each tier's reported settings (as in experiment 19); the 0.2 reference is experiment 19's paired
# twenty-replicate run of the reported C-arm on that tier. The default is the headline tier above.
DATASETS=()
case "${TIER:-headline}" in
  headline) ;;
  full_pool)
    MEMORY_PATH="workspace/results/gpt/tier_a_full_pool_v2_20rep/experience_memory.json"
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv"
    OUT_DIR="workspace/results/fusion_weight/full_pool" ;;
  pairs|pairs_interaction)
    MEMORY_PATH="workspace/results/gpt/tier_a_full_pool_20rep/experience_memory.json"
    export WADDINGTON_LEGACY_TASK_PROMPTS=1  # that memory predates the protocol field; nothing else changes
    # pairs: the pair-mean table of experiment 09 (build_horlbeck_pairs_mean.py); pairs_interaction: the same
    # table with the eight interaction columns that experiment 12 later added to it
    PT="lgbm_training_data_horlbeck_pairs_mean"; [ "${TIER:-}" = pairs_interaction ] && PT="lgbm_training_data_horlbeck_pairs"
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/${PT}.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/${PT}_v3.csv"
    export WADDINGTON_EXTRA_DATASETS="Horlbeck:128"
    export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-4}"
    DATASETS=(--datasets Horlbeck)
    OUT_DIR="workspace/results/fusion_weight/${TIER}" ;;
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
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-6}"
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"

[ -s "$MEMORY_PATH" ] || { echo "ERROR: reference memory $MEMORY_PATH missing" >&2; exit 1; }
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

for w in $WEIGHTS; do
  echo "=== Experiment 17: C-arm at w_llm = $w, $REPS replicates, memory $MEMORY_PATH ==="
  WADDINGTON_FORCE_WLLM="$w" conda run --no-capture-output -n waddington-bio python3 -m waddington_select \
    --arms waddington_c ${DATASETS[@]+"${DATASETS[@]}"} \
    --replicates "$REPS" \
    --seed-mode local \
    --rounds 5 \
    --resume \
    --out "$OUT_DIR/carm_w${w}_${REPS}rep.json"
done
echo "done -> $OUT_DIR"
