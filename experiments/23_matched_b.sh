#!/usr/bin/env bash
# Experiment 23: the padded LLM baseline B with its padding ranker on the C-arm's own features.
#
# B (llm_reasoning) pads the slots its LLM does not fill from a static LOO ranking. In every reported run that
# ranking read the original benchmark configuration (six gene-intrinsic + three anchor features, no DepMap), so
# C minus B mixed "who leads, the LLM or the ML" with a feature difference (anchors in, DepMap out). The ML
# baselines were moved to C's configuration on 2026-10-23 (analysis/matched_baselines.py); this re-runs B the
# same way: arm llm_reasoning_matched = B with the padding ranker on waddington_c_arm._get_feature_config
# (no anchor, DepMap by design metadata). The LLM side is unchanged, but the LLM sees the outcomes of the padded
# genes from round 2 on, so the whole arm is re-run rather than re-scored.
#
# Each tier reproduces the environment of the run that produced the reported B (experiments 05, 08, 09, 18),
# changing only the arm. Results go to a new directory; nothing existing is overwritten.
#
#   bash experiments/23_matched_b.sh <tier> [replicates]
#   tier: headline | full_pool | pairs128 | pairs32 | jurkat128 | jurkat32 | all
set -euo pipefail
cd "$(dirname "$0")/.."

TIER="${1:?tier: headline | full_pool | pairs128 | pairs32 | jurkat128 | jurkat32 | all}"
REPS="${2:-20}"
MODEL="openai-codex/gpt-5.6-luna"
EVAL="$PWD/workspace/evaluation"
ROOT="${WADDINGTON_MATCHED_B_DIR:-workspace/results/gpt/matched_b}"

if [[ "$TIER" == "all" ]]; then
  for t in pairs32 jurkat32 jurkat128 pairs128 headline full_pool; do
    bash "$0" "$t" "$REPS"
  done
  exit 0
fi

unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY WADDINGTON_DROP_ANCHOR_FEATS
unset WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE WADDINGTON_EXCLUDE_STUDY WADDINGTON_EXCLUDE_FEATS
unset WADDINGTON_TRAINING_CSV WADDINGTON_TRAINING_CSV_V3 WADDINGTON_EXTRA_DATASETS WADDINGTON_PAIR_DATASETS
unset PYTHONOPTIMIZE

DATASETS=()
case "$TIER" in
  headline)    # experiment 05 (reported run replogle_transcriptomics_v2_20rep)
    export WADDINGTON_LEGACY_TASK_PROMPTS=0
    PAR=6 ;;
  full_pool)   # experiment 08
    export WADDINGTON_LEGACY_TASK_PROMPTS=0
    export WADDINGTON_TRAINING_CSV="$EVAL/lgbm_training_data_tier_a.csv"
    export WADDINGTON_TRAINING_CSV_V3="$EVAL/lgbm_training_data_tier_a_v3.csv"
    PAR=6 ;;
  pairs128|pairs32)   # experiment 09; the *_pairs_mean tables equal the ones it read on every column B uses
    export WADDINGTON_LEGACY_TASK_PROMPTS=0
    export WADDINGTON_TRAINING_CSV="$EVAL/lgbm_training_data_horlbeck_pairs_mean.csv"
    export WADDINGTON_TRAINING_CSV_V3="$EVAL/lgbm_training_data_horlbeck_pairs_mean_v3.csv"
    export WADDINGTON_EXTRA_DATASETS="Horlbeck:${TIER#pairs}"
    DATASETS=(--datasets Horlbeck)
    PAR=4 ;;
  jurkat128|jurkat32)  # experiment 18 (frozen protocol, legacy task-prompt wording)
    export WADDINGTON_LEGACY_TASK_PROMPTS=1
    export WADDINGTON_TRAINING_CSV="$EVAL/lgbm_training_data_horlbeck_jurkat_pairs.csv"
    export WADDINGTON_TRAINING_CSV_V3="$EVAL/lgbm_training_data_horlbeck_jurkat_pairs_v3.csv"
    export WADDINGTON_PAIR_DATASETS=Horlbeck_Jurkat
    export WADDINGTON_EXTRA_DATASETS="Horlbeck_Jurkat:${TIER#jurkat}"
    DATASETS=(--datasets Horlbeck_Jurkat)
    PAR=4 ;;
  *) echo "unknown tier $TIER" >&2; exit 2 ;;
esac

RESULT_DIR="$ROOT/$TIER"
OUT="$RESULT_DIR/llm_reasoning_matched_${REPS}rep.json"
export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="$MODEL"
export WADDINGTON_CHAT_MODEL="$MODEL"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export WADDINGTON_LLM_SLEEP="${WADDINGTON_LLM_SLEEP:-0}"
export WADDINGTON_FULL_SELECTED_CONTEXT=1
export WADDINGTON_LLM_NAMED_ONLY=1
export WADDINGTON_DUMP_SELECTIONS=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-$PAR}"
export WADDINGTON_GPT_RESULT_DIR="$RESULT_DIR"

mkdir -p "$RESULT_DIR"
exec 9>"$RESULT_DIR/.experiment.lock"
if ! flock -n 9; then
  echo "ERROR: another experiment already holds $RESULT_DIR/.experiment.lock" >&2
  exit 1
fi

node --input-type=module -e '
  import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
  const target = process.env.WADDINGTON_PI_MODEL;
  const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
  if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
'

echo "=== Experiment 23: matched B, tier $TIER, $REPS replicates ($(date '+%m-%d %H:%M')) ==="
conda run -n waddington-bio python3 -m waddington_select \
  --arms llm_reasoning_matched "${DATASETS[@]}" \
  --replicates "$REPS" --seed-mode local --rounds 5 --resume \
  --out "$OUT"
echo "### matched B $TIER DONE $(date '+%m-%d %H:%M') -> $OUT"
