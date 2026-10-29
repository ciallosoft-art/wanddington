#!/usr/bin/env bash
# Experiment 24: the held-out Jurkat map in the reported C-arm's own pair encoding (no interaction features).
#
# Experiment 18 read Jurkat's pair table with the eight pair_* interaction columns, and an arm reads every such
# column a table carries (tables.with_pair_features). The K562 C-arm it is set against ran on the pair-mean table
# without them. Static rankers cannot use the columns (their training rows are degenerate single-gene pairs), but
# every model retrained on revealed Jurkat pairs does: experiment 18's online ML has 87 hits at 640 picks with them,
# 72 without. So this re-runs, on the Jurkat *_pairs_mean tables (built 2026-10-17 by
# workspace/evaluation/build_horlbeck_pairs_mean.py --screen Horlbeck_Jurkat, identical apart from the eight columns):
#   deterministic   coreset, static ranker, online ML in their original configuration (one run each)
#   carm            the reported C-arm, twenty replicates
# Everything else is experiment 18's environment: legacy task-prompt switch (experiment 08's memory, as on K562),
# the same memory file, model and switches. Experiment 18 stays the frozen analysis; nothing there is overwritten.
# The matched baselines on these tables come from analysis/matched_baselines.py (tiers jurkat_mean_b*), the B from
# experiment 23 (its padding ranking is identical on both tables), the numbers from analysis/jurkat_no_interaction.py.
#
#   bash experiments/24_jurkat_no_interaction.sh [replicates] [batch]      batch 128 or 32 (default 32)
#   ONLY=deterministic|carm runs one step
set -euo pipefail
cd "$(dirname "$0")/.."
export WADDINGTON_LEGACY_TASK_PROMPTS=1
REPS="${1:-20}"
BATCH="${2:-32}"
ONLY="${ONLY:-}"
DS=Horlbeck_Jurkat
MODEL="openai-codex/gpt-5.6-luna"
R="workspace/results/gpt/jurkat_mean_b${BATCH}_20rep"

unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY \
      WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE \
      WADDINGTON_EXCLUDE_STUDY WADDINGTON_EXCLUDE_FEATS WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD \
      WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN PYTHONOPTIMIZE
export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_jurkat_pairs_mean.csv"
export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_jurkat_pairs_mean_v3.csv"
export WADDINGTON_PAIR_DATASETS="$DS"
export WADDINGTON_EXTRA_DATASETS="$DS:${BATCH}"
export WADDINGTON_MEMORY_PATH="$PWD/workspace/results/gpt/tier_a_full_pool_20rep/experience_memory.json"
export WADDINGTON_LLM_BACKEND=pi WADDINGTON_PI_MODEL="$MODEL" WADDINGTON_CHAT_MODEL="$MODEL"
export WADDINGTON_PI_REASONING=none WADDINGTON_PI_STDIN=1 WADDINGTON_LLM_SLEEP="${WADDINGTON_LLM_SLEEP:-0}"
export WADDINGTON_FULL_SELECTED_CONTEXT=1 WADDINGTON_LLM_NAMED_ONLY=1 WADDINGTON_DUMP_SELECTIONS=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-4}"

for f in "$WADDINGTON_TRAINING_CSV" "$WADDINGTON_TRAINING_CSV_V3" "$WADDINGTON_MEMORY_PATH"; do
  [[ -f "$f" ]] || { echo "ERROR: $f missing" >&2; exit 1; }
done
mkdir -p "$R"
exec 9>"$R/.experiment.lock"
flock -n 9 || { echo "ERROR: another experiment already holds $R/.experiment.lock" >&2; exit 1; }

step () { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
runner () {   # out-file  replicates  arm...
  local out="$1" reps="$2"; shift 2
  echo "### $out  $(date '+%m-%d %H:%M')"
  conda run -n waddington-bio python3 -m waddington_select --arms "$@" \
    --datasets "$DS" --replicates "$reps" --seed-mode local --rounds 5 --resume --out "$R/$out"
}

step deterministic && runner deterministic_1rep.json 1 coreset static_ranker online_adaptive
if step carm; then
  node --input-type=module -e '
    import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
    const target = process.env.WADDINGTON_PI_MODEL;
    const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
    if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
  '
  runner "waddington_c_${REPS}rep.json" "$REPS" waddington_c
fi
echo "### JURKAT (no interaction features) b${BATCH} DONE $(date '+%m-%d %H:%M')"
