#!/usr/bin/env bash
# The K562 combinatorial runs this test is frozen against read experiment 08's cross-experiment memory
# (four most relevant entries per prompt). That memory predates the task-prompt protocol field, and
# today's code accepts it only under the legacy protocol switch, which for these pair screens changes
# nothing else (it only swaps the two Replogle prompts). So the switch is on: same memory as K562.
export WADDINGTON_LEGACY_TASK_PROMPTS=1
# Held-out test of the combinatorial repair on Horlbeck et al. 2018's Jurkat map (docs/jurkat_protocol.md).
#
# Everything is frozen at the K562 settings: the same arms and switches, the 0.2 verifier weight, the
# 50/50 fill, the cold start, the eight interaction features, the same cross-experiment memory, and a
# training half of the nine single-gene screens only (no K562 pair labels). Nothing is tuned on Jurkat.
#
# Usage: bash experiments/18_jurkat_heldout.sh [replicates] [batch]
#   batch 32 (default) is BDA's own protocol and the primary budget (160 picks); 128 the secondary one.
#   ONLY=<step> runs one step: deterministic random llm carm no_llm names_first verifier bda
#
# Prerequisites, once:
#   python3 workspace/evaluation/build_horlbeck_jurkat.py
#   python3 workspace/evaluation/build_pair_interaction_features.py \
#       --pairs-csv workspace/data/horlbeck2018/pairs_Horlbeck_Jurkat.csv \
#       --out workspace/evaluation/pair_interaction_features_horlbeck_jurkat.csv --tag horlbeck_jurkat
#   python3 workspace/evaluation/build_horlbeck_pairs.py --screen Horlbeck_Jurkat
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${1:-20}"
BATCH="${2:-32}"
ONLY="${ONLY:-}"
DS=Horlbeck_Jurkat
R="workspace/results/gpt/jurkat_b${BATCH}_20rep"

unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY \
      WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE \
      WADDINGTON_EXCLUDE_STUDY PYTHONOPTIMIZE
export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_jurkat_pairs.csv"
export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_jurkat_pairs_v3.csv"
export WADDINGTON_PAIR_DATASETS="$DS"
export WADDINGTON_EXTRA_DATASETS="$DS:${BATCH}"
# The same cross-experiment memory as every K562 combinatorial run (experiments 09, 12, 13).
export WADDINGTON_MEMORY_PATH="$PWD/workspace/results/gpt/tier_a_full_pool_20rep/experience_memory.json"
export WADDINGTON_LLM_BACKEND=pi WADDINGTON_PI_MODEL=openai-codex/gpt-5.6-luna
export WADDINGTON_CHAT_MODEL=openai-codex/gpt-5.6-luna WADDINGTON_PI_REASONING=none WADDINGTON_PI_STDIN=1
export WADDINGTON_FULL_SELECTED_CONTEXT=1 WADDINGTON_LLM_NAMED_ONLY=1 WADDINGTON_DUMP_SELECTIONS=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-4}"

for f in "$WADDINGTON_TRAINING_CSV" "$WADDINGTON_TRAINING_CSV_V3"; do
  [[ -f "$f" ]] || { echo "ERROR: $f missing -- run the prerequisites above" >&2; exit 1; }
done
mkdir -p "$R"

step () { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
runner () {   # out-file  replicates  arm...
  local out="$1" reps="$2"; shift 2
  echo "### $out  $(date '+%m-%d %H:%M')"
  conda run -n waddington-bio python3 -m waddington_select --arms "$@" \
    --datasets "$DS" --replicates "$reps" --seed-mode local --rounds 5 --resume --out "$R/$out"
}
pair_env () {  # named_first prior_fill cold_start -- complex_first and exploration off, as in K562's final repair
  export WADDINGTON_PAIR_NAMED_FIRST="$1" WADDINGTON_PAIR_PRIOR_FILL="$2" WADDINGTON_PAIR_COLD_START="$3" \
         WADDINGTON_PAIR_COMPLEX_FIRST=0 WADDINGTON_PAIR_EXPLORE_FRAC=0
}

step deterministic && runner deterministic_1rep.json 1 coreset static_ranker online_adaptive
step random        && runner "random_${REPS}rep.json" "$REPS" random
step llm           && runner "llm_reasoning_${REPS}rep.json" "$REPS" llm_reasoning
step carm          && runner "waddington_c_${REPS}rep.json" "$REPS" waddington_c
step no_llm        && ( pair_env 1 1 1; runner pair_no_llm_1rep.json 1 waddington_c_pair_no_llm )
step names_first   && ( pair_env 1 1 1; runner "pair_names_first_${REPS}rep.json" "$REPS" waddington_c_pair )
step verifier      && ( pair_env 0 1 1; runner "pair_verifier_${REPS}rep.json" "$REPS" waddington_c_pair )
if step bda; then
  echo "### bda_${REPS}rep.json  $(date '+%m-%d %H:%M')"
  conda run -n waddington-bio python3 workspace/evaluation/run_bda_controlled.py \
    --datasets "$DS" --combinatorial --pair-pool real --replicates "$REPS" --rounds 5 \
    --out "$R/bda_${REPS}rep.json"
fi
echo "### JURKAT b${BATCH} DONE $(date '+%m-%d %H:%M')"
