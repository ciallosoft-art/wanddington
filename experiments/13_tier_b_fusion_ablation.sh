#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# Attribute the combinatorial-tier fusion repair to its two independent parts.
#
# The reported C-arm scores Score(g) = 0.80*ml(g) + 0.20*1[g in named]. On Horlbeck that is the wrong
# control law -- the ML branch is near chance, so a 0.20 bonus cannot lift a pair it scores low, and
# only ~24% of each batch ends up being what the LLM actually named, even though LLM-named picks hit
# at 0.329 against 0.162 for ML-driven ones. WaddingtonCPairArm changes only the combination:
#
#   named_first   take the LLM's names first (ordered among themselves by the fill score), ML fills
#                 the remainder -- the same named-only vote expressed as an ordering, not a weight
#   prior_fill    rank the filler by the label-free interaction prior (equal-weight rank-sum of the
#                 pair_* columns) blended 50/50 with the online ML score, instead of ML alone
#
# Run separately so the gain can be attributed rather than asserted. Neither knob is tuned on
# Horlbeck: the 50/50 blend is fixed, not searched.
#
# Usage: bash experiments/13_tier_b_fusion_ablation.sh [replicates]
set -uo pipefail
cd "$(dirname "$0")/.."
REPS="${1:-20}"
# Batch size. 128 matches the single-gene tiers and everything else in this directory; pass 32 to
# reproduce BioDiscoveryAgent's own Table 2 protocol (160 picks), the only budget at which our numbers
# and theirs are comparable.
BATCH="${2:-128}"
R=workspace/results/gpt/tier_b_pairs_interaction_20rep
[[ "$BATCH" != "128" ]] && R="workspace/results/gpt/tier_b_pairs_interaction_b${BATCH}_20rep"

export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs.csv"
export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_v3.csv"
export WADDINGTON_EXTRA_DATASETS="Horlbeck:${BATCH}"
export WADDINGTON_MEMORY_PATH="$PWD/workspace/results/gpt/tier_a_full_pool_20rep/experience_memory.json"
export WADDINGTON_LLM_BACKEND=pi WADDINGTON_PI_MODEL=openai-codex/gpt-5.6-luna
export WADDINGTON_CHAT_MODEL=openai-codex/gpt-5.6-luna WADDINGTON_PI_REASONING=none WADDINGTON_PI_STDIN=1
export WADDINGTON_FULL_SELECTED_CONTEXT=1 WADDINGTON_LLM_NAMED_ONLY=1 WADDINGTON_DUMP_SELECTIONS=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-4}"
mkdir -p "$R"

run () {   # label  named_first  prior_fill  [cold_start] [complex_first] [explore_frac]
  echo "### $1  (nf=$2 pf=$3 cs=${4:-0} cx=${5:-0} ex=${6:-0})  $(date '+%m-%d %H:%M')"
  env WADDINGTON_PAIR_NAMED_FIRST="$2" WADDINGTON_PAIR_PRIOR_FILL="$3" \
      WADDINGTON_PAIR_COLD_START="${4:-0}" WADDINGTON_PAIR_COMPLEX_FIRST="${5:-0}" \
      WADDINGTON_PAIR_EXPLORE_FRAC="${6:-0}" \
    conda run -n waddington-bio python3 -m waddington_select --arms waddington_c_pair \
      --datasets Horlbeck --replicates "$REPS" --seed-mode local --rounds 5 --resume \
      --out "$R/pair_$1.json"
}
run namedfirst_only  1 0
run namedfirst_prior 1 1
# Round 1 on the interaction prior alone; named-first from round 2. The per-round breakdown showed
# round 1 is where the LLM is weakest (22.9% vs the prior's own 28.9% over the same 128 slots).
run coldstart_prior  1 1 1
# Curated-complex pairs ahead of everything, including the LLM: 34.4% per pick vs its 32.9%, and the
# source is fixed in advance so it cannot run dry mid-campaign the way LLM proposals do.
run complex_first    1 1 1 1
# Matched control for complex_first. Adding pair_complex_shared to the table changed the runtime
# fingerprint, so the three configs above could not be resumed and remain on the 7-feature table --
# complex_first would otherwise differ from them by two things at once (the extra column in the
# feature table AND the ordering rule). This rerun isolates the ordering.
run coldstart_prior_v8 1 1 1 0
# Explore/exploit. Selection is heavily redundant: round 1 touches 47 of 449 genes, five rounds reach
# 61%, the most-sampled gene appears 31 times against 2.9 under uniform sampling -- so the online model
# trains on a narrow slice, and its learning curve is far from saturated. Spend part of each batch on
# the best pair containing the least-sampled gene and see whether better rounds 3-5 repay round 1.
run explore_12pct    1 1 1 0 0.125
run explore_25pct    1 1 1 0 0.25
echo "### ABLATION DONE $(date '+%m-%d %H:%M')"
