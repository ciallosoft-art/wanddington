#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# BioDiscoveryAgent's own agent, run on our benchmark and scored by our oracle, across BDA's tiers.
#
# The headline tier's controlled run already exists (workspace/results/bda_controlled/bda_gpt.json,
# 0.146). This adds the other two tiers under the same protocol, so the external comparison is
# available wherever our own arms are reported.
#
# Agent logic is untouched. Three adaptations, each documented at its patch site in the BDA repo:
#   LLM.py:complete_text     every completion routed through our LLMClient at a fixed model
#   tools.py:parse_entries   tolerate a missing literal "Solution:" header on top-up re-prompts
#   tools.py:agent_loop      credit a round the agent could not fill instead of discarding it
# plus workspace/evaluation/_bda_stubs/ to satisfy tool imports that the no-tools setting never calls.
#
# Two-gene runs are done TWICE. BDA's shipped pair prompt hardcodes a 124-gene Schmidt T-cell list,
# but Horlbeck's library is a different 449 genes (overlap 8 => at most 28 of 100,576 pairs
# proposable, 3 of them hits), and its own reported Table 2 reaches 45.1 hits — so the released prompt
# cannot be what produced their numbers. `real` substitutes the actual library and measures the
# method; `shipped` measures the released artifact. Report both.
#
# Usage: bash experiments/11_tiers_bda.sh [tier_a|tier_b|both] [replicates]
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:-both}"
REPS="${2:-3}"

export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_CHAT_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset WADDINGTON_LEGACY_ROUTER PYTHONOPTIMIZE

TIER_A_V1="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv"
TIER_A_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv"
PAIR_V1="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs.csv"
PAIR_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_v3.csv"

if [[ "$WHICH" == "tier_a" || "$WHICH" == "both" ]]; then
  echo "=== BDA on the 1-gene FULL pool tier ($REPS replicates) ==="
  env WADDINGTON_TRAINING_CSV="$TIER_A_V1" WADDINGTON_TRAINING_CSV_V3="$TIER_A_V3" \
    conda run -n waddington-bio python3 workspace/evaluation/run_bda_controlled.py \
      --datasets IFNG IL2 Sanchez21 Sanchez21_down Carnevale22 Scharenberg22 Steinhart \
                 Replogle_K562_essential Replogle_K562_gwps \
      --replicates "$REPS" --rounds 5 \
      --out workspace/results/gpt/tier_a_full_pool_20rep/bda_controlled.json
fi

if [[ "$WHICH" == "tier_b" || "$WHICH" == "both" ]]; then
  # The `shipped` control is ~30x slower per round than `real` (26 min vs 50 s, measured twice): the
  # released gene list leaves the agent unable to propose in-pool pairs, so every round grinds through
  # retries and the 101-call summarize_remaining_genes fallback. Its finding is qualitative -- can it
  # finish, and at what cost -- so one complete replicate is enough; spending 3x the quota on variance
  # buys nothing. Override with BDA_SHIPPED_REPS if you want more.
  for pool in real shipped; do
    pool_reps="$REPS"
    [[ "$pool" == "shipped" ]] && pool_reps="${BDA_SHIPPED_REPS:-1}"
    echo "=== BDA on the 2-gene tier, pair library = $pool ($pool_reps replicates, 32 pairs/round) ==="
    env WADDINGTON_TRAINING_CSV="$PAIR_V1" WADDINGTON_TRAINING_CSV_V3="$PAIR_V3" \
        WADDINGTON_EXTRA_DATASETS="Horlbeck:32" \
      conda run -n waddington-bio python3 workspace/evaluation/run_bda_controlled.py \
        --datasets Horlbeck --combinatorial --pair-pool "$pool" \
        --replicates "$pool_reps" --rounds 5 \
        --out "workspace/results/gpt/tier_b_pairs_b32_20rep/bda_controlled_${pool}.json"
  done
fi
echo "Done."
