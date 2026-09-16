#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# Raise the BioDiscoveryAgent two-gene baseline from three replicates to twenty.
#
# Three was a cost decision when each run was expected to be slow; with the real gene library injected
# a campaign takes ~250 s, so the precision was cheap to buy and we were comparing our 20-replicate
# arms against a baseline whose 95% interval was [33.7, 71.6] -- wide enough to swallow every method in
# the table. The p = 0.47 against our repaired arm was mostly that imprecision, not closeness.
# BDA's own Table 2 uses ten runs.
#
# `real` only: the shipped-library control is a separate, ~30x slower question and is not repeated.
set -euo pipefail
cd "$(dirname "$0")/.."
REPS="${1:-20}"

export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs.csv"
export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_v3.csv"
export WADDINGTON_EXTRA_DATASETS="Horlbeck:32"
export WADDINGTON_LLM_BACKEND=pi WADDINGTON_PI_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_CHAT_MODEL="openai-codex/gpt-5.6-luna" WADDINGTON_PI_REASONING=none WADDINGTON_PI_STDIN=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset WADDINGTON_LEGACY_ROUTER PYTHONOPTIMIZE

conda run -n waddington-bio python3 workspace/evaluation/run_bda_controlled.py \
  --datasets Horlbeck --combinatorial --pair-pool real \
  --replicates "$REPS" --rounds 5 \
  --out workspace/results/gpt/tier_b_pairs_b32_20rep/bda_controlled_real.json
echo "### BDA 20REP DONE $(date '+%m-%d %H:%M')"
