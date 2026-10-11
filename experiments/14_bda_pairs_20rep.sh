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
#
# Usage: bash experiments/14_bda_pairs_20rep.sh [replicates] [batch]
#   batch 32 (default) is BDA's own protocol (160 picks); 128 is the single-gene tiers' batch (640 picks),
#   run 2026-10-11 so the K562 map has BDA's agent at both budgets, as the Jurkat map does. At 128 per
#   round BDA's agent often names too few valid pairs and falls back on its own top-up path, so expect
#   ~20-30 min per replicate; the runner resumes from --out if interrupted.
set -euo pipefail
cd "$(dirname "$0")/.."
REPS="${1:-20}"
BATCH="${2:-32}"
case "$BATCH" in
  32)  OUT=workspace/results/gpt/tier_b_pairs_b32_20rep/bda_controlled_real.json ;;
  128) OUT=workspace/results/gpt/tier_b_pairs_20rep/bda_controlled_real.json ;;
  *)   echo "batch must be 32 or 128" >&2; exit 2 ;;
esac

export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs.csv"
export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_v3.csv"
export WADDINGTON_EXTRA_DATASETS="Horlbeck:$BATCH"
export WADDINGTON_LLM_BACKEND=pi WADDINGTON_PI_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_CHAT_MODEL="openai-codex/gpt-5.6-luna" WADDINGTON_PI_REASONING=none WADDINGTON_PI_STDIN=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset WADDINGTON_LEGACY_ROUTER PYTHONOPTIMIZE

conda run -n waddington-bio python3 workspace/evaluation/run_bda_controlled.py \
  --datasets Horlbeck --combinatorial --pair-pool real \
  --replicates "$REPS" --rounds 5 \
  --out "$OUT"
echo "### BDA b${BATCH} ${REPS}REP DONE $(date '+%m-%d %H:%M')"
