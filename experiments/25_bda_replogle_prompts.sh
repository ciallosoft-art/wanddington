#!/usr/bin/env bash
# Experiment 25: BioDiscoveryAgent on the two Replogle screens with the corrected task prompts.
#
# BDA ships no task prompt for the two Replogle Perturb-seq screens, so run_bda_controlled.py writes ours into
# its datasets/task_prompts/. It wrote them once and never refreshed them: the controlled BDA runs of 2026-08-30
# (headline tier, workspace/results/bda_controlled/bda_gpt.json) and 2026-09-11 (full pool,
# gpt/tier_a_full_pool_20rep/bda_controlled.json) read the fitness / survival wording that the 2026-09-18
# correction replaced for our own arms (docs/fix_impact_map.md). The runner now always writes the prompt our
# LLM arms read (llm_reasoning_arm._load_task) and records it in the manifest. This re-runs only those two screens
# on both single-gene tiers, one replicate each as before, with everything else as in experiment 11. The original
# runs are kept; analysis/bda_replogle_v2.py merges the two new screens into copies of them.
#
#   bash experiments/25_bda_replogle_prompts.sh headline|full_pool
set -euo pipefail
cd "$(dirname "$0")/.."
TIER="${1:?tier: headline | full_pool}"
export WADDINGTON_LEGACY_TASK_PROMPTS=0
export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_CHAT_MODEL="openai-codex/gpt-5.6-luna"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset WADDINGTON_LEGACY_ROUTER PYTHONOPTIMIZE WADDINGTON_TRAINING_CSV WADDINGTON_TRAINING_CSV_V3 \
      WADDINGTON_EXTRA_DATASETS WADDINGTON_PAIR_DATASETS

case "$TIER" in
  headline) ;;
  full_pool)
    export WADDINGTON_TRAINING_CSV="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv"
    export WADDINGTON_TRAINING_CSV_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv" ;;
  *) echo "unknown tier $TIER" >&2; exit 2 ;;
esac

R="workspace/results/gpt/bda_replogle_v2"
mkdir -p "$R"
exec 9>"$R/.experiment_${TIER}.lock"
flock -n 9 || { echo "ERROR: another run holds $R/.experiment_${TIER}.lock" >&2; exit 1; }

echo "=== Experiment 25: BDA, Replogle screens, corrected prompts, tier $TIER ($(date '+%m-%d %H:%M')) ==="
conda run -n waddington-bio python3 workspace/evaluation/run_bda_controlled.py \
  --datasets Replogle_K562_essential Replogle_K562_gwps --replicates 1 --rounds 5 \
  --out "$R/${TIER}.json"
echo "### BDA Replogle $TIER DONE $(date '+%m-%d %H:%M')"
