#!/usr/bin/env bash
# Reproduce the reported 20-seed headline comparison and its matched legacy-router control.
# Requires a valid Anthropic API token (see setup_auth.py). This is a long-running API experiment.
#
# Outputs:
#   workspace/results/router/main_table_20seed.json
#   workspace/results/router/legacy_waddington_20seed.json
#
# By default the frozen outputs are recomputed from scratch. Set WADDINGTON_RESUME=1 only when
# continuing an interrupted rerun whose output files already contain partial checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
# Prevent an interactive/development shell from silently changing the paper configuration.
unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE
unset WADDINGTON_FEATURE_POLICY WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY
unset WADDINGTON_REASON_FROM_STRUCTURE WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
export WADDINGTON_LLM_BACKEND=anthropic

echo "=== Experiment 00: Reported 20-seed main table ==="
echo "Arms: random  coreset  static_ranker  online_adaptive  llm_reasoning  waddington_c"
echo "Seeds: 20  |  Rounds: 5  |  Datasets: all 9"
echo

python3 experiments/setup_auth.py --check || {
    echo "ERROR: API token expired or missing. Run: python3 experiments/setup_auth.py --token sk-ant-..."
    exit 1
}

resume_args=()
if [[ "${WADDINGTON_RESUME:-0}" == "1" ]]; then
    resume_args+=(--resume)
fi

mkdir -p workspace/results/router

# (1) Reported leakage-free default: all six main-table methods, matched at 20 seeds.
conda run -n waddington-bio python3 -m waddington_select \
    --arms random coreset static_ranker online_adaptive llm_reasoning waddington_c \
    --seeds 20 \
    --rounds 5 \
    "${resume_args[@]}" \
    --out workspace/results/router/main_table_20seed.json

# (2) Matched 20-seed diagnostic control: the old target-aware routed C-arm.
WADDINGTON_LEGACY_ROUTER=1 conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c \
    --seeds 20 \
    --rounds 5 \
    "${resume_args[@]}" \
    --out workspace/results/router/legacy_waddington_20seed.json

echo
echo "Done. Reported main table → workspace/results/router/main_table_20seed.json"
echo "      Matched legacy control → workspace/results/router/legacy_waddington_20seed.json"
