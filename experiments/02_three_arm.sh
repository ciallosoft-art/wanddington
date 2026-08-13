#!/usr/bin/env bash
# Reproduce the supporting 5-seed three-arm comparison
# (A=Coreset vs B=LLMReasoning vs C=Waddington).
# Requires a valid Anthropic API token (see setup_auth.py).
# Runtime: ~2-3 hours on 9 datasets × 5 seeds (dominated by LLM API calls).
#
# Two C-arm configurations (see README / paper §honest-router):
#   - three_arm.json           : LEGACY target-aware routed C (avg 0.256) — feeds the legacy
#                                SHAP / decomposition / attribution tables. Pinned with LEGACY_ROUTER.
#   - clean_headline_...json    : 5-seed companion for the reported leakage-free C (avg 0.251).
# The reported 20-seed main table (avg 0.253) is produced by 00_main_table_20seed.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
unset WADDINGTON_LEGACY_ROUTER WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE
unset WADDINGTON_FEATURE_POLICY WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY
unset WADDINGTON_REASON_FROM_STRUCTURE WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
export WADDINGTON_LLM_BACKEND=anthropic

echo "=== Experiment 02: Supporting five-seed three-arm comparison ==="
echo "Arms: coreset  llm_reasoning  waddington_c"
echo "Seeds: 5  |  Rounds: 5  |  Datasets: all 9"
echo

# Check token
python3 experiments/setup_auth.py --check || {
    echo "ERROR: API token expired or missing. Run: python3 experiments/setup_auth.py --token sk-ant-..."
    exit 1
}

# (1) Legacy target-aware routed C (0.256) — source of the legacy SHAP/decomposition tables.
WADDINGTON_LEGACY_ROUTER=1 conda run -n waddington-bio python3 -m waddington_select \
    --arms coreset llm_reasoning waddington_c \
    --seeds 5 \
    --rounds 5 \
    --out workspace/results/sequential/three_arm.json

# (2) Five-seed companion for the reported leakage-free C (0.251) — the plain default (no env flags).
#     Coreset/LLM are unaffected by the router, so only waddington_c needs re-running here.
mkdir -p workspace/results/router
conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c \
    --seeds 5 \
    --rounds 5 \
    --out workspace/results/router/clean_headline_w0.2_5seed.json

echo
echo "Done. Legacy three-arm → workspace/results/sequential/three_arm.json"
echo "      Five-seed leakage-free companion → workspace/results/router/clean_headline_w0.2_5seed.json"
