#!/usr/bin/env bash
# Reproduce the three-arm comparison (A=Coreset vs B=LLMReasoning vs C=Waddington).
# Requires a valid Anthropic API token (see setup_auth.py).
# Runtime: ~2-3 hours on 9 datasets × 5 seeds (dominated by LLM API calls).
#
# Two C-arm configurations (see README / paper §honest-router):
#   - three_arm.json           : LEGACY target-aware routed C (avg 0.256) — feeds the legacy
#                                SHAP / decomposition / attribution tables. Pinned with LEGACY_ROUTER.
#   - clean_headline_...json    : the REPORTED leakage-free C (avg 0.251) — the plain default run.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Experiment 02: Three-arm comparison (A vs B vs C) ==="
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

# (2) Reported leakage-free C (0.251) — the plain default (no env flags). Coreset/LLM are baselines
#     unaffected by the router, so only waddington_c needs re-running here.
mkdir -p workspace/results/router
conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c \
    --seeds 5 \
    --rounds 5 \
    --out workspace/results/router/clean_headline_w0.2_5seed.json

echo
echo "Done. Legacy three-arm → workspace/results/sequential/three_arm.json"
echo "      Reported leakage-free C → workspace/results/router/clean_headline_w0.2_5seed.json"
