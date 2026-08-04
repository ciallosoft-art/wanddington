#!/usr/bin/env bash
# Reproduce Table 4: all four ablation experiments.
# Each ablation is run paired against the full C arm (waddington_c)
# so results are directly comparable within each run.
# Requires a valid Anthropic API token (see setup_auth.py).
# Runtime: ~6-8 hours total (4 × ~1.5h each).
#
# Outputs:
#   workspace/results/sequential/ablation_memory.json
#   workspace/results/sequential/ablation_llm.json
#   workspace/results/sequential/ablation_ml.json
#   workspace/results/sequential/ablation_shuffled.json
#
# These ablation tables were computed on the LEGACY target-aware routed C-arm (avg 0.256). The default
# code path is now the reported leakage-free config (0.251), so we pin the legacy router here to keep the
# committed ablation_*.json reproducible. (The final leakage-free system's paired stats are in
# workspace/results/router/clean_headline_w0.2_5seed.json — the plain default run; see README.)
set -euo pipefail
cd "$(dirname "$0")/.."
export WADDINGTON_LEGACY_ROUTER=1

echo "=== Experiment 03: Ablation study ==="
echo

python3 experiments/setup_auth.py --check || {
    echo "ERROR: API token expired or missing. Run: python3 experiments/setup_auth.py --token sk-ant-..."
    exit 1
}

run_ablation() {
    local label="$1"; local arm="$2"; local out="$3"
    echo "--- Ablation: $label ---"
    conda run -n waddington-bio python3 -m waddington_select \
        --arms waddington_c "$arm" \
        --seeds 5 \
        --rounds 5 \
        --resume \
        --out "workspace/results/sequential/$out"
    echo "  Saved: workspace/results/sequential/$out"
    echo
    # Refresh token between long runs to avoid expiry
    python3 experiments/setup_auth.py --check 2>/dev/null || true
}

run_ablation "C - memory"         waddington_c_no_memory   ablation_memory.json
run_ablation "C - LLM"            waddington_c_no_llm      ablation_llm.json
run_ablation "C - ML (retrain)"   waddington_c_no_ml       ablation_ml.json
run_ablation "C - shuffled names" waddington_c_shuffled_names ablation_shuffled.json

echo "=== All ablations complete ==="
echo "Results in workspace/results/sequential/ablation_*.json"
