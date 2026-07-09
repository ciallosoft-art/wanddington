#!/usr/bin/env bash
# Reproduce Table 2 main result (A=Coreset vs B=LLMReasoning vs C=Waddington).
# Requires a valid Anthropic API token (see setup_auth.py).
# Runtime: ~2-3 hours on 9 datasets × 5 seeds (dominated by LLM API calls).
#
# Output: workspace/results/sequential/three_arm.json
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

conda run -n waddington-bio python3 -m waddington_select \
    --arms coreset llm_reasoning waddington_c \
    --seeds 5 \
    --rounds 5 \
    --out workspace/results/sequential/three_arm.json

echo
echo "Done. Results saved to workspace/results/sequential/three_arm.json"
