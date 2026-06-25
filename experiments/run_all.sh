#!/usr/bin/env bash
# Run all paper experiments in sequence.
# Total runtime: ~8-12 hours.
# Requires Anthropic API token (see setup_auth.py).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Waddington: full paper reproduction ==="
echo "Estimated time: 8-12 hours"
echo

bash experiments/01_baselines.sh
bash experiments/02_three_arm.sh
bash experiments/03_ablations.sh

echo
echo "=== All experiments complete ==="
echo "Results directory: workspace/results/sequential/"
ls workspace/results/sequential/*.json 2>/dev/null || true
