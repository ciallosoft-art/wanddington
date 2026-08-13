#!/usr/bin/env bash
# Run all paper experiments in sequence.
# Long-running: the headline alone contains three 20-seed LLM-backed runs; total time is API-dependent.
# Requires Anthropic API token (see setup_auth.py).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Waddington: full paper reproduction ==="
echo "This is a long-running API reproduction (often well over 12 hours)."
echo

bash experiments/00_main_table_20seed.sh
bash experiments/01_baselines.sh
bash experiments/02_three_arm.sh
bash experiments/03_ablations.sh

echo
echo "=== All experiments complete ==="
echo "Headline results: workspace/results/router/"
ls workspace/results/router/main_table_20seed.json \
   workspace/results/router/legacy_waddington_20seed.json 2>/dev/null || true
echo "Supporting results: workspace/results/sequential/"
ls workspace/results/sequential/*.json 2>/dev/null || true
