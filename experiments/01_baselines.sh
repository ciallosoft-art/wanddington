#!/usr/bin/env bash
# Reproduce Table 2 baseline rows (Random, LOO-LightGBM, OnlineAdaptive).
# These arms do NOT require an API token (no LLM calls).
# Runtime: ~5-10 min on a modern CPU.
#
# Output: workspace/results/sequential/baselines.json
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Experiment 01: Baselines (no LLM) ==="
echo "Arms: random  static_ranker  online_adaptive  coreset"
echo "Seeds: 5  |  Rounds: 5  |  Datasets: all 9"
echo

conda run -n waddington-bio python3 workspace/agent/run_sequential.py \
    --arms random coreset static_ranker online_adaptive \
    --seeds 5 \
    --rounds 5 \
    --out workspace/results/sequential/baselines.json

echo
echo "Done. Results saved to workspace/results/sequential/baselines.json"
