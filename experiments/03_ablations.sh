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
# code path is now the reported 20-seed leakage-free config (0.253), so we pin the legacy router here to
# keep the committed 5-seed ablation_*.json reproducible. (The final system and its paired statistics use
# workspace/results/router/main_table_20seed.json; see README.)
set -euo pipefail
cd "$(dirname "$0")/.."
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE
unset WADDINGTON_FEATURE_POLICY WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY
unset WADDINGTON_REASON_FROM_STRUCTURE WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
unset WADDINGTON_PI_MODEL WADDINGTON_PI_REASONING WADDINGTON_PI_CMD
export WADDINGTON_LEGACY_ROUTER=1
export WADDINGTON_LLM_BACKEND=anthropic
export WADDINGTON_LLM_MODEL=claude-haiku-4-5-20251001
export WADDINGTON_FULL_SELECTED_CONTEXT=0
export WADDINGTON_LLM_NAMED_ONLY=0
export WADDINGTON_MEMORY_PATH=workspace/results/sequential/experience_memory.json
export WADDINGTON_SKILL_LIBRARY_PATH=workspace/results/sequential/skill_library.json

echo "=== Experiment 03: Ablation study ==="
echo

python3 experiments/setup_auth.py --check || {
    echo "ERROR: API token expired or missing. Run: python3 experiments/setup_auth.py --token sk-ant-..."
    exit 1
}

resume_args=()
if [[ "${WADDINGTON_RESUME:-0}" == "1" ]]; then
    resume_args+=(--resume)
fi

run_ablation() {
    local label="$1"; local arm="$2"; local out="$3"
    echo "--- Ablation: $label ---"
    conda run -n waddington-bio python3 -m waddington_select \
        --arms waddington_c "$arm" \
        --seeds 5 \
        --rounds 5 \
        "${resume_args[@]}" \
        --out "workspace/results/sequential/$out"
    echo "  Saved: workspace/results/sequential/$out"
    echo
}

run_ablation "C - memory"         waddington_c_no_memory   ablation_memory.json
run_ablation "C - LLM"            waddington_c_no_llm      ablation_llm.json
run_ablation "C - ML (retrain)"   waddington_c_no_ml       ablation_ml.json
run_ablation "C - shuffled names" waddington_c_shuffled_names ablation_shuffled.json

echo "=== All ablations complete ==="
echo "Results in workspace/results/sequential/ablation_*.json"
