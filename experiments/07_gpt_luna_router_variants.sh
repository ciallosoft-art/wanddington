#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# GPT-5.6 Luna honest-router variant base runs (three evaluation replicates each).
#
# Produces the four base C-arm runs that analysis/router_analysis.py needs to derive the honest
# routing table (global_fixed / nested_router / study-out) on GPT: the global fixed-weight runs at
# w_llm in {0.2,0.3,0.4} WITH the anchor-relative features kept, plus the strict study-out-prior run
# at w=0.2. All are leakage-free EXCEPT that anchor features are kept here (that is the point — the
# "with anchor" rows); the reported no-anchor Final Waddington is the honest GPT main C (0.260).
# Reuses the strictly verified GPT memory from 05; runs in parallel with the ablation job via its own
# result dir and lock.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="openai-codex/gpt-5.6-luna"
SRC_DIR="workspace/results/gpt/gpt_5_6_luna_none_20rep"
MEMORY_PATH="$SRC_DIR/experience_memory.json"
OUT_DIR="workspace/results/gpt/router"

if [[ ! -f "$MEMORY_PATH" ]]; then
  echo "ERROR: GPT memory $MEMORY_PATH not found; run 05_gpt_luna_20repeat.sh first" >&2
  exit 1
fi
mkdir -p "$OUT_DIR"

# Honest protocol (no legacy router, metadata features, named-only vote), GPT backend, GPT memory.
unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY WADDINGTON_HITFREQ_ONLY
unset WADDINGTON_REASON_FROM_STRUCTURE WADDINGTON_TRAINING_CSV PYTHONOPTIMIZE
export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="$MODEL"
export WADDINGTON_CHAT_MODEL="$MODEL"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export WADDINGTON_LLM_SLEEP="${WADDINGTON_LLM_SLEEP:-0}"
export WADDINGTON_FULL_SELECTED_CONTEXT=1
export WADDINGTON_LLM_NAMED_ONLY=1
export WADDINGTON_DROP_ANCHOR_FEATS=0     # keep anchor-relative features (the "with anchor" variants)
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Repeat-level parallelism (default 6; export WADDINGTON_PARALLEL=1 to force sequential).
# N is not part of the provenance manifest and does not change results.
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-6}"
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"

exec 8>"$OUT_DIR/.router.lock"
if ! flock -n 8; then
  echo "ERROR: another GPT router job already holds $OUT_DIR/.router.lock" >&2
  exit 1
fi

node --input-type=module -e '
  import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
  const target = process.env.WADDINGTON_PI_MODEL;
  const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
  if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
  console.log(`Authorized model: ${target}`);
'

echo "=== Experiment 07: GPT-5.6 Luna honest-router variant base runs ==="
echo "Reusing GPT memory: $MEMORY_PATH"

run_cfg() {
  local out="$1"; shift
  echo "--- $out  ($*) ---"
  env "$@" conda run -n waddington-bio python3 -m waddington_select \
    --arms waddington_c \
    --replicates 3 \
    --seed-mode local \
    --rounds 5 \
    --resume \
    --out "$OUT_DIR/$out"
}

# Global fixed-weight sweep (with anchor). WADDINGTON_EXCLUDE_STUDY explicitly cleared per run.
run_cfg global_w0.2.json           WADDINGTON_FORCE_WLLM=0.2 WADDINGTON_EXCLUDE_STUDY=0
run_cfg global_w0.3.json           WADDINGTON_FORCE_WLLM=0.3 WADDINGTON_EXCLUDE_STUDY=0
run_cfg global_w0.4.json           WADDINGTON_FORCE_WLLM=0.4 WADDINGTON_EXCLUDE_STUDY=0
# Strict study-out prior at w=0.2 (same-study siblings removed from the LOO prior).
run_cfg study_out_prior_w0.2.json  WADDINGTON_FORCE_WLLM=0.2 WADDINGTON_EXCLUDE_STUDY=1

echo
echo "Done. GPT router variants in: $OUT_DIR"
