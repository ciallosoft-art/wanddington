#!/usr/bin/env bash
# Experiment 22 — the reported GPT C-arm without hub_score_norm and ppi_score_sum (2026-10-22).
#
# Both columns are counted over a cache of STRING queries that holds the nine screens' anchors and hits,
# so they are not gene-intrinsic (analysis/hub_leak.py; paper Appendix, "Hub score and PPI weight are not
# gene-intrinsic"). Dropping them raises the matched static prior from 0.231 to 0.240. This run asks
# whether the full system's gain survives against that cleaner prior. Everything else is held at the
# reported protocol: GPT-5.6 Luna / reasoning none, corrected task prompts, named-only vote, global
# w_llm = 0.2, and the reported run's verified memory (replogle_transcriptomics_v2_20rep). The only change
# is WADDINGTON_EXCLUDE_FEATS, which removes the two columns from the ML branch and the padding ranker and
# is recorded in the run manifest. Read with: python -m waddington_select.analysis.clean_connectivity
#
#   experiments/22_clean_connectivity.sh             # 20 replicates (default)
#   REPS=5 experiments/22_clean_connectivity.sh
export WADDINGTON_LEGACY_TASK_PROMPTS=0
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="openai-codex/gpt-5.6-luna"
REF_DIR="workspace/results/gpt/replogle_transcriptomics_v2_20rep"
MEMORY_PATH="$REF_DIR/experience_memory.json"
OUT_DIR="workspace/results/gpt/clean_connectivity"
REPS="${REPS:-20}"

unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE
unset WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV WADDINGTON_TRAINING_CSV_V3
unset PYTHONOPTIMIZE
export WADDINGTON_EXCLUDE_FEATS="hub_score_norm,ppi_score_sum"
export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="$MODEL"
export WADDINGTON_CHAT_MODEL="$MODEL"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export WADDINGTON_LLM_SLEEP="${WADDINGTON_LLM_SLEEP:-0}"
export WADDINGTON_FULL_SELECTED_CONTEXT=1
export WADDINGTON_LLM_NAMED_ONLY=1
export WADDINGTON_DUMP_SELECTIONS=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-6}"
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"

[ -s "$MEMORY_PATH" ] || { echo "ERROR: reference memory $MEMORY_PATH missing" >&2; exit 1; }
mkdir -p "$OUT_DIR"
exec 9>"$OUT_DIR/.experiment.lock"
if ! flock -n 9; then
  echo "ERROR: another experiment already holds $OUT_DIR/.experiment.lock" >&2
  exit 1
fi

node --input-type=module -e '
  import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
  const target = process.env.WADDINGTON_PI_MODEL;
  const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
  if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
  console.log(`Authorized model: ${target}`);
'

echo "=== Experiment 22: C-arm without [$WADDINGTON_EXCLUDE_FEATS], $REPS replicates, memory $MEMORY_PATH ==="
conda run --no-capture-output -n waddington-bio python3 -m waddington_select \
  --arms waddington_c \
  --replicates "$REPS" \
  --seed-mode local \
  --rounds 5 \
  --resume \
  --out "$OUT_DIR/waddington_c_${REPS}rep.json"
echo "done -> $OUT_DIR"
