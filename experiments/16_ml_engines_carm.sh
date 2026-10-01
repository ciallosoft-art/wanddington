#!/usr/bin/env bash
# Experiment 16 — the full GPT C-arm with an alternative ML engine (docs/ml_engines_plan.md, step 7).
#
# Everything is held at the reported protocol except the engine: the same GPT-5.6 Luna settings,
# task prompts, named-only vote, global w_llm = 0.2 (never tuned on a target screen) and the same verified
# experience memory as the corrected-prompt 20-replicate run (replogle_transcriptomics_v2_20rep), whose
# C-arm (0.2603) is the paired reference. Engines are the stage-B candidates of step 5.
#
#   experiments/16_ml_engines_carm.sh            # gnn mlp, 5 replicates
#   REPS=20 experiments/16_ml_engines_carm.sh gnn
export WADDINGTON_LEGACY_TASK_PROMPTS=0
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="openai-codex/gpt-5.6-luna"
REF_DIR="workspace/results/gpt/replogle_transcriptomics_v2_20rep"
MEMORY_PATH="$REF_DIR/experience_memory.json"
OUT_DIR="workspace/results/ml_engines/carm"
REPS="${REPS:-5}"
ENGINES="${*:-gnn mlp}"

unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE
unset WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
unset PYTHONOPTIMIZE
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

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

ARMS=""
for e in $ENGINES; do ARMS="$ARMS waddington_c__$e"; done
echo "=== Experiment 16: C-arm with engines [$ENGINES], $REPS replicates, memory $MEMORY_PATH ==="
conda run --no-capture-output -n waddington-bio python3 -m waddington_select \
  --arms $ARMS \
  --replicates "$REPS" \
  --seed-mode local \
  --rounds 5 \
  --resume \
  --out "$OUT_DIR/carm_${REPS}rep_$(echo $ENGINES | tr ' ' '_').json"
echo "done -> $OUT_DIR"
