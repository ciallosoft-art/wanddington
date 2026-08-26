#!/usr/bin/env bash
# GPT-5.6 Luna component ablations (twenty evaluation replicates).
#
# Re-runs the paired component ablation of the C-arm on the GPT backend, reusing the strictly verified
# GPT memory already built by 05_gpt_luna_20repeat.sh (no re-calibration). The four ablation variants
# (− memory, − LLM, − online ML, shuffled names) predate the leakage-free refactor and hardcode the
# legacy target-aware router, so the base waddington_c is run in the SAME legacy mode
# (WADDINGTON_LEGACY_ROUTER=1, legacy padded voting) — this keeps every arm on one protocol so the
# paired deltas are clean. The headline main table stays the leakage-free honest GPT run; this is the
# "Legacy component decomposition and ablation" section, now GPT-backed rather than Claude-backed.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="openai-codex/gpt-5.6-luna"
RESULT_DIR="workspace/results/gpt/gpt_5_6_luna_none_20rep"
MEMORY_PATH="$RESULT_DIR/experience_memory.json"
ABLATION_RESULT="$RESULT_DIR/ablations_20rep.json"

if [[ ! -f "$MEMORY_PATH" ]]; then
  echo "ERROR: GPT memory $MEMORY_PATH not found; run 05_gpt_luna_20repeat.sh first" >&2
  exit 1
fi

# Legacy-consistent protocol for the paired ablation: the variants hardcode the target-aware router
# and legacy padded voting, so the base arm is put in the same mode. WADDINGTON_PI_CMD unset -> the
# fingerprinted in-repository tool-less bridge is used.
unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE
unset WADDINGTON_EXCLUDE_STUDY WADDINGTON_TRAINING_CSV
unset WADDINGTON_LLM_NAMED_ONLY PYTHONOPTIMIZE
export WADDINGTON_LEGACY_ROUTER=1
export WADDINGTON_LLM_BACKEND=pi
export WADDINGTON_PI_MODEL="$MODEL"
export WADDINGTON_CHAT_MODEL="$MODEL"
export WADDINGTON_PI_REASONING=none
export WADDINGTON_PI_STDIN=1
export WADDINGTON_LLM_SLEEP="${WADDINGTON_LLM_SLEEP:-0}"
export WADDINGTON_FULL_SELECTED_CONTEXT=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"

mkdir -p "$RESULT_DIR"

exec 9>"$RESULT_DIR/.ablation.lock"
if ! flock -n 9; then
  echo "ERROR: another GPT ablation already holds $RESULT_DIR/.ablation.lock" >&2
  exit 1
fi

# Fail before an expensive run if the exact active model is not authorized.
node --input-type=module -e '
  import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
  const target = process.env.WADDINGTON_PI_MODEL;
  const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
  if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
  console.log(`Authorized model: ${target}`);
'

echo "=== Experiment 06: GPT-5.6 Luna component ablations ==="
echo "Protocol: nine screens x twenty evaluation replicates x five rounds, legacy-consistent"
echo "Arms: waddington_c (base) + no_memory + no_llm + no_ml + shuffled_names"
echo "Reusing GPT memory: $MEMORY_PATH"
echo

conda run -n waddington-bio python3 -m waddington_select \
  --arms waddington_c waddington_c_no_memory waddington_c_no_llm waddington_c_no_ml waddington_c_shuffled_names \
  --replicates 20 \
  --seed-mode local \
  --rounds 5 \
  --resume \
  --out "$ABLATION_RESULT"

echo
echo "Done. GPT ablations: $ABLATION_RESULT"
echo "      Manifest:      ${ABLATION_RESULT%.json}.manifest.json"
