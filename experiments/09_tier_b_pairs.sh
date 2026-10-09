#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# BDA difficulty tier "harder": 2-gene combinatorial selection (Horlbeck 2018 GI map, K562).
#
# Same protocol as the headline run — five rounds, batches of 128, twenty evaluation replicates, the
# same six arms — but a candidate is a PAIR token "GENEA|GENEB" drawn from the 100 576 measured
# pairs, and the tables are the ones built by workspace/evaluation/build_horlbeck_pairs.py.
#
# Memory: the cross-experiment prior for this screen is the NINE SINGLE-GENE screens, which enter the
# pair table as degenerate pairs. Their memory is exactly experiment 08's, built on the same full-pool
# single-gene data, so it is reused rather than regenerated — regenerating it would produce the same
# nine summaries from the same calibration at the cost of another 18 LLM calls. Run 08 first.
set -euo pipefail
cd "$(dirname "$0")/.."

# BDA's own two-gene protocol (their Table 2) is 32 pairs/round, not the 128 the single-gene tiers
# use. Pass the batch size as $1 to produce a run directly comparable to their published numbers:
#   bash experiments/09_tier_b_pairs.sh 32
HORLBECK_BATCH="${1:-128}"

MODEL="openai-codex/gpt-5.6-luna"
RESULT_DIR="workspace/results/gpt/tier_b_pairs_20rep"
[[ "$HORLBECK_BATCH" != "128" ]] && RESULT_DIR="workspace/results/gpt/tier_b_pairs_b${HORLBECK_BATCH}_20rep"
CALIBRATION_RESULT="$RESULT_DIR/calibration_20rep.json"
MEMORY_PATH="${WADDINGTON_TIER_B_MEMORY:-$PWD/workspace/results/gpt/tier_a_full_pool_20rep/experience_memory.json}"
C_RESULT="$RESULT_DIR/waddington_c_20rep.json"
SKILL_PATH="$RESULT_DIR/skill_library.json"

PAIR_V1="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs.csv"
PAIR_V3="$PWD/workspace/evaluation/lgbm_training_data_horlbeck_pairs_v3.csv"

unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE
unset WADDINGTON_EXCLUDE_STUDY
unset PYTHONOPTIMIZE

export WADDINGTON_TRAINING_CSV="$PAIR_V1"
export WADDINGTON_TRAINING_CSV_V3="$PAIR_V3"
export WADDINGTON_EXTRA_DATASETS="Horlbeck:${HORLBECK_BATCH}"

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
# The pair pool is 100 576 candidates and each arm holds its own copy of a 46 MB table, so keep the
# default repeat-level parallelism lower than the single-gene runs.
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-4}"
export WADDINGTON_GPT_RESULT_DIR="$RESULT_DIR"
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"
export WADDINGTON_SKILL_LIBRARY_PATH="$SKILL_PATH"

for f in "$PAIR_V1" "$PAIR_V3"; do
  [[ -f "$f" ]] || { echo "ERROR: $f missing — run workspace/evaluation/build_horlbeck_pairs.py" >&2; exit 1; }
done
[[ -f "$MEMORY_PATH" ]] || {
  echo "ERROR: $MEMORY_PATH missing — run experiments/08_tier_a_full_pool.sh first" >&2; exit 1; }

python3 - "$MEMORY_PATH" <<'PY'
import json, sys
entries = json.load(open(sys.argv[1]))
expected = {
    "IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22", "Scharenberg22",
    "Steinhart", "Replogle_K562_essential", "Replogle_K562_gwps",
}
if not isinstance(entries, list) or {e.get("dataset") for e in entries} != expected:
    raise RuntimeError("memory gate: the nine single-gene screens must all be present")
if not all(e.get("verified") is True for e in entries):
    raise RuntimeError("memory gate: every entry must be strictly verified")
print(f"Memory gate: {len(entries)}/9 verified single-gene entries reused as the pair prior")
PY

mkdir -p "$RESULT_DIR"
exec 9>"$RESULT_DIR/.experiment.lock"
if ! flock -n 9; then
  echo "ERROR: another experiment already holds $RESULT_DIR/.experiment.lock" >&2
  exit 1
fi

echo "=== Experiment 09: BDA tier 'harder' (2-gene combinations, Horlbeck) ==="
echo "Tables: $(basename "$PAIR_V1") / $(basename "$PAIR_V3")"
echo "Pool:   100576 pairs over 448 genes plus a negative control, 9959 hits (9.90%); batch=${HORLBECK_BATCH}"
echo "Memory: $MEMORY_PATH"
echo "Checkpoint directory: $RESULT_DIR"
echo

node --input-type=module -e '
  import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
  const target = process.env.WADDINGTON_PI_MODEL;
  const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
  if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
  console.log(`Authorized model: ${target}`);
'

echo "--- Stage 1/2: calibration (five arms) ---"
conda run -n waddington-bio python3 -m waddington_select \
  --arms random coreset static_ranker online_adaptive llm_reasoning \
  --datasets Horlbeck \
  --replicates 20 --seed-mode local --rounds 5 --resume \
  --out "$CALIBRATION_RESULT"

echo "--- Stage 2/2: GPT-C with the single-gene cross-experiment memory ---"
conda run -n waddington-bio python3 -m waddington_select \
  --arms waddington_c \
  --datasets Horlbeck \
  --replicates 20 --seed-mode local --rounds 5 --resume \
  --out "$C_RESULT"

echo
echo "Done. Tier-B results: $CALIBRATION_RESULT + $C_RESULT"
