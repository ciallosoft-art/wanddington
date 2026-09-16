#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# BDA difficulty tier "easier": 1-gene selection over the FULL gene pool (essentials included).
#
# Identical protocol to 05_gpt_luna_20repeat.sh — same nine screens, same batch sizes, same task
# prompts, same three stages, same twenty evaluation replicates — with ONE change: every consumer
# reads the unfiltered feature tables built by workspace/evaluation/build_tier_a_unfiltered.py
# instead of the frozen CEGv2-filtered ones. The paper's headline tier (1-gene, non-essential) is
# untouched; this is written to its own result directory.
#
# Memory is rebuilt from THIS tier's calibration: the memory summarises which strategy won on each
# screen, and that is a property of the pool, so reusing the frozen tier's memory would describe a
# different experiment.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="openai-codex/gpt-5.6-luna"
RESULT_DIR="workspace/results/gpt/tier_a_full_pool_20rep"
CALIBRATION_RESULT="$RESULT_DIR/calibration_20rep.json"
MEMORY_PATH="$RESULT_DIR/experience_memory.json"
C_RESULT="$RESULT_DIR/waddington_c_20rep.json"
MAIN_RESULT="$RESULT_DIR/main_table_20rep.json"
SKILL_PATH="$RESULT_DIR/skill_library.json"

TIER_V1="$PWD/workspace/evaluation/lgbm_training_data_tier_a.csv"
TIER_V3="$PWD/workspace/evaluation/lgbm_training_data_tier_a_v3.csv"

unset WADDINGTON_LLM_MODEL WADDINGTON_PI_CMD WADDINGTON_LEGACY_ROUTER
unset WADDINGTON_AUTH_PATH WADDINGTON_REUSE_FEYNMAN
unset WADDINGTON_FORCE_WLLM WADDINGTON_FORCE_ROUTE WADDINGTON_FEATURE_POLICY
unset WADDINGTON_DROP_ANCHOR_FEATS WADDINGTON_HITFREQ_ONLY WADDINGTON_REASON_FROM_STRUCTURE
unset WADDINGTON_EXCLUDE_STUDY WADDINGTON_EXTRA_DATASETS
unset PYTHONOPTIMIZE

# The one substantive difference from the headline protocol.
export WADDINGTON_TRAINING_CSV="$TIER_V1"
export WADDINGTON_TRAINING_CSV_V3="$TIER_V3"

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
export WADDINGTON_GPT_RESULT_DIR="$RESULT_DIR"
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"
export WADDINGTON_SKILL_LIBRARY_PATH="$SKILL_PATH"

for f in "$TIER_V1" "$TIER_V3"; do
  [[ -f "$f" ]] || { echo "ERROR: $f missing — run workspace/evaluation/build_tier_a_unfiltered.py" >&2; exit 1; }
done

mkdir -p "$RESULT_DIR"
exec 9>"$RESULT_DIR/.experiment.lock"
if ! flock -n 9; then
  echo "ERROR: another experiment already holds $RESULT_DIR/.experiment.lock" >&2
  exit 1
fi

echo "=== Experiment 08: BDA tier 'easier' (1-gene, full pool) ==="
echo "Tables: $(basename "$TIER_V1") / $(basename "$TIER_V3")"
echo "Protocol: nine screens x twenty evaluation replicates x five rounds"
echo "Checkpoint directory: $RESULT_DIR"
echo

node --input-type=module -e '
  import { listAvailableModels } from "./frontend/src/llm/complete.mjs";
  const target = process.env.WADDINGTON_PI_MODEL;
  const available = new Set(listAvailableModels().map((m) => `${m.provider}/${m.id}`));
  if (!available.has(target)) throw new Error(`${target} is not authorized; run: node frontend/bin/waddington.js setup`);
  console.log(`Authorized model: ${target}`);
'

BENCHMARK_DATASETS=(
  IFNG IL2 Sanchez21 Sanchez21_down Carnevale22 Scharenberg22 Steinhart
  Replogle_K562_essential Replogle_K562_gwps
)

echo "--- Stage 1/3: calibration (five arms) ---"
conda run -n waddington-bio python3 -m waddington_select \
  --arms random coreset static_ranker online_adaptive llm_reasoning \
  --replicates 20 --seed-mode local --rounds 5 --resume \
  --out "$CALIBRATION_RESULT"

echo "--- Stage 2/3: GPT-native, fail-closed memory (this tier's calibration) ---"
conda run -n waddington-bio python3 -m waddington_select.memory_builder \
  --datasets "${BENCHMARK_DATASETS[@]}" \
  --performance-json "$CALIBRATION_RESULT" \
  --expected-repeats 20 --resume \
  --out "$MEMORY_PATH"

python3 - "$MEMORY_PATH" "$CALIBRATION_RESULT" "$MODEL" <<'PY'
import hashlib, json, sys
memory_path, calibration_path, model = sys.argv[1:]
manifest_path = calibration_path.rsplit(".json", 1)[0] + ".manifest.json"
with open(memory_path) as handle:
    entries = json.load(handle)
expected = {
    "IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22", "Scharenberg22",
    "Steinhart", "Replogle_K562_essential", "Replogle_K562_gwps",
}
if not isinstance(entries, list) or len(entries) != len(expected):
    raise RuntimeError("memory gate: expected exactly nine entries")
datasets = [entry.get("dataset") for entry in entries if isinstance(entry, dict)]
if len(datasets) != len(entries) or len(set(datasets)) != len(datasets) or set(datasets) != expected:
    raise RuntimeError("memory gate: dataset identities are missing, duplicated, or unexpected")
with open(calibration_path, "rb") as handle:
    cal_hash = hashlib.sha256(handle.read()).hexdigest()
with open(manifest_path) as handle:
    protocol = json.load(handle)["protocol"]
runtime_fingerprint = protocol["runtime_fingerprint"]
protocol_hash = hashlib.sha256(
    json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
for entry in entries:
    generator = entry.get("generator", {})
    verifier = entry.get("verification", {})
    if entry.get("verified") is not True or verifier.get("status") != "passed":
        raise RuntimeError(f"memory gate: {entry.get('dataset')} is not strictly verified")
    if generator.get("backend") != "pi" or generator.get("model") != model:
        raise RuntimeError(f"memory gate: wrong generator for {entry.get('dataset')}")
    if generator.get("reasoning") != "none" or verifier.get("reasoning") != "none":
        raise RuntimeError(f"memory gate: wrong reasoning provenance for {entry.get('dataset')}")
    if verifier.get("backend") != "pi" or verifier.get("model") != model:
        raise RuntimeError(f"memory gate: wrong verifier for {entry.get('dataset')}")
    source = generator.get("performance_source", {})
    if (
        source.get("sha256") != cal_hash
        or source.get("protocol_sha256") != protocol_hash
        or source.get("runtime_fingerprint") != runtime_fingerprint
    ):
        raise RuntimeError(f"memory gate: stale calibration provenance for {entry.get('dataset')}")
print("Memory gate: 9/9 strictly verified GPT-calibrated entries")
PY

echo "--- Stage 3/3: GPT-C with this tier's memory ---"
conda run -n waddington-bio python3 -m waddington_select \
  --arms waddington_c \
  --replicates 20 --seed-mode local --rounds 5 --resume \
  --out "$C_RESULT"

conda run -n waddington-bio python3 -m waddington_select.merge_results \
  --calibration "$CALIBRATION_RESULT" \
  --carm "$C_RESULT" \
  --memory "$MEMORY_PATH" \
  --repeats 20 \
  --out "$MAIN_RESULT"

echo
echo "Done. Tier-A main table: $MAIN_RESULT"
