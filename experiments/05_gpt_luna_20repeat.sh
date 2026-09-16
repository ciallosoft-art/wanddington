#!/usr/bin/env bash
# Use task prompts matched to the actual Perturb-seq readout definitions.
export WADDINGTON_LEGACY_TASK_PROMPTS=0
# End-to-end GPT Waddington evaluation (twenty evaluation replicates).
#
# Stage 1 calibrates the four non-LLM comparators plus GPT-B on all nine screens. Stage 2 builds
# strictly verified GPT memory from those calibration results. Stage 3 evaluates GPT-C. Keeping the
# stages separate prevents Claude-era performance summaries from entering the active GPT memory.
#
# GPT-5.6 Luna is OpenAI's cost-sensitive GPT-5.6 tier. We use the officially supported reasoning
# effort `none`. The Codex OAuth endpoint exposes no sampling seed, so the LLM-backed conditions are
# five repeated API invocations, not reproducible model seeds; independence is not assumed. Of the
# formal baselines, only Random consumes the local replicate-index seed. Every repeat is atomically
# checkpointed and runtime/input hashes gate resume.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="openai-codex/gpt-5.6-luna"
RESULT_DIR="workspace/results/gpt/gpt_5_6_luna_none_20rep"
CALIBRATION_RESULT="$RESULT_DIR/calibration_20rep.json"
MEMORY_PATH="$RESULT_DIR/experience_memory.json"
C_RESULT="$RESULT_DIR/waddington_c_20rep.json"
MAIN_RESULT="$RESULT_DIR/main_table_20rep.json"
SKILL_PATH="$RESULT_DIR/skill_library.json"

# Prevent a caller's development shell from silently changing the protocol. WADDINGTON_PI_CMD is
# deliberately unset so the fingerprinted in-repository tool-less bridge is used.
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
# Repeat-level parallelism (default 6; export WADDINGTON_PARALLEL=1 to force sequential).
# N is not part of the provenance manifest and does not change results.
export WADDINGTON_PARALLEL="${WADDINGTON_PARALLEL:-6}"
export WADDINGTON_GPT_RESULT_DIR="$RESULT_DIR"
export WADDINGTON_MEMORY_PATH="$MEMORY_PATH"
export WADDINGTON_SKILL_LIBRARY_PATH="$SKILL_PATH"

mkdir -p "$RESULT_DIR"

# Hold one lock for the entire staged workflow. Atomic JSON replacement prevents truncation, while
# this lock also prevents two runners from issuing the same GPT repeat and overwriting each other's
# checkpoint. The file descriptor is released automatically when the script exits.
exec 9>"$RESULT_DIR/.experiment.lock"
if ! flock -n 9; then
  echo "ERROR: another GPT experiment already holds $RESULT_DIR/.experiment.lock" >&2
  exit 1
fi

echo "=== Experiment 04: GPT-5.6 Luna / reasoning none ==="
echo "Protocol: nine screens x twenty evaluation replicates x five rounds"
echo "LLM calls: GPT-B 225 + memory generation/verification + GPT-C 225"
echo "Checkpoint directory: $RESULT_DIR"
echo

# Fail before an expensive run if the exact active model is not authorized. Compatibility providers
# are never selected implicitly.
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

echo "--- Stage 1/3: GPT calibration (five arms) ---"
conda run -n waddington-bio python3 -m waddington_select \
  --arms random coreset static_ranker online_adaptive llm_reasoning \
  --replicates 20 \
  --seed-mode local \
  --rounds 5 \
  --resume \
  --out "$CALIBRATION_RESULT"

echo "--- Stage 2/3: GPT-native, fail-closed memory ---"
conda run -n waddington-bio python3 -m waddington_select.memory_builder \
  --datasets "${BENCHMARK_DATASETS[@]}" \
  --performance-json "$CALIBRATION_RESULT" \
  --expected-repeats 20 \
  --resume \
  --out "$MEMORY_PATH"

# Require exactly one verified, provenance-matched entry for every benchmark phenotype. These are
# explicit exceptions rather than asserts, so Python optimisation cannot remove the scientific gate.
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

# Skill variants are not part of the six-arm headline. Build them only when explicitly requested so
# the default experiment does not spend GPT quota on an unused diagnostic artifact.
if [[ "${WADDINGTON_BUILD_GPT_SKILLS:-0}" == "1" ]]; then
  echo "--- Optional: GPT skill library (not used by the main table) ---"
  conda run -n waddington-bio python3 -m waddington_select.skill_builder --out "$SKILL_PATH"
fi

echo "--- Stage 3/3: GPT-C with GPT-calibrated memory ---"
conda run -n waddington-bio python3 -m waddington_select \
  --arms waddington_c \
  --replicates 20 \
  --seed-mode local \
  --rounds 5 \
  --resume \
  --out "$C_RESULT"

conda run -n waddington-bio python3 -m waddington_select.merge_results \
  --calibration "$CALIBRATION_RESULT" \
  --carm "$C_RESULT" \
  --memory "$MEMORY_PATH" \
  --repeats 20 \
  --out "$MAIN_RESULT"

echo
echo "Done. GPT main table: $MAIN_RESULT"
echo "      Manifest:       ${MAIN_RESULT%.json}.manifest.json"
