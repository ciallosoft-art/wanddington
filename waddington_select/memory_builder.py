"""
memory_builder.py — Cross-experiment memory generation for WaddingtonArm.

For each BDA dataset, generates a strategy insight summary using the configured LLM,
capturing what gene families/pathways are hits and which observed arm works best. The GPT proposal arm
is explicitly treated as a composite because unmatched slots are padded by a static LOO ranker.
Stored as experience_memory.json for LOO use by WaddingtonArm.

Verified admission (DeLM-inspired): before any entry is written to disk it
passes two gates:
  1. Mechanical check — `best_strategy` label must be consistent with the
     actual arm_performance numbers (no LLM cost, deterministic).
  2. LLM verifier call — a separate model call confirms that strategy_insight
     claims are grounded in the performance data and sample hit genes, and
     rejects summaries that misattribute which arm won.
If verification fails the generator retries with the issues injected into the
next prompt. Entries that still fail after MAX_VERIFY_RETRIES are admitted
with a `"verified": false` flag so callers can filter if needed.

Usage (after a completed calibration):
    conda run -n waddington-bio python3 -m waddington_select.memory_builder \
        --performance-json workspace/results/gpt/gpt_5_6_luna_none/calibration_5rep.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from .oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from .arms.llm_reasoning_arm import _load_task
from .config import (
    ACTIVE_GPT_RESULT_DIR, ACTIVE_MEMORY_PATH, ACTIVE_TASK_PROMPT_PROTOCOL, task_prompt_protocol,
)
from .llm_client import (
    DEFAULT_BACKEND,
    DEFAULT_PI_REASONING,
    LLMClient,
    default_model_for_backend,
)

LLM_MODEL = default_model_for_backend()

MEMORY_PATH = ACTIVE_MEMORY_PATH

# Frozen Claude-era evidence, retained only for explicit historical reconstruction. Active GPT
# memory must be conditioned on a completed GPT calibration JSON via --performance-json.
LEGACY_KNOWN_PERFORMANCE: dict[str, dict[str, float]] = {
    "IFNG":                   {"random": 0.029, "coreset": 0.100, "static_ranker": 0.168, "online_adaptive": 0.183, "llm_reasoning": 0.156},
    "IL2":                    {"random": 0.031, "coreset": 0.139, "static_ranker": 0.306, "online_adaptive": 0.314, "llm_reasoning": 0.253},
    "Sanchez21":              {"random": 0.037, "coreset": 0.031, "static_ranker": 0.077, "online_adaptive": 0.087, "llm_reasoning": 0.060},
    "Sanchez21_down":         {"random": 0.029, "coreset": 0.078, "static_ranker": 0.091, "online_adaptive": 0.101, "llm_reasoning": 0.069},
    "Carnevale22":            {"random": 0.024, "coreset": 0.054, "static_ranker": 0.048, "online_adaptive": 0.047, "llm_reasoning": 0.058},
    "Scharenberg22":          {"random": 0.102, "coreset": 0.286, "static_ranker": 0.449, "online_adaptive": 0.449, "llm_reasoning": 0.469},
    "Steinhart":              {"random": 0.021, "coreset": 0.090, "static_ranker": 0.076, "online_adaptive": 0.090, "llm_reasoning": 0.152},
    "Replogle_K562_essential":{"random": 0.254, "coreset": 0.270, "static_ranker": 0.492, "online_adaptive": 0.476, "llm_reasoning": 0.550},
    "Replogle_K562_gwps":     {"random": 0.065, "coreset": 0.156, "static_ranker": 0.247, "online_adaptive": 0.273, "llm_reasoning": 0.214},
}

ACTIVE_CALIBRATION_PATH = ACTIVE_GPT_RESULT_DIR / "calibration_5rep.json"
REQUIRED_PERFORMANCE_ARMS = (
    "random", "coreset", "static_ranker", "online_adaptive", "llm_reasoning",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_calibration_performance(
    path: Path,
    datasets: list[str],
    expected_repeats: int,
) -> tuple[dict[str, dict[str, float]], dict]:
    """Load authoritative GPT calibration results and reduce final ratios to per-arm means."""
    manifest_path = path.with_name(f"{path.stem}.manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"calibration provenance manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    protocol = manifest.get("protocol") if isinstance(manifest, dict) else None
    if not isinstance(protocol, dict):
        raise ValueError(f"calibration manifest has no protocol object: {manifest_path}")
    # Local import avoids adding the full experiment runner to module import time.
    from .run_sequential import _runtime_fingerprint, _selection_config
    expected_protocol = {
        **_selection_config(datasets, list(REQUIRED_PERFORMANCE_ARMS)),
        "task_prompt_protocol": ACTIVE_TASK_PROMPT_PROTOCOL,
        "datasets": datasets,
        "arms": list(REQUIRED_PERFORMANCE_ARMS),
        "rounds": 5,
        "repeats": expected_repeats,
        "seed_mode": "local",
        "llm_backend": DEFAULT_BACKEND,
        "llm_model": LLM_MODEL,
        "llm_reasoning": (
            os.environ.get("WADDINGTON_PI_REASONING", DEFAULT_PI_REASONING)
            if DEFAULT_BACKEND == "pi" else None
        ),
        "full_selected_context": True,
        "llm_vote_policy": "gpt_named_only",
        "legacy_router": False,
        "force_route": None,
        "force_wllm": None,
        "effective_wllm": 0.2,
        "feature_policy": "metadata",
        "drop_anchor_features": True,
        "hit_frequency_only": False,
        "exclude_study": False,
        # The gate's job is that this stage was calibrated on the SAME feature/label table that is
        # active now — not that the table is the frozen benchmark one. Hardcoding None here made the
        # provenance check unsatisfiable for any isolated superset table (the CRISPRa validation, and
        # BDA's two other difficulty tiers), which is a configuration the runner explicitly supports.
        "training_csv_override": os.environ.get("WADDINGTON_TRAINING_CSV"),
        "training_csv_v3_override": os.environ.get("WADDINGTON_TRAINING_CSV_V3"),
        "reason_from_structure": False,
        "gene_level_audit": True,
        "dump_revealed_hit_genes": True,
        "cross_experiment_split": "leave_one_phenotype_out",
        "same_study_siblings_allowed": True,
        "process_determinism": {
            "pythonhashseed": "0",
            "omp_num_threads": "1",
            "openblas_num_threads": "1",
            "mkl_num_threads": "1",
        },
    }
    mismatches = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected_protocol.items() if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"calibration manifest mismatch: {json.dumps(mismatches, sort_keys=True)}")
    if protocol.get("runtime_fingerprint") != _runtime_fingerprint():
        raise ValueError("runtime/input fingerprint changed since calibration; start a new calibration")
    protocol_sha256 = _sha256_bytes(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
    )

    raw_bytes = path.read_bytes()
    data = json.loads(raw_bytes)
    if not isinstance(data, dict) or set(data) != set(datasets):
        raise ValueError("calibration dataset keys do not exactly match the requested datasets")
    performance: dict[str, dict[str, float]] = {}
    dataset_diagnostics: dict[str, dict[str, dict]] = {}
    for dataset in datasets:
        if dataset not in data:
            raise ValueError(f"calibration is missing dataset '{dataset}': {path}")
        if not isinstance(data[dataset], dict) or set(data[dataset]) != set(REQUIRED_PERFORMANCE_ARMS):
            raise ValueError(f"calibration arm keys are incomplete or unexpected for '{dataset}'")
        performance[dataset] = {}
        dataset_diagnostics[dataset] = {}
        for arm in REQUIRED_PERFORMANCE_ARMS:
            repeats = data[dataset].get(arm)
            if not isinstance(repeats, list) or len(repeats) != expected_repeats:
                got = len(repeats) if isinstance(repeats, list) else 0
                raise ValueError(
                    f"calibration {dataset}/{arm} has {got} repeats; expected {expected_repeats}"
                )
            finals: list[float] = []
            fallback_slots = 0
            total_slots = 0
            for idx, result in enumerate(repeats):
                if result.get("dataset") != dataset or result.get("arm_name") != arm:
                    raise ValueError(f"calibration identity mismatch at {dataset}/{arm}/repeat {idx}")
                ratios = result.get("hit_ratio_per_round")
                if result.get("n_rounds") != 5 or not isinstance(ratios, list) or len(ratios) != 5:
                    raise ValueError(f"invalid five-round result at {dataset}/{arm}/repeat {idx}")
                if result.get("repeat_index") != idx:
                    raise ValueError(f"non-contiguous repeat_index at {dataset}/{arm}/repeat {idx}")
                finals.append(float(ratios[-1]))
                batch_size = result.get("batch_size")
                if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
                        or batch_size != BATCH_SIZES[dataset]):
                    raise ValueError(f"invalid batch_size at {dataset}/{arm}/repeat {idx}")
                if arm == "llm_reasoning":
                    fallback = result.get("llm_fallback_per_round")
                    if not isinstance(fallback, list) or len(fallback) != 5:
                        raise ValueError(
                            f"missing five-round LLM fallback diagnostics at {dataset}/{arm}/repeat {idx}"
                        )
                    if any(
                        isinstance(value, bool) or not isinstance(value, int)
                        or value < 0 or value > batch_size
                        for value in fallback
                    ):
                        raise ValueError(
                            f"invalid LLM fallback count at {dataset}/{arm}/repeat {idx}"
                        )
                    fallback_slots += sum(fallback)
                    total_slots += batch_size * 5
            performance[dataset][arm] = round(sum(finals) / len(finals), 6)
            dataset_diagnostics[dataset][arm] = {
                "method": (
                    "GPT proposals padded with a static leave-one-phenotype-out ML ranking"
                    if arm == "llm_reasoning" else arm
                ),
                **({
                    "fallback_slots": fallback_slots,
                    "total_slots": total_slots,
                    "fallback_fraction": round(fallback_slots / total_slots, 6),
                } if arm == "llm_reasoning" else {}),
            }
    return performance, {
        "kind": "gpt_calibration_json",
        "path": str(path),
        "sha256": _sha256_bytes(raw_bytes),
        "manifest_path": str(manifest_path),
        # The full manifest has mutable session timestamps. Bind memory to the stable scientific
        # protocol object so a no-op --resume cannot invalidate already-generated memory/C results.
        "protocol_sha256": protocol_sha256,
        "runtime_fingerprint": protocol["runtime_fingerprint"],
        "repeats": expected_repeats,
        "rounds": 5,
        "arms": list(REQUIRED_PERFORMANCE_ARMS),
        "dataset_diagnostics": dataset_diagnostics,
    }


def _legacy_performance_source() -> tuple[dict[str, dict[str, float]], dict]:
    payload = json.dumps(LEGACY_KNOWN_PERFORMANCE, sort_keys=True).encode()
    return LEGACY_KNOWN_PERFORMANCE, {
        "kind": "historical_v9_v11_claude_era_summaries",
        "sha256": _sha256_bytes(payload),
        "repeats": 3,
        "rounds": 5,
        "arms": list(REQUIRED_PERFORMANCE_ARMS),
    }

# Top hit genes (sampled from oracle ground truth) to give the configured model concrete examples
# These are the actual hits - legitimate to use in memory since they were "revealed" in our experiments
def get_top_hit_genes(dataset_name: str, n: int = 20) -> list[str]:
    import pandas as pd
    from .tables import TRAINING_DATA_CSV
    df = pd.read_csv(TRAINING_DATA_CSV)
    df["gene"] = df["gene"].str.strip().str.upper()
    sub = df[(df["dataset"] == dataset_name) & (df["label"] == 1)]
    # Return genes with highest PPI scores (most biologically "visible" hits)
    if "g1_ppi_score" in df.columns:
        sub = sub.sort_values("g1_ppi_score", ascending=False)
    return sub["gene"].head(n).tolist()


MAX_VERIFY_RETRIES = 2

# ---------------------------------------------------------------------------
# Verification (DeLM-style admitted-context gate)
# ---------------------------------------------------------------------------

# Deterministic arm → expected strategy label mapping.
_ARM_TO_STRATEGY: dict[str, str] = {
    "llm_reasoning": "gpt_plus_ml_padding",
    "online_adaptive": "ml",
    "static_ranker": "ml",
    "coreset": "diversity",
    "random": "random",
}

_VERIFIER_PROMPT = """\
A strategy summary was auto-generated for a CRISPR experiment and will be stored
as reusable memory injected into future gene-selection agents.
Verify it is grounded in the authoritative data below.

AUTHORITATIVE DATA:
- Arm performance (hit_ratio @ R5, higher = better): {perf_str}
- Best arm by performance: {best_arm}
- Arm identity/diagnostics: {arm_diagnostics}
- Sample hit genes (top-scoring by PPI connectivity): {top_hits}

The `llm_reasoning` arm is NOT a pure LLM intervention: valid GPT-proposed genes are used first and
all unmatched slots are filled by a static leave-one-phenotype-out ML ranker. Its performance cannot
identify a causal GPT-vs-ML advantage. If it wins, describe the winning composite branch, not "LLM
biological knowledge beating ML".

GENERATED SUMMARY:
- best_strategy: "{best_strategy}"
- top_hit_families: {top_hit_families}
- key_genes: {key_genes}
- strategy_insight: "{strategy_insight}"

Verify ONLY these points:
1. Does `strategy_insight` correctly identify which observed arm performed best without treating the
   padded `llm_reasoning` arm as a pure LLM intervention?
2. Are `key_genes` plausible given the sample hit genes? They need not match
   exactly, but must not be completely unrelated genes invented without basis.
3. Does `strategy_insight` contain causal claims unsupported by the data?

Reply with ONLY valid JSON — no prose, no markdown fences:
{{"valid": true}} if trustworthy, or
{{"valid": false, "issues": ["specific problem 1", "specific problem 2"]}}"""


def _expected_strategy(best_arm: str) -> str:
    """Deterministic arm name → best_strategy label."""
    return _ARM_TO_STRATEGY.get(best_arm, "hybrid")


def _verify_entry(
    entry: dict,
    top_hits: list[str],
    client: LLMClient,
) -> tuple[bool, list[str], dict]:
    """Two-phase admission gate (DeLM §3.2).

    Phase 1 — mechanical (free): check best_strategy label consistency.
    Phase 2 — LLM verifier: check strategy_insight is data-grounded.
    Returns (is_valid, issues, provenance). Verifier errors and malformed replies fail closed.
    Provider/auth/quota failures propagate so they cannot be mislabeled as scientific rejection.
    """
    issues: list[str] = []

    # --- Phase 1: mechanical ---
    expected = _expected_strategy(entry["best_arm"])
    actual = entry.get("best_strategy", "")
    if actual != expected:
        issues.append(
            f"best_strategy='{actual}' inconsistent with best_arm='{entry['best_arm']}'"
            f" (expected '{expected}')"
        )
    insight = entry.get("strategy_insight")
    if not isinstance(insight, str) or not insight.strip():
        issues.append("empty strategy_insight")
    elif len(insight) > 4000:
        issues.append("strategy_insight exceeds 4000 characters")
    families = entry.get("top_hit_families")
    if (
        not isinstance(families, list) or not 1 <= len(families) <= 12
        or any(not isinstance(item, str) or not item.strip() or len(item) > 200 for item in families)
    ):
        issues.append("top_hit_families must be a non-empty list of at most 12 strings")
    key_genes = entry.get("key_genes")
    if (
        not isinstance(key_genes, list) or not 1 <= len(key_genes) <= 25
        or any(not isinstance(item, str) or not item.strip() or len(item) > 40 for item in key_genes)
    ):
        issues.append("key_genes must be a non-empty list of at most 25 strings")
    perf = entry.get("arm_performance")
    if (
        not isinstance(perf, dict) or set(perf) != set(REQUIRED_PERFORMANCE_ARMS)
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in perf.values())
    ):
        issues.append("arm_performance has an invalid schema")
    diagnostics = entry.get("arm_diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(REQUIRED_PERFORMANCE_ARMS):
        issues.append("arm_diagnostics has an invalid schema")

    # --- Phase 2: LLM verifier ---
    perf = entry.get("arm_performance", {})
    perf_str = ", ".join(
        f"{arm}={score:.3f}"
        for arm, score in sorted(perf.items(), key=lambda x: -x[1])
    )
    prompt = _VERIFIER_PROMPT.format(
        perf_str=perf_str,
        best_arm=entry["best_arm"],
        top_hits=", ".join(top_hits[:15]),
        arm_diagnostics=json.dumps(entry.get("arm_diagnostics", {}), sort_keys=True),
        best_strategy=actual,
        top_hit_families=entry.get("top_hit_families", []),
        key_genes=entry.get("key_genes", []),
        strategy_insight=entry.get("strategy_insight", ""),
    )
    text = client.complete(prompt, temperature=0.0, max_tokens=200)
    verifier = {
        "backend": client.backend,
        "model": client.model,
        "reasoning": (
            os.environ.get("WADDINGTON_PI_REASONING", DEFAULT_PI_REASONING)
            if client.backend == "pi" else None
        ),
        "response_sha256": _sha256_bytes(text.encode()),
    }
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        result = json.loads(m.group()) if m else None
    except json.JSONDecodeError:
        result = None
    if not isinstance(result, dict) or not isinstance(result.get("valid"), bool):
        issues.append("verifier returned malformed JSON or omitted boolean 'valid'")
        verifier["status"] = "invalid_response"
    elif result["valid"] is False:
        rejected = result.get("issues")
        issues.extend(rejected if isinstance(rejected, list) and rejected else ["LLM verifier rejected summary"])
        verifier["status"] = "rejected"
    else:
        verifier["status"] = "passed"

    return len(issues) == 0, issues, verifier


# ---------------------------------------------------------------------------
# Entry generation (with verified admission)
# ---------------------------------------------------------------------------

def _build_generation_prompt(
    dataset_name: str,
    task: dict,
    oracle,
    perf_str: str,
    best_arm: str,
    top_hits: list[str],
    arm_diagnostics: dict,
    prior_issues: list[str],
) -> str:
    correction = ""
    if prior_issues:
        correction = (
            "\n\nPREVIOUS ATTEMPT WAS REJECTED by the verifier. Fix these issues:\n"
            + "\n".join(f"  - {i}" for i in prior_issues)
            + "\n"
        )
    return f"""You are analyzing results from a completed CRISPR perturbation screen experiment.
Generate a strategy insight that will help a future agent select genes for a SIMILAR task.

COMPLETED EXPERIMENT:
Dataset: {dataset_name}
Task: {task.get('Task', dataset_name)}
Measurement: {task.get('Measurement', '')}
Total genes screened: {oracle.n_genes}
Total hits: {oracle.total_hits} ({oracle.total_hits/oracle.n_genes*100:.1f}% hit rate)

ARM PERFORMANCE (hit_ratio @ 5 rounds of sequential selection):
{perf_str}
Best arm: {best_arm}

ARM IDENTITY / DIAGNOSTICS:
{json.dumps(arm_diagnostics, sort_keys=True)}

IMPORTANT: `llm_reasoning` means GPT proposals PLUS static LOO-ML padding for every unmatched slot.
Its score is a composite outcome and does not isolate a causal GPT contribution. Never say that GPT
or "LLM biological knowledge" beat ML solely because this composite arm scored highest.

SAMPLE HIT GENES (top-scoring hits by PPI connectivity):
{', '.join(top_hits)}{correction}
Generate a concise strategy insight (4-6 sentences) for this experiment type. Include:
1. Which biological pathways/gene families appear most enriched (based on the sample hit genes)
2. Which OBSERVED ARM was more effective. Describe `llm_reasoning` as the GPT+ML-padding composite
   and avoid causal component attribution. Your best_strategy label MUST match the best arm above.
3. Specific advice for a future agent facing a similar task
4. Any caveats or surprising findings

Return ONLY a JSON object with these exact keys:
{{
  "strategy_insight": "...",
  "top_hit_families": ["pathway1", "pathway2", "pathway3"],
  "best_strategy": "ml" | "gpt_plus_ml_padding" | "diversity" | "random",
  "key_genes": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
}}"""


def _parse_llm_json(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def generate_memory_entry(
    dataset_name: str,
    client: LLMClient,
    performance: dict[str, dict[str, float]],
    performance_source: dict,
) -> dict:
    task = _load_task(dataset_name)
    oracle = DatasetOracle(dataset_name)
    perf = performance.get(dataset_name, {})
    if not perf:
        raise ValueError(f"no calibration performance for {dataset_name}")
    top_hits = get_top_hit_genes(dataset_name, n=25)

    best_arm = max(perf, key=perf.get) if perf else "unknown"
    arm_diagnostics = (
        performance_source.get("dataset_diagnostics", {}).get(dataset_name, {})
    )
    perf_str = "\n".join(
        f"  {arm}: {score:.3f}"
        for arm, score in sorted(perf.items(), key=lambda x: -x[1])
    )

    prior_issues: list[str] = []
    entry: dict = {}
    verified = False

    for attempt in range(MAX_VERIFY_RETRIES + 1):
        prompt = _build_generation_prompt(
            dataset_name, task, oracle, perf_str, best_arm, top_hits, arm_diagnostics, prior_issues
        )
        text = client.complete(prompt, temperature=0.3, max_tokens=600)
        parsed = _parse_llm_json(text)

        entry = {
            "dataset": dataset_name,
            "task": task.get("Task", dataset_name),
            "measurement": task.get("Measurement", ""),
            "n_genes": oracle.n_genes,
            "n_hits": oracle.total_hits,
            "hit_rate": round(oracle.total_hits / oracle.n_genes, 4),
            "arm_performance": perf,
            "best_arm": best_arm,
            "arm_diagnostics": arm_diagnostics,
            "strategy_insight": parsed.get("strategy_insight", ""),
            "top_hit_families": parsed.get("top_hit_families", []),
            # This label is a deterministic derivative of the measured winning arm, not a creative
            # field. Normalising it here prevents a semantically correct generation such as
            # "online_adaptive" from failing merely because the schema requires the coarser "ml".
            "best_strategy": _expected_strategy(best_arm),
            "key_genes": parsed.get("key_genes", top_hits[:5]),
            "generator": {
                "task_prompt_protocol": task_prompt_protocol(),
                "backend": client.backend,
                "model": client.model,
                "reasoning": (
                    os.environ.get("WADDINGTON_PI_REASONING", DEFAULT_PI_REASONING)
                    if client.backend == "pi" else None
                ),
                "performance_source": performance_source,
            },
        }

        valid, issues, verifier = _verify_entry(entry, top_hits, client)
        entry["verification"] = verifier
        if valid:
            verified = True
            break
        prior_issues = issues
        print(f"    [verify] attempt {attempt + 1}/{MAX_VERIFY_RETRIES + 1} failed: {issues}", flush=True)

    entry["verified"] = verified
    return entry


def build_memory(
    datasets: list[str],
    performance: dict[str, dict[str, float]],
    performance_source: dict,
    client: LLMClient | None = None,
) -> list[dict]:
    client = client or LLMClient(model=LLM_MODEL, temperature=0.3, max_tokens=600)

    entries: list[dict] = []
    for ds in datasets:
        print(f"  Generating memory for {ds}...", flush=True)
        entry = generate_memory_entry(ds, client, performance, performance_source)
        entries.append(entry)
        status = "✓ verified" if entry.get("verified") else "✗ unverified"
        print(
            f"    [{status}] best_arm={entry['best_arm']}  "
            f"best_strategy={entry['best_strategy']}  "
            f"families={entry['top_hit_families'][:3]}",
            flush=True,
        )

    n_ok = sum(1 for e in entries if e.get("verified"))
    print(f"\n  Verification summary: {n_ok}/{len(entries)} entries passed.", flush=True)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-experiment memory")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--out", type=Path, default=MEMORY_PATH)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--performance-json",
        type=Path,
        default=None,
        help=f"completed GPT calibration results (active default: {ACTIVE_CALIBRATION_PATH})",
    )
    source.add_argument(
        "--use-legacy-performance",
        action="store_true",
        help="explicitly use frozen Claude-era V9-V11 evidence for historical reconstruction",
    )
    parser.add_argument("--expected-repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true",
                        help="keep only verified entries whose model and performance provenance still match")
    args = parser.parse_args()

    if args.use_legacy_performance:
        performance, performance_source = _legacy_performance_source()
    else:
        calibration_path = args.performance_json or ACTIVE_CALIBRATION_PATH
        if not calibration_path.exists():
            raise SystemExit(
                f"GPT calibration not found: {calibration_path}. Run the calibration stage first, "
                "or pass --use-legacy-performance only for historical reconstruction."
            )
        performance, performance_source = _load_calibration_performance(
            calibration_path, args.datasets, args.expected_repeats,
        )

    client = LLMClient(model=LLM_MODEL, temperature=0.3, max_tokens=600)
    expected_reasoning = (
        os.environ.get("WADDINGTON_PI_REASONING", DEFAULT_PI_REASONING)
        if client.backend == "pi" else None
    )

    existing: dict[str, dict] = {}
    if args.resume and args.out.exists():
        with open(args.out) as f:
            prior = json.load(f)
        existing = {
            e["dataset"]: e for e in prior
            if (
                e.get("dataset") in args.datasets
                and e.get("verified") is True
                and e.get("verification", {}).get("status") == "passed"
                and e.get("generator", {}).get("backend") == client.backend
                and e.get("generator", {}).get("model") == client.model
                and e.get("generator", {}).get("reasoning") == expected_reasoning
                and e.get("generator", {}).get("task_prompt_protocol") == task_prompt_protocol()
                and e.get("generator", {}).get("performance_source", {}).get("sha256")
                    == performance_source["sha256"]
                and e.get("generator", {}).get("performance_source", {}).get("protocol_sha256")
                    == performance_source.get("protocol_sha256")
            )
        }
    pending = [d for d in args.datasets if d not in existing]
    if existing:
        print(f"[resume] keeping {len(existing)} verified entries; regenerating {len(pending)}")
    print(f"Building experience memory for {len(pending)} datasets...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined = dict(existing)

    def checkpoint() -> list[dict]:
        current = [combined[d] for d in args.datasets if d in combined]
        tmp = args.out.with_name(f".{args.out.name}.tmp")
        with open(tmp, "w") as f:
            json.dump(current, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, args.out)
        return current

    for dataset in pending:
        print(f"  Generating memory for {dataset}...", flush=True)
        entry = generate_memory_entry(
            dataset, client, performance, performance_source,
        )
        combined[dataset] = entry
        entries = checkpoint()
        status = "✓ verified" if entry.get("verified") else "✗ unverified"
        print(
            f"    [{status}] best_arm={entry['best_arm']}  "
            f"best_strategy={entry['best_strategy']}  "
            f"families={entry['top_hit_families'][:3]}",
            flush=True,
        )
    entries = checkpoint()
    n_ok = sum(1 for e in entries if e.get("verified") is True)
    print(f"\n  Verification summary: {n_ok}/{len(entries)} entries passed.", flush=True)
    print(f"\nMemory saved to {args.out}  ({len(entries)} entries)")
    for e in entries:
        print(f"  {e['dataset']:30s}: best={e['best_arm']:18s}  insight={e['strategy_insight'][:60]}...")
    if n_ok != len(args.datasets):
        raise SystemExit(
            f"memory admission failed: {n_ok}/{len(args.datasets)} requested entries are strictly verified"
        )


if __name__ == "__main__":
    main()
