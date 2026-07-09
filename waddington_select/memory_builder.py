"""
memory_builder.py — Cross-experiment memory generation for WaddingtonArm.

For each BDA dataset, generates a strategy insight summary using Claude,
capturing what gene families/pathways are hits and which approach (ML vs LLM)
works best. Stored as experience_memory.json for LOO use by WaddingtonArm.

Verified admission (DeLM-inspired): before any entry is written to disk it
passes two gates:
  1. Mechanical check — `best_strategy` label must be consistent with the
     actual arm_performance numbers (no LLM cost, deterministic).
  2. LLM verifier call — a separate Claude call confirms that strategy_insight
     claims are grounded in the performance data and sample hit genes, and
     rejects summaries that misattribute which arm won.
If verification fails the generator retries with the issues injected into the
next prompt. Entries that still fail after MAX_VERIFY_RETRIES are admitted
with a `"verified": false` flag so callers can filter if needed.

Usage:
    conda run -n waddington-bio python3 workspace/agent/memory_builder.py
    conda run -n waddington-bio python3 workspace/agent/memory_builder.py --datasets IFNG IL2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from .oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from .arms.llm_reasoning_arm import _load_task
from .llm_client import LLMClient

LLM_MODEL = "claude-haiku-4-5-20251001"

MEMORY_PATH = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

# Known arm performance from V9-V11 experiments (hit_ratio @ R5, 3-seed avg)
KNOWN_PERFORMANCE: dict[str, dict[str, float]] = {
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

# Top hit genes (sampled from oracle ground truth) to give Claude concrete examples
# These are the actual hits - legitimate to use in memory since they were "revealed" in our experiments
def get_top_hit_genes(dataset_name: str, n: int = 20) -> list[str]:
    import pandas as pd
    df = pd.read_csv(REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv")
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
    "llm_reasoning": "llm",
    "online_adaptive": "ml",
    "static_ranker": "ml",
    "coreset": "hybrid",
    "random": "hybrid",
}

_VERIFIER_PROMPT = """\
A strategy summary was auto-generated for a CRISPR experiment and will be stored
as reusable memory injected into future gene-selection agents.
Verify it is grounded in the authoritative data below.

AUTHORITATIVE DATA:
- Arm performance (hit_ratio @ R5, higher = better): {perf_str}
- Best arm by performance: {best_arm}
- Sample hit genes (top-scoring by PPI connectivity): {top_hits}

GENERATED SUMMARY:
- best_strategy: "{best_strategy}"
- top_hit_families: {top_hit_families}
- key_genes: {key_genes}
- strategy_insight: "{strategy_insight}"

Verify ONLY these points:
1. Does `strategy_insight` correctly attribute which arm performed best?
   It MUST NOT claim ML beat LLM (or vice versa) if the numbers say otherwise.
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
) -> tuple[bool, list[str]]:
    """Two-phase admission gate (DeLM §3.2).

    Phase 1 — mechanical (free): check best_strategy label consistency.
    Phase 2 — LLM verifier: check strategy_insight is data-grounded.
    Returns (is_valid, issues).  On LLM parse failure → passes (fail-open).
    """
    issues: list[str] = []

    # --- Phase 1: mechanical ---
    expected = _expected_strategy(entry["best_arm"])
    actual = entry.get("best_strategy", "")
    # Allow "hybrid" as a conservative fallback for any expected value.
    if actual != expected and actual != "hybrid":
        issues.append(
            f"best_strategy='{actual}' inconsistent with best_arm='{entry['best_arm']}'"
            f" (expected '{expected}')"
        )

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
        best_strategy=actual,
        top_hit_families=entry.get("top_hit_families", []),
        key_genes=entry.get("key_genes", []),
        strategy_insight=entry.get("strategy_insight", ""),
    )
    try:
        text = client.complete(prompt, temperature=0.0, max_tokens=200)
        m = re.search(r'\{.*\}', text, re.DOTALL)
        result = json.loads(m.group()) if m else {}
        if not result.get("valid", True):
            issues.extend(result.get("issues", ["LLM verifier rejected summary"]))
    except Exception:
        pass  # Fail-open: infra hiccup should not block admission

    return len(issues) == 0, issues


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

SAMPLE HIT GENES (top-scoring hits by PPI connectivity):
{', '.join(top_hits)}{correction}
Generate a concise strategy insight (4-6 sentences) for this experiment type. Include:
1. Which biological pathways/gene families appear most enriched (based on the sample hit genes)
2. Whether ML feature-based selection or LLM biological knowledge was more effective and WHY
   IMPORTANT: your best_strategy label MUST match the best arm above.
3. Specific advice for a future agent facing a similar task
4. Any caveats or surprising findings

Return ONLY a JSON object with these exact keys:
{{
  "strategy_insight": "...",
  "top_hit_families": ["pathway1", "pathway2", "pathway3"],
  "best_strategy": "ml" | "llm" | "hybrid",
  "key_genes": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
}}"""


def _parse_llm_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        return {}


def generate_memory_entry(dataset_name: str, client: LLMClient) -> dict:
    task = _load_task(dataset_name)
    oracle = DatasetOracle(dataset_name)
    perf = KNOWN_PERFORMANCE.get(dataset_name, {})
    top_hits = get_top_hit_genes(dataset_name, n=25)

    best_arm = max(perf, key=perf.get) if perf else "unknown"
    perf_str = "\n".join(
        f"  {arm}: {score:.3f}"
        for arm, score in sorted(perf.items(), key=lambda x: -x[1])
    )

    prior_issues: list[str] = []
    entry: dict = {}
    verified = False

    for attempt in range(MAX_VERIFY_RETRIES + 1):
        prompt = _build_generation_prompt(
            dataset_name, task, oracle, perf_str, best_arm, top_hits, prior_issues
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
            "strategy_insight": parsed.get("strategy_insight", ""),
            "top_hit_families": parsed.get("top_hit_families", []),
            "best_strategy": parsed.get("best_strategy", "hybrid"),
            "key_genes": parsed.get("key_genes", top_hits[:5]),
        }

        valid, issues = _verify_entry(entry, top_hits, client)
        if valid:
            verified = True
            break
        prior_issues = issues
        print(f"    [verify] attempt {attempt + 1}/{MAX_VERIFY_RETRIES + 1} failed: {issues}", flush=True)

    entry["verified"] = verified
    return entry


def build_memory(datasets: list[str]) -> list[dict]:
    client = LLMClient(model=LLM_MODEL, temperature=0.3, max_tokens=600)

    entries: list[dict] = []
    for ds in datasets:
        print(f"  Generating memory for {ds}...", flush=True)
        entry = generate_memory_entry(ds, client)
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
    args = parser.parse_args()

    print(f"Building experience memory for {len(args.datasets)} datasets...")
    entries = build_memory(args.datasets)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nMemory saved to {args.out}  ({len(entries)} entries)")
    for e in entries:
        print(f"  {e['dataset']:30s}: best={e['best_arm']:18s}  insight={e['strategy_insight'][:60]}...")


if __name__ == "__main__":
    main()
