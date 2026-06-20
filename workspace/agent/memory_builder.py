"""
memory_builder.py — Cross-experiment memory generation for WaddingtonArm.

For each BDA dataset, generates a strategy insight summary using Claude,
capturing what gene families/pathways are hits and which approach (ML vs LLM)
works best. Stored as experience_memory.json for LOO use by WaddingtonArm.

Usage:
    conda run -n waddington-bio python3 workspace/agent/memory_builder.py
    conda run -n waddington-bio python3 workspace/agent/memory_builder.py --datasets IFNG IL2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workspace" / "agent"))

from oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from arms.llm_reasoning_arm import _load_task, _load_auth_token

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


def generate_memory_entry(dataset_name: str, client: anthropic.Anthropic) -> dict:
    task = _load_task(dataset_name)
    oracle = DatasetOracle(dataset_name)
    perf = KNOWN_PERFORMANCE.get(dataset_name, {})
    top_hits = get_top_hit_genes(dataset_name, n=25)

    best_arm = max(perf, key=perf.get) if perf else "unknown"
    perf_str = "\n".join(f"  {arm}: {score:.3f}" for arm, score in sorted(perf.items(), key=lambda x: -x[1]))

    prompt = f"""You are analyzing results from a completed CRISPR perturbation screen experiment.
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
{', '.join(top_hits)}

Generate a concise strategy insight (4-6 sentences) for this experiment type. Include:
1. Which biological pathways/gene families appear most enriched (based on the sample hit genes)
2. Whether ML feature-based selection or LLM biological knowledge was more effective and WHY
3. Specific advice for a future agent facing a similar task
4. Any caveats or surprising findings

Return ONLY a JSON object with these exact keys:
{{
  "strategy_insight": "...",
  "top_hit_families": ["pathway1", "pathway2", "pathway3"],
  "best_strategy": "ml" | "llm" | "hybrid",
  "key_genes": ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"]
}}"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Extract JSON block if wrapped in markdown
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        parsed = json.loads(m.group()) if m else {}

    return {
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


def build_memory(datasets: list[str]) -> list[dict]:
    token = _load_auth_token()
    client = anthropic.Anthropic(auth_token=token)

    entries: list[dict] = []
    for ds in datasets:
        print(f"  Generating memory for {ds}...", flush=True)
        entry = generate_memory_entry(ds, client)
        entries.append(entry)
        print(f"    best_arm={entry['best_arm']}  families={entry['top_hit_families'][:3]}", flush=True)

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
