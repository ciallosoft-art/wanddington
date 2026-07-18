"""
data.py — load the committed benchmark results into tidy frames.

Every figure reads from `workspace/results/sequential/*.json`, which are the frozen benchmark
outputs. Nothing here re-runs an experiment, so a figure can never disagree with the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "workspace" / "results" / "sequential"

# Which file each method's runs live in (baselines + the three-arm comparison).
METHOD_FILES = {
    "random": "baselines.json",
    "coreset": "baselines.json",
    "static_ranker": "baselines.json",
    "online_adaptive": "baselines.json",
    "llm_reasoning": "three_arm.json",
    "waddington_c": "three_arm.json",
}

ABLATIONS = {
    "waddington_c_no_memory": ("ablation_memory.json", "− cross-experiment memory"),
    "waddington_c_no_llm": ("ablation_llm.json", "− LLM reasoning"),
    "waddington_c_no_ml": ("ablation_ml.json", "− online ML"),
    "waddington_c_shuffled_names": ("ablation_shuffled.json", "gene names shuffled"),
    "waddington_c_feature_reasoning": ("ablation_feature_reasoning.json", "anonymized + structural features"),
    "waddington_c_pool_only": ("ablation_feature_reasoning.json", "anonymized + ML pool, no features"),
    "waddington_c_skills": ("skills_ablation.json", "+ skill library"),
    "waddington_c_enrich": ("enrich_gated.json", "+ runtime enrichment"),
}


def _runs(path: Path) -> pd.DataFrame:
    """One JSON → tidy rows (dataset, arm, seed, round, hit_ratio, …)."""
    data = json.loads(path.read_text())
    rows = []
    for dataset, arms in data.items():
        for arm, seeds in arms.items():
            for seed, r in enumerate(seeds):
                for i, hr in enumerate(r["hit_ratio_per_round"]):
                    rows.append({
                        "dataset": dataset, "arm": arm, "seed": seed, "round": i + 1,
                        "hit_ratio": hr,
                        "cumulative_hits": r["cumulative_hits"][i],
                        "total_hits": r["total_hits"],
                        "auc": r["auc_normalized"],
                    })
    return pd.DataFrame(rows)


def load_methods() -> pd.DataFrame:
    """All six methods, tidy. (Coreset appears in two files; keep the baselines copy.)"""
    frames = []
    for path in sorted({f for f in METHOD_FILES.values()}):
        p = RESULTS / path
        if p.exists():
            frames.append(_runs(p))
    df = pd.concat(frames, ignore_index=True)
    keep = [(m, METHOD_FILES[m]) for m in METHOD_FILES]
    out = []
    for arm, _file in keep:
        sub = df[df["arm"] == arm]
        if not sub.empty:
            # a method may appear in >1 file — take the first file that has it, consistently
            out.append(sub.drop_duplicates(["dataset", "seed", "round"]))
    return pd.concat(out, ignore_index=True)


def final_round(df: pd.DataFrame) -> pd.DataFrame:
    """hit_ratio at the last round — the headline metric (hit@R5)."""
    last = df["round"].max()
    return df[df["round"] == last]


def method_means(df: pd.DataFrame) -> pd.Series:
    """arm → mean hit@R5 over datasets (each dataset averaged over its seeds first)."""
    fr = final_round(df)
    per_ds = fr.groupby(["arm", "dataset"])["hit_ratio"].mean()
    return per_ds.groupby("arm").mean().sort_values(ascending=False)


def load_ablation(arm: str) -> pd.DataFrame | None:
    """Paired frame for one ablation (waddington_c vs the variant), or None if absent."""
    fname, _label = ABLATIONS[arm]
    p = RESULTS / fname
    if not p.exists():
        return None
    df = _runs(p)
    return df[df["arm"].isin(["waddington_c", arm])]


def ablation_deltas() -> pd.DataFrame:
    """Δ hit@R5 (variant − waddington_c), paired within each file (same seeds/datasets)."""
    rows = []
    for arm, (_f, label) in ABLATIONS.items():
        df = load_ablation(arm)
        if df is None or df["arm"].nunique() < 2:
            continue
        fr = final_round(df)
        per = fr.groupby(["arm", "dataset"])["hit_ratio"].mean().groupby("arm").mean()
        rows.append({"arm": arm, "label": label,
                     "base": per.get("waddington_c"), "variant": per.get(arm),
                     "delta": per.get(arm) - per.get("waddington_c")})
    return pd.DataFrame(rows).sort_values("delta")


def load_agent() -> pd.DataFrame | None:
    """The tool-using agent vs the pipeline, per dataset (from agent_benchmark.json)."""
    p = RESULTS / "agent_benchmark.json"
    if not p.exists():
        return None
    agent = json.loads(p.read_text())
    base = final_round(_runs(RESULTS / "three_arm.json"))
    pipe = base[base["arm"] == "waddington_c"].groupby("dataset")["hit_ratio"].mean()
    rows = []
    for ds, v in agent.items():
        if ds in pipe.index:
            rows.append({"dataset": ds, "agent": v["mean"], "pipeline": float(pipe[ds]),
                         "delta": v["mean"] - float(pipe[ds])})
    return pd.DataFrame(rows).sort_values("delta")
