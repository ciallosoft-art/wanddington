"""
data.py — load the committed benchmark results into tidy frames.

Figures read frozen benchmark outputs under `workspace/results/sequential/` and
`workspace/results/router/`. The principal figures use `router/main_table_20seed.json`; supporting
diagnostic figures use their matched 5-seed files. Nothing here re-runs an experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "workspace" / "results" / "sequential"
# the external GeneDisco reference (docs/ml_engines_plan.md, P2-6): equal information, best single rule
GENEDISCO_ARM = "gdfb_achilles_bnn_coreset"
# The reported run of all six main-table arms (figures 1-3; analysis/sources.py); the legacy
# three_arm.json waddington_c stays the reference for the component-ablation figure.
from .sources import REPORTED_RUN as FINAL_WADD  # noqa: E402
GPT_RESULTS = REPO_ROOT / "workspace" / "results" / "gpt"

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
    # The two augmentations were re-run on GPT-5.6 Luna; the July files of the same name under
    # sequential/ are the Claude predecessor and must not be mixed into the GPT figure.
    # (An absolute path here wins over RESULTS in load_ablation's join.)
    "waddington_c_skills": (GPT_RESULTS / "skills_ablation.json", "+ skill library"),
    "waddington_c_enrich": (GPT_RESULTS / "enrich_gated.json", "+ runtime enrichment"),
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


def load_methods(final: bool = False) -> pd.DataFrame:
    """All six methods, tidy. (Coreset appears in two files; keep the baselines copy.)

    final=True swaps every arm present in the 20-seed run (FINAL_WADD) for the principal figures, so the
    figures match the 20-seed main table; arms absent from that file fall back to their 5-seed runs.
    """
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
    result = pd.concat(out, ignore_index=True)
    if final and FINAL_WADD.exists():
        fw = _runs(FINAL_WADD)
        arms20 = set(fw["arm"].unique())
        result = pd.concat([result[~result["arm"].isin(arms20)], fw], ignore_index=True)
    return result


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


def load_external() -> pd.DataFrame:
    """The two external prior methods on the full nine-screen benchmark, as tidy rows (round 5, seed 0)
    so they append to the method frame for the comparison figures (Fig.~2 bars, Fig.~3 heatmap):
      * ``bda``       — BioDiscoveryAgent's no-tools agent on GPT-5.6 Luna, our-oracle hit@R5 per screen.
      * ``genedisco`` — GeneDisco run with its own models and rules but our system's 0/1 feedback (equal
                        information), its strongest single rule (Achilles features, Bayesian MLP,
                        core-set); per-screen hit@R5 per round, mean over its five seeds. The earlier
                        port (sequential/genedisco_ported.json) broke ties by a row order that lists some
                        screens' hits first and is no longer used (analysis/order_audit.py).
    Both carry the caveats spelled out in the external-comparison tables; they are shown in neutral grey.
    """
    rows = []
    bda_p = REPO_ROOT / "workspace" / "results" / "bda_controlled" / "bda_gpt.json"
    if bda_p.exists():
        for ds, v in json.loads(bda_p.read_text()).get("results", {}).items():
            prr = v.get("hit_ratio_per_round") or [v["hit_at_r5"]]
            for i, hr in enumerate(prr):
                rows.append({"dataset": ds, "arm": "bda", "seed": 0, "round": i + 1,
                             "hit_ratio": float(hr), "cumulative_hits": float("nan"),
                             "total_hits": v["total_hits"], "auc": float("nan")})
    for gd_p in sorted((REPO_ROOT / "workspace" / "results" / "ml_engines").glob("gdfb_seeded_*.json")):
        if gd_p.name.endswith(".manifest.json"):
            continue
        gd = json.loads(gd_p.read_text())
        for ds, arms in gd.items():
            if not isinstance(arms, dict):
                continue
            seeds = arms.get(GENEDISCO_ARM)
            if seeds:
                nr = len(seeds[0]["hit_ratio_per_round"])
                for i in range(nr):
                    hr = sum(r["hit_ratio_per_round"][i] for r in seeds) / len(seeds)
                    rows.append({"dataset": ds, "arm": "genedisco", "seed": 0, "round": i + 1,
                                 "hit_ratio": float(hr), "cumulative_hits": float("nan"),
                                 "total_hits": float("nan"), "auc": float("nan")})
    return pd.DataFrame(rows)


def load_agent() -> pd.DataFrame | None:
    """The tool-using agent vs the pipeline, per dataset. Agent run and pipeline reference are both the
    reported GPT system (agent under gpt/agent_benchmark.json; pipeline from the GPT main table)."""
    p = REPO_ROOT / "workspace" / "results" / "gpt" / "agent_benchmark.json"
    if not p.exists():
        return None
    agent = json.loads(p.read_text())
    base = final_round(_runs(FINAL_WADD))
    pipe = base[base["arm"] == "waddington_c"].groupby("dataset")["hit_ratio"].mean()
    rows = []
    for ds, v in agent.items():
        if ds in pipe.index:
            rows.append({"dataset": ds, "agent": v["mean"], "pipeline": float(pipe[ds]),
                         "delta": v["mean"] - float(pipe[ds])})
    return pd.DataFrame(rows).sort_values("delta")
