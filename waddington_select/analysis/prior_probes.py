"""
prior_probes.py — what does the cross-experiment LOO prior actually learn?

A reviewer asked whether the LOO-LightGBM's gain is genuine phenotype-specific transfer or something
more trivial (sibling-screen leakage, generic essentiality, plain hit-recurrence). These probes answer
that, all on a single deterministic LightGBM (no LLM), so they run in seconds:

  1. sibling-exclusion  — retrain with the target's same-study sibling ALSO removed from training.
  2. feature-family     — the prior restricted to intrinsic / anchor-relative / DepMap subsets.
  3. hit-frequency      — a trivial baseline: rank a gene by how often it is a hit in the OTHER screens.
  4. novel-hit rate     — what fraction of a screen's hits are never a hit in any other screen.

Baseline convention: a UNIFORM all-feature LOO (no per-screen feature routing), so the feature-family
ablation is clean; this scores slightly above the routed LOO used in the main tables (it keeps DepMap
on every screen). The deltas, not the absolute baseline, are the point.

    python -m waddington_select.analysis prior_probes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from ..oracle import BATCH_SIZES, BENCHMARK_DATASETS

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "workspace" / "evaluation" / "lgbm_training_data_v3.csv"
OUT = REPO / "workspace" / "results" / "prior_probes.json"

LGBM = dict(objective="binary", metric="auc", learning_rate=0.05, n_estimators=300,
            num_leaves=31, verbose=-1, n_jobs=-1, random_state=42)
INTRINSIC = ["hub_score_norm", "ppi_score_sum", "pli_score", "string_degree_norm",
             "kegg_pathway_count_norm", "reactome_pathway_count_norm"]
ANCHOR = ["g1_ppi_score", "archs4_coexpr", "kegg_overlap"]
DEPMAP = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm", "depmap_K562_norm"]
ALLF = INTRINSIC + ANCHOR + DEPMAP
# same-study sibling screens (same paper / platform / raw source)
SIBLING = {"IFNG": ["IL2"], "IL2": ["IFNG"],
           "Sanchez21": ["Sanchez21_down"], "Sanchez21_down": ["Sanchez21"],
           "Replogle_K562_essential": ["Replogle_K562_gwps"],
           "Replogle_K562_gwps": ["Replogle_K562_essential"]}


def _load():
    df = pd.read_csv(CSV)
    df["gene"] = df["gene"].str.strip().str.upper()
    return df


def _loo_hit_at_r5(df, target, feats, drop_from_train=()):
    sub = df[df["dataset"] == target]
    pool, y = sub["gene"].tolist(), dict(zip(sub["gene"], sub["label"]))
    total, bs = int(sub["label"].sum()), BATCH_SIZES[target]
    train = df[(df["dataset"] != target) & (~df["dataset"].isin(drop_from_train))]
    pos, neg = (train["label"] == 1).sum(), (train["label"] == 0).sum()
    m = lgb.LGBMClassifier(**LGBM, scale_pos_weight=neg / max(pos, 1))
    m.fit(train[feats].values, train["label"].values)
    order = np.argsort(-m.predict_proba(sub[feats].values)[:, 1])
    top = [pool[i] for i in order[:5 * bs]]
    return (sum(y[g] for g in top) / total) if total else 0.0


def _hit_frequency(df, target):
    bench = list(BENCHMARK_DATASETS)
    hitset = {d: set(df[(df.dataset == d) & (df.label == 1)].gene) for d in bench}
    sub = df[df.dataset == target]
    pool, total, bs = sub.gene.tolist(), sub.label.sum(), BATCH_SIZES[target]
    freq = {g: sum(g in hitset[o] for o in bench if o != target) for g in pool}
    top = sorted(pool, key=lambda g: -freq[g])[:5 * bs]
    return (sum(g in hitset[target] for g in top) / total) if total else 0.0


def run() -> dict:
    df = _load()
    bench = list(BENCHMARK_DATASETS)
    hitset = {d: set(df[(df.dataset == d) & (df.label == 1)].gene) for d in bench}
    mean = lambda xs: float(np.mean(xs))

    res: dict = {"per_screen": {}, "summary": {}}
    for d in bench:
        seen = set().union(*[hitset[o] for o in bench if o != d])
        res["per_screen"][d] = {
            "loo_all": _loo_hit_at_r5(df, d, ALLF),
            "loo_no_sibling": _loo_hit_at_r5(df, d, ALLF, SIBLING.get(d, [])),
            "hit_frequency": _hit_frequency(df, d),
            "novel_hit_frac": round(len(hitset[d] - seen) / max(len(hitset[d]), 1), 3),
            "has_sibling": d in SIBLING,
        }
    ps = res["per_screen"]
    sib = [d for d in bench if d in SIBLING]
    res["summary"] = {
        "loo_all_avg": mean([ps[d]["loo_all"] for d in bench]),
        "loo_no_sibling_avg": mean([ps[d]["loo_no_sibling"] for d in bench]),
        "sibling_pairs_normal": mean([ps[d]["loo_all"] for d in sib]),
        "sibling_pairs_excluded": mean([ps[d]["loo_no_sibling"] for d in sib]),
        "hit_frequency_avg": mean([ps[d]["hit_frequency"] for d in bench]),
        "feature_family": {
            "intrinsic": mean([_loo_hit_at_r5(df, d, INTRINSIC) for d in bench]),
            "anchor": mean([_loo_hit_at_r5(df, d, ANCHOR) for d in bench]),
            "depmap": mean([_loo_hit_at_r5(df, d, DEPMAP) for d in bench]),
            "intrinsic_anchor_no_depmap": mean([_loo_hit_at_r5(df, d, INTRINSIC + ANCHOR) for d in bench]),
            "all": mean([_loo_hit_at_r5(df, d, ALLF) for d in bench]),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    s = r["summary"]
    print(f"LOO (all feats) avg              {s['loo_all_avg']:.3f}")
    print(f"LOO, siblings excluded avg       {s['loo_no_sibling_avg']:.3f}  (paired: "
          f"{s['sibling_pairs_normal']:.3f} -> {s['sibling_pairs_excluded']:.3f})")
    print(f"hit-frequency baseline avg       {s['hit_frequency_avg']:.3f}")
    print("feature family:", {k: round(v, 3) for k, v in s["feature_family"].items()})
    print(f"saved -> {OUT}")
