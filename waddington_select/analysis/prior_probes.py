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

from ..arms.base import label_free_order
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


TIE_SEEDS = 20  # label-free tie-break orders averaged over in the hit-frequency ranking


def _hit_frequency(df, target):
    """Rank the target's genes by how often each is a hit in the OTHER screens, take the top 5*B.

    Thousands of genes share a count (mostly 1 or 2), so the cut-off falls inside a tie. The table stores
    some screens' hits first (Scharenberg22, K562-Essential, K562-GWPS; IL2 partly), so breaking ties by
    stored row order reads labels: it gave 0.243 over the nine screens, against 0.175 with label-free ties
    (2026-10-16). Ties are now broken in TIE_SEEDS label-free orders and the hit ratio averaged."""
    bench = list(BENCHMARK_DATASETS)
    hitset = {d: set(df[(df.dataset == d) & (df.label == 1)].gene) for d in bench}
    sub = df[df.dataset == target]
    pool, total, bs = sub.gene.tolist(), sub.label.sum(), BATCH_SIZES[target]
    if not total:
        return 0.0
    freq = {g: sum(g in hitset[o] for o in bench if o != target) for g in pool}
    vals = []
    for seed in range(TIE_SEEDS):
        order = [pool[i] for i in label_free_order(pool, seed)]
        top = sorted(order, key=lambda g: -freq[g])[:5 * bs]
        vals.append(sum(g in hitset[target] for g in top) / total)
    return float(np.mean(vals))


def _anchor_leakage(df, bench, hitset) -> dict:
    """How many of each screen's anchor genes (used to build the anchor-relative features) are hits in
    that same screen — a privileged-information audit. Anchors live in the benchmark feature-prep table
    (gene_ranker.DATASET_ANCHORS), not the deployment registry."""
    import sys
    sys.path.insert(0, str(REPO / "workspace" / "evaluation"))
    try:
        from gene_ranker import DATASET_ANCHORS  # type: ignore
    except Exception:
        return {"error": "gene_ranker.DATASET_ANCHORS unavailable"}
    per = {}
    ta = th = 0
    for d in bench:
        anch = [a.upper() for a in DATASET_ANCHORS.get(d, [])]
        ah = [a for a in anch if a in hitset[d]]
        per[d] = {"n_anchors": len(anch), "n_anchor_hits": len(ah), "anchor_hits": ah}
        ta += len(anch)
        th += len(ah)
    return {"per_screen": per, "total_anchors": ta, "total_anchor_hits": th,
            "frac_anchors_that_are_hits": round(th / ta, 3) if ta else 0.0}


def permutation_importance(df, feats=None, n_shuffles: int = 8, seed: int = 0) -> dict:
    """Per-feature permutation importance for the LOO prior, in hit@R5 units. For each screen the LightGBM
    is trained on the other screens (as in ``_loo_hit_at_r5``); then each feature is permuted across the
    target genes and we measure how far hit@R5 falls (averaged over shuffles, then screens). Reported over
    the no-anchor feature set the paper ships, so it corroborates the SHAP attributions on the same
    features. An independent method (SHAP attributes the score; this destroys a feature and measures the
    ranking loss). Deterministic (fixed LightGBM + permutation seed), no LLM -> model-independent."""
    if feats is None:
        feats = INTRINSIC + DEPMAP  # no-anchor reported configuration
    rng = np.random.default_rng(seed)
    drops = {f: [] for f in feats}
    base_all = []
    for target in BENCHMARK_DATASETS:
        sub = df[df["dataset"] == target]
        pool, y = sub["gene"].tolist(), dict(zip(sub["gene"], sub["label"]))
        total, bs = int(sub["label"].sum()), BATCH_SIZES[target]
        train = df[df["dataset"] != target]
        pos, neg = (train["label"] == 1).sum(), (train["label"] == 0).sum()
        m = lgb.LGBMClassifier(**LGBM, scale_pos_weight=neg / max(pos, 1))
        m.fit(train[feats].values, train["label"].values)
        X = sub[feats].values.astype(float)

        def hit(matrix):
            order = np.argsort(-m.predict_proba(matrix)[:, 1])
            top = [pool[i] for i in order[:5 * bs]]
            return (sum(y[g] for g in top) / total) if total else 0.0

        base = hit(X)
        base_all.append(base)
        for j, f in enumerate(feats):
            ds = []
            for _ in range(n_shuffles):
                perturbed = X.copy()
                perturbed[:, j] = rng.permutation(perturbed[:, j])
                ds.append(base - hit(perturbed))
            drops[f].append(float(np.mean(ds)))
    return {
        "baseline_hit_at_r5": float(np.mean(base_all)),
        "feature_set": "intrinsic+depmap (no anchor)",
        "n_shuffles": n_shuffles,
        "importance": {f: float(np.mean(v)) for f, v in drops.items()},
    }


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
            "intrinsic_depmap_no_anchor": mean([_loo_hit_at_r5(df, d, INTRINSIC + DEPMAP) for d in bench]),
            "all": mean([_loo_hit_at_r5(df, d, ALLF) for d in bench]),
        },
        "anchor_leakage": _anchor_leakage(df, bench, hitset),
        "permutation_importance": permutation_importance(df),
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
    pi = s["permutation_importance"]
    print(f"\npermutation importance (no-anchor, hit@R5 drop; baseline {pi['baseline_hit_at_r5']:.3f}):")
    for f, v in sorted(pi["importance"].items(), key=lambda kv: -kv[1]):
        print(f"  {f:28s} {v:+.4f}")
    print(f"saved -> {OUT}")
