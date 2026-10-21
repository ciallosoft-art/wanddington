"""
coreset_probe.py — what the diversity baseline A (CoresetArm) actually selects, and what its 0.134 comes from.
(2026-10-21; local, deterministic, no LLM calls.)

Arm A is greedy k-center (farthest-point) over the nine V7 features (arms/coreset_arm.py): start at the gene
farthest from the centroid, then repeatedly add the gene whose distance to its nearest already-selected gene
is largest. It reads no labels. This module measures:

  reproduce        the arm as reported (hit@R5 per screen; must match Table main, mean 0.134)
  row order        the arm with the target rows in five random orders (argmax ties would otherwise follow the
                   stored order, which lists three screens' hits first)
  no anchors       the same k-center on the six gene-intrinsic features (the three anchor-relative features
                   are privileged: 51% of anchors are own-screen hits)
  label-free controls at the same budget (5 x batch), no diversity: the top genes by STRING degree, and the
                   top genes by distance from the centroid of the six intrinsic features (ties broken in
                   twenty random orders); and, paired against the arm, the single-feature rankings already
                   reported post hoc (analysis/ml_engines.py label_free_baselines, ml_engines/summary.json)
  geometry         where the picks sit: their percentile on each feature, their share in each decile of
                   distance-from-centroid (the arm's own nine-feature space) against the pool's 10%, the hit
                   rate per decile, and the hit rate of the picks inside vs outside the budget-many most
                   outlying genes

Run:  python -m waddington_select.analysis.coreset_probe          -> workspace/results/coreset_probe.json
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

from ..arms import coreset_arm as CA
from ..oracle import DatasetOracle
from ..router_protocol import BENCH
from ..sequential_runner import run_arm_on_dataset
from ..tables import TRAINING_DATA_CSV
from .final_system_stats import _paired

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "workspace" / "results" / "coreset_probe.json"
ML_ENGINES_SUMMARY = REPO / "workspace" / "results" / "ml_engines" / "summary.json"
FULL = list(CA.FEATURE_COLS)
ANCHOR = ["g1_ppi_score", "archs4_coexpr", "kegg_overlap"]
INTRINSIC = [c for c in FULL if c not in ANCHOR]
N_ORDERS = 5          # row orders for the row-order check
N_TIES = 20           # tie-break orders for the ranking controls


class _ShuffledCoreset(CA.CoresetArm):
    """The same arm with the target rows in a random, label-free order."""

    perm_seed = 0

    def _load_features(self, dataset_name):
        genes, X = super()._load_features(dataset_name)
        idx = np.random.default_rng(self.perm_seed).permutation(len(genes))
        return [genes[i] for i in idx], X[idx]


def _coreset(ds: str, cols: list[str], perm_seed: int | None = None):
    CA.FEATURE_COLS = cols
    try:
        batch = DatasetOracle(ds).batch_size
        if perm_seed is None:
            arm = CA.CoresetArm(ds, batch)
        else:
            _ShuffledCoreset.perm_seed = perm_seed
            arm = _ShuffledCoreset(ds, batch)
        return run_arm_on_dataset(arm, ds, n_rounds=5)
    finally:
        CA.FEATURE_COLS = FULL


def _top_k(ds: str, genes: np.ndarray, key: np.ndarray) -> float:
    """hit@R5 of taking the budget-many genes with the largest key, ties in N_TIES random orders."""
    o = DatasetOracle(ds)
    k = 5 * o.batch_size
    vals = []
    for sd in range(N_TIES):
        perm = np.random.default_rng(1000 + sd).permutation(len(genes))
        order = genes[perm][np.argsort(-key[perm], kind="stable")]
        vals.append(sum(o.reveal(list(order[:k])).values()) / o.total_hits)
    return st.mean(vals)


def _outlyingness(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - X.mean(axis=0), axis=1)


def probe_screen(ds: str, table: pd.DataFrame) -> dict:
    sub = table[table["dataset"] == ds].reset_index(drop=True)
    genes = sub["gene"].to_numpy()
    o = DatasetOracle(ds)
    budget = 5 * o.batch_size
    is_hit = np.array([o.reveal([g])[g] for g in genes])

    rep = _coreset(ds, FULL)
    picks_by_round = rep.selected_genes_per_round
    picks = [g for rnd in picks_by_round for g in rnd]
    pick_mask = np.isin(genes, picks)

    # geometry, in the arm's own nine-feature space
    X9 = sub[FULL].fillna(0.0).to_numpy(dtype=float)
    out9 = _outlyingness(X9)
    pct9 = pd.Series(out9).rank(pct=True).to_numpy()
    decile = np.minimum((pct9 * 10).astype(int), 9)          # 0 = most central ... 9 = most outlying 10%
    most_outlying = np.zeros(len(genes), dtype=bool)
    most_outlying[np.argsort(-out9, kind="stable")[:budget]] = True
    inside, outside = pick_mask & most_outlying, pick_mask & ~most_outlying
    by_round = [float(pct9[np.isin(genes, rnd)].mean()) for rnd in picks_by_round]

    shuffled = [_coreset(ds, FULL, perm_seed=k).hit_ratio_per_round[-1] for k in range(N_ORDERS)]
    X6 = sub[INTRINSIC].fillna(0.0).to_numpy(dtype=float)
    return {
        "n_pool": len(sub), "budget": budget, "pool_hit_rate": float(is_hit.mean()),
        "coreset": rep.hit_ratio_per_round[-1],
        "coreset_row_shuffled": st.mean(shuffled), "coreset_row_shuffled_range": max(shuffled) - min(shuffled),
        "coreset_no_anchor": _coreset(ds, INTRINSIC).hit_ratio_per_round[-1],
        "top_degree": _top_k(ds, genes, sub["string_degree_norm"].fillna(0.0).to_numpy(dtype=float)),
        "top_outlying_intrinsic": _top_k(ds, genes, _outlyingness(X6)),
        "pick_feature_percentile": {c: float(sub[c].rank(pct=True)[pick_mask].mean()) for c in FULL},
        "pick_outlying_percentile_by_round": by_round,
        "picks_per_outlying_decile": [float((decile[pick_mask] == d).mean()) for d in range(10)],
        "hit_rate_per_outlying_decile": [float(is_hit[decile == d].mean()) for d in range(10)],
        "picks_in_most_outlying_budget": float(inside.sum() / pick_mask.sum()),
        "hit_rate_picks_inside_most_outlying": float(is_hit[inside].mean()) if inside.any() else None,
        "hit_rate_picks_outside_most_outlying": float(is_hit[outside].mean()) if outside.any() else None,
        "hit_rate_most_outlying_not_picked": float(is_hit[most_outlying & ~pick_mask].mean())
        if (most_outlying & ~pick_mask).any() else None,
        "anchor_signal_share_pool": float((sub[ANCHOR].fillna(0) > 0).any(axis=1).mean()),
        "anchor_signal_share_picks": float((sub.loc[pick_mask, ANCHOR].fillna(0) > 0).any(axis=1).mean()),
        "feature_quantiles": {c: {q: float(sub[c].quantile(q)) for q in (0.5, 0.9, 0.99)} | {"max": float(sub[c].max())}
                              for c in INTRINSIC},
        "first_picks": [{"gene": g, "hit": bool(o.reveal([g])[g]),
                         **{c: float(sub.loc[sub["gene"] == g, c].iloc[0]) for c in ("string_degree_norm",
                            "hub_score_norm", "kegg_pathway_count_norm", "reactome_pathway_count_norm")}}
                        for g in picks_by_round[0][:12]],
    }


def _wtl(d: list[float], tol: float = 0.001) -> str:
    return f"{sum(x > tol for x in d)}/{sum(abs(x) <= tol for x in d)}/{sum(x < -tol for x in d)}"


def main() -> None:
    table = pd.read_csv(TRAINING_DATA_CSV)
    table["gene"] = table["gene"].str.strip().str.upper()
    per = {}
    for ds in BENCH:
        per[ds] = probe_screen(ds, table)
        r = per[ds]
        print(f"{ds:24s} coreset {r['coreset']:.3f}  shuffled {r['coreset_row_shuffled']:.3f}  "
              f"no-anchor {r['coreset_no_anchor']:.3f}  top-degree {r['top_degree']:.3f}  "
              f"top-outlying {r['top_outlying_intrinsic']:.3f}", flush=True)
    mean = lambda k: st.mean(per[d][k] for d in BENCH)
    summary = {k: mean(k) for k in ("coreset", "coreset_row_shuffled", "coreset_no_anchor", "top_degree",
                                     "top_outlying_intrinsic", "picks_in_most_outlying_budget",
                                     "anchor_signal_share_pool", "anchor_signal_share_picks")}
    for k in ("top_degree", "top_outlying_intrinsic", "coreset_no_anchor"):
        d = [per[ds][k] - per[ds]["coreset"] for ds in BENCH]
        summary[f"{k}_minus_coreset"] = st.mean(d)
        summary[f"{k}_wtl_vs_coreset"] = _wtl(d)
    coreset = {ds: per[ds]["coreset"] for ds in BENCH}
    paired = {k: _paired({ds: per[ds][k] for ds in BENCH}, coreset)
              for k in ("top_degree", "top_outlying_intrinsic", "coreset_no_anchor")}
    single = json.loads(ML_ENGINES_SUMMARY.read_text())["label_free_post_hoc"]
    for f, v in single.items():
        paired[f"single_feature:{f}"] = {"mean": v["mean"], **_paired(v["per_screen"], coreset)}
    summary["paired_vs_coreset"] = paired
    summary["row_order_max_range"] = max(per[d]["coreset_row_shuffled_range"] for d in BENCH)
    summary["pick_feature_percentile"] = {c: st.mean(per[d]["pick_feature_percentile"][c] for d in BENCH) for c in FULL}
    summary["picks_per_outlying_decile"] = [st.mean(per[d]["picks_per_outlying_decile"][i] for d in BENCH)
                                            for i in range(10)]
    summary["hit_rate_per_outlying_decile_over_pool_rate"] = [
        st.mean(per[d]["hit_rate_per_outlying_decile"][i] / per[d]["pool_hit_rate"] for d in BENCH) for i in range(10)]
    summary["pick_outlying_percentile_by_round"] = [st.mean(per[d]["pick_outlying_percentile_by_round"][i]
                                                            for d in BENCH) for i in range(5)]
    both = [d for d in BENCH if per[d]["hit_rate_picks_outside_most_outlying"] is not None
            and per[d]["hit_rate_picks_inside_most_outlying"] is not None]
    summary["picks_hit_rate_inside_over_pool"] = st.mean(per[d]["hit_rate_picks_inside_most_outlying"]
                                                         / per[d]["pool_hit_rate"] for d in both)
    summary["picks_hit_rate_outside_over_pool"] = st.mean(per[d]["hit_rate_picks_outside_most_outlying"]
                                                          / per[d]["pool_hit_rate"] for d in both)
    OUT.write_text(json.dumps({"summary": summary, "per_screen": per}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))}, indent=1))
    print("pick percentile:", {k: round(v, 3) for k, v in summary["pick_feature_percentile"].items()})
    print("picks per outlying decile (central->outlying):", [round(x, 3) for x in summary["picks_per_outlying_decile"]])
    print("hit rate per decile / pool:", [round(x, 2) for x in summary["hit_rate_per_outlying_decile_over_pool_rate"]])
    print("pick outlying percentile by round:", [round(x, 3) for x in summary["pick_outlying_percentile_by_round"]])
    for k, v in paired.items():
        print(f"vs coreset  {k:42s} {v.get('mean', summary.get(k, float('nan'))):.4f}  "
              f"{v['delta']:+.4f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['wins']}/{v['ties']}/{v['losses']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
