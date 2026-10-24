"""
feature_families.py — the prior's feature families, regrouped by where each feature comes from (2026-10-23; no LLM).

prior_probes.py splits the thirteen features into intrinsic (6) / anchor-relative (3) / DepMap (4). Two of its
"intrinsic" six, hub_score_norm and ppi_score_sum, are counted over a STRING query set that holds the nine
screens' anchors and hits (analysis/hub_leak.py), so they are regrouped here by source:

  global   string_degree_norm, pli_score, kegg_pathway_count_norm, reactome_pathway_count_norm
           (genome-wide STRING degree, gnomAD constraint, per-gene pathway counts: no anchor, no label)
  pooled   hub_score_norm, ppi_score_sum (count / score sum over the 315-gene STRING query set: the nine
           screens' anchors plus 251 genes half of which are benchmark hits; same value in every screen)
  depmap   the four DepMap knockout-essentiality features
  anchor   g1_ppi_score, archs4_coexpr, kegg_overlap (proximity to the target screen's own anchors)

Same convention as prior_probes (a uniform LOO LightGBM, DepMap on every screen, label-free tie order), so the
numbers sit next to its feature-family probe. Paired, screen-clustered comparisons are against global+depmap,
the prior built only from features with no anchor and no label. Also saved: that clean prior against the
plain cross-screen hit-frequency ranking (prior_probes.json, label-free ties), and the within-screen Spearman
correlation of the thirteen features, averaged over the nine screens.

Run:  python -m waddington_select.analysis.feature_families      -> workspace/results/feature_families.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..oracle import BENCHMARK_DATASETS
from .final_system_stats import _paired
from .prior_probes import _load, _loo_hit_at_r5

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "workspace" / "results" / "feature_families.json"
PRIOR_PROBES = REPO / "workspace" / "results" / "prior_probes.json"
GLOBAL = ["string_degree_norm", "pli_score", "kegg_pathway_count_norm", "reactome_pathway_count_norm"]
POOLED = ["hub_score_norm", "ppi_score_sum"]
DEPMAP = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm", "depmap_K562_norm"]
ANCHOR = ["g1_ppi_score", "archs4_coexpr", "kegg_overlap"]
SETS = {
    "global": GLOBAL, "pooled": POOLED, "depmap": DEPMAP, "anchor": ANCHOR,
    "global+pooled (prior_probes 'intrinsic')": GLOBAL + POOLED,
    "global+depmap (clean)": GLOBAL + DEPMAP,
    "global+depmap+pooled": GLOBAL + DEPMAP + POOLED,
    "global+depmap+anchor": GLOBAL + DEPMAP + ANCHOR,
    "all thirteen": GLOBAL + POOLED + DEPMAP + ANCHOR,
}
CLEAN = "global+depmap (clean)"


def correlations(df) -> dict:
    feats = GLOBAL + POOLED + DEPMAP + ANCHOR
    mats = [g[feats].rank().corr() for _, g in df.groupby("dataset")]
    mean = sum(mats) / len(mats)
    return {"features": feats, "spearman_within_screen_mean": mean.round(4).values.tolist()}


def main() -> None:
    df = _load()
    bench = [d for d in BENCHMARK_DATASETS if d in set(df["dataset"])]
    per = {name: {d: _loo_hit_at_r5(df, d, feats) for d in bench} for name, feats in SETS.items()}
    out = {
        "groups": {"global": GLOBAL, "pooled": POOLED, "depmap": DEPMAP, "anchor": ANCHOR},
        "means": {k: float(np.mean(list(v.values()))) for k, v in per.items()},
        "per_screen": per,
        "paired_vs_clean": {k: _paired(v, per[CLEAN]) for k, v in per.items() if k != CLEAN},
        "correlation": correlations(df),
    }
    hit_freq = {d: v["hit_frequency"] for d, v in json.loads(PRIOR_PROBES.read_text())["per_screen"].items()}
    out["clean_minus_hit_frequency"] = _paired(per[CLEAN], hit_freq)
    OUT.write_text(json.dumps(out, indent=1))
    p = out["clean_minus_hit_frequency"]
    print(f"clean prior minus hit-frequency ranking {p['delta']:+.4f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] "
          f"{p['wins']}/{p['ties']}/{p['losses']}")
    for k, v in out["means"].items():
        p = out["paired_vs_clean"].get(k)
        tail = "" if p is None else (f"  vs clean {p['delta']:+.4f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] "
                                     f"{p['wins']}/{p['ties']}/{p['losses']}")
        print(f"{k:42s} {v:.4f}{tail}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
