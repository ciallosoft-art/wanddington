"""
label_free_vs_bda.py — label-free connectivity rankings against the BioDiscoveryAgent run (2026-10-21; no LLM).

The post-hoc single-feature rankings (ml_engines.label_free_baselines: walk down one gene feature, top 5B,
ties in a label-free order) sit at or above BDA's 0.146 on the headline tier. The comparison has a stated
asymmetry in our favour (Method, Relationship to BioDiscoveryAgent): our arms, and these rankings, choose
from the non-essential pool, while BDA's agent proposes from the whole genome and every core-essential (or
out-of-pool) name it gives is a wasted slot. So each ranking is scored two ways, paired by screen against
BDA's GPT-5.6 Luna run (bda_controlled/bda_gpt.json, one run per screen):

  ne_pool     candidates = the non-essential pool (as the paper's rankings, and our arms)
  full_pool   candidates = the full pool (core-essential genes restored, tables.tier_tables("full_pool")),
              scored exactly as BDA is: hits counted against the non-essential hit set, so a
              core-essential pick is a wasted slot here too

and BDA's own waste is reported, as shares of its 5B budget: picks that are CEGv2 genes, and picks that
never landed in the pool. bda_gpt.json stores only the in-pool names (on K562-Essential 64 of 160), so the
second share is (5B - stored) / 5B and cannot say which of those lost names were core-essential.

Run:  python -m waddington_select.analysis.label_free_vs_bda        -> workspace/results/label_free_vs_bda.json
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pandas as pd

from ..arms.base import label_free_order
from ..oracle import BATCH_SIZES
from ..router_protocol import BENCH
from ..tables import TIER_FULL_POOL, core_essential_genes, tier_tables
from .final_system_stats import _paired

REPO = Path(__file__).resolve().parents[2]
NE_V3 = REPO / "workspace" / "evaluation" / "lgbm_training_data_v3.csv"   # as ml_engines.label_free_baselines
BDA = REPO / "workspace" / "results" / "bda_controlled" / "bda_gpt.json"
OUT = REPO / "workspace" / "results" / "label_free_vs_bda.json"
FEATURES = ["hub_score_norm", "ppi_score_sum", "string_degree_norm", "depmap_frac_ess"]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["gene"] = df["gene"].str.strip().str.upper()
    return df


def _ranking(cands: pd.DataFrame, feature: str, budget: int, ne_hits: set[str]) -> float:
    tf = cands.iloc[label_free_order(cands["gene"].tolist(), 0)]
    top = tf.sort_values(feature, ascending=False, kind="stable").head(budget)["gene"]
    return sum(g in ne_hits for g in top) / len(ne_hits)


def main() -> None:
    ne, full = _load(NE_V3), _load(tier_tables(TIER_FULL_POOL)[1])
    bda = json.loads(BDA.read_text())["results"]
    ceg = core_essential_genes()
    out = {"bda": {}, "rankings": {}}
    for ds in BENCH:
        ne_ds = ne[ne["dataset"] == ds]
        pool = set(ne_ds["gene"])
        picks = [g.strip().upper() for g in bda[ds]["selected_genes"]]
        budget = 5 * BATCH_SIZES[ds]
        stray = [g for g in picks if g not in pool and g not in ceg]
        assert not stray, (ds, stray[:5])   # stored picks are in-pool or core-essential
        out["bda"][ds] = {"hit_at_r5": bda[ds]["hit_at_r5"], "budget": budget, "n_stored": len(picks),
                          "share_ceg": sum(g in ceg for g in picks) / budget,
                          "share_not_in_pool": (budget - len(picks)) / budget}
    bda_ps = {ds: out["bda"][ds]["hit_at_r5"] for ds in BENCH}
    for f in FEATURES:
        r = {"ne_pool": {}, "full_pool": {}}
        for ds in BENCH:
            ne_ds = ne[ne["dataset"] == ds]
            ne_hits = set(ne_ds.loc[ne_ds["label"] == 1, "gene"])
            budget = 5 * BATCH_SIZES[ds]
            r["ne_pool"][ds] = _ranking(ne_ds, f, budget, ne_hits)
            r["full_pool"][ds] = _ranking(full[full["dataset"] == ds], f, budget, ne_hits)
        out["rankings"][f] = {k: {"mean": st.mean(v.values()), "per_screen": v, "vs_bda": _paired(v, bda_ps)}
                              for k, v in r.items()}
    out["summary"] = {
        "bda_mean": st.mean(bda_ps.values()),
        "bda_share_ceg_mean": st.mean(v["share_ceg"] for v in out["bda"].values()),
        "bda_share_not_in_pool_mean": st.mean(v["share_not_in_pool"] for v in out["bda"].values()),
    }
    OUT.write_text(json.dumps(out, indent=1))
    s = out["summary"]
    print(f"BDA mean {s['bda_mean']:.4f}  budget spent on CEGv2 {s['bda_share_ceg_mean']:.1%}  "
          f"never in the pool {s['bda_share_not_in_pool_mean']:.1%}")
    for ds, v in out["bda"].items():
        print(f"  {ds:24s} BDA {v['hit_at_r5']:.3f}  CEG {v['share_ceg']:.1%}  not in pool {v['share_not_in_pool']:.1%}")
    for f, v in out["rankings"].items():
        for k in ("ne_pool", "full_pool"):
            p = v[k]["vs_bda"]
            print(f"{f:20s} {k:9s} {v[k]['mean']:.4f}  vs BDA {p['delta']:+.4f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"
                  f"  {p['wins']}/{p['ties']}/{p['losses']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
