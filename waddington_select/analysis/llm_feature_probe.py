"""
llm_feature_probe.py — a direct feature probe of the (black-box) LLM branch.

The LLM never sees the structural feature table; it names genes from the task description and its own
parametric knowledge. We probe what gene properties its naming *correlates with* by fitting two LightGBM
models over the same no-anchor feature set (+ screen fixed effects) and comparing their native SHAP:

  ML model  : features -> is the gene a hit?          (what the ML ranker uses)
  LLM model : features -> did the LLM ever name it?   (what the LLM's naming tracks)

If the two rankings agree, the LLM adds nothing the ML feature space does not already encode; if they
differ, the LLM's endorsement is driven by orthogonal gene properties. Reported as each feature's share
of the total mean |SHAP| (gene features only; the screen one-hots absorb per-screen naming rates).

    python -m waddington_select.analysis.llm_feature_probe
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..oracle import BENCHMARK_DATASETS
from .prior_probes import _load, LGBM, INTRINSIC, DEPMAP

REPO = Path(__file__).resolve().parents[2]
from .sources import REPORTED_RUN_DIR  # noqa: E402  (the reported run; analysis/sources.py)
RUN = REPORTED_RUN_DIR / "waddington_c_20rep.json"
OUT = REPO / "workspace" / "results" / "llm_feature_probe.json"
FEATS = INTRINSIC + DEPMAP  # no-anchor reported configuration
SCREENS = list(BENCHMARK_DATASETS)  # --pairs: ["Horlbeck"], and FEATS gains each feature's within-pair spread


def _named_counts(run: dict) -> dict:
    """screen -> {gene(upper): number of repeats in which the LLM named it}."""
    out: dict[str, collections.Counter] = {}
    for s in SCREENS:
        cnt: collections.Counter = collections.Counter()
        for r in run[s]["waddington_c"]:
            named: set[str] = set()
            for round_genes in r.get("selected_llm_named_genes_per_round", []):
                named |= {g.strip().upper() for g in round_genes}
            for g in named:
                cnt[g] += 1
        out[s] = cnt
    return out


def _pooled_table(df: pd.DataFrame, named: dict) -> pd.DataFrame:
    rows = []
    for s in SCREENS:
        sub = df[df["dataset"] == s].copy()
        sub["llm_named"] = (sub["gene"].map(lambda g: named[s].get(g, 0)) > 0).astype(int)
        sub["screen"] = s
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _shap_share(table: pd.DataFrame, target: str) -> dict:
    """Each gene-feature's share of total mean |SHAP| in a LightGBM predicting `target`, with screen
    fixed effects included in the fit but excluded from the reported shares."""
    screen_oh = pd.get_dummies(table["screen"])
    X = pd.concat([table[FEATS], screen_oh], axis=1).values.astype(float)
    y = table[target].values
    pos, neg = (y == 1).sum(), (y == 0).sum()
    model = lgb.LGBMClassifier(**LGBM, scale_pos_weight=neg / max(pos, 1))
    model.fit(X, y)
    contrib = model.predict(X, pred_contrib=True)[:, : len(FEATS)]  # gene-feature SHAP only
    mean_abs = {f: float(np.mean(np.abs(contrib[:, j]))) for j, f in enumerate(FEATS)}
    total = sum(mean_abs.values()) or 1.0
    return {f: v / total for f, v in mean_abs.items()}


def run() -> dict:
    df = _load()
    run_json = json.loads(RUN.read_text())
    table = _pooled_table(df, _named_counts(run_json))
    ml = _shap_share(table, "label")
    llm = _shap_share(table, "llm_named")
    res = {
        "feature_set": "intrinsic+depmap (no anchor)",
        "ml_hit_share": ml,
        "llm_named_share": llm,
        "family_share": {
            "ml": {"depmap": sum(ml[f] for f in DEPMAP), "connectivity_constraint": sum(ml[f] for f in INTRINSIC)},
            "llm": {"depmap": sum(llm[f] for f in DEPMAP), "connectivity_constraint": sum(llm[f] for f in INTRINSIC)},
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    import argparse
    from . import prior_probes
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None, help="a run with the C-arm's selection dumps")
    ap.add_argument("--csv", type=Path, default=None, help="that tier's v3 table")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pairs", action="store_true", help="the two-gene tier: Horlbeck only, pair encoding")
    a = ap.parse_args()
    RUN, OUT = a.run or RUN, a.out or OUT
    if a.csv:
        prior_probes.CSV = a.csv
    if a.pairs:
        SCREENS = ["Horlbeck"]
        FEATS = FEATS + [f + "_absdiff" for f in FEATS]
        INTRINSIC = INTRINSIC + [f + "_absdiff" for f in INTRINSIC]
        DEPMAP = DEPMAP + [f + "_absdiff" for f in DEPMAP]
    r = run()
    print(f"{'feature':28s} {'ML(hit)':>9s} {'LLM(named)':>11s}")
    for f in sorted(FEATS, key=lambda f: -r["ml_hit_share"][f]):
        print(f"  {f:26s} {r['ml_hit_share'][f]*100:7.1f}% {r['llm_named_share'][f]*100:9.1f}%")
    fs = r["family_share"]
    print(f"\nDepMap family share      ML {fs['ml']['depmap']*100:.0f}%  vs  LLM {fs['llm']['depmap']*100:.0f}%")
    print(f"connectivity/constraint  ML {fs['ml']['connectivity_constraint']*100:.0f}%  vs  LLM {fs['llm']['connectivity_constraint']*100:.0f}%")
    print(f"saved -> {OUT}")
