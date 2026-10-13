"""
ml_engines_tier_a.py — the alternative ML engines on BDA's full-pool single-gene tier (core-essential genes
kept; Table tiers' "full pool" column). Phase 3 of docs/ml_engines_plan.md, registered before the runs.
Reads experiment 15 run with TIER=full_pool (workspace/results/ml_engines/tier_a/); trains nothing.

The rule is the headline tier's: the primary metric is the nine-screen mean hit@R5 of the online engine;
an engine is better or worse than LightGBM only if the paired, screen-clustered bootstrap 95% CI of the
difference excludes 0. The reference is the online LightGBM engine run on this tier in the same
experiment. Each engine's headline-tier difference is reported next to its full-pool one.

    python -m waddington_select.analysis.ml_engines_tier_a
"""

from __future__ import annotations

import json

import numpy as np

from ..ml_models import ENGINES, is_stochastic
from . import ml_engines as H
from .final_system_stats import BENCH, _paired
from .sources import REPORTED_TIER_A_RUN

RES = H.RES / "tier_a"
OUT = RES / "summary.json"
TIER_A_V3 = H.REPO / "workspace" / "evaluation" / "lgbm_training_data_tier_a_v3.csv"


def _tier_a_reference() -> dict[str, dict[str, float]]:
    """Context rows from the reported full-pool run (experiment 08): the full system and its baselines."""
    main = json.loads(REPORTED_TIER_A_RUN.read_text())
    return {arm: {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in main[ds][arm]])) for ds in BENCH}
            for arm in ("waddington_c", "online_adaptive", "static_ranker")}


def summarize() -> dict:
    arms = H._load(RES)
    complete = {a: v for a, v in arms.items() if all(ds in v for ds in BENCH)}
    if H.REF not in complete:
        raise SystemExit(f"{H.REF} missing or incomplete; run TIER=full_pool experiments/15_ml_engines.sh tabular")
    ref = H._per_screen(complete[H.REF])
    headline = json.loads(H.OUT.read_text())["rows"] if H.OUT.exists() else {}
    rows = {}
    for arm, by_ds in sorted(complete.items()):
        m = H._model(arm)
        ps = H._per_screen(by_ds)
        h = headline.get(arm)
        d = None if arm == H.REF else _paired(ps, ref)
        rows[arm] = {
            "model": m,
            "online": arm.startswith("ml_engine_"),
            "probabilistic": m == "lightgbm" or ENGINES[m].probabilistic,
            "replicates": min(len(v) for v in by_ds.values()),
            "per_screen": ps,
            "mean": float(np.mean([ps[ds] for ds in BENCH])),
            "vs_lightgbm_online": d,
            "deterministic_check": None if is_stochastic(m) else H._determinism(by_ds),
            "seed_sd_of_mean": H._seed_spread(by_ds) if is_stochastic(m) else None,
            "headline_mean": None if h is None else h["mean"],
            "headline_vs_lightgbm": None if h is None else h["vs_lightgbm_online"],
        }
        hv = rows[arm]["headline_vs_lightgbm"]
        rows[arm]["same_sign_as_headline"] = None if (d is None or hv is None) else bool(
            np.sign(d["delta"]) == np.sign(hv["delta"]))
    for m in {r["model"] for r in rows.values()}:
        on, st = rows.get(f"ml_engine_{m}"), rows.get(f"ml_static_{m}")
        if on and st:
            on["online_minus_static"] = _paired(on["per_screen"], st["per_screen"])
    ctx = _tier_a_reference()
    free = {}
    for f, ps in H.label_free_baselines(TIER_A_V3).items():
        free[f] = {"per_screen": ps, "mean": float(np.mean([ps[ds] for ds in BENCH])),
                   "ml_engine_minus_this": _paired(ref, ps), "c_minus_this": _paired(ctx["waddington_c"], ps)}
    res = {"tier": "full_pool", "reference": H.REF, "rows": rows,
           "incomplete": sorted(set(arms) - set(complete)),
           "context_full_pool_run": {a: {"per_screen": ps, "mean": float(np.mean(list(ps.values())))}
                                     for a, ps in ctx.items()},
           "c_minus_ml_engine": _paired(ctx["waddington_c"], ref),
           "label_free_post_hoc": free}
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = summarize()
    print(f"full-pool tier, reference online LightGBM {r['rows'][H.REF]['mean']:.4f}")
    for arm, v in sorted(r["rows"].items(), key=lambda kv: (not kv[1]["online"], -kv[1]["mean"])):
        d, hv = v["vs_lightgbm_online"], v["headline_vs_lightgbm"]
        txt = "reference" if d is None else (f"{d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
                                             f"{d['wins']}/{d['ties']}/{d['losses']}")
        htxt = "" if hv is None else f" | headline {hv['delta']:+.4f}"
        print(f"  {arm:28s} {v['mean']:.4f} x{v['replicates']}  {txt}{htxt}")
    print("incomplete arms:", r["incomplete"])
    ctx = r["context_full_pool_run"]
    print("context (experiment 08): " + ", ".join(f"{a} {v['mean']:.4f}" for a, v in ctx.items()))
    c = r["c_minus_ml_engine"]
    print(f"C - ML engine {c['delta']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] {c['wins']}/{c['ties']}/{c['losses']}")
    v = r["label_free_post_hoc"]["depmap_frac_ess"]
    print(f"DepMap essentiality ranking {v['mean']:.4f}; C - this {v['c_minus_this']['delta']:+.4f}")
