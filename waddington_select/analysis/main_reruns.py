"""
main_reruns.py — main-text analyses re-run on the reported system (docs/tier_extension_plan.md, part 丁).

  ladder   reasoning vs recall: A0 anonymous names, A0.5 + the ML-ranked pool, A1 + structural features,
           A2 real names, all on the reported C-arm (experiment 19, headline: ablations_20rep.json for A0 and
           A2, ladder_20rep.json for A0.5 and A1); the C-arm's ML engine alone as the GBM reference
  loso     full leave-one-study-out (experiment 19, EXCLUDE_STUDY=1, the six screens with a sibling; the other
           three are the paired full-system run), against the matched static prior with the same siblings
           removed (computed here, in the reported feature space)
  router   global-fixed, nested router and study-excluded selection under the frozen router protocol, on the
           reported configuration's weight grid (0.2: the reported run; 0.3, 0.4: experiment 17)
  rf_pairs the full C-arm with a random-forest engine on the two-gene tier (experiment 19, TIER=pairs)

Single-gene comparisons: paired over the nine screens, screen-clustered bootstrap (final_system_stats._paired).

    python -m waddington_select.analysis.main_reruns
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .final_system_stats import BENCH, _matched_loo, _paired
from .sources import REPORTED_RUN
from .tier_extension import _arms, _hits, _per_screen, _summ, _vs

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "workspace" / "results"
OUT = RES / "main_reruns.json"
HEAD = RES / "tier_ablations" / "headline"


def _mean(ps: dict) -> float:
    return float(np.mean([ps[s] for s in BENCH]))


def ladder() -> dict:
    a = _arms(HEAD / "ablations_20rep.json")
    lad = HEAD / "ladder_20rep.json"
    if not lad.exists():
        return {}
    b = _arms(lad)
    rungs = {"A0": _per_screen(a["waddington_c_anon_names"]), "A0.5": _per_screen(b["waddington_c_anon_pool"]),
             "A1": _per_screen(b["waddington_c_anon_features"]), "A2": _per_screen(a["waddington_c"])}
    gbm = json.loads((RES / "ml_engines" / "summary.json").read_text())["rows"]["ml_engine_lightgbm"]["per_screen"]
    return {"means": {k: _mean(v) for k, v in rungs.items()} | {"GBM": _mean(gbm)},
            "per_screen": rungs | {"GBM": gbm},
            "pool (A0.5-A0)": _paired(rungs["A0.5"], rungs["A0"]),
            "features (A1-A0.5)": _paired(rungs["A1"], rungs["A0.5"]),
            "name (A2-A1)": _paired(rungs["A2"], rungs["A1"]),
            "A1-GBM": _paired(rungs["A1"], gbm)}


def loso() -> dict:
    f = HEAD / "loso_20rep.json"
    if not f.exists():
        return {}
    lo = _per_screen(_arms(f)["waddington_c"])
    c = _per_screen(_arms(HEAD / "ablations_20rep.json")["waddington_c"])
    strict = {s: lo.get(s, c[s]) for s in BENCH}
    old = os.environ.get("WADDINGTON_EXCLUDE_STUDY")
    os.environ["WADDINGTON_EXCLUDE_STUDY"] = "1"
    try:
        loo_out = _matched_loo()
    finally:
        if old is None:
            os.environ.pop("WADDINGTON_EXCLUDE_STUDY", None)
        else:
            os.environ["WADDINGTON_EXCLUDE_STUDY"] = old
    loo = _matched_loo()
    return {"screens_rerun": sorted(lo), "mean": _mean(strict), "per_screen": strict,
            "matched_loo_siblings_excluded": _mean(loo_out), "matched_loo": _mean(loo), "full_system": _mean(c),
            "vs_matched_loo_siblings_excluded": _paired(strict, loo_out),
            "vs_matched_loo": _paired(strict, loo), "vs_full_system": _paired(strict, c)}


def router() -> dict:
    from . import router_analysis as R
    from ..router_protocol import CANDIDATE_WLLM
    rep = json.loads(Path(REPORTED_RUN).read_text())
    grid = {0.2: {s: float(np.mean([r["hit_ratio_per_round"][-1] for r in rep[s]["waddington_c"]])) for s in BENCH}}
    for w in (0.3, 0.4):
        grid[w] = _per_screen(_arms(RES / "fusion_weight" / f"carm_w{w}_5rep.json")["waddington_c"])
    W = {(s, w): grid[w][s] for s in BENCH for w in CANDIDATE_WLLM}
    os.environ.pop("WADDINGTON_EXCLUDE_STUDY", None)
    loo = _matched_loo()
    out = {"candidates": CANDIDATE_WLLM, "reference_c": _mean(grid[0.2]), "matched_loo": _mean(loo), "variants": {}}
    for name, (per, choice) in {"global_fixed": R.global_fixed(W), "nested_router": R.nested_router(W),
                                "global_fixed_study_out": R.global_fixed(W, study_out=True),
                                "nested_router_study_out": R.nested_router(W, study_out=True)}.items():
        out["variants"][name] = {"mean": _mean(per), "per_screen": per, "config_per_fold": choice,
                                 "vs_matched_loo": _paired(per, loo), "vs_reported_c": _paired(per, grid[0.2])}
    return out


def rf_pairs() -> dict:
    f = RES / "tier_ablations" / "pairs" / "c_rf_20rep.json"
    if not f.exists():
        return {}
    rf = _hits(_arms(f)["waddington_c__rf"]["Horlbeck"])
    c = _hits(_arms(RES / "tier_ablations" / "pairs" / "ablations_20rep.json")["waddington_c"]["Horlbeck"])
    return {"c_rf": _summ(rf), "c_lightgbm": _summ(c), "rf_minus_lightgbm": _vs(rf, c)}


def run() -> dict:
    res = {"ladder": ladder(), "loso": loso(), "router": router(), "rf_pairs": rf_pairs()}
    OUT.write_text(json.dumps(res, indent=2))
    return res


def _fmt(v: dict) -> str:
    if "wins" in v:
        return f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['wins']}/{v['ties']}/{v['losses']}"
    return f"{v['delta']:+.1f} (p={v.get('p')})"


if __name__ == "__main__":
    r = run()
    if r["ladder"]:
        L = r["ladder"]
        print("[ladder] means:", {k: round(v, 4) for k, v in L["means"].items()})
        for k in ("pool (A0.5-A0)", "features (A1-A0.5)", "name (A2-A1)", "A1-GBM"):
            print(f"  {k:20s} {_fmt(L[k])}")
    if r["loso"]:
        o = r["loso"]
        print(f"[loso] {o['mean']:.4f} (reran {o['screens_rerun']}); matched LOO w/o siblings "
              f"{o['matched_loo_siblings_excluded']:.4f}: {_fmt(o['vs_matched_loo_siblings_excluded'])}; "
              f"vs matched LOO {_fmt(o['vs_matched_loo'])}; vs C {_fmt(o['vs_full_system'])}")
    ro = r["router"]
    print(f"[router] reported C {ro['reference_c']:.4f}, matched LOO {ro['matched_loo']:.4f}")
    for k, v in ro["variants"].items():
        print(f"  {k:26s} {v['mean']:.4f}  vs matched LOO {_fmt(v['vs_matched_loo'])}  vs C {_fmt(v['vs_reported_c'])}")
    if r["rf_pairs"]:
        x = r["rf_pairs"]
        print(f"[rf pairs] C+RF {x['c_rf']['mean']:.1f} vs C {x['c_lightgbm']['mean']:.1f}: {_fmt(x['rf_minus_lightgbm'])}")
    print(f"saved -> {OUT}")
