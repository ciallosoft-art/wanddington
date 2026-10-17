"""
main_reruns.py — main-text analyses re-run on the reported system (docs/tier_extension_plan.md, part 丁).

  ladder   reasoning vs recall: A0 anonymous names, A0.5 + the ML-ranked pool, A1 + structural features,
           A2 real names, all on the reported C-arm (experiment 19, headline: ablations_20rep.json for A0 and
           A2, ladder_20rep.json for A0.5 and A1); the C-arm's ML engine alone as the GBM reference; A1's LLM
           replaced by the hit-centroid rule (linear_2rep.json, no LLM, deterministic). Per screen, a rung
           difference is a difference of means +- its replicate standard error (the two runs' replicates are
           not paired); the averages are paired over screens
  loso     full leave-one-study-out (experiment 19, EXCLUDE_STUDY=1, the six screens with a sibling; the other
           three are the paired full-system run), against the matched static prior with the same siblings
           removed (computed here, in the reported feature space)
  router   global-fixed, nested router and study-excluded selection under the frozen router protocol, on the
           reported configuration's weight grid (0.2: the reported run; 0.3, 0.4: experiment 17)
  rf_pairs the full C-arm with a random-forest engine on the two-gene tier (experiment 19, TIER=pairs)

  mechanism  what each rung's vote does to the batch (--mechanism; replays the ML branch, no LLM calls): of the
           genes the LLM names, how many the ML would have taken anyway; how many the vote swaps in; the hits
           the swapped-in genes bring against the hits of the ML picks they displace (first-order, along the
           recorded trajectory, as replay_attribution); saved to main_reruns_mechanism.json and merged

  old_anchor  did the earlier "anchor-free" ladder (gpt/router/reasoning_noanchor.json) drop the anchor features
           in its anonymous arm? Its round 1 replayed from the recorded LLM names with the arm's own weights,
           the ML prior trained with the anchor features kept and dropped: which reproduces the recorded batch
           (--old-anchor; screens on a weighted route; saved to main_reruns_old_anchor.json and merged)

  crispra  the gain-of-function probe on the reported system (part 己): A2 - A1 on Schmidt et al.'s two CRISPRa
           screens (experiment 19, CRISPRA=1) against their CRISPRi twins and Steinhart from the ladder, difference
           of means +- replicate standard error
  jurkat   random forest vs LightGBM as the ML engine alone on the held-out Jurkat map (part 己, experiment 15,
           TIER=jurkat_pairs / jurkat_pairs_interaction), at 640 and 160 picks, Welch

Single-gene comparisons: paired over the nine screens, screen-clustered bootstrap (final_system_stats._paired).

    python -m waddington_select.analysis.main_reruns [--mechanism] [--old-anchor]
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

from .final_system_stats import BENCH, _matched_loo, _paired
from .sources import GPT_ROUTER, LEGACY_GPT_ABLATIONS, REPORTED_RUN
from .tier_extension import _arms, _hits, _per_screen, _summ, _vs

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "workspace" / "results"
OUT = RES / "main_reruns.json"
MECH = RES / "main_reruns_mechanism.json"
OLD_ANCHOR = RES / "main_reruns_old_anchor.json"
HEAD = RES / "tier_ablations" / "headline"


def _mean(ps: dict) -> float:
    return float(np.mean([ps[s] for s in BENCH]))


def _last(runs: list[dict]) -> np.ndarray:
    return np.array([r["hit_ratio_per_round"][-1] for r in runs], dtype=float)


def _diff_se(x: list[dict], y: list[dict]) -> dict:
    a, b = _last(x), _last(y)
    return {"delta": float(a.mean() - b.mean()),
            "se": float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))}


def ladder() -> dict:
    a = _arms(HEAD / "ablations_20rep.json")
    lad = HEAD / "ladder_20rep.json"
    if not lad.exists():
        return {}
    b = _arms(lad)
    runs = {"A0": a["waddington_c_anon_names"], "A0.5": b["waddington_c_anon_pool"],
            "A1": b["waddington_c_anon_features"], "A2": a["waddington_c"]}
    rungs = {k: _per_screen(v) for k, v in runs.items()}
    gbm = json.loads((RES / "ml_engines" / "summary.json").read_text())["rows"]["ml_engine_lightgbm"]["per_screen"]
    steps = {"pool (A0.5-A0)": ("A0.5", "A0"), "features (A1-A0.5)": ("A1", "A0.5"), "name (A2-A1)": ("A2", "A1")}
    out = {"means": {k: _mean(v) for k, v in rungs.items()} | {"GBM": _mean(gbm)},
           "per_screen": rungs | {"GBM": gbm},
           "per_screen_steps": {k: {s: _diff_se(runs[hi][s], runs[lo][s]) for s in BENCH}
                                for k, (hi, lo) in steps.items()},
           "A1-GBM": _paired(rungs["A1"], gbm), "A2-GBM": _paired(rungs["A2"], gbm)}
    out |= {k: _paired(rungs[hi], rungs[lo]) for k, (hi, lo) in steps.items()}
    lin_f = HEAD / "linear_2rep.json"
    if lin_f.exists():
        lin = _arms(lin_f)["waddington_c_anon_linear"]
        lp = _per_screen(lin)
        out["linear"] = {"mean": _mean(lp), "per_screen": lp,
                         "deterministic": all(r["selected_genes_per_round"] == lin[s][0]["selected_genes_per_round"]
                                              for s in BENCH for r in lin[s]),
                         "A1-linear": _paired(rungs["A1"], lp), "linear-GBM": _paired(lp, gbm)}
    return out


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
    # the older with-anchor grid (3 seeds, before the task-prompt correction) and the legacy routed C, kept in
    # the table as older runs, paired with the reported system
    old = json.loads((GPT_ROUTER / "router_analysis.json").read_text())["variants"]
    leg = json.loads(LEGACY_GPT_ABLATIONS.read_text())
    legacy_c = {s: float(np.mean([r["hit_ratio_per_round"][-1] for r in leg[s]["waddington_c"]])) for s in BENCH}
    out["older_runs"] = {k: {"mean": _mean(v["per_screen"]), "vs_reported_c": _paired(v["per_screen"], grid[0.2])}
                         for k, v in old.items()}
    out["older_runs"]["legacy_routed_c"] = {"mean": _mean(legacy_c), "vs_reported_c": _paired(legacy_c, grid[0.2])}
    return out


def rf_pairs() -> dict:
    f = RES / "tier_ablations" / "pairs" / "c_rf_20rep.json"
    if not f.exists():
        return {}
    from .pair_tier import RUNS
    rf = _hits(_arms(f)["waddington_c__rf"]["Horlbeck"])
    c = _hits(_arms(RES / "tier_ablations" / "pairs" / "ablations_20rep.json")["waddington_c"]["Horlbeck"])
    rf_ml = _hits(_arms(RES / "ml_engines" / "pairs")["ml_engine_rf"]["Horlbeck"])
    bda_file, bda_arm = RUNS[128]["bda"]
    bda = _hits(json.loads(bda_file.read_text())["Horlbeck"][bda_arm])
    return {"c_rf": _summ(rf), "c_lightgbm": _summ(c), "rf_engine_alone": _summ(rf_ml), "bda_640": _summ(bda),
            "rf_minus_lightgbm": _vs(rf, c), "c_rf_minus_rf_engine": _vs(rf, rf_ml), "c_rf_minus_bda": _vs(rf, bda)}


def _vote_effect(ds: str, runs: list[dict]) -> dict:
    from ..arms.online_adaptive_arm import OnlineAdaptiveArm
    from ..arms.waddington_c_arm import _drop_anchor_default, _get_feature_config
    from ..oracle import BATCH_SIZES, DatasetOracle
    from .replay_attribution import W_LLM, W_ML
    bs = BATCH_SIZES[ds]
    csv, extra = _get_feature_config(ds)
    base = OnlineAdaptiveArm(ds, bs, training_csv=csv, extra_feature_cols=extra, drop_anchor_feats=_drop_anchor_default())
    oracle = DatasetOracle(ds)
    acc = dict.fromkeys(("rounds", "verified", "named", "named_in_ml_top", "swapped", "gained", "lost"), 0)
    for rec in runs:
        online, selected = copy.deepcopy(base), set()
        n = len(rec["selected_genes_per_round"])
        for r, recorded in enumerate(rec["selected_genes_per_round"]):
            ml = online.all_scores()
            named = set(rec["llm_named_proposals_per_round"][r])
            combined = {g: W_ML * ml.get(g, 0.0) + W_LLM * (1.0 if g in named else 0.0)
                        for g in online._genes if g not in selected}
            replayed = sorted(combined, key=combined.__getitem__, reverse=True)[:bs]
            ml_top = set(sorted(combined, key=lambda g: -ml.get(g, 0.0))[:bs])
            rec_set = set(recorded)
            add, drop = rec_set - ml_top, ml_top - rec_set
            acc["rounds"] += 1
            acc["verified"] += replayed == list(recorded)
            acc["named"] += len(named)
            acc["named_in_ml_top"] += len(named & ml_top)
            acc["swapped"] += len(add)
            acc["gained"] += sum(oracle.reveal(list(add)).values())
            acc["lost"] += sum(oracle.reveal(list(drop)).values())
            selected |= rec_set
            if r < n - 1:
                online.update(r, oracle.reveal(list(recorded)))
    acc["total_hits"] = oracle.total_hits
    acc["replicates"] = len(runs)
    return acc


def mechanism() -> dict:
    a, b = _arms(HEAD / "ablations_20rep.json"), _arms(HEAD / "ladder_20rep.json")
    src = {"A0": a["waddington_c_anon_names"], "A0.5": b["waddington_c_anon_pool"],
           "A1": b["waddington_c_anon_features"], "A2": a["waddington_c"]}
    out = {}
    for rung, by in src.items():
        per = {ds: _vote_effect(ds, by[ds]) for ds in BENCH}
        tot = {k: sum(p[k] for p in per.values()) for k in ("rounds", "verified", "named", "named_in_ml_top",
                                                             "swapped", "gained", "lost")}
        out[rung] = {"per_screen": per, "total": tot,
                     "named_already_in_ml_top": tot["named_in_ml_top"] / max(1, tot["named"]),
                     "swapped_per_round": tot["swapped"] / tot["rounds"],
                     "precision_swapped_in": tot["gained"] / max(1, tot["swapped"]),
                     "precision_displaced": tot["lost"] / max(1, tot["swapped"]),
                     # hit@R5 the vote adds over the ML picks it displaces, mean over screens (first-order)
                     "net_hit_ratio": _mean({ds: (p["gained"] - p["lost"]) / (p["replicates"] * p["total_hits"])
                                             for ds, p in per.items()})}
        print(rung, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[rung].items()
                     if k not in ("per_screen",)}, flush=True)
    MECH.write_text(json.dumps(out, indent=1))
    return out


def old_anchor() -> dict:
    os.environ["WADDINGTON_FEATURE_POLICY"] = "metadata"  # that run's feature policy (its manifest)
    from ..arms.online_adaptive_arm import OnlineAdaptiveArm
    from ..arms.waddington_c_arm import (W_LLM_DEFAULT, W_LLM_LARGE, W_ML_DEFAULT, W_ML_LARGE, _classify,
                                         _get_dataset_stats, _get_feature_config)
    from ..oracle import BATCH_SIZES
    from .sources import GPT_ROUTER
    data = json.loads((GPT_ROUTER / "reasoning_noanchor.json").read_text())
    out = {}
    for ds in BENCH:
        route = _classify(*_get_dataset_stats(ds))
        if route == "two_stage":
            continue
        legacy_w = (W_ML_LARGE, W_LLM_LARGE) if route == "ml_heavy" else (W_ML_DEFAULT, W_LLM_DEFAULT)
        csv, extra = _get_feature_config(ds)
        bs = BATCH_SIZES[ds]
        row = {"route": route}
        for drop in (False, True):
            ml = OnlineAdaptiveArm(ds, bs, training_csv=csv, extra_feature_cols=extra, drop_anchor_feats=drop).all_scores()
            for arm, (w_ml, w_llm) in (("waddington_c_feature_reasoning", legacy_w), ("waddington_c", (0.8, 0.2))):
                n = 0
                for r in data[ds][arm]:
                    named = set(r["llm_named_proposals_per_round"][0])
                    comb = {g: w_ml * ml.get(g, 0.0) + w_llm * (1.0 if g in named else 0.0) for g in ml}
                    n += set(sorted(comb, key=comb.__getitem__, reverse=True)[:bs]) == set(r["selected_genes_per_round"][0])
                row[f"{arm}|anchors_{'dropped' if drop else 'kept'}"] = f"{n}/{len(data[ds][arm])}"
        out[ds] = row
        print(ds, row, flush=True)
    OLD_ANCHOR.write_text(json.dumps(out, indent=1))
    return out


def crispra() -> dict:
    f = HEAD / "crispra_probe_20rep.json"
    if not f.exists():
        return {}
    new = _arms(f)
    a = _arms(HEAD / "ablations_20rep.json")
    lad = _arms(HEAD / "ladder_20rep.json")
    runs = {"A1": {**lad["waddington_c_anon_features"], **new["waddington_c_anon_features"]},
            "A2": {**a["waddington_c"], **new["waddington_c"]}}
    screens = ("IL2_crispra", "IFNG_crispra", "IL2", "IFNG", "Steinhart")
    per = {ds: {"A1": float(_last(runs["A1"][ds]).mean()), "A2": float(_last(runs["A2"][ds]).mean()),
                "name": _diff_se(runs["A2"][ds], runs["A1"][ds])} for ds in screens}
    twin = {}
    for ca, ci in (("IL2_crispra", "IL2"), ("IFNG_crispra", "IFNG")):
        d = per[ca]["name"]["delta"] - per[ci]["name"]["delta"]
        twin[ca] = {"delta": d, "se": float(np.hypot(per[ca]["name"]["se"], per[ci]["name"]["se"]))}
    return {"per_screen": per, "crispra_minus_crispri_twin": twin}


def jurkat() -> dict:
    from scipy import stats
    out = {}
    for t in ("jurkat_pairs", "jurkat_pairs_interaction"):
        for b, picks in ((128, 640), (32, 160)):
            d = RES / "ml_engines" / f"{t}_b{b}"
            if not d.exists():
                continue
            arms = _arms(d)
            lg = _hits(arms["ml_engine_lightgbm"]["Horlbeck_Jurkat"])
            rf = _hits(arms["ml_engine_rf"]["Horlbeck_Jurkat"])
            p = None if lg.std() > 0 else float(stats.ttest_1samp(rf, lg.mean()).pvalue)
            out[f"{t}@{picks}"] = {"lightgbm": _summ(lg), "lightgbm_deterministic": bool(lg.std() == 0),
                                   "rf": _summ(rf), "rf_minus_lightgbm": float(rf.mean() - lg.mean()), "p": p}
    return out


def run() -> dict:
    res = {"ladder": ladder(), "loso": loso(), "router": router(), "rf_pairs": rf_pairs(),
           "crispra": crispra(), "jurkat": jurkat()}
    if MECH.exists():
        res["mechanism"] = json.loads(MECH.read_text())
    if OLD_ANCHOR.exists():
        res["old_anchor"] = json.loads(OLD_ANCHOR.read_text())
    OUT.write_text(json.dumps(res, indent=2))
    return res


def _fmt(v: dict) -> str:
    if "wins" in v:
        return f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['wins']}/{v['ties']}/{v['losses']}"
    return f"{v['delta']:+.1f} (p={v.get('p')})"


if __name__ == "__main__":
    if "--mechanism" in sys.argv:
        mechanism()
    if "--old-anchor" in sys.argv:
        old_anchor()
    r = run()
    if r["ladder"]:
        L = r["ladder"]
        print("[ladder] means:", {k: round(v, 4) for k, v in L["means"].items()})
        for k in ("pool (A0.5-A0)", "features (A1-A0.5)", "name (A2-A1)", "A1-GBM", "A2-GBM"):
            print(f"  {k:20s} {_fmt(L[k])}")
        if "linear" in L:
            x = L["linear"]
            print(f"  centroid rule {x['mean']:.4f} (deterministic {x['deterministic']}); A1-rule {_fmt(x['A1-linear'])}; "
                  f"rule-GBM {_fmt(x['linear-GBM'])}")
    if r["loso"]:
        o = r["loso"]
        print(f"[loso] {o['mean']:.4f} (reran {o['screens_rerun']}); matched LOO w/o siblings "
              f"{o['matched_loo_siblings_excluded']:.4f}: {_fmt(o['vs_matched_loo_siblings_excluded'])}; "
              f"vs matched LOO {_fmt(o['vs_matched_loo'])}; vs C {_fmt(o['vs_full_system'])}")
    ro = r["router"]
    print(f"[router] reported C {ro['reference_c']:.4f}, matched LOO {ro['matched_loo']:.4f}")
    for k, v in ro["variants"].items():
        print(f"  {k:26s} {v['mean']:.4f}  vs matched LOO {_fmt(v['vs_matched_loo'])}  vs C {_fmt(v['vs_reported_c'])}")
    for k, v in ro["older_runs"].items():
        print(f"  older {k:20s} {v['mean']:.4f}  vs C {_fmt(v['vs_reported_c'])}")
    if r["rf_pairs"]:
        x = r["rf_pairs"]
        print(f"[rf pairs] C+RF {x['c_rf']['mean']:.1f}+-{x['c_rf']['sd']:.1f} vs C {x['c_lightgbm']['mean']:.1f}: "
              f"{_fmt(x['rf_minus_lightgbm'])}; vs RF alone {x['rf_engine_alone']['mean']:.1f}: "
              f"{_fmt(x['c_rf_minus_rf_engine'])}; vs BDA@640 {x['bda_640']['mean']:.1f}: {_fmt(x['c_rf_minus_bda'])}")
    if r["crispra"]:
        for ds, v in r["crispra"]["per_screen"].items():
            print(f"[crispra] {ds:14s} A1 {v['A1']:.4f}  A2 {v['A2']:.4f}  name {v['name']['delta']:+.4f} +- {v['name']['se']:.4f}")
        for ds, v in r["crispra"]["crispra_minus_crispri_twin"].items():
            print(f"[crispra] {ds} name value minus its CRISPRi twin's: {v['delta']:+.4f} +- {v['se']:.4f}")
    for k, v in r["jurkat"].items():
        print(f"[jurkat] {k:32s} LightGBM {v['lightgbm']['mean']:.1f} (deterministic {v['lightgbm_deterministic']}), "
              f"RF {v['rf']['mean']:.1f} +- {v['rf']['sd']:.1f}: {v['rf_minus_lightgbm']:+.1f} (p={v['p']})")
    print(f"saved -> {OUT}")
