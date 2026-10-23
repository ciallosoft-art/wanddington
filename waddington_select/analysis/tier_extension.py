"""
tier_extension.py — the follow-up analyses on BDA's other two tiers (docs/tier_extension_plan.md).

Reads the runs of experiments 19, 17 (TIER=...), 15 (TIER=full_pool / pairs) and the offline replays of
experiment 21, and writes workspace/results/tier_extension_summary.json. Trains nothing.

  ablations     the reported C-arm and its component ablations on each tier (experiment 19); the ablation
                without the LLM is the C-arm's own ML engine (ml_engine_lightgbm, experiment 15)
  fusion        the fusion-weight grid on the full pool and the pairs (experiment 17); 0.2 is experiment 19's
                paired run of the reported C-arm, 0 the ML engine
  pair_engines  the ML engines on the pair tier, both tables (experiment 15, TIER=pairs / pairs_interaction)
  acquisition   acquisition rules on the full pool and the pairs (experiment 15, group acquisition)
  genedisco     GeneDisco with its own models and rules, equal information, on the full pool (gdfb arms)
  attribution   source attribution and the ML-adjusted value of LLM endorsement (experiment 21 replays)
  shap          the online model's most-used features over its selections, from the same replays
  novel_hits    recurrent vs novel hits on the full pool (experiment 21)
  full_pool_per_screen  every method of Table tiers on the full pool, per screen
  pair_features the eight interaction features ranked one at a time on the K562 map (no training)

Single-gene tiers: paired over the nine screens, screen-clustered bootstrap 95% CI, wins/ties/losses
(final_system_stats._paired). The pair tier has one screen: Welch t-test over replicates, or a one-sample
test against a deterministic arm.

    python -m waddington_select.analysis.tier_extension
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .final_system_stats import BENCH, _paired

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "workspace" / "results"
OUT = RES / "tier_extension_summary.json"
ABL = RES / "tier_ablations"
FUS = RES / "fusion_weight"
ENG = RES / "ml_engines"
SINGLE = ("headline", "full_pool")
PAIRS = ("pairs", "pairs_interaction")
VARIANTS = ("waddington_c_nomem", "waddington_c_static_ml", "waddington_c_anon_names")


def _arms(path: Path) -> dict[str, dict[str, list[dict]]]:
    """arm -> screen -> runs, merged over one or more run files."""
    out: dict[str, dict[str, list[dict]]] = {}
    for p in ([path] if path.is_file() else sorted(path.glob("*.json"))):
        if p.name.endswith(".manifest.json") or "summary" in p.name:
            continue
        for ds, by in json.loads(p.read_text()).items():
            if isinstance(by, dict):
                for arm, runs in by.items():
                    if isinstance(runs, list) and runs:
                        out.setdefault(arm, {})[ds] = runs
    return out


def _per_screen(by_ds: dict[str, list[dict]]) -> dict[str, float]:
    return {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in runs])) for ds, runs in by_ds.items()}


def _hits(runs: list[dict]) -> np.ndarray:
    return np.array([r["cumulative_hits"][-1] for r in runs], dtype=float)


def _summ(x: np.ndarray) -> dict:
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}


def _vs(x: np.ndarray, ref: np.ndarray) -> dict:
    """mean(x) - mean(ref) for the pair tier: Welch, or one-sample against a deterministic side."""
    d = float(x.mean() - ref.mean())
    if len(x) > 1 and len(ref) > 1 and (x.std() > 0 or ref.std() > 0):
        t = stats.ttest_ind(x, ref, equal_var=False)
        se = np.sqrt(x.var(ddof=1) / len(x) + ref.var(ddof=1) / len(ref))
        return {"delta": d, "p": float(t.pvalue), "ci95": [d - 1.96 * se, d + 1.96 * se], "test": "welch"}
    rep, const = (x, float(ref.mean())) if len(x) > 1 else (ref, float(x.mean()))
    if rep.std() == 0:
        return {"delta": d, "p": None, "test": "both constant"}
    return {"delta": d, "p": float(stats.ttest_1samp(rep, const).pvalue), "test": "one-sample"}


def _ml_engine(tier: str) -> dict | None:
    """The C-arm's ML engine alone on a tier (the ablation without the LLM, and the grid's w = 0)."""
    if tier == "headline":
        s = ENG / "summary.json"
    elif tier == "full_pool":
        s = ENG / "tier_a" / "summary.json"
    else:
        a = _arms(ENG / tier).get("ml_engine_lightgbm") if (ENG / tier).exists() else None
        return None if not a else {"hits": _hits(a["Horlbeck"])}
    return {"per_screen": json.loads(s.read_text())["rows"]["ml_engine_lightgbm"]["per_screen"]} if s.exists() else None


def ablations() -> dict:
    res = {}
    for tier in SINGLE + PAIRS:
        f = ABL / tier / "ablations_20rep.json"
        if not f.exists():
            continue
        arms = _arms(f)
        ml = _ml_engine(tier)
        if tier in SINGLE:
            c = _per_screen(arms["waddington_c"])
            rows = {"waddington_c": {"mean": float(np.mean(list(c.values()))), "per_screen": c}}
            for v in VARIANTS:
                ps = _per_screen(arms[v])
                rows[v] = {"mean": float(np.mean(list(ps.values()))), "per_screen": ps, "vs_c": _paired(ps, c)}
            if ml:
                ps = ml["per_screen"]
                rows["no_llm"] = {"mean": float(np.mean(list(ps.values()))), "per_screen": ps, "vs_c": _paired(ps, c)}
        else:
            c = _hits(arms["waddington_c"]["Horlbeck"])
            rows = {"waddington_c": _summ(c)}
            for v in VARIANTS:
                x = _hits(arms[v]["Horlbeck"])
                rows[v] = {**_summ(x), "vs_c": _vs(x, c)}
            if ml:
                rows["no_llm"] = {**_summ(ml["hits"]), "vs_c": _vs(ml["hits"], c)}
        res[tier] = rows
    return res


def fusion() -> dict:
    res = {}
    for tier in ("full_pool",) + PAIRS:
        d = FUS / tier
        if not d.exists():
            continue
        ref_arms = _arms(ABL / tier / "ablations_20rep.json")["waddington_c"]
        grid = {}
        ml = _ml_engine(tier)
        for w in ("0.1", "0.3", "0.4", "0.5"):
            f = d / f"carm_w{w}_5rep.json"
            if f.exists():
                grid[w] = _arms(f)["waddington_c"]
        if tier == "full_pool":
            ref = _per_screen(ref_arms)
            rows = {"0.2": {"mean": float(np.mean(list(ref.values()))), "replicates": 20}}
            if ml:
                rows["0.0"] = {"mean": float(np.mean(list(ml["per_screen"].values()))),
                               "vs_0.2": _paired(ml["per_screen"], ref)}
            for w, by in grid.items():
                ps = _per_screen(by)
                rows[w] = {"mean": float(np.mean(list(ps.values()))), "per_screen": ps, "vs_0.2": _paired(ps, ref)}
        else:
            ref = _hits(ref_arms["Horlbeck"])
            rows = {"0.2": _summ(ref)}
            if ml:
                rows["0.0"] = {**_summ(ml["hits"]), "vs_0.2": _vs(ml["hits"], ref)}
            for w, by in grid.items():
                x = _hits(by["Horlbeck"])
                rows[w] = {**_summ(x), "vs_0.2": _vs(x, ref)}
        res[tier] = rows
    return res


def pair_engines() -> dict:
    res = {}
    for tier in PAIRS:
        d = ENG / tier
        if not d.exists():
            continue
        arms = _arms(d)
        if "ml_engine_lightgbm" not in arms:
            continue
        ref = _hits(arms["ml_engine_lightgbm"]["Horlbeck"])
        rows = {}
        for arm, by in sorted(arms.items()):
            if not arm.startswith(("ml_engine_", "ml_static_")) or "Horlbeck" not in by:
                continue
            x = _hits(by["Horlbeck"])
            rows[arm] = {**_summ(x), "vs_lightgbm_online": None if arm == "ml_engine_lightgbm" else _vs(x, ref)}
        res[tier] = rows
    return res


def acquisition() -> dict:
    res = {}
    for tier, d in (("full_pool", ENG / "tier_a"), ("pairs", ENG / "pairs")):
        arms = {a: v for a, v in _arms(d).items() if a.startswith("ml_acq_")} if d.exists() else {}
        if "ml_acq_greedy" not in arms:
            continue
        if tier == "full_pool":
            g = _per_screen(arms["ml_acq_greedy"])
            res[tier] = {a: {"mean": float(np.mean(list(_per_screen(v).values()))),
                             "vs_greedy": None if a == "ml_acq_greedy" else _paired(_per_screen(v), g)}
                         for a, v in arms.items()}
        else:
            g = _hits(arms["ml_acq_greedy"]["Horlbeck"])
            res[tier] = {a: {**_summ(_hits(v["Horlbeck"])),
                             "vs_greedy": None if a == "ml_acq_greedy" else _vs(_hits(v["Horlbeck"]), g)}
                         for a, v in arms.items()}
    return res


def genedisco() -> dict:
    d = ENG / "tier_a"
    arms = {a: v for a, v in _arms(d).items() if a.startswith("gdfb_") and all(s in v for s in BENCH)} if d.exists() else {}
    if not arms:
        return {}
    ref = json.loads((d / "summary.json").read_text())["rows"]["ml_engine_lightgbm"]["per_screen"]
    rows = {a: {"mean": float(np.mean(list(_per_screen(v).values()))), "per_screen": _per_screen(v),
                "vs_lightgbm_online": _paired(_per_screen(v), ref)} for a, v in arms.items()}
    best = max(rows, key=lambda a: rows[a]["mean"])
    oracle = {s: max(r["per_screen"][s] for r in rows.values()) for s in BENCH}
    return {"rows": rows, "best_rule": best, "per_screen_oracle": float(np.mean(list(oracle.values())))}


def _pair_or(replay: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """hit ~ ml_rank_pct + endorsed on the pair tier's selected pairs; bootstrap over replicates."""
    from .conditional_attribution import _irls
    rows = []
    for r in replay["pooled"]["rounds"]:
        if not r.get("verified", True):
            continue
        for g in r["genes"]:
            rows.append((r["repeat"], float(g["ml_score"]), g["source"] in ("both", "llm_only"), bool(g["hit"])))
    rep = np.array([x[0] for x in rows])
    ml = np.array([x[1] for x in rows])
    rank = stats.rankdata(ml) / len(ml)
    X = np.column_stack([np.ones(len(rows)), rank, [float(x[2]) for x in rows]])
    y = np.array([float(x[3]) for x in rows])
    beta = _irls(X, y)
    reps = np.unique(rep)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(reps, size=len(reps), replace=True)
        idx = np.concatenate([np.flatnonzero(rep == k) for k in pick])
        boots.append(_irls(X[idx], y[idx])[2])
    return {"endorsed_odds_ratio": float(np.exp(beta[2])),
            "or_ci95": [float(np.exp(np.percentile(boots, 2.5))), float(np.exp(np.percentile(boots, 97.5)))],
            "n_pairs": int(len(rows)), "n_replicates": int(len(reps)), "bootstrap": "replicate-clustered"}


def attribution() -> dict:
    from .conditional_attribution import run as cond_run
    res = {}
    for tier in ("full_pool", "pairs"):
        f = RES / "tier_offline" / tier / "attribution_replay.json"
        if not f.exists():
            continue
        rep = json.loads(f.read_text())
        v = rep["verification"]
        row = {"verified_rounds": sum(x["verified"] for x in v.values()), "rounds": sum(x["rounds"] for x in v.values()),
               "by_source": rep["overall"]}
        if tier == "pairs":
            row["conditional"] = _pair_or(rep)
        else:
            c = cond_run(f, f.with_name("conditional_attribution.json"))
            row["conditional"] = {"tertiles": c["tertiles"], "logit": c["logit"], "marginal": c["marginal"]}
        res[tier] = row
    return res


def shap() -> dict:
    """Mean |SHAP| of the online model over the genes it selected, per screen and pooled (verified rounds)."""
    res = {}
    for tier in ("full_pool", "pairs"):
        f = RES / "tier_offline" / tier / "attribution_replay.json"
        if not f.exists():
            continue
        acc: dict[str, dict[str, list[float]]] = {}
        for r in json.loads(f.read_text())["pooled"]["rounds"]:
            if r.get("verified", True):
                for feat, v in r["shap"].items():
                    acc.setdefault(r["dataset"], {}).setdefault(feat, []).append(abs(v))
        per = {ds: dict(sorted(((k, float(np.mean(v))) for k, v in fs.items()), key=lambda kv: -kv[1]))
               for ds, fs in acc.items()}
        feats = {k for fs in per.values() for k in fs}
        pooled = {k: float(np.mean([fs[k] for fs in per.values() if k in fs])) for k in feats}
        res[tier] = {"per_screen_top3": {ds: list(fs.items())[:3] for ds, fs in per.items()},
                     "pooled": dict(sorted(pooled.items(), key=lambda kv: -kv[1]))}
    return res


def full_pool_per_screen() -> dict:
    """Table tiers' full-pool column, per screen: the six arms of experiment 08, GeneDisco (equal information,
    its best rule on the headline tier) and BDA's agent."""
    from .sources import REPORTED_TIER_A_RUN
    main = _arms(REPORTED_TIER_A_RUN)
    cols = {"random": main["random"], "coreset": main["coreset"], "static_ranker": main["static_ranker"],
            "online_ml": main["online_adaptive"], "padded_llm": main["llm_reasoning"], "waddington_c": main["waddington_c"]}
    gd = _arms(ENG / "port" / "tier_a_gdfb_coreset_seeded.json").get("gdfb_achilles_bnn_coreset")
    bda = _arms(RES / "gpt" / "tier_a_full_pool_20rep" / "bda_controlled.json").get("bda")
    if gd:
        cols["genedisco"] = gd
    if bda:
        cols["bda"] = bda
    per = {k: _per_screen(v) for k, v in cols.items()}
    return {"per_screen": per, "mean": {k: float(np.mean([v[s] for s in BENCH])) for k, v in per.items()}}


def pair_features(seeds: int = 20) -> dict:
    """Top-k hits on K562's pair map when ranking by one interaction feature (|r| for co-essentiality), or by
    the repaired arm's equal-weight rank sum of all eight; ties broken in label-free orders, averaged."""
    import pandas as pd
    from ..arms.base import label_free_order
    from ..arms.waddington_c_pair_arm import interaction_prior
    t = REPO / "workspace" / "evaluation" / "lgbm_training_data_horlbeck_pairs.csv"
    df = pd.read_csv(t, usecols=lambda c: c.startswith("pair_") or c in ("gene", "dataset", "label"))
    df = df[df["dataset"] == "Horlbeck"].reset_index(drop=True)
    y = df["label"].to_numpy()
    cols = [c for c in df.columns if c.startswith("pair_")]
    scores = {c: (df[c].abs() if c.endswith("coessential") else df[c]).to_numpy(float) for c in cols}
    scores["rank_sum_all"] = df["gene"].map(interaction_prior(df[["gene"] + cols])).to_numpy(float)
    out = {}
    for name, sc in scores.items():
        row = {}
        for k in (160, 640):
            vals = []
            for seed in range(seeds):
                order = label_free_order(df["gene"].tolist(), seed)
                idx = order[np.argsort(-sc[order], kind="stable")[:k]]
                vals.append(int(y[idx].sum()))
            row[f"top{k}"] = float(np.mean(vals))
        row["nonzero"] = float((sc != 0).mean())
        out[name] = row
    return out


def rules_on_engines() -> dict:
    """The acquisition rules on the Bayesian MLP (against its greedy engine) and UCB on the sparse GP (against
    its Probability-of-Hit engine), on the full pool and the pairs (experiment 15, bnn_rules / svgp_ucb)."""
    res = {}
    for tier, d in (("full_pool", ENG / "tier_a"), ("pairs", ENG / "pairs")):
        if not d.exists():
            continue
        arms = _arms(d)
        rows = {}
        for ref, prefix in (("ml_engine_bnn", "ml_bnnacq_"), ("ml_engine_svgp", "ml_svgpacq_")):
            if ref not in arms:
                continue
            for a in (x for x in arms if x.startswith(prefix)):
                if tier == "full_pool":
                    if not all(s in arms[a] for s in BENCH):
                        continue
                    ps, r = _per_screen(arms[a]), _per_screen(arms[ref])
                    rows[a] = {"reference": ref, "mean": float(np.mean(list(ps.values()))),
                               "reference_mean": float(np.mean(list(r.values()))), "vs_reference": _paired(ps, r)}
                else:
                    x, r = _hits(arms[a]["Horlbeck"]), _hits(arms[ref]["Horlbeck"])
                    rows[a] = {"reference": ref, **_summ(x), "reference_mean": float(r.mean()), "vs_reference": _vs(x, r)}
        res[tier] = rows
    return res


def genedisco_scores() -> dict:
    """GeneDisco's rules fed the revealed genes' continuous scores (its own information setting), full pool."""
    d = ENG / "tier_a"
    arms = {a: v for a, v in _arms(d).items() if a.startswith("gdf_") and all(s in v for s in BENCH)} if d.exists() else {}
    if not arms:
        return {}
    ref = json.loads((d / "summary.json").read_text())["rows"]["ml_engine_lightgbm"]["per_screen"]
    rows = {a: {"mean": float(np.mean(list(_per_screen(v).values()))), "per_screen": _per_screen(v),
                "vs_lightgbm_online": _paired(_per_screen(v), ref)} for a, v in arms.items()}
    best = max(rows, key=lambda a: rows[a]["mean"])
    oracle = {s: max(r["per_screen"][s] for r in rows.values()) for s in BENCH}
    return {"rows": rows, "best_rule": best, "per_screen_oracle": float(np.mean(list(oracle.values()))),
            "n_rules": len(rows)}


def full_pool_probes() -> dict:
    """What the prior learns, the matched-LOO comparisons, the LLM feature probe and the hit-frequency system
    on the full pool; the LLM feature probe and the static prior's permutation importance on the pairs."""
    off = RES / "tier_offline"
    out = {}
    f = off / "full_pool" / "prior_probes.json"
    if f.exists():
        pp = json.loads(f.read_text())
        ps = pp["per_screen"]
        out["prior_probes"] = {"summary": {k: v for k, v in pp["summary"].items() if k != "anchor_leakage"},
                               "loo_minus_hit_frequency": _paired({d: ps[d]["loo_all"] for d in BENCH},
                                                                  {d: ps[d]["hit_frequency"] for d in BENCH})}
        h = RES / "hitfreq_system" / "full_pool" / "hitfreq_c_20rep.json"
        if h.exists():
            from .sources import REPORTED_TIER_A_RUN
            hs = _per_screen(_arms(h)["waddington_c"])
            cs = _per_screen(_arms(REPORTED_TIER_A_RUN)["waddington_c"])
            hf = {d: ps[d]["hit_frequency"] for d in BENCH}
            out["hit_frequency_system"] = {"mean": float(np.mean(list(hs.values()))),
                                           "c_minus_it": _paired(cs, hs), "it_minus_static": _paired(hs, hf)}
    for name, path in (("final_stats", off / "full_pool" / "final_system_stats.json"),
                       ("llm_feature_probe", off / "full_pool" / "llm_feature_probe.json"),
                       ("llm_feature_probe_pairs", off / "pairs" / "llm_feature_probe.json"),
                       ("prior_permutation_pairs", off / "pairs" / "prior_permutation.json")):
        if path.exists():
            out[name] = json.loads(path.read_text())
    return out


def novel_hits() -> dict:
    f = RES / "tier_offline" / "full_pool" / "novel_hit_analysis.json"
    return json.loads(f.read_text()) if f.exists() else {}


def run() -> dict:
    res = {"ablations": ablations(), "fusion": fusion(), "pair_engines": pair_engines(),
           "acquisition": acquisition(), "genedisco_full_pool": genedisco(), "attribution": attribution(),
           "shap": shap(), "novel_hits_full_pool": novel_hits(), "full_pool_per_screen": full_pool_per_screen(),
           "pair_features": pair_features(), "rules_on_engines": rules_on_engines(),
           "genedisco_scores_full_pool": genedisco_scores(), "full_pool_probes": full_pool_probes()}
    OUT.write_text(json.dumps(res, indent=2))
    return res


def _fmt(v: dict | None) -> str:
    if v is None:
        return "reference"
    if "wins" in v:
        return f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['wins']}/{v['ties']}/{v['losses']}"
    p = "-" if v.get("p") is None else f"{v['p']:.2g}"
    return f"{v['delta']:+.1f} (p={p})"


if __name__ == "__main__":
    r = run()
    for tier, rows in r["ablations"].items():
        print(f"[ablation {tier}]")
        for a, v in rows.items():
            print(f"  {a:26s} {v['mean']:.4f}  {_fmt(v.get('vs_c'))}")
    for tier, rows in r["fusion"].items():
        print(f"[fusion {tier}]")
        for w in sorted(rows):
            print(f"  w={w}  {rows[w]['mean']:.4f}  {_fmt(rows[w].get('vs_0.2'))}")
    for tier, rows in r["pair_engines"].items():
        print(f"[pair engines {tier}]")
        for a, v in sorted(rows.items(), key=lambda kv: -kv[1]["mean"]):
            print(f"  {a:26s} {v['mean']:.1f}  {_fmt(v['vs_lightgbm_online'])}")
    for tier, rows in r["acquisition"].items():
        print(f"[acquisition {tier}]")
        for a, v in rows.items():
            print(f"  {a:20s} {v['mean']:.4f}  {_fmt(v['vs_greedy'])}")
    if r["genedisco_full_pool"]:
        g = r["genedisco_full_pool"]
        print(f"[GeneDisco full pool] best {g['best_rule']} {g['rows'][g['best_rule']]['mean']:.4f}; oracle {g['per_screen_oracle']:.4f}")
    for tier, v in r["attribution"].items():
        print(f"[attribution {tier}] verified {v['verified_rounds']}/{v['rounds']}; "
              + ", ".join(f"{s} {x['hit_rate']*100:.1f}% (n={x['picked']})" for s, x in v["by_source"].items()))
        lg = v["conditional"].get("logit", v["conditional"])
        print(f"   endorsed OR {lg['endorsed_odds_ratio']:.2f} [{lg['or_ci95'][0]:.2f}, {lg['or_ci95'][1]:.2f}]")
    for tier, v in r["shap"].items():
        print(f"[shap {tier}] pooled top: " + ", ".join(f"{k} {x:.3f}" for k, x in list(v["pooled"].items())[:5]))
    nh = r["novel_hits_full_pool"]
    if nh:
        print("[novel hits full pool]", nh["totals"], {k: {a: round(b, 3) for a, b in v.items()} for k, v in nh["macro_avg"].items()})
    fp = r["full_pool_per_screen"]["mean"]
    print("[full pool means]", {k: round(v, 4) for k, v in fp.items()})
    print("[pair features]", {k: (v["top160"], v["top640"]) for k, v in r["pair_features"].items()})
    for tier, rows in r["rules_on_engines"].items():
        print(f"[rules on engines {tier}]")
        for a, v in rows.items():
            print(f"  {a:28s} {v['mean']:.4f}  vs {v['reference']} ({v['reference_mean']:.4f}) {_fmt(v['vs_reference'])}")
    g = r["genedisco_scores_full_pool"]
    if g:
        b = g["rows"][g["best_rule"]]
        print(f"[GeneDisco scores, full pool] {g['n_rules']} rules; best {g['best_rule']} {b['mean']:.4f} "
              f"{_fmt(b['vs_lightgbm_online'])}; oracle {g['per_screen_oracle']:.4f}")
    fp = r["full_pool_probes"]
    if "hit_frequency_system" in fp:
        hs = fp["hit_frequency_system"]
        print(f"[full pool hit-frequency system] {hs['mean']:.4f}; C - it {_fmt(hs['c_minus_it'])}; it - static {_fmt(hs['it_minus_static'])}")
    print(f"saved -> {OUT}")
