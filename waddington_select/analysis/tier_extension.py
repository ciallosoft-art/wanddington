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
  novel_hits    recurrent vs novel hits on the full pool (experiment 21)

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
        scores = [g["ml_score"] for g in r["genes"]]
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
    from .trace import hit_rate_by_source
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


def novel_hits() -> dict:
    f = RES / "tier_offline" / "full_pool" / "novel_hit_analysis.json"
    return json.loads(f.read_text()) if f.exists() else {}


def run() -> dict:
    res = {"ablations": ablations(), "fusion": fusion(), "pair_engines": pair_engines(),
           "acquisition": acquisition(), "genedisco_full_pool": genedisco(), "attribution": attribution(),
           "novel_hits_full_pool": novel_hits()}
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
    print(f"saved -> {OUT}")
