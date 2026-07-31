"""
conditional_attribution.py — is LLM endorsement predictive *beyond* the ML score?

The headline attribution (Explainability section) shows genes that BOTH the ML and the LLM endorse hit
at 46.5%, vs.14.7% for ML-only picks. A reviewer rightly asked whether that is a real effect of
endorsement or a confound: agreement genes might simply be the ones with the highest ML score, so the
LLM adds nothing once the ML score is held fixed.

This probe controls for the ML score two ways, on the SELECTED genes of the eight weighted-route screens
(the `two_stage` screen K562-GWPS is excluded: there the ML shortlists and the LLM picks from the
shortlist, so the "would the ML have picked it" counterfactual — and therefore the source label — is
degenerate and every pick is tagged llm_only):

  1. Stratified: bin each screen's selected genes into ML-score deciles (within screen, so per-screen
     model scale cancels), then within each decile compare the hit rate of LLM-endorsed vs. non-endorsed
     genes. If endorsement helps at matched ML score, the endorsed hit rate exceeds the non-endorsed one
     inside deciles, not just marginally.
  2. Logistic regression: hit ~ ml_rank_pct + endorsed, pooled over the eight screens with screen fixed
     effects. The sign and magnitude of the `endorsed` coefficient is the ML-score-adjusted value of
     endorsement; we report it as a conditional odds ratio with a screen-clustered bootstrap CI.

    python -m waddington_select.analysis.conditional_attribution
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ATTR = REPO / "workspace" / "results" / "attribution_9ds.json"
OUT = REPO / "workspace" / "results" / "conditional_attribution.json"

# two_stage screen: source attribution is degenerate (LLM picks from the ML shortlist), so drop it.
EXCLUDE = {"Replogle_K562_gwps"}


def _rows_from_pooled(attr: dict, screen_of_round: list[str]) -> list[dict]:
    rows = []
    for r, ds in zip(attr["pooled"]["rounds"], screen_of_round):
        if ds in EXCLUDE:
            continue
        for g in r["genes"]:
            if "hit" not in g:
                continue
            rows.append({
                "screen": ds,
                "ml_score": float(g["ml_score"]),
                "endorsed": g["source"] in ("both", "llm_only"),
                "hit": bool(g["hit"]),
            })
    return rows


def _screen_order(attr: dict) -> list[str]:
    """pooled.rounds are 5 consecutive rounds per screen, in per_dataset key order."""
    order = []
    for ds in attr["per_dataset"]:
        order += [ds] * 5
    return order


def _decile_table(rows: list[dict]) -> dict:
    """Within-screen ML-score deciles, then pooled endorsed-vs-not hit rates per decile."""
    by_screen: dict[str, list[dict]] = {}
    for r in rows:
        by_screen.setdefault(r["screen"], []).append(r)
    # assign within-screen decile (0..9) by ml_score rank
    for rs in by_screen.values():
        rs.sort(key=lambda r: r["ml_score"])
        n = len(rs)
        for i, r in enumerate(rs):
            r["decile"] = min(9, int(10 * i / n))
    cells: dict[int, dict[bool, list[int]]] = {d: {True: [], False: []} for d in range(10)}
    for r in rows:
        cells[r["decile"]][r["endorsed"]].append(int(r["hit"]))
    table = []
    for d in range(10):
        e, ne = cells[d][True], cells[d][False]
        table.append({
            "decile": d,
            "endorsed_n": len(e), "endorsed_hit_rate": (sum(e) / len(e)) if e else None,
            "not_endorsed_n": len(ne), "not_endorsed_hit_rate": (sum(ne) / len(ne)) if ne else None,
        })
    return {"deciles": table}


def _irls(X, y, iters=50):
    """Newton/IRLS logistic fit; returns the coefficient vector. Pure numpy, no statsmodels dep."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        W = np.clip(p * (1 - p), 1e-6, None)
        z = X @ b + (y - p) / W
        XtW = X.T * W
        try:
            b = np.linalg.solve(XtW @ X + 1e-6 * np.eye(X.shape[1]), XtW @ z)
        except np.linalg.LinAlgError:
            break
    return b


def _design(rr, screens):
    """hit ~ intercept + within-screen ml percentile + endorsed + screen fixed effects."""
    by_s: dict[str, list[dict]] = {}
    for r in rr:
        by_s.setdefault(r["screen"], []).append(r)
    for rs in by_s.values():
        rs.sort(key=lambda r: r["ml_score"])
        n = len(rs)
        for i, r in enumerate(rs):
            r["ml_pct"] = (i + 0.5) / n
    X = [[1.0, r["ml_pct"], 1.0 if r["endorsed"] else 0.0]
         + [1.0 if r["screen"] == s else 0.0 for s in screens[1:]] for r in rr]
    y = [1.0 if r["hit"] else 0.0 for r in rr]
    return np.array(X), np.array(y)


def _or_jackknife(rows: list[dict]) -> dict:
    """Leave-one-screen-out robustness for the endorsement OR: with only eight clusters, a single screen
    could drive the effect. Drop each screen, refit, and report the range of the adjusted OR."""
    all_screens = sorted({r["screen"] for r in rows})
    per = {}
    for drop in all_screens:
        sub = [dict(r) for r in rows if r["screen"] != drop]
        screens = sorted({r["screen"] for r in sub})
        X, y = _design(sub, screens)
        per[drop] = float(np.exp(_irls(X, y)[2]))
    vals = list(per.values())
    return {"per_dropped_screen": per, "or_min": min(vals), "or_max": max(vals),
            "all_above_1": bool(all(v > 1 for v in vals))}


def _logit_or(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """hit ~ ml_rank_pct + endorsed with screen fixed effects (one-hot). Return the endorsed odds ratio
    and a screen-clustered bootstrap CI (resample whole screens). Pure-numpy IRLS, no statsmodels dep."""
    screens = sorted({r["screen"] for r in rows})
    by_s: dict[str, list[dict]] = {}
    for r in rows:
        by_s.setdefault(r["screen"], []).append(r)

    X, y = _design(rows, screens)
    endorsed_coef = float(_irls(X, y)[2])  # index 2 = endorsed

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(screens, size=len(screens), replace=True)
        rr = [r for s in pick for r in by_s[s]]
        if len({r["endorsed"] for r in rr}) < 2:
            continue
        Xb, yb = _design(rr, sorted({r["screen"] for r in rr}))
        try:
            boots.append(float(_irls(Xb, yb)[2]))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return {
        "endorsed_logodds": endorsed_coef,
        "endorsed_odds_ratio": float(np.exp(endorsed_coef)),
        "or_ci95": [float(np.exp(lo)), float(np.exp(hi))],
        "leave_one_screen_out": _or_jackknife(rows),
        "n_screens": len(screens), "n_genes": len(rows),
        "note": "OR>1 with CI excluding 1 => LLM endorsement predicts hits after adjusting for ML rank.",
    }


def run() -> dict:
    attr = json.loads(ATTR.read_text())
    rows = _rows_from_pooled(attr, _screen_order(attr))
    res = {
        "n_rows": len(rows),
        "excluded": sorted(EXCLUDE),
        "marginal": {
            "endorsed": _rate([r for r in rows if r["endorsed"]]),
            "not_endorsed": _rate([r for r in rows if not r["endorsed"]]),
        },
        **_decile_table(rows),
        "logit": _logit_or(rows),
    }
    OUT.write_text(json.dumps(res, indent=2))
    return res


def _rate(rr):
    return {"n": len(rr), "hit_rate": (sum(r["hit"] for r in rr) / len(rr)) if rr else None}


if __name__ == "__main__":
    r = run()
    print(f"rows={r['n_rows']}  (excluded {r['excluded']})")
    print(f"marginal: endorsed {r['marginal']['endorsed']['hit_rate']*100:.1f}% "
          f"(n={r['marginal']['endorsed']['n']})  vs  "
          f"not {r['marginal']['not_endorsed']['hit_rate']*100:.1f}% "
          f"(n={r['marginal']['not_endorsed']['n']})")
    print("\nML-score decile |  endorsed hit%(n)  |  not-endorsed hit%(n)")
    for d in r["deciles"]:
        e = f"{d['endorsed_hit_rate']*100:4.1f}% ({d['endorsed_n']})" if d["endorsed_hit_rate"] is not None else "   -   "
        ne = f"{d['not_endorsed_hit_rate']*100:4.1f}% ({d['not_endorsed_n']})" if d["not_endorsed_hit_rate"] is not None else "   -   "
        print(f"      {d['decile']:2d}        |  {e:18s} |  {ne}")
    lg = r["logit"]
    print(f"\nlogistic (hit ~ ml_rank + endorsed + screen FE):")
    print(f"  endorsed OR = {lg['endorsed_odds_ratio']:.2f}  "
          f"95% CI [{lg['or_ci95'][0]:.2f}, {lg['or_ci95'][1]:.2f}]  "
          f"(screen-clustered bootstrap, {lg['n_genes']} genes / {lg['n_screens']} screens)")
    jk = lg["leave_one_screen_out"]
    print(f"  leave-one-screen-out OR range: [{jk['or_min']:.2f}, {jk['or_max']:.2f}]  "
          f"(all above 1: {jk['all_above_1']})")
    for s, v in sorted(jk["per_dropped_screen"].items(), key=lambda kv: kv[1]):
        print(f"      drop {s:24s} OR={v:.2f}")
    print(f"saved -> {OUT}")
