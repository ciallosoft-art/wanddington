"""
jurkat_heldout.py — the pre-registered analysis of the held-out test on Horlbeck's Jurkat map
(docs/jurkat_protocol.md). Written and committed before any Jurkat run.

  arms        every arm at both budgets (160 picks primary, 640 secondary): mean, s.d., 95% CI
  prior_alone the repaired arm's label-free interaction prior ranked on its own (top 160 / 640)
  comparisons the five pre-registered comparisons at each budget, Holm-adjusted within the budget:
                P1 verifier repair vs BDA's agent          predicted higher   (primary endpoint)
                P2 verifier repair vs the repair without LLM predicted >= 0
                P3 names-first repair vs the repair without LLM predicted lower
                P4 interaction prior alone vs random       predicted higher
                P5 reported C-arm vs BDA's agent           predicted lower
              Two replicated arms: Welch t-test. One deterministic arm: one-sample t-test of the
              replicated arm against its value. p is two-sided; the direction is checked separately.
  replicated  P1 and P2 both in the predicted direction with Holm-adjusted p < 0.05 at 160 picks
  transfer    secondary, exploratory: rankers trained on K562's pair labels, scored on the Jurkat pool;
              and ranking by the same pair's K562 GI score
  llm_picks   exploratory, added after the runs (2026-10-11; no pre-registered result depends on it): in
              each LLM arm, the share of the batch the LLM named and the hit rate of its named picks
              against the rest; for BDA's agent, the picks it actually made and its hits per pick

    python -m waddington_select.analysis.jurkat_heldout
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..arms.waddington_c_pair_arm import interaction_prior
from ..tables import INTERACTION_PREFIX, pair_token
from .sources import GPT, REPO

DATASET = "Horlbeck_Jurkat"
BUDGETS = {32: 160, 128: 640}
EVAL = REPO / "workspace" / "evaluation"
JURKAT_TABLE = EVAL / "lgbm_training_data_horlbeck_jurkat_pairs_v3.csv"
K562_TABLE = EVAL / "lgbm_training_data_horlbeck_pairs_v3.csv"
SCORES = REPO / "workspace" / "data" / "horlbeck2018" / "extracted" / "gene_gi_scores.csv"
OUT = REPO / "workspace" / "results" / "jurkat_heldout.json"
REPS = 20

# label -> (file stem, arm); file is <stem>_<REPS>rep.json, or <stem>_1rep.json for deterministic arms
ARMS = {
    "random": (f"random_{REPS}rep", "random"),
    "coreset": ("deterministic_1rep", "coreset"),
    "static_ranker": ("deterministic_1rep", "static_ranker"),
    "online_ml": ("deterministic_1rep", "online_adaptive"),
    "padded_llm": (f"llm_reasoning_{REPS}rep", "llm_reasoning"),
    "c_arm": (f"waddington_c_{REPS}rep", "waddington_c"),
    "repair_no_llm": ("pair_no_llm_1rep", "waddington_c_pair_no_llm"),
    "repair_names_first": (f"pair_names_first_{REPS}rep", "waddington_c_pair"),
    "repair_verifier": (f"pair_verifier_{REPS}rep", "waddington_c_pair"),
    "bda": (f"bda_{REPS}rep", "bda"),
}
# (id, a, b, predicted sign of mean(a) - mean(b)); "prior_alone" is the offline ranking
COMPARISONS = [
    ("P1", "repair_verifier", "bda", +1),
    ("P2", "repair_verifier", "repair_no_llm", 0),     # predicted >= 0
    ("P3", "repair_names_first", "repair_no_llm", -1),
    ("P4", "prior_alone", "random", +1),
    ("P5", "c_arm", "bda", -1),
]


def _rng_top(y: np.ndarray, score: np.ndarray, k: int, seed: int = 0) -> int:
    """Hits among the top k, score ties broken at random (never by row order)."""
    rng = np.random.default_rng(seed)
    return int(y[np.lexsort((rng.random(len(score)), -score))[:k]].sum())


def jurkat_pool() -> pd.DataFrame:
    df = pd.read_csv(JURKAT_TABLE)
    df["gene"] = df["gene"].str.strip().str.upper()
    return df[df["dataset"] == DATASET].reset_index(drop=True)


def load_arm(budget_batch: int, label: str) -> np.ndarray | None:
    stem, arm = ARMS[label]
    f = GPT / f"jurkat_b{budget_batch}_20rep" / f"{stem}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    runs = d.get(DATASET, {}).get(arm)
    if not runs:
        return None
    return np.array([r["cumulative_hits"][-1] for r in runs], dtype=float)


def load_runs(budget_batch: int, label: str) -> list[dict] | None:
    """The arm's per-replicate records (selections per round), or None if it has not been run."""
    stem, arm = ARMS[label]
    f = GPT / f"jurkat_b{budget_batch}_20rep" / f"{stem}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get(DATASET, {}).get(arm) or None


def llm_picks(pool: pd.DataFrame) -> dict:
    """Exploratory: how the LLM's named picks hit against the rest of the batch, and BDA's picks per run."""
    hits = set(pool.loc[pool["label"] == 1, "gene"])
    out = {}
    for batch, budget in BUDGETS.items():
        for label in ("padded_llm", "c_arm", "repair_names_first", "repair_verifier"):
            rs = load_runs(batch, label)
            if not rs or "selected_llm_named_genes_per_round" not in rs[0]:
                continue
            named, other, share = [], [], []
            for r in rs:
                for i, sel in enumerate(r["selected_genes_per_round"]):
                    nm = set(r["selected_llm_named_genes_per_round"][i])
                    named += [g in hits for g in sel if g in nm]
                    other += [g in hits for g in sel if g not in nm]
                    share.append(len(nm & set(sel)) / batch)
            out[f"{budget}:{label}"] = {"llm_share_of_batch": float(np.mean(share)),
                                        "hit_rate_llm_named": float(np.mean(named)) if named else None,
                                        "hit_rate_other": float(np.mean(other)) if other else None}
        rs = load_runs(batch, "bda")
        if rs:
            picks = np.array([sum(len(b) for b in r["selected_genes_per_round"]) for r in rs], dtype=float)
            h = np.array([r["cumulative_hits"][-1] for r in rs], dtype=float)
            out[f"{budget}:bda"] = {"picks_mean": float(picks.mean()), "picks_min": int(picks.min()),
                                    "runs_short": int((picks < budget).sum()), "n": len(rs),
                                    "hits_per_pick": float(h.sum() / picks.sum())}
    return out


def _summary(x: np.ndarray) -> dict:
    out = {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0}
    if len(x) > 1:
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        out["ci95"] = [float(x.mean() - h), float(x.mean() + h)]
    return out


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided p for mean(a) - mean(b); a one-sample test when one side is a single deterministic value."""
    delta = float(a.mean() - b.mean())
    if len(a) > 1 and len(b) > 1:
        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        dfree = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
        h = stats.t.ppf(0.975, dfree) * se
        return {"delta": delta, "p": p, "cohens_d": delta / sd if sd > 0 else None, "ci95": [delta - h, delta + h],
                "test": "welch"}
    rep, const, fixed = (a, float(b[0]), "b") if len(a) > 1 else (b, float(a[0]), "a")
    p = float(stats.ttest_1samp(rep, const).pvalue)
    h = stats.t.ppf(0.975, len(rep) - 1) * rep.std(ddof=1) / np.sqrt(len(rep))
    return {"delta": delta, "p": p, "ci95": [delta - h, delta + h], "test": "one-sample",
            "deterministic_side": fixed, "replicates_above_deterministic": int((rep > const).sum()), "n": len(rep)}


def _holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def transfer(pool: pd.DataFrame) -> dict:
    """Secondary: rankers trained on K562's pair labels, scored on the Jurkat pool."""
    import lightgbm as lgb
    k = pd.read_csv(K562_TABLE)
    k = k[k["dataset"] == "Horlbeck"]
    feats = [c for c in pool.columns if c.startswith(INTERACTION_PREFIX)]
    y = pool["label"].to_numpy()
    out = {}
    models = {
        "lightgbm_small": lgb.LGBMClassifier(objective="binary", learning_rate=0.05, n_estimators=200, num_leaves=7,
                                             min_child_samples=5, verbose=-1, n_jobs=4, random_state=42,
                                             scale_pos_weight=(k["label"] == 0).sum() / (k["label"] == 1).sum()),
        "logreg_balanced": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
    }
    for name, m in models.items():
        m.fit(k[feats].to_numpy(), k["label"].to_numpy())
        s = m.predict_proba(pool[feats].to_numpy())[:, 1]
        out[name] = {"auc": float(roc_auc_score(y, s)), **{f"top{n}": _rng_top(y, s, n) for n in BUDGETS.values()}}
    gi = pd.read_csv(SCORES)
    gi["gene"] = [pair_token(a, b) for a, b in zip(gi["gene_a"].astype(str), gi["gene_b"].astype(str))]
    k562 = gi.set_index("gene")["k562_avg_gi"].reindex(pool["gene"]).abs().fillna(-1.0).to_numpy()
    out["k562_abs_gi"] = {"auc": float(roc_auc_score(y, k562)), "covered": int((k562 >= 0).sum()),
                          **{f"top{n}": _rng_top(y, k562, n) for n in BUDGETS.values()}}
    return out


def run() -> dict:
    pool = jurkat_pool()
    y = pool["label"].to_numpy()
    prior = interaction_prior(pool)
    pscore = pool["gene"].map(prior).to_numpy()
    res = {"pool": len(pool), "hits": int(y.sum()),
           "prior_alone": {str(n): _rng_top(y, pscore, n) for n in BUDGETS.values()},
           "arms": {}, "comparisons": {}, "replicated": None}
    for batch, budget in BUDGETS.items():
        arms = {label: load_arm(batch, label) for label in ARMS}
        arms["prior_alone"] = np.array([res["prior_alone"][str(budget)]], dtype=float)
        res["arms"][str(budget)] = {k: _summary(v) for k, v in arms.items() if v is not None}
        rows = []
        for cid, a, b, sign in COMPARISONS:
            if arms.get(a) is None or arms.get(b) is None:
                continue
            c = _compare(arms[a], arms[b])
            c.update({"id": cid, "a": a, "b": b, "predicted": {1: "higher", 0: ">= 0", -1: "lower"}[sign],
                      "direction_ok": c["delta"] > 0 if sign > 0 else c["delta"] < 0 if sign < 0 else c["delta"] >= 0})
            rows.append(c)
        for c, adj in zip(rows, _holm([c["p"] for c in rows])):
            c["p_holm"] = adj
        res["comparisons"][str(budget)] = rows
    primary = {c["id"]: c for c in res["comparisons"].get("160", [])}
    if "P1" in primary and "P2" in primary:
        res["replicated"] = all(primary[i]["direction_ok"] and primary[i]["p_holm"] < 0.05 for i in ("P1", "P2"))
    res["transfer"] = transfer(pool)
    res["llm_picks"] = llm_picks(pool)
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    print(f"Jurkat pool {r['pool']} pairs, {r['hits']} hits; prior alone {r['prior_alone']}")
    for budget, arms in r["arms"].items():
        print(f"[{budget} picks] " + ", ".join(f"{k} {v['mean']:.1f}" + (f"±{v['sd']:.1f}" if v['n'] > 1 else "")
                                              for k, v in arms.items()))
        for c in r["comparisons"][budget]:
            print(f"   {c['id']} {c['a']} vs {c['b']}: {c['delta']:+.1f} p={c['p']:.2g} Holm {c['p_holm']:.2g} "
                  f"predicted {c['predicted']} -> {'ok' if c['direction_ok'] else 'NOT as predicted'}")
    print("replicated (P1 and P2 at 160 picks):", r["replicated"])
    print("transfer from K562:", json.dumps(r["transfer"]))
    for k, v in r["llm_picks"].items():
        print(f"   picks {k}: " + ", ".join(f"{a} {b:.3f}" if isinstance(b, float) else f"{a} {b}" for a, b in v.items()))
