"""
norman_transfer.py — does a ranker trained on a second genetic-interaction screen transfer to Horlbeck?
(paper: "A second combinatorial screen, and why it does not rescue the prior")

Train on the 130 scoreable Norman pairs (workspace/evaluation/build_norman_gi.py; the phenotype and hit
rule are ours), score all 100,576 Horlbeck pairs on the eight pair-interaction features, and report AUC
and hits among the top 160 / 640. The two gene sets are disjoint, so nothing passes except through the
features. Rows:

  Norman classifier / regressor  the original 2026-09-13 configuration (small LightGBM: 7 leaves,
                                 min_child_samples 5, 200 trees), on the binary label / on |GI|
  static LOO prior               the static_ranker arm on the pair table (nine single-gene screens)
  equal-weight rank sum          label-free; |co-essentiality| plus the other seven features
  random                         expected hits

A sensitivity panel retrains the Norman ranker with two other learners (the C-arm's own LightGBM
settings; class-balanced logistic regression), since 130 pairs with 14 hits is a small training set.

    python -m waddington_select.analysis.norman_transfer
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
NORMAN = REPO / "workspace" / "data" / "norman_gi"
PAIR_FEATS = REPO / "workspace" / "evaluation" / "pair_interaction_features.csv"
PAIR_TABLE = REPO / "workspace" / "evaluation" / "lgbm_training_data_horlbeck_pairs.csv"
BDA = REPO / "workspace" / "data" / "bda_benchmark"
OUT = REPO / "workspace" / "results" / "norman_transfer.json"
BUDGETS = (160, 640)

SMALL_LGBM = dict(learning_rate=0.05, n_estimators=200, num_leaves=7, min_child_samples=5,
                  verbose=-1, n_jobs=4, random_state=42)


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a.upper(), b.upper())))


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    nf = pd.read_csv(NORMAN / "norman_pair_features.csv")
    ng = pd.read_csv(NORMAN / "norman_gi_pairs.csv")
    norman = nf.merge(ng[["pair", "gi_score", "label"]], left_on="gene", right_on="pair")
    hf = pd.read_csv(PAIR_FEATS)
    gt = pd.read_csv(BDA / "ground_truth_Horlbeck.csv")
    keys = [_pair_key(a, b) for a, b in (ast.literal_eval(s) for s in gt["Gene_pairs"])]
    hits = {_pair_key(a, b) for a, b in np.load(BDA / "topmovers_Horlbeck.npy", allow_pickle=True)}
    horlbeck = hf.merge(pd.DataFrame({"gene": keys, "y": [k in hits for k in keys]}), on="gene")
    feats = [c for c in hf.columns if c != "gene"]
    return norman, horlbeck, feats


def static_prior_scores(horlbeck: pd.DataFrame) -> np.ndarray:
    """The static_ranker arm's LOO scores on the Horlbeck pairs (trained on the nine single-gene screens)."""
    from ..arms.static_ranker_arm import FEATURE_COLS, LGBM_PARAMS
    from ..tables import with_pair_spread
    df = pd.read_csv(PAIR_TABLE)
    df["gene"] = df["gene"].str.strip().str.upper()
    feats = with_pair_spread([c for c in FEATURE_COLS if c in df.columns], df)
    train, test = df[df["dataset"] != "Horlbeck"], df[df["dataset"] == "Horlbeck"]
    pos, neg = (train["label"] == 1).sum(), (train["label"] == 0).sum()
    m = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=neg / pos, random_state=42)
    m.fit(train[feats].values, train["label"].values)
    s = pd.Series(m.predict_proba(test[feats].values)[:, 1], index=test["gene"].values)
    return s.reindex(horlbeck["gene"].str.upper()).values


def diagnostic_loo() -> dict:
    """The diagnosis paragraph ("the pair encoding is orthogonal to the label"): a leave-one-phenotype-out
    LightGBM on the v3 pair table's symmetric encoding (mean and |difference| of the two genes' features,
    DepMap and anchor features included), and the largest |r| between any of those features and the label.
    The pair-interaction columns (pair_*) were added to the table after this diagnosis and are excluded.
    Not the static_ranker arm, which uses fewer features."""
    df = pd.read_csv(PAIR_TABLE.with_name(PAIR_TABLE.stem + "_v3.csv"))
    feat = [c for c in df.columns if c not in ("gene", "label", "dataset") and not c.startswith("pair_")]
    tr, te = df[df["dataset"] != "Horlbeck"], df[df["dataset"] == "Horlbeck"]
    m = lgb.LGBMClassifier(objective="binary", learning_rate=0.05, n_estimators=300, num_leaves=31, verbose=-1,
                           n_jobs=4, random_state=42,
                           scale_pos_weight=(tr["label"] == 0).sum() / max((tr["label"] == 1).sum(), 1))
    m.fit(tr[feat].values, tr["label"].values)
    p, y = m.predict_proba(te[feat].values)[:, 1], te["label"].values
    r = [abs(np.corrcoef(te[c].values, y)[0, 1]) for c in feat if te[c].std() > 0]
    out = {"auc": float(roc_auc_score(y, p)), "hits_top640": int(y[np.argsort(-p)[:640]].sum()),
           "random_top640": float(640 * y.mean()), "max_abs_r": float(max(r)), "n_train_rows": int(len(tr))}

    # each gene's measured solo effect, read from its (gene, negative-control) pair; rank the real pairs
    # by |solo_a| + |solo_b|
    gt = pd.read_csv(BDA / "ground_truth_Horlbeck.csv")
    pairs = [tuple(x.upper() for x in ast.literal_eval(s)) for s in gt["Gene_pairs"]]
    solo = {(a if b == "NEGATIVE" else b): s for (a, b), s in zip(pairs, gt["Score"]) if "NEGATIVE" in (a, b)}
    hits = {_pair_key(a, b) for a, b in np.load(BDA / "topmovers_Horlbeck.npy", allow_pickle=True)}
    real = [(a, b) for a, b in pairs if "NEGATIVE" not in (a, b)]
    ys = np.array([_pair_key(a, b) in hits for a, b in real], int)
    s = np.array([abs(solo.get(a, 0.0)) + abs(solo.get(b, 0.0)) for a, b in real])
    order = np.lexsort((np.random.default_rng(0).random(len(s)), -s))
    out["solo_effect"] = {"n_negative_control_pairs": len(solo), "auc": float(roc_auc_score(ys, s)),
                          "hits_top640": int(ys[order[:640]].sum())}
    return out


def run() -> dict:
    norman, horlbeck, F = load()
    y = horlbeck["y"].values.astype(int)
    rng = np.random.default_rng(0)

    def top(s: np.ndarray, k: int) -> int:  # ties broken at random, never by row order
        return int(y[np.lexsort((rng.random(len(s)), -s))[:k]].sum())

    def row(s: np.ndarray, trained_on: str) -> dict:
        return {"trained_on": trained_on, "auc": float(roc_auc_score(y, s)),
                **{f"hits_top{k}": top(s, k) for k in BUDGETS}}

    Xn, yn = norman[F].values, norman["label"].values
    spw = (yn == 0).sum() / max((yn == 1).sum(), 1)
    rows = {}
    m = lgb.LGBMClassifier(objective="binary", **SMALL_LGBM, scale_pos_weight=spw).fit(Xn, yn)
    rows["norman_classifier"] = row(m.predict_proba(horlbeck[F].values)[:, 1], "130 Norman pairs, 14 hits")
    m = lgb.LGBMRegressor(objective="regression", **SMALL_LGBM).fit(Xn, np.abs(norman["gi_score"].values))
    rows["norman_regressor"] = row(m.predict(horlbeck[F].values), "130 Norman pairs, |GI|")
    rows["static_loo_prior"] = row(static_prior_scores(horlbeck), "nine single-gene screens")
    z = np.zeros(len(horlbeck))
    for c in F:
        v = np.abs(horlbeck[c].values) if c == "pair_coessential" else horlbeck[c].values
        z += pd.Series(v).rank(pct=True).values
    rows["equal_weight_rank_sum"] = row(z, "nothing")
    rows["random"] = {"trained_on": "-", "auc": 0.5, **{f"hits_top{k}": float(k * y.mean()) for k in BUDGETS}}

    # sensitivity: the same 130 Norman pairs, two other learners
    from ..arms.online_adaptive_arm import LGBM_PARAMS as CARM_LGBM
    sens = {}
    m = lgb.LGBMClassifier(**CARM_LGBM, scale_pos_weight=spw).fit(Xn, yn)
    sens["norman_lightgbm_carm_params"] = row(m.predict_proba(horlbeck[F].values)[:, 1], "130 Norman pairs")
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(Xn, yn)
    sens["norman_logreg_balanced"] = row(m.predict_proba(horlbeck[F].values)[:, 1], "130 Norman pairs")

    res = {"n_norman_pairs": int(len(norman)), "n_norman_hits": int(yn.sum()),
           "n_horlbeck_pairs": int(len(horlbeck)), "n_horlbeck_hits": int(y.sum()),
           "features": F, "rows": rows, "sensitivity": sens, "diagnostic_loo_all_features": diagnostic_loo()}
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    print(f"Norman {r['n_norman_pairs']} pairs ({r['n_norman_hits']} hits) -> "
          f"Horlbeck {r['n_horlbeck_pairs']} pairs ({r['n_horlbeck_hits']} hits)")
    for block in ("rows", "sensitivity"):
        print(f"[{block}]")
        for k, v in r[block].items():
            print(f"  {k:32s} AUC {v['auc']:.4f}  top160 {v['hits_top160']:>6}  top640 {v['hits_top640']:>6}")
    print("[diagnostic LOO, symmetric pair encoding]", r["diagnostic_loo_all_features"])
