"""
screen_scores.py — continuous screen scores as a training target (docs/ml_engines_plan.md, phase 2 item 1).

The benchmark's hit labels are thresholds on a continuous per-gene screen score (the BDA ground-truth
files). GeneDisco regresses that score; our engines classify the thresholded label. This module turns a
screen's score into a hit-aligned target in [0, 1]:

  1. pick the transform t in {s, -s, |s|} whose ranking best separates the screen's own hits (ROC AUC);
  2. target = within-screen percentile of t(s), over the screen's candidate pool.

A screen whose hit list is not a threshold on its shipped score (best AUC < MIN_AUC; Sanchez21, where every
transform gives AUC ~0.5) gets no score target; callers put its rows on the same scale from the label.

Only TRAINING screens are read. The leave-one-screen-out engines call this for the other eight screens and
never for the target, and the target screen's revealed genes stay binary, so the protocol is unchanged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BDA_DIR = Path(__file__).resolve().parents[1] / "workspace" / "data" / "bda_benchmark"
# training-frame dataset name -> BDA ground-truth file stem (the rest match)
BDA_STEM = {"Carnevale22": "Carnevale22_Adenosine", "Steinhart": "Steinhart_crispra_GD2_D22"}
TRANSFORMS = {"s": lambda s: s, "-s": lambda s: -s, "|s|": np.abs}
MIN_AUC = 0.9


@lru_cache(maxsize=None)
def load_scores(dataset: str) -> pd.Series:
    """Upper-cased gene -> score for one screen (the Steinhart file ships with a '0,1' header)."""
    path = BDA_DIR / f"ground_truth_{BDA_STEM.get(dataset, dataset)}.csv"
    df = pd.read_csv(path)
    df.columns = ["Gene", "Score"]
    df["Gene"] = df["Gene"].astype(str).str.strip().str.upper()
    return df.drop_duplicates("Gene").set_index("Gene")["Score"].astype(float)


def aligned_target(dataset: str, genes, labels) -> tuple[np.ndarray | None, dict]:
    """Hit-aligned score percentile for one screen's rows, or None when the score does not define its hits.

    `genes`/`labels` are that screen's training rows. Returns (target, info) where info records the chosen
    transform, its AUC and the score coverage; rows without a score get NaN (callers impute from the label).
    """
    from sklearn.metrics import roc_auc_score
    s = load_scores(dataset).reindex([str(g).upper() for g in genes]).to_numpy()
    y = np.asarray(labels).astype(int)
    have = ~np.isnan(s)
    info = {"coverage": float(have.mean())}
    if have.sum() < 2 or len(np.unique(y[have])) < 2:
        return None, {**info, "transform": None, "auc": float("nan")}
    aucs = {k: float(roc_auc_score(y[have], f(s[have]))) for k, f in TRANSFORMS.items()}
    best = max(aucs, key=aucs.get)
    info.update(transform=best, auc=aucs[best], aucs=aucs)
    if aucs[best] < MIN_AUC:
        return None, info
    from scipy.stats import rankdata
    t = TRANSFORMS[best](s[have])
    out = np.full(len(s), np.nan)
    out[have] = (rankdata(t) - 1.0) / max(len(t) - 1.0, 1.0)
    return out, info


def score_targets(y, groups, genes, revealed: str = "imputed") -> tuple[np.ndarray, dict]:
    """Per-row regression target for an engine's training rows.

    Rows of score-defined training screens get their hit-aligned percentile. Everything else — the target
    screen's revealed genes (group "__target__"), screens whose score does not define their hits, and rows
    without a score — gets E[target | label], estimated on the score-defined rows, so a label sits on the
    same scale as a score. Returns (target, info) with per-screen transform/AUC and the two imputed values.

    revealed="hard" instead gives the target screen's revealed genes their bare label (0 or 1): a post-hoc
    mechanism check, since E[target | non-hit] ~0.48 makes a revealed non-hit a much weaker negative than
    the classifier's 0.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups, dtype=object)
    genes = np.asarray(genes, dtype=object)
    target = np.full(len(y), np.nan)
    screens = {}
    for g in pd.unique(groups):
        if g == "__target__":
            continue
        idx = np.flatnonzero(groups == g)
        t, info = aligned_target(g, genes[idx], y[idx])
        screens[g] = info
        if t is not None:
            target[idx] = t
    scored = ~np.isnan(target)
    if not scored.any():
        raise ValueError("no training screen has a score that defines its hits")
    imputed = {lab: float(target[scored & (y == lab)].mean()) if (scored & (y == lab)).any() else float(lab)
               for lab in (0, 1)}
    fill = ~scored
    target[fill] = np.where(y[fill] == 1, imputed[1], imputed[0])
    if revealed == "hard":
        tgt = groups == "__target__"
        target[tgt] = y[tgt]
    elif revealed != "imputed":
        raise ValueError(f"revealed must be 'imputed' or 'hard', not {revealed!r}")
    return target, {"screens": screens, "imputed": imputed, "n_imputed": int(fill.sum()), "revealed": revealed}
