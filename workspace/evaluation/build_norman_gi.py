#!/usr/bin/env python3
"""
Derive a genetic-interaction dataset from the Norman Perturb-seq screen.

Why this exists
---------------
The combinatorial tier has exactly one dataset in the whole benchmark -- Horlbeck -- and every claim
we make about it rests on that single screen. It also has a structural gap: the cross-experiment
prior is trained on nine SINGLE-gene screens, which contain no pair interactions at all, so the
leave-one-phenotype-out prior sits at chance (AUC 0.4959; the static_ranker arm 0.4990) no matter how the
features are engineered. A second combinatorial screen is the only thing that can close either gap.

Norman et al.'s Perturb-seq is not a hit list; it is an expression matrix. Turning it into the same
shape as Horlbeck requires defining a phenotype and a hit rule, and those choices are OURS, not
BioDiscoveryAgent's. They are stated here rather than buried:

  phenotype   For a perturbation P, delta_P = mean expression over its cells minus mean over controls,
              a vector over the 5045 measured genes.
  expectation A genetic interaction is deviation from additivity, so the null for pair (A,B) is
              delta_A + delta_B.
  GI score    ||delta_AB|| - ||delta_A + delta_B||, signed. Negative means the double perturbation
              moved the transcriptome LESS than the two singles predict (buffering); positive means
              more (synergy). This mirrors Horlbeck's signed fitness-interaction score, so the same
              two-tailed hit rule applies for the same reason.
  hits        The most extreme ~5% in each tail by signed GI, matching BDA's convention on Horlbeck.

Scope and caveats, all of which matter when reading any transfer result:
  * Only pairs where BOTH genes also have a single-perturbation arm can be scored (130 of 179).
  * Norman is CRISPRa (over-expression); Horlbeck is CRISPRi (knock-down). The modality differs, the
    same way Steinhart differs from the other eight single-gene screens in this benchmark.
  * The two gene sets are disjoint (0 of 72 Norman genes appear among Horlbeck's 448). For transfer
    this is a feature, not a bug: nothing can pass between them except through the features.
  * 130 pairs is a small training set. The learning curve on Horlbeck suggests this is around where
    a within-screen model starts to beat chance, so treat any transfer result as directional.

Usage:
    conda run -n waddington-bio python3 workspace/evaluation/build_norman_gi.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
# GEARS' processed Norman Perturb-seq file (not shipped): $WADDINGTON_NORMAN_H5AD, else the author's local copy.
DEFAULT_H5AD = Path(os.environ.get("WADDINGTON_NORMAN_H5AD",
                                   "/home/duanyu/Python/Myproject/GEARS (副本)/data/norman/perturb_processed.h5ad"))
OUT_DIR = REPO_ROOT / "workspace" / "data" / "norman_gi"
TAIL_FRACTION = 0.05     # per tail, as BDA does on Horlbeck


def condition_means(adata, conditions: list[str]) -> dict[str, np.ndarray]:
    """Mean expression vector per condition, read in one pass over the backed matrix."""
    cond = adata.obs["condition"].astype(str).values
    out = {}
    for c in conditions:
        idx = np.flatnonzero(cond == c)
        if len(idx) == 0:
            continue
        block = adata[idx].to_memory().X
        out[c] = np.asarray(block.mean(axis=0)).ravel()
    return out


def main(h5ad: str = str(DEFAULT_H5AD)) -> None:
    import anndata as ad

    print(f"reading {Path(h5ad).name} ...")
    a = ad.read_h5ad(h5ad, backed="r")
    cond = a.obs["condition"].astype(str)
    uniq = list(cond.unique())
    singles = [c for c in uniq if c.endswith("+ctrl")]
    doubles = [c for c in uniq if c != "ctrl" and "+" in c and not c.endswith("+ctrl")]
    single_gene = {c.split("+")[0] for c in singles}
    usable = [c for c in doubles if all(g in single_gene for g in c.split("+"))]
    print(f"  {len(singles)} singles, {len(doubles)} doubles, {len(usable)} doubles fully scoreable")

    print("computing per-condition means ...")
    means = condition_means(a, ["ctrl"] + singles + usable)
    ctrl = means["ctrl"]
    delta = {c: means[c] - ctrl for c in means if c != "ctrl"}

    rows = []
    for c in usable:
        g1, g2 = c.split("+")
        d_ab = delta[c]
        expected = delta[f"{g1}+ctrl"] + delta[f"{g2}+ctrl"]
        obs_norm, exp_norm = float(np.linalg.norm(d_ab)), float(np.linalg.norm(expected))
        resid = float(np.linalg.norm(d_ab - expected))
        cos = float(d_ab @ expected / max(np.linalg.norm(d_ab) * np.linalg.norm(expected), 1e-12))
        rows.append({
            "gene_a": min(g1, g2), "gene_b": max(g1, g2),
            "pair": "|".join(sorted((g1.upper(), g2.upper()))),
            "gi_score": obs_norm - exp_norm,       # signed: <0 buffering, >0 synergy
            "residual_norm": resid,
            "cosine_obs_exp": cos,
            "observed_norm": obs_norm, "expected_norm": exp_norm,
            "n_cells": int((cond == c).sum()),
        })
    df = pd.DataFrame(rows)

    lo, hi = df["gi_score"].quantile([TAIL_FRACTION, 1 - TAIL_FRACTION])
    df["label"] = ((df["gi_score"] <= lo) | (df["gi_score"] >= hi)).astype(int)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.sort_values("gi_score").to_csv(OUT_DIR / "norman_gi_pairs.csv", index=False)
    (OUT_DIR / "provenance.json").write_text(json.dumps({
        "source_h5ad": str(h5ad),
        "cells": int(a.shape[0]), "measured_genes": int(a.shape[1]),
        "cell_type": sorted(a.obs["cell_type"].astype(str).unique()),
        "phenotype": "delta_P = mean(cells of P) - mean(control cells), over all measured genes",
        "gi_score": "||delta_AB|| - ||delta_A + delta_B||  (signed; <0 buffering, >0 synergy)",
        "hit_rule": f"two-tailed, most extreme {TAIL_FRACTION:.0%} per tail by signed gi_score",
        "note": "phenotype and hit rule are ours, not from BioDiscoveryAgent; Norman is CRISPRa "
                "while Horlbeck is CRISPRi, and the two gene sets are disjoint",
    }, indent=2))

    print(f"\nwrote {OUT_DIR/'norman_gi_pairs.csv'}: {len(df)} pairs, {int(df.label.sum())} hits "
          f"({100*df.label.mean():.1f}%)")
    print(f"  gi_score range [{df.gi_score.min():.3f}, {df.gi_score.max():.3f}], "
          f"cutoffs {lo:.3f} / {hi:.3f}")
    print("\n  most buffering:"); print(df.nsmallest(4, "gi_score")[["pair","gi_score"]].to_string(index=False))
    print("\n  most synergistic:"); print(df.nlargest(4, "gi_score")[["pair","gi_score"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=str(DEFAULT_H5AD))
    main(**vars(ap.parse_args()))
