#!/usr/bin/env python3
"""
Build the *easier* BDA difficulty tier: 1-gene selection over the FULL gene pool.

BioDiscoveryAgent defines three tiers (Roohani et al., ICLR'25):

  easier   1-gene, full pool          <- this script
  harder   1-gene, non-essential      <- the frozen benchmark table (lgbm_training_data*.csv)
  hardest  2-gene combinations        <- build_horlbeck_pairs.py

The frozen tables drop every CEGv2 core-essential gene from each screen's pool *after* labelling, so
the committed benchmark is the middle tier. This script rebuilds the same feature frames with the
CEGv2 filter switched off, writing ISOLATED superset tables. The frozen files, the committed models
and the published results are never touched; a run selects a tier purely by env var:

    WADDINGTON_TRAINING_CSV=.../lgbm_training_data_tier_a.csv \
    WADDINGTON_TRAINING_CSV_V3=.../lgbm_training_data_tier_a_v3.csv \
        conda run -n waddington-bio python3 -m waddington_select.run_sequential ...

Everything else — the nine dataset names, batch sizes, task prompts, hit@R5 — is unchanged, so the
two tiers are directly comparable.

Faithfulness
------------
Only `kegg_overlap` is pool-dependent (it is normalised by the max overlap *within the screen*), so
a larger pool can rescale it. Every other feature is a per-gene lookup. The script asserts that the
non-essential rows of the new table reproduce the frozen table exactly outside `kegg_overlap`, and
reports the largest `kegg_overlap` drift, so any silent change to the shared pipeline is caught here
rather than in the results.

DepMap columns (v3) are reproduced from CRISPRGeneEffect.csv with the normalisation recovered from
the committed v3 table: a global, negatively-oriented min-max over all 17 916 screened genes
(essentials included), so adding essentials back does not move the scale.

Usage:
    conda run -n waddington-bio python3 workspace/evaluation/build_tier_a_unfiltered.py
    conda run -n waddington-bio python3 workspace/evaluation/build_tier_a_unfiltered.py --skip-fetch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import bootstrap_lgbm as B                                    # noqa: E402
from gene_ranker import prefetch_kegg_batch, get_kegg_pathways  # noqa: E402
from prep_pathway_features import prefetch_reactome, REACTOME_CACHE_DIR  # noqa: E402

FROZEN_V1 = HERE / "lgbm_training_data.csv"
FROZEN_V3 = HERE / "lgbm_training_data_v3.csv"
OUT_V1 = HERE / "lgbm_training_data_tier_a.csv"
OUT_V3 = HERE / "lgbm_training_data_tier_a_v3.csv"
PATHWAY_CSV = REPO_ROOT / "workspace" / "data" / "pathway_features.csv"
PATHWAY_OUT = REPO_ROOT / "workspace" / "data" / "pathway_features_tier_a.csv"
DEPMAP_CSV = REPO_ROOT / "workspace" / "data" / "depmap" / "CRISPRGeneEffect.csv"

DEPMAP_COLS = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm", "depmap_K562_norm"]

PPI_CACHE = HERE / "_ppi_cache"
# `hub_score_norm` and `ppi_score_sum` are computed over the WHOLE STRING PPI cache and normalised by
# its running maximum, so they silently rescale whenever a new anchor is queried. Eleven anchor files
# were added after the frozen benchmark table was written (the ferroptosis/p53 probe of 2026-07-13 and
# the Schmidt CRISPRa validation of 2026-08-30); including them shifts every gene's value by up to
# 0.04-0.05 and would make the two difficulty tiers incomparable. Excluding exactly these eleven
# reproduces the frozen table bit for bit — verified below, and asserted on every run.
POST_FREEZE_ANCHORS = {
    "TP53", "RB1", "BRCA2", "CDKN1A", "SLC7A11", "ACSL4", "FSP1",
    "STAT1", "JAK1", "JAK2", "IFNGR1",
}
K562_LINE = "ACH-000551"
CHRONOS_HIT = -0.5


# ---------------------------------------------------------------------------
# Pathway features over the *unfiltered* gene universe
# ---------------------------------------------------------------------------

def build_pathway_features(pool_genes: set[str], skip_fetch: bool) -> dict[str, tuple[float, float]]:
    """{gene: (kegg_norm, reactome_norm)} covering the unfiltered pools.

    Normalisation constants are taken from the committed table when the extra genes do not exceed
    them, so every gene already in the frozen benchmark keeps its exact value.
    """
    committed = pd.read_csv(PATHWAY_CSV)
    committed["gene"] = committed["gene"].astype(str).str.strip().str.upper()

    universe = sorted(set(committed["gene"]) | {g.upper() for g in pool_genes})
    missing_kegg = [g for g in universe if not (HERE / "_kegg_cache" / f"{g}.json").exists()]
    missing_reac = [g for g in universe if not (REACTOME_CACHE_DIR / f"{g}.json").exists()]
    print(f"  pathway universe: {len(universe)} genes "
          f"(KEGG missing {len(missing_kegg)}, Reactome missing {len(missing_reac)})")

    if not skip_fetch:
        if missing_kegg:
            prefetch_kegg_batch(missing_kegg, verbose=True)
        if missing_reac:
            prefetch_reactome(missing_reac, verbose=True)
    elif missing_kegg or missing_reac:
        print("  [--skip-fetch] uncached genes will get pathway count 0")

    kegg_cnt: dict[str, int] = {}
    reac_cnt: dict[str, int] = {}
    for g in universe:
        kp = HERE / "_kegg_cache" / f"{g}.json"
        try:
            kegg_cnt[g] = len(json.loads(kp.read_text()))
        except Exception:
            kegg_cnt[g] = 0
        rp = REACTOME_CACHE_DIR / f"{g}.json"
        try:
            reac_cnt[g] = int(json.loads(rp.read_text()))
        except Exception:
            reac_cnt[g] = 0

    # PIN the normalisers to the frozen table's constants. The divisor is an arbitrary scale, and
    # keeping it fixed is what puts both difficulty tiers on the same feature scale: every gene already
    # in the benchmark keeps its exact value. A handful of newly added essentials sit in more pathways
    # than any non-essential gene did, so their normalised count lands slightly above 1.0 — harmless
    # for a tree model, and preferable to silently rescaling all 20 000 other genes.
    max_kegg = int(committed["kegg_pathway_count"].max())
    max_reac = int(committed["reactome_pathway_count"].max())
    over_k = sum(1 for v in kegg_cnt.values() if v > max_kegg)
    over_r = sum(1 for v in reac_cnt.values() if v > max_reac)
    print(f"  normalisation pinned to the frozen table: kegg_max={max_kegg}, reactome_max={max_reac}"
          f" ({over_k} KEGG / {over_r} Reactome genes exceed it -> norm slightly >1)")

    out = pd.DataFrame({
        "gene": universe,
        "kegg_pathway_count": [kegg_cnt[g] for g in universe],
        "reactome_pathway_count": [reac_cnt[g] for g in universe],
    })
    out["kegg_pathway_count_norm"] = (out["kegg_pathway_count"] / max(max_kegg, 1)).round(4)
    out["reactome_pathway_count_norm"] = (out["reactome_pathway_count"] / max(max_reac, 1)).round(4)
    out.to_csv(PATHWAY_OUT, index=False)
    print(f"  wrote {PATHWAY_OUT} ({len(out)} genes)")
    return {r.gene: (float(r.kegg_pathway_count_norm), float(r.reactome_pathway_count_norm))
            for r in out.itertuples()}


# ---------------------------------------------------------------------------
# DepMap columns (v3), reproduced from the raw Chronos matrix
# ---------------------------------------------------------------------------

def build_depmap_map() -> dict[str, tuple[float, float, float, float]]:
    print(f"  reading {DEPMAP_CSV.name} ({DEPMAP_CSV.stat().st_size / 1e6:.0f} MB)...")
    df = pd.read_csv(DEPMAP_CSV, index_col=0)
    df.columns = [_sym(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    n_lines = len(df)
    frac = (df <= CHRONOS_HIT).sum(axis=0) / n_lines
    mean = df.mean(axis=0, skipna=True)
    mn = df.min(axis=0, skipna=True)
    k562 = df.loc[K562_LINE]

    def neg_minmax(s: pd.Series) -> pd.Series:
        lo, hi = float(s.min()), float(s.max())
        return ((hi - s) / (hi - lo)).round(4)

    out = pd.DataFrame({
        "depmap_frac_ess": frac,
        "depmap_mean_norm": neg_minmax(mean),
        "depmap_min_norm": neg_minmax(mn),
        "depmap_K562_norm": neg_minmax(k562),
    })
    print(f"  DepMap: {len(out)} genes, {n_lines} cell lines")
    return {g: tuple(float(v) for v in row) for g, row in out.iterrows()}


def frozen_ppi_cache_files() -> list[Path]:
    """The STRING PPI cache as it stood when the frozen benchmark table was built."""
    return sorted(f for f in PPI_CACHE.glob("*.json") if f.stem.upper() not in POST_FREEZE_ANCHORS)


def hub_scores_frozen_scope() -> dict[str, float]:
    """`compute_hub_scores`, restricted to the frozen cache scope."""
    counts: dict[str, int] = {}
    for f in frozen_ppi_cache_files():
        for g in json.loads(f.read_text()):
            counts[g] = counts.get(g, 0) + 1
    if not counts:
        return {}
    mx = max(counts.values())
    return {g: round(c / mx, 4) for g, c in counts.items()}


def ppi_sum_frozen_scope() -> dict[str, float]:
    """`compute_ppi_sum_from_cache`, restricted to the frozen cache scope."""
    totals: dict[str, float] = {}
    for f in frozen_ppi_cache_files():
        for gene, score in json.loads(f.read_text()).items():
            totals[gene] = totals.get(gene, 0.0) + score
    if not totals:
        return {}
    mx = max(totals.values())
    return {g: round(s / mx, 4) for g, s in totals.items()}


def _sym(col: str) -> str:
    m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*\(", col)
    return m.group(1).upper() if m else col.strip().upper()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_fetch: bool = False) -> None:
    print("Loading BDA screens WITHOUT the CEGv2 filter...")
    data: dict[str, pd.DataFrame] = {}
    for ds in B.DATASETS:
        df = B.load_dataset_labeled(ds, essential=set())   # <- the only substantive change
        data[ds] = df
        print(f"  {ds:26s} {len(df):6d} genes, {int(df['label'].sum()):5d} hits "
              f"({100 * df['label'].mean():.2f}%)")

    pool_genes = {g for df in data.values() for g in df["gene"]}
    print(f"\nUnfiltered universe: {len(pool_genes)} distinct genes")

    print("\nPathway features (KEGG + Reactome over the unfiltered universe)...")
    pathway_features = build_pathway_features(pool_genes, skip_fetch)

    print("\nUniversal features (pLI + STRING degree)...")
    univ_features = B.load_universal_features()

    print("\nSTRING PPI anchor expansion (cached, frozen-benchmark scope)...")
    g1_all = {ds: B.compute_g1_scores_for_dataset(ds) for ds in data}
    n_all = len(list(PPI_CACHE.glob("*.json")))
    n_frozen = len(frozen_ppi_cache_files())
    print(f"  cache files: {n_all} on disk, {n_frozen} in the frozen scope "
          f"({n_all - n_frozen} post-freeze anchors excluded)")
    hub_scores = hub_scores_frozen_scope()
    ppi_sum = ppi_sum_frozen_scope()
    print(f"  hub={len(hub_scores)} genes, ppi_sum={len(ppi_sum)} genes")

    print("\nBuilding feature frames (rank_norm=False, matching the frozen table)...")
    frames = B._precompute_features(
        data, {}, hub_scores, ppi_sum, g1_all,
        univ_features=univ_features, use_rank_norm=False,
        pathway_features=pathway_features,
    )
    v1 = pd.concat(frames.values(), ignore_index=True)
    v1["gene"] = v1["gene"].str.strip().str.upper()

    # ---- faithfulness check against the frozen benchmark ----
    print("\nVerifying the non-essential rows reproduce the frozen table...")
    frozen = pd.read_csv(FROZEN_V1)
    frozen["gene"] = frozen["gene"].str.strip().str.upper()
    key = ["dataset", "gene"]
    # BDA's Scharenberg22 ground truth lists 5 genes twice; both copies carry identical features, so
    # compare on unique keys (the tables themselves keep the duplicates, exactly as the benchmark does).
    frozen_u = frozen.drop_duplicates(key)
    v1_u = v1.drop_duplicates(key)
    merged = frozen_u.merge(v1_u, on=key, suffixes=("_old", "_new"), how="inner")
    if len(merged) != len(frozen_u):
        raise RuntimeError(
            f"frozen has {len(frozen_u)} unique (dataset, gene) keys but only {len(merged)} matched")
    worst: dict[str, float] = {}
    for col in B.FEATURE_COLS + ["label"]:
        diff = np.abs(merged[f"{col}_old"] - merged[f"{col}_new"])
        if diff.isna().any():
            raise RuntimeError(f"column {col} has NaN in the rebuild or the frozen table")
        d = float(diff.max())
        worst[col] = d
        flag = "" if d == 0.0 else ("   <- pool-dependent, expected" if col == "kegg_overlap" else "   <-- UNEXPECTED")
        print(f"  {col:30s} max|Δ| = {d:.6g}{flag}")
    bad = {c: d for c, d in worst.items() if d > 0 and c != "kegg_overlap"}
    if bad:
        raise RuntimeError(f"rebuild does not reproduce the frozen table on {bad}")
    print("  OK — identical except kegg_overlap (whose normaliser is the within-screen max).")

    v1.to_csv(OUT_V1, index=False)
    print(f"\nWrote {OUT_V1}  ({len(v1)} rows)")

    print("\nBuilding the v3 (DepMap) variant...")
    dm = build_depmap_map()
    v3 = v1.copy()
    for i, col in enumerate(DEPMAP_COLS):
        v3[col] = v3["gene"].map(lambda g, i=i: dm.get(g, (0.0, 0.0, 0.0, 0.0))[i]).astype(float)
    # A gene screened in DepMap but not measured in K562 (129 of them) yields NaN, not 0. The frozen
    # v3 table stores 0.0 for exactly these, same as for a gene absent from DepMap entirely — match it.
    n_nan = int(v3[DEPMAP_COLS].isna().sum().sum())
    if n_nan:
        print(f"  filling {n_nan} unmeasured DepMap cells with 0.0 (frozen-table convention)")
        v3[DEPMAP_COLS] = v3[DEPMAP_COLS].fillna(0.0)
    v3 = v3[list(pd.read_csv(FROZEN_V3, nrows=0).columns)]

    frozen3 = pd.read_csv(FROZEN_V3, usecols=key + DEPMAP_COLS)
    frozen3["gene"] = frozen3["gene"].str.strip().str.upper()
    m3 = frozen3.drop_duplicates(key).merge(
        v3[key + DEPMAP_COLS].drop_duplicates(key), on=key, suffixes=("_old", "_new"), how="inner")
    for col in DEPMAP_COLS:
        diff = np.abs(m3[f"{col}_old"] - m3[f"{col}_new"])
        if diff.isna().any():   # a NaN silently satisfies every `> tol` test — reject it explicitly
            raise RuntimeError(f"DepMap column {col} has NaN after the merge")
        d = float(diff.max())
        print(f"  {col:30s} max|Δ| vs frozen v3 = {d:.6g}")
        if d > 1e-4:
            raise RuntimeError(f"DepMap column {col} does not reproduce the frozen v3 table (Δ={d})")
    if v3.isna().any().any():
        raise RuntimeError(f"NaNs remain in the v3 table: "
                           f"{ {k: int(v) for k, v in v3.isna().sum().items() if v} }")
    if v1.isna().any().any():
        raise RuntimeError(f"NaNs in the v1 table: "
                           f"{ {k: int(v) for k, v in v1.isna().sum().items() if v} }")
    v3.to_csv(OUT_V3, index=False)
    print(f"Wrote {OUT_V3}  ({len(v3)} rows)")

    print("\nTier comparison (hit rate per screen):")
    print(f"  {'screen':26s} {'harder (frozen)':>18s} {'easier (full pool)':>20s}")
    for ds in B.DATASETS:
        f = frozen[frozen["dataset"] == ds]
        n = v1[v1["dataset"] == ds]
        print(f"  {ds:26s} {len(f):7d}/{100*f['label'].mean():6.2f}%   "
              f"{len(n):7d}/{100*n['label'].mean():6.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true",
                    help="do not call KEGG/MyGene.info; uncached genes get pathway count 0")
    main(**vars(ap.parse_args()))
