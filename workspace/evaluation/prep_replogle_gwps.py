#!/usr/bin/env python3
"""
Prepare Replogle K562 GWPS (Genome-Wide Perturb-seq) data for G5 BenchmarkEval.

Input : K562_gwps_normalized_bulk_01.h5ad  (Figshare bulk pseudo-bulk format)
Output: ground_truth_Replogle_K562_gwps.csv  (Gene, Score)
        topmovers_Replogle_K562_gwps.npy      (hit gene names)

Data format (bulk h5ad, 11258 obs × 8248 vars):
  Each obs = one perturbation condition (gene × guide combo)
  Index: "{idx}_{GENE}_{guide}_{ENSEMBL}"
  Key columns: anderson_darling_counts, energy_test_p_value, core_control

Hit definition:
  Score = max anderson_darling_counts across guides per gene.
  AD counts = number of expressed genes significantly shifted (A-D test).
  Top-10% by score = hit  (matches Replogle essential convention).
  CEGv2 filter applied (BDA convention).

Why anderson_darling_counts over energy_test_p_value:
  energy_test_p_value is clamped at 0.0001 for most significant genes,
  causing ~2000 genes to tie at the same score. AD counts have full range
  (0–5528, median=2) and cleanly rank by transcriptional program size.

Usage:
  conda run -n waddington-bio python3 workspace/evaluation/prep_replogle_gwps.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT      = Path(__file__).resolve().parents[2]
H5AD_PATH      = REPO_ROOT / "workspace/data/raw_h5ad/replogle_k562_gwps/K562_gwps_normalized_bulk_01.h5ad"
BDA_DIR        = REPO_ROOT / "workspace/data/bda_benchmark"
CEG_PATH       = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")
OUT_NAME       = "Replogle_K562_gwps"
HIT_PERCENTILE = 90  # top 10% by AD counts = hit


def load_essential() -> set[str]:
    ceg = pd.read_csv(CEG_PATH, sep="\t")
    return set(ceg["GENE"].dropna().str.strip().str.upper())


def parse_gene(idx: str) -> str:
    """'{num}_{GENE}_{guide}_{ENSEMBL}' → GENE (uppercase)."""
    parts = idx.split("_")
    return parts[1].upper() if len(parts) >= 2 else idx.upper()


def main():
    import anndata as ad

    essential = load_essential()
    print(f"CEGv2: {len(essential)} essential genes")

    print(f"Loading {H5AD_PATH} ...")
    d = ad.read_h5ad(H5AD_PATH, backed="r")
    print(f"  {d.n_obs} obs × {d.n_vars} vars")

    obs = d.obs.copy()
    obs["gene"] = [parse_gene(i) for i in obs.index]

    # Remove control rows
    n_ctrl = obs["core_control"].astype(bool).sum()
    obs = obs[~obs["core_control"].astype(bool)].copy()
    print(f"  Removed {n_ctrl} core_control rows → {len(obs)} perturbation rows")

    # Per-gene: take best (max) AD counts across guides
    gene_df = (
        obs.groupby("gene")["anderson_darling_counts"]
        .max()
        .reset_index()
        .rename(columns={"anderson_darling_counts": "Score"})
    )
    print(f"  Unique genes (after aggregating guides): {len(gene_df)}")

    # CEGv2 filter
    gene_df = gene_df[~gene_df["gene"].isin(essential)].reset_index(drop=True)
    print(f"  After CEGv2 filter: {len(gene_df)} genes")

    # Hit definition: top 10% by AD counts
    threshold = float(np.percentile(gene_df["Score"], HIT_PERCENTILE))
    print(f"AD counts p{HIT_PERCENTILE} threshold: {threshold:.0f}")
    hits = gene_df.loc[gene_df["Score"] >= threshold, "gene"].tolist()
    print(f"Hits (AD counts ≥ {threshold:.0f}): {len(hits)} ({100*len(hits)/len(gene_df):.1f}%)")

    # Sort by score descending
    gene_df = gene_df.sort_values("Score", ascending=False).reset_index(drop=True)

    # Ground truth CSV
    csv_path = BDA_DIR / f"ground_truth_{OUT_NAME}.csv"
    gene_df.rename(columns={"gene": "Gene"}).to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}  ({len(gene_df)} rows)")

    # Topmovers NPY
    npy_path = BDA_DIR / f"topmovers_{OUT_NAME}.npy"
    np.save(npy_path, np.array(hits))
    print(f"Saved: {npy_path}  ({len(hits)} hits)")

    print(f"\nTop 20 by AD counts:")
    hits_set = set(hits)
    for _, row in gene_df.head(20).iterrows():
        tag = " ← hit" if row["gene"] in hits_set else ""
        print(f"  {row['gene']:15s}  AD={row['Score']:.0f}{tag}")


if __name__ == "__main__":
    main()
