#!/usr/bin/env python3
"""
Prepare Replogle K562 Essential Perturb-seq data for the G5 BenchmarkEval framework.

Input : perturb_processed.h5ad  (GEARS format, local)
Output: ground_truth_Replogle_K562_essential.csv  (Gene, Score)
        topmovers_Replogle_K562_essential.npy      (hit gene names)

Hit definition:
  L2 distance between perturbed-cell mean expression and control mean.
  Top-10% by L2 (threshold = p90) = "hit" (strong phenotypic effect).
  Genes in CEGv2 are filtered out (same convention as other BDA datasets).

Usage:
  conda run -n waddington-bio python3 workspace/evaluation/prep_replogle.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

H5AD_PATH = Path("/home/duanyu/scouter-repro/scouter_misc/data/Data_GEARS"
                 "/replogle_k562_essential/perturb_processed.h5ad")
BDA_DIR   = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/datasets")
CEG_PATH  = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")
OUT_NAME  = "Replogle_K562_essential"
HIT_PERCENTILE = 90  # top 10% by L2 = hit


def load_essential() -> set[str]:
    ceg = pd.read_csv(CEG_PATH, sep="\t")
    return set(ceg["GENE"].dropna().str.strip().str.upper())


def compute_l2_scores(h5ad_path: Path) -> list[tuple[str, float, int]]:
    """Return [(gene, l2_dist_from_ctrl, n_cells), ...] sorted by L2 desc."""
    import anndata as ad
    print(f"Loading {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} cells × {adata.n_vars} genes")

    ctrl_mean = np.asarray(adata[adata.obs["control"] == 1].X.mean(axis=0)).flatten()
    print(f"  Control cells: {(adata.obs['control'] == 1).sum()}")

    conditions = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
    results = []
    for cond in conditions:
        gene = cond.split("+")[0].upper()
        cells = adata[adata.obs["condition"] == cond]
        if cells.n_obs < 5:
            continue
        pm = np.asarray(cells.X.mean(axis=0)).flatten()
        l2 = float(np.linalg.norm(pm - ctrl_mean))
        results.append((gene, l2, cells.n_obs))

    results.sort(key=lambda x: -x[1])
    return results


def main():
    essential = load_essential()

    results = compute_l2_scores(H5AD_PATH)

    # Filter CEGv2 essential genes (BDA convention)
    results_filtered = [(g, l2, n) for g, l2, n in results if g not in essential]
    print(f"\nUniverse: {len(results)} genes → after CEGv2 filter: {len(results_filtered)}")

    l2s = [l2 for _, l2, _ in results_filtered]
    threshold = float(np.percentile(l2s, HIT_PERCENTILE))
    print(f"L2 p{HIT_PERCENTILE} threshold: {threshold:.3f}")

    hits = [g for g, l2, _ in results_filtered if l2 >= threshold]
    print(f"Hits (L2 ≥ threshold): {len(hits)} ({100*len(hits)/len(results_filtered):.1f}%)")

    # Ground truth CSV: all universe genes sorted by L2 score desc
    df = pd.DataFrame([(g, round(l2, 4)) for g, l2, _ in results_filtered],
                      columns=["Gene", "Score"])
    csv_path = BDA_DIR / f"ground_truth_{OUT_NAME}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Topmovers NPY: hit gene names
    npy_path = BDA_DIR / f"topmovers_{OUT_NAME}.npy"
    np.save(npy_path, np.array(hits))
    print(f"Saved: {npy_path}")

    print(f"\nTop 20 hits:")
    for g, l2, n in results_filtered[:20]:
        tag = " ← hit" if g in hits else ""
        print(f"  {g:15s}  L2={l2:.3f}  cells={n}{tag}")


if __name__ == "__main__":
    main()
