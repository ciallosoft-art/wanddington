#!/usr/bin/env python3
"""
Preprocess DepMap 24Q4 CRISPRGeneEffect.csv into BDA-compatible
ground_truth / topmovers files for use as additional training data.

Design:
  Universe : all 17 916 screened genes MINUS CEGv2 core essentials
  Hits     : Chronos score < HIT_THRESHOLD (cell-line-specific dependencies)
  Anchors  : top N_ANCHORS most-dependent genes per cell line (after CEG filter)
             These are excluded from "hits" to avoid circularity.

Cell-line selection: greedy maximum-spread over the binary hit matrix so the
50 chosen lines are as biologically diverse as possible.

Outputs (per cell line, into workspace/data/depmap/processed/):
  ground_truth_DepMap_{ACH_ID}.csv   — "Gene" column + placeholder scores
  topmovers_DepMap_{ACH_ID}.npy     — hit gene symbols (string array)
  anchors_DepMap_{ACH_ID}.json      — anchor gene symbols (list of str)

Usage:
  conda run -n waddington-bio python3 prep_depmap.py
  conda run -n waddington-bio python3 prep_depmap.py --n-lines 30
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).resolve().parents[2]
DEPMAP_DIR         = REPO_ROOT / "workspace" / "data" / "depmap"
OUT_DIR            = DEPMAP_DIR / "processed"
CEG_PATH           = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")
DEPMAP_CSV         = DEPMAP_DIR / "CRISPRGeneEffect.csv"
SUPER_ESSENTIAL_TXT = DEPMAP_DIR / "super_essentials.txt"   # essential in >80% of cell lines

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

HIT_THRESHOLD = -0.5    # Chronos ≤ this → "hit" (dependent gene)
N_ANCHORS     = 12      # top-N most-dependent genes per cell line → anchors
N_CELL_LINES  = 50      # number of cell lines to select
MIN_HITS      = 100     # minimum hits per cell line (after CEG filter)
MAX_HIT_FRAC  = 0.20    # maximum fraction of universe that can be hits
MIN_COVERAGE  = 0.85    # minimum fraction of genes that must be non-NaN


def load_ceg(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t")
    return set(df["GENE"].dropna().str.strip().str.upper())


def load_super_essentials(path: Path) -> set[str]:
    """Genes essential in >80% of DepMap cell lines — too universal for anchors."""
    if not path.exists():
        return set()
    return {g.strip().upper() for g in path.read_text().split("\n") if g.strip()}


def parse_gene_symbol(col: str) -> str:
    """Extract gene symbol from 'SYMBOL (ENTREZ_ID)' format."""
    m = re.match(r"^([A-Za-z0-9_\-\.]+)\s*\(", col)
    return m.group(1).upper() if m else col.strip().upper()


def select_diverse_lines(
    hit_matrix: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> list[str]:
    """
    Greedy maximum-spread selection of n cell lines.
    Starts with the line that has the median hit count, then iteratively
    picks the line least correlated with already-selected lines.
    """
    rng = np.random.default_rng(seed)
    # Binary hit indicators
    B = hit_matrix.values.astype(np.float32)           # (n_lines, n_genes)
    norms = np.sqrt((B ** 2).sum(axis=1, keepdims=True)) + 1e-9
    B_norm = B / norms                                 # unit vectors

    n_lines = len(hit_matrix)
    # Start with the line closest to median hit count
    hit_counts = B.sum(axis=1)
    med = np.median(hit_counts)
    start = int(np.argmin(np.abs(hit_counts - med)))

    selected = [start]
    similarity = B_norm @ B_norm[start]                # shape (n_lines,)

    for _ in range(n - 1):
        # Mask already-selected
        similarity[selected] = np.inf
        nxt = int(np.argmin(similarity))
        # Update similarity: take running max (closest to any selected)
        similarity = np.maximum(similarity, B_norm @ B_norm[nxt])
        selected.append(nxt)

    return [hit_matrix.index[i] for i in selected]


def process(n_cell_lines: int = N_CELL_LINES) -> None:
    print(f"Loading CEGv2 from {CEG_PATH}...")
    ceg = load_ceg(CEG_PATH)
    super_ess = load_super_essentials(SUPER_ESSENTIAL_TXT)
    # Combined exclusion: CEGv2 ∪ genes essential in >80% of DepMap cell lines
    # The second set catches ribosomal proteins, chaperones etc. not in CEGv2
    exclude_set = ceg | super_ess
    print(f"  CEGv2={len(ceg)}, super_essentials={len(super_ess)}, union={len(exclude_set)} genes excluded")

    print(f"\nLoading DepMap CRISPRGeneEffect.csv ({DEPMAP_CSV.stat().st_size / 1e6:.0f} MB)...")
    df = pd.read_csv(DEPMAP_CSV, index_col=0)
    print(f"  Raw matrix: {df.shape[0]} cell lines × {df.shape[1]} genes")

    # Strip gene names to symbols
    gene_map  = {col: parse_gene_symbol(col) for col in df.columns}
    df.columns = [gene_map[c] for c in df.columns]

    # Deduplicate gene columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    # Remove CEGv2 + super-essential genes from universe
    keep_genes = [g for g in df.columns if g not in exclude_set]
    df = df[keep_genes]
    print(f"  After combined filter: {len(keep_genes)} genes in universe")

    # Filter cell lines by coverage and hit count
    coverage = df.notna().mean(axis=1)
    n_hits   = (df < HIT_THRESHOLD).sum(axis=1)
    n_univ   = len(keep_genes)

    mask = (
        (coverage >= MIN_COVERAGE) &
        (n_hits >= MIN_HITS) &
        (n_hits <= n_univ * MAX_HIT_FRAC)
    )
    df_filt = df.loc[mask].copy()
    print(f"  After QC filter: {len(df_filt)} eligible cell lines")
    print(f"  n_hits stats: min={int(n_hits[mask].min())}, "
          f"median={int(n_hits[mask].median())}, max={int(n_hits[mask].max())}")

    if len(df_filt) < n_cell_lines:
        print(f"  Warning: only {len(df_filt)} eligible lines; using all.")
        selected_ids = list(df_filt.index)
    else:
        print(f"\nSelecting {n_cell_lines} maximally diverse cell lines...")
        # Build binary hit matrix for diversity selection
        hit_mat = (df_filt < HIT_THRESHOLD).astype(np.int8)
        selected_ids = select_diverse_lines(hit_mat, n_cell_lines)
        print(f"  Selected: {selected_ids[:5]} ...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for cl_id in selected_ids:
        row = df_filt.loc[cl_id]
        valid = row.dropna()
        universe = list(valid.index)

        # Top-N anchors: most negative Chronos scores
        sorted_scores = valid.sort_values()
        anchors = list(sorted_scores.head(N_ANCHORS).index)

        # Hits: Chronos < threshold, excluding anchors
        anchor_set = set(anchors)
        hits = [g for g in universe
                if valid[g] < HIT_THRESHOLD and g not in anchor_set]

        if len(hits) < MIN_HITS // 2:
            skipped += 1
            continue

        hit_ratio = len(hits) / len(universe)
        safe_id   = cl_id.replace("-", "_")

        # --- ground_truth CSV (same format as BDA) ---
        gt_df = pd.DataFrame({"Gene": universe, "Score": valid[universe].values})
        gt_path = OUT_DIR / f"ground_truth_DepMap_{safe_id}.csv"
        gt_df.to_csv(gt_path, index=False)

        # --- topmovers NPY ---
        tm_path = OUT_DIR / f"topmovers_DepMap_{safe_id}.npy"
        np.save(tm_path, np.array(hits, dtype=object))

        # --- anchors JSON ---
        anc_path = OUT_DIR / f"anchors_DepMap_{safe_id}.json"
        anc_path.write_text(json.dumps(anchors))

        print(f"  {cl_id}: {len(universe)} genes, {len(hits)} hits ({hit_ratio:.1%}), "
              f"anchors: {anchors[:3]}...")

    n_done = len(selected_ids) - skipped
    print(f"\nDone. {n_done} cell lines written to {OUT_DIR}")
    print(f"Skipped {skipped} cell lines with too few hits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-lines", type=int, default=N_CELL_LINES,
                        help=f"Number of cell lines to process (default {N_CELL_LINES})")
    args = parser.parse_args()
    process(n_cell_lines=args.n_lines)
