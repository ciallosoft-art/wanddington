#!/usr/bin/env python3
"""
Compute anchor-independent pathway membership features for Waddington V7.

Features:
  kegg_pathway_count_norm    — number of KEGG pathways a gene belongs to (normalised)
  reactome_pathway_count_norm — number of Reactome pathways a gene belongs to (normalised)

Both are global gene properties, completely independent of anchor genes.
KEGG counts are derived from the existing _kegg_cache/ directory (no extra download).
Reactome counts are fetched via MyGene.info batch API and cached to _reactome_cache/.

Outputs:
  workspace/data/pathway_features.csv

Usage:
  python3 prep_pathway_features.py
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT         = Path(__file__).resolve().parents[2]
DATA_DIR          = REPO_ROOT / "workspace" / "data"
OUT_CSV           = DATA_DIR / "pathway_features.csv"
KEGG_CACHE_DIR    = Path(__file__).parent / "_kegg_cache"
REACTOME_CACHE_DIR = Path(__file__).parent / "_reactome_cache"

MYGENE_URL   = "https://mygene.info/v3/query"
BATCH_SIZE   = 500
RETRY_DELAY  = 5   # seconds between retries on API error


# ---------------------------------------------------------------------------
# KEGG pathway count — from existing _kegg_cache/
# ---------------------------------------------------------------------------

def compute_kegg_counts() -> pd.DataFrame:
    """
    Count KEGG pathways per gene using _kegg_cache/{GENE}.json files.
    Each file contains a list of KEGG pathway IDs the gene belongs to.
    """
    if not KEGG_CACHE_DIR.exists():
        print("  [WARNING] _kegg_cache/ not found — KEGG counts will be 0")
        return pd.DataFrame(columns=["gene", "kegg_pathway_count"])

    rows = []
    for f in KEGG_CACHE_DIR.glob("*.json"):
        gene = f.stem.upper()
        try:
            data = json.loads(f.read_text())
            count = len(data) if isinstance(data, list) else 0
        except Exception:
            count = 0
        rows.append({"gene": gene, "kegg_pathway_count": count})

    df = pd.DataFrame(rows)
    n_nonzero = (df["kegg_pathway_count"] > 0).sum()
    print(f"  KEGG: {len(df)} genes, {n_nonzero} with at least 1 pathway, "
          f"max={df['kegg_pathway_count'].max()}")
    return df


# ---------------------------------------------------------------------------
# Reactome pathway count — via MyGene.info batch API
# ---------------------------------------------------------------------------

def _fetch_reactome_batch(genes: list[str]) -> dict[str, int]:
    """
    Batch-query MyGene.info for Reactome pathway counts.
    Returns {GENE: count}.
    """
    data = json.dumps({
        "q":      [g.upper() for g in genes],
        "scopes": "symbol",
        "fields": "pathway.reactome",
        "species": "human",
    }).encode()
    req = urllib.request.Request(
        MYGENE_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            hits = json.loads(resp.read())
    except Exception as e:
        print(f"  [WARNING] MyGene.info error: {e}", flush=True)
        return {}

    result: dict[str, int] = {}
    for hit in hits:
        sym = hit.get("query", "").upper()
        pathways = hit.get("pathway", {}).get("reactome", [])
        if isinstance(pathways, dict):
            pathways = [pathways]
        result[sym] = len(pathways) if isinstance(pathways, list) else 0
    return result


def prefetch_reactome(genes: list[str], verbose: bool = True) -> None:
    """
    Pre-populate _reactome_cache/{GENE}.json for all genes.
    Skips genes already cached.
    """
    REACTOME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_fetch = [g for g in genes
                if not (REACTOME_CACHE_DIR / f"{g.upper()}.json").exists()]
    if not to_fetch:
        if verbose:
            print(f"  Reactome: all {len(genes)} genes already cached")
        return

    if verbose:
        print(f"  Reactome: fetching {len(to_fetch)} genes "
              f"({len(genes)-len(to_fetch)} cached)…", flush=True)

    for i in range(0, len(to_fetch), BATCH_SIZE):
        chunk = to_fetch[i: i + BATCH_SIZE]
        fetched = _fetch_reactome_batch(chunk)
        for g in chunk:
            cache_path = REACTOME_CACHE_DIR / f"{g.upper()}.json"
            count = fetched.get(g.upper(), 0)
            cache_path.write_text(json.dumps(count))
        done = min(i + BATCH_SIZE, len(to_fetch))
        if verbose:
            print(f"  Reactome: {done}/{len(to_fetch)} fetched", flush=True)
        time.sleep(0.5)   # polite delay


def compute_reactome_counts(genes: list[str]) -> pd.DataFrame:
    """
    Return DataFrame with columns: gene, reactome_pathway_count.
    Reads from _reactome_cache/ (populated by prefetch_reactome).
    """
    prefetch_reactome(genes, verbose=True)

    rows = []
    for g in genes:
        cache_path = REACTOME_CACHE_DIR / f"{g.upper()}.json"
        try:
            count = int(json.loads(cache_path.read_text()))
        except Exception:
            count = 0
        rows.append({"gene": g.upper(), "reactome_pathway_count": count})

    df = pd.DataFrame(rows)
    n_nonzero = (df["reactome_pathway_count"] > 0).sum()
    print(f"  Reactome: {len(df)} genes, {n_nonzero} with at least 1 pathway, "
          f"max={df['reactome_pathway_count'].max()}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== KEGG pathway counts (from _kegg_cache/) ===")
    df_kegg = compute_kegg_counts()

    # Collect all gene symbols from KEGG cache for Reactome query
    all_genes = df_kegg["gene"].tolist()
    print(f"\n=== Reactome pathway counts (MyGene.info, {len(all_genes)} genes) ===")
    df_reactome = compute_reactome_counts(all_genes)

    # Merge and normalise
    df = df_kegg.merge(df_reactome, on="gene", how="outer")
    df["kegg_pathway_count"]     = df["kegg_pathway_count"].fillna(0).astype(int)
    df["reactome_pathway_count"] = df["reactome_pathway_count"].fillna(0).astype(int)

    max_kegg     = df["kegg_pathway_count"].max()
    max_reactome = df["reactome_pathway_count"].max()
    df["kegg_pathway_count_norm"]     = (df["kegg_pathway_count"] / max(max_kegg, 1)).round(4)
    df["reactome_pathway_count_norm"] = (df["reactome_pathway_count"] / max(max_reactome, 1)).round(4)

    df = df.sort_values("gene").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(df)} genes to {OUT_CSV}")
    print(f"Normalisation: kegg_max={max_kegg}, reactome_max={max_reactome}")
    print("\nSample (top by reactome count):")
    print(df.nlargest(8, "reactome_pathway_count")[
        ["gene", "kegg_pathway_count", "kegg_pathway_count_norm",
         "reactome_pathway_count", "reactome_pathway_count_norm"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
