#!/usr/bin/env python3
"""
Compute anchor-independent biological features for Waddington Plan B.

Features:
  pli_score         — gnomAD v2.1.1 pLI (0→1; high = loss-of-function intolerant)
  string_degree_norm — degree in STRING v12 human PPI (combined_score ≥ 700), normalised

These are GLOBAL (not anchor-specific) and should generalise across biological
contexts better than archs4_coexpr / kegg_overlap / g1_ppi_score.

Outputs:
  workspace/data/universal_features.parquet   — gene → feature values
  workspace/data/universal_features.csv        — same, for inspection

Usage:
  python3 prep_universal_features.py
  python3 prep_universal_features.py --no-string   # skip STRING (pLI only)
"""

import argparse
import gzip
import io
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT  = Path(__file__).resolve().parents[2]
DATA_DIR   = REPO_ROOT / "workspace" / "data"
OUT_CSV    = DATA_DIR / "universal_features.csv"

# STRING downloads (v12, human 9606)
STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download"
    "/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL  = (
    "https://stringdb-downloads.org/download"
    "/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
)
STRING_SCORE_THRESHOLD = 700   # high-confidence interactions only

# gnomAD v2.1.1 gene constraint metrics
GNOMAD_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad"
    "/release/2.1.1/constraint/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"
)

# Local cache paths
CACHE_DIR          = DATA_DIR / "_universal_cache"
STRING_LINKS_CACHE = CACHE_DIR / "string_links.txt.gz"
STRING_INFO_CACHE  = CACHE_DIR / "string_info.txt.gz"
GNOMAD_CACHE       = CACHE_DIR / "gnomad_lof.txt.gz"


def download(url: str, dest: Path, desc: str = "") -> None:
    if dest.exists():
        print(f"  Already cached: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {desc} …")
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    done  = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = 100 * done / total
                print(f"\r    {pct:5.1f}%  ({done / 1e6:.0f} / {total / 1e6:.0f} MB)",
                      end="", flush=True)
    print(f"\r    Done ({done / 1e6:.0f} MB)          ")


# ---------------------------------------------------------------------------
# gnomAD pLI
# ---------------------------------------------------------------------------

def compute_pli() -> pd.DataFrame:
    """Return DataFrame with columns: gene (symbol), pli_score."""
    download(GNOMAD_URL, GNOMAD_CACHE, "gnomAD v2.1.1 constraint")

    print("  Parsing gnomAD constraint file …", flush=True)
    # .bgz is block gzip — readable with standard gzip.open
    rows = []
    with gzip.open(GNOMAD_CACHE, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        gene_col = header.index("gene")
        pli_col  = header.index("pLI")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            gene = parts[gene_col].strip().upper()
            try:
                pli = float(parts[pli_col])
            except (ValueError, IndexError):
                pli = 0.0
            rows.append({"gene": gene, "pli_score": round(pli, 4)})

    df = pd.DataFrame(rows)
    # Keep highest pLI per gene (some genes have multiple transcript entries)
    df = df.groupby("gene", as_index=False)["pli_score"].max()
    print(f"  gnomAD: {len(df)} unique genes, "
          f"{(df['pli_score'] > 0.9).sum()} with pLI > 0.9")
    return df


# ---------------------------------------------------------------------------
# STRING global degree
# ---------------------------------------------------------------------------

def _build_ensp_to_symbol(info_path: Path) -> dict[str, str]:
    """Map Ensembl protein IDs → gene symbols from STRING protein.info."""
    mapping: dict[str, str] = {}
    with gzip.open(info_path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        prot_col   = header.index("protein_external_id") if "protein_external_id" in header else 0
        symbol_col = header.index("preferred_name")       if "preferred_name"       in header else 1
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            ensp   = parts[prot_col].strip()
            symbol = parts[symbol_col].strip().upper()
            if ensp and symbol:
                mapping[ensp] = symbol
    print(f"  STRING info: {len(mapping)} Ensembl protein ID → symbol mappings")
    return mapping


def compute_string_degree() -> pd.DataFrame:
    """Return DataFrame with columns: gene (symbol), string_degree_norm."""
    download(STRING_LINKS_URL, STRING_LINKS_CACHE, "STRING v12 human PPI links")
    download(STRING_INFO_URL,  STRING_INFO_CACHE,  "STRING v12 human protein info")

    print("  Building Ensembl → gene symbol mapping …", flush=True)
    ensp2sym = _build_ensp_to_symbol(STRING_INFO_CACHE)

    print(f"  Counting STRING interactions (score ≥ {STRING_SCORE_THRESHOLD}) …",
          flush=True)
    degree: dict[str, int] = {}
    n_edges = 0
    with gzip.open(STRING_LINKS_CACHE, "rt") as fh:
        header = fh.readline()   # skip header
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            score = int(parts[2])
            if score < STRING_SCORE_THRESHOLD:
                continue
            a = ensp2sym.get(parts[0], "")
            b = ensp2sym.get(parts[1], "")
            if a:
                degree[a] = degree.get(a, 0) + 1
            if b:
                degree[b] = degree.get(b, 0) + 1
            n_edges += 1

    print(f"  STRING: {n_edges} high-confidence interactions, "
          f"{len(degree)} genes with at least 1 partner")

    max_deg = max(degree.values()) if degree else 1
    rows = [{"gene": g, "string_degree_norm": round(c / max_deg, 4)}
            for g, c in degree.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(include_string: bool = True) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== gnomAD pLI ===")
    df_pli = compute_pli()

    if include_string:
        print("\n=== STRING global degree ===")
        df_str = compute_string_degree()
        df = df_pli.merge(df_str, on="gene", how="outer")
        df["pli_score"] = df["pli_score"].fillna(0.0)
        df["string_degree_norm"] = df["string_degree_norm"].fillna(0.0)
    else:
        df = df_pli
        df["string_degree_norm"] = 0.0

    df = df.sort_values("gene").reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(df)} genes to:")
    print(f"  {OUT_CSV}")
    print("\nSample:")
    print(df.sort_values("pli_score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-string", dest="no_string", action="store_true",
                        help="Skip STRING download (pLI only, much faster)")
    args = parser.parse_args()
    main(include_string=not args.no_string)
