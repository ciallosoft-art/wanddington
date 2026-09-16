"""
ingest.py — turn a CRISPR screen readout file into hit/no-hit labels for a tested batch.

This is the *real experiment* counterpart to the benchmark oracle (`tools.reveal`): instead of the
ground truth, the scientist provides their own screen readout each round, and we derive which of the
tested genes hit — the same way BioDiscoveryAgent does (`data/screen.py: identify_hits`).

Accepted files:
  - MAGeCK `gene_summary.txt` (tab-delimited; score column e.g. `pos|lfc` / `neg|lfc`)
  - a `Gene,Score` CSV (like BDA's `ground_truth_*.csv`)
  - any table with a gene-id column + a numeric score column, or an explicit hit/label column

Hit derivation (ported from BDA):
  - if a `hit`/`is_hit`/`label` column exists → use it directly (0/1 / bool / yes/no)
  - else `gaussian` (default): the top `top_ratio` (0.05) genes by |score| are hits
  - else `castle`: score > 95th percentile
  - or an explicit absolute `--threshold`

Given the tested `--genes` batch, returns which of them hit (batch ∩ hitset) and the genome-wide
hit count (for a cumulative hit_ratio comparable to the oracle path).

Usage:
    python -m waddington_select.ingest --file screen.txt --genes STAT1 JAK2 ACTB \
        [--score-col "pos|lfc"] [--top-ratio 0.05] [--method gaussian|castle] [--threshold X] --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_CANDIDATES = ["gene", "id", "gene_symbol", "symbol", "genes"]
SCORE_CANDIDATES = ["score", "log-fold-change", "lfc", "pos|lfc", "neg|lfc", "logfc", "z", "zscore"]
HIT_CANDIDATES = ["hit", "is_hit", "hits", "label", "hit_label"]

_TRUE = {"1", "true", "yes", "y", "hit", "t"}
_FALSE = {"0", "false", "no", "n", "miss", "f"}


def _norm(genes) -> list[str]:
    return list(dict.fromkeys(str(g).strip().upper() for g in (genes or []) if str(g).strip()))


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _read_table(path: str) -> pd.DataFrame:
    """Read a screen file (.csv → comma; .txt/.tsv/MAGeCK gene_summary → tab) and add _GENE."""
    if Path(path).suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    id_col = _pick(list(df.columns), ID_CANDIDATES) or df.columns[0]
    df["_GENE"] = df[id_col].astype("string").str.strip().str.upper()
    return df.loc[df["_GENE"].notna() & df["_GENE"].ne("")].copy()


def read_gene_pool(path: str) -> list[str]:
    """Every gene in a screen/library file — the candidate pool for a new phenotype."""
    df = _read_table(path)
    return sorted({g for g in df["_GENE"].tolist() if g and g != "NAN"})


def derive_hits(
    path: str,
    genes,
    *,
    score_col: str | None = None,
    method: str = "gaussian",
    top_ratio: float = 0.05,
    threshold: float | None = None,
) -> dict:
    """Label observed genes; absent genes and missing measurements stay unknown."""
    if not np.isfinite(top_ratio) or not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1]")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if method not in {"gaussian", "castle"}:
        raise ValueError(f"Unknown hit derivation method: {method}")
    df = _read_table(path)

    # 1) explicit hit/label column wins
    hit_col = _pick(list(df.columns), HIT_CANDIDATES)
    if hit_col is not None:
        text = df[hit_col].astype("string").str.strip().str.lower()
        numeric = pd.to_numeric(df[hit_col], errors="coerce")
        is_hit = (text.isin(_TRUE) | numeric.eq(1)).fillna(False)
        is_miss = (text.isin(_FALSE) | numeric.eq(0)).fillna(False)
        missing = df[hit_col].isna() | text.eq("").fillna(False)
        invalid = ~(is_hit | is_miss | missing)
        if invalid.any():
            raise ValueError(f"Unrecognised hit labels in {hit_col}: "
                             f"{df.loc[invalid, hit_col].astype(str).unique()[:5].tolist()}")
        hitset = set(df.loc[is_hit, "_GENE"])
        negative = set(df.loc[is_miss, "_GENE"])
        if hitset & negative:
            raise ValueError(f"Conflicting labels for genes: {', '.join(sorted(hitset & negative))}")
        observed = hitset | negative
        used_method = f"label:{hit_col}"
        used_score = None
    else:
        sc = score_col or _pick(list(df.columns), SCORE_CANDIDATES)
        if sc is None:
            raise ValueError(
                f"No score/hit column found. Columns: {list(df.columns)}. "
                f"Pass --score-col explicitly."
            )
        if sc not in df.columns:
            raise ValueError(f"Score column {sc!r} is not in the readout")
        vals = pd.to_numeric(df[sc], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            raise ValueError(f"No finite scores found in {sc!r}")
        observed = set(df.loc[finite, "_GENE"])
        used_score = sc
        if threshold is not None:
            hit_idx = np.where(finite & (vals > threshold))[0]
            used_method = f"threshold>{threshold}"
        elif method == "castle":
            thr = np.nanpercentile(vals[finite], 95)
            hit_idx = np.where(finite & (vals > thr))[0]
            used_method = "castle(>p95)"
        else:  # gaussian: top `top_ratio` by |score|
            valid_idx = np.flatnonzero(finite)
            order = valid_idx[np.argsort(np.abs(vals[finite]))[::-1]]
            count = max(1, int(finite.sum() * top_ratio))
            hit_idx = order[:count]
            used_method = f"gaussian(top{top_ratio:g})"
        hitset = set(df["_GENE"].to_numpy()[hit_idx])

    batch = _norm(genes)
    hits = [g for g in batch if g in hitset]
    misses = [g for g in batch if g in observed and g not in hitset]
    unknown = [g for g in batch if g not in observed]
    return {
        "tool": "ingest",
        "tested": len(batch),
        "hits": hits,
        "misses": misses,
        "n_hits": len(hits),
        "total_hits": len(hitset),
        "unknown_genes": unknown,
        "method": used_method,
        "score_col": used_score,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Derive hits for a tested batch from a screen readout file.")
    p.add_argument("--file", required=True, help="MAGeCK gene_summary.txt / Gene,Score CSV / labelled table")
    p.add_argument("--genes", nargs="+", required=True, help="the tested batch")
    p.add_argument("--score-col", default=None, help="score column (else auto: Score/log-fold-change/pos|lfc/…)")
    p.add_argument("--method", default="gaussian", choices=["gaussian", "castle"])
    p.add_argument("--top-ratio", type=float, default=0.05, help="gaussian hit fraction (default 0.05)")
    p.add_argument("--threshold", type=float, default=None, help="absolute score cutoff (overrides method)")
    p.add_argument("--json", action="store_true", help="emit JSON only")
    args = p.parse_args()

    try:
        res = derive_hits(
            args.file, args.genes,
            score_col=args.score_col, method=args.method,
            top_ratio=args.top_ratio, threshold=args.threshold,
        )
    except ValueError as exc:
        p.error(str(exc))
    if args.json:
        json.dump(res, sys.stdout)
        sys.stdout.write("\n")
        return
    print(f"Screen readout: {args.file}  (method={res['method']}, score_col={res['score_col']})")
    print(f"Genome-wide hits: {res['total_hits']}")
    if res["unknown_genes"]:
        print(f"[warn] missing gene or measurement (not labelled): {', '.join(res['unknown_genes'])}")
    print(f"\nTested {res['tested']} genes → {res['n_hits']} hits:")
    print("  hits:   " + (", ".join(res["hits"]) or "(none)"))
    print("  misses: " + (", ".join(res["misses"]) or "(none)"))


if __name__ == "__main__":
    main()
