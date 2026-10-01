"""
string_graph.py — the STRING v12 human PPI network as a gene-symbol adjacency, for the graph ML engines.

Read from the local copies prep_universal_features.py downloaded (workspace/data/_universal_cache/), at
the same high-confidence cut-off string_degree_norm uses (combined_score >= 700), symmetric and binary.
The parsed graph is cached as a small npz under workspace/cache/ (gitignored, rebuilt on demand).
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STRING_LINKS = REPO_ROOT / "workspace" / "data" / "_universal_cache" / "string_links.txt.gz"
STRING_INFO = REPO_ROOT / "workspace" / "data" / "_universal_cache" / "string_info.txt.gz"
CACHE_DIR = REPO_ROOT / "workspace" / "cache"
THRESHOLD = 700

_GRAPH: tuple[list[str], "object"] | None = None


def _ensp_to_symbol() -> dict[str, str]:
    mapping: dict[str, str] = {}
    with gzip.open(STRING_INFO, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        pc = header.index("#string_protein_id") if "#string_protein_id" in header else 0
        sc = header.index("preferred_name") if "preferred_name" in header else 1
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > sc and parts[sc].strip():
                mapping[parts[pc].strip()] = parts[sc].strip().upper()
    return mapping


def string_graph() -> tuple[list[str], "object"]:
    """(genes, A): node names and a symmetric binary scipy CSR adjacency over them."""
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    import scipy.sparse as sp
    cache = CACHE_DIR / f"string_graph_{THRESHOLD}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        genes = [str(g) for g in z["genes"]]
        A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(len(genes), len(genes)))
    else:
        ensp2sym = _ensp_to_symbol()
        idx: dict[str, int] = {}
        rows, cols = [], []
        with gzip.open(STRING_LINKS, "rt") as fh:
            fh.readline()
            for line in fh:
                a, b, s = line.split()
                if int(s) < THRESHOLD:
                    continue
                ga, gb = ensp2sym.get(a), ensp2sym.get(b)
                if not ga or not gb or ga == gb:
                    continue
                rows.append(idx.setdefault(ga, len(idx)))
                cols.append(idx.setdefault(gb, len(idx)))
        n = len(idx)
        A = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
        A = ((A + A.T) > 0).astype(np.float32).tocsr()
        genes = [""] * n
        for g, i in idx.items():
            genes[i] = g
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, genes=np.array(genes), data=A.data, indices=A.indices, indptr=A.indptr)
    _GRAPH = (genes, A)
    return _GRAPH
