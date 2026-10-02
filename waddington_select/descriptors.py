"""
descriptors.py — high-dimensional, label-free gene descriptors (docs/ml_engines_plan.md, phase 2 item 2).

GeneDisco describes each gene by a long vector — its DepMap (Achilles) dependency profile, or a STRING
network embedding — where our engines use ten summary features. Two such descriptors, each reduced to
DIM dimensions and cached (gitignored workspace/cache/descriptors/):

  depmap_pca       the gene's CRISPR gene-effect profile across every DepMap cell line, PCA to DIM
  string_spectral  Laplacian eigenmap of the STRING >= 700 graph's largest connected component

Both depend on gene identity only, never on a screen's labels. `rows(name, genes)` looks them up and
returns NaN for genes a descriptor does not cover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "workspace" / "cache" / "descriptors"
DEPMAP_CSV = REPO / "workspace" / "data" / "depmap" / "CRISPRGeneEffect.csv"
DIM = 32
_LOADED: dict[str, tuple[dict[str, int], np.ndarray]] = {}


def _depmap_pca() -> tuple[list[str], np.ndarray]:
    import pandas as pd
    from sklearn.decomposition import PCA
    d = pd.read_csv(DEPMAP_CSV, index_col=0)  # cell lines x "SYMBOL (entrez)"
    d.columns = [c.split(" (")[0].strip().upper() for c in d.columns]
    d = d.T.groupby(level=0).mean()  # genes x cell lines; merges the odd duplicated symbol
    d = d.apply(lambda r: r.fillna(r.mean()), axis=1).dropna()  # a gene's missing lines -> its own mean
    Z = PCA(n_components=DIM, svd_solver="full").fit_transform(d.to_numpy(dtype=np.float64))
    return list(d.index), Z.astype(np.float32)


def _string_spectral() -> tuple[list[str], np.ndarray]:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh
    from .string_graph import string_graph
    genes, A = string_graph()
    _, comp = connected_components(A, directed=False)
    keep = np.flatnonzero(comp == np.bincount(comp).argmax())
    A = A[keep][:, keep].astype(np.float64)
    dinv = 1.0 / np.sqrt(np.asarray(A.sum(axis=1)).ravel())
    S = sp.diags(dinv) @ A @ sp.diags(dinv)
    # top DIM+1 eigenvectors of the normalised adjacency; the first is the trivial sqrt(degree) one.
    # A fixed start vector makes ARPACK deterministic; sign is irrelevant to tree models.
    vals, vecs = eigsh(S, k=DIM + 1, which="LA", v0=np.full(len(keep), 1.0 / np.sqrt(len(keep))))
    order = np.argsort(-vals)[1:]
    Z = vecs[:, order] * dinv[:, None]  # random-walk eigenvectors (Laplacian eigenmaps)
    return [genes[i] for i in keep], Z.astype(np.float32)


_BUILDERS = {"depmap_pca": _depmap_pca, "string_spectral": _string_spectral}


def descriptor(name: str) -> tuple[dict[str, int], np.ndarray]:
    """(gene -> row, matrix[n_genes, DIM]) for one descriptor, built once and cached on disk."""
    if name in _LOADED:
        return _LOADED[name]
    cache = CACHE_DIR / f"{name}{DIM}.npz"
    if not cache.exists():
        genes, Z = _BUILDERS[name]()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, genes=np.array(genes), Z=Z)
    z = np.load(cache, allow_pickle=False)
    _LOADED[name] = ({str(g): i for i, g in enumerate(z["genes"])}, z["Z"])
    return _LOADED[name]


def rows(name: str, genes) -> np.ndarray:
    index, Z = descriptor(name)
    out = np.full((len(genes), Z.shape[1]), np.nan, dtype=np.float32)
    for j, g in enumerate(genes):
        i = index.get(str(g).upper())
        if i is not None:
            out[j] = Z[i]
    return out
