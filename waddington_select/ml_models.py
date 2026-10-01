"""
ml_models.py — interchangeable ML engines for the online ranker (OnlineAdaptiveArm).

LightGBM is the engine the paper reports, and it keeps its own code path in OnlineAdaptiveArm so the
reported numbers stay bit-identical. Every other engine here answers "why LightGBM?" under the same
features, the same leave-one-screen-out training rows, the same x10 in-experiment weighting and the same
round protocol: only the model changes.

An engine is fitted on the training rows (other screens' labelled genes, plus the target screen's
revealed genes) and then scores the target screen's genes:

    engine.fit(X, y, sample_weight, pos_scale, genes, groups)
    engine.score(X, genes) -> ndarray in [0, 1], higher = more likely a hit

`sample_weight` is the protocol weight (1 for historical rows, IN_EXPERIMENT_WEIGHT for revealed ones);
`pos_scale` is neg/pos over the historical rows, the same quantity LightGBM receives as
scale_pos_weight, and each engine applies it the way its loss allows. `genes` and `groups` (the screen
of each training row, "__target__" for revealed rows) are passed for engines that need them.

`probabilistic` marks engines whose scores are probabilities, which the C-arm's 0.8/0.2 fusion assumes;
the others return a rank percentile and are compared only as ML-only engines. `stochastic` marks engines
whose result depends on the seed, so replicates are seeded rather than repeated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ENGINES: dict[str, type] = {}


def register(cls):
    ENGINES[cls.name] = cls
    return cls


def make_engine(name: str, seed: int = 42):
    if name not in ENGINES:
        raise ValueError(f"unknown ML engine {name!r}; known: {sorted(ENGINES)}")
    return ENGINES[name](seed=seed)


def engine_names() -> list[str]:
    """Every selectable engine, LightGBM (handled natively by OnlineAdaptiveArm) included."""
    return ["lightgbm"] + sorted(ENGINES)


def is_stochastic(name: str) -> bool:
    return name in ENGINES and ENGINES[name].stochastic


def _rank_percentile(s: np.ndarray) -> np.ndarray:
    """Map arbitrary scores to [0, 1] by rank (ties averaged), for engines without probabilities."""
    from scipy.stats import rankdata
    if len(s) < 2:
        return np.ones_like(s, dtype=float)
    return (rankdata(s) - 1.0) / (len(s) - 1.0)


def _class_weighted(y: np.ndarray, sample_weight, pos_scale: float) -> np.ndarray:
    """Protocol weights with positives up-weighted by neg/pos: what scale_pos_weight does inside LightGBM."""
    w = np.asarray(sample_weight, dtype=float)
    return w * np.where(np.asarray(y) == 1, pos_scale, 1.0)


@register
class LogRegEngine:
    """L2 logistic regression on standardised features: does the prior need a non-linear model at all?"""

    name = "logreg"
    probabilistic = True
    stochastic = False

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler().fit(X)
        self._m = LogisticRegression(C=1.0, max_iter=5000)
        self._m.fit(self._scaler.transform(X), y, sample_weight=_class_weighted(y, sample_weight, pos_scale))
        return self

    def score(self, X, genes=None) -> np.ndarray:
        return self._m.predict_proba(self._scaler.transform(X))[:, 1]


@register
class RandomForestEngine:
    """Bagged trees instead of boosted ones. min_samples_leaf=20 mirrors LightGBM's min_child_samples."""

    name = "rf"
    probabilistic = True
    stochastic = True

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        from sklearn.ensemble import RandomForestClassifier
        self._m = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                         n_jobs=-1, random_state=self.seed)
        self._m.fit(X, y, sample_weight=_class_weighted(y, sample_weight, pos_scale))
        # Parallel predict_proba sums the trees in thread-completion order, so identical forests can
        # differ in the last ulp and reorder exact ties; summing on one thread makes a seed reproducible.
        self._m.n_jobs = 1
        return self

    def score(self, X, genes=None) -> np.ndarray:
        return self._m.predict_proba(X)[:, 1]


@register
class MLPEngine:
    """Two-layer MLP (64-64, ReLU, dropout 0.1), weighted BCE, Adam, 20 epochs, on CPU so a seed fixes it."""

    name = "mlp"
    probabilistic = True
    stochastic = True
    HIDDEN = 64
    EPOCHS = 20
    BATCH = 1024

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        import torch
        from sklearn.preprocessing import StandardScaler
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._scaler = StandardScaler().fit(X)
        Xt = torch.tensor(self._scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        wt = torch.tensor(_class_weighted(y, sample_weight, pos_scale), dtype=torch.float32)
        d = Xt.shape[1]
        self._net = torch.nn.Sequential(
            torch.nn.Linear(d, self.HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(self.HIDDEN, self.HIDDEN), torch.nn.ReLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(self.HIDDEN, 1),
        )
        opt = torch.optim.Adam(self._net.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        n = len(yt)
        self._net.train()
        for _ in range(self.EPOCHS):
            perm = torch.as_tensor(rng.permutation(n))
            for k in range(0, n, self.BATCH):
                idx = perm[k:k + self.BATCH]
                opt.zero_grad()
                loss = (loss_fn(self._net(Xt[idx]).squeeze(1), yt[idx]) * wt[idx]).sum() / wt[idx].sum()
                loss.backward()
                opt.step()
        self._net.eval()
        return self

    def score(self, X, genes=None) -> np.ndarray:
        import torch
        with torch.no_grad():
            z = self._net(torch.tensor(self._scaler.transform(X), dtype=torch.float32)).squeeze(1)
        return torch.sigmoid(z).numpy().astype(float)


@register
class LambdaRankEngine:
    """LambdaMART (LightGBM, objective=lambdarank): the same trees, trained to rank hits above non-hits
    within a screen instead of to classify. LightGBM caps a query at 10,000 rows, so each screen is split
    into fixed random sub-queries of 2,000 rows (seeded, hence deterministic); revealed rows form their own
    query. Ranking is insensitive to class balance, so pos_scale is unused. Scores are rank percentiles."""

    name = "lambdarank"
    probabilistic = False
    stochastic = False
    CHUNK = 2000

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        import lightgbm as lgb
        X = np.asarray(X)
        y = np.asarray(y)
        w = np.asarray(sample_weight, dtype=float)
        groups = np.asarray(groups if groups is not None else ["all"] * len(y), dtype=object)
        rng = np.random.default_rng(42)
        qid = np.empty(len(y), dtype=object)
        for g in dict.fromkeys(groups):  # first-seen order, so the split is reproducible
            idx = rng.permutation(np.flatnonzero(groups == g))
            for k in range(0, len(idx), self.CHUNK):
                qid[idx[k:k + self.CHUNK]] = f"{g}#{k // self.CHUNK}"
        order = np.argsort(qid, kind="stable")
        _, first, counts = np.unique(qid[order], return_index=True, return_counts=True)
        sizes = counts[np.argsort(first)]
        self._m = lgb.LGBMRanker(objective="lambdarank", learning_rate=0.05, n_estimators=300,
                                 num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        self._m.fit(X[order], y[order], group=sizes, sample_weight=w[order])
        return self

    def score(self, X, genes=None) -> np.ndarray:
        return _rank_percentile(self._m.predict(np.asarray(X)))


@register
class RWREngine:
    """Network propagation (random walk with restart, restart 0.5) on STRING >= 700. Seeds are the hit
    rows of the training data, each with its protocol weight: a gene's seed mass is the number of other
    screens it is a hit in, plus 10 per hit revealed in this screen. Features and non-hits are unused —
    guilt by association alone. Scores are rank percentiles; genes absent from STRING rank last."""

    name = "rwr"
    probabilistic = False
    stochastic = False
    needs_graph = True
    RESTART = 0.5
    ITERS = 100

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        import scipy.sparse as sp
        from .string_graph import string_graph
        node_genes, A = string_graph()
        self._index = {g: i for i, g in enumerate(node_genes)}
        p0 = np.zeros(len(node_genes))
        for g, yy, w in zip(genes, np.asarray(y), np.asarray(sample_weight, dtype=float)):
            if yy == 1 and g in self._index:
                p0[self._index[g]] += w
        if p0.sum() == 0:
            self._p = p0
            return self
        p0 /= p0.sum()
        deg = np.asarray(A.sum(axis=0)).ravel()
        W = A @ sp.diags(1.0 / np.maximum(deg, 1.0))  # column-stochastic transition matrix
        p = p0.copy()
        for _ in range(self.ITERS):
            p_new = (1 - self.RESTART) * (W @ p) + self.RESTART * p0
            done = np.abs(p_new - p).sum() < 1e-10
            p = p_new
            if done:
                break
        self._p = p
        return self

    def score(self, X, genes=None) -> np.ndarray:
        raw = np.array([self._p[self._index[g]] if g in self._index else -1.0 for g in genes])
        return _rank_percentile(raw)


_SAGE_CLS = None


def _sage(d: int, h: int):
    """Two-layer GraphSAGE (mean aggregator): [x || mean_N(x)] -> h -> [h || mean_N(h)] -> h -> logit.
    `M` is the row-normalised adjacency, sparse on CPU or dense on GPU; `M @ x` is the neighbour mean."""
    global _SAGE_CLS
    import torch
    if _SAGE_CLS is None:
        class SAGE(torch.nn.Module):
            def __init__(self, d, h):
                super().__init__()
                self.l1 = torch.nn.Linear(2 * d, h)
                self.l2 = torch.nn.Linear(2 * h, h)
                self.out = torch.nn.Linear(h, 1)
                self.drop = torch.nn.Dropout(0.1)

            def forward(self, x, M):
                h = self.drop(torch.relu(self.l1(torch.cat([x, M @ x], 1))))
                h = self.drop(torch.relu(self.l2(torch.cat([h, M @ h], 1))))
                return self.out(h).squeeze(1)
        _SAGE_CLS = SAGE
    return _SAGE_CLS(d, h)


@register
class GraphSAGEEngine:
    """GraphSAGE over STRING >= 700 with the same gene-level features as the tabular engines (they are
    identical across screens for every non-anchor feature). A gene's loss is its labelled rows collapsed
    onto its node: weighted hits k and non-hits n-k (protocol weights; hits x neg/pos), i.e. exactly the
    row-level weighted BCE. Full-batch Adam, 200 epochs. On a GPU the neighbour mean is a dense matmul
    (~1 GB for 16k genes): cuBLAS is reproducible, while sparse CUDA matmul accumulates with atomics and
    drifted by up to 0.18 in a probability between identical-seed fits."""

    name = "gnn"
    probabilistic = True
    stochastic = True
    needs_graph = True
    HIDDEN = 64
    EPOCHS = 200
    LR = 5e-3

    def __init__(self, seed: int = 42, device: str | None = None) -> None:
        import os
        import torch
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before the first cuBLAS call
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _adjacency(self, n_extra: int):
        import scipy.sparse as sp
        import torch
        from .string_graph import string_graph
        node_genes, A = string_graph()
        if n_extra:
            A = sp.block_diag([A, sp.csr_matrix((n_extra, n_extra))]).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        M = (sp.diags(1.0 / np.maximum(deg, 1.0)) @ A).tocoo()
        n = A.shape[0]
        Mt = torch.sparse_coo_tensor(np.vstack([M.row, M.col]).astype(np.int64), M.data.astype(np.float32),
                                     (n, n)).coalesce()
        return Mt.to_dense().to(self.device) if self.device != "cpu" else Mt

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        import torch
        from .string_graph import string_graph
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        w = np.asarray(sample_weight, dtype=float)
        node_genes = string_graph()[0]
        in_graph = set(node_genes)
        extra = [g for g in dict.fromkeys(genes) if g not in in_graph]  # labelled genes STRING lacks
        names = list(node_genes) + extra
        self._index = {g: i for i, g in enumerate(names)}
        n, d = len(names), X.shape[1]
        feats = np.zeros((n, d), dtype=np.float32)
        has = np.zeros(n, dtype=bool)
        k = np.zeros(n)
        tot = np.zeros(n)
        for row, g, yy, ww in zip(X, genes, y, w):
            i = self._index[g]
            if not has[i]:
                feats[i], has[i] = row, True
            tot[i] += ww
            k[i] += ww * yy
        self._mu = feats[has].mean(0)
        self._sd = feats[has].std(0) + 1e-6
        feats[~has] = self._mu  # STRING genes with no labelled row: average features (0 after scaling)
        self._feats, self._has = feats, has
        self._M = self._adjacency(len(extra))
        dev = self.device
        Xn = torch.tensor((feats - self._mu) / self._sd, device=dev)
        kp = torch.tensor(k * pos_scale, dtype=torch.float32, device=dev)
        kn = torch.tensor(tot - k, dtype=torch.float32, device=dev)
        lab = torch.tensor(tot > 0, device=dev)
        denom = (kp + kn)[lab].sum()
        self._net = _sage(d, self.HIDDEN).to(dev)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.LR, weight_decay=1e-4)
        ls = torch.nn.functional.logsigmoid
        self._net.train()
        for _ in range(self.EPOCHS):
            opt.zero_grad()
            z = self._net(Xn, self._M)
            loss = -(kp * ls(z) + kn * ls(-z))[lab].sum() / denom
            loss.backward()
            opt.step()
        self._net.eval()
        with torch.no_grad():
            self._z = self._net(Xn, self._M).cpu().numpy()
        return self

    def score(self, X, genes=None) -> np.ndarray:
        import torch
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(genes))
        refill, isolated = [], []
        for j, g in enumerate(genes):
            i = self._index.get(g)
            if i is None:
                isolated.append(j)   # in no training screen and not in STRING
            elif not self._has[i]:
                refill.append(j)     # in STRING; features first seen now
            else:
                out[j] = self._z[i]
        if refill:
            feats = self._feats.copy()
            for j in refill:
                feats[self._index[genes[j]]] = X[j]
            with torch.no_grad():
                z = self._net(torch.tensor((feats - self._mu) / self._sd, device=self.device),
                              self._M).cpu().numpy()
            for j in refill:
                out[j] = z[self._index[genes[j]]]
        if isolated:                 # no neighbours: the aggregation term is zero
            xi = torch.tensor((X[isolated] - self._mu) / self._sd, device=self.device)
            zero = torch.zeros((len(isolated), len(isolated)), device=self.device)
            with torch.no_grad():
                out[isolated] = self._net(xi, zero).cpu().numpy()
        return 1.0 / (1.0 + np.exp(-out))


# ---------------------------------------------------------------------------------------------------
# GenePT text embeddings (Chen & Zou 2024; Zenodo 10.5281/zenodo.10833191, CC-BY-4.0): ada-002 embeddings
# of NCBI/UniProt gene summaries. Reduced once, label-free, to 64 principal components and cached under
# the gitignored workspace/cache/genept/. Genes GenePT lacks get the zero vector (the PCA mean).
# ---------------------------------------------------------------------------------------------------

GENEPT_DIR = Path(__file__).resolve().parents[1] / "workspace" / "cache" / "genept"
GENEPT_DIM = 64
_GENEPT = None


def genept_embedding() -> tuple[dict[str, int], np.ndarray]:
    """(gene -> row, matrix[n_genes, GENEPT_DIM]) from the cached PCA of the GenePT ada embeddings."""
    global _GENEPT
    if _GENEPT is not None:
        return _GENEPT
    cache = GENEPT_DIR / f"genept_pca{GENEPT_DIM}.npz"
    if not cache.exists():
        import pickle
        from sklearn.decomposition import PCA
        pkl = next(GENEPT_DIR.rglob("GenePT_gene_embedding_ada_text.pickle"), None)
        if pkl is None:
            raise FileNotFoundError(f"GenePT embeddings not found under {GENEPT_DIR}; see docs/ml_engines_plan.md")
        with open(pkl, "rb") as fh:
            emb = pickle.load(fh)
        # upper-casing merges a few hundred case variants of one symbol; keep one vector per symbol
        raw = {str(g).strip().upper(): v for g, v in emb.items()}
        genes = sorted(raw)
        M = np.vstack([np.asarray(raw[g], dtype=np.float32) for g in genes])
        Z = PCA(n_components=GENEPT_DIM, random_state=0).fit_transform(M).astype(np.float32)
        np.savez_compressed(cache, genes=np.array(genes), Z=Z)
    z = np.load(cache, allow_pickle=False)
    _GENEPT = ({str(g): i for i, g in enumerate(z["genes"])}, z["Z"])
    return _GENEPT


def _genept_rows(genes) -> np.ndarray:
    index, Z = genept_embedding()
    out = np.zeros((len(genes), Z.shape[1]), dtype=np.float32)
    for j, g in enumerate(genes):
        i = index.get(g)
        if i is not None:
            out[j] = Z[i]
    return out


@register
class GenePTLogRegEngine:
    """GenePert-style: a linear model on the gene's text embedding alone (no tabular features)."""

    name = "genept_lr"
    probabilistic = True
    stochastic = False

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        E = _genept_rows(genes)
        self._scaler = StandardScaler().fit(E)
        self._m = LogisticRegression(C=1.0, max_iter=5000)
        self._m.fit(self._scaler.transform(E), y, sample_weight=_class_weighted(y, sample_weight, pos_scale))
        return self

    def score(self, X, genes=None) -> np.ndarray:
        return self._m.predict_proba(self._scaler.transform(_genept_rows(genes)))[:, 1]


@register
class GenePTLightGBMEngine:
    """The reported LightGBM, same parameters, on the tabular features plus the 64 GenePT components."""

    name = "genept_lgbm"
    probabilistic = True
    stochastic = False

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
        import lightgbm as lgb
        from .arms.online_adaptive_arm import LGBM_PARAMS
        self._m = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=pos_scale)
        self._m.fit(np.hstack([np.asarray(X, dtype=np.float32), _genept_rows(genes)]), y,
                    sample_weight=np.asarray(sample_weight, dtype=float))
        return self

    def score(self, X, genes=None) -> np.ndarray:
        return self._m.predict_proba(np.hstack([np.asarray(X, dtype=np.float32), _genept_rows(genes)]))[:, 1]
