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
