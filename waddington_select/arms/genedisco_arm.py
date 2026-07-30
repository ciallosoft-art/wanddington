"""
GeneDiscoArm — GeneDisco's acquisition functions, ported onto our benchmark.

GeneDisco (Mehrjou et al., 2021) is not a model; it is a benchmark of *batch active-learning
acquisition functions*. Each baseline = a surrogate trained on the genes tested so far + an
acquisition rule that picks the next batch from the surrogate's predictions / uncertainty / embedding.

We could not run their package over our full benchmark: it natively implements only two of our screens
(schmidt_2021_ifng / _il2; sanchez_2021_neurons_tau's target is `NotImplementedError` in their
evaluator) and the other four were never GeneDisco tasks. BioDiscoveryAgent had the same problem and
solved it the same way we do here --- by running GeneDisco's *methods* on their own datasets.

So this arm ports the acquisition rules verbatim from
`genedisco/active_learning_methods/acquisition_functions/` and applies them to our feature table and
oracle, which makes them directly comparable to every other arm (identical features, identical hit
definition, identical 5-round budget).

The one thing we deliberately keep faithful is the part that defines GeneDisco: **the surrogate sees
only within-experiment labels.** It is retrained each round on the genes revealed so far in *this*
screen and gets no cross-experiment prior. Round 1 therefore has no training data and falls back to a
random batch, exactly as GeneDisco's loop does. That is the whole point of the comparison: it isolates
"active learning from scratch inside one screen" from "a prior transferred across screens".
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .base import BaseArm
from .llm_reasoning_arm import FEATURE_COLS, LGBM_PARAMS
from .waddington_c_arm import _get_feature_config
from ..features import load_training_frame

ACQUISITIONS = ("topuncertain", "softuncertain", "marginsample", "coreset", "kmeans_data", "random")


class GeneDiscoArm(BaseArm):
    """One GeneDisco acquisition function on our benchmark."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        acquisition: str = "topuncertain",
        seed: int = 42,
    ) -> None:
        super().__init__(f"genedisco_{acquisition}", dataset_name, batch_size)
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition {acquisition}; valid: {ACQUISITIONS}")
        self.acquisition = acquisition
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        training_csv, extra_feats = _get_feature_config(dataset_name)
        df = load_training_frame(training_csv, dataset_name)
        df["gene"] = df["gene"].str.strip().str.upper()
        sub = df[df["dataset"] == dataset_name].reset_index(drop=True)
        self._genes: list[str] = sub["gene"].tolist()
        feats = [c for c in FEATURE_COLS + (extra_feats or []) if c in sub.columns]
        X = sub[feats].to_numpy(dtype=float)
        # standardise: k-center / k-means distances are scale-sensitive
        mu, sd = X.mean(0), X.std(0)
        self._X = (X - mu) / np.where(sd > 0, sd, 1.0)
        self._idx = {g: i for i, g in enumerate(self._genes)}
        self._revealed: dict[str, bool] = {}

    def _on_reset(self) -> None:
        self._revealed = {}
        self._rng = np.random.default_rng(self._seed)

    # ── surrogate: within-experiment only (no cross-experiment prior) ──────────
    def _fit_surrogate(self):
        rows = [(self._idx[g], y) for g, y in self._revealed.items() if g in self._idx]
        if not rows:
            return None
        ii = np.array([r[0] for r in rows])
        yy = np.array([1 if r[1] else 0 for r in rows])
        if yy.min() == yy.max():          # single-class: nothing to learn yet
            return None
        pos, neg = int(yy.sum()), int((1 - yy).sum())
        model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=(neg / max(pos, 1)))
        model.fit(self._X[ii], yy)
        return model

    def _predict(self, model, cand_idx: np.ndarray):
        """Return (mean, uncertainty, margin) as GeneDisco's models do."""
        p = model.predict_proba(self._X[cand_idx])[:, 1]
        return p, p * (1.0 - p), np.abs(p - 0.5)   # Bernoulli variance; distance to boundary

    # ── acquisition rules, ported from the genedisco source ────────────────────
    def _acquire(self, model, cand_idx: np.ndarray) -> np.ndarray:
        k = min(self.batch_size, len(cand_idx))
        if self.acquisition == "random" or model is None:
            return self._rng.choice(cand_idx, size=k, replace=False)

        if self.acquisition in ("topuncertain", "softuncertain", "marginsample"):
            _, unc, margin = self._predict(model, cand_idx)
            if self.acquisition == "topuncertain":       # highest uncertainty
                return cand_idx[np.flip(np.argsort(unc))[:k]]
            if self.acquisition == "marginsample":       # smallest margin
                return cand_idx[np.argsort(margin)[:k]]
            # softuncertain: softmax(temperature) sampling over uncertainty
            e = np.exp(unc / 0.9)
            pr = np.exp(e - e.max()); pr = pr / pr.sum()
            return self._rng.choice(cand_idx, size=k, replace=False, p=pr)

        if self.acquisition == "coreset":                # greedy k-center
            opts = self._X[cand_idx]
            sel_names = [g for g in self._revealed if g in self._idx]
            if sel_names:
                prev = self._X[np.array([self._idx[g] for g in sel_names])]
                min_d = pairwise_distances(opts, prev).min(axis=1)
            else:
                min_d = np.full(len(opts), np.inf)
            chosen = []
            for _ in range(k):
                j = int(min_d.argmax())
                chosen.append(j)
                min_d = np.minimum(min_d, pairwise_distances(opts, opts[[j]])[:, 0])
            return cand_idx[np.array(chosen)]

        # kmeans_data: cluster raw features, take the point closest to each centre
        km = KMeans(init="k-means++", n_init=10, n_clusters=k, random_state=self._seed)
        km.fit(self._X[cand_idx])
        closest = pairwise_distances(self._X[cand_idx], km.cluster_centers_).argmin(axis=0)
        picked = list(dict.fromkeys(closest.tolist()))
        for j in self._rng.permutation(len(cand_idx)):   # top up if centres collided
            if len(picked) >= k:
                break
            if j not in picked:
                picked.append(int(j))
        return cand_idx[np.array(picked[:k])]

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        cand_idx = np.array([i for g, i in self._idx.items() if g not in self._selected])
        model = self._fit_surrogate()
        chosen = self._acquire(model, cand_idx)
        return [self._genes[i] for i in chosen]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        self._revealed.update({g.strip().upper(): bool(v) for g, v in revealed_new.items()})
        super().update(round_idx, revealed_new)
