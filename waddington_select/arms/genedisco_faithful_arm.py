"""
GeneDiscoFaithfulArm — GeneDisco run the way GeneDisco runs (docs/ml_engines_plan.md, phase 2 item 3a).

GeneDiscoArm keeps GeneDisco's acquisition rules but swaps in our LightGBM classifier on binary labels.
This arm keeps the rest of GeneDisco as well, ported from keypaper/code/genedisco:
  - its models: a random-forest regressor (100 trees; uncertainty = spread over 100 trees drawn with
    replacement) and the consistent-MC-dropout Bayesian MLP of bnn.py (MSE, batch 64, 100 epochs);
  - its target: the screen's continuous score of the genes revealed so far. That is GeneDisco's
    information setting, and more than our protocol reveals (a binary hit label per gene);
  - its rules as its source implements them (marginsample = largest spread of the sampled predictions,
    softuncertain = sampling by softmax(exp(log(1e-7 + std^2) / 0.9)), BADGE's noise-scaled embedding);
  - its loop: a random first batch, then retrain each round on this screen's revealed genes only;
  - optionally its features: Achilles, the full DepMap dependency profile ("achilles"), or ours ("ours",
    the C-arm's own feature set, standardised as GeneDiscoArm does).

Arm names:
    gdf_random                        model-free
    gdf_<feats>_kmeans_data           model-free: k-means on the raw features
    gdf_<feats>_<model>_<rule>        feats achilles|ours; rf rules RF_RULES; bnn rules BNN_RULES
    gdfb_<feats>_<model>_<rule>       the same with equal information (item 3c): each round the model
                                      regresses the revealed genes' 0/1 hit labels, exactly the feedback
                                      our system gets, and the target screen's scores are never loaded.
                                      While every revealed gene has the same label the batch is random,
                                      as in GeneDiscoArm. The model-free arms need no such variant.

Deviations, stated: genes of our pool without a DepMap profile get zeros (GeneDisco's pool is the genes
it has features for); the model is fitted on every revealed gene (GeneDisco fits on an 80% split); the
adversarialBIM distance is computed for 100 genes drawn at random from the untested pool (GeneDisco takes
the first 100 of a set-ordered list, an arbitrary subset), and a batch larger than 100 is filled from the
same random draw.
"""

from __future__ import annotations

import numpy as np

from .base import BaseArm
from .online_adaptive_arm import FEATURE_COLS
from .waddington_c_arm import _drop_anchor_default, _get_feature_config
from .. import bnn
from ..features import load_training_frame
from ..screen_scores import load_scores
from ..tables import ABSDIFF_SUFFIX, with_pair_spread

FEATURES = ("achilles", "ours")
RF_RULES = ("topuncertain", "softuncertain", "marginsample")
BNN_RULES = RF_RULES + ("badge", "coreset", "kmeans_embedding", "adversarialBIM")
RF_TREES = 100


def arm_names() -> list[str]:
    names = ["gdf_random"]
    for f in FEATURES:
        names.append(f"gdf_{f}_kmeans_data")
        names += [f"gdf_{f}_rf_{r}" for r in RF_RULES] + [f"gdf_{f}_bnn_{r}" for r in BNN_RULES]
    return names


BINARY_PREFIX = "gdfb_"


def binary_arm_names() -> list[str]:
    """The equal-information variants: every arm with a model, fed 0/1 labels instead of scores."""
    return [BINARY_PREFIX + n[len("gdf_"):] for n in arm_names() if parse_name(n)[1] is not None]


def parse_name(name: str) -> tuple[str | None, str | None, str] | None:
    """'gdf_achilles_bnn_badge' -> ('achilles', 'bnn', 'badge'); 'gdf_random' -> (None, None, 'random').
    A 'gdfb_' name parses as its 'gdf_' twin (only model-based arms have one)."""
    if name.startswith(BINARY_PREFIX):
        twin = parse_name("gdf_" + name[len(BINARY_PREFIX):])
        return twin if twin is not None and twin[1] is not None else None
    if name not in arm_names():
        return None
    if name == "gdf_random":
        return None, None, "random"
    feats, rest = name[len("gdf_"):].split("_", 1)
    if rest == "kmeans_data":
        return feats, None, rest
    model, rule = rest.split("_", 1)
    return feats, model, rule


def our_features(dataset_name: str):
    """(pool genes, feature matrix) exactly as the C-arm's ML engine sees them (metadata DepMap policy,
    anchor features dropped). Mirrors OnlineAdaptiveArm's column choice; a test checks they agree."""
    csv, extra = _get_feature_config(dataset_name)
    df = load_training_frame(csv, dataset_name)
    df["gene"] = df["gene"].str.strip().str.upper()
    sub = df[df["dataset"] == dataset_name].reset_index(drop=True)
    feats = with_pair_spread([c for c in FEATURE_COLS + (extra or []) if c in sub.columns], sub)
    if _drop_anchor_default():
        anchor = {"g1_ppi_score", "archs4_coexpr", "kegg_overlap"}
        anchor |= {f + ABSDIFF_SUFFIX for f in anchor}
        feats = [c for c in feats if c not in anchor]
    return sub["gene"].tolist(), feats, sub[feats].to_numpy(dtype=float)


class GeneDiscoFaithfulArm(BaseArm):
    def __init__(self, dataset_name: str, batch_size: int, name: str, seed: int = 42) -> None:
        super().__init__(name, dataset_name, batch_size)
        parsed = parse_name(name)
        if parsed is None:
            raise ValueError(f"unknown faithful-GeneDisco arm {name!r}; "
                             f"known: {arm_names() + binary_arm_names()}")
        self.features, self.model, self.rule = parsed
        self.binary = name.startswith(BINARY_PREFIX)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        genes, _, X = our_features(dataset_name)
        self._genes = genes
        self._idx = {g: i for i, g in enumerate(genes)}
        if self.features == "achilles":
            from ..descriptors import achilles_profile
            index, M = achilles_profile()
            X = np.zeros((len(genes), M.shape[1]), dtype=np.float32)
            have = [(j, index[g]) for j, g in enumerate(genes) if g in index]
            X[[j for j, _ in have]] = M[[i for _, i in have]]
        else:
            mu, sd = X.mean(0), X.std(0)
            X = (X - mu) / np.where(sd > 0, sd, 1.0)
        self._X = np.asarray(X, dtype=np.float32)
        # the target screen's score, read only for genes the oracle has revealed (GeneDisco's setting);
        # the equal-information variant never loads it and learns from the revealed labels instead
        self._score = (None if self.binary else
                       load_scores(dataset_name).reindex(genes).fillna(0.0).to_numpy(dtype=float))
        self._label = np.zeros(len(genes))
        self._revealed: list[int] = []
        self._last_batch: list[int] = []

    def _on_reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._revealed, self._last_batch = [], []
        self._label = np.zeros(len(self._genes))

    # -- models --------------------------------------------------------------------------------------
    def _round_seed(self) -> int:
        return self._seed * 1000 + len(self._revealed)

    def _fit(self):
        ii = np.array(self._revealed)
        y = self._label[ii] if self.binary else self._score[ii]
        if self.model == "rf":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=RF_TREES, random_state=self._round_seed(), n_jobs=-1)
            m.fit(self._X[ii], y)
            return m
        return bnn.fit(self._X[ii], y, loss="mse", epochs=100, batch=64, seed=self._round_seed())

    def _samples(self, m, cand: np.ndarray) -> np.ndarray:
        """(K, n_candidates) sampled predictions."""
        if self.model == "rf":
            trees = self._rng.integers(RF_TREES, size=RF_TREES)  # GeneDisco draws trees with replacement
            return np.vstack([m.estimators_[t].predict(self._X[cand]) for t in trees])
        return bnn.mc_predict(m, self._X[cand], seed=self._round_seed())

    # -- acquisition ---------------------------------------------------------------------------------
    def _acquire(self, cand: np.ndarray) -> np.ndarray:
        k = min(self.batch_size, len(cand))
        if self.rule == "random" or not self._revealed:  # GeneDisco's first batch is random
            return self._rng.choice(cand, size=k, replace=False)
        if self.rule == "kmeans_data":
            return cand[bnn.kmeans_closest(self._X[cand], k, self._round_seed(), self._rng)]
        if self.binary and np.unique(self._label[self._revealed]).size < 2:
            return self._rng.choice(cand, size=k, replace=False)  # one class only: nothing to learn yet
        m = self._fit()
        if self.rule in RF_RULES:
            S = self._samples(m, cand)
            if self.rule == "topuncertain":
                return cand[np.argsort(-S.std(0), kind="stable")[:k]]
            if self.rule == "marginsample":
                return cand[np.argsort(-(S.max(0) - S.min(0)), kind="stable")[:k]]
            from scipy.special import softmax
            # float64 throughout: the MLP's samples are float32, where the 1e-300 floor below underflows
            # back to 0 and screens with count-scale scores (Scharenberg22, K562-GWPS) run out of genes
            p = softmax(np.exp(np.log(1e-7 + S.std(0).astype(np.float64) ** 2) / 0.9))
            p = np.maximum(p, 1e-300)  # underflowed zeros would make sampling without replacement fail
            return self._rng.choice(cand, size=k, replace=False, p=p / p.sum())
        if self.rule == "kmeans_embedding":
            return cand[bnn.kmeans_closest(bnn.embedding(m, self._X[cand]), k, self._round_seed(), self._rng)]
        if self.rule == "badge":
            E = bnn.embedding(m, self._X[cand]) * self._rng.standard_normal((len(cand), 1))
            return cand[bnn.kmeanspp_seeds(E, k, self._rng)]
        if self.rule == "coreset":
            from sklearn.metrics import pairwise_distances
            E = bnn.embedding(m, self._X[cand])
            prev = bnn.embedding(m, self._X[np.array(self._last_batch)])
            d = pairwise_distances(E, prev).min(axis=1)
            chosen = []
            for _ in range(k):
                j = int(d.argmax())
                chosen.append(j)
                d = np.minimum(d, pairwise_distances(E, E[[j]]).ravel())
            return cand[np.array(chosen)]
        # adversarialBIM
        perm = self._rng.permutation(len(cand))
        head = perm[:100]
        dist = np.array([bnn.bim_distance(m, self._X[cand[j]], seed=self._round_seed() + j) for j in head])
        order = list(head[np.argsort(dist, kind="stable")]) + list(perm[100:])
        return cand[np.array(order[:k])]

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        cand = np.array([i for g, i in self._idx.items() if g not in self._selected])
        chosen = self._acquire(cand)
        self._last_batch = [int(i) for i in chosen]
        return [self._genes[i] for i in chosen]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        for g, hit in revealed_new.items():
            i = self._idx.get(g.strip().upper())
            if i is not None:
                self._revealed.append(i)
                self._label[i] = float(bool(hit))
        super().update(round_idx, revealed_new)
