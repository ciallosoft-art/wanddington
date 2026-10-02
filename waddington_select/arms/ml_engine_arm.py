"""
MLEngineArm — the C-arm's ML engine on its own (= C − LLM), with a swappable model.

Built exactly as WaddingtonCArm builds its online ranker: the same metadata-only DepMap feature policy,
the same anchor-feature drop, the same leave-one-screen-out rows and x10 in-experiment weighting. Only the
model differs (`waddington_select.ml_models`). Two variants per engine:

    ml_engine_<model>   online: retrained on each round's revealed genes (the C-arm's ML)
    ml_static_<model>   static: the round-0 cross-experiment prior, walked down without retraining

`ml_engine_lightgbm` is the reported engine and the reference every other engine is compared against.
"""

from __future__ import annotations

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .waddington_c_arm import _drop_anchor_default, _get_feature_config


class MLEngineArm(OnlineAdaptiveArm):
    def __init__(self, dataset_name: str, batch_size: int, model: str = "lightgbm",
                 online: bool = True, seed: int = 42) -> None:
        training_csv, extra_feats = _get_feature_config(dataset_name)
        super().__init__(
            dataset_name, batch_size,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
            drop_anchor_feats=_drop_anchor_default(),
            model=model,
            seed=seed,
        )
        self._online_updates = online
        self.name = f"ml_{'engine' if online else 'static'}_{model}"

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        if self._online_updates:
            super().update(round_idx, revealed_new)
        else:
            BaseArm.update(self, round_idx, revealed_new)  # remember what was tested; never retrain


def parse_arm_name(name: str) -> tuple[str, bool] | None:
    """'ml_engine_rf' -> ('rf', True); 'ml_static_rf' -> ('rf', False); anything else -> None."""
    for prefix, online in (("ml_engine_", True), ("ml_static_", False)):
        if name.startswith(prefix):
            return name[len(prefix):], online
    return None


# ---------------------------------------------------------------------------------------------------
# Acquisition rules (step 6 of docs/ml_engines_plan.md). The engine is held fixed at LightGBM — no
# alternative engine beat it with a CI excluding 0 in steps 2-4 — and replaced by an ensemble of
# ENSEMBLE_SIZE bootstrap members (rows resampled with replacement, protocol weights kept), so every rule
# below ranks from the same ensemble and only the rule differs:
#   greedy    mean probability (= Probability-of-Hit for binary labels; the reported rule, ensembled)
#   ucb05/1   mean + beta * sd, beta fixed in advance at 0.5 and 1 (never tuned on a target screen)
#   thompson  batch Thompson: each slot draws a member and takes its best gene not yet chosen
#   margin    GeneDisco-style uncertainty: closest mean probability to 0.5 first (a contrast, not a bet)
# ---------------------------------------------------------------------------------------------------

ACQUISITIONS = ("greedy", "ucb05", "ucb1", "thompson", "margin")
ENSEMBLE_SIZE = 5


class AcquisitionArm(MLEngineArm):
    def __init__(self, dataset_name: str, batch_size: int, acquisition: str = "greedy", seed: int = 0) -> None:
        if acquisition not in ACQUISITIONS:
            raise ValueError(f"unknown acquisition {acquisition!r}; known: {ACQUISITIONS}")
        self._acquisition = acquisition
        self._acq_seed = seed
        super().__init__(dataset_name, batch_size, model="lightgbm", online=True, seed=seed)
        self.name = f"ml_acq_{acquisition}"

    def _rebuild_ranking(self) -> None:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
        from .online_adaptive_arm import IN_EXPERIMENT_WEIGHT, LGBM_PARAMS
        train = self._loo_train
        pos = (train["label"] == 1).sum()
        scale_pos = (train["label"] == 0).sum() / pos if pos > 0 else 1.0
        X = train[self._available_feats].values
        y = train["label"].values
        w = np.ones(len(train))
        if self._revealed_records:
            extra = pd.DataFrame(self._revealed_records)
            X = np.vstack([X, extra[self._available_feats].values])
            y = np.concatenate([y, extra["label"].values])
            w = np.concatenate([w, np.full(len(extra), IN_EXPERIMENT_WEIGHT)])
        # members differ by their bootstrap draw; the round index keeps successive rounds independent
        rng = np.random.default_rng([self._acq_seed, len(self._revealed_records)])
        Xt = self._test_frame[self._available_feats].values
        P = []
        for _ in range(ENSEMBLE_SIZE):
            idx = rng.integers(0, len(y), size=len(y))
            m = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos)
            m.fit(X[idx], y[idx], sample_weight=w[idx])
            P.append(m.predict_proba(Xt)[:, 1])
        P = np.vstack(P)
        mean, sd = P.mean(0), P.std(0)
        a = self._acquisition
        if a == "thompson":  # built over untested genes only, so tested ones never consume a draw
            order = self._thompson_order(P, rng, done=np.array([g in self._selected for g in self._genes]))
        else:
            key = {"greedy": mean, "ucb05": mean + 0.5 * sd, "ucb1": mean + 1.0 * sd,
                   "margin": -np.abs(mean - 0.5)}[a]
            order = np.argsort(-key, kind="stable")
        self._model = None  # an ensemble; SHAP and the single-model hooks do not apply
        self._ranking = [self._genes[i] for i in order]
        self._scores = {self._genes[i]: float(mean[i]) for i in range(len(self._genes))}

    @staticmethod
    def _thompson_order(P, rng, done):
        """An ordering of the untested genes built slot by slot: draw a member, append its best gene not
        yet placed. Tested genes (`done`) are pre-placed so they never consume a draw."""
        import numpy as np
        ranks = [np.argsort(-p, kind="stable") for p in P]
        ptr = [0] * len(P)
        placed = done.copy()
        order = []
        while len(order) < int((~done).sum()):
            k = int(rng.integers(len(P)))
            while placed[ranks[k][ptr[k]]]:
                ptr[k] += 1
            g = ranks[k][ptr[k]]
            placed[g] = True
            order.append(g)
        return order


def parse_acq_name(name: str) -> str | None:
    """'ml_acq_ucb1' -> 'ucb1'; anything else -> None."""
    return name[len("ml_acq_"):] if name.startswith("ml_acq_") and name[len("ml_acq_"):] in ACQUISITIONS else None


# ---------------------------------------------------------------------------------------------------
# Acquisition rules on GeneDisco's Bayesian MLP in our protocol (phase 2, item 3b). The engine is the `bnn`
# engine with the same seed, so `ml_engine_bnn` (greedy on the MC mean) is the paired reference:
#   bald              mutual information H[mean p] - mean_k H[p_k] over the 100 MC samples
#   ucb1              mean + 1 * sd over the MC samples
#   badge             BADGE (Ash et al.): k-means++ seeding on gradient embeddings (p - y_hat) * h, where
#                     y_hat = 1[p >= 0.5] and h the hidden layer (GeneDisco's regression port scales h by
#                     noise instead; a classifier has the real gradient)
#   kmeans_embedding  k-means on the hidden layer, the gene nearest each centre
# The set rules pick this round's batch among untested genes; the other genes follow by MC mean.
# ---------------------------------------------------------------------------------------------------

BNN_ACQUISITIONS = ("bald", "ucb1", "badge", "kmeans_embedding")


class BNNAcquisitionArm(MLEngineArm):
    def __init__(self, dataset_name: str, batch_size: int, acquisition: str = "bald", seed: int = 0) -> None:
        if acquisition not in BNN_ACQUISITIONS:
            raise ValueError(f"unknown acquisition {acquisition!r}; known: {BNN_ACQUISITIONS}")
        self._acquisition = acquisition
        self._acq_seed = seed
        super().__init__(dataset_name, batch_size, model="bnn", online=True, seed=seed)
        self.name = f"ml_bnnacq_{acquisition}"

    def _rebuild_ranking(self) -> None:
        import numpy as np
        from .. import bnn
        self._model = self._train_model(self._revealed_records)
        Xt = self._test_frame[self._available_feats].values
        P = self._model.mc(Xt)
        mean = P.mean(0)
        a = self._acquisition
        if a in ("bald", "ucb1"):
            if a == "ucb1":
                key = mean + P.std(0)
            else:
                def H(p):
                    p = np.clip(p, 1e-12, 1 - 1e-12)
                    return -(p * np.log(p) + (1 - p) * np.log(1 - p))
                key = H(mean) - H(P).mean(0)
            order = list(np.argsort(-key, kind="stable"))
        else:
            rng = np.random.default_rng([self._acq_seed, len(self._revealed_records)])
            untested = np.array([i for i, g in enumerate(self._genes) if g not in self._selected])
            k = min(self.batch_size, len(untested))
            if a == "kmeans_embedding":
                picked = bnn.kmeans_closest(self._model.embedding(Xt[untested]), k,
                                            self._acq_seed * 1000 + len(self._revealed_records), rng)
            else:
                h = np.maximum(self._model.embedding(Xt[untested]), 0.0)
                p = mean[untested]
                picked = bnn.kmeanspp_seeds((p - (p >= 0.5))[:, None] * h, k, rng)
            first = [int(untested[j]) for j in picked]
            placed = set(first)
            rest = [i for i in np.argsort(-mean, kind="stable") if i not in placed]
            order = first + rest
        self._ranking = [self._genes[i] for i in order]
        self._scores = {self._genes[i]: float(mean[i]) for i in range(len(self._genes))}


def parse_bnn_acq_name(name: str) -> str | None:
    """'ml_bnnacq_bald' -> 'bald'; anything else -> None."""
    rule = name[len("ml_bnnacq_"):] if name.startswith("ml_bnnacq_") else None
    return rule if rule in BNN_ACQUISITIONS else None


# ---------------------------------------------------------------------------------------------------
# UCB on the sparse GP (phase 2, item 4b): rank by the latent function's mean + 1 * sd instead of by
# P(hit). The engine is `svgp` with the same seed, so `ml_engine_svgp` (Probability-of-Hit) is the paired
# reference.
# ---------------------------------------------------------------------------------------------------


class SVGPUCBArm(MLEngineArm):
    def __init__(self, dataset_name: str, batch_size: int, seed: int = 0) -> None:
        super().__init__(dataset_name, batch_size, model="svgp", online=True, seed=seed)
        self.name = "ml_svgpacq_ucb1"

    def _rebuild_ranking(self) -> None:
        import numpy as np
        self._model = self._train_model(self._revealed_records)
        Xt = self._test_frame[self._available_feats].values
        mean, sd = self._model.latent(Xt)
        order = np.argsort(-(mean + sd), kind="stable")
        p = self._model.score(Xt)
        self._ranking = [self._genes[i] for i in order]
        self._scores = {self._genes[i]: float(p[i]) for i in range(len(self._genes))}
