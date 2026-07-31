"""
OnlineAdaptiveArm — PerTurboAgent ML Inference component.

Implements the self-adapting ML model that PerTurboAgent uses to rerank
genes after each round of experimental feedback, without requiring an LLM.

Algorithm:
  Round 0: identical to StaticRankerArm (LOO LightGBM prior, no data yet)
  Round R>0:
    - Accumulate revealed (gene, label) pairs as in-experiment training data
    - Retrain LightGBM: LOO historical data + in-experiment data (upweighted)
    - Rerank remaining genes with updated model
    - Select next batch from new ranking

The upweighting of in-experiment samples (IN_EXPERIMENT_WEIGHT=10) lets the
model rapidly adapt its prior toward the current experiment's hit pattern,
mimicking PerTurboAgent's "Train perturbation prediction model" action.
"""

from __future__ import annotations

import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseArm
from ..features import load_training_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_CSV = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"

FEATURE_COLS = [
    "g1_ppi_score",
    "hub_score_norm",
    "archs4_coexpr",
    "ppi_score_sum",
    "kegg_overlap",
    "pli_score",
    "string_degree_norm",
    "kegg_pathway_count_norm",
    "reactome_pathway_count_norm",
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

IN_EXPERIMENT_WEIGHT = 10.0  # upweight in-experiment samples vs historical
MIN_REVEALED_TO_RETRAIN = 8  # don't retrain until this many genes are revealed


class OnlineAdaptiveArm(BaseArm):
    """
    Online-adaptive LightGBM ranker: updates its model each round using
    the gene labels revealed by the oracle.

    Round 0 is identical to StaticRankerArm. Subsequent rounds refine the
    ranking by incorporating in-experiment feedback.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        training_csv: Path | None = None,
        extra_feature_cols: list[str] | None = None,
    ) -> None:
        super().__init__("online_adaptive", dataset_name, batch_size)
        csv_path = training_csv if training_csv is not None else TRAINING_DATA_CSV
        df = load_training_frame(csv_path, dataset_name)
        df["gene"] = df["gene"].str.strip().str.upper()
        all_feats = FEATURE_COLS + (extra_feature_cols or [])
        self._available_feats = [c for c in all_feats if c in df.columns]
        # Clean-headline variant: drop the anchor-relative features (proximity to the phenotype's seed
        # genes), which the audit flagged as privileged target information (51% of anchors are own-screen
        # hits). Uses no realized labels; removes the leak from the prior AND the online model.
        if os.environ.get("WADDINGTON_DROP_ANCHOR_FEATS") == "1":
            _ANCHOR_FEATS = {"g1_ppi_score", "archs4_coexpr", "kegg_overlap"}
            self._available_feats = [c for c in self._available_feats if c not in _ANCHOR_FEATS]

        # LOO train split: all datasets except this one
        self._loo_train: pd.DataFrame = df[df["dataset"] != dataset_name].copy()
        # Stricter leave-one-study-out: also drop the target's same-study sibling screens from the
        # cross-experiment prior, so a sister screen from the same paper cannot leak its hit structure.
        if os.environ.get("WADDINGTON_EXCLUDE_STUDY") == "1":
            try:
                from ..router_protocol import siblings
                sibs = siblings(dataset_name)
                if sibs:
                    self._loo_train = self._loo_train[~self._loo_train["dataset"].isin(sibs)].copy()
            except Exception:
                pass
        # Full feature frame for this dataset (no labels exposed)
        self._test_frame: pd.DataFrame = (
            df[df["dataset"] == dataset_name].reset_index(drop=True).copy()
        )
        self._genes: list[str] = self._test_frame["gene"].tolist()

        # State updated each round
        self._ranking: list[str] = []
        self._revealed_records: list[dict] = []  # {gene, label, features...}
        self._model: lgb.LGBMClassifier | None = None
        self._scores: dict[str, float] = {}  # gene → current ML probability

        # Build initial LOO ranking (Round 0 prior)
        self._rebuild_ranking()

    def _on_reset(self) -> None:
        self._ranking = []
        self._revealed_records = []
        self._model = None
        self._scores = {}
        self._rebuild_ranking()

    def _train_model(self, extra_records: list[dict]) -> lgb.LGBMClassifier:
        """Train LightGBM on LOO data + in-experiment revealed samples."""
        train = self._loo_train.copy()

        pos = (train["label"] == 1).sum()
        neg = (train["label"] == 0).sum()
        scale_pos = neg / pos if pos > 0 else 1.0

        weights = [1.0] * len(train)

        if extra_records:
            extra_df = pd.DataFrame(extra_records)
            # Align columns
            for col in self._available_feats:
                if col not in extra_df.columns:
                    extra_df[col] = 0.0
            train = pd.concat([train, extra_df[self._available_feats + ["label"]]],
                               ignore_index=True)
            weights += [IN_EXPERIMENT_WEIGHT] * len(extra_records)

        model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=scale_pos)
        model.fit(
            train[self._available_feats].values,
            train["label"].values,
            sample_weight=weights,
        )
        return model

    def _rebuild_ranking(self) -> None:
        """Retrain model and rerank all test genes."""
        self._model = self._train_model(self._revealed_records)
        scores = self._model.predict_proba(
            self._test_frame[self._available_feats].values
        )[:, 1]
        order = np.argsort(-scores)
        self._ranking = [self._genes[i] for i in order]
        self._scores = {self._genes[i]: float(scores[i]) for i in range(len(self._genes))}

    def all_scores(self) -> dict[str, float]:
        """Return current ML confidence scores for all genes (gene → probability)."""
        return dict(self._scores)

    def shap_for(self, genes: list[str]) -> dict[str, float]:
        """Mean |SHAP| per feature over `genes` — which features drove this batch's ML score.

        Uses LightGBM's native contribution output (`pred_contrib=True`), so no `shap` dependency.
        The last column is the base value (bias) and is dropped.
        """
        if self._model is None:
            return {}
        want = set(genes)
        idx = [i for i, g in enumerate(self._genes) if g in want]
        if not idx:
            return {}
        X = self._test_frame.iloc[idx][self._available_feats].values
        contrib = self._model.booster_.predict(X, pred_contrib=True)
        mean_abs = np.abs(np.asarray(contrib)[:, :-1]).mean(axis=0)
        return {f: float(v) for f, v in zip(self._available_feats, mean_abs)}

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        remaining = [g for g in self._ranking if g not in self._selected]
        return remaining[: self.batch_size]

    def ranked_candidates(self, n: int, exclude: set[str]) -> list[tuple[str, float]]:
        """Return top-n unselected genes with their current ML confidence scores."""
        if self._model is None:
            return []
        scores = self._model.predict_proba(
            self._test_frame[self._available_feats].values
        )[:, 1]
        order = np.argsort(-scores)
        result: list[tuple[str, float]] = []
        for i in order:
            if len(result) >= n:
                break
            g = self._genes[i]
            if g not in exclude:
                result.append((g, float(scores[i])))
        return result

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        super().update(round_idx, revealed_new)

        # Accumulate revealed records with their features
        gene_to_row = {
            row["gene"]: row
            for _, row in self._test_frame.iterrows()
        }
        for gene, is_hit in revealed_new.items():
            if gene in gene_to_row:
                row = gene_to_row[gene]
                record = {col: row[col] for col in self._available_feats}
                record["label"] = int(is_hit)
                self._revealed_records.append(record)

        # Retrain only when enough data has accumulated
        if len(self._revealed_records) >= MIN_REVEALED_TO_RETRAIN:
            self._rebuild_ranking()
