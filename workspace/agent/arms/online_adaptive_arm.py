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

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .base import BaseArm

REPO_ROOT = Path(__file__).resolve().parents[3]
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

    def __init__(self, dataset_name: str, batch_size: int) -> None:
        super().__init__("online_adaptive", dataset_name, batch_size)
        df = pd.read_csv(TRAINING_DATA_CSV)
        df["gene"] = df["gene"].str.strip().str.upper()
        self._available_feats = [c for c in FEATURE_COLS if c in df.columns]

        # LOO train split: all datasets except this one
        self._loo_train: pd.DataFrame = df[df["dataset"] != dataset_name].copy()
        # Full feature frame for this dataset (no labels exposed)
        self._test_frame: pd.DataFrame = (
            df[df["dataset"] == dataset_name].reset_index(drop=True).copy()
        )
        self._genes: list[str] = self._test_frame["gene"].tolist()

        # State updated each round
        self._ranking: list[str] = []
        self._revealed_records: list[dict] = []  # {gene, label, features...}
        self._model: lgb.LGBMClassifier | None = None

        # Build initial LOO ranking (Round 0 prior)
        self._rebuild_ranking()

    def _on_reset(self) -> None:
        self._ranking = []
        self._revealed_records = []
        self._model = None
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
