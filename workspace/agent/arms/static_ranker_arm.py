"""
StaticRankerArm — V7 LightGBM trained LOO, one-shot ranking.

The ranking is computed once at init using the leave-one-out (LOO) model:
train on the other 8 datasets, score all genes in this dataset, then rank.
Subsequent calls to select() simply walk down the pre-sorted list.

This arm never reads feedback (update() is a no-op).
It represents the best static prior we can build without in-experiment data.
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
}


class StaticRankerArm(BaseArm):
    """
    Pre-computes a static gene ranking using V7 LightGBM (LOO).
    Each round simply advances a pointer into the sorted list.
    """

    def __init__(self, dataset_name: str, batch_size: int) -> None:
        super().__init__("static_ranker_v7", dataset_name, batch_size)
        self._ranking: list[str] = self._build_ranking(dataset_name)
        self._pointer: int = 0

    def _on_reset(self) -> None:
        self._pointer = 0

    def _build_ranking(self, dataset_name: str) -> list[str]:
        df = pd.read_csv(TRAINING_DATA_CSV)
        df["gene"] = df["gene"].str.strip().str.upper()

        available_feats = [c for c in FEATURE_COLS if c in df.columns]

        train_df = df[df["dataset"] != dataset_name]
        test_df = df[df["dataset"] == dataset_name].copy()

        if train_df.empty or test_df.empty:
            raise RuntimeError(f"Cannot build LOO split for '{dataset_name}'")

        pos = (train_df["label"] == 1).sum()
        neg = (train_df["label"] == 0).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        model = lgb.LGBMClassifier(
            **LGBM_PARAMS, scale_pos_weight=scale_pos_weight, random_state=42
        )
        model.fit(
            train_df[available_feats].values,
            train_df["label"].values,
        )

        scores = model.predict_proba(test_df[available_feats].values)[:, 1]
        test_df = test_df.copy()
        test_df["score"] = scores
        ranked = test_df.sort_values("score", ascending=False)
        return ranked["gene"].tolist()

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        remaining_ranking = [g for g in self._ranking if g not in self._selected]
        batch = remaining_ranking[: self.batch_size]
        return batch
