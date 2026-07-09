"""
DatasetOracle — strict truth-revelation mechanism.

Only reveals gene labels when explicitly queried via reveal().
No arm should ever read labels directly; all truth access goes through here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DATA_CSV = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
BDA_DIR = REPO_ROOT / "workspace" / "data" / "bda_benchmark"

# Batch sizes matching the BDA benchmark convention
BATCH_SIZES: dict[str, int] = {
    "IFNG": 128,
    "IL2": 128,
    "Sanchez21": 128,
    "Sanchez21_down": 128,
    "Carnevale22": 128,
    "Scharenberg22": 32,
    "Steinhart": 128,
    "Replogle_K562_essential": 32,
    "Replogle_K562_gwps": 128,
}

ALL_DATASETS = list(BATCH_SIZES.keys())


class DatasetOracle:
    """
    Wraps a single CRISPR dataset and controls access to ground-truth labels.

    Labels are stored privately; only reveal() exposes them on demand.
    """

    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in BATCH_SIZES:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid: {list(BATCH_SIZES)}")
        self.dataset_name = dataset_name
        self.batch_size = BATCH_SIZES[dataset_name]
        self._load()

    def _load(self) -> None:
        df = pd.read_csv(TRAINING_DATA_CSV)
        df["gene"] = df["gene"].str.strip().str.upper()
        sub = df[df["dataset"] == self.dataset_name].copy()
        if sub.empty:
            raise RuntimeError(
                f"Dataset '{self.dataset_name}' not found in {TRAINING_DATA_CSV}"
            )
        self._genes: list[str] = sub["gene"].tolist()
        self._labels: dict[str, bool] = {
            row["gene"]: bool(row["label"]) for _, row in sub.iterrows()
        }

    def all_genes(self) -> list[str]:
        """Candidate gene pool (no labels)."""
        return list(self._genes)

    def reveal(self, genes: list[str]) -> dict[str, bool]:
        """
        Reveal ground-truth hit/non-hit for a batch of genes.
        Genes not in this dataset are silently mapped to False.
        """
        return {g: self._labels.get(g.strip().upper(), False) for g in genes}

    @property
    def total_hits(self) -> int:
        return sum(self._labels.values())

    @property
    def n_genes(self) -> int:
        return len(self._genes)

    def __repr__(self) -> str:
        return (
            f"DatasetOracle({self.dataset_name!r}, "
            f"n={self.n_genes}, hits={self.total_hits}, batch={self.batch_size})"
        )
