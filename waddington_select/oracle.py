"""
DatasetOracle — strict truth-revelation mechanism.

Only reveals gene labels when explicitly queried via reveal().
No arm should ever read labels directly; all truth access goes through here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# The ground-truth / feature table. WADDINGTON_TRAINING_CSV overrides it — used by the CRISPRa
# validation to point every consumer at an isolated superset CSV, so the frozen benchmark file (and
# its committed results) are never touched.
TRAINING_DATA_CSV = Path(
    os.environ.get("WADDINGTON_TRAINING_CSV",
                   REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv")
)
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

# Reason-vs-recall validation screens (Schmidt CRISPRa arm). Registered only when the isolated
# validation CSV is active, so they never appear as benchmark datasets in normal runs.
if os.environ.get("WADDINGTON_TRAINING_CSV"):
    BATCH_SIZES.update({"IL2_crispra": 128, "IFNG_crispra": 128})

BENCHMARK_DATASETS = list(BATCH_SIZES.keys())

# Phenotypes the scientist registered from their own screen (see phenotype.py). They are rankable
# (suggest / upload campaigns) but have NO ground truth, so the oracle refuses to serve them.
from .phenotype import load_registry as _load_registry  # noqa: E402

BATCH_SIZES.update(
    {name: int(cfg.get("batch_size", 32)) for name, cfg in _load_registry().items()}
)

ALL_DATASETS = list(BATCH_SIZES.keys())


class DatasetOracle:
    """
    Wraps a single CRISPR dataset and controls access to ground-truth labels.

    Labels are stored privately; only reveal() exposes them on demand.
    """

    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in BATCH_SIZES:
            raise ValueError(f"Unknown dataset: {dataset_name}. Valid: {list(BATCH_SIZES)}")
        if dataset_name not in BENCHMARK_DATASETS:
            raise ValueError(
                f"'{dataset_name}' is a registered phenotype with no ground truth, so it has no "
                f"oracle. Use the upload path instead (provide your own screen readout): "
                f"python -m waddington_select.ingest, or the frontend's upload experiment mode."
            )
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
