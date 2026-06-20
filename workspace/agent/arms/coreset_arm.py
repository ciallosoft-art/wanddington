"""
CoresetArm — greedy k-center acquisition (WADDINGTON_PLAN A-arm).

At each round, selects batch_size genes that maximise coverage of the
feature space, given everything already selected in prior rounds.

Algorithm: farthest-point (greedy k-center)
  1. d[g] = min_{s in selected} dist(g, s)   for all unselected g
  2. Repeat batch_size times:
       g* = argmax d[g]
       add g* to batch
       update d[g] = min(d[g], dist(g, g*))

No labels, no biological prior — pure geometric diversity in V7 feature space.
This is the GeneDisco-style A-arm baseline.
"""

from __future__ import annotations

from pathlib import Path

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


class CoresetArm(BaseArm):
    """
    Greedy k-center coreset selection in V7 feature space.

    Self-adapts each round: the selected set grows, so new selections
    cover previously unexplored regions of feature space.
    """

    def __init__(self, dataset_name: str, batch_size: int, seed: int = 42) -> None:
        super().__init__("coreset", dataset_name, batch_size)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._genes, self._X = self._load_features(dataset_name)
        # Indices of genes in selected set (grows across rounds)
        self._selected_idx: set[int] = set()

    def _on_reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._selected_idx = set()

    def _load_features(self, dataset_name: str) -> tuple[list[str], np.ndarray]:
        df = pd.read_csv(TRAINING_DATA_CSV)
        df["gene"] = df["gene"].str.strip().str.upper()
        sub = df[df["dataset"] == dataset_name].reset_index(drop=True)
        if sub.empty:
            raise RuntimeError(f"Dataset '{dataset_name}' not found")
        available = [c for c in FEATURE_COLS if c in sub.columns]
        X = sub[available].fillna(0.0).values.astype(np.float32)
        return sub["gene"].tolist(), X

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        n = len(self._genes)
        unselected_idx = [i for i in range(n) if self._genes[i] not in self._selected]

        if not unselected_idx:
            return []

        X_all = self._X

        if not self._selected_idx:
            # Round 0: start from the gene farthest from the global centroid
            centroid = X_all.mean(axis=0)
            dists_to_centroid = np.linalg.norm(X_all[unselected_idx] - centroid, axis=1)
            seed_local = int(np.argmax(dists_to_centroid))
            seed_global = unselected_idx[seed_local]
            batch_idx = [seed_global]
            self._selected_idx.add(seed_global)
            unselected_idx = [i for i in unselected_idx if i != seed_global]
        else:
            batch_idx = []

        # Compute initial distances from unselected to selected set
        selected_arr = np.array(list(self._selected_idx), dtype=int)
        X_sel = X_all[selected_arr]           # (|S|, D)
        X_pool = X_all[unselected_idx]        # (|P|, D)

        # d[j] = min distance from pool point j to any selected point
        # shape: (|P|, |S|) -> min over S axis
        diffs = X_pool[:, None, :] - X_sel[None, :, :]   # (|P|, |S|, D)
        d = np.linalg.norm(diffs, axis=2).min(axis=1)    # (|P|,)

        remaining = list(range(len(unselected_idx)))  # local indices into X_pool

        while len(batch_idx) < self.batch_size and remaining:
            # Pick farthest point
            local_best = int(np.argmax(d[remaining]))
            best_local = remaining[local_best]
            best_global = unselected_idx[best_local]
            batch_idx.append(best_global)
            self._selected_idx.add(best_global)
            remaining.pop(local_best)

            if not remaining:
                break

            # Update distances
            new_pt = X_all[best_global]
            d_to_new = np.linalg.norm(X_pool[remaining] - new_pt, axis=1)
            d[remaining] = np.minimum(d[remaining], d_to_new)

        return [self._genes[i] for i in batch_idx]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        super().update(round_idx, revealed_new)
        # Keep _selected_idx in sync with _selected (gene name set)
        for i, g in enumerate(self._genes):
            if g in self._selected:
                self._selected_idx.add(i)
