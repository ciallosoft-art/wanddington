"""
WaddingtonV4Arm — C-arm v4: static pre-classification routing.

Replaces V14's failed online EMA with offline routing: at init, read
n_genes and n_hits from training data, assign fixed w_ml/w_llm for the
entire experiment based on dataset characteristics.

Routing rules:
  n_genes < 2000              → (0.30, 0.70)  LLM-heavy small screens
  hit_rate < 0.015            → (0.35, 0.65)  LLM-heavy sparse targets
  n_genes > 15000 & hr 2-7%  → (0.80, 0.20)  ML-heavy large screens
  default                     → (0.60, 0.40)  balanced

Classification of all 9 BDA datasets:
  ML-heavy  (0.80/0.20): IFNG, IL2, Sanchez21, Sanchez21_down, Carnevale22
  LLM-heavy (0.30/0.70): Scharenberg22, Replogle_essential
  LLM-heavy (0.35/0.65): Steinhart
  Balanced  (0.60/0.40): Replogle_gwps

Expected: avg hit@R5 ≥ 0.242 (V13=0.232).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import BaseArm
from ..online_adaptive_arm import OnlineAdaptiveArm
from ..llm_reasoning_arm import LLMReasoningArm
from ..waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA_CSV = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
MEMORY_PATH = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

# Routing weight constants
ROUTE_SMALL   = (0.30, 0.70)  # n_genes < 2000
ROUTE_SPARSE  = (0.35, 0.65)  # hit_rate < 1.5%
ROUTE_LARGE   = (0.80, 0.20)  # n_genes > 15000, hit_rate 2-7%
ROUTE_DEFAULT = (0.60, 0.40)  # balanced


def _get_dataset_stats(dataset_name: str) -> tuple[int, int]:
    """Return (n_genes, n_hits) for routing, read from training data."""
    df = pd.read_csv(TRAINING_DATA_CSV)
    df["gene"] = df["gene"].str.strip().str.upper()
    sub = df[df["dataset"] == dataset_name]
    return len(sub), int(sub["label"].sum())


def _route_weights(n_genes: int, n_hits: int) -> tuple[float, float]:
    """Return (w_ml, w_llm) based on dataset characteristics."""
    hit_rate = n_hits / max(n_genes, 1)
    if n_genes < 2000:
        return ROUTE_SMALL
    if hit_rate < 0.015:
        return ROUTE_SPARSE
    if n_genes > 15000 and 0.02 < hit_rate < 0.07:
        return ROUTE_LARGE
    return ROUTE_DEFAULT


class WaddingtonV4Arm(BaseArm):
    """
    C-arm v4: same dual-signal ensemble as V13 (WaddingtonV2) but weights
    are determined once at init by routing based on (n_genes, hit_rate).
    No online learning — eliminates V14's noise-driven weight instability.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v4", dataset_name, batch_size)

        # Determine static weights from dataset characteristics
        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._w_ml, self._w_llm = _route_weights(n_genes, n_hits)

        # ML component
        self._online = OnlineAdaptiveArm(dataset_name, batch_size)

        # LLM component: free picks + cross-experiment memory
        task = _load_task(dataset_name)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm = LLMReasoningArm(dataset_name, batch_size, memory_entries=memory)

    def _on_reset(self) -> None:
        self._online.reset()
        self._llm.reset()

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        ml_scores = self._online.all_scores()
        llm_set = set(self._llm.select(round_idx, revealed))

        combined: dict[str, float] = {}
        for g in self._online._genes:
            if g in self._selected:
                continue
            combined[g] = (
                self._w_ml * ml_scores.get(g, 0.0)
                + self._w_llm * (1.0 if g in llm_set else 0.0)
            )

        ranked = sorted(combined, key=combined.__getitem__, reverse=True)
        return ranked[: self.batch_size]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        self._online.update(round_idx, revealed_new)
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
