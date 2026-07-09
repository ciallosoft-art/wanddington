"""
WaddingtonCNoLLMArm — Ablation: C − LLM.

Identical feature routing to V14 (v1/v2/v3 DepMap), but LLM is removed:
  - weighted route: pure ML ranking (no LLM votes)
  - two_stage route: pure ML top-128 (no LLM reranking)

Isolates the contribution of the LLM component to C-arm performance.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .waddington_c_arm import (
    _get_feature_config,
    _get_dataset_stats,
    _classify,
    MEMORY_PATH,
)


class WaddingtonCNoLLMArm(BaseArm):
    """C − LLM ablation: online adaptive ML only, same DepMap feature routing as V14."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_c_no_llm", dataset_name, batch_size)

        training_csv, extra_feats = _get_feature_config(dataset_name)
        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._route = _classify(n_genes, n_hits)

        self._online = OnlineAdaptiveArm(
            dataset_name, batch_size,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
        )

    def _on_reset(self) -> None:
        self._online.reset()

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        # Pure ML ranking regardless of route
        remaining = [g for g in self._online._ranking if g not in self._selected]
        return remaining[: self.batch_size]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        self._online.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
