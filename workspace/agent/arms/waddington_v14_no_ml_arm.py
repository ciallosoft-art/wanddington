"""
WaddingtonV14NoMLArm — Ablation: C − ML (online retraining).

Same DepMap feature routing and LLM as V14, but the ML component
never retrains on revealed in-experiment data. Round 0 LOO prior
is used throughout all 5 rounds (static prior only).

Isolates the contribution of online ML adaptation (rounds 1-5 retraining)
to C-arm performance. The remaining signal comes from:
  - Static LOO LightGBM prior (trained on other datasets)
  - LLM biological prior (same as full C arm)
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task
from .waddington_v14_arm import (
    _get_feature_config,
    _get_dataset_stats,
    _classify,
    LLM_TEMPERATURE,
    LLM_MODEL,
    MEMORY_PATH,
    SHORTLIST_SIZE,
    W_ML_LARGE, W_LLM_LARGE,
    W_ML_DEFAULT, W_LLM_DEFAULT,
)


class _FrozenOnlineArm(OnlineAdaptiveArm):
    """OnlineAdaptiveArm that never retrains — LOO prior frozen after Round 0."""

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        # Track selected genes but skip retraining
        BaseArm.update(self, round_idx, revealed_new)


class WaddingtonV14NoMLArm(BaseArm):
    """C − ML ablation: static LOO prior + LLM, no online retraining."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v14_no_ml", dataset_name, batch_size)

        training_csv, extra_feats = _get_feature_config(dataset_name)
        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._route = _classify(n_genes, n_hits)

        if self._route == "ml_heavy":
            self._w_ml, self._w_llm = W_ML_LARGE, W_LLM_LARGE
        else:
            self._w_ml, self._w_llm = W_ML_DEFAULT, W_LLM_DEFAULT

        self._online = _FrozenOnlineArm(
            dataset_name, batch_size,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
        )

        task = _load_task(dataset_name)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm = LLMReasoningArm(
            dataset_name, batch_size,
            memory_entries=memory,
            temperature=LLM_TEMPERATURE,
            model=LLM_MODEL,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
        )

    def _on_reset(self) -> None:
        self._online.reset()
        self._llm.reset()

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        if self._route == "two_stage":
            return self._select_two_stage(round_idx, revealed)
        return self._select_weighted(round_idx, revealed)

    def _select_weighted(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        ml_scores = self._online.all_scores()
        llm_set = set(self._llm.select(round_idx, revealed))
        combined: dict[str, float] = {
            g: self._w_ml * ml_scores.get(g, 0.0)
               + self._w_llm * (1.0 if g in llm_set else 0.0)
            for g in self._online._genes
            if g not in self._selected
        }
        return sorted(combined, key=combined.__getitem__, reverse=True)[: self.batch_size]

    def _select_two_stage(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        candidates = self._online.ranked_candidates(SHORTLIST_SIZE, exclude=self._selected)
        if not candidates:
            return [g for g in self._online._ranking if g not in self._selected][: self.batch_size]
        return self._llm.select_with_shortlist(round_idx, candidates)

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        self._online.update(round_idx, revealed_new)  # frozen: tracks _selected only
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
