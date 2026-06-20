"""
WaddingtonV2Arm — C-arm improved: dual-signal weighted ensemble.

Fixes V12's ML-shortlist bottleneck by running ML scoring and LLM free
picks in parallel, then combining via weighted score:

    score(g) = w_ml × ml_score(g) + w_llm × llm_bonus(g)

where:
  ml_score(g)  = OnlineAdaptiveArm probability for gene g (0–1, all genes)
  llm_bonus(g) = 1.0 if g named by LLM, else 0.0

LLM uses unconstrained free picks (V11 style) WITH cross-experiment memory
(V12 style). No candidate pool restriction — LLM can name any gene it knows.

Expected: avg hit@R5 > 0.235, outperforming all single-method arms.
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm, _load_auth_token
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

MEMORY_PATH = Path(__file__).resolve().parents[3] / "workspace" / "results" / "sequential" / "experience_memory.json"

W_ML  = 0.6   # ML score weight
W_LLM = 0.4   # LLM mention bonus weight


class WaddingtonV2Arm(BaseArm):
    """
    Improved C-arm: ML scores all genes; LLM freely names top picks (with memory);
    combined by weighted score. No candidate pool restriction on LLM.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
        w_ml: float = W_ML,
        w_llm: float = W_LLM,
    ) -> None:
        super().__init__("waddington_v2", dataset_name, batch_size)
        self._w_ml = w_ml
        self._w_llm = w_llm

        # ML component: online-adaptive ranking + all_scores()
        self._online = OnlineAdaptiveArm(dataset_name, batch_size)

        # LLM component: free picks + cross-experiment memory
        task = _load_task(dataset_name)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm = LLMReasoningArm(
            dataset_name, batch_size, memory_entries=memory
        )

    def _on_reset(self) -> None:
        self._online.reset()
        self._llm.reset()

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        # Signal 1: ML scores for all unselected genes
        ml_scores = self._online.all_scores()

        # Signal 2: LLM free picks (unconstrained, with memory)
        llm_picks = set(self._llm.select(round_idx, revealed))

        # Weighted combination
        combined: dict[str, float] = {}
        for g in self._online._genes:
            if g in self._selected:
                continue
            combined[g] = (
                self._w_ml * ml_scores.get(g, 0.0)
                + self._w_llm * (1.0 if g in llm_picks else 0.0)
            )

        ranked = sorted(combined, key=combined.__getitem__, reverse=True)
        return ranked[: self.batch_size]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        self._online.update(round_idx, revealed_new)
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
