"""
WaddingtonCEnsembleArm — C-arm with TWO LLM views fused together.

The reason-vs-recall ablation showed the real-name LLM (A2/C) and the anonymized-structure LLM (A1)
are *complementary*: names win where they carry biology the features miss (Steinhart), structure wins
where names mislead (Scharenberg22). Neither dominates; the oracle upper bound of picking the better
per screen is +0.023 over C. You cannot route per screen (modality routing was refuted), so this arm
instead runs BOTH LLMs every round and lets the fusion decide.

Fusion (weighted, both routes): each LLM contributes its full weight independently,

    score(g) = w_ML · ml_score(g) + w_LLM · [g named by name-LLM] + w_LLM · [g named by structure-LLM]

so a gene either LLM endorses keeps the single-view weight (preserving Steinhart/Scharenberg where only
one view is right), and a gene BOTH endorse gets 2·w_LLM — the agreement bonus, directly motivated by
the paper's core finding (genes multiple independent signals agree on hit far above chance).

Cost: two LLM calls per round instead of one. Whether the realized gain justifies that is the question
this arm exists to answer.
"""

from __future__ import annotations

from pathlib import Path

from .waddington_c_arm import (
    WaddingtonCArm,
    _get_feature_config,
    _load_memory,
    _rank_memory_by_relevance,
    _load_task,
    MEMORY_PATH,
    LLM_TEMPERATURE,
    LLM_MODEL,
)


class WaddingtonCEnsembleArm(WaddingtonCArm):
    """C-arm + a second, anonymized-structure LLM view, fused with the ML score."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        # super() builds the ML (self._online), the real-name LLM (self._llm), the route and weights.
        super().__init__(dataset_name, batch_size, memory_path=memory_path,
                         name="waddington_c_ensemble")

        # Add the structure LLM alongside the name LLM, same feature routing and memory.
        from .waddington_c_feature_reasoning_arm import _AnonymousFeaturesLLMArm
        training_csv, extra_feats = _get_feature_config(dataset_name)
        task = _load_task(dataset_name)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm_struct = _AnonymousFeaturesLLMArm(
            dataset_name, batch_size,
            memory_entries=memory,
            temperature=LLM_TEMPERATURE,
            model=LLM_MODEL,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
        )

    def _on_reset(self) -> None:
        super()._on_reset()          # resets self._online and self._llm (name)
        self._llm_struct.reset()

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        # Always the weighted ensemble (both routes): each LLM votes at full w_llm; agreement stacks.
        ml_scores = self._online.all_scores()
        name_set = set(self._llm.select(round_idx, revealed))
        struct_set = set(self._llm_struct.select(round_idx, revealed))
        combined: dict[str, float] = {
            g: self._w_ml * ml_scores.get(g, 0.0)
               + self._w_llm * (1.0 if g in name_set else 0.0)
               + self._w_llm * (1.0 if g in struct_set else 0.0)
            for g in self._online._genes
            if g not in self._selected
        }
        return sorted(combined, key=combined.__getitem__, reverse=True)[: self.batch_size]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        super().update(round_idx, revealed_new)   # online + name LLM + selected tracking
        self._llm_struct.update(round_idx, revealed_new)
