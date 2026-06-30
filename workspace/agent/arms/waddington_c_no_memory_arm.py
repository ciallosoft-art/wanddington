"""
WaddingtonCNoMemoryArm — Ablation: C − memory.

Identical to WaddingtonCArm except cross-experiment memory is disabled
(memory_entries=[] passed to LLMReasoningArm). Used to quantify how much
the cross-experiment memory module contributes to C-arm performance.
"""

from __future__ import annotations

from pathlib import Path

from .waddington_c_arm import (
    WaddingtonCArm,
    _get_feature_config,
    _get_dataset_stats,
    _classify,
    LLM_TEMPERATURE,
    LLM_MODEL,
    MEMORY_PATH,
)
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm
from .waddington_arm import _load_task
from .base import BaseArm


class WaddingtonCNoMemoryArm(WaddingtonCArm):
    """C − memory ablation: same as V14 but with no cross-experiment memory."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        # Call BaseArm directly to set arm name, then rebuild without memory
        BaseArm.__init__(self, "waddington_c_no_memory", dataset_name, batch_size)

        training_csv, extra_feats = _get_feature_config(dataset_name)
        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._route = _classify(n_genes, n_hits)

        if self._route == "ml_heavy":
            self._w_ml, self._w_llm = 0.80, 0.20
        else:
            self._w_ml, self._w_llm = 0.60, 0.40

        self._online = OnlineAdaptiveArm(
            dataset_name, batch_size,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
        )

        self._llm = LLMReasoningArm(
            dataset_name, batch_size,
            memory_entries=[],          # ← ablation: no cross-experiment memory
            temperature=LLM_TEMPERATURE,
            model=LLM_MODEL,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
        )
