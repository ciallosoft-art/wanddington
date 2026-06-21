"""
WaddingtonV6Arm — C-arm v6: two-bucket routing + LLM temperature=0.

Identical to WaddingtonV5 (V16) except LLM uses temperature=0.0 (greedy
decoding) instead of 0.5, eliminating ±0.020 per-dataset variance from
LLM stochasticity observed across V13/V15/V16 for identical routing weights.

With temperature=0:
  - Same prompt → same output every time
  - 3 seeds still run (ML online learning varies by revealed sequence order,
    but LLM picks are identical across seeds)
  - Cross-run comparison becomes reliable: noise from routing/weighting
    changes is no longer confounded with LLM sampling randomness

Routing (same as WaddingtonV5):
  n_genes > 15000 & hit_rate 2-7%  →  (0.80, 0.20)  ML-heavy
  everything else                  →  (0.60, 0.40)  V13 baseline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA_CSV = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
MEMORY_PATH = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

LLM_TEMPERATURE = 0.0   # deterministic output

W_ML_LARGE    = 0.80
W_LLM_LARGE   = 0.20
W_ML_DEFAULT  = 0.60
W_LLM_DEFAULT = 0.40


def _get_dataset_stats(dataset_name: str) -> tuple[int, int]:
    df = pd.read_csv(TRAINING_DATA_CSV)
    df["gene"] = df["gene"].str.strip().str.upper()
    sub = df[df["dataset"] == dataset_name]
    return len(sub), int(sub["label"].sum())


def _route_weights(n_genes: int, n_hits: int) -> tuple[float, float]:
    hit_rate = n_hits / max(n_genes, 1)
    if n_genes > 15000 and 0.02 < hit_rate < 0.07:
        return W_ML_LARGE, W_LLM_LARGE
    return W_ML_DEFAULT, W_LLM_DEFAULT


class WaddingtonV6Arm(BaseArm):
    """V16 two-bucket routing + LLM temperature=0 for stable, noise-free picks."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v6", dataset_name, batch_size)

        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._w_ml, self._w_llm = _route_weights(n_genes, n_hits)

        self._online = OnlineAdaptiveArm(dataset_name, batch_size)

        task = _load_task(dataset_name)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm = LLMReasoningArm(
            dataset_name, batch_size,
            memory_entries=memory,
            temperature=LLM_TEMPERATURE,
        )

    def _on_reset(self) -> None:
        self._online.reset()
        self._llm.reset()

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        ml_scores = self._online.all_scores()
        llm_set = set(self._llm.select(round_idx, revealed))

        combined: dict[str, float] = {
            g: self._w_ml * ml_scores.get(g, 0.0)
               + self._w_llm * (1.0 if g in llm_set else 0.0)
            for g in self._online._genes
            if g not in self._selected
        }
        return sorted(combined, key=combined.__getitem__, reverse=True)[: self.batch_size]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        self._online.update(round_idx, revealed_new)
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
