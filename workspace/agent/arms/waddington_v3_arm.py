"""
WaddingtonV3Arm — C-arm improved: EMA precision-adaptive weights.

Fixes V13's fixed-weight compromise by measuring each signal's actual
hit precision each round and updating weights via EMA:

    ml_ema  = α × ml_prec  + (1-α) × ml_ema
    llm_ema = α × llm_prec + (1-α) × llm_ema
    w_ml    = ml_ema  / (ml_ema + llm_ema)
    w_llm   = 1 - w_ml   (clamped to [W_LLM_MIN, W_LLM_MAX])

ML-dominant datasets (IFNG, IL2): ml_prec >> llm_prec → w_ml rises to ~0.8
LLM-dominant datasets (Steinhart): llm_prec >> ml_prec → w_llm rises to ~0.7
Both-strong datasets (Replogle_essential): balance preserved near 0.4/0.6

Expected: avg hit@R5 > 0.240 (V13=0.232, OA=0.224).
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

MEMORY_PATH = Path(__file__).resolve().parents[3] / "workspace" / "results" / "sequential" / "experience_memory.json"

EMA_ALPHA  = 0.9   # fast adaptation: current round counts 90%
W_ML_INIT  = 0.6   # prior — same as V13 for Round 0 consistency
W_LLM_INIT = 0.4
W_LLM_MIN  = 0.10  # never fully suppress LLM
W_LLM_MAX  = 0.90  # never fully suppress ML


class WaddingtonV3Arm(BaseArm):
    """
    Improved C-arm: same dual-signal ensemble as V13 (WaddingtonV2) but
    weights adapt online by tracking per-round hit precision of each signal.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
        ema_alpha: float = EMA_ALPHA,
        w_ml_init: float = W_ML_INIT,
        w_llm_init: float = W_LLM_INIT,
    ) -> None:
        super().__init__("waddington_v3", dataset_name, batch_size)
        self._ema_alpha = ema_alpha

        # ML component
        self._online = OnlineAdaptiveArm(dataset_name, batch_size)

        # LLM component: free picks + cross-experiment memory
        task = _load_task(dataset_name)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm = LLMReasoningArm(dataset_name, batch_size, memory_entries=memory)

        # Adaptive weight state
        self._ml_ema  = w_ml_init
        self._llm_ema = w_llm_init
        self._w_ml    = w_ml_init
        self._w_llm   = w_llm_init

        # Per-round independent picks (for precision tracking after reveal)
        self._round_ml_picks:  list[str] = []
        self._round_llm_picks: list[str] = []

    def _on_reset(self) -> None:
        self._online.reset()
        self._llm.reset()
        self._ml_ema   = W_ML_INIT
        self._llm_ema  = W_LLM_INIT
        self._w_ml     = W_ML_INIT
        self._w_llm    = W_LLM_INIT
        self._round_ml_picks  = []
        self._round_llm_picks = []

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        ml_scores = self._online.all_scores()

        # ML independent ranking (pure ML, for precision tracking)
        self._round_ml_picks = [
            g for g in sorted(ml_scores, key=ml_scores.__getitem__, reverse=True)
            if g not in self._selected
        ][: self.batch_size]

        # LLM free picks (unconstrained, with cross-experiment memory)
        self._round_llm_picks = self._llm.select(round_idx, revealed)
        llm_set = set(self._round_llm_picks)

        # Weighted combination using current (adaptive) weights
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
        hits = {g for g, is_hit in revealed_new.items() if is_hit}

        # Update ML precision EMA
        if self._round_ml_picks:
            ml_prec = len(hits & set(self._round_ml_picks)) / len(self._round_ml_picks)
            self._ml_ema = self._ema_alpha * ml_prec + (1 - self._ema_alpha) * self._ml_ema

        # Update LLM precision EMA
        if self._round_llm_picks:
            llm_prec = len(hits & set(self._round_llm_picks)) / len(self._round_llm_picks)
            self._llm_ema = self._ema_alpha * llm_prec + (1 - self._ema_alpha) * self._llm_ema

        # Recompute weights from EMA estimates
        total = self._ml_ema + self._llm_ema + 1e-9
        self._w_llm = max(W_LLM_MIN, min(W_LLM_MAX, self._llm_ema / total))
        self._w_ml  = 1.0 - self._w_llm

        # Sub-arm state updates
        self._online.update(round_idx, revealed_new)
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
