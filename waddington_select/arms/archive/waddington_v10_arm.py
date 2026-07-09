"""
WaddingtonV10Arm — C-arm v10: DepMap-enriched ML features.

Identical to WaddingtonV8Arm (V19) in routing, weights, LLM settings,
and shortlist strategy. Single change: LOO LightGBM trained on 12 features
instead of 9, with 3 new DepMap-derived features:

  depmap_frac_ess   — fraction of 1178 cell lines with Chronos ≤ -0.5
  depmap_mean_norm  — normalized mean Chronos score (0=not essential, 1=pan-essential)
  depmap_min_norm   — normalized min Chronos score across all cell lines

LOO AUC improvement (new vs old features):
  Replogle_gwps:   0.652 → 0.751 (+0.099)
  Scharenberg22:   0.712 → 0.792 (+0.080)
  IL2:             0.733 → 0.790 (+0.057)
  Sanchez21:       0.564 → 0.599 (+0.035)
  IFNG:            0.629 → 0.651 (+0.022)
  Replogle_essential: 0.658 → 0.622 (-0.036, pan-essentiality ≠ K562-specific)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import BaseArm
from ..online_adaptive_arm import OnlineAdaptiveArm
from ..llm_reasoning_arm import LLMReasoningArm
from ..waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA_V2  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data_v2.csv"
MEMORY_PATH       = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

DEPMAP_EXTRA_FEATS = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm"]

LLM_TEMPERATURE = 0.0
LLM_MODEL       = "claude-haiku-4-5-20251001"
SHORTLIST_SIZE  = 384

W_ML_LARGE    = 0.80
W_LLM_LARGE   = 0.20
W_ML_DEFAULT  = 0.60
W_LLM_DEFAULT = 0.40


def _get_dataset_stats(dataset_name: str) -> tuple[int, int]:
    df = pd.read_csv(TRAINING_DATA_V2)
    df["gene"] = df["gene"].str.strip().str.upper()
    sub = df[df["dataset"] == dataset_name]
    return len(sub), int(sub["label"].sum())


def _classify(n_genes: int, n_hits: int) -> str:
    hit_rate = n_hits / max(n_genes, 1)
    if n_genes > 15000 and 0.02 < hit_rate < 0.07:
        return "ml_heavy"
    if 3000 < n_genes <= 15000 and hit_rate > 0.08:
        return "two_stage"
    return "baseline"


class WaddingtonV10Arm(BaseArm):
    """Three-bucket routing with DepMap-enriched ML features; LLM unchanged."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v10", dataset_name, batch_size)

        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._route = _classify(n_genes, n_hits)
        if self._route == "ml_heavy":
            self._w_ml, self._w_llm = W_ML_LARGE, W_LLM_LARGE
        else:
            self._w_ml, self._w_llm = W_ML_DEFAULT, W_LLM_DEFAULT

        self._online = OnlineAdaptiveArm(
            dataset_name, batch_size,
            training_csv=TRAINING_DATA_V2,
            extra_feature_cols=DEPMAP_EXTRA_FEATS,
        )

        task = _load_task(dataset_name)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm = LLMReasoningArm(
            dataset_name, batch_size,
            memory_entries=memory,
            temperature=LLM_TEMPERATURE,
            model=LLM_MODEL,
            training_csv=TRAINING_DATA_V2,
            extra_feature_cols=DEPMAP_EXTRA_FEATS,
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
        self._online.update(round_idx, revealed_new)
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
