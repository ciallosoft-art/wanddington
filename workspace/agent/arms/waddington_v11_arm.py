"""
WaddingtonV11Arm — C-arm v11: selective DepMap features per dataset.

Identical to WaddingtonV10Arm except that DepMap features are only used
for datasets where they improve LOO AUC. Replogle_K562_essential is
excluded from DepMap features (LOO AUC drops 0.658→0.622 with DepMap,
causing -0.089 hit@R5 regression in V21).

Dataset → training data:
  Replogle_K562_essential → lgbm_training_data.csv    (9 features, no DepMap)
  all other 8 datasets    → lgbm_training_data_v2.csv (12 features, +DepMap)

LOO AUC with selective DepMap:
  gwps:      0.652→0.751 (+0.099)
  IL2:       0.733→0.790 (+0.057)
  IFNG:      0.629→0.651 (+0.022)
  essential: 0.658 (unchanged, back to v1 features)

Expected: avg ≈ 0.255 = V21 gwps gain + essential regression fixed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

REPO_ROOT         = Path(__file__).resolve().parents[3]
TRAINING_DATA_V1  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
TRAINING_DATA_V2  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data_v2.csv"
MEMORY_PATH       = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

DEPMAP_EXTRA_FEATS = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm"]

# Datasets where DepMap features hurt (LOO AUC regression): use v1 features
DEPMAP_EXCLUDED = {"Replogle_K562_essential"}

LLM_TEMPERATURE = 0.0
LLM_MODEL       = "claude-haiku-4-5-20251001"
SHORTLIST_SIZE  = 384

W_ML_LARGE    = 0.80
W_LLM_LARGE   = 0.20
W_ML_DEFAULT  = 0.60
W_LLM_DEFAULT = 0.40


def _get_dataset_stats(dataset_name: str) -> tuple[int, int]:
    csv = TRAINING_DATA_V1 if dataset_name in DEPMAP_EXCLUDED else TRAINING_DATA_V2
    df = pd.read_csv(csv)
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


class WaddingtonV11Arm(BaseArm):
    """Three-bucket routing with selective DepMap features (excluded for essential)."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v11", dataset_name, batch_size)

        use_depmap = dataset_name not in DEPMAP_EXCLUDED
        training_csv = TRAINING_DATA_V2 if use_depmap else TRAINING_DATA_V1
        extra_feats = DEPMAP_EXTRA_FEATS if use_depmap else []

        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._route = _classify(n_genes, n_hits)
        if self._route == "ml_heavy":
            self._w_ml, self._w_llm = W_ML_LARGE, W_LLM_LARGE
        else:
            self._w_ml, self._w_llm = W_ML_DEFAULT, W_LLM_DEFAULT

        self._online = OnlineAdaptiveArm(
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
        self._online.update(round_idx, revealed_new)
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
