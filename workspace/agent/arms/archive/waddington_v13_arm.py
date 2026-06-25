"""
WaddingtonV13Arm — C-arm v13: exclude Steinhart from DepMap features.

V23 revealed that ml_heavy routing for Steinhart (reducing LLM to 20%)
hurt performance badly: 0.124→0.087. Root cause is two-fold:
  1. LLM knows GD2 synthesis pathway (B4GALNT1, ST8SIA1) and is helpful
     at 40% weight; reducing to 20% removed this benefit.
  2. DepMap (pan-cancer essentiality) anti-correlates with GD2 synthesis
     genes, disrupting online ML adaptation (V22 R4→R5=+0.007 vs V19=+0.032).

Fix: add Steinhart to DEPMAP_EXCLUDED → use v1 features (no DepMap).
Routing reverts to V22 style (ml_heavy only for n>15000 AND 2%<hr<7%),
keeping Steinhart in baseline route (w_llm=0.40 to preserve LLM benefit).

DEPMAP_EXCLUDED = {"Replogle_K562_essential", "Steinhart"}

Expected: Steinhart ~0.154 (V19 level), avg ≈ 0.256
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import BaseArm
from ..online_adaptive_arm import OnlineAdaptiveArm
from ..llm_reasoning_arm import LLMReasoningArm
from ..waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

REPO_ROOT         = Path(__file__).resolve().parents[3]
TRAINING_DATA_V1  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
TRAINING_DATA_V2  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data_v2.csv"
MEMORY_PATH       = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

DEPMAP_EXTRA_FEATS = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm"]
DEPMAP_EXCLUDED    = {"Replogle_K562_essential", "Steinhart"}

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


class WaddingtonV13Arm(BaseArm):
    """V22 routing + DepMap excluded for both essential and Steinhart."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v13", dataset_name, batch_size)

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
