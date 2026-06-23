"""
WaddingtonV15Arm — C-arm v15: uncertainty-aware dynamic weights.

Key innovation over V14: instead of fixed ML/LLM weights, dynamically adjust
each round based on uncertainty signals from both components:

  ML confidence  = std(LightGBM probability scores)
                   Higher spread → ML has clear opinions → higher ML weight
                   Typical range: LOO prior ~0.05-0.10, adapted ~0.12-0.22

  LLM confidence = LLM's self-reported float in [0, 1] returned alongside genes
                   Prompt asks: {"genes": [...], "confidence": 0.75}
                   Reflects how strongly the LLM believes in its biological picks

Dynamic weight formula (per round):
  w_ml  = clip(ml_conf / (ml_conf + llm_conf), w_min, w_max)
  w_llm = 1 - w_ml

Route-aware clipping prevents extreme values:
  ml_heavy (n>15000, 2%<hr<7%): w_ml ∈ [0.60, 0.95]
  baseline:                      w_ml ∈ [0.35, 0.80]

DepMap feature routing, two_stage handling, and memory are identical to V14.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task

REPO_ROOT         = Path(__file__).resolve().parents[3]
TRAINING_DATA_V1  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
TRAINING_DATA_V3  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data_v3.csv"
MEMORY_PATH       = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

DEPMAP_PAN_FEATS   = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm"]
DEPMAP_K562_FEATS  = DEPMAP_PAN_FEATS + ["depmap_K562_norm"]

DEPMAP_EXCLUDED = {"Replogle_K562_essential", "Steinhart"}
K562_EXCLUDED   = {"Replogle_K562_gwps", "IL2", "Sanchez21_down"}

LLM_TEMPERATURE = 0.0
LLM_MODEL       = "claude-haiku-4-5-20251001"
SHORTLIST_SIZE  = 384

# Calibration: ML score std at which confidence saturates to 1.0
# LOO prior typically 0.05-0.10; well-adapted model 0.15-0.22
ML_CONF_SCALE = 0.15

# Weight bounds per route
ML_HEAVY_W_MIN, ML_HEAVY_W_MAX = 0.60, 0.95
BASELINE_W_MIN, BASELINE_W_MAX = 0.35, 0.80


def _get_feature_config(dataset_name: str) -> tuple[Path, list[str]]:
    if dataset_name in DEPMAP_EXCLUDED:
        return TRAINING_DATA_V1, []
    elif dataset_name in K562_EXCLUDED:
        return TRAINING_DATA_V3, DEPMAP_PAN_FEATS
    else:
        return TRAINING_DATA_V3, DEPMAP_K562_FEATS


def _get_dataset_stats(dataset_name: str) -> tuple[int, int]:
    csv, _ = _get_feature_config(dataset_name)
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


class WaddingtonV15Arm(BaseArm):
    """Uncertainty-aware dynamic ML/LLM weight allocation per round."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v15", dataset_name, batch_size)

        training_csv, extra_feats = _get_feature_config(dataset_name)
        n_genes, n_hits = _get_dataset_stats(dataset_name)
        self._route = _classify(n_genes, n_hits)

        if self._route == "ml_heavy":
            self._w_min, self._w_max = ML_HEAVY_W_MIN, ML_HEAVY_W_MAX
        else:
            self._w_min, self._w_max = BASELINE_W_MIN, BASELINE_W_MAX

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

    def _compute_ml_confidence(self, ml_scores: dict[str, float]) -> float:
        """ML confidence from score spread. std=0 → 0, std≥ML_CONF_SCALE → 1."""
        if not ml_scores:
            return 0.5
        scores = np.array(list(ml_scores.values()), dtype=float)
        return float(min(np.std(scores) / ML_CONF_SCALE, 1.0))

    def _dynamic_weights(
        self, ml_conf: float, llm_conf: float
    ) -> tuple[float, float]:
        total = ml_conf + llm_conf
        if total < 1e-9:
            raw_w_ml = (self._w_min + self._w_max) / 2
        else:
            raw_w_ml = ml_conf / total
        w_ml = float(np.clip(raw_w_ml, self._w_min, self._w_max))
        return w_ml, 1.0 - w_ml

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        if self._route == "two_stage":
            return self._select_two_stage(round_idx, revealed)
        return self._select_weighted_dynamic(round_idx, revealed)

    def _select_weighted_dynamic(
        self, round_idx: int, revealed: dict[str, bool]
    ) -> list[str]:
        ml_scores = self._online.all_scores()
        ml_conf = self._compute_ml_confidence(ml_scores)

        llm_genes, llm_conf = self._llm.select_with_confidence(round_idx, revealed)
        llm_set = set(llm_genes)

        w_ml, w_llm = self._dynamic_weights(ml_conf, llm_conf)
        print(
            f"    [V15] round={round_idx} "
            f"ml_conf={ml_conf:.3f} llm_conf={llm_conf:.3f} "
            f"→ w_ml={w_ml:.3f} w_llm={w_llm:.3f}"
        )

        combined: dict[str, float] = {
            g: w_ml * ml_scores.get(g, 0.0)
               + w_llm * (1.0 if g in llm_set else 0.0)
            for g in self._online._genes
            if g not in self._selected
        }
        return sorted(combined, key=combined.__getitem__, reverse=True)[: self.batch_size]

    def _select_two_stage(
        self, round_idx: int, revealed: dict[str, bool]
    ) -> list[str]:
        candidates = self._online.ranked_candidates(SHORTLIST_SIZE, exclude=self._selected)
        if not candidates:
            return [g for g in self._online._ranking if g not in self._selected][: self.batch_size]
        return self._llm.select_with_shortlist(round_idx, candidates)

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        self._online.update(round_idx, revealed_new)
        self._llm.update(round_idx, revealed_new)
        super().update(round_idx, revealed_new)
