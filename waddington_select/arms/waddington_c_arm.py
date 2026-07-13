"""
WaddingtonCArm — C-arm v14: per-dataset optimal feature version.

Adds K562-specific DepMap feature (depmap_K562_norm from ACH-000551)
for datasets where it improves LOO AUC. Three-tier feature routing:

  DEPMAP_EXCLUDED  → v1 (9 features, no DepMap)
    essential: pan-DepMap hurts (K562 curated set), v1 still best (0.658)
    Steinhart: DepMap disrupts online learning for GD2 biology

  K562_EXCLUDED    → v2 (12 features, pan-cancer DepMap only)
    gwps:          K562 feature redundant with pan-cancer (0.751→0.744)
    IL2:           marginal hurt (0.790→0.784)
    Sanchez21_down: marginal hurt/tie

  K562_INCLUDED    → v3 (13 features, pan-cancer + K562 specific)
    Scharenberg22: 0.792→0.825 (+0.033, largest gain)
    IFNG:          0.651→0.660 (+0.009)
    Carnevale22:   0.519→0.525 (+0.006)
    Sanchez21:     0.599→0.600 (+0.001)

Routing unchanged from V22/V24 (ml_heavy only for n>15000, 2%<hr<7%).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import LLMReasoningArm
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task
from ..skills import SkillLibrary, SKILL_LIBRARY_PATH
from ..features import load_training_frame
from ..phenotype import load_registry

REPO_ROOT         = Path(__file__).resolve().parents[2]
TRAINING_DATA_V1  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
TRAINING_DATA_V3  = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data_v3.csv"
MEMORY_PATH       = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

DEPMAP_PAN_FEATS   = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm"]
DEPMAP_K562_FEATS  = DEPMAP_PAN_FEATS + ["depmap_K562_norm"]

# v1 (no DepMap): online learning disrupted by DepMap for these datasets
DEPMAP_EXCLUDED = {"Replogle_K562_essential", "Steinhart"}

# v2 mode (pan-cancer only, no K562): K562 feature hurts or is redundant
K562_EXCLUDED   = {"Replogle_K562_gwps", "IL2", "Sanchez21_down"}

# all others → v3 mode (pan-cancer + K562)

LLM_TEMPERATURE = 0.0
LLM_MODEL       = "claude-haiku-4-5-20251001"
SHORTLIST_SIZE  = 384

W_ML_LARGE    = 0.80
W_LLM_LARGE   = 0.20
W_ML_DEFAULT  = 0.60
W_LLM_DEFAULT = 0.40


def _get_feature_config(dataset_name: str) -> tuple[Path, list[str]]:
    if dataset_name in DEPMAP_EXCLUDED:
        return TRAINING_DATA_V1, []
    elif dataset_name in K562_EXCLUDED:
        return TRAINING_DATA_V3, DEPMAP_PAN_FEATS
    else:
        return TRAINING_DATA_V3, DEPMAP_K562_FEATS


def _get_dataset_stats(dataset_name: str) -> tuple[int, int]:
    csv, _ = _get_feature_config(dataset_name)
    df = load_training_frame(csv, dataset_name)
    df["gene"] = df["gene"].str.strip().str.upper()
    sub = df[df["dataset"] == dataset_name]
    n_genes, n_hits = len(sub), int(sub["label"].sum())
    if n_hits == 0:
        # A newly-registered phenotype has no labels yet. Use the scientist's expected hit rate to
        # route if they gave one; otherwise a hit rate of 0 routes to the safe `baseline` fusion.
        rate = (load_registry().get(dataset_name) or {}).get("expected_hit_rate")
        if rate:
            n_hits = int(n_genes * float(rate))
    return n_genes, n_hits


def _classify(n_genes: int, n_hits: int) -> str:
    hit_rate = n_hits / max(n_genes, 1)
    if n_genes > 15000 and 0.02 < hit_rate < 0.07:
        return "ml_heavy"
    if 3000 < n_genes <= 15000 and hit_rate > 0.08:
        return "two_stage"
    return "baseline"


class WaddingtonCArm(BaseArm):
    """Three-tier DepMap feature routing for per-dataset optimal ML signal."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
        skill_library: "SkillLibrary | None" = None,
        use_memory: bool = True,
        use_enrichment: bool = False,
        name: str = "waddington_c",
    ) -> None:
        super().__init__(name, dataset_name, batch_size)

        training_csv, extra_feats = _get_feature_config(dataset_name)

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
        if use_memory:
            raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
            memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        else:
            memory = []
        self._llm = LLMReasoningArm(
            dataset_name, batch_size,
            memory_entries=memory,
            temperature=LLM_TEMPERATURE,
            model=LLM_MODEL,
            training_csv=training_csv,
            extra_feature_cols=extra_feats,
            skill_library=skill_library,
            dataset_hit_rate=n_hits / max(n_genes, 1),
            use_enrichment=use_enrichment,
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


class WaddingtonCSkillsArm(WaddingtonCArm):
    """C-arm variant that replaces the flat cross-experiment memory with the
    trigger-conditioned skill library (Phase 1).

    For the ablation: waddington_c (flat top-4 memory) vs waddington_c_skills
    (verified skill library, injected only when a skill's trigger fires).
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        skill_path: Path = SKILL_LIBRARY_PATH,
    ) -> None:
        super().__init__(
            dataset_name,
            batch_size,
            skill_library=SkillLibrary.load(skill_path),
            use_memory=False,
            name="waddington_c_skills",
        )


class WaddingtonCMemSkillsArm(WaddingtonCArm):
    """Additive variant: flat cross-experiment memory AND the skill library together.

    Tests whether keeping memory avoids the specificity loss seen when skills *replace* it
    (e.g. the K562-essential regression in the Phase 1 ablation), while retaining the datasets
    where skills helped.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        skill_path: Path = SKILL_LIBRARY_PATH,
    ) -> None:
        super().__init__(
            dataset_name,
            batch_size,
            skill_library=SkillLibrary.load(skill_path),
            use_memory=True,
            name="waddington_c_memskills",
        )


class WaddingtonCEnrichArm(WaddingtonCArm):
    """Enrichment-augmented hybrid: the full C-arm (memory + ML+LLM fusion) PLUS runtime Enrichr
    enrichment of the hits revealed so far, injected into the LLM prompt each round.

    Transplants the tool-using agent's one winning ingredient (runtime enrichment, which beat the
    pipeline on Scharenberg22) into the strong pipeline, without giving up its ML fusion on the
    strong-ML datasets where the free agent lost.
    """

    def __init__(self, dataset_name: str, batch_size: int) -> None:
        super().__init__(
            dataset_name,
            batch_size,
            use_memory=True,
            use_enrichment=True,
            name="waddington_c_enrich",
        )
