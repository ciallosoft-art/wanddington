"""
skills.py — Evolving skill library (Phase 1: trigger-conditioned, no evolution yet).

A *skill* is a distilled, reusable "when-then" strategy fragment, in contrast to the
per-experiment episodic entries in experience_memory.json. Where memory says *what happened*
in one experiment, a skill says *what to do* when a condition holds — and is injected into the
LLM prompt only in the rounds where its trigger fires.

Phase 1 provides:
  - a skill schema (plain dicts, mirroring the memory entries),
  - trigger-conditioned retrieval against the current round state,
  - a leakage guard for the leave-one-out benchmark (drop skills sourced from the held-out
    dataset, or whose directive names a held-out hit gene).

Utility tracking and evolution (promote/retire/merge) are Phase 2 — deliberately absent here so
the mechanism can be measured against the flat top-4 memory in isolation.

Skill schema
------------
{
  "id": "sk_001",
  "type": "pathway_prior | selection_heuristic | calibration",
  "trigger": {                      # all present conditions must hold for the skill to fire
    "min_round": 2,                 # 1-based round index
    "max_round": 5,
    "min_n_genes": 15000,
    "max_n_genes": 3000,
    "min_hit_rate": 0.02,           # dataset hit rate (same signal waddington_c._classify uses)
    "max_hit_rate": 0.07,
    "requires_revealed_hits": true  # only fire once at least one hit has been revealed
  },
  "directive": "…natural-language when-then rule (structure over identity)…",
  "evidence_datasets": ["IFNG", "IL2"],   # source experiments — used for the LOO leakage guard
  "marker_genes": ["ZAP70", "LCK", …],    # canonical pathway genes; a pathway_prior fires only
                                          # when the observed hits include one (discriminative
                                          # trigger). Used internally only — never shown in the
                                          # prompt, so it is not a leakage vector. [] for others.
  "verified": true
}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_LIBRARY_PATH = REPO_ROOT / "workspace" / "results" / "sequential" / "skill_library.json"

# HGNC-like token, for the mechanical leakage scan of a directive.
_GENE_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

# Rendering priority when several skills fire (higher = shown first).
_TYPE_PRIORITY = {"pathway_prior": 3, "selection_heuristic": 2, "calibration": 1}


class SkillLibrary:
    """A read-only (Phase 1) collection of verified skills with trigger-conditioned retrieval."""

    def __init__(self, skills: list[dict]) -> None:
        self._skills = skills

    @classmethod
    def load(cls, path: Path = SKILL_LIBRARY_PATH) -> "SkillLibrary":
        if not Path(path).exists():
            return cls([])
        with open(path) as f:
            data = json.load(f)
        return cls(data if isinstance(data, list) else [])

    def __len__(self) -> int:
        return len(self._skills)

    # -- retrieval ---------------------------------------------------------------

    def retrieve(
        self,
        state: dict,
        exclude_dataset: str,
        block_genes: set[str] | None = None,
        k: int = 4,
    ) -> list[dict]:
        """Return the firing skills for the current state, honouring the LOO leakage guard.

        state keys: round (1-based), n_genes, hit_rate, n_revealed_hits.
        A skill is dropped if it was distilled from `exclude_dataset` (fold discipline) or if its
        directive names a gene in `block_genes` (the held-out set's hits).
        """
        block_genes = block_genes or set()
        firing: list[dict] = []
        revealed_hits = {g.upper() for g in state.get("revealed_hits", ())}
        for sk in self._skills:
            if not sk.get("verified", False):
                continue
            if exclude_dataset in sk.get("evidence_datasets", []):
                continue  # leakage guard 1: skill sourced from the held-out dataset
            if self._directive_leaks(sk, block_genes):
                continue  # leakage guard 2: directive names a held-out hit gene
            if not self._fires(sk.get("trigger", {}), state):
                continue
            # Discriminative gate: a skill with marker genes fires only when the observed hits
            # actually include one of them (i.e. the phenotype really touches this pathway).
            markers = {g.upper() for g in sk.get("marker_genes", [])}
            if markers and revealed_hits.isdisjoint(markers):
                continue
            firing.append(sk)

        firing.sort(
            key=lambda s: (
                _TYPE_PRIORITY.get(s.get("type"), 0),
                len(s.get("evidence_datasets", [])),
            ),
            reverse=True,
        )
        return firing[:k]

    @staticmethod
    def _fires(trigger: dict, state: dict) -> bool:
        rnd = state.get("round", 1)
        n_genes = state.get("n_genes", 0)
        hit_rate = state.get("hit_rate", 0.0)
        n_hits = state.get("n_revealed_hits", 0)

        if "min_round" in trigger and rnd < trigger["min_round"]:
            return False
        if "max_round" in trigger and rnd > trigger["max_round"]:
            return False
        if "min_n_genes" in trigger and n_genes < trigger["min_n_genes"]:
            return False
        if "max_n_genes" in trigger and n_genes > trigger["max_n_genes"]:
            return False
        if "min_hit_rate" in trigger and hit_rate < trigger["min_hit_rate"]:
            return False
        if "max_hit_rate" in trigger and hit_rate > trigger["max_hit_rate"]:
            return False
        if trigger.get("requires_revealed_hits") and n_hits <= 0:
            return False
        return True

    @staticmethod
    def _directive_leaks(skill: dict, block_genes: set[str]) -> bool:
        if not block_genes:
            return False
        tokens = set(_GENE_TOKEN.findall(skill.get("directive", "")))
        return bool(tokens & block_genes)

    # -- rendering ---------------------------------------------------------------

    @staticmethod
    def render(skills: list[dict]) -> str:
        if not skills:
            return ""
        lines = [f"\nLEARNED STRATEGY SKILLS ({len(skills)} applicable to this round):"]
        for i, sk in enumerate(skills, 1):
            src = ", ".join(sk.get("evidence_datasets", [])[:4])
            lines.append(
                f"\n[S{i}] ({sk.get('type', '?')}; learned from {src})\n"
                f"    {sk.get('directive', '').strip()}"
            )
        lines.append("\nApply the skills whose conditions match the feedback you see.")
        return "\n".join(lines)


def load_dataset_hits(dataset_name: str) -> set[str]:
    """All hit genes (label==1) for a dataset — used as the leakage block set for that fold."""
    import pandas as pd

    df = pd.read_csv(REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv")
    df["gene"] = df["gene"].str.strip().str.upper()
    sub = df[(df["dataset"] == dataset_name) & (df["label"] == 1)]
    return set(sub["gene"].tolist())
