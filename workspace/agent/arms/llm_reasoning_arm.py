"""
LLMReasoningArm — B-arm (BDA-style LLM gene selection).

Uses Claude's parametric biological knowledge to select genes each round.
No ML features, no cross-experiment memory — pure LLM reasoning.

Each round:
  1. Builds a prompt: task description + revealed hits/non-hits so far
  2. Calls Claude API → receives a list of gene names
  3. Matches names to actual gene pool (case-insensitive)
  4. Fills unmatched slots with top StaticRanker genes (fallback)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import anthropic
import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseArm

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA_CSV = REPO_ROOT / "workspace" / "evaluation" / "lgbm_training_data.csv"
TASK_PROMPTS_DIR = REPO_ROOT / "workspace" / "data" / "bda_benchmark" / "task_prompts"
AUTH_JSON = Path.home() / ".feynman" / "agent" / "auth.json"

FEATURE_COLS = [
    "g1_ppi_score", "hub_score_norm", "archs4_coexpr", "ppi_score_sum",
    "kegg_overlap", "pli_score", "string_degree_norm",
    "kegg_pathway_count_norm", "reactome_pathway_count_norm",
]

LGBM_PARAMS = {
    "objective": "binary", "metric": "auc", "learning_rate": 0.05,
    "n_estimators": 300, "num_leaves": 31, "verbose": -1, "n_jobs": -1, "random_state": 42,
}

# Dataset → task prompt file (None = hardcoded description)
TASK_PROMPT_FILES: dict[str, Optional[str]] = {
    "IFNG": "IFNG.json",
    "IL2": "IL2.json",
    "Sanchez21": "Sanchez21.json",
    "Sanchez21_down": "Sanchez21_down.json",
    "Carnevale22": "Carnevale22_Adenosine.json",
    "Scharenberg22": "Scharenberg22.json",
    "Steinhart": "Steinhart_crispra_GD2_D22.json",
    "Replogle_K562_essential": None,
    "Replogle_K562_gwps": None,
}

HARDCODED_TASKS: dict[str, dict[str, str]] = {
    "Replogle_K562_essential": {
        "Task": "Identify essential genes required for the survival and proliferation of K562 chronic myelogenous leukemia (CML) cells",
        "Measurement": "the fitness effect (depletion) of gene knockout as measured by sgRNA read count changes in a genome-wide CRISPR screen",
    },
    "Replogle_K562_gwps": {
        "Task": "Identify genes whose knockout significantly affects the fitness (growth or survival) of K562 chronic myelogenous leukemia (CML) cells",
        "Measurement": "log fold change in sgRNA abundance comparing post-selection to pre-selection timepoints in a genome-wide perturbation screen",
    },
}

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 1200


def _load_auth_token() -> str:
    with open(AUTH_JSON) as f:
        auth = json.load(f)
    return auth["anthropic"]["access"]


def _load_task(dataset_name: str) -> dict[str, str]:
    prompt_file = TASK_PROMPT_FILES.get(dataset_name)
    if prompt_file:
        path = TASK_PROMPTS_DIR / prompt_file
        with open(path) as f:
            return json.load(f)
    return HARDCODED_TASKS.get(dataset_name, {"Task": dataset_name, "Measurement": "gene fitness"})


class LLMReasoningArm(BaseArm):
    """
    B-arm: LLM-based gene selection using Claude's parametric knowledge.

    Each round, Claude receives the task description and revealed feedback,
    then names genes it believes will be hits. Names are matched against the
    actual gene pool; unmatched slots fall back to StaticRanker order.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        seed: int = 42,
        memory_entries: list[dict] | None = None,
    ) -> None:
        super().__init__("llm_reasoning", dataset_name, batch_size)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._memory: list[dict] = memory_entries or []

        df = pd.read_csv(TRAINING_DATA_CSV)
        df["gene"] = df["gene"].str.strip().str.upper()
        self._all_feats = [c for c in FEATURE_COLS if c in df.columns]

        # Gene pool for this dataset
        test_df = df[df["dataset"] == dataset_name].reset_index(drop=True)
        self._genes: list[str] = test_df["gene"].tolist()
        self._gene_set: set[str] = set(self._genes)

        # Build StaticRanker fallback order (LOO LightGBM)
        train_df = df[df["dataset"] != dataset_name].copy()
        pos = (train_df["label"] == 1).sum()
        neg = (train_df["label"] == 0).sum()
        spw = neg / pos if pos > 0 else 1.0
        model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=spw)
        model.fit(train_df[self._all_feats].values, train_df["label"].values)
        scores = model.predict_proba(test_df[self._all_feats].values)[:, 1]
        order = np.argsort(-scores)
        self._static_ranking: list[str] = [self._genes[i] for i in order]

        # Task description
        self._task = _load_task(dataset_name)

        # Revealed history: cumulative hits and non-hits by round
        self._round_hits: list[list[str]] = []    # hits per round
        self._round_nonhits: list[list[str]] = [] # non-hits per round

        # Anthropic client
        token = _load_auth_token()
        self._client = anthropic.Anthropic(auth_token=token)

    def _on_reset(self) -> None:
        self._round_hits = []
        self._round_nonhits = []
        self._rng = np.random.default_rng(self._seed)

    def _build_memory_section(self) -> str:
        if not self._memory:
            return ""
        lines = [f"\nCROSS-EXPERIMENT MEMORY ({len(self._memory)} past experiments):"]
        for i, m in enumerate(self._memory[:4], 1):
            lines.append(
                f"\n[{i}] Dataset: {m['dataset']} | Task: {m['task']}\n"
                f"    Best strategy: {m.get('best_strategy','?')} "
                f"(top arm: {m.get('best_arm','?')})\n"
                f"    Key gene families: {', '.join(m.get('top_hit_families',[])[:3])}\n"
                f"    Insight: {m.get('strategy_insight','')}"
            )
        lines.append("\nApply these patterns to the current task.")
        return "\n".join(lines)

    def _build_prompt(self, round_idx: int) -> str:
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")

        memory_section = self._build_memory_section()

        history_lines = []
        for r, (hits, nonhits) in enumerate(zip(self._round_hits, self._round_nonhits)):
            hit_str = ", ".join(hits[:20]) + ("..." if len(hits) > 20 else "")
            nonhit_sample = nonhits[:10]
            nonhit_str = ", ".join(nonhit_sample) + f"... ({len(nonhits)} total)"
            history_lines.append(
                f"Round {r+1} — Hits ({len(hits)}): {hit_str or 'none'} | "
                f"Non-hits sample: {nonhit_str}"
            )

        history_section = ""
        if history_lines:
            history_section = "\n\nEXPERIMENTAL FEEDBACK FROM PREVIOUS ROUNDS:\n" + "\n".join(history_lines)
            all_hits = [g for rnd in self._round_hits for g in rnd]
            if all_hits:
                history_section += f"\n\nCumulative hits found so far: {', '.join(all_hits[:30])}"
                history_section += "\nUse the pattern of hits to infer which biological pathways or gene families to prioritize next."

        already_selected = list(self._selected)
        already_note = ""
        if already_selected:
            already_note = f"\n\nALREADY SELECTED (do NOT repeat): {', '.join(already_selected[:50])}{'...' if len(already_selected) > 50 else ''}"

        prompt = f"""You are a CRISPR screen expert selecting genes for a perturbation experiment.

TASK: {task_text}
MEASUREMENT: {measurement}
{memory_section}
You are in round {round_idx + 1} of a sequential CRISPR screen.{history_section}{already_note}

Select exactly {self.batch_size} human protein-coding gene symbols to perturb in this round.

Rules:
- Use standard HGNC gene symbols (e.g., TP53, EGFR, BRCA1)
- Do not repeat previously selected genes
- Choose genes most likely to affect the measured phenotype
- Prioritize biological relevance over coverage

Return ONLY a JSON array of {self.batch_size} gene symbols. No explanation, no markdown, just the JSON array.
Example: ["TP53", "EGFR", "BRCA1", "MYC"]"""
        return prompt

    def _call_llm(self, prompt: str) -> list[str]:
        response = self._client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Try JSON array parse
        try:
            genes = json.loads(text)
            if isinstance(genes, list):
                return [str(g).strip().upper() for g in genes if g]
        except json.JSONDecodeError:
            pass

        # Fallback: extract HGNC-like patterns
        raw = re.findall(r'\b[A-Z][A-Z0-9\-]{1,9}\b', text)
        return [g.upper() for g in raw]

    def _match_to_pool(self, llm_genes: list[str]) -> list[str]:
        """Return genes from llm_genes that are in the actual gene pool and not yet selected."""
        seen: set[str] = set()
        matched: list[str] = []
        for g in llm_genes:
            g = g.strip().upper()
            if g in self._gene_set and g not in self._selected and g not in seen:
                matched.append(g)
                seen.add(g)
        return matched

    def _fill_from_static(self, matched: list[str], n_needed: int) -> list[str]:
        """Fill remaining slots from StaticRanker order."""
        matched_set = set(matched)
        extra: list[str] = []
        for g in self._static_ranking:
            if len(extra) >= n_needed:
                break
            if g not in self._selected and g not in matched_set:
                extra.append(g)
        return extra

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        prompt = self._build_prompt(round_idx)
        llm_genes = self._call_llm(prompt)
        matched = self._match_to_pool(llm_genes)

        n_needed = self.batch_size - len(matched)
        if n_needed > 0:
            fallback = self._fill_from_static(matched, n_needed)
            matched = matched + fallback

        return matched[: self.batch_size]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        super().update(round_idx, revealed_new)
        hits = [g for g, is_hit in revealed_new.items() if is_hit]
        nonhits = [g for g, is_hit in revealed_new.items() if not is_hit]
        self._round_hits.append(hits)
        self._round_nonhits.append(nonhits)
