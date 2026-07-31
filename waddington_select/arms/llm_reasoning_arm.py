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
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from .base import BaseArm
from ..skills import SkillLibrary, load_dataset_hits
from ..llm_client import LLMClient
from ..features import load_training_frame
from ..phenotype import task_prompt as _user_task_prompt

REPO_ROOT = Path(__file__).resolve().parents[2]
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

# Enrichment confidence gate: inject a pathway only if significant AND coherent (>=N hits in it).
_ENRICH_GATE_PVAL = 1e-3
_ENRICH_GATE_MIN_OVERLAP = 2

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
    # Reason-vs-recall validation: the Schmidt CRISPRa (gain-of-function) arm. Framed like Steinhart
    # ("upon activation ...") so the LLM knows it is an OVER-EXPRESSION screen, not knockdown — without
    # this the gene name carries no usable signal because the model does not know the question.
    "IL2_crispra": {
        "Task": "identify genes that, when activated (over-expressed via CRISPRa), regulate the production of Interleukin-2 (IL-2) in primary human T cells",
        "Measurement": "the log fold change in normalized sgRNA read counts between IL-2-high and IL-2-low sorting bins in a genome-wide CRISPR activation (CRISPRa) screen",
    },
    "IFNG_crispra": {
        "Task": "identify genes that, when activated (over-expressed via CRISPRa), regulate the production of Interferon-gamma (IFN-gamma) in primary human T cells",
        "Measurement": "the log fold change in normalized sgRNA read counts between IFN-gamma-high and IFN-gamma-low sorting bins in a genome-wide CRISPR activation (CRISPRa) screen",
    },
}

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE = 0.5
LLM_MAX_TOKENS = 1500


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
    if dataset_name in HARDCODED_TASKS:
        return HARDCODED_TASKS[dataset_name]
    user = _user_task_prompt(dataset_name)  # a registered new phenotype describes itself
    if user:
        return user
    # No task description anywhere. The LLM would be told to select for a phenotype named literally
    # "<dataset>" with measurement "gene fitness" — a silent garbage prompt that quietly nullifies the
    # LLM's contribution (it can't reason about a phenotype it was never told). Warn loudly; a benchmark
    # dataset reaching here is a bug (add a TASK_PROMPT_FILES / HARDCODED_TASKS entry).
    import sys
    print(f"    [WARN] no task prompt for '{dataset_name}' — the LLM will get a placeholder and "
          f"contribute little. Add it to HARDCODED_TASKS or TASK_PROMPT_FILES.", file=sys.stderr, flush=True)
    return {"Task": dataset_name, "Measurement": "gene fitness"}


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
        temperature: float = LLM_TEMPERATURE,
        model: str = LLM_MODEL,
        training_csv: Path | None = None,
        extra_feature_cols: list[str] | None = None,
        skill_library: SkillLibrary | None = None,
        dataset_hit_rate: float = 0.0,
        use_enrichment: bool = False,
    ) -> None:
        super().__init__("llm_reasoning", dataset_name, batch_size)
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._memory: list[dict] = memory_entries or []
        # Evolving skill library (Phase 1: trigger-conditioned retrieval, no evolution).
        self._skill_library = skill_library
        self._dataset_hit_rate = dataset_hit_rate
        self._block_genes: set[str] = (
            load_dataset_hits(dataset_name) if skill_library is not None else set()
        )
        # Runtime enrichment: inject Enrichr pathways of the hits found so far (the C-arm's
        # enrichment-augmented "hybrid" — transplants the agent's winning ingredient).
        self._use_enrichment = use_enrichment
        self._enrich_cache: dict = {}
        self._temperature = temperature
        self._model = model

        csv_path = training_csv if training_csv is not None else TRAINING_DATA_CSV
        df = load_training_frame(csv_path, dataset_name)
        df["gene"] = df["gene"].str.strip().str.upper()
        all_feats = FEATURE_COLS + (extra_feature_cols or [])
        self._all_feats = [c for c in all_feats if c in df.columns]
        # Clean-headline variant (see online_adaptive_arm): drop anchor-relative features from the static
        # padding ranker too, so no privileged anchor signal enters the batch by any path.
        if os.environ.get("WADDINGTON_DROP_ANCHOR_FEATS") == "1":
            self._all_feats = [c for c in self._all_feats
                               if c not in {"g1_ppi_score", "archs4_coexpr", "kegg_overlap"}]

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

        # Seconds to pause before each LLM call. Multi-seed sweeps issue hundreds of large prompts and
        # will trip a sustained account rate limit, which the retry backoff (capped at 120s) cannot
        # outlast — it just burns 8 attempts and dies. Throttling up front is what keeps a sweep alive.
        self._inter_call_sleep = float(os.environ.get("WADDINGTON_LLM_SLEEP", "0"))

        # Provider-agnostic LLM client (default: anthropic; opt-in pi/codex via
        # WADDINGTON_LLM_BACKEND=pi, reusing feynman's provider routing + OAuth).
        self._llm = LLMClient(
            model=self._model,
            temperature=self._temperature,
            max_tokens=LLM_MAX_TOKENS,
        )

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

    def _build_skill_section(self, round_idx: int) -> str:
        """Retrieve and render the skills whose triggers fire in the current round state."""
        if self._skill_library is None or len(self._skill_library) == 0:
            return ""
        revealed_hits = {g for rnd in self._round_hits for g in rnd}
        state = {
            "round": round_idx + 1,
            "n_genes": len(self._genes),
            "hit_rate": self._dataset_hit_rate,
            "n_revealed_hits": len(revealed_hits),
            "revealed_hits": revealed_hits,
        }
        firing = self._skill_library.retrieve(
            state,
            exclude_dataset=self.dataset_name,
            block_genes=self._block_genes,
            k=4,
        )
        return SkillLibrary.render(firing)

    def _build_enrichment_section(self, round_idx: int) -> str:
        """Enrichr pathways of the hits revealed so far (runtime, this experiment's own hits).

        Confidence-gated: a pathway is injected only if it is both significant (p < 1e-3) and
        coherent (>=2 revealed hits fall in it). Suppresses the noisy/diffuse enrichment that
        distracted the LLM on genome-wide / weak-signal phenotypes, keeping only strong signals.
        """
        if not self._use_enrichment:
            return ""
        hits = [g for rnd in self._round_hits for g in rnd]
        if len(hits) < 3:
            return ""
        key = frozenset(hits)
        if key not in self._enrich_cache:
            from ..tools import enrich  # lazy import to avoid a circular import at module load
            self._enrich_cache[key] = enrich(hits, top=12)
        terms = self._enrich_cache[key].get("terms", [])
        strong = [t for t in terms
                  if t.get("pval", 1.0) < _ENRICH_GATE_PVAL
                  and len(t.get("overlap_genes", [])) >= _ENRICH_GATE_MIN_OVERLAP]
        if not strong:
            return ""  # no confident, coherent pathway → inject nothing (avoid noise)
        lines = ["\nACTIVE PATHWAYS (significant enrichment of the hits found so far):"]
        for t in strong[:5]:
            lines.append(f"  - {t['term']} (hit genes: {', '.join(t.get('overlap_genes', [])[:5])})")
        lines.append("Prioritise untested genes in these active pathways.")
        return "\n".join(lines)

    def _build_prompt(self, round_idx: int) -> str:
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")

        memory_section = self._build_memory_section()
        skill_section = self._build_skill_section(round_idx)
        enrichment_section = self._build_enrichment_section(round_idx)

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
{memory_section}{skill_section}{enrichment_section}
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
        # Proactive inter-call sleep to stay within rate limits for larger models
        if getattr(self, "_inter_call_sleep", 0) > 0:
            time.sleep(self._inter_call_sleep)

        text = self._llm.complete(prompt).strip()

        # Some backends (the pi bridge) wrap the JSON in a ```json … ``` fence. Strip it before parsing,
        # otherwise json.loads fails and the regex fallback below splits underscored identifiers like
        # GENE_00001 into "GENE"+"00001" — which match no anonymized pool, silently forcing 100% padding.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text).strip()

        # Try JSON array parse
        try:
            genes = json.loads(text)
            if isinstance(genes, list):
                return [str(g).strip().upper() for g in genes if g]
        except json.JSONDecodeError:
            pass

        # Fallback: extract HGNC-like patterns (allow underscores so GENE_00001 survives)
        raw = re.findall(r'\b[A-Z][A-Z0-9_\-]{1,12}\b', text)
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

    def _build_shortlist_prompt(
        self, round_idx: int, shortlist: list[tuple[str, float]]
    ) -> str:
        """Prompt variant that shows an ML candidate shortlist."""
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")
        memory_section = self._build_memory_section()
        skill_section = self._build_skill_section(round_idx)
        enrichment_section = self._build_enrichment_section(round_idx)

        history_lines = []
        for r, (hits, nonhits) in enumerate(zip(self._round_hits, self._round_nonhits)):
            hit_str = ", ".join(hits[:20]) + ("..." if len(hits) > 20 else "")
            nonhit_str = ", ".join(nonhits[:10]) + f"... ({len(nonhits)} total)"
            history_lines.append(
                f"Round {r+1} — Hits ({len(hits)}): {hit_str or 'none'} | "
                f"Non-hits sample: {nonhit_str}"
            )
        history_section = ""
        if history_lines:
            history_section = "\n\nEXPERIMENTAL FEEDBACK:\n" + "\n".join(history_lines)
            all_hits = [g for rnd in self._round_hits for g in rnd]
            if all_hits:
                history_section += (
                    f"\n\nCumulative hits so far: {', '.join(all_hits[:30])}"
                    "\nUse the hit pattern to infer pathways and prioritize next candidates."
                )

        already_selected = list(self._selected)
        already_note = ""
        if already_selected:
            already_note = (
                f"\n\nALREADY SELECTED (do NOT repeat): "
                f"{', '.join(already_selected[:50])}{'...' if len(already_selected) > 50 else ''}"
            )

        cand_lines = [f"  {g} (ML confidence: {s:.3f})" for g, s in shortlist]
        shortlist_section = "\n\nML-RANKED CANDIDATE POOL:\n" + "\n".join(cand_lines)

        return f"""You are a CRISPR screen expert selecting genes for a perturbation experiment.

TASK: {task_text}
MEASUREMENT: {measurement}
{memory_section}{skill_section}{enrichment_section}
You are in round {round_idx + 1} of a sequential CRISPR screen.{history_section}{already_note}
{shortlist_section}

INSTRUCTIONS:
Select exactly {self.batch_size} genes for this round.
- Prefer genes from the ML candidate pool above (they are pre-filtered by the model).
- You MAY also suggest genes NOT in the pool if you have strong biological justification.
- Use standard HGNC gene symbols. Do not repeat already-selected genes.

Return ONLY a JSON array of {self.batch_size} gene symbols. No explanation.
Example: ["TP53", "EGFR", "BRCA1", "MYC"]"""

    def select_with_shortlist(
        self, round_idx: int, shortlist: list[tuple[str, float]]
    ) -> list[str]:
        """Select batch_size genes from an ML shortlist (two-stage strategy)."""
        prompt = self._build_shortlist_prompt(round_idx, shortlist)
        llm_genes = self._call_llm(prompt)
        matched = self._match_to_pool(llm_genes)

        n_needed = self.batch_size - len(matched)
        if n_needed > 0:
            # Fill from ML shortlist in ranked order
            shortlist_set = set(matched)
            for g, _ in shortlist:
                if n_needed <= 0:
                    break
                if g not in self._selected and g not in shortlist_set:
                    matched.append(g)
                    shortlist_set.add(g)
                    n_needed -= 1
        if len(matched) < self.batch_size:
            # Final fallback: StaticRanker
            fallback = self._fill_from_static(matched, self.batch_size - len(matched))
            matched = matched + fallback

        return matched[: self.batch_size]

    def _build_confidence_prompt(self, round_idx: int) -> str:
        """Like _build_prompt but also requests an overall confidence score."""
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")
        memory_section = self._build_memory_section()
        skill_section = self._build_skill_section(round_idx)
        enrichment_section = self._build_enrichment_section(round_idx)

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

        return f"""You are a CRISPR screen expert selecting genes for a perturbation experiment.

TASK: {task_text}
MEASUREMENT: {measurement}
{memory_section}{skill_section}{enrichment_section}
You are in round {round_idx + 1} of a sequential CRISPR screen.{history_section}{already_note}

Select exactly {self.batch_size} human protein-coding gene symbols to perturb in this round.

Rules:
- Use standard HGNC gene symbols (e.g., TP53, EGFR, BRCA1)
- Do not repeat previously selected genes
- Choose genes most likely to affect the measured phenotype
- Prioritize biological relevance over coverage

Return ONLY a JSON object with two fields:
- "genes": an array of exactly {self.batch_size} gene symbols
- "confidence": your overall confidence (0.0-1.0) that these selections are biologically relevant

Example: {{"genes": ["TP53", "EGFR", "BRCA1", "MYC"], "confidence": 0.75}}
No explanation, no markdown."""

    def select_with_confidence(
        self, round_idx: int, revealed: dict[str, bool]
    ) -> tuple[list[str], float]:
        """Select genes and return (matched_genes, llm_confidence_0_to_1)."""
        if getattr(self, "_inter_call_sleep", 0) > 0:
            time.sleep(self._inter_call_sleep)

        prompt = self._build_confidence_prompt(round_idx)
        text = self._llm.complete(prompt)
        llm_genes: list[str] = []
        llm_conf: float = 0.5

        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "genes" in obj:
                llm_genes = [str(g).strip().upper() for g in obj["genes"] if g]
                raw_conf = obj.get("confidence", 0.5)
                llm_conf = float(max(0.0, min(1.0, float(raw_conf))))
            elif isinstance(obj, list):
                llm_genes = [str(g).strip().upper() for g in obj if g]
                llm_conf = 0.5
        except (json.JSONDecodeError, ValueError, TypeError):
            raw = re.findall(r'\b[A-Z][A-Z0-9\-]{1,9}\b', text)
            llm_genes = [g.upper() for g in raw]
            conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
            if conf_match:
                try:
                    llm_conf = max(0.0, min(1.0, float(conf_match.group(1))))
                except ValueError:
                    pass

        matched = self._match_to_pool(llm_genes)
        n_needed = self.batch_size - len(matched)
        if n_needed > 0:
            matched = matched + self._fill_from_static(matched, n_needed)
        return matched[: self.batch_size], llm_conf

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        prompt = self._build_prompt(round_idx)
        llm_genes = self._call_llm(prompt)
        matched = self._match_to_pool(llm_genes)

        # Record what the LLM ACTUALLY named, before any back-fill. The batch is padded from the
        # static LOO ranker when the LLM names too few valid genes, and on some screens that padding
        # dominates (K562-Essential: the LLM matched 1 gene of 32). Attribution must not credit those
        # static-ranker genes to the LLM — the fusion still scores them, but they are not LLM picks.
        self._llm_named: set[str] = set(matched[: self.batch_size])

        n_needed = self.batch_size - len(matched)
        if n_needed > 0:
            fallback = self._fill_from_static(matched, n_needed)
            matched = matched + fallback

        self._n_fallback: int = max(0, n_needed)
        return matched[: self.batch_size]

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        super().update(round_idx, revealed_new)
        hits = [g for g, is_hit in revealed_new.items() if is_hit]
        nonhits = [g for g, is_hit in revealed_new.items() if not is_hit]
        self._round_hits.append(hits)
        self._round_nonhits.append(nonhits)
