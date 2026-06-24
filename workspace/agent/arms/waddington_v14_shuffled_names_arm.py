"""
WaddingtonV14ShuffledNamesArm — Ablation: shuffled gene names.

Same architecture as V14 (online ML + LLM + routing), but ALL gene names
passed to the LLM are replaced with anonymous identifiers (GENE_00001, ...).

Tests whether LLM's contribution comes from biological gene-name semantics
or from other signals (task description, ML scores in shortlist, round feedback).

Expected result:
- Datasets where LLM was critical (Steinhart, essential): drop to ~C-LLM levels
- Large screens where LLM was neutral: no change
- Scharenberg22: if LLM selection was driven by gene names, should recover toward C-LLM level
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import anthropic

from .base import BaseArm
from .llm_reasoning_arm import LLMReasoningArm, LLM_MAX_TOKENS
from .online_adaptive_arm import OnlineAdaptiveArm
from .waddington_arm import _load_memory, _rank_memory_by_relevance, _load_task
from .waddington_v14_arm import (
    _get_feature_config,
    _get_dataset_stats,
    _classify,
    LLM_TEMPERATURE,
    LLM_MODEL,
    MEMORY_PATH,
    SHORTLIST_SIZE,
    W_ML_LARGE, W_LLM_LARGE,
    W_ML_DEFAULT, W_LLM_DEFAULT,
)


class _AnonymousNamesLLMArm(LLMReasoningArm):
    """LLMReasoningArm where all gene names are replaced with GENE_XXXXX identifiers.

    The LLM sees no real HGNC symbols anywhere: history, shortlist, already-selected.
    The only signals available are: biological task context and ML confidence scores.
    """

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        super().__init__(dataset_name, batch_size, **kwargs)
        # Fixed bijection: sort genes alphabetically for reproducibility
        sorted_genes = sorted(self._genes)
        self._r2f: dict[str, str] = {g: f"GENE_{i:05d}" for i, g in enumerate(sorted_genes)}
        self._f2r: dict[str, str] = {v: k for k, v in self._r2f.items()}
        self._gene_set_anon: set[str] = set(self._r2f.values())
        self._static_ranking_anon: list[str] = [self._r2f[g] for g in self._static_ranking]

    def _selected_anon(self) -> set[str]:
        return {self._r2f[g] for g in self._selected if g in self._r2f}

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        # Store anonymous names in history (used in prompt building)
        hits = [self._r2f[g] for g, v in revealed_new.items() if v and g in self._r2f]
        nonhits = [self._r2f[g] for g, v in revealed_new.items() if not v and g in self._r2f]
        self._round_hits.append(hits)
        self._round_nonhits.append(nonhits)
        # Track selected using real names (for BaseArm._selected)
        BaseArm.update(self, round_idx, revealed_new)

    def _on_reset(self) -> None:
        self._round_hits = []
        self._round_nonhits = []

    # ── Anonymous prompt builders ──────────────────────────────────────────────

    def _build_anon_prompt(self, round_idx: int) -> str:
        """Open-ended prompt: ask LLM to name GENE_XXXXX identifiers."""
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")
        memory_section = self._build_memory_section()

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
                history_section += f"\n\nCumulative hits: {', '.join(all_hits[:30])}"
                history_section += "\nUse these patterns to infer which identifiers to prioritize."

        sel_anon = list(self._selected_anon())
        already_note = ""
        if sel_anon:
            already_note = f"\n\nALREADY SELECTED (do NOT repeat): {', '.join(sel_anon[:50])}"

        return f"""You are selecting gene identifiers for a perturbation experiment.

TASK: {task_text}
MEASUREMENT: {measurement}
{memory_section}
You are in round {round_idx + 1} of a sequential screen.{history_section}{already_note}

Select exactly {self.batch_size} gene identifiers for this round.
Identifiers are in the format GENE_XXXXX (e.g., GENE_00042, GENE_00137).

Return ONLY a JSON array of {self.batch_size} identifiers. No explanation.
Example: ["GENE_00042", "GENE_00137", "GENE_00891"]"""

    def _build_anon_shortlist_prompt(
        self, round_idx: int, fake_shortlist: list[tuple[str, float]]
    ) -> str:
        """Shortlist prompt with anonymous gene names and ML scores."""
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")
        memory_section = self._build_memory_section()

        history_lines = []
        for r, (hits, nonhits) in enumerate(zip(self._round_hits, self._round_nonhits)):
            hit_str = ", ".join(hits[:20]) + ("..." if len(hits) > 20 else "")
            nonhit_str = ", ".join(nonhits[:10]) + f"... ({len(nonhits)} total)"
            history_lines.append(
                f"Round {r+1} — Hits ({len(hits)}): {hit_str or 'none'} | "
                f"Non-hits: {nonhit_str}"
            )
        history_section = ""
        if history_lines:
            history_section = "\n\nEXPERIMENTAL FEEDBACK:\n" + "\n".join(history_lines)
            all_hits = [g for rnd in self._round_hits for g in rnd]
            if all_hits:
                history_section += f"\n\nCumulative hits: {', '.join(all_hits[:30])}"

        sel_anon = list(self._selected_anon())
        already_note = ""
        if sel_anon:
            already_note = f"\n\nALREADY SELECTED: {', '.join(sel_anon[:50])}"

        cand_lines = [f"  {g} (ML confidence: {s:.3f})" for g, s in fake_shortlist]
        shortlist_section = "\n\nML-RANKED CANDIDATE POOL:\n" + "\n".join(cand_lines)

        return f"""You are selecting gene identifiers for a perturbation experiment.

TASK: {task_text}
MEASUREMENT: {measurement}
{memory_section}
Round {round_idx + 1} of a sequential screen.{history_section}{already_note}
{shortlist_section}

Select exactly {self.batch_size} identifiers from the candidate pool above.
- Prefer high ML-confidence candidates unless you have other signals from feedback.
- Do not repeat already-selected identifiers.

Return ONLY a JSON array of {self.batch_size} identifiers. No explanation.
Example: ["GENE_00042", "GENE_00137"]"""

    # ── Anonymous match / fill ─────────────────────────────────────────────────

    def _match_anon(self, llm_genes: list[str]) -> list[str]:
        seen: set[str] = set()
        matched: list[str] = []
        sel_anon = self._selected_anon()
        for g in llm_genes:
            g = g.strip().upper()
            if g in self._gene_set_anon and g not in sel_anon and g not in seen:
                matched.append(g)
                seen.add(g)
        return matched

    def _fill_anon_from_static(self, matched_anon: list[str], n_needed: int) -> list[str]:
        matched_set = set(matched_anon)
        sel_anon = self._selected_anon()
        extra: list[str] = []
        for g in self._static_ranking_anon:
            if len(extra) >= n_needed:
                break
            if g not in sel_anon and g not in matched_set:
                extra.append(g)
        return extra

    # ── Public interface (overrides parent) ────────────────────────────────────

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        prompt = self._build_anon_prompt(round_idx)
        llm_raw = self._call_llm(prompt)
        matched_anon = self._match_anon(llm_raw)
        n_needed = self.batch_size - len(matched_anon)
        if n_needed > 0:
            matched_anon = matched_anon + self._fill_anon_from_static(matched_anon, n_needed)
        result_anon = matched_anon[: self.batch_size]
        return [self._f2r[g] for g in result_anon if g in self._f2r]

    def select_with_shortlist(
        self, round_idx: int, shortlist: list[tuple[str, float]]
    ) -> list[str]:
        # Translate real names in shortlist to fake names
        fake_shortlist = [(self._r2f.get(g, g), s) for g, s in shortlist]
        prompt = self._build_anon_shortlist_prompt(round_idx, fake_shortlist)
        llm_raw = self._call_llm(prompt)
        matched_anon = self._match_anon(llm_raw)

        n_needed = self.batch_size - len(matched_anon)
        if n_needed > 0:
            # Fill from fake shortlist in ML order
            sel_anon = self._selected_anon()
            matched_set = set(matched_anon)
            for fg, _ in fake_shortlist:
                if n_needed <= 0:
                    break
                if fg not in sel_anon and fg not in matched_set:
                    matched_anon.append(fg)
                    matched_set.add(fg)
                    n_needed -= 1
        if len(matched_anon) < self.batch_size:
            matched_anon += self._fill_anon_from_static(matched_anon, self.batch_size - len(matched_anon))

        result_anon = matched_anon[: self.batch_size]
        return [self._f2r.get(g, g) for g in result_anon]


class WaddingtonV14ShuffledNamesArm(BaseArm):
    """C-arm with anonymous gene names passed to LLM. Tests LLM gene-name semantics."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__("waddington_v14_shuffled_names", dataset_name, batch_size)

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
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        memory = _rank_memory_by_relevance(raw_memory, task)[:4]
        self._llm = _AnonymousNamesLLMArm(
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
