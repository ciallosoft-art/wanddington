"""
WaddingtonArm — C-arm: ML shortlist + cross-experiment memory + LLM reasoning.

This is the core Waddington innovation: three-component sequential selection.

Round R flow:
  1. OnlineAdaptiveArm provides top-K candidates with ML confidence scores
     (K = SHORTLIST_FACTOR × batch_size, after in-experiment retraining)
  2. Cross-experiment memory (LOO) provides strategy insights from other datasets
  3. Claude reasons over candidates + memory + revealed feedback → selects batch

Expected: avg hit@R5 > max(OnlineAdaptive=0.224, LLMReasoning=0.220)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import anthropic

from .base import BaseArm
from .online_adaptive_arm import OnlineAdaptiveArm
from .llm_reasoning_arm import _load_auth_token, _load_task

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMORY_PATH = REPO_ROOT / "workspace" / "results" / "sequential" / "experience_memory.json"

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_TEMPERATURE = 0.3   # Lower than B-arm: memory context makes more determinism better
LLM_MAX_TOKENS = 1500
SHORTLIST_FACTOR = 3    # Show LLM top (3 × batch_size) ML candidates
MAX_MEMORY_ENTRIES = 4  # Max past experiments to show in prompt
MAX_CANDIDATES_IN_PROMPT = 80  # Cap shortlist shown to LLM to control prompt size


def _load_memory(memory_path: Path, exclude_dataset: str) -> list[dict]:
    if not memory_path.exists():
        return []
    with open(memory_path) as f:
        all_entries = json.load(f)
    return [e for e in all_entries if e.get("dataset") != exclude_dataset]


def _rank_memory_by_relevance(entries: list[dict], task: dict) -> list[dict]:
    """Sort memory entries so most relevant (similar task type) come first."""
    task_text = task.get("Task", "").lower()

    def relevance(entry: dict) -> float:
        score = 0.0
        entry_task = entry.get("task", "").lower()
        # Simple keyword overlap
        for word in task_text.split():
            if len(word) > 4 and word in entry_task:
                score += 1.0
        return score

    return sorted(entries, key=relevance, reverse=True)


class WaddingtonArm(BaseArm):
    """
    C-arm combining OnlineAdaptive ML + cross-experiment memory + LLM reasoning.

    The ML component provides an online-retrained shortlist of top candidates.
    The LLM reasons over this shortlist using:
      - Task description
      - Cross-experiment memory (strategy insights from LOO past experiments)
      - Revealed feedback from current experiment
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
        shortlist_factor: int = SHORTLIST_FACTOR,
    ) -> None:
        super().__init__("waddington", dataset_name, batch_size)

        # ML component (handles online retraining)
        self._online = OnlineAdaptiveArm(dataset_name, batch_size)

        # Task context
        self._task = _load_task(dataset_name)

        # Cross-experiment memory (LOO: exclude this dataset)
        raw_memory = _load_memory(memory_path, exclude_dataset=dataset_name)
        ranked = _rank_memory_by_relevance(raw_memory, self._task)
        self._memory = ranked[:MAX_MEMORY_ENTRIES]

        # LLM client
        self._client = anthropic.Anthropic(auth_token=_load_auth_token())
        self._shortlist_factor = shortlist_factor

        # Revealed history for LLM context
        self._round_hits: list[list[str]] = []
        self._round_nonhits: list[list[str]] = []

    def _on_reset(self) -> None:
        self._online.reset()
        self._round_hits = []
        self._round_nonhits = []

    def _build_memory_section(self) -> str:
        if not self._memory:
            return ""
        lines = [f"\nCROSS-EXPERIMENT MEMORY ({len(self._memory)} past experiments):"]
        for i, m in enumerate(self._memory, 1):
            lines.append(
                f"\n[{i}] Dataset: {m['dataset']} | Task: {m['task']}\n"
                f"    Best approach: {m.get('best_strategy','unknown')} "
                f"(top arm: {m.get('best_arm','?')})\n"
                f"    Key gene families: {', '.join(m.get('top_hit_families',[])[:4])}\n"
                f"    Notable genes: {', '.join(m.get('key_genes',[])[:6])}\n"
                f"    Insight: {m.get('strategy_insight','')}"
            )
        lines.append(
            "\nUse these patterns to calibrate your strategy for the current task."
        )
        return "\n".join(lines)

    def _build_history_section(self) -> str:
        if not self._round_hits and not self._round_nonhits:
            return ""
        lines = ["\nCURRENT EXPERIMENT FEEDBACK:"]
        all_hits: list[str] = []
        for r, (hits, nonhits) in enumerate(zip(self._round_hits, self._round_nonhits), 1):
            all_hits.extend(hits)
            hit_str = ", ".join(hits[:15]) + ("..." if len(hits) > 15 else "")
            lines.append(
                f"  Round {r}: {len(hits)} hits ({hit_str or 'none'}), "
                f"{len(nonhits)} non-hits"
            )
        if all_hits:
            lines.append(f"  Cumulative hits: {', '.join(all_hits[:25])}")
            lines.append("  → Infer biological theme from these hits to guide next selection.")
        return "\n".join(lines)

    def _build_candidates_section(self, candidates: list[tuple[str, float]]) -> str:
        shown = candidates[:MAX_CANDIDATES_IN_PROMPT]
        lines = [f"\nML CANDIDATE SHORTLIST (top {len(shown)} by current model confidence):"]
        lines.append("  Gene          ML_score")
        lines.append("  ────────────  ────────")
        for gene, score in shown:
            lines.append(f"  {gene:<14s}  {score:.3f}")
        if len(candidates) > MAX_CANDIDATES_IN_PROMPT:
            lines.append(f"  ... ({len(candidates) - MAX_CANDIDATES_IN_PROMPT} more not shown)")
        return "\n".join(lines)

    def _build_prompt(
        self, round_idx: int, candidates: list[tuple[str, float]]
    ) -> str:
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")

        memory_sec = self._build_memory_section()
        history_sec = self._build_history_section()
        candidates_sec = self._build_candidates_section(candidates)

        already = list(self._selected)
        already_note = ""
        if already:
            already_note = (
                f"\nALREADY SELECTED (do NOT include): "
                f"{', '.join(already[:30])}{'...' if len(already) > 30 else ''}"
            )

        return f"""You are the Waddington gene selection agent combining ML predictions with biological reasoning.

TASK: {task_text}
MEASUREMENT: {measurement}
ROUND: {round_idx + 1}{memory_sec}{history_sec}{candidates_sec}{already_note}

INSTRUCTIONS:
Select exactly {self.batch_size} genes for this round. You SHOULD primarily choose from the ML candidate shortlist (they are pre-filtered by the model). You MAY also suggest genes NOT in the shortlist if you have strong biological justification from the cross-experiment memory or task knowledge.

Strategy guidance:
- If memory shows similar tasks favored LLM knowledge → feel free to add genes you know from biology
- If memory shows similar tasks favored ML features → trust the ML shortlist order
- Blend both if unclear

Return ONLY a JSON array of {self.batch_size} gene symbols. No explanation.
Example: ["JAK1", "STAT1", "IRF1", ...]"""

    def _call_llm(self, prompt: str, candidates: list[tuple[str, float]]) -> list[str]:
        response = self._client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse JSON array
        try:
            genes = json.loads(text)
            if isinstance(genes, list):
                return [str(g).strip().upper() for g in genes if g]
        except json.JSONDecodeError:
            pass

        # Fallback: extract HGNC-like patterns
        raw = re.findall(r'\b[A-Z][A-Z0-9\-]{1,9}\b', text)
        return [g.upper() for g in raw]

    def _match_and_fill(
        self,
        llm_genes: list[str],
        candidates: list[tuple[str, float]],
    ) -> list[str]:
        """
        Match LLM suggestions to actual gene pool.
        Fill remaining slots from ML candidates in order.
        """
        # Build gene pool from oracle (all unselected)
        candidate_genes = {g for g, _ in candidates}
        # Also allow LLM to pick from full gene pool (if it suggests valid genes)
        # We'll validate against the online arm's gene list
        all_genes_set = set(self._online._genes)

        seen: set[str] = set()
        matched: list[str] = []
        for g in llm_genes:
            g = g.strip().upper()
            if g in all_genes_set and g not in self._selected and g not in seen:
                matched.append(g)
                seen.add(g)

        # Fill remaining from ML candidates in ranked order
        if len(matched) < self.batch_size:
            for g, _ in candidates:
                if len(matched) >= self.batch_size:
                    break
                if g not in self._selected and g not in seen:
                    matched.append(g)
                    seen.add(g)

        return matched[: self.batch_size]

    def select(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        # Get ML shortlist
        n_candidates = min(
            self._shortlist_factor * self.batch_size,
            len(self._online._genes) - len(self._selected),
        )
        candidates = self._online.ranked_candidates(n_candidates, exclude=self._selected)

        # Build prompt and call LLM
        prompt = self._build_prompt(round_idx, candidates)
        llm_genes = self._call_llm(prompt, candidates)

        return self._match_and_fill(llm_genes, candidates)

    def update(self, round_idx: int, revealed_new: dict[str, bool]) -> None:
        # Update ML model (online retraining)
        self._online.update(round_idx, revealed_new)
        # Update base (selected set)
        super().update(round_idx, revealed_new)
        # Track history for LLM context
        hits = [g for g, h in revealed_new.items() if h]
        nonhits = [g for g, h in revealed_new.items() if not h]
        self._round_hits.append(hits)
        self._round_nonhits.append(nonhits)
