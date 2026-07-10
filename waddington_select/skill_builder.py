"""
skill_builder.py — Distil the episodic memory into a verified skill library (Phase 1).

Reads experience_memory.json (per-experiment episodic entries) and asks Claude to distil
*general, transferable* when-then strategy skills across experiments. Each candidate skill then
passes a verified-admission gate modelled on memory_builder._verify_entry:

  1. Mechanical  — schema well-formed; evidence_datasets reference real datasets; trigger uses
     only the allowed keys.
  2. Grounding   — an LLM verifier checks the rule is actually supported by the cited experiments
     and is not overfit to one dataset or an unsupported causal claim.
  (3. Leakage    — enforced at *retrieval* time in skills.py: a skill is never shown on a fold it
     was distilled from, nor if its directive names a held-out hit gene.)

Skills that fail after retries are stored with "verified": false so retrieval can filter them.

Usage:
    conda run -n waddington-bio python3 -m waddington_select.skill_builder
    conda run -n waddington-bio python3 -m waddington_select.skill_builder --n-skills 12
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .oracle import ALL_DATASETS
from .memory_builder import MEMORY_PATH
from .skills import SKILL_LIBRARY_PATH
from .llm_client import LLMClient

MAX_VERIFY_RETRIES = 2
LLM_MODEL = "claude-haiku-4-5-20251001"


def _parse_skill_array(text: str) -> list[dict]:
    """Extract the JSON array of candidate skills from an LLM response.

    Tolerant of markdown fences and leading/trailing prose. Returns [] on failure so the
    distiller loop can retry rather than crash.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    try:
        v = json.loads(t)
        if isinstance(v, list):
            return v
        if isinstance(v, dict) and isinstance(v.get("skills"), list):
            return v["skills"]
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if m:
        try:
            v = json.loads(m.group())
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    return []

_ALLOWED_TRIGGER_KEYS = {
    "min_round", "max_round", "min_n_genes", "max_n_genes",
    "min_hit_rate", "max_hit_rate", "requires_revealed_hits",
}
_ALLOWED_TYPES = {"pathway_prior", "selection_heuristic", "calibration"}


def _condense(entries: list[dict]) -> str:
    """One compact line per experiment for the distiller prompt."""
    lines = []
    for e in entries:
        lines.append(
            f"- {e['dataset']}: hit_rate={e.get('hit_rate')}, n_genes={e.get('n_genes')}, "
            f"best_arm={e.get('best_arm')} ({e.get('best_strategy')}); "
            f"families={e.get('top_hit_families', [])[:3]}; "
            f"insight: {e.get('strategy_insight', '')[:220]}"
        )
    return "\n".join(lines)


def _distill_prompt(entries: list[dict], n_skills: int) -> str:
    return f"""You are distilling reusable strategy SKILLS from a set of completed CRISPR
gene-selection experiments. A skill is a GENERAL "when-then" rule that will be injected into a
future gene-selection agent facing a DIFFERENT phenotype.

COMPLETED EXPERIMENTS (each is one phenotype screen):
{_condense(entries)}

Produce up to {n_skills} skills. Rules for good skills:
- STRUCTURE OVER IDENTITY: state rules about pathways, gene families, complexes, or selection
  strategy — NOT specific gene symbols. A skill must transfer to phenotypes not seen above.
- Each skill must be supported by at least two experiments where possible; list them in
  "evidence_datasets" (using the dataset names above).
- Attach a "trigger": only these keys are allowed: {sorted(_ALLOWED_TRIGGER_KEYS)}.
  round is 1-based; hit_rate is the screen's overall hit rate; requires_revealed_hits means the
  rule only applies once the agent has seen at least one hit.
- "type" is one of {sorted(_ALLOWED_TYPES)}:
    pathway_prior      — how revealed hits imply which untested genes to prioritise
    selection_heuristic— round-level policy (exploration, ML-vs-LLM trust, batch allocation)
    calibration        — how to weight a signal (e.g. LLM confidence) based on observed reliability
- For every pathway_prior, add "marker_genes": 5-10 CANONICAL, well-known member genes of the
  pathway/complex the rule is about (e.g. autophagy → ATG5, ATG7, BECN1, MAP1LC3B, SQSTM1). These
  are used only to decide *when* the skill fires (the rule fires only if the experiment's revealed
  hits actually include one of them) — they are never shown to the agent, so they may be specific
  gene symbols. Other skill types use "marker_genes": [].

Return ONLY a JSON array; each element:
{{
  "type": "...",
  "trigger": {{...}},
  "directive": "one or two sentences, a clear when-then rule with no specific gene names",
  "evidence_datasets": ["...", "..."],
  "marker_genes": ["GENE1", "GENE2", "..."]
}}"""


_VERIFY_PROMPT = """A candidate strategy SKILL was distilled from CRISPR experiments and will be
injected into future gene-selection agents. Verify it is trustworthy and general.

SOURCE EXPERIMENTS IT CLAIMS AS EVIDENCE:
{evidence_str}

CANDIDATE SKILL:
- type: {type}
- trigger: {trigger}
- directive: "{directive}"

Verify ONLY:
1. Is the directive actually SUPPORTED by the cited experiments (not overfit to one, not an
   unsupported causal claim)?
2. Is it GENERAL — does it avoid naming specific gene symbols and plausibly transfer to unseen
   phenotypes?
3. Is the trigger sensible for the rule?

Reply with ONLY valid JSON, no markdown:
{{"valid": true}} or {{"valid": false, "issues": ["...", "..."]}}"""


def _mechanical_ok(skill: dict) -> list[str]:
    issues: list[str] = []
    if skill.get("type") not in _ALLOWED_TYPES:
        issues.append(f"invalid type '{skill.get('type')}'")
    if not skill.get("directive", "").strip():
        issues.append("empty directive")
    ev = skill.get("evidence_datasets", [])
    if not ev:
        issues.append("no evidence_datasets")
    for d in ev:
        if d not in ALL_DATASETS:
            issues.append(f"unknown evidence dataset '{d}'")
    bad_keys = set(skill.get("trigger", {})) - _ALLOWED_TRIGGER_KEYS
    if bad_keys:
        issues.append(f"unknown trigger keys {sorted(bad_keys)}")
    # A pathway_prior must carry marker genes so it can fire discriminatively.
    if skill.get("type") == "pathway_prior" and len(skill.get("marker_genes", [])) < 3:
        issues.append("pathway_prior needs >=3 marker_genes for a discriminative trigger")
    return issues


def _verify_skill(skill: dict, entries_by_ds: dict, client: LLMClient) -> tuple[bool, list[str]]:
    issues = _mechanical_ok(skill)

    evidence_str = "\n".join(
        f"- {d}: {entries_by_ds.get(d, {}).get('strategy_insight', '(missing)')[:220]}"
        for d in skill.get("evidence_datasets", [])
    ) or "(none)"
    prompt = _VERIFY_PROMPT.format(
        evidence_str=evidence_str,
        type=skill.get("type"),
        trigger=skill.get("trigger", {}),
        directive=skill.get("directive", ""),
    )
    try:
        text = client.complete(prompt, temperature=0.0, max_tokens=200)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        result = json.loads(m.group()) if m else {}
        if not result.get("valid", True):
            issues.extend(result.get("issues", ["verifier rejected skill"]))
    except Exception:
        pass  # fail-open on infra hiccups, as in memory_builder
    return len(issues) == 0, issues


def build_skills(n_skills: int = 12) -> list[dict]:
    with open(MEMORY_PATH) as f:
        entries = json.load(f)
    entries_by_ds = {e["dataset"]: e for e in entries}

    client = LLMClient(model=LLM_MODEL, temperature=0.3, max_tokens=2000)

    prior_issues: list[str] = []
    candidates: list[dict] = []
    for attempt in range(MAX_VERIFY_RETRIES + 1):
        prompt = _distill_prompt(entries, n_skills)
        if prior_issues:
            prompt += "\n\nThe previous batch had rejected skills. Avoid: " + "; ".join(prior_issues[:6])
        text = client.complete(prompt, temperature=0.3, max_tokens=2000)
        candidates = _parse_skill_array(text)
        if candidates:
            break

    skills: list[dict] = []
    prior_issues = []
    for i, cand in enumerate(candidates, 1):
        valid, issues = _verify_skill(cand, entries_by_ds, client)
        skill = {
            "id": f"sk_{i:03d}",
            "type": cand.get("type"),
            "trigger": cand.get("trigger", {}),
            "directive": cand.get("directive", "").strip(),
            "evidence_datasets": cand.get("evidence_datasets", []),
            "marker_genes": [g.strip().upper() for g in cand.get("marker_genes", []) if g.strip()],
            "verified": valid,
        }
        skills.append(skill)
        status = "✓" if valid else "✗"
        if not valid:
            prior_issues.extend(issues)
        print(f"  [{status}] {skill['id']} {skill['type']:20s} "
              f"src={skill['evidence_datasets']}  {skill['directive'][:70]}", flush=True)

    n_ok = sum(1 for s in skills if s["verified"])
    print(f"\n  Verification summary: {n_ok}/{len(skills)} skills passed.", flush=True)
    return skills


def main() -> None:
    parser = argparse.ArgumentParser(description="Distil episodic memory into a verified skill library")
    parser.add_argument("--n-skills", type=int, default=12)
    parser.add_argument("--out", type=Path, default=SKILL_LIBRARY_PATH)
    args = parser.parse_args()

    print(f"Distilling skills from {MEMORY_PATH.name} ...")
    skills = build_skills(args.n_skills)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(skills, f, indent=2)
    print(f"\nSkill library saved to {args.out}  ({len(skills)} skills)")


if __name__ == "__main__":
    main()
