"""
WaddingtonCFeatureReasoningArm — Ablation A1: anonymized names + structural features.

The `waddington_c_shuffled_names` arm (A0) renames every gene to GENE_XXXXX and gives the LLM nothing
else — only anonymous IDs and which past IDs were hits. That measures a blindfolded guesser, not a
reasoner. This arm (A1) keeps the anonymization but hands the LLM the SAME structural feature vectors
the GBM uses (PPI proximity to the task anchors, co-expression, pathway overlap, network degree,
constraint, essentiality), plus, each round, the mean feature profile of the revealed hits vs
non-hits. The LLM can no longer *recall* a gene by name, but it *can* reason:
"hits so far run high on ppi_task and low on essentiality; this candidate matches — pick it."

The decomposition this enables (all measured in the identical routing / fusion harness):

    A1 - A0   value of giving the anonymized LLM structure to reason over
    A2 - A1   value of the real gene name (parametric recall) on top of reasoning   ← the question
    A1 vs online_adaptive   LLM in-context reasoning vs a GBM on identical features

Design invariant: the LLM never sees the ML confidence score. If it did, "reasoning" would collapse to
copying the GBM. It sees only the raw (anonymized) feature vectors and must induce the decision rule
itself. The fusion downstream still uses the ML score — but that is the harness, held constant across
A0/A1/A2, so the contrasts stay clean.
"""

from __future__ import annotations

from pathlib import Path

from .llm_reasoning_arm import TRAINING_DATA_CSV
from ..features import load_training_frame
from .waddington_c_arm import MEMORY_PATH
from .waddington_c_shuffled_names_arm import (
    _AnonymousNamesLLMArm,
    WaddingtonCShuffledNamesArm,
)

# Raw feature column → readable, still-anonymous tag. The "_task" suffix flags anchor-relative features
# (proximity to the phenotype's core genes); the rest are gene-intrinsic. No tag names a gene.
_FEATURE_TAGS: dict[str, str] = {
    "g1_ppi_score": "ppi_task",
    "archs4_coexpr": "coexpr_task",
    "kegg_overlap": "pathway_task",
    "string_degree_norm": "net_degree",
    "hub_score_norm": "hubness",
    "ppi_score_sum": "ppi_total",
    "pli_score": "constraint",
    "kegg_pathway_count_norm": "n_paths_kegg",
    "reactome_pathway_count_norm": "n_paths_react",
    "depmap_frac_ess": "essentiality_frac",
    "depmap_mean_norm": "essentiality_mean",
    "depmap_min_norm": "essentiality_min",
    "depmap_K562_norm": "essentiality_k562",
}


class _AnonymousFeaturesLLMArm(_AnonymousNamesLLMArm):
    """Anonymous-name LLM arm that additionally exposes each candidate's structural feature vector and
    the learned hit/non-hit feature profile, so the LLM can reason rather than recall."""

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        super().__init__(dataset_name, batch_size, **kwargs)

        # A tractable, selective candidate pool: present ~3x the batch so the LLM actually chooses.
        self._feature_shortlist_size = max(200, 3 * batch_size)

        # Build anon_id -> {tag: value} using the routed feature set (self._all_feats already reflects
        # per-dataset DepMap routing, so e.g. Steinhart carries no essentiality columns).
        csv_path = kwargs.get("training_csv") or TRAINING_DATA_CSV
        df = load_training_frame(csv_path, dataset_name)
        df["gene"] = df["gene"].str.strip().str.upper()
        sub = df[df["dataset"] == dataset_name]
        feats = [c for c in self._all_feats if c in sub.columns]
        self._feat_by_anon: dict[str, dict[str, float]] = {}
        for row in sub.itertuples(index=False):
            real = getattr(row, "gene")
            anon = self._r2f.get(real)
            if anon is None:
                continue
            self._feat_by_anon[anon] = {
                _FEATURE_TAGS.get(c, c): float(getattr(row, c)) for c in feats
            }
        # Stable tag order for rendering (follow _all_feats order, mapped to tags).
        self._feat_tags: list[str] = [_FEATURE_TAGS.get(c, c) for c in feats]

    # ── Rendering helpers ──────────────────────────────────────────────────────

    def _feat_line(self, anon: str) -> str:
        d = self._feat_by_anon.get(anon, {})
        parts = " ".join(f"{t} {d.get(t, 0.0):.2f}" for t in self._feat_tags)
        return f"  {anon} | {parts}"

    def _hit_profile_section(self) -> str:
        """Mean feature vector of revealed hits vs non-hits — the induction signal the LLM applies."""
        all_hits = [g for rnd in self._round_hits for g in rnd]
        all_non = [g for rnd in self._round_nonhits for g in rnd]
        if not all_hits:
            return ""

        def mean_vec(ids: list[str]) -> dict[str, float]:
            rows = [self._feat_by_anon[g] for g in ids if g in self._feat_by_anon]
            if not rows:
                return {}
            return {t: sum(r.get(t, 0.0) for r in rows) / len(rows) for t in self._feat_tags}

        hp = mean_vec(all_hits)
        npf = mean_vec(all_non)
        lines = []
        for t in self._feat_tags:
            hv = hp.get(t, 0.0)
            nv = npf.get(t)
            if nv is None:
                lines.append(f"  {t}: hits avg {hv:.2f}")
            else:
                if hv > nv + 0.02:
                    tag = "HIGHER in hits"
                elif hv < nv - 0.02:
                    tag = "LOWER in hits"
                else:
                    tag = "similar"
                lines.append(f"  {t}: hits {hv:.2f} vs non-hits {nv:.2f}  ({tag})")
        return (
            f"\n\nLEARNED PROFILE ({len(all_hits)} hits, {len(all_non)} non-hits so far) — "
            "the feature signature that separated hits from non-hits:\n" + "\n".join(lines)
        )

    def _candidate_block(self, anon_candidates: list[str], limit: int | None = None) -> str:
        """Feature lines for the candidate pool, sorted by anon id so ML rank cannot leak via order.

        `limit` MUST stay None whenever the caller already supplies the arm's real candidate set (the
        two-stage shortlist): truncating it to the ML top-N would hand A1 a tighter, richer pool than
        A0/A2 see, and A1 would beat them without reasoning at all. Only the open route — where the
        alternative is 18k unreadable rows — may cap the pool.
        """
        sel = self._selected_anon()
        pool = [g for g in anon_candidates if g not in sel]
        if limit is not None:
            pool = pool[:limit]
        pool = sorted(pool)  # break any rank ordering
        lines = "\n".join(self._feat_line(g) for g in pool)
        return f"\nCANDIDATE POOL ({len(pool)} genes, each with its structural features):\n{lines}"

    def _instructions(self) -> str:
        return (
            f"\nSelect exactly {self.batch_size} identifiers from the pool above.\n"
            "You do NOT know the gene names. Decide only by reasoning over the features:\n"
            "- Prefer candidates whose feature profile matches the LEARNED PROFILE of hits (once "
            "feedback exists).\n"
            "- Before any feedback, prefer candidates well-connected to the task anchors "
            "(high ppi_task / coexpr_task / pathway_task).\n"
            "- Do not repeat already-selected identifiers.\n\n"
            f"Return ONLY a JSON array of {self.batch_size} identifiers. No explanation.\n"
            'Example: ["GENE_00042", "GENE_00137"]'
        )

    # ── Prompt builders (override parent to inject features, drop ML score) ──────

    def _build_anon_prompt(self, round_idx: int) -> str:
        """Weighted / open route: no shortlist is passed, so draw the pool from the static ranking."""
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")
        memory_section = self._build_memory_section()
        profile = self._hit_profile_section()
        # Open route: no shortlist exists, so the pool must be capped (18k rows are unreadable).
        candidates = self._candidate_block(
            self._static_ranking_anon, limit=self._feature_shortlist_size
        )
        return f"""You are selecting genes for a perturbation experiment, identified only by anonymous IDs.

TASK: {task_text}
MEASUREMENT: {measurement}
{memory_section}
Round {round_idx + 1} of a sequential screen.{profile}
{candidates}
{self._instructions()}"""

    def _build_anon_shortlist_prompt(
        self, round_idx: int, fake_shortlist: list[tuple[str, float]]
    ) -> str:
        """Two-stage route: the online arm supplies the candidate pool; drop its scores, show features."""
        task_text = self._task.get("Task", self.dataset_name)
        measurement = self._task.get("Measurement", "")
        memory_section = self._build_memory_section()
        profile = self._hit_profile_section()
        # Two-stage: use the arm's FULL shortlist, exactly the pool A0/A2 see. Never truncate here.
        candidates = self._candidate_block([fg for fg, _ in fake_shortlist], limit=None)
        return f"""You are selecting genes for a perturbation experiment, identified only by anonymous IDs.

TASK: {task_text}
MEASUREMENT: {measurement}
{memory_section}
Round {round_idx + 1} of a sequential screen.{profile}
{candidates}
{self._instructions()}"""


class _AnonymousPoolOnlyLLMArm(_AnonymousFeaturesLLMArm):
    """A0.5 control: the SAME anonymized, ML-curated candidate pool A1 sees, but with the feature
    values stripped out — only the bare IDs.

    Without this control A1 - A0 is uninterpretable. A0's open prompt lists no candidates at all (the
    LLM must invent IDs out of an 18k space), whereas A1 is handed the ML top-N. On the small screens
    (K562-Essential: 623 genes, Scharenberg22: 1029) that pool is itself a strong ML prior, so A1
    could beat A0 without reasoning over a single feature. Holding the pool fixed isolates the
    features:  A0.5 - A0 = the pool;  A1 - A0.5 = the features.
    """

    def _feat_line(self, anon: str) -> str:
        return f"  {anon}"

    def _hit_profile_section(self) -> str:
        # No feature values => no feature profile. Report only the count, mirroring A1's framing.
        all_hits = [g for rnd in self._round_hits for g in rnd]
        all_non = [g for rnd in self._round_nonhits for g in rnd]
        if not all_hits:
            return ""
        return f"\n\nFEEDBACK SO FAR: {len(all_hits)} hits, {len(all_non)} non-hits."

    def _instructions(self) -> str:
        return (
            f"\nSelect exactly {self.batch_size} identifiers from the pool above.\n"
            "You do NOT know the gene names and have no other information about them.\n"
            "- Do not repeat already-selected identifiers.\n\n"
            f"Return ONLY a JSON array of {self.batch_size} identifiers. No explanation.\n"
            'Example: ["GENE_00042", "GENE_00137"]'
        )


class WaddingtonCFeatureReasoningArm(WaddingtonCShuffledNamesArm):
    """A1: the anonymized C-arm, but the LLM reasons over structural feature vectors instead of names.

    Identical routing / fusion / update to A0 (WaddingtonCShuffledNamesArm); only the inner LLM's view
    changes, via _INNER_LLM_CLS.
    """

    _INNER_LLM_CLS = _AnonymousFeaturesLLMArm

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__(
            dataset_name,
            batch_size,
            memory_path=memory_path,
            name="waddington_c_feature_reasoning",
        )


class WaddingtonCPoolOnlyArm(WaddingtonCShuffledNamesArm):
    """A0.5: A1's anonymized ML-curated candidate pool, WITHOUT the features. The control that makes
    A1 - A0.5 a measurement of the features rather than of the pool."""

    _INNER_LLM_CLS = _AnonymousPoolOnlyLLMArm

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        memory_path: Path = MEMORY_PATH,
    ) -> None:
        super().__init__(
            dataset_name,
            batch_size,
            memory_path=memory_path,
            name="waddington_c_pool_only",
        )
