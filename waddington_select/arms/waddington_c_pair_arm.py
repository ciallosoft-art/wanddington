"""
WaddingtonCPairArm — the C-arm's fusion, repaired for the combinatorial tier (paper: "Diagnosing and
repairing the combinatorial failure"; every number below is recomputed by analysis/pair_tier.py).

Why a separate arm
------------------
The reported C-arm fuses a single global weight, Score(g) = 0.80*ml(g) + 0.20*1[g in named]. That is
right where it was calibrated: on single-gene screens the ML branch is the stronger source. On Horlbeck
its prior is at chance (58 hits in 640 picks, random 60) and the ordering inverts. In the C-arm's
twenty-replicate runs at 128 pairs per round, LLM-named picks hit at 26% and the other picks at 15%
(29% and 16% at 32 per round), yet only 23% of the batch is LLM-named (28%), although the LLM names
close to a full batch of valid pairs each round: a 0.20 bonus cannot lift a pair the ML branch scores
low. This arm changes how the sources are combined, not the sources. Each change is a switch:

  named_first   WADDINGTON_PAIR_NAMED_FIRST, default on. The LLM's names go first, ordered among
                themselves by the fill score, and the fill score ranks the rest of the batch. The same
                named-only vote, expressed as an ordering, so a branch at chance cannot outvote it.
  prior_fill    WADDINGTON_PAIR_PRIOR_FILL, default on. The fill score is 0.5 * online ML score +
                0.5 * the label-free interaction prior (interaction_prior below) instead of the ML score
                alone. Ranked alone, the prior puts 182 hits in the top 640 with the seven original
                pair columns and 196 with the eighth (complex co-membership), against 117 for the online
                model.
  cold_start    WADDINGTON_PAIR_COLD_START, default off. Round 1 is the prior's own top B and named-first
                starts in round 2. Before any feedback the LLM is the weaker source: in the named-first
                runs its round-1 picks hit at 23%, where the prior's top 128 hit at 29% (seven columns).
  complex_first WADDINGTON_PAIR_COMPLEX_FIRST, default off. The 445 pairs whose genes share an EBI
                Complex Portal complex (34% hit rate, the most precise single signal) go ahead of
                everything else. Tested and rejected: at 640 picks it costs 11 hits against having the
                column in the table (209.9 vs 220.9), because what it displaces is LLM proposals and pairs
                the online model has learned to rank. At 160 picks it scores 70.2 against 68.5.
  explore_frac  WADDINGTON_PAIR_EXPLORE_FRAC, default 0. A share of each batch spent on coverage. Also
                rejected: 12.5% and 25% cost 6 and 18 hits at 640 picks.

The repair the paper reports is named_first + prior_fill + cold_start on the eight-column table
(experiments/13_tier_b_fusion_ablation.sh, run "coldstart_prior_v8"). The reported C-arm is untouched.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .waddington_c_arm import WaddingtonCArm, _get_feature_config, _named_only_llm_vote
from ..features import load_training_frame
from ..tables import INTERACTION_PREFIX, split_pair


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def interaction_prior(sub: pd.DataFrame) -> dict[str, float]:
    """Equal-weight rank-sum of one screen's pair_* columns, in [0, 1]. Label-free: no outcome is read.

    `sub` is that screen's rows, with pair tokens in "gene"."""
    cols = [c for c in sub.columns if c.startswith(INTERACTION_PREFIX)]
    if not cols:
        return {}
    z = np.zeros(len(sub), dtype=float)
    for c in cols:
        v = sub[c].to_numpy(dtype=float)
        # co-essentiality is informative in BOTH directions (synergy and buffering), so rank |r|
        if c.endswith("coessential"):
            v = np.abs(v)
        z += pd.Series(v).rank(pct=True).to_numpy()
    z /= max(len(cols), 1)
    return dict(zip(sub["gene"].tolist(), z.tolist()))


def pair_selection_config() -> dict:
    """Effective pair-arm settings, shared with provenance/resume validation."""
    return {
        "named_first": _flag("WADDINGTON_PAIR_NAMED_FIRST"),
        "prior_fill": _flag("WADDINGTON_PAIR_PRIOR_FILL"),
        "cold_start": _flag("WADDINGTON_PAIR_COLD_START", "0"),
        "complex_first": _flag("WADDINGTON_PAIR_COMPLEX_FIRST", "0"),
        "explore_frac": float(os.environ.get("WADDINGTON_PAIR_EXPLORE_FRAC", "0")),
    }


class WaddingtonCPairArm(WaddingtonCArm):
    """C-arm with named-first ordering and an interaction-prior filler."""

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        kwargs.setdefault("name", "waddington_c_pair")
        super().__init__(dataset_name, batch_size, **kwargs)
        config = pair_selection_config()
        self._named_first = config["named_first"]
        self._prior_fill = config["prior_fill"]
        self._cold_start = config["cold_start"]
        self._complex_first = config["complex_first"]
        self._explore_frac = config["explore_frac"]
        self._complex: set[str] = set()
        if self._complex_first:
            self._complex = self._load_complex_pairs(dataset_name)
        self._prior = self._build_interaction_prior(dataset_name) if self._prior_fill else {}

    def _build_interaction_prior(self, dataset_name: str) -> dict[str, float]:
        """Equal-weight rank-sum of the pair_* columns. Label-free: no target outcome is read."""
        csv, _ = _get_feature_config(dataset_name)
        df = load_training_frame(csv, dataset_name)
        df["gene"] = df["gene"].str.strip().str.upper()
        return interaction_prior(df[df["dataset"] == dataset_name])

    def _load_complex_pairs(self, dataset_name: str) -> set[str]:
        """Candidates whose two genes share a curated complex. Label-free."""
        csv, _ = _get_feature_config(dataset_name)
        df = load_training_frame(csv, dataset_name)
        df["gene"] = df["gene"].str.strip().str.upper()
        sub = df[df["dataset"] == dataset_name]
        col = "pair_complex_shared"
        if col not in sub.columns:
            return set()
        return set(sub.loc[sub[col] > 0, "gene"].tolist())

    def _explore_picks(self, pool: list[str], exclude: set[str], n: int,
                       score: dict[str, float] | None = None) -> list[str]:
        """n pairs chosen for coverage: the best-scoring pair containing the least-sampled gene.

        Not random exploration -- the prior still orders the candidates, coverage only decides which
        gene to look at next. Selection is otherwise very redundant: round 1 touches 47 of the 448 genes,
        five rounds reach 61%, and the most-sampled gene appears 31 times against 2.9 under uniform
        sampling, so the online model sees a narrow slice of the feature space.
        """
        from collections import Counter
        counts: Counter = Counter()
        for tok in self._selected:
            for g in split_pair(tok):
                counts[g] += 1
        rank = score if score is not None else self._prior
        by_gene: dict[str, list[str]] = {}
        for tok in pool:
            if tok in exclude:
                continue
            for g in split_pair(tok):
                by_gene.setdefault(g, []).append(tok)
        for lst in by_gene.values():
            lst.sort(key=lambda t: rank.get(t, 0.0), reverse=True)
        picked: list[str] = []
        taken: set[str] = set()
        while len(picked) < n and by_gene:
            gene = min(by_gene, key=lambda g: (counts[g], -len(by_gene[g])))
            lst = by_gene[gene]
            while lst and lst[0] in taken:
                lst.pop(0)
            if not lst:
                del by_gene[gene]
                continue
            tok = lst.pop(0)
            picked.append(tok)
            taken.add(tok)
            for g in split_pair(tok):
                counts[g] += 1
        return picked

    def _llm_vote(self, round_idx: int, revealed: dict[str, bool]) -> set[str]:
        """The pairs the LLM branch votes for this round: its own names under the named-only policy."""
        padded = set(self._llm.select(round_idx, revealed))
        named = set(getattr(self._llm, "_llm_named", padded))
        self._last_llm_named_proposals = named
        self._last_llm_static_padding_proposals = set(getattr(self._llm, "_static_padding", []))
        return named if _named_only_llm_vote() else padded

    def _select_weighted(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        ml_scores = self._online.all_scores()
        llm_set = self._llm_vote(round_idx, revealed)

        pool = [g for g in self._online._genes if g not in self._selected]
        n_explore = int(round(self.batch_size * self._explore_frac))

        def _take(order: list[str], batch: list[str]) -> list[str]:
            """Complex-sharing candidates first, then the given order."""
            if not self._complex_first or not self._complex:
                return (batch + order)[: self.batch_size]
            seen = set(batch)
            head = [g for g in order if g in self._complex and g not in seen]
            tail = [g for g in order if g not in self._complex and g not in seen]
            return (batch + head + tail)[: self.batch_size]

        if self._cold_start and round_idx == 0 and self._prior:
            # Cold start: no feedback exists yet, so neither the LLM nor the online model has anything
            # this screen has taught it. Rank on the one source that is informative from the outset.
            order = sorted(pool, key=lambda g: self._prior.get(g, 0.0), reverse=True)
            batch = _take(order, [])
            if n_explore:
                keep = batch[: self.batch_size - n_explore]
                batch = keep + self._explore_picks(pool, set(keep), n_explore)
            return batch[: self.batch_size]

        def fill_score(g: str) -> float:
            ml = ml_scores.get(g, 0.0)
            if not self._prior_fill or not self._prior:
                return ml
            # Both terms are in [0, 1]; an unweighted mean keeps this untuned on the target screen.
            return 0.5 * ml + 0.5 * self._prior.get(g, 0.0)

        if self._named_first:
            # With complex_first on, complex-sharing pairs outrank even the LLM's names (34.4% hit rate
            # against about 31% for LLM-named picks at 128 per round). That switch was rejected; see the
            # module docstring.
            named = sorted((g for g in pool if g in llm_set), key=fill_score, reverse=True)
            rest = sorted((g for g in pool if g not in llm_set), key=fill_score, reverse=True)
            batch = _take(named + rest, [])
            if n_explore:
                keep = batch[: self.batch_size - n_explore]
                scores = {g: fill_score(g) for g in pool}
                batch = keep + self._explore_picks(pool, set(keep), n_explore, scores)
        else:
            combined = {g: self._w_ml * fill_score(g) + self._w_llm * (1.0 if g in llm_set else 0.0)
                        for g in pool}
            batch = sorted(combined, key=combined.__getitem__, reverse=True)[: self.batch_size]
        return batch


class WaddingtonCPairNoLLMArm(WaddingtonCPairArm):
    """The repaired pair arm with its LLM removed: the same prior, cold start, online ML and fill score,
    but no named pairs, so the fill score orders every slot. Against the full arm it isolates what the
    LLM adds on top of the label-free prior and the online model. Makes no LLM call and is deterministic,
    so one replicate is the whole answer."""

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        kwargs.setdefault("name", "waddington_c_pair_no_llm")
        super().__init__(dataset_name, batch_size, **kwargs)

    def _llm_vote(self, round_idx: int, revealed: dict[str, bool]) -> set[str]:
        self._last_llm_named_proposals = set()
        self._last_llm_static_padding_proposals = set()
        return set()
