"""
WaddingtonCPairArm — the C-arm's fusion, repaired for the combinatorial tier.

Why a separate arm
------------------
The reported C-arm fuses a single global weight, Score(g) = 0.80*ml(g) + 0.20*1[g in named]. That is
correct where it was calibrated: on single-gene screens the ML branch is the stronger source
(LOO prior 0.217 hit@R5), so letting it rank and using the LLM as a light endorsement is an upgrade.

On Horlbeck the ordering inverts and the same arithmetic becomes the failure. Measured on the
twenty-replicate runs:

  per-pick hit rate, LLM-named selections      0.329   (3.32x random -- identical to BDA's agent)
  per-pick hit rate, ML-driven selections      0.162   (1.64x random)
  share of the batch that was LLM-named          24%

A 0.20 bonus cannot lift a pair the ML branch scores low, so three quarters of each batch is drawn
from the weaker source; the LLM names 26-30 valid pairs every round and the fusion discards most of
them. This arm changes only how the two branches are combined -- not the branches themselves:

  named_first   the LLM's names are taken FIRST (ordered among themselves by ML score), and the ML
                ranking fills only what is left. This is the same "named-only vote" semantics
                expressed as an ordering instead of an additive weight, so the LLM can no longer be
                outvoted by a branch that is at chance.
  prior_fill    the filler is ranked by the label-free interaction prior (an equal-weight rank-sum of
                the pair_* columns) blended with the online ML score, rather than by the ML score
                alone. Ranking the pool by that prior alone puts 182 hits in the top 640 against 117
                for the online model, so it is the better source for the slots the LLM cannot fill.

Both are on by default here and individually switchable, so their contributions can be attributed.
The reported C-arm is untouched.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .waddington_c_arm import WaddingtonCArm, _get_feature_config, _named_only_llm_vote
from ..features import load_training_frame
from ..tables import INTERACTION_PREFIX


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


class WaddingtonCPairArm(WaddingtonCArm):
    """C-arm with named-first ordering and an interaction-prior filler."""

    def __init__(self, dataset_name: str, batch_size: int, **kwargs) -> None:
        kwargs.setdefault("name", "waddington_c_pair")
        super().__init__(dataset_name, batch_size, **kwargs)
        self._named_first = _flag("WADDINGTON_PAIR_NAMED_FIRST")
        self._prior_fill = _flag("WADDINGTON_PAIR_PRIOR_FILL")
        self._prior = self._build_interaction_prior(dataset_name) if self._prior_fill else {}

    def _build_interaction_prior(self, dataset_name: str) -> dict[str, float]:
        """Equal-weight rank-sum of the pair_* columns. Label-free: no target outcome is read."""
        csv, _ = _get_feature_config(dataset_name)
        df = load_training_frame(csv, dataset_name)
        df["gene"] = df["gene"].str.strip().str.upper()
        sub = df[df["dataset"] == dataset_name]
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

    def _select_weighted(self, round_idx: int, revealed: dict[str, bool]) -> list[str]:
        ml_scores = self._online.all_scores()
        padded = set(self._llm.select(round_idx, revealed))
        named = set(getattr(self._llm, "_llm_named", padded))
        self._last_llm_named_proposals = named
        self._last_llm_static_padding_proposals = set(getattr(self._llm, "_static_padding", []))
        llm_set = named if _named_only_llm_vote() else padded

        pool = [g for g in self._online._genes if g not in self._selected]

        def fill_score(g: str) -> float:
            ml = ml_scores.get(g, 0.0)
            if not self._prior_fill or not self._prior:
                return ml
            # Both terms are in [0, 1]; an unweighted mean keeps this untuned on the target screen.
            return 0.5 * ml + 0.5 * self._prior.get(g, 0.0)

        if self._named_first:
            head = sorted((g for g in pool if g in llm_set), key=fill_score, reverse=True)
            batch = head[: self.batch_size]
            if len(batch) < self.batch_size:
                seen = set(batch)
                tail = sorted((g for g in pool if g not in seen), key=fill_score, reverse=True)
                batch += tail[: self.batch_size - len(batch)]
        else:
            combined = {g: self._w_ml * fill_score(g) + self._w_llm * (1.0 if g in llm_set else 0.0)
                        for g in pool}
            batch = sorted(combined, key=combined.__getitem__, reverse=True)[: self.batch_size]
        return batch
