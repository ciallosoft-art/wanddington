"""
trace.py — record WHY each gene was chosen, so a run can be explained.

The C-arm fuses an online ML ranking with an LLM's picks. The interesting question is not "did the LLM
speak" but **"what did the LLM add that the ML would not have chosen anyway"** — so attribution is a
counterfactual against the ML's own top-k:

    both      — the LLM named it AND the ML would have picked it anyway
    llm_only  — the LLM named it and the ML would NOT have picked it   ← the LLM's marginal contribution
    ml_only   — the LLM didn't name it; it came in on ML score alone

Pair that with the revealed outcome and you get the hit rate per source, which answers the question
the ablations can only answer in aggregate: *does the LLM find hits the ML misses?*
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceRecorder:
    """Collects one record per round; attach to a WaddingtonCArm via `trace=`."""

    rounds: list[dict] = field(default_factory=list)

    def record_selection(
        self,
        round_idx: int,
        route: str,
        w_ml: float,
        w_llm: float,
        ml_scores: dict[str, float],
        llm_set: set[str],
        batch: list[str],
        ml_candidates: list[str] | None = None,
        shap: dict[str, float] | None = None,
        n_fallback: int = 0,
    ) -> None:
        """`llm_set` must be the genes the LLM actually NAMED (not the static-ranker back-fill);
        `ml_candidates` = the genes the ML would have taken on its own (defaults to its top-k)."""
        k = len(batch)
        if ml_candidates is None:
            ranked = sorted(ml_scores, key=lambda g: -ml_scores.get(g, 0.0))
            ml_candidates = ranked[:k]
        ml_topk = set(ml_candidates[:k])

        genes = []
        for g in batch:
            in_llm = g in llm_set
            in_ml = g in ml_topk
            source = "both" if (in_llm and in_ml) else ("llm_only" if in_llm else "ml_only")
            genes.append({
                "gene": g,
                "source": source,
                "ml_score": round(float(ml_scores.get(g, 0.0)), 5),
            })

        self.rounds.append({
            "round": round_idx + 1,
            "route": route,
            "w_ml": w_ml,
            "w_llm": w_llm,
            "batch_size": k,
            "n_llm_named": len(llm_set),
            "n_fallback": n_fallback,
            "genes": genes,
            "shap": {f: round(v, 5) for f, v in sorted((shap or {}).items(), key=lambda kv: -kv[1])},
            "counts": {s: sum(1 for x in genes if x["source"] == s) for s in ("both", "ml_only", "llm_only")},
        })

    def record_outcome(self, round_idx: int, revealed: dict[str, bool]) -> None:
        """Mark which of the round's picks turned out to be hits."""
        for r in self.rounds:
            if r["round"] == round_idx + 1:
                for g in r["genes"]:
                    if g["gene"] in revealed:
                        g["hit"] = bool(revealed[g["gene"]])
                r["n_hits"] = sum(1 for g in r["genes"] if g.get("hit"))
                return

    def to_dict(self) -> dict:
        return {"rounds": self.rounds}


def hit_rate_by_source(trace: dict) -> dict[str, dict]:
    """Across all traced rounds: picks and hits per attribution source."""
    agg: dict[str, dict] = {s: {"picked": 0, "hits": 0} for s in ("both", "ml_only", "llm_only")}
    for r in trace.get("rounds", []):
        for g in r["genes"]:
            s = g["source"]
            agg[s]["picked"] += 1
            if g.get("hit"):
                agg[s]["hits"] += 1
    for s, v in agg.items():
        v["hit_rate"] = (v["hits"] / v["picked"]) if v["picked"] else 0.0
    return agg


def traced_campaign(dataset: str, rounds: int = 5, batch_size: int | None = None) -> dict:
    """Run a full oracle-driven campaign with tracing on — decisions AND outcomes in one artifact."""
    from ..oracle import BATCH_SIZES, DatasetOracle
    from ..arms.waddington_c_arm import WaddingtonCArm

    oracle = DatasetOracle(dataset)
    bs = batch_size or BATCH_SIZES[dataset]
    rec = TraceRecorder()
    arm = WaddingtonCArm(dataset, bs, trace=rec)

    history = []
    cumulative = 0
    for r in range(rounds):
        batch = arm.select(round_idx=r, revealed={})
        revealed = oracle.reveal(batch)
        rec.record_outcome(r, revealed)
        arm.update(round_idx=r, revealed_new=revealed)
        hits = [g for g, v in revealed.items() if v]
        cumulative += len(hits)
        history.append({
            "round": r + 1, "tested": len(batch), "hits": hits,
            "cumulative": cumulative,
            "ratio": cumulative / oracle.total_hits if oracle.total_hits else 0.0,
        })

    return {
        "dataset": dataset,
        "rounds_run": rounds,
        "batch_size": bs,
        "total_hits": oracle.total_hits,
        "n_genes": oracle.n_genes,
        "cumulative_hits": cumulative,
        "hit_ratio": round(cumulative / oracle.total_hits, 4) if oracle.total_hits else 0.0,
        "history": history,
        "trace": rec.to_dict(),
    }
