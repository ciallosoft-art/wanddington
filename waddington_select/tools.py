"""
tools.py — Pure-compute actions for the gene-selection agent.

The agent's *reasoning* is the LLM; its *actions that touch data* are these functions. Each returns
a plain dict. They are callable natively (by the Python agent loop, agent_loop.py) and also exposed
as a CLI (`python -m waddington_select.tools <cmd> ...`) that prints the same dict as JSON — handy
for a feynman/pi-driven agent that shells out. No function calls the LLM.

  ml_rank   ML (online LightGBM) top candidate genes, given feedback so far.
  enrich    Pathway/GO enrichment of a gene set (Enrichr) — turns revealed hits into pathways.
  reveal    "Run the experiment": hit/no-hit for a batch. Benchmark = ground-truth oracle;
            real deployment = the scientist entering wet-lab results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from .arms.online_adaptive_arm import OnlineAdaptiveArm
from .arms.waddington_c_arm import _get_feature_config


def _norm(genes) -> list[str]:
    return [g.strip().upper() for g in (genes or []) if g.strip()]


# ── pure functions (callable natively) ───────────────────────────────────────

def ml_rank(dataset: str, n: int | None = None, tested_hits=None,
            tested_misses=None, exclude=None) -> dict:
    training_csv, extra = _get_feature_config(dataset)
    n = n or BATCH_SIZES[dataset]
    arm = OnlineAdaptiveArm(dataset, n, training_csv=training_csv, extra_feature_cols=extra)
    hits, misses = _norm(tested_hits), _norm(tested_misses)
    revealed = {g: True for g in hits}
    revealed.update({g: False for g in misses})
    retrained = bool(revealed)
    if retrained:
        arm.update(0, revealed)
    excl = set(hits) | set(misses) | set(_norm(exclude))
    ranked = arm.ranked_candidates(n, exclude=excl)
    return {
        "tool": "ml_rank", "dataset": dataset, "retrained_on_feedback": retrained,
        "candidates": [{"gene": g, "score": round(float(s), 4)} for g, s in ranked],
    }


def enrich(genes, top: int = 10) -> dict:
    genes = _norm(genes)
    result = {"tool": "enrich", "n_genes": len(genes), "terms": []}
    if not genes:
        result["error"] = "no genes provided"
        return result
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace" / "evaluation"))
        from go_enrichment import enrich_genes
        raw = enrich_genes(genes)  # list of {rank, term, pval, z_score, combined_score, genes}
        result["terms"] = [
            {"term": row["term"], "pval": round(float(row["pval"]), 6),
             "combined_score": round(float(row["combined_score"]), 2),
             "overlap_genes": row.get("genes", [])}
            for row in raw[:top]
        ]
    except Exception as e:
        result["error"] = f"enrichment unavailable: {type(e).__name__}: {e}"
    return result


def reveal(dataset: str, genes) -> dict:
    genes = _norm(genes)
    oracle = DatasetOracle(dataset)
    revealed = oracle.reveal(genes)
    hits = [g for g, is_hit in revealed.items() if is_hit]
    return {
        "tool": "reveal", "dataset": dataset, "tested": len(genes),
        "n_hits": len(hits), "hits": hits, "reveal": revealed,
        "total_hits": oracle.total_hits,  # dataset-wide hit count, for cumulative hit_ratio
        "note": "ground-truth oracle (benchmark). In deployment, replace with real wet-lab results.",
    }


# ── CLI wrappers ─────────────────────────────────────────────────────────────

def _emit(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2)
    sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gene-selection agent tools (pure compute).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ml_rank")
    p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--tested-hits", nargs="*", default=None)
    p.add_argument("--tested-misses", nargs="*", default=None)
    p.add_argument("--exclude", nargs="*", default=None)
    p.set_defaults(func=lambda a: _emit(ml_rank(a.dataset, a.n, a.tested_hits, a.tested_misses, a.exclude)))

    p = sub.add_parser("enrich")
    p.add_argument("--genes", nargs="+", required=True)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=lambda a: _emit(enrich(a.genes, a.top)))

    p = sub.add_parser("reveal")
    p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p.add_argument("--genes", nargs="+", required=True)
    p.set_defaults(func=lambda a: _emit(reveal(a.dataset, a.genes)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
