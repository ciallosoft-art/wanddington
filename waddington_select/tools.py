"""
tools.py — Pure-compute actions the feynman gene-selection agent calls (via its bash tool).

The agent's *reasoning* is the LLM (feynman's brain, provider-switchable); its *actions* that touch
data are these tools. Each subcommand prints a single JSON object to stdout so the agent can parse
it. No tool calls the LLM — the intelligence/planning lives in the agent loop.

Subcommands:
  ml_rank   ML (online LightGBM) top candidate genes, given feedback so far.
  enrich    Pathway/GO enrichment of a gene set (Enrichr) — turns revealed hits into pathways.
  reveal    "Run the experiment": return hit/no-hit for a batch. Benchmark uses the ground-truth
            oracle; in real deployment this is replaced by the scientist entering wet-lab results.

Usage:
  python -m waddington_select.tools ml_rank --dataset IFNG --n 30 --tested-hits ZAP70 --tested-misses ACTB
  python -m waddington_select.tools enrich  --genes ZAP70 LCK PLCG1 --top 10
  python -m waddington_select.tools reveal  --dataset IFNG --genes ZAP70 LCK ACTB
"""

from __future__ import annotations

import argparse
import json
import sys

from .oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from .arms.online_adaptive_arm import OnlineAdaptiveArm
from .arms.waddington_c_arm import _get_feature_config


def _emit(obj: dict) -> None:
    json.dump(obj, sys.stdout, indent=2)
    sys.stdout.write("\n")


def _norm(genes) -> list[str]:
    return [g.strip().upper() for g in (genes or []) if g.strip()]


# ── ml_rank ──────────────────────────────────────────────────────────────────

def cmd_ml_rank(args) -> None:
    training_csv, extra = _get_feature_config(args.dataset)
    n = args.n or BATCH_SIZES[args.dataset]
    arm = OnlineAdaptiveArm(args.dataset, n, training_csv=training_csv, extra_feature_cols=extra)

    hits = _norm(args.tested_hits)
    misses = _norm(args.tested_misses)
    revealed = {g: True for g in hits}
    revealed.update({g: False for g in misses})
    retrained = False
    if revealed:
        arm.update(0, revealed)
        retrained = True

    exclude = set(hits) | set(misses) | set(_norm(args.exclude))
    ranked = arm.ranked_candidates(n, exclude=exclude)
    _emit({
        "tool": "ml_rank",
        "dataset": args.dataset,
        "retrained_on_feedback": retrained,
        "candidates": [{"gene": g, "score": round(float(s), 4)} for g, s in ranked],
    })


# ── enrich ───────────────────────────────────────────────────────────────────

def cmd_enrich(args) -> None:
    from pathlib import Path
    genes = _norm(args.genes)
    result = {"tool": "enrich", "n_genes": len(genes), "terms": []}
    if not genes:
        result["error"] = "no genes provided"
        _emit(result)
        return
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace" / "evaluation"))
        from go_enrichment import enrich_genes
        raw = enrich_genes(genes)  # list of {rank, term, pval, z_score, combined_score, genes}
        result["terms"] = [
            {
                "term": row["term"],
                "pval": round(float(row["pval"]), 6),
                "combined_score": round(float(row["combined_score"]), 2),
                "overlap_genes": row.get("genes", []),
            }
            for row in raw[: args.top]
        ]
    except Exception as e:  # network / API hiccup — degrade gracefully
        result["error"] = f"enrichment unavailable: {type(e).__name__}: {e}"
    _emit(result)


# ── reveal (oracle / wet-lab stand-in) ───────────────────────────────────────

def cmd_reveal(args) -> None:
    genes = _norm(args.genes)
    oracle = DatasetOracle(args.dataset)
    revealed = oracle.reveal(genes)
    hits = [g for g, is_hit in revealed.items() if is_hit]
    _emit({
        "tool": "reveal",
        "dataset": args.dataset,
        "tested": len(genes),
        "n_hits": len(hits),
        "hits": hits,
        "reveal": revealed,
        "note": "ground-truth oracle (benchmark). In deployment, replace with real wet-lab results.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Gene-selection agent tools (pure compute).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ml_rank", help="ML top candidate genes given feedback")
    p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--tested-hits", nargs="*", default=None)
    p.add_argument("--tested-misses", nargs="*", default=None)
    p.add_argument("--exclude", nargs="*", default=None)
    p.set_defaults(func=cmd_ml_rank)

    p = sub.add_parser("enrich", help="Pathway/GO enrichment of a gene set")
    p.add_argument("--genes", nargs="+", required=True)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_enrich)

    p = sub.add_parser("reveal", help="Return hit/no-hit for a batch (benchmark oracle)")
    p.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    p.add_argument("--genes", nargs="+", required=True)
    p.set_defaults(func=cmd_reveal)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
