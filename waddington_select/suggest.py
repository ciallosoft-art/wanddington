"""
suggest.py — Forward gene recommendation (the deployment entry point).

The main runner (`python -m waddington_select`) is a *benchmark* loop that needs the oracle to
reveal hits. This module is the *forward* action an experimenter actually wants: "for this
phenotype, which genes should I perturb next?" — the C-arm's ranking with no oracle involved.

It is what feynman invokes from chat (see repo AGENTS.md).

Two modes:
- Cold start: no feedback → the C-arm's first-round recommendation.
- Feedback-driven: pass the results of what you already tested with --tested-hits / --tested-misses.
  Those labels retrain the online ML model and are fed to the LLM as experimental history, so the
  next recommendation adapts (true sequential selection, one feedback round).

Usage:
    conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG
    conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG --n 20 \
        --tested-hits ZAP70 LCK PLCG1 --tested-misses ACTB GAPDH TUBB
    # run the LLM component on codex instead of Claude:
    WADDINGTON_LLM_BACKEND=pi python3 -m waddington_select.suggest --dataset IFNG

Note (v1): --dataset must be one of the nine benchmark phenotypes, since the ML features are
precomputed per dataset. Suggesting for an arbitrary new phenotype needs the feature pipeline to
run on that gene pool first (follow-up work).
"""

from __future__ import annotations

import argparse
import json

from .oracle import ALL_DATASETS, BATCH_SIZES
from .arms.waddington_c_arm import WaddingtonCArm, WaddingtonCSkillsArm


def _norm(genes: list[str] | None) -> list[str]:
    return [g.strip().upper() for g in (genes or []) if g.strip()]


def _anchor_genes(dataset: str) -> list[str]:
    """Seed genes of a registered user phenotype (empty for the benchmark datasets)."""
    from .phenotype import load_registry
    return _norm((load_registry().get(dataset) or {}).get("anchors"))


def suggest(
    dataset: str,
    n: int | None = None,
    exclude: list[str] | None = None,
    use_skills: bool = False,
    tested_hits: list[str] | None = None,
    tested_misses: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Recommend the next batch of genes to perturb for `dataset`.

    Returns (recommended_genes, info) where info reports how feedback was incorporated and which
    supplied gene symbols were not recognised in this phenotype's pool.
    """
    if dataset not in ALL_DATASETS:
        raise SystemExit(f"Unknown dataset '{dataset}'. Available: {', '.join(ALL_DATASETS)}")

    batch_size = n or BATCH_SIZES[dataset]
    arm = (WaddingtonCSkillsArm if use_skills else WaddingtonCArm)(dataset, batch_size)
    pool: set[str] = arm._llm._gene_set  # canonical gene pool for this phenotype

    hits = _norm(tested_hits)
    misses = _norm(tested_misses)
    excluded = _norm(exclude)

    # A registered phenotype's anchor genes are the scientist's own seed biology: they score top by
    # construction (an anchor's PPI/co-expression similarity to itself is maximal), and recommending
    # them back would be circular — the scientist already knows them. Never propose them.
    anchors = _anchor_genes(dataset)
    if anchors:
        excluded = sorted(set(excluded) | set(anchors))
    unknown = sorted({g for g in hits + misses + excluded if g not in pool})

    round_idx = 0
    revealed = {g: True for g in hits if g in pool}
    revealed.update({g: False for g in misses if g in pool})
    if revealed:
        # One feedback round: retrain ML + feed LLM history, then recommend the next round.
        arm.update(round_idx=0, revealed_new=revealed)
        round_idx = 1
    if excluded:
        arm._selected.update(g for g in excluded if g in pool)

    genes = arm.select(round_idx=round_idx, revealed={})
    info = {
        "round": round_idx + 1,
        "n_feedback": len(revealed),
        "n_hits": sum(revealed.values()),
        "unknown_genes": unknown,
        "route": getattr(arm, "_route", "?"),
    }
    return genes, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend genes to perturb next for a phenotype (System 2 C-arm).",
    )
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                        help="Benchmark phenotype to recommend genes for")
    parser.add_argument("--n", type=int, default=None,
                        help="How many genes to recommend (default: dataset batch size)")
    parser.add_argument("--tested-hits", nargs="*", default=None,
                        help="Already-tested genes that WERE hits (used as positive feedback)")
    parser.add_argument("--tested-misses", nargs="*", default=None,
                        help="Already-tested genes that were NOT hits (negative feedback)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Genes to exclude from recommendations (outcome unknown/irrelevant)")
    parser.add_argument("--skills", action="store_true",
                        help="Use the skill library instead of flat cross-experiment memory")
    parser.add_argument("--json", action="store_true",
                        help="Emit a single JSON object {dataset, genes, info} instead of human text "
                             "(machine contract for the conversational frontend)")
    args = parser.parse_args()

    genes, info = suggest(
        args.dataset, args.n, args.exclude, use_skills=args.skills,
        tested_hits=args.tested_hits, tested_misses=args.tested_misses,
    )

    if args.json:
        print(json.dumps({"dataset": args.dataset, "genes": genes, "info": info}))
        return

    if info["unknown_genes"]:
        print(f"\n[warn] not in '{args.dataset}' gene pool, ignored: "
              f"{', '.join(info['unknown_genes'])}")
    if info["n_feedback"]:
        print(f"\nIncorporated {info['n_feedback']} tested genes "
              f"({info['n_hits']} hits) as feedback → recommending round {info['round']}.")

    print(f"\nRecommended genes to perturb next for '{args.dataset}' "
          f"({len(genes)} genes, C-arm{' + skills' if args.skills else ''}, route={info['route']}):\n")
    for i, g in enumerate(genes, 1):
        print(f"  {i:3d}. {g}")
    print()


if __name__ == "__main__":
    main()
