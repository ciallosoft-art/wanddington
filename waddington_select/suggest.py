"""
suggest.py — Forward gene recommendation (the deployment entry point).

The main runner (`python -m waddington_select`) is a *benchmark* loop that needs the oracle to
reveal hits. This module is the *forward* action an experimenter actually wants: "for this
phenotype, which genes should I perturb next?" — the C-arm's ranking with no oracle involved.

It is what the Waddington frontend invokes from chat (see repo AGENTS.md).

Two modes:
- Cold start: no feedback → the C-arm's first-round recommendation.
- Feedback-driven: pass the results of what you already tested with --tested-hits / --tested-misses.
  Those labels retrain the online ML model and are fed to the LLM as experimental history, so the
  next recommendation adapts (true sequential selection, one feedback round).

Usage:
    conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG
    conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG --n 20 \
        --tested-hits ZAP70 LCK PLCG1 --tested-misses ACTB GAPDH TUBB
    # GPT-5.6 Luna/none through Codex OAuth is the default:
    python3 -m waddington_select.suggest --dataset IFNG

--dataset is one of the nine benchmark phenotypes or a phenotype registered with
`python -m waddington_select.phenotype register` (which builds its features). A registered phenotype's
single-gene tier (non_essential by default, or full_pool) decides the training tables and which reported
run's cross-experiment memory is read (deployment_memory_path). Single-gene only: a two-gene (pair) screen
is refused (the reported system loses to BioDiscoveryAgent on that tier).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .oracle import ALL_DATASETS, BATCH_SIZES
from .tables import is_pair_dataset
from .arms.waddington_c_arm import WaddingtonCArm, WaddingtonCSkillsArm


def _norm(genes: list[str] | None) -> list[str]:
    return [g.strip().upper() for g in (genes or []) if g.strip()]


def _anchor_genes(dataset: str) -> list[str]:
    """Seed genes of a registered user phenotype (empty for the benchmark datasets)."""
    from .phenotype import load_registry
    return _norm((load_registry().get(dataset) or {}).get("anchors"))


def deployment_memory_path(dataset: str) -> Path:
    """The cross-experiment memory a recommendation reads: $WADDINGTON_MEMORY_PATH if set, else the memory
    of the reported run for the phenotype's tier (analysis/sources.py) -- the full-pool run's for a phenotype
    registered in the full-pool tier, the headline run's otherwise. (config.ACTIVE_MEMORY_PATH is where the
    memory builder writes, not a calibrated memory, and is absent from a fresh clone: recommendations used to
    run without memory.)"""
    env = os.environ.get("WADDINGTON_MEMORY_PATH")
    if env:
        return Path(env)
    from .analysis.sources import REPORTED_RUN_DIR, REPORTED_TIER_A_RUN
    from .phenotype import phenotype_tier
    from .tables import TIER_FULL_POOL
    if phenotype_tier(dataset) == TIER_FULL_POOL:
        return REPORTED_TIER_A_RUN.parent / "experience_memory.json"
    return REPORTED_RUN_DIR / "experience_memory.json"


def suggest(
    dataset: str,
    n: int | None = None,
    exclude: list[str] | None = None,
    use_skills: bool = False,
    tested_hits: list[str] | None = None,
    tested_misses: list[str] | None = None,
    trace: bool = False,
    reason_from_structure: bool | None = None,
) -> tuple[list[str], dict]:
    """Recommend the next batch of genes to perturb for `dataset`.

    Returns (recommended_genes, info) where info reports how feedback was incorporated and which
    supplied gene symbols were not recognised in this phenotype's pool.
    """
    if is_pair_dataset(dataset):
        raise SystemExit(f"'{dataset}' is a two-gene (combinatorial) screen; recommendations are single-gene only. "
                         "On the two-gene tier the reported system loses to BioDiscoveryAgent, so that tier is a "
                         "benchmark (experiments/), not offered for deployment.")
    if dataset not in ALL_DATASETS:
        raise SystemExit(f"Unknown dataset '{dataset}'. Available: {', '.join(ALL_DATASETS)}")

    batch_size = n or BATCH_SIZES[dataset]
    recorder = None
    if trace and not use_skills:  # the skills arm (a negative result) takes no recorder
        from .analysis.trace import TraceRecorder
        recorder = TraceRecorder()
    if use_skills:
        arm = WaddingtonCSkillsArm(dataset, batch_size)
    else:
        arm = WaddingtonCArm(dataset, batch_size, trace=recorder, reason_from_structure=reason_from_structure,
                             memory_path=deployment_memory_path(dataset))
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
    if recorder is not None and recorder.rounds:
        # one selection per suggest() call; the caller renumbers it to its own campaign round
        info_trace = recorder.rounds[-1]
    else:
        info_trace = None
    info = {
        "round": round_idx + 1,
        "n_feedback": len(revealed),
        "n_hits": sum(revealed.values()),
        "unknown_genes": unknown,
        "route": getattr(arm, "_route", "?"),
        "reason_from_structure": getattr(arm, "_reason_from_structure", False),
        "memory_entries": len(getattr(arm._llm, "_memory", []) or []),
    }
    if info_trace is not None:
        info["trace"] = info_trace
    return genes, info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend genes to perturb next for a phenotype (Waddington C-arm).",
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
    parser.add_argument("--trace", action="store_true",
                        help="record WHY each gene was chosen (attribution + SHAP) into info.trace")
    parser.add_argument("--reason-from-structure", action="store_true", default=None,
                        help="LLM reasons over anonymized structural features instead of gene names "
                             "(opt-in; removes the dependence on recognising gene names, at a cost: in the "
                             "reported configuration it is no better than the ML engine alone). Defaults to "
                             "$WADDINGTON_REASON_FROM_STRUCTURE.")
    parser.add_argument("--json", action="store_true",
                        help="Emit a single JSON object {dataset, genes, info} instead of human text "
                             "(machine contract for the conversational frontend)")
    args = parser.parse_args()

    genes, info = suggest(
        args.dataset, args.n, args.exclude, use_skills=args.skills,
        tested_hits=args.tested_hits, tested_misses=args.tested_misses,
        trace=args.trace, reason_from_structure=args.reason_from_structure,
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

    mode = " · reasoning from structure (names hidden)" if info.get("reason_from_structure") else ""
    print(f"\nRecommended genes to perturb next for '{args.dataset}' "
          f"({len(genes)} genes, C-arm{' + skills' if args.skills else ''}, route={info['route']}"
          f"{mode}):\n")
    for i, g in enumerate(genes, 1):
        print(f"  {i:3d}. {g}")
    print()


if __name__ == "__main__":
    main()
