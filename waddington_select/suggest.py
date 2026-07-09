"""
suggest.py — Forward gene recommendation (the deployment entry point).

The main runner (`python -m waddington_select`) is a *benchmark* loop that needs the oracle to
reveal hits. This module is the *forward* action an experimenter actually wants: "for this
phenotype, which genes should I perturb next?" — the C-arm's ranking with no oracle involved.

It is what feynman invokes from chat (see repo AGENTS.md).

Usage:
    conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG
    conda run -n waddington-bio python3 -m waddington_select.suggest --dataset IFNG --n 20 \
        --exclude ZAP70 LCK
    # run the LLM component on codex instead of Claude:
    WADDINGTON_LLM_BACKEND=pi python3 -m waddington_select.suggest --dataset IFNG

Note (v1): --dataset must be one of the nine benchmark phenotypes, since the ML features are
precomputed per dataset. Suggesting for an arbitrary new phenotype needs the feature pipeline to
run on that gene pool first (follow-up work).
"""

from __future__ import annotations

import argparse

from .oracle import ALL_DATASETS, BATCH_SIZES
from .arms.waddington_c_arm import WaddingtonCArm, WaddingtonCSkillsArm


def suggest(
    dataset: str,
    n: int | None = None,
    exclude: list[str] | None = None,
    use_skills: bool = False,
) -> list[str]:
    """Return the C-arm's recommended first batch of genes to perturb for `dataset`."""
    if dataset not in ALL_DATASETS:
        raise SystemExit(
            f"Unknown dataset '{dataset}'. Available: {', '.join(ALL_DATASETS)}"
        )
    batch_size = n or BATCH_SIZES[dataset]
    arm = (WaddingtonCSkillsArm if use_skills else WaddingtonCArm)(dataset, batch_size)
    if exclude:
        # Treat already-tested genes as selected so they are not recommended again.
        arm._selected.update(g.strip().upper() for g in exclude)
    return arm.select(round_idx=0, revealed={})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend genes to perturb next for a phenotype (System 2 C-arm).",
    )
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                        help="Benchmark phenotype to recommend genes for")
    parser.add_argument("--n", type=int, default=None,
                        help="How many genes to recommend (default: dataset batch size)")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Genes already tested — excluded from recommendations")
    parser.add_argument("--skills", action="store_true",
                        help="Use the skill library instead of flat cross-experiment memory")
    args = parser.parse_args()

    genes = suggest(args.dataset, args.n, args.exclude, use_skills=args.skills)

    print(f"\nRecommended genes to perturb next for '{args.dataset}' "
          f"({len(genes)} genes, C-arm{' + skills' if args.skills else ''}):\n")
    for i, g in enumerate(genes, 1):
        print(f"  {i:3d}. {g}")
    print()


if __name__ == "__main__":
    main()
