"""
simulate.py — Replay a full sequential CRISPR campaign, narrated round by round.

This demonstrates, on the nine benchmark phenotypes, what using System 2 for a *researcher's own
sequential experiment* would look like: each round the C-arm recommends a batch, the "wet-lab
result" is revealed, and the next round adapts to it. Here the benchmark's ground-truth oracle
plays the wet lab (that is the "simulation") — in real use the same loop is driven by the
experimenter's actual hit/no-hit results instead (see suggest.py --tested-hits/--tested-misses).

Usage:
    conda run -n waddington-bio python3 -m waddington_select.simulate --dataset IFNG --rounds 5
    conda run -n waddington-bio python3 -m waddington_select.simulate --dataset Scharenberg22 \
        --rounds 5 --skills
    WADDINGTON_LLM_BACKEND=pi python3 -m waddington_select.simulate --dataset IFNG   # run on codex
"""

from __future__ import annotations

import argparse

from .oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from .arms.waddington_c_arm import WaddingtonCArm, WaddingtonCSkillsArm, WaddingtonCMemSkillsArm

_ARMS = {
    "waddington_c": WaddingtonCArm,
    "waddington_c_skills": WaddingtonCSkillsArm,
    "waddington_c_memskills": WaddingtonCMemSkillsArm,
}


def simulate(dataset: str, rounds: int = 5, n: int | None = None,
             arm_name: str = "waddington_c", sample: int = 8) -> dict:
    """Run a narrated sequential campaign and return the trajectory."""
    if dataset not in ALL_DATASETS:
        raise SystemExit(f"Unknown dataset '{dataset}'. Available: {', '.join(ALL_DATASETS)}")

    batch_size = n or BATCH_SIZES[dataset]
    arm = _ARMS[arm_name](dataset, batch_size)
    oracle = DatasetOracle(dataset)
    arm.reset()

    print(f"\nSimulating a sequential CRISPR campaign on '{dataset}'")
    print(f"  phenotype pool: {oracle.n_genes} genes, {oracle.total_hits} true hits "
          f"({oracle.total_hits / oracle.n_genes:.1%}) | arm={arm_name} | batch={batch_size}")
    print("  (the benchmark oracle stands in for wet-lab results)\n")

    revealed: dict[str, bool] = {}
    cumulative = 0
    trajectory = []
    for r in range(rounds):
        picks = arm.select(r, revealed)
        new = oracle.reveal(picks)
        arm.update(r, new)
        revealed.update(new)

        hit_genes = [g for g, is_hit in new.items() if is_hit]
        cumulative += len(hit_genes)
        rate = cumulative / oracle.total_hits if oracle.total_hits else 0.0
        trajectory.append({"round": r + 1, "tested": len(picks), "hits": len(hit_genes),
                           "cumulative": cumulative, "hit_ratio": round(rate, 4)})

        shown = ", ".join(hit_genes[:sample]) + ("  …" if len(hit_genes) > sample else "")
        print(f"  Round {r + 1}: tested {len(picks):>3d} → {len(hit_genes):>3d} hits"
              f"  [{shown}]")
        print(f"           cumulative {cumulative}/{oracle.total_hits} hits "
              f"({rate:.1%} of all hits recovered)")

    # If the researcher continued, this is what the tool would test next.
    next_batch = arm.select(rounds, revealed)
    print(f"\n  Next round would test: {', '.join(next_batch[:sample])}"
          f"{'  …' if len(next_batch) > sample else ''}")
    print(f"\n  Summary: {cumulative}/{oracle.total_hits} hits in {rounds} rounds "
          f"(hit_ratio@R{rounds} = {trajectory[-1]['hit_ratio']:.3f})\n")

    return {"dataset": dataset, "arm": arm_name, "trajectory": trajectory,
            "next_batch": next_batch}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a narrated sequential gene-selection campaign (System 2 C-arm).",
    )
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--n", type=int, default=None,
                        help="Genes per round (default: dataset batch size)")
    parser.add_argument("--arm", choices=list(_ARMS), default="waddington_c",
                        help="Which C-arm variant to drive")
    parser.add_argument("--skills", action="store_true",
                        help="Shorthand for --arm waddington_c_skills")
    args = parser.parse_args()

    arm_name = "waddington_c_skills" if args.skills else args.arm
    simulate(args.dataset, rounds=args.rounds, n=args.n, arm_name=arm_name)


if __name__ == "__main__":
    main()
