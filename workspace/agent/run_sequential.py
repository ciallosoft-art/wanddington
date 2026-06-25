#!/usr/bin/env python3
"""
Sequential oracle evaluation — paper experiment runner.

Runs one or more arms across all BDA benchmark datasets for N seeds
and reports hit_ratio@R5 per dataset plus averages.

Usage:
    conda run -n waddington-bio python3 run_sequential.py \\
        --arms waddington_v14 llm_reasoning coreset \\
        --seeds 5

Arm names
---------
  random                  Random selection baseline
  coreset                 Coreset diversity-maximising (A arm)
  static_ranker           LOO LightGBM static prior
  online_adaptive         PerTurboAgent-style online ML
  llm_reasoning           Pure LLM reasoning (B arm)
  waddington_v14          Waddington (C arm, paper final)

Ablation arms:
  waddington_v14_no_memory    C minus cross-experiment memory
  waddington_v14_no_llm       C minus LLM (online ML only)
  waddington_v14_no_ml        C minus online retraining (static LOO + LLM)
  waddington_v14_shuffled_names  C with anonymous gene identifiers

Development arms (archived, not needed for paper results):
  waddington, waddington_v2 … waddington_v13, waddington_v15
  → see workspace/agent/arms/archive/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workspace" / "agent"))

from oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from sequential_runner import RunResult, SequentialRunner

# ── Paper arms ──────────────────────────────────────────────────────────────
from arms.random_arm import RandomArm
from arms.static_ranker_arm import StaticRankerArm
from arms.coreset_arm import CoresetArm
from arms.online_adaptive_arm import OnlineAdaptiveArm
from arms.llm_reasoning_arm import LLMReasoningArm
from arms.waddington_v14_arm import WaddingtonV14Arm

# ── Ablation arms ────────────────────────────────────────────────────────────
from arms.waddington_v14_no_memory_arm import WaddingtonV14NoMemoryArm
from arms.waddington_v14_no_llm_arm import WaddingtonV14NoLLMArm
from arms.waddington_v14_no_ml_arm import WaddingtonV14NoMLArm
from arms.waddington_v14_shuffled_names_arm import WaddingtonV14ShuffledNamesArm

RESULTS_DIR = REPO_ROOT / "workspace" / "results" / "sequential"

# Arms available via --arms flag
_PAPER_ARMS = {
    "random", "coreset", "static_ranker", "online_adaptive",
    "llm_reasoning", "waddington_v14",
    "waddington_v14_no_memory", "waddington_v14_no_llm",
    "waddington_v14_no_ml", "waddington_v14_shuffled_names",
}

_ARCHIVE_ARMS = {
    "waddington", "waddington_v2", "waddington_v3", "waddington_v4",
    "waddington_v5", "waddington_v6", "waddington_v7", "waddington_v8",
    "waddington_v9", "waddington_v10", "waddington_v11", "waddington_v12",
    "waddington_v13", "waddington_v15",
}


def _load_archive_arm(name: str, dataset_name: str, bs: int):
    """Lazy-load a development-history arm from the archive directory."""
    import importlib
    mod = importlib.import_module(f"arms.archive.{name}_arm")
    cls_name = "".join(p.capitalize() for p in name.split("_")) + "Arm"
    # e.g. "waddington_v2" → "WaddingtonV2Arm"
    cls = getattr(mod, cls_name)
    return cls(dataset_name, bs)


def make_arms(dataset_name: str, arm_names: list[str]) -> list:
    oracle = DatasetOracle(dataset_name)
    pool = oracle.all_genes()
    bs = BATCH_SIZES[dataset_name]
    arms = []
    for name in arm_names:
        if name == "random":
            arms.append(RandomArm(dataset_name, bs, pool, seed=42))
        elif name == "static_ranker":
            arms.append(StaticRankerArm(dataset_name, bs))
        elif name == "coreset":
            arms.append(CoresetArm(dataset_name, bs, seed=42))
        elif name == "online_adaptive":
            arms.append(OnlineAdaptiveArm(dataset_name, bs))
        elif name == "llm_reasoning":
            arms.append(LLMReasoningArm(dataset_name, bs))
        elif name == "waddington_v14":
            arms.append(WaddingtonV14Arm(dataset_name, bs))
        elif name == "waddington_v14_no_memory":
            arms.append(WaddingtonV14NoMemoryArm(dataset_name, bs))
        elif name == "waddington_v14_no_llm":
            arms.append(WaddingtonV14NoLLMArm(dataset_name, bs))
        elif name == "waddington_v14_no_ml":
            arms.append(WaddingtonV14NoMLArm(dataset_name, bs))
        elif name == "waddington_v14_shuffled_names":
            arms.append(WaddingtonV14ShuffledNamesArm(dataset_name, bs))
        elif name in _ARCHIVE_ARMS:
            print(f"    [archive] Loading {name} from arms/archive/ ...")
            arms.append(_load_archive_arm(name, dataset_name, bs))
        else:
            print(f"    [WARN] Unknown arm '{name}', skipping")
    return arms


def print_result(r: RunResult) -> None:
    ratios = "  ".join(f"R{i+1}={v:.3f}" for i, v in enumerate(r.hit_ratio_per_round))
    print(
        f"  {r.arm_name:30s}  {ratios}  "
        f"AUC_norm={r.auc_normalized:.3f}  hits={r.cumulative_hits[-1]}/{r.total_hits}"
    )


def run_all(
    datasets: list[str],
    arm_names: list[str],
    n_rounds: int,
    n_seeds: int,
    out_path: Path | None = None,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, list]] = {}

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  (batch={BATCH_SIZES[ds]}, rounds={n_rounds})")
        print(f"{'='*60}")

        oracle = DatasetOracle(ds)
        print(f"  Genes: {oracle.n_genes}, Hits: {oracle.total_hits}")

        ds_results: dict[str, list[RunResult]] = {}

        for arm_name in arm_names:
            seed_results = []
            for seed in range(n_seeds):
                arms = make_arms(ds, [arm_name])
                if not arms:
                    continue
                arm = arms[0]
                runner = SequentialRunner(arm, oracle, n_rounds=n_rounds)
                r = runner.run(seed=seed)
                seed_results.append(r)

            if not seed_results:
                continue

            avg_result = _average_results(seed_results)
            ds_results[arm_name] = seed_results
            print_result(avg_result)

        all_results[ds] = {
            arm: [_result_to_dict(r) for r in results]
            for arm, results in ds_results.items()
        }

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY  (hit_ratio @ final round, avg across seeds)")
    print(f"{'='*60}")
    header = f"{'Dataset':28s}" + "".join(f"  {a:>22s}" for a in arm_names)
    print(header)
    for ds in datasets:
        row = f"{ds:28s}"
        for arm_name in arm_names:
            if arm_name in all_results.get(ds, {}):
                seed_results = all_results[ds][arm_name]
                avg_ratio = sum(r["hit_ratio_per_round"][-1] for r in seed_results) / len(seed_results)
                row += f"  {avg_ratio:>22.3f}"
            else:
                row += f"  {'N/A':>22s}"
        print(row)

    save_path = out_path or (RESULTS_DIR / "results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    return all_results


def _average_results(results: list[RunResult]) -> RunResult:
    n = len(results)
    r0 = results[0]
    avg = RunResult(
        dataset=r0.dataset,
        arm_name=r0.arm_name,
        batch_size=r0.batch_size,
        n_rounds=r0.n_rounds,
        total_hits=r0.total_hits,
        n_genes=r0.n_genes,
    )
    n_rounds = r0.n_rounds
    avg.hits_per_round = [
        round(sum(r.hits_per_round[i] for r in results) / n)
        for i in range(n_rounds)
    ]
    avg.cumulative_hits = [
        round(sum(r.cumulative_hits[i] for r in results) / n)
        for i in range(n_rounds)
    ]
    avg.hit_ratio_per_round = [
        round(sum(r.hit_ratio_per_round[i] for r in results) / n, 4)
        for i in range(n_rounds)
    ]
    avg.auc_normalized = round(sum(r.auc_normalized for r in results) / n, 4)
    return avg


def _result_to_dict(r: RunResult) -> dict:
    return {
        "dataset": r.dataset,
        "arm_name": r.arm_name,
        "batch_size": r.batch_size,
        "n_rounds": r.n_rounds,
        "total_hits": r.total_hits,
        "n_genes": r.n_genes,
        "hits_per_round": r.hits_per_round,
        "cumulative_hits": r.cumulative_hits,
        "hit_ratio_per_round": r.hit_ratio_per_round,
        "auc_normalized": r.auc_normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential oracle evaluation for paper experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Paper arms:  " + "  ".join(sorted(_PAPER_ARMS)),
            "Archive arms (dev history): " + "  ".join(sorted(_ARCHIVE_ARMS)),
        ]),
    )
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="Datasets to evaluate (default: all 9)")
    parser.add_argument("--arms", nargs="+",
                        default=["coreset", "llm_reasoning", "waddington_v14"],
                        help="Arms to run")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of selection rounds (default: 5)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (default: workspace/results/sequential/results.json)")
    args = parser.parse_args()

    run_all(args.datasets, args.arms, args.rounds, args.seeds, args.out)


if __name__ == "__main__":
    main()
