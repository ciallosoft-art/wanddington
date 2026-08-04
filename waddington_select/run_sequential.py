#!/usr/bin/env python3
"""
Sequential oracle evaluation — paper experiment runner.

Runs one or more arms across all BDA benchmark datasets for N seeds
and reports hit_ratio@R5 per dataset plus averages.

Usage:
    conda run -n waddington-bio python3 run_sequential.py \\
        --arms waddington_c llm_reasoning coreset \\
        --seeds 5

Arm names
---------
  random                  Random selection baseline
  coreset                 Coreset diversity-maximising (A arm)
  static_ranker           LOO LightGBM static prior
  online_adaptive         PerTurboAgent-style online ML
  llm_reasoning           Pure LLM reasoning (B arm)
  waddington_c            Waddington (C arm, paper final)

Ablation arms:
  waddington_c_no_memory      C minus cross-experiment memory
  waddington_c_no_llm         C minus LLM (online ML only)
  waddington_c_no_ml          C minus online retraining (static LOO + LLM)
  waddington_c_shuffled_names   C with anonymous gene identifiers (A0: no info to reason over)
  waddington_c_feature_reasoning C with anonymous IDs + structural feature vectors (A1: reason, not recall)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from .oracle import BENCHMARK_DATASETS, BATCH_SIZES, DatasetOracle
from .sequential_runner import RunResult, SequentialRunner

# ── Paper arms ──────────────────────────────────────────────────────────────
from .arms.random_arm import RandomArm
from .arms.static_ranker_arm import StaticRankerArm
from .arms.coreset_arm import CoresetArm
from .arms.online_adaptive_arm import OnlineAdaptiveArm
from .arms.llm_reasoning_arm import LLMReasoningArm
from .arms.waddington_c_arm import WaddingtonCArm, WaddingtonCSkillsArm, WaddingtonCMemSkillsArm, WaddingtonCEnrichArm

# ── Ablation arms ────────────────────────────────────────────────────────────
from .arms.waddington_c_no_memory_arm import WaddingtonCNoMemoryArm
from .arms.waddington_c_no_llm_arm import WaddingtonCNoLLMArm
from .arms.waddington_c_no_ml_arm import WaddingtonCNoMLArm
from .arms.waddington_c_shuffled_names_arm import WaddingtonCShuffledNamesArm
from .arms.waddington_c_feature_reasoning_arm import WaddingtonCFeatureReasoningArm, WaddingtonCPoolOnlyArm, WaddingtonCLinearArm
from .arms.waddington_c_ensemble_arm import WaddingtonCEnsembleArm
from .arms.genedisco_arm import GeneDiscoArm, ACQUISITIONS as _GD_ACQ

RESULTS_DIR = REPO_ROOT / "workspace" / "results" / "sequential"

# Arms available via --arms flag
_PAPER_ARMS = {
    "random", "coreset", "static_ranker", "online_adaptive",
    "llm_reasoning", "waddington_c", "waddington_c_skills", "waddington_c_memskills", "waddington_c_enrich",
    "waddington_c_no_memory", "waddington_c_no_llm",
    "waddington_c_no_ml", "waddington_c_shuffled_names",
    "waddington_c_feature_reasoning", "waddington_c_pool_only", "waddington_c_ensemble", "waddington_c_linear",
} | {f"genedisco_{a}" for a in _GD_ACQ}


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
        elif name == "waddington_c":
            arms.append(WaddingtonCArm(dataset_name, bs))
        elif name == "waddington_c_skills":
            arms.append(WaddingtonCSkillsArm(dataset_name, bs))
        elif name == "waddington_c_memskills":
            arms.append(WaddingtonCMemSkillsArm(dataset_name, bs))
        elif name == "waddington_c_enrich":
            arms.append(WaddingtonCEnrichArm(dataset_name, bs))
        elif name == "waddington_c_no_memory":
            arms.append(WaddingtonCNoMemoryArm(dataset_name, bs))
        elif name == "waddington_c_no_llm":
            arms.append(WaddingtonCNoLLMArm(dataset_name, bs))
        elif name == "waddington_c_no_ml":
            arms.append(WaddingtonCNoMLArm(dataset_name, bs))
        elif name == "waddington_c_shuffled_names":
            arms.append(WaddingtonCShuffledNamesArm(dataset_name, bs))
        elif name == "waddington_c_feature_reasoning":
            arms.append(WaddingtonCFeatureReasoningArm(dataset_name, bs))
        elif name == "waddington_c_pool_only":
            arms.append(WaddingtonCPoolOnlyArm(dataset_name, bs))
        elif name == "waddington_c_linear":
            arms.append(WaddingtonCLinearArm(dataset_name, bs))
        elif name == "waddington_c_ensemble":
            arms.append(WaddingtonCEnsembleArm(dataset_name, bs))
        elif name.startswith("genedisco_"):
            arms.append(GeneDiscoArm(dataset_name, bs, acquisition=name[len("genedisco_"):]))
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
    resume: bool = False,
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = out_path or (RESULTS_DIR / "results.json")

    # --- Resume: load existing checkpoint ---
    all_results: dict[str, dict[str, list]] = {}
    if resume and save_path.exists():
        with open(save_path) as f:
            all_results = json.load(f)
        completed = {
            ds for ds, arms in all_results.items()
            if all(arm in arms for arm in arm_names)
        }
        if completed:
            print(f"[resume] Skipping {len(completed)} already-completed datasets: "
                  f"{', '.join(sorted(completed))}")
    else:
        completed = set()

    for ds in datasets:
        if ds in completed:
            continue

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

        # Checkpoint: save after every dataset so progress survives crashes
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [checkpoint] {save_path}")

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
        # Gene identities of the hits found, dumped only when requested (for the novel-hit analysis);
        # off by default so ordinary result files stay compact.
        **({"revealed_hits": r.revealed_hits}
           if os.environ.get("WADDINGTON_DUMP_SELECTIONS") == "1" else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential oracle evaluation for paper experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Paper arms:  " + "  ".join(sorted(_PAPER_ARMS)),
    )
    parser.add_argument("--datasets", nargs="+", default=BENCHMARK_DATASETS,
                        help="Datasets to evaluate (default: the 9 benchmark screens with ground "
                             "truth; registered user phenotypes have no oracle and are excluded)")
    parser.add_argument("--arms", nargs="+",
                        default=["coreset", "llm_reasoning", "waddington_c"],
                        help="Arms to run")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of selection rounds (default: 5)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of random seeds (default: 5)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (default: workspace/results/sequential/results.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint: skip already-completed datasets")
    args = parser.parse_args()

    run_all(args.datasets, args.arms, args.rounds, args.seeds, args.out, args.resume)


if __name__ == "__main__":
    main()
