#!/usr/bin/env python3
"""
V8 entry point: sequential oracle evaluation of all arms on all BDA datasets.

Usage:
    conda run -n waddington-bio python3 run_sequential.py
    conda run -n waddington-bio python3 run_sequential.py --datasets IFNG IL2
    conda run -n waddington-bio python3 run_sequential.py --arms random static_ranker
    conda run -n waddington-bio python3 run_sequential.py --rounds 5 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "workspace" / "agent"))

from oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from arms.random_arm import RandomArm
from arms.static_ranker_arm import StaticRankerArm
from arms.coreset_arm import CoresetArm
from arms.online_adaptive_arm import OnlineAdaptiveArm
from arms.llm_reasoning_arm import LLMReasoningArm
from arms.waddington_arm import WaddingtonArm
from arms.waddington_v2_arm import WaddingtonV2Arm
from arms.waddington_v3_arm import WaddingtonV3Arm
from arms.waddington_v4_arm import WaddingtonV4Arm
from arms.waddington_v5_arm import WaddingtonV5Arm
from arms.waddington_v6_arm import WaddingtonV6Arm
from arms.waddington_v7_arm import WaddingtonV7Arm
from arms.waddington_v8_arm import WaddingtonV8Arm
from arms.waddington_v9_arm import WaddingtonV9Arm
from arms.waddington_v10_arm import WaddingtonV10Arm
from arms.waddington_v11_arm import WaddingtonV11Arm
from arms.waddington_v12_arm import WaddingtonV12Arm
from arms.waddington_v13_arm import WaddingtonV13Arm
from arms.waddington_v14_arm import WaddingtonV14Arm
from arms.waddington_v15_arm import WaddingtonV15Arm
from arms.waddington_v14_no_memory_arm import WaddingtonV14NoMemoryArm
from arms.waddington_v14_no_llm_arm import WaddingtonV14NoLLMArm
from sequential_runner import RunResult, SequentialRunner

RESULTS_DIR = REPO_ROOT / "workspace" / "results" / "sequential"


def make_arms(dataset_name: str, arm_names: list[str]) -> list:
    oracle = DatasetOracle(dataset_name)
    pool = oracle.all_genes()
    bs = BATCH_SIZES[dataset_name]
    arms = []
    for name in arm_names:
        if name == "random":
            arms.append(RandomArm(dataset_name, bs, pool, seed=42))
        elif name == "static_ranker":
            print(f"    [StaticRanker] Building LOO ranking for {dataset_name}...")
            arms.append(StaticRankerArm(dataset_name, bs))
        elif name == "coreset":
            arms.append(CoresetArm(dataset_name, bs, seed=42))
        elif name == "online_adaptive":
            arms.append(OnlineAdaptiveArm(dataset_name, bs))
        elif name == "llm_reasoning":
            print(f"    [LLMReasoning] Building StaticRanker fallback + loading task for {dataset_name}...")
            arms.append(LLMReasoningArm(dataset_name, bs))
        elif name == "waddington":
            print(f"    [Waddington] Building C-arm (ML + memory + LLM) for {dataset_name}...")
            arms.append(WaddingtonArm(dataset_name, bs))
        elif name == "waddington_v2":
            print(f"    [WaddingtonV2] Building C-arm v2 (weighted ensemble) for {dataset_name}...")
            arms.append(WaddingtonV2Arm(dataset_name, bs))
        elif name == "waddington_v3":
            print(f"    [WaddingtonV3] Building C-arm v3 (EMA adaptive weights) for {dataset_name}...")
            arms.append(WaddingtonV3Arm(dataset_name, bs))
        elif name == "waddington_v4":
            print(f"    [WaddingtonV4] Building C-arm v4 (static routing) for {dataset_name}...")
            arms.append(WaddingtonV4Arm(dataset_name, bs))
        elif name == "waddington_v5":
            print(f"    [WaddingtonV5] Building C-arm v5 (two-bucket routing) for {dataset_name}...")
            arms.append(WaddingtonV5Arm(dataset_name, bs))
        elif name == "waddington_v6":
            print(f"    [WaddingtonV6] Building C-arm v6 (routing + temp=0) for {dataset_name}...")
            arms.append(WaddingtonV6Arm(dataset_name, bs))
        elif name == "waddington_v7":
            print(f"    [WaddingtonV7] Building C-arm v7 (3-bucket routing + two-stage) for {dataset_name}...")
            arms.append(WaddingtonV7Arm(dataset_name, bs))
        elif name == "waddington_v8":
            print(f"    [WaddingtonV8] Building C-arm v8 (two-stage shortlist=384, full display) for {dataset_name}...")
            arms.append(WaddingtonV8Arm(dataset_name, bs))
        elif name == "waddington_v9":
            print(f"    [WaddingtonV9] Building C-arm v9 (Sonnet LLM + three-bucket routing) for {dataset_name}...")
            arms.append(WaddingtonV9Arm(dataset_name, bs))
        elif name == "waddington_v10":
            print(f"    [WaddingtonV10] Building C-arm v10 (DepMap features) for {dataset_name}...")
            arms.append(WaddingtonV10Arm(dataset_name, bs))
        elif name == "waddington_v11":
            print(f"    [WaddingtonV11] Building C-arm v11 (selective DepMap: essential→v1) for {dataset_name}...")
            arms.append(WaddingtonV11Arm(dataset_name, bs))
        elif name == "waddington_v12":
            print(f"    [WaddingtonV12] Building C-arm v12 (ml_heavy for all n>15000) for {dataset_name}...")
            arms.append(WaddingtonV12Arm(dataset_name, bs))
        elif name == "waddington_v13":
            print(f"    [WaddingtonV13] Building C-arm v13 (DepMap excluded: essential+Steinhart) for {dataset_name}...")
            arms.append(WaddingtonV13Arm(dataset_name, bs))
        elif name == "waddington_v14":
            print(f"    [WaddingtonV14] Building C-arm v14 (per-dataset optimal features: v1/v2/v3) for {dataset_name}...")
            arms.append(WaddingtonV14Arm(dataset_name, bs))
        elif name == "waddington_v15":
            print(f"    [WaddingtonV15] Building C-arm v15 (uncertainty-aware dynamic weights) for {dataset_name}...")
            arms.append(WaddingtonV15Arm(dataset_name, bs))
        elif name == "waddington_v14_no_memory":
            print(f"    [WaddingtonV14-NoMemory] Building C-memory ablation for {dataset_name}...")
            arms.append(WaddingtonV14NoMemoryArm(dataset_name, bs))
        elif name == "waddington_v14_no_llm":
            print(f"    [WaddingtonV14-NoLLM] Building C-LLM ablation for {dataset_name}...")
            arms.append(WaddingtonV14NoLLMArm(dataset_name, bs))
        else:
            print(f"    [WARN] Unknown arm '{name}', skipping")
    return arms


def print_result(r: RunResult) -> None:
    ratios = "  ".join(f"R{i+1}={v:.3f}" for i, v in enumerate(r.hit_ratio_per_round))
    print(
        f"  {r.arm_name:20s}  {ratios}  "
        f"AUC_norm={r.auc_normalized:.3f}  hits={r.cumulative_hits[-1]}/{r.total_hits}"
    )


def run_all(
    datasets: list[str],
    arm_names: list[str],
    n_rounds: int,
    n_seeds: int,
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

            # Average across seeds
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
    header = f"{'Dataset':22s}" + "".join(f"  {a:>18s}" for a in arm_names)
    print(header)
    for ds in datasets:
        row = f"{ds:22s}"
        for arm_name in arm_names:
            if arm_name in all_results.get(ds, {}):
                seed_results = all_results[ds][arm_name]
                avg_ratio = sum(r["hit_ratio_per_round"][-1] for r in seed_results) / len(seed_results)
                row += f"  {avg_ratio:>18.3f}"
            else:
                row += f"  {'N/A':>18s}"
        print(row)

    # Save results JSON
    out_path = RESULTS_DIR / "v8_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

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
    parser = argparse.ArgumentParser(description="V8 sequential oracle evaluation")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--arms", nargs="+", default=["random", "coreset", "static_ranker", "online_adaptive", "llm_reasoning", "waddington", "waddington_v2", "waddington_v3"])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    run_all(args.datasets, args.arms, args.rounds, args.seeds)


if __name__ == "__main__":
    main()
