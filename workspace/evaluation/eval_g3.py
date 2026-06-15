#!/usr/bin/env python3
"""
9.3 — G3 NegativeFilter Quantitative Evaluation

Simulates technical failures and true negatives in a controlled setting,
measures how well the G3 rule set distinguishes the two failure modes.

Decision paths mirrored from src/web/negative-filter.ts:
  Path 1 (E3 type)       : failure.type ∈ {environment_failure, data_access_failure, …}
  Path 2 (protocol low)  : rfs.protocol < 0.5
  Path 3 (discrepancy)   : rfs.overall < 0.40 AND rfs.biology > 0.6
  Path 4 (quality + low) : rfs.overall ≥ 0.65 AND metric low AND ≥N prior quality runs
  Path 5 (metric mismatch): failure.type = metric_mismatch
  Default                 : needs_investigation

Key metrics:
  tf_recovery_rate    — P(verdict=technical_failure | true_hit, TF signal injected)
  false_blacklist_rate — P(verdict=true_negative   | true_hit, TF signal injected)
  tn_precision        — P(verdict=true_negative    | true_non_hit, multi-run quality)
  tn_recall           — P(verdict=true_negative    | true_non_hit, multi-run quality)

Usage:
  conda run -n waddington-bio python3 workspace/evaluation/eval_g3.py
  conda run -n waddington-bio python3 workspace/evaluation/eval_g3.py --dataset IL2 --n-genes 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
BDA_DIR   = REPO_ROOT / "workspace" / "data" / "bda_benchmark"
CEG_PATH  = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# RFS + E3 data structures (mirrors TypeScript types)
# ---------------------------------------------------------------------------

@dataclass
class RFSResult:
    overall:  float        # 0-1, weighted average
    result:   float        # primary metric fidelity
    metric:   float        # measurement fidelity
    protocol: float        # protocol adherence (-1 if no spec)
    biology:  float        # GO/PPI enrichment

@dataclass
class FailureRecord:
    type: str              # E3 failure type
    message: str = ""
    evidence: list[str] = field(default_factory=list)
    suggested_fix: str = ""

Verdict = Literal["technical_failure", "true_negative", "needs_investigation"]

@dataclass
class FilterResult:
    verdict:    Verdict
    confidence: str
    reason:     str
    path:       int        # which decision path fired (1-6 or 0 for default)

# ---------------------------------------------------------------------------
# G3 rule engine (port of negative-filter.ts, stateless version)
# ---------------------------------------------------------------------------

PROTOCOL_FIDELITY_MIN   = 0.5
OVERALL_RFS_GOOD        = 0.65
PROTOCOL_RFS_GOOD       = 0.70
BIOLOGY_SCORE_OK        = 0.6
TECHNICAL_FAILURE_TYPES = {"environment_failure", "data_access_failure", "protocol_ambiguity"}

def classify(
    rfs: RFSResult,
    primary_metric: float,        # the main experimental readout (e.g. pearson_de)
    failure: FailureRecord | None = None,
    prior_quality_runs_with_low_metric: int = 0,
    primary_metric_threshold: float = 0.30,
) -> FilterResult:
    """Stateless G3 classifier — caller supplies prior run count."""

    # Path 1 — clear E3 technical failure type
    if failure and failure.type in TECHNICAL_FAILURE_TYPES:
        return FilterResult("technical_failure", "high",
                            f"E3 type: {failure.type}", path=1)

    # Path 2 — protocol fidelity too low
    if rfs.protocol >= 0 and rfs.protocol < PROTOCOL_FIDELITY_MIN:
        return FilterResult("technical_failure", "high",
                            f"RFS.protocol={rfs.protocol:.2f} < {PROTOCOL_FIDELITY_MIN}", path=2)

    # Path 3 — overall low but biology fine → technical discrepancy
    if rfs.overall >= 0 and rfs.overall < 0.40 and rfs.biology > BIOLOGY_SCORE_OK:
        return FilterResult("technical_failure", "medium",
                            f"RFS.overall={rfs.overall:.2f}<0.40 but biology={rfs.biology:.2f}>0.6",
                            path=3)

    # Path 4 — good experiment quality, evaluate metric
    protocol_ok = (rfs.protocol < 0) or (rfs.protocol >= PROTOCOL_RFS_GOOD)
    quality_ok  = rfs.overall >= OVERALL_RFS_GOOD and protocol_ok
    metric_low  = primary_metric < primary_metric_threshold

    if quality_ok and metric_low:
        total_quality_runs = prior_quality_runs_with_low_metric + 1
        if total_quality_runs >= 3:
            return FilterResult("true_negative", "high",
                                f"{total_quality_runs} quality runs all low", path=4)
        if total_quality_runs == 2:
            return FilterResult("true_negative", "medium",
                                "2 quality runs both low", path=4)
        return FilterResult("needs_investigation", "low",
                            "1 quality run low — need replicate", path=4)

    # Path 5 — metric mismatch
    if failure and failure.type == "metric_mismatch":
        return FilterResult("technical_failure", "medium",
                            "E3 type: metric_mismatch", path=5)

    # Default
    return FilterResult("needs_investigation", "low",
                        "insufficient signal", path=0)

# ---------------------------------------------------------------------------
# RFS injection scenarios
# ---------------------------------------------------------------------------

def rfs_tf_protocol(rng: random.Random) -> tuple[RFSResult, FailureRecord | None]:
    """Path 2 trigger: protocol < 0.5"""
    return RFSResult(
        overall  = rng.uniform(0.35, 0.55),
        result   = rng.uniform(0.20, 0.45),
        metric   = rng.uniform(0.50, 0.70),
        protocol = rng.uniform(0.15, 0.45),   # below 0.5
        biology  = rng.uniform(0.50, 0.75),
    ), None

def rfs_tf_discrepancy(rng: random.Random) -> tuple[RFSResult, FailureRecord | None]:
    """Path 3 trigger: overall < 0.40 but biology > 0.6"""
    return RFSResult(
        overall  = rng.uniform(0.25, 0.38),   # below 0.40
        result   = rng.uniform(0.20, 0.35),
        metric   = rng.uniform(0.55, 0.70),
        protocol = rng.uniform(0.60, 0.85),   # protocol is fine
        biology  = rng.uniform(0.62, 0.85),   # biology is fine → discrepancy
    ), None

def rfs_tf_e3(rng: random.Random) -> tuple[RFSResult, FailureRecord | None]:
    """Path 1 trigger: E3 environment failure"""
    return RFSResult(
        overall  = rng.uniform(0.10, 0.50),
        result   = rng.uniform(0.10, 0.40),
        metric   = rng.uniform(0.30, 0.60),
        protocol = rng.uniform(0.30, 0.70),
        biology  = rng.uniform(0.30, 0.70),
    ), FailureRecord(type="environment_failure", message="conda env broken")

def rfs_good_experiment(rng: random.Random) -> RFSResult:
    """High-quality experiment RFS (no TF signal)."""
    return RFSResult(
        overall  = rng.uniform(0.68, 0.90),
        result   = rng.uniform(0.60, 0.85),
        metric   = rng.uniform(0.65, 0.88),
        protocol = rng.uniform(0.72, 0.92),
        biology  = rng.uniform(0.60, 0.82),
    )

TF_SCENARIOS: dict[str, callable] = {
    "tf_protocol":     rfs_tf_protocol,
    "tf_discrepancy":  rfs_tf_discrepancy,
    "tf_e3":           rfs_tf_e3,
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> tuple[list[str], set[str]]:
    """Returns (all_genes, hit_genes_set) after CEGv2 filter."""
    datasets = {
        "IFNG":       ("ground_truth_IFNG.csv",                      "topmovers_IFNG.npy"),
        "IL2":        ("ground_truth_IL2.csv",                       "topmovers_IL2.npy"),
        "Sanchez21":  ("ground_truth_Sanchez21.csv",                 "topmovers_Sanchez21.npy"),
        "Steinhart":  ("ground_truth_Steinhart_crispra_GD2_D22.csv", "topmovers_Steinhart_crispra_GD2_D22.npy"),
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets)}")

    csv_f, npy_f = datasets[name]
    df = pd.read_csv(BDA_DIR / csv_f)
    gene_col = df.columns[0]
    all_genes = df[gene_col].str.strip().str.upper().tolist()

    ceg = pd.read_csv(CEG_PATH, sep="\t")
    essential = set(ceg["GENE"].dropna().str.strip().str.upper())
    all_genes = [g for g in all_genes if g not in essential]

    hits = {str(g).strip().upper()
            for g in np.load(BDA_DIR / npy_f, allow_pickle=True)}
    hits = {g for g in hits if g not in essential}

    return all_genes, hits


def evaluate_tf_recovery(
    hit_genes: list[str],
    n_genes: int,
    seed: int = 42,
) -> dict[str, dict]:
    """
    For each TF scenario: inject TF signal into true hit genes.
    Measure tf_recovery_rate and false_blacklist_rate per scenario.
    """
    rng = random.Random(seed)
    sample = rng.sample(hit_genes, min(n_genes, len(hit_genes)))

    results: dict[str, dict] = {}
    for scenario_name, inject_fn in TF_SCENARIOS.items():
        verdicts = {"technical_failure": 0, "true_negative": 0, "needs_investigation": 0}
        paths_fired = []

        for gene in sample:
            rfs, failure = inject_fn(rng)
            # Primary metric is also low (hit didn't show phenotype due to TF)
            metric = rng.uniform(0.05, 0.25)
            r = classify(rfs, metric, failure, prior_quality_runs_with_low_metric=0)
            verdicts[r.verdict] += 1
            paths_fired.append(r.path)

        n = len(sample)
        tf_rate   = verdicts["technical_failure"] / n
        fn_rate   = verdicts["true_negative"] / n
        ni_rate   = verdicts["needs_investigation"] / n
        results[scenario_name] = {
            "n": n,
            "tf_recovery_rate":    round(tf_rate, 4),
            "false_blacklist_rate": round(fn_rate, 4),
            "needs_investigation": round(ni_rate, 4),
            "paths": {p: paths_fired.count(p) for p in sorted(set(paths_fired))},
        }
    return results


def evaluate_tn_precision(
    non_hit_genes: list[str],
    n_genes: int,
    n_runs_per_gene: int = 3,   # simulate this many rounds per gene
    seed: int = 42,
) -> dict:
    """
    Simulate multi-round good-quality experiments with consistently low primary metrics.
    Measure: after N rounds, how many non-hits does G3 correctly label true_negative?
    """
    rng = random.Random(seed)
    sample = rng.sample(non_hit_genes, min(n_genes, len(non_hit_genes)))

    final_verdicts: dict[str, int] = {"technical_failure": 0, "true_negative": 0,
                                       "needs_investigation": 0}
    round_verdicts: list[list[str]] = [[] for _ in range(n_runs_per_gene)]

    for gene in sample:
        prior_quality_with_low = 0
        for run_idx in range(n_runs_per_gene):
            rfs = rfs_good_experiment(rng)
            metric = rng.uniform(0.02, 0.20)    # consistently low
            r = classify(rfs, metric, failure=None,
                         prior_quality_runs_with_low_metric=prior_quality_with_low)
            round_verdicts[run_idx].append(r.verdict)
            # If this run was quality OK + metric low, increment prior count
            protocol_ok = rfs.protocol < 0 or rfs.protocol >= PROTOCOL_RFS_GOOD
            quality_ok  = rfs.overall >= OVERALL_RFS_GOOD and protocol_ok
            if quality_ok and metric < 0.30:
                prior_quality_with_low += 1

        final_verdicts[r.verdict] += 1    # last round verdict

    n = len(sample)
    return {
        "n": n,
        "n_runs_per_gene": n_runs_per_gene,
        "final_tn_rate":   round(final_verdicts["true_negative"] / n, 4),
        "final_ni_rate":   round(final_verdicts["needs_investigation"] / n, 4),
        "final_tf_rate":   round(final_verdicts["technical_failure"] / n, 4),
        "per_round": [
            {f"round_{i+1}": {v: round_verdicts[i].count(v) / n for v in
                               ["technical_failure", "true_negative", "needs_investigation"]}}
            for i in range(n_runs_per_gene)
        ],
    }


def evaluate_tn_false_save(
    non_hit_genes: list[str],
    n_genes: int,
    seed: int = 42,
) -> dict:
    """
    Inject TF signals into NON-HIT genes.
    G3 will (correctly) classify them as technical_failure and retry them.
    This is a 'false save': we waste a retry slot on a gene with no effect.
    Report the rate to quantify the cost of G3's conservative approach.
    """
    rng = random.Random(seed)
    sample = rng.sample(non_hit_genes, min(n_genes, len(non_hit_genes)))

    per_scenario: dict[str, dict] = {}
    for scenario_name, inject_fn in TF_SCENARIOS.items():
        verdicts = {"technical_failure": 0, "true_negative": 0, "needs_investigation": 0}
        for gene in sample:
            rfs, failure = inject_fn(rng)
            metric = rng.uniform(0.05, 0.25)
            r = classify(rfs, metric, failure, prior_quality_runs_with_low_metric=0)
            verdicts[r.verdict] += 1
        n = len(sample)
        per_scenario[scenario_name] = {
            "false_save_rate": round(verdicts["technical_failure"] / n, 4),
        }
    return {"n": len(sample), "per_scenario": per_scenario}


def evaluate_boundary_cases(seed: int = 42) -> list[dict]:
    """
    Test G3 behavior near decision thresholds.
    Returns list of {label, rfs, metric, verdict, path}.
    """
    rng = random.Random(seed)
    cases = [
        # Near Path 2 threshold (protocol=0.5)
        ("protocol=0.49 (just below TF threshold)",
         RFSResult(overall=0.60, result=0.55, metric=0.65, protocol=0.49, biology=0.70), 0.10, None),
        ("protocol=0.51 (just above TF threshold)",
         RFSResult(overall=0.60, result=0.55, metric=0.65, protocol=0.51, biology=0.70), 0.10, None),
        # Near Path 3 threshold (overall=0.40)
        ("overall=0.39 + biology=0.65 (TF discrepancy)",
         RFSResult(overall=0.39, result=0.35, metric=0.65, protocol=0.75, biology=0.65), 0.10, None),
        ("overall=0.41 + biology=0.65 (escapes Path 3)",
         RFSResult(overall=0.41, result=0.38, metric=0.65, protocol=0.75, biology=0.65), 0.10, None),
        # Path 4: good quality, 0/1/2 prior runs
        ("good quality, 0 prior runs (NI expected)",
         RFSResult(overall=0.70, result=0.65, metric=0.70, protocol=0.75, biology=0.65), 0.10, None),
        ("good quality, 1 prior run  (TN-medium)",
         RFSResult(overall=0.70, result=0.65, metric=0.70, protocol=0.75, biology=0.65), 0.10, None),
        ("good quality, 2 prior runs (TN-high)",
         RFSResult(overall=0.70, result=0.65, metric=0.70, protocol=0.75, biology=0.65), 0.10, None),
    ]
    results = []
    prior_runs = [0, 0, 0, 0, 0, 1, 2]  # aligned with cases
    for (label, rfs, metric, failure), prior in zip(cases, prior_runs):
        r = classify(rfs, metric, failure, prior_quality_runs_with_low_metric=prior)
        results.append({
            "label": label,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "path": r.path,
        })
    return results


def evaluate_no_g3_baseline(
    hit_genes: list[str],
    n_genes: int,
    seed: int = 42,
) -> dict:
    """
    Counterfactual: what if every low result was classified as true_negative?
    Shows the false_blacklist_rate WITHOUT G3 (naive baseline).
    """
    rng = random.Random(seed)
    sample = rng.sample(hit_genes, min(n_genes, len(hit_genes)))

    false_blacklisted = 0
    for gene in sample:
        # Pick a random TF scenario
        inject_fn = random.choice(list(TF_SCENARIOS.values()))
        rfs, _ = inject_fn(rng)
        # Naive: classify as true_negative whenever metric is low (no TF check)
        metric = rng.uniform(0.05, 0.25)
        if metric < 0.30:
            false_blacklisted += 1     # naive system blacklists this true hit

    return {
        "n": len(sample),
        "false_blacklist_rate_no_g3": round(false_blacklisted / len(sample), 4),
    }


def print_report(
    dataset: str,
    tf_results: dict,
    tn_results: dict,
    fs_results: dict,
) -> None:
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"G3 NegativeFilter Evaluation  —  dataset: {dataset}")
    print(sep)

    print("\n── A. Technical Failure Recovery (true hits, TF signal injected) ──")
    print(f"{'Scenario':<20} {'TF Recovery':>12} {'False Blacklist':>15} {'Needs Invest.':>14}")
    print("-" * 64)
    for scenario, r in tf_results.items():
        print(f"  {scenario:<18}  {r['tf_recovery_rate']:>11.1%}  "
              f"{r['false_blacklist_rate']:>14.1%}  {r['needs_investigation']:>13.1%}")
        print(f"    paths fired: {r['paths']}")

    print("\n── B. True Negative Identification (non-hits, multi-round quality) ──")
    r = tn_results
    print(f"  Gene sample: {r['n']}  |  Rounds per gene: {r['n_runs_per_gene']}")
    for rd in r["per_round"]:
        for rnd, v in rd.items():
            print(f"  {rnd}: TN={v['true_negative']:.1%}  NI={v['needs_investigation']:.1%}  "
                  f"TF={v['technical_failure']:.1%}")
    print(f"  → Final TN rate after {r['n_runs_per_gene']} rounds: {r['final_tn_rate']:.1%}")

    print("\n── C. False Save Rate (non-hits given TF signal → wasted retry) ──")
    print(f"  Gene sample: {fs_results['n']}")
    for scenario, v in fs_results["per_scenario"].items():
        print(f"  {scenario:<20}  false_save_rate = {v['false_save_rate']:.1%}")

    print(f"\n{sep}")
    print("Summary:")
    avg_tf  = sum(r["tf_recovery_rate"]    for r in tf_results.values()) / len(tf_results)
    avg_fn  = sum(r["false_blacklist_rate"] for r in tf_results.values()) / len(tf_results)
    print(f"  TF Recovery (avg across 3 scenarios):  {avg_tf:.1%}  ← higher is better")
    print(f"  False Blacklist (avg):                 {avg_fn:.1%}  ← lower is better (0% ideal)")
    print(f"  TN Precision (after {tn_results['n_runs_per_gene']} rounds):         "
          f"{tn_results['final_tn_rate']:.1%}  ← higher is better")
    print(f"  False Save (avg across scenarios):     "
          f"{sum(v['false_save_rate'] for v in fs_results['per_scenario'].values())/len(fs_results['per_scenario']):.1%}"
          f"  ← cost of conservative approach")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  default="IFNG",
                        choices=["IFNG", "IL2", "Sanchez21", "Steinhart"])
    parser.add_argument("--n-genes",  type=int, default=300,
                        help="genes sampled per condition")
    parser.add_argument("--n-rounds", type=int, default=3,
                        help="rounds simulated per non-hit gene (TN test)")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--json",     action="store_true",
                        help="also print JSON output")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} ...")
    all_genes, hits = load_dataset(args.dataset)
    non_hits  = [g for g in all_genes if g not in hits]
    hit_genes = [g for g in all_genes if g in hits]
    print(f"  {len(all_genes)} genes | {len(hit_genes)} hits ({100*len(hit_genes)/len(all_genes):.1f}%) "
          f"| {len(non_hits)} non-hits")

    tf_results = evaluate_tf_recovery(hit_genes, n_genes=args.n_genes, seed=args.seed)
    tn_results = evaluate_tn_precision(non_hits, n_genes=args.n_genes,
                                       n_runs_per_gene=args.n_rounds, seed=args.seed)
    fs_results = evaluate_tn_false_save(non_hits, n_genes=args.n_genes, seed=args.seed)

    print_report(args.dataset, tf_results, tn_results, fs_results)

    # Boundary cases
    print("\n── D. Boundary / Edge Case Behavior ──")
    boundary = evaluate_boundary_cases(seed=args.seed)
    for b in boundary:
        print(f"  [{b['verdict']:<20} conf={b['confidence']:<6} path={b['path']}]  {b['label']}")

    # No-G3 baseline
    no_g3 = evaluate_no_g3_baseline(hit_genes, n_genes=args.n_genes, seed=args.seed)
    print(f"\n── E. Without G3 (naive baseline) ──")
    print(f"  If every low metric → true_negative (no TF check):")
    print(f"  false_blacklist_rate = {no_g3['false_blacklist_rate_no_g3']:.1%}  "
          f"(vs {sum(r['false_blacklist_rate'] for r in tf_results.values())/len(tf_results):.1%} with G3)")
    print(f"  G3 reduces false blacklisting by "
          f"{no_g3['false_blacklist_rate_no_g3'] - sum(r['false_blacklist_rate'] for r in tf_results.values())/len(tf_results):.1%}")

    if args.json:
        output = {
            "dataset": args.dataset,
            "n_genes": args.n_genes,
            "tf_recovery": tf_results,
            "tn_precision": tn_results,
            "false_save": fs_results,
            "boundary_cases": boundary,
            "no_g3_baseline": no_g3,
        }
        print("\nJSON output:")
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
