"""
G5 BenchmarkEval — evaluate gene selection strategies against BioDiscoveryAgent datasets.

Datasets cloned from: https://github.com/snap-stanford/BioDiscoveryAgent
Ground truth format:
  ground_truth_<name>.csv  — Gene, Score columns (full universe with scores)
  topmovers_<name>.npy     — array of true hit gene symbols

Usage:
  python benchmark.py                          # run all baselines, print table
  python benchmark.py --dataset IFNG           # single dataset
  python benchmark.py --ranker waddington      # use Waddington GeneRanker stub
  python benchmark.py --trials 10 --seed 42   # reproducibility
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# G1 GeneRanker (Phase 1)
sys.path.insert(0, str(Path(__file__).parent))
from gene_ranker import waddington_ranker as _waddington_ranker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "workspace" / "benchmarks" / "biodiscovery_datasets"
CEG_PATH  = REPO_ROOT / "workspace" / "benchmarks" / "CEGv2.txt"

# Fallback: look in the cloned BioDiscoveryAgent repo
BDA_DIR   = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/datasets")
BDA_CEG   = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")

def _resolve_data_dir() -> Path:
    if DATA_DIR.exists():
        return DATA_DIR
    if BDA_DIR.exists():
        return BDA_DIR
    raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR} or {BDA_DIR}")

def _resolve_ceg_path() -> Path:
    if CEG_PATH.exists():
        return CEG_PATH
    if BDA_CEG.exists():
        return BDA_CEG
    raise FileNotFoundError(f"CEGv2.txt not found: {CEG_PATH} or {BDA_CEG}")

# ---------------------------------------------------------------------------
# Dataset registry
# All single-gene datasets from BioDiscoveryAgent (Horlbeck is 2-gene, handled separately)
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "IFNG": {
        "csv": "ground_truth_IFNG.csv",
        "npy": "topmovers_IFNG.npy",
        "batch_size": 128,
        "description": "IFN-γ production in T cells (Schmidt et al. 2022)",
    },
    "IL2": {
        "csv": "ground_truth_IL2.csv",
        "npy": "topmovers_IL2.npy",
        "batch_size": 128,
        "description": "IL-2 production in T cells (Schmidt et al. 2022)",
    },
    "Sanchez21": {
        "csv": "ground_truth_Sanchez21.csv",
        "npy": "topmovers_Sanchez21.npy",
        "batch_size": 128,
        "description": "Neuronal choline recycling (Sanchez et al. 2021)",
    },
    "Sanchez21_down": {
        "csv": "ground_truth_Sanchez21_down.csv",
        "npy": "topmovers_Sanchez21_down.npy",
        "batch_size": 128,
        "description": "Neuronal choline recycling - downregulated (Sanchez et al. 2021)",
    },
    "Carnevale22": {
        "csv": "ground_truth_Carnevale22_Adenosine.csv",
        "npy": "topmovers_Carnevale22_Adenosine.npy",
        "batch_size": 128,
        "description": "CAR-T adenosine signaling (Carnevale et al. 2022)",
    },
    "Scharenberg22": {
        "csv": "ground_truth_Scharenberg22.csv",
        "npy": "topmovers_Scharenberg22.npy",
        "batch_size": 32,
        "description": "T cell proliferation (Scharenberg et al. 2022)",
    },
    "Steinhart": {
        "csv": "ground_truth_Steinhart_crispra_GD2_D22.csv",
        "npy": "topmovers_Steinhart_crispra_GD2_D22.npy",
        "batch_size": 128,
        "description": "CRISPRa GD2 expression (Steinhart et al.)",
    },
}

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_essential_genes() -> set[str]:
    """Load CEGv2 core essential genes — filter these out to avoid false positives."""
    ceg = pd.read_csv(_resolve_ceg_path(), sep="\t")
    return set(ceg["GENE"].dropna().str.strip().str.upper())


def load_dataset(
    name: str,
    filter_essential: bool = True,
) -> tuple[list[str], set[str]]:
    """
    Returns:
        universe: all testable gene symbols (after optional essential-gene filtering)
        hits:     set of true positive gene symbols
    """
    data_dir = _resolve_data_dir()
    cfg = DATASETS[name]

    df = pd.read_csv(data_dir / cfg["csv"])
    gene_col = df.columns[0]
    all_genes = df[gene_col].str.strip().str.upper().tolist()

    raw_hits = np.load(data_dir / cfg["npy"], allow_pickle=True)
    hits = {str(g).strip().upper() for g in raw_hits}

    if filter_essential:
        essential = load_essential_genes()
        all_genes = [g for g in all_genes if g not in essential]
        hits = hits - essential

    return all_genes, hits

# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

RankerFn = Callable[[list[str], list[str], int], list[str]]
"""
Signature: ranker(universe, already_selected, round_number) -> batch of genes to test next.
- universe:          full list of testable genes (essential already filtered)
- already_selected:  genes selected in all previous rounds (cumulative)
- round_number:      1-indexed current round
Returns a list of at most batch_size gene symbols (already_selected excluded automatically).
"""


def run_evaluation(
    ranker: RankerFn,
    dataset_name: str,
    n_rounds: int = 5,
    filter_essential: bool = True,
    seed: int = 0,
) -> dict:
    """
    Simulate n_rounds of sequential gene selection.

    Returns a dict with:
      hit_ratios:  list[float], cumulative hit_ratio after each round
      hits_found:  list[int],   cumulative number of true hits found
      total_hits:  int,         total true hits in the dataset
      auc:         float,       area under the hit_ratio curve
      batch_size:  int
    """
    random.seed(seed)
    np.random.seed(seed)

    universe, hits = load_dataset(dataset_name, filter_essential=filter_essential)
    batch_size = DATASETS[dataset_name]["batch_size"]
    total_hits = len(hits)

    if total_hits == 0:
        raise ValueError(f"No hits found in dataset '{dataset_name}' after filtering")

    selected: list[str] = []
    hit_ratios: list[float] = []
    hits_found: list[int] = []

    for rnd in range(1, n_rounds + 1):
        remaining = [g for g in universe if g not in selected]
        if not remaining:
            break

        batch = ranker(remaining, selected, rnd)
        # Safety: deduplicate and cap at batch_size
        batch = list(dict.fromkeys(g for g in batch if g in set(remaining)))[:batch_size]

        selected.extend(batch)
        n_hits_so_far = len(set(selected) & hits)
        hit_ratios.append(n_hits_so_far / total_hits)
        hits_found.append(n_hits_so_far)

    # AUC: trapezoidal over rounds 1..n_rounds (normalised to [0,1] x-axis)
    auc = float(np.trapezoid(hit_ratios, dx=1.0 / max(len(hit_ratios), 1))
               if hasattr(np, "trapezoid") else
               float(np.trapz(hit_ratios, dx=1.0 / max(len(hit_ratios), 1))))

    return {
        "hit_ratios": hit_ratios,
        "hits_found": hits_found,
        "total_hits": total_hits,
        "auc": auc,
        "batch_size": batch_size,
        "n_rounds": len(hit_ratios),
    }


def run_trials(
    ranker: RankerFn,
    dataset_name: str,
    n_rounds: int = 5,
    n_trials: int = 5,
    filter_essential: bool = True,
) -> dict:
    """Run multiple random seeds and aggregate."""
    all_ratios = []
    all_aucs = []
    for seed in range(n_trials):
        result = run_evaluation(ranker, dataset_name, n_rounds, filter_essential, seed=seed)
        all_ratios.append(result["hit_ratios"])
        all_aucs.append(result["auc"])

    # Pad to equal length
    max_len = max(len(r) for r in all_ratios)
    padded = [r + [r[-1]] * (max_len - len(r)) for r in all_ratios]

    mean_ratios = np.mean(padded, axis=0).tolist()
    std_ratios  = np.std(padded, axis=0).tolist()

    return {
        "mean_hit_ratios": mean_ratios,
        "std_hit_ratios": std_ratios,
        "mean_auc": float(np.mean(all_aucs)),
        "std_auc": float(np.std(all_aucs)),
        "total_hits": run_evaluation(ranker, dataset_name, n_rounds, filter_essential, seed=0)["total_hits"],
        "n_trials": n_trials,
    }

# ---------------------------------------------------------------------------
# Built-in rankers
# ---------------------------------------------------------------------------

def random_ranker(batch_size: int) -> RankerFn:
    """Pure random selection from remaining universe."""
    def _ranker(universe: list[str], already_selected: list[str], round_num: int) -> list[str]:
        pool = [g for g in universe if g not in set(already_selected)]
        return random.sample(pool, min(batch_size, len(pool)))
    return _ranker


def score_guided_ranker(dataset_name: str, batch_size: int) -> RankerFn:
    """
    Oracle upper-bound: selects highest-score genes from ground truth CSV.
    Not realistic — shows theoretical ceiling.
    """
    data_dir = _resolve_data_dir()
    cfg = DATASETS[dataset_name]
    df = pd.read_csv(data_dir / cfg["csv"])
    gene_col, score_col = df.columns[0], df.columns[1]
    df["_gene"] = df[gene_col].str.strip().str.upper()
    # For Sanchez21 (negative scores = hits), invert; others higher = better
    scores_map = dict(zip(df["_gene"], df[score_col]))

    # Auto-detect score direction: hits may have highest OR lowest scores
    hits_raw = np.load(data_dir / cfg["npy"], allow_pickle=True)
    hit_set = {str(g).strip().upper() for g in hits_raw}
    hit_scores   = [scores_map[g] for g in hit_set   if g in scores_map]
    nohit_scores = [scores_map[g] for g in scores_map if g not in hit_set]
    # If mean hit score < mean non-hit score, lower is better
    descending = (np.mean(hit_scores) >= np.mean(nohit_scores)) if hit_scores and nohit_scores else True

    def _ranker(universe: list[str], already_selected: list[str], round_num: int) -> list[str]:
        pool = [(g, scores_map.get(g, 0.0)) for g in universe if g not in set(already_selected)]
        pool_sorted = sorted(pool, key=lambda x: x[1], reverse=descending)
        return [g for g, _ in pool_sorted[:batch_size]]
    return _ranker


def waddington_ranker_stub(batch_size: int) -> RankerFn:
    """Kept for backwards-compat; use waddington_ranker instead."""
    return _waddington_ranker("", batch_size)

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

RANKER_PAPER_RESULTS = {
    # BioDiscoveryAgent Table 1 — mean hit_ratio at round 5 (no essential filtering)
    # Source: Roohani et al. 2025, Table 1 average across 6 datasets
    "BioDiscoveryAgent": {
        "IFNG": 0.096, "IL2": 0.100, "Sanchez21": None,
        "Carnevale22": 0.043, "Scharenberg22": None, "Steinhart": None,
        "avg": 0.128,
    },
    "Random (paper)": {
        "IFNG": 0.037, "IL2": 0.031, "Sanchez21": None,
        "Carnevale22": 0.036, "Scharenberg22": None, "Steinhart": None,
        "avg": 0.055,
    },
}


def fmt(v, decimals=3):
    return f"{v:.{decimals}f}" if v is not None else "  —  "


def run_all(
    n_rounds: int = 5,
    n_trials: int = 5,
    filter_essential: bool = True,
    datasets: list[str] | None = None,
    ranker_name: str = "all",
):
    target_datasets = datasets or list(DATASETS.keys())

    rankers: dict[str, Callable[[str, int], RankerFn]] = {
        "random":      lambda ds, bs: random_ranker(bs),
        "oracle":      lambda ds, bs: score_guided_ranker(ds, bs),
        "waddington":  lambda ds, bs: _waddington_ranker(ds, bs, verbose=True),
    }
    active_rankers = rankers if ranker_name == "all" else {ranker_name: rankers[ranker_name]}

    results: dict[str, dict[str, dict]] = {}

    print(f"\nG5 BenchmarkEval  |  rounds={n_rounds}  trials={n_trials}  filter_essential={filter_essential}")
    print("=" * 90)

    for rname, ranker_factory in active_rankers.items():
        results[rname] = {}
        ratios_final = []
        aucs = []

        for ds in target_datasets:
            bs = DATASETS[ds]["batch_size"]
            try:
                r = run_trials(ranker_factory(ds, bs), ds, n_rounds, n_trials, filter_essential)
                results[rname][ds] = r
                ratios_final.append(r["mean_hit_ratios"][-1])
                aucs.append(r["mean_auc"])
            except Exception as e:
                print(f"  WARNING: {rname}/{ds} failed: {e}")
                results[rname][ds] = None

        # Per-ranker row
        avg_ratio = float(np.mean(ratios_final)) if ratios_final else float("nan")
        avg_auc   = float(np.mean(aucs))          if aucs         else float("nan")
        results[rname]["_avg_ratio"] = avg_ratio
        results[rname]["_avg_auc"]   = avg_auc

    # Print hit_ratio table (round 5)
    header = f"{'Ranker':20s}" + "".join(f"  {ds:12s}" for ds in target_datasets) + f"  {'Avg':8s}  {'Avg AUC':8s}"
    print(f"\nHit ratio at round {n_rounds}  (mean ± std over {n_trials} trials):")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for rname, ds_results in results.items():
        row = f"{rname:20s}"
        for ds in target_datasets:
            r = ds_results.get(ds)
            if r is None:
                row += f"  {'ERR':12s}"
            else:
                val = r["mean_hit_ratios"][-1]
                std = r["std_hit_ratios"][-1]
                row += f"  {val:.3f}±{std:.3f}"
        avg_r = ds_results.get("_avg_ratio", float("nan"))
        avg_a = ds_results.get("_avg_auc", float("nan"))
        row += f"  {avg_r:.3f}     {avg_a:.3f}"
        print(row)

    # Print reference rows from paper
    print()
    print("Reference (from paper, no essential filtering, 1 trial):")
    for rname, ref in RANKER_PAPER_RESULTS.items():
        row = f"{rname:20s}"
        for ds in target_datasets:
            v = ref.get(ds)
            row += f"  {fmt(v):12s}"
        row += f"  {fmt(ref.get('avg'))}"
        print(row)

    print()
    return results


def save_results(results: dict, out_path: Path):
    """Save full results to JSON for later analysis."""
    serialisable = {}
    for rname, ds_dict in results.items():
        serialisable[rname] = {}
        for ds, r in ds_dict.items():
            serialisable[rname][ds] = r
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serialisable, indent=2))
    print(f"Results saved to {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G5 BenchmarkEval")
    parser.add_argument("--dataset",  nargs="+", choices=list(DATASETS), default=None,
                        help="Datasets to evaluate (default: all)")
    parser.add_argument("--ranker",   choices=["all", "random", "oracle", "waddington"],
                        default="all")
    parser.add_argument("--rounds",   type=int, default=5)
    parser.add_argument("--trials",   type=int, default=5)
    parser.add_argument("--no-filter-essential", action="store_true",
                        help="Disable CEGv2 essential gene filtering")
    parser.add_argument("--save",     type=str, default=None,
                        help="Save results JSON to this path")
    args = parser.parse_args()

    results = run_all(
        n_rounds=args.rounds,
        n_trials=args.trials,
        filter_essential=not args.no_filter_essential,
        datasets=args.dataset,
        ranker_name=args.ranker,
    )

    if args.save:
        save_results(results, Path(args.save))
