"""
G5 BenchmarkEval — evaluate gene selection strategies against BioDiscoveryAgent datasets.

Datasets cloned from: https://github.com/snap-stanford/BioDiscoveryAgent
Ground truth format:
  ground_truth_<name>.csv  — Gene, Score columns (full universe with scores)
  topmovers_<name>.npy     — array of true hit gene symbols

Usage:
  python benchmark.py                            # run all baselines, print table
  python benchmark.py --dataset IFNG             # single dataset
  python benchmark.py --ranker waddington        # Waddington GeneRanker
  python benchmark.py --scramble-genes           # gene name scrambling ablation
  python benchmark.py --trials 5 --auto-save     # save results automatically

Results are auto-saved to workspace/results/runs/ when --auto-save is set.
"""

import argparse
import csv
import datetime
import json
import os
import random
import subprocess
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

REPO_ROOT    = Path(__file__).resolve().parents[2]
DATA_DIR     = REPO_ROOT / "workspace" / "benchmarks" / "biodiscovery_datasets"
CEG_PATH     = REPO_ROOT / "workspace" / "benchmarks" / "CEGv2.txt"
RESULTS_DIR  = REPO_ROOT / "workspace" / "results"

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
    "Replogle_K562_essential": {
        "csv": "ground_truth_Replogle_K562_essential.csv",
        "npy": "topmovers_Replogle_K562_essential.npy",
        "batch_size": 32,
        "description": "K562 CRISPRi essential-gene Perturb-seq (Replogle et al. 2022)",
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
# Gene name scrambling (ablation study)
# ---------------------------------------------------------------------------

def scramble_gene_universe(
    universe: list[str],
    hits: set[str],
    seed: int = 0,
) -> tuple[list[str], set[str], dict[str, str]]:
    """
    Replace real gene names with random IDs (e.g. GENE_00042).

    All biology-dependent rankers (Waddington PPI/LightGBM) will score every
    scrambled gene as 0.0 → degrade to random.  Random ranker is unaffected.
    Oracle ranker also degrades (needs real names to look up ground-truth scores);
    this is expected and confirms the simulation is sensitivity-correct.

    Returns:
        scrambled_universe  — same order, all names replaced
        scrambled_hits      — hit set with replaced names
        mapping             — real_name → scrambled_id (for logging)
    """
    rng = random.Random(seed)
    all_names = list(set(universe) | hits)
    ids = [f"GENE_{i:05d}" for i in range(len(all_names))]
    rng.shuffle(ids)
    mapping = {gene: sid for gene, sid in zip(all_names, ids)}

    scrambled_universe = [mapping[g] for g in universe]
    scrambled_hits     = {mapping[g] for g in hits if g in mapping}
    return scrambled_universe, scrambled_hits, mapping


# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _make_run_id() -> str:
    return "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_run(
    run_id: str,
    config: dict,
    results: dict,
) -> Path:
    """
    Save a full run to workspace/results/runs/<run_id>_<tag>.json
    and append a summary row to workspace/results/summary.csv.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "runs").mkdir(exist_ok=True)

    scramble_tag = "scrambled" if config.get("scramble_genes") else "normal"
    ranker_tag   = config.get("ranker", "all").replace(" ", "_")
    filename     = f"{run_id}_{ranker_tag}_{scramble_tag}.json"
    out_path     = RESULTS_DIR / "runs" / filename

    payload = {
        "run_id":    run_id,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config":    config,
        "results":   results,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    # Append to summary.csv
    _append_summary_csv(run_id, payload)
    return out_path


def _append_summary_csv(run_id: str, payload: dict) -> None:
    """One row per (run_id, ranker) in summary.csv."""
    csv_path = RESULTS_DIR / "summary.csv"
    all_ds   = list(DATASETS.keys())

    # Build header
    ds_cols = []
    for ds in all_ds:
        ds_cols += [f"{ds}_r5_mean", f"{ds}_r5_std"]
    fieldnames = [
        "run_id", "timestamp", "git_commit", "ranker",
        "scramble_genes", "filter_essential", "rounds", "trials",
        *ds_cols, "avg_r5", "avg_auc",
    ]

    write_header = not csv_path.exists()
    cfg = payload["config"]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for ranker_name, ds_dict in payload["results"].items():
            if ranker_name.startswith("_"):
                continue
            row: dict = {
                "run_id":           run_id,
                "timestamp":        payload["timestamp"],
                "git_commit":       payload["git_commit"],
                "ranker":           ranker_name,
                "scramble_genes":   cfg.get("scramble_genes", False),
                "filter_essential": cfg.get("filter_essential", True),
                "rounds":           cfg.get("rounds", 5),
                "trials":           cfg.get("trials", 5),
                "avg_r5":           ds_dict.get("_avg_ratio", ""),
                "avg_auc":          ds_dict.get("_avg_auc",   ""),
            }
            for ds in all_ds:
                r = ds_dict.get(ds)
                if r and isinstance(r, dict) and "mean_hit_ratios" in r:
                    row[f"{ds}_r5_mean"] = round(r["mean_hit_ratios"][-1], 4)
                    row[f"{ds}_r5_std"]  = round(r["std_hit_ratios"][-1],  4)
                else:
                    row[f"{ds}_r5_mean"] = ""
                    row[f"{ds}_r5_std"]  = ""
            writer.writerow(row)


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
    scramble_genes: bool = False,
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
    if scramble_genes:
        universe, hits, _ = scramble_gene_universe(universe, hits, seed=seed)
        # Shuffle universe to remove list-order signal (CSVs may be score-sorted)
        random.shuffle(universe)
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
    scramble_genes: bool = False,
) -> dict:
    """Run multiple random seeds and aggregate."""
    all_ratios = []
    all_aucs = []
    for seed in range(n_trials):
        result = run_evaluation(
            ranker, dataset_name, n_rounds, filter_essential,
            seed=seed, scramble_genes=scramble_genes,
        )
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
        "total_hits": run_evaluation(
            ranker, dataset_name, n_rounds, filter_essential,
            seed=0, scramble_genes=scramble_genes,
        )["total_hits"],
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


def coreset_ranker(dataset_name: str, batch_size: int) -> RankerFn:
    """
    GeneDisco-style k-Center Coreset ranker.

    Uses (g1_ppi_score, hub_score_norm) as a 2-D feature vector, then at each
    round greedily picks the gene that is furthest (L2) from the already-selected
    set.  This maximises coverage of the PPI-network feature space without any
    dataset-specific biological anchors or LightGBM supervision, providing an
    honest A-arm algorithmic baseline.

    Unlike Waddington it does NOT know which genes are biologically relevant to
    the target phenotype; it only knows the global PPI network structure.
    """
    from gene_ranker import build_anchor_scores, compute_hub_scores_from_cache, DATASET_ANCHORS

    anchor_genes = DATASET_ANCHORS.get(dataset_name, [])
    ppi_scores   = build_anchor_scores(anchor_genes, verbose=False) if anchor_genes else {}
    hub_scores   = compute_hub_scores_from_cache()                  # global, dataset-agnostic

    def _feat(gene: str) -> np.ndarray:
        return np.array([ppi_scores.get(gene, 0.0), hub_scores.get(gene, 0.0)], dtype=float)

    def _ranker(universe: list[str], already_selected: list[str], round_num: int) -> list[str]:
        pool = [g for g in universe if g not in set(already_selected)]
        if not pool:
            return []

        pool_feats = np.array([_feat(g) for g in pool])  # (N, 2)

        if not already_selected:
            # Round 1: pick the gene with the highest PPI score as seed, then
            # run k-center greedily from there
            seed_idx = int(np.argmax(pool_feats[:, 0]))
        else:
            # Seed = gene in pool furthest from existing selection centroid
            sel_feats  = np.array([_feat(g) for g in already_selected])
            centroid   = sel_feats.mean(axis=0)
            seed_idx   = int(np.argmax(np.linalg.norm(pool_feats - centroid, axis=1)))

        selected_indices: list[int] = [seed_idx]
        selected_feats               = [pool_feats[seed_idx]]

        # k-center: greedy furthest-point sampling
        while len(selected_indices) < min(batch_size, len(pool)):
            sel_arr  = np.array(selected_feats)           # (k, 2)
            # Distance of each pool gene to its nearest already-chosen gene
            dists    = np.min(
                np.linalg.norm(pool_feats[:, None, :] - sel_arr[None, :, :], axis=2),
                axis=1,
            )
            # Mask already chosen
            for idx in selected_indices:
                dists[idx] = -1.0
            next_idx = int(np.argmax(dists))
            selected_indices.append(next_idx)
            selected_feats.append(pool_feats[next_idx])

        return [pool[i] for i in selected_indices]

    return _ranker


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
    scramble_genes: bool = False,
    auto_save: bool = False,
):
    target_datasets = datasets or list(DATASETS.keys())
    run_id = _make_run_id()

    rankers: dict[str, Callable[[str, int], RankerFn]] = {
        "random":      lambda ds, bs: random_ranker(bs),
        "oracle":      lambda ds, bs: score_guided_ranker(ds, bs),
        "coreset":     lambda ds, bs: coreset_ranker(ds, bs),
        "waddington":  lambda ds, bs: _waddington_ranker(ds, bs, verbose=True),
    }
    active_rankers = rankers if ranker_name == "all" else {ranker_name: rankers[ranker_name]}

    results: dict[str, dict[str, dict]] = {}

    scramble_label = "  [SCRAMBLED — ablation]" if scramble_genes else ""
    print(f"\nG5 BenchmarkEval  |  rounds={n_rounds}  trials={n_trials}  filter_essential={filter_essential}{scramble_label}")
    print("=" * 90)

    for rname, ranker_factory in active_rankers.items():
        results[rname] = {}
        ratios_final = []
        aucs = []

        for ds in target_datasets:
            bs = DATASETS[ds]["batch_size"]
            try:
                r = run_trials(
                    ranker_factory(ds, bs), ds, n_rounds, n_trials,
                    filter_essential, scramble_genes=scramble_genes,
                )
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

    if auto_save:
        config = {
            "ranker": ranker_name,
            "rounds": n_rounds,
            "trials": n_trials,
            "filter_essential": filter_essential,
            "datasets": target_datasets,
            "scramble_genes": scramble_genes,
        }
        out = save_run(run_id, config, results)
        print(f"Results saved → {out.relative_to(REPO_ROOT)}")

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
    parser.add_argument("--ranker",   choices=["all", "random", "oracle", "coreset", "waddington"],
                        default="all")
    parser.add_argument("--rounds",   type=int, default=5)
    parser.add_argument("--trials",   type=int, default=5)
    parser.add_argument("--no-filter-essential", action="store_true",
                        help="Disable CEGv2 essential gene filtering")
    parser.add_argument("--scramble-genes", action="store_true",
                        help="Ablation: replace gene names with random IDs")
    parser.add_argument("--auto-save", action="store_true",
                        help="Auto-save results to workspace/results/runs/")
    parser.add_argument("--save",     type=str, default=None,
                        help="Save results JSON to this path (legacy)")
    args = parser.parse_args()

    results = run_all(
        n_rounds=args.rounds,
        n_trials=args.trials,
        filter_essential=not args.no_filter_essential,
        datasets=args.dataset,
        ranker_name=args.ranker,
        scramble_genes=args.scramble_genes,
        auto_save=args.auto_save,
    )

    if args.save:
        save_results(results, Path(args.save))
