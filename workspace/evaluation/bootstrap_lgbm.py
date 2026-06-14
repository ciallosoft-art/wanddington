#!/usr/bin/env python3
"""
G1 Phase 2 — Bootstrap LightGBM ranker from BioDiscoveryAgent benchmark data.

Workflow:
  1. For every gene in every BDA dataset, compute pre-experiment features:
       g1_ppi_score   — PPI distance to biological anchor genes (gene_ranker Phase 1)
       hub_score_norm — how many anchor-gene PPI lists mention this gene (network centrality)
       is_essential   — whether gene is in CEGv2 core essential list
  2. Label each gene as hit (1) or non-hit (0) using topmovers ground truth.
  3. Train one LightGBM classifier per dataset (+ one cross-dataset model).
  4. Evaluate with AUC-ROC and simulated hit_ratio.
  5. Save models to workspace/models/*.pkl for use by gene_ranker.py Phase 2.

Usage:
  conda run -n waddington-bio python3 bootstrap_lgbm.py
  conda run -n waddington-bio python3 bootstrap_lgbm.py --dataset IFNG
  conda run -n waddington-bio python3 bootstrap_lgbm.py --eval-only  # skip training
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).resolve().parents[2]
BDA_DIR    = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/datasets")
BDA_CEG    = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")
CACHE_DIR  = Path(__file__).parent / "_ppi_cache"
MODEL_DIR  = REPO_ROOT / "workspace" / "models"
DATA_DIR   = REPO_ROOT / "workspace" / "evaluation"

sys.path.insert(0, str(Path(__file__).parent))
from gene_ranker import DATASET_ANCHORS, build_anchor_scores, build_archs4_scores

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def load_essential_genes() -> set[str]:
    ceg = pd.read_csv(BDA_CEG, sep="\t")
    return set(ceg["GENE"].dropna().str.strip().str.upper())


def compute_hub_scores() -> dict[str, float]:
    """
    Inverted PPI: for each gene, count how many anchor-gene PPI lists mention it.
    Normalised to [0,1] by max count.
    Genes that appear in many anchor neighborhoods = highly connected hubs.
    """
    counts: dict[str, int] = {}
    for f in CACHE_DIR.glob("*.json"):
        for g in json.loads(f.read_text()):
            counts[g] = counts.get(g, 0) + 1
    if not counts:
        return {}
    max_count = max(counts.values())
    return {g: round(c / max_count, 4) for g, c in counts.items()}


def compute_g1_scores_for_dataset(dataset_name: str) -> dict[str, float]:
    """Build anchor-expansion scores for a dataset (uses disk cache)."""
    anchors = DATASET_ANCHORS.get(dataset_name, [])
    if not anchors:
        return {}
    return build_anchor_scores(anchors, verbose=False)


def compute_archs4_scores_for_dataset(dataset_name: str) -> dict[str, float]:
    """Build ARCHS4 co-expression scores anchored to dataset anchor genes."""
    anchors = DATASET_ANCHORS.get(dataset_name, [])
    if not anchors:
        return {}
    print(f"  [ARCHS4] Building co-expression scores for {dataset_name}: {anchors}")
    return build_archs4_scores(anchors, verbose=True)


def build_features(
    genes: list[str],
    g1_scores: dict[str, float],
    hub_scores: dict[str, float],
    essential: set[str],
    archs4_scores: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for gene in genes:
        row: dict = {
            "gene":           gene,
            "g1_ppi_score":   g1_scores.get(gene, 0.0),
            "hub_score_norm": hub_scores.get(gene, 0.0),
            "is_essential":   int(gene in essential),
            "archs4_coexpr":  archs4_scores.get(gene, 0.0) if archs4_scores else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASETS = {
    "IFNG":          ("ground_truth_IFNG.csv",                    "topmovers_IFNG.npy"),
    "IL2":           ("ground_truth_IL2.csv",                     "topmovers_IL2.npy"),
    "Sanchez21":     ("ground_truth_Sanchez21.csv",               "topmovers_Sanchez21.npy"),
    "Sanchez21_down":("ground_truth_Sanchez21_down.csv",          "topmovers_Sanchez21_down.npy"),
    "Carnevale22":   ("ground_truth_Carnevale22_Adenosine.csv",   "topmovers_Carnevale22_Adenosine.npy"),
    "Scharenberg22": ("ground_truth_Scharenberg22.csv",           "topmovers_Scharenberg22.npy"),
    "Steinhart":     ("ground_truth_Steinhart_crispra_GD2_D22.csv", "topmovers_Steinhart_crispra_GD2_D22.npy"),
}

BATCH_SIZES = {
    "IFNG": 128, "IL2": 128, "Sanchez21": 128, "Sanchez21_down": 128,
    "Carnevale22": 128, "Scharenberg22": 32, "Steinhart": 128,
}


def load_dataset_labeled(name: str, essential: set[str]) -> pd.DataFrame:
    csv_name, npy_name = DATASETS[name]
    df = pd.read_csv(BDA_DIR / csv_name)
    gene_col = df.columns[0]
    df["gene"] = df[gene_col].str.strip().str.upper()
    hits = {str(g).strip().upper() for g in np.load(BDA_DIR / npy_name, allow_pickle=True)}
    df["label"] = df["gene"].isin(hits).astype(int)
    # Filter essential genes (BDA convention)
    df = df[~df["gene"].isin(essential)].reset_index(drop=True)
    df["dataset"] = name
    return df[["gene", "label", "dataset"]]


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "learning_rate":    0.05,
    "n_estimators":     300,
    "num_leaves":       31,
    "min_child_samples": 20,
    "verbose":          -1,
    "n_jobs":           -1,
}

FEATURE_COLS = ["g1_ppi_score", "hub_score_norm", "is_essential", "archs4_coexpr"]


def train_model(df_feat: pd.DataFrame) -> lgb.LGBMClassifier:
    """Train a LightGBM binary classifier on the feature DataFrame."""
    X = df_feat[FEATURE_COLS].values
    y = df_feat["label"].values
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model = lgb.LGBMClassifier(**LGBM_PARAMS, scale_pos_weight=pos_weight)
    model.fit(X, y)
    return model


def evaluate_model(
    model: lgb.LGBMClassifier,
    df_feat: pd.DataFrame,
    dataset_name: str,
    batch_size: int,
    n_rounds: int = 5,
) -> dict:
    """Simulate sequential selection using model probabilities."""
    X = df_feat[[c for c in FEATURE_COLS if c in df_feat.columns]].values
    y = df_feat["label"].values
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)

    # Simulate hit_ratio over n_rounds (static ranking)
    order = np.argsort(-probs)
    total_hits = y.sum()
    hit_ratios = []
    for r in range(1, n_rounds + 1):
        selected_idx = order[: r * batch_size]
        found = y[selected_idx].sum()
        hit_ratios.append(round(found / total_hits, 4))

    return {"auc_roc": round(auc, 4), "hit_ratios": hit_ratios}


# ---------------------------------------------------------------------------
# Cross-dataset evaluation (leave-one-out)
# ---------------------------------------------------------------------------

def cross_dataset_eval(
    all_data: dict[str, pd.DataFrame],
    hub_scores: dict[str, float],
    essential: set[str],
) -> dict:
    """Leave-one-dataset-out evaluation."""
    results = {}
    for test_ds in all_data:
        train_frames = []
        for ds, df in all_data.items():
            if ds == test_ds:
                continue
            g1 = compute_g1_scores_for_dataset(ds)
            a4 = compute_archs4_scores_for_dataset(ds)
            feats = build_features(df["gene"].tolist(), g1, hub_scores, essential, a4)
            feats["label"] = df["label"].values
            train_frames.append(feats)

        train_df = pd.concat(train_frames, ignore_index=True)
        model = train_model(train_df)

        test_df_raw = all_data[test_ds]
        g1_test = compute_g1_scores_for_dataset(test_ds)
        a4_test = compute_archs4_scores_for_dataset(test_ds)
        test_feats = build_features(test_df_raw["gene"].tolist(), g1_test, hub_scores, essential, a4_test)
        test_feats["label"] = test_df_raw["label"].values
        bs = BATCH_SIZES[test_ds]
        results[test_ds] = evaluate_model(model, test_feats, test_ds, bs)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(datasets_to_train: list[str] | None = None, eval_only: bool = False):
    print("Loading essential genes...")
    essential = load_essential_genes()
    print(f"  CEGv2: {len(essential)} essential genes")

    print("Computing hub scores from PPI cache...")
    hub_scores = compute_hub_scores()
    print(f"  Hub scores: {len(hub_scores)} genes scored")

    # Load all datasets
    print("\nLoading datasets...")
    all_data: dict[str, pd.DataFrame] = {}
    for ds in (datasets_to_train or list(DATASETS.keys())):
        df = load_dataset_labeled(ds, essential)
        all_data[ds] = df
        print(f"  {ds}: {len(df)} genes, {df['label'].sum()} hits ({100*df['label'].mean():.1f}%)")

    # Build features per dataset (includes ARCHS4 co-expression)
    print("\nBuilding features (ARCHS4 queries will be cached after first run)...")
    feature_frames: dict[str, pd.DataFrame] = {}
    for ds, df in all_data.items():
        g1    = compute_g1_scores_for_dataset(ds)
        a4    = compute_archs4_scores_for_dataset(ds)
        feats = build_features(df["gene"].tolist(), g1, hub_scores, essential, a4)
        feats["label"]   = df["label"].values
        feats["dataset"] = ds
        feature_frames[ds] = feats

    # Save training data CSV for inspection
    csv_path = DATA_DIR / "lgbm_training_data.csv"
    pd.concat(feature_frames.values(), ignore_index=True).to_csv(csv_path, index=False)
    print(f"\nTraining data saved to {csv_path}")

    if eval_only:
        print("\n[eval-only] Skipping model training.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Per-dataset LightGBM training + evaluation")
    print("=" * 70)

    per_dataset_results: dict[str, dict] = {}
    for ds, df_feat in feature_frames.items():
        model = train_model(df_feat)
        bs = BATCH_SIZES[ds]
        result = evaluate_model(model, df_feat, ds, bs)
        per_dataset_results[ds] = result

        # Save model
        model_path = MODEL_DIR / f"lgbm_{ds}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        hr5 = result["hit_ratios"][-1] if result["hit_ratios"] else float("nan")
        print(f"  {ds:20s}  AUC-ROC={result['auc_roc']:.3f}  hit_ratio@R5={hr5:.3f}")

    # Cross-dataset model (trained on all datasets combined)
    print("\nTraining cross-dataset model...")
    all_feats = pd.concat(feature_frames.values(), ignore_index=True)
    cross_model = train_model(all_feats)
    cross_model_path = MODEL_DIR / "lgbm_cross_dataset.pkl"
    with open(cross_model_path, "wb") as f:
        pickle.dump(cross_model, f)
    print(f"  Saved to {cross_model_path}")

    # Feature importances
    print("\nFeature importances (cross-dataset model):")
    for feat, imp in zip(FEATURE_COLS, cross_model.feature_importances_):
        print(f"  {feat:25s}  {imp}")

    # Leave-one-dataset-out generalization
    if len(all_data) >= 3:
        print("\nLeave-one-dataset-out AUC (generalization check):")
        loo_results = cross_dataset_eval(all_data, hub_scores, essential)
        for ds, r in loo_results.items():
            hr5 = r["hit_ratios"][-1] if r["hit_ratios"] else float("nan")
            print(f"  {ds:20s}  AUC-ROC={r['auc_roc']:.3f}  hit_ratio@R5={hr5:.3f}")

    print("\nDone. Models saved to", MODEL_DIR)
    return per_dataset_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", choices=list(DATASETS), default=None)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    run(args.dataset, eval_only=args.eval_only)
