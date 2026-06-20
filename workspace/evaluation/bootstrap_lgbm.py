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

REPO_ROOT      = Path(__file__).resolve().parents[2]
BDA_DIR        = REPO_ROOT / "workspace" / "data" / "bda_benchmark"
BDA_CEG        = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")
CACHE_DIR      = Path(__file__).parent / "_ppi_cache"
MODEL_DIR      = REPO_ROOT / "workspace" / "models"
DATA_DIR       = REPO_ROOT / "workspace" / "evaluation"
DEPMAP_OUT_DIR    = REPO_ROOT / "workspace" / "data" / "depmap" / "processed"
UNIV_FEAT_CSV     = REPO_ROOT / "workspace" / "data" / "universal_features.csv"
PATHWAY_FEAT_CSV  = REPO_ROOT / "workspace" / "data" / "pathway_features.csv"

sys.path.insert(0, str(Path(__file__).parent))
from gene_ranker import (
    DATASET_ANCHORS, build_anchor_scores, build_archs4_scores,
    compute_ppi_sum_from_cache, build_kegg_scores, LGBMWrapper,
)
from scipy.stats import rankdata as _rankdata

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def load_essential_genes() -> set[str]:
    ceg = pd.read_csv(BDA_CEG, sep="\t")
    return set(ceg["GENE"].dropna().str.strip().str.upper())


def load_pathway_features() -> dict[str, tuple[float, float]]:
    """Load KEGG and Reactome pathway counts. Returns {gene: (kegg_norm, reactome_norm)}."""
    if not PATHWAY_FEAT_CSV.exists():
        print("  [WARNING] pathway_features.csv not found — run prep_pathway_features.py")
        return {}
    df = pd.read_csv(PATHWAY_FEAT_CSV)
    df["gene"] = df["gene"].str.strip().str.upper()
    result = {
        row["gene"]: (float(row["kegg_pathway_count_norm"]),
                      float(row["reactome_pathway_count_norm"]))
        for _, row in df.iterrows()
    }
    print(f"  Pathway features: {len(result)} genes (KEGG count + Reactome count)")
    return result


def load_universal_features() -> dict[str, tuple[float, float]]:
    """Load pLI and STRING degree. Returns {gene: (pli_score, string_degree_norm)}."""
    if not UNIV_FEAT_CSV.exists():
        print("  [WARNING] universal_features.csv not found — run prep_universal_features.py")
        return {}
    df = pd.read_csv(UNIV_FEAT_CSV)
    df["gene"] = df["gene"].str.strip().str.upper()
    result = {
        row["gene"]: (float(row["pli_score"]), float(row["string_degree_norm"]))
        for _, row in df.iterrows()
    }
    print(f"  Universal features: {len(result)} genes (pLI + STRING degree)")
    return result


def compute_hub_scores() -> dict[str, float]:
    """Count-based PPI hub: how many anchor PPI lists mention each gene."""
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


def compute_kegg_scores_for_dataset(
    dataset_name: str,
    genes: list[str],
) -> dict[str, float]:
    """KEGG pathway overlap scores for all genes in a dataset."""
    anchors = DATASET_ANCHORS.get(dataset_name, [])
    if not anchors:
        return {}
    print(f"  [KEGG] Building pathway overlap scores for {dataset_name}: {anchors}")
    return build_kegg_scores(genes, anchors, verbose=True)


def compute_g1_scores_with_anchors(anchors: list[str]) -> dict[str, float]:
    """Build anchor-expansion scores for a dynamic anchor list (uses disk cache)."""
    if not anchors:
        return {}
    return build_anchor_scores(anchors, verbose=False)


def compute_archs4_scores_with_anchors(anchors: list[str], label: str = "") -> dict[str, float]:
    """Build ARCHS4 co-expression scores for a dynamic anchor list."""
    if not anchors:
        return {}
    print(f"  [ARCHS4] Building co-expression scores for {label}: {anchors[:5]}")
    return build_archs4_scores(anchors, verbose=True)


def compute_kegg_scores_with_anchors(
    genes: list[str],
    anchors: list[str],
    label: str = "",
) -> dict[str, float]:
    """KEGG pathway overlap scores for a dynamic anchor list."""
    if not anchors:
        return {}
    print(f"  [KEGG] Building pathway overlap scores for {label}: {anchors[:5]}")
    return build_kegg_scores(genes, anchors, verbose=True)


# Features whose distributions depend on the dataset's anchor genes.
# These are rank-normalised per dataset when --rank-norm is used.
_ANCHOR_FEATS = ["g1_ppi_score", "archs4_coexpr", "kegg_overlap"]


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rank-normalise anchor-specific features to [0, 1] within this DataFrame."""
    df = df.copy()
    n  = len(df)
    for feat in _ANCHOR_FEATS:
        if feat in df.columns:
            r = _rankdata(df[feat].values, method="average") - 1
            df[feat] = r / max(n - 1, 1)
    return df


def build_features(
    genes: list[str],
    g1_scores: dict[str, float],
    hub_scores: dict[str, float],
    archs4_scores: dict[str, float],
    ppi_sum_scores: dict[str, float],
    kegg_scores: dict[str, float] | None = None,
    univ_features: dict[str, tuple[float, float]] | None = None,
    pathway_features: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    _kg  = kegg_scores      or {}
    _uf  = univ_features    or {}
    _pf  = pathway_features or {}
    rows = []
    for gene in genes:
        pli, deg       = _uf.get(gene, (0.0, 0.0))
        kegg_cnt, reac = _pf.get(gene, (0.0, 0.0))
        row: dict = {
            "gene":                      gene,
            "g1_ppi_score":              g1_scores.get(gene, 0.0),
            "hub_score_norm":            hub_scores.get(gene, 0.0),
            "archs4_coexpr":             archs4_scores.get(gene, 0.0),
            "ppi_score_sum":             ppi_sum_scores.get(gene, 0.0),
            "kegg_overlap":              _kg.get(gene, 0.0),
            "pli_score":                 pli,
            "string_degree_norm":        deg,
            "kegg_pathway_count_norm":   kegg_cnt,
            "reactome_pathway_count_norm": reac,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASETS = {
    "IFNG":                   ("ground_truth_IFNG.csv",                      "topmovers_IFNG.npy"),
    "IL2":                    ("ground_truth_IL2.csv",                       "topmovers_IL2.npy"),
    "Sanchez21":              ("ground_truth_Sanchez21.csv",                 "topmovers_Sanchez21.npy"),
    "Sanchez21_down":         ("ground_truth_Sanchez21_down.csv",            "topmovers_Sanchez21_down.npy"),
    "Carnevale22":            ("ground_truth_Carnevale22_Adenosine.csv",     "topmovers_Carnevale22_Adenosine.npy"),
    "Scharenberg22":          ("ground_truth_Scharenberg22.csv",             "topmovers_Scharenberg22.npy"),
    "Steinhart":              ("ground_truth_Steinhart_crispra_GD2_D22.csv", "topmovers_Steinhart_crispra_GD2_D22.npy"),
    "Replogle_K562_essential":("ground_truth_Replogle_K562_essential.csv",   "topmovers_Replogle_K562_essential.npy"),
    "Replogle_K562_gwps":     ("ground_truth_Replogle_K562_gwps.csv",        "topmovers_Replogle_K562_gwps.npy"),
}

BATCH_SIZES = {
    "IFNG": 128, "IL2": 128, "Sanchez21": 128, "Sanchez21_down": 128,
    "Carnevale22": 128, "Scharenberg22": 32, "Steinhart": 128,
    "Replogle_K562_essential": 32, "Replogle_K562_gwps": 128,
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
    "random_state":     42,   # deterministic training
}

FEATURE_COLS = [
    "g1_ppi_score", "hub_score_norm", "archs4_coexpr",
    "ppi_score_sum", "kegg_overlap",
    "pli_score", "string_degree_norm",
    "kegg_pathway_count_norm", "reactome_pathway_count_norm",
]


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
    ppi_sum: dict[str, float],
    use_rank_norm: bool = False,
    univ_features: dict[str, tuple[float, float]] | None = None,
    pathway_features: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """Leave-one-dataset-out evaluation."""
    g1_all = {ds: compute_g1_scores_for_dataset(ds) for ds in all_data}
    results = {}
    for test_ds in all_data:
        train_frames = []
        for ds, df in all_data.items():
            if ds == test_ds:
                continue
            genes = df["gene"].tolist()
            a4 = compute_archs4_scores_for_dataset(ds)
            kg = compute_kegg_scores_for_dataset(ds, genes)
            feats = build_features(genes, g1_all[ds], hub_scores, a4, ppi_sum, kg,
                                   univ_features, pathway_features)
            if use_rank_norm:
                feats = rank_normalize(feats)
            feats["label"] = df["label"].values
            train_frames.append(feats)

        train_df = pd.concat(train_frames, ignore_index=True)
        model = train_model(train_df)

        test_df_raw = all_data[test_ds]
        test_genes = test_df_raw["gene"].tolist()
        a4_test = compute_archs4_scores_for_dataset(test_ds)
        kg_test = compute_kegg_scores_for_dataset(test_ds, test_genes)
        test_feats = build_features(
            test_genes, g1_all[test_ds], hub_scores, a4_test, ppi_sum, kg_test,
            univ_features, pathway_features
        )
        if use_rank_norm:
            test_feats = rank_normalize(test_feats)
        test_feats["label"] = test_df_raw["label"].values
        bs = BATCH_SIZES[test_ds]
        results[test_ds] = evaluate_model(model, test_feats, test_ds, bs)

    return results


# ---------------------------------------------------------------------------
# DepMap dataset loading
# ---------------------------------------------------------------------------

def load_depmap_datasets(
    essential: set[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    """Load DepMap processed cell-line datasets.

    Returns:
      all_data   — {dataset_name: labeled DataFrame}
      anchor_map — {dataset_name: list of anchor gene symbols}
    """
    all_data: dict[str, pd.DataFrame] = {}
    anchor_map: dict[str, list[str]] = {}

    if not DEPMAP_OUT_DIR.exists():
        print(f"  [DepMap] {DEPMAP_OUT_DIR} not found — skipping.")
        return all_data, anchor_map

    for gt_path in sorted(DEPMAP_OUT_DIR.glob("ground_truth_DepMap_*.csv")):
        name = gt_path.stem.replace("ground_truth_", "")          # DepMap_ACH_000106
        tm_path  = DEPMAP_OUT_DIR / f"topmovers_{name}.npy"
        anc_path = DEPMAP_OUT_DIR / f"anchors_{name}.json"

        if not tm_path.exists() or not anc_path.exists():
            continue

        df = pd.read_csv(gt_path)
        gene_col = df.columns[0]
        df["gene"] = df[gene_col].str.strip().str.upper()

        hits = {str(g).strip().upper()
                for g in np.load(tm_path, allow_pickle=True)}
        df["label"]   = df["gene"].isin(hits).astype(int)
        df            = df[~df["gene"].isin(essential)].reset_index(drop=True)
        df["dataset"] = name

        anchors = [g.strip().upper() for g in json.loads(anc_path.read_text())]

        all_data[name]   = df[["gene", "label", "dataset"]]
        anchor_map[name] = anchors

    return all_data, anchor_map


def _precompute_features(
    all_data: dict[str, pd.DataFrame],
    anchor_map_override: dict[str, list[str]],
    hub_scores: dict[str, float],
    ppi_sum: dict[str, float],
    g1_all: dict[str, dict[str, float]],
    univ_features: dict[str, tuple[float, float]] | None = None,
    use_rank_norm: bool = False,
    pathway_features: dict[str, tuple[float, float]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute full feature frames for all datasets (BDA + DepMap) once."""
    frames: dict[str, pd.DataFrame] = {}
    for ds, df in all_data.items():
        genes = df["gene"].tolist()
        if ds in anchor_map_override:
            anchors = anchor_map_override[ds]
            a4 = compute_archs4_scores_with_anchors(anchors, label=ds)
            kg = compute_kegg_scores_with_anchors(genes, anchors, label=ds)
        else:
            a4 = compute_archs4_scores_for_dataset(ds)
            kg = compute_kegg_scores_for_dataset(ds, genes)
        feats = build_features(genes, g1_all[ds], hub_scores, a4, ppi_sum, kg,
                               univ_features, pathway_features)
        if use_rank_norm:
            feats = rank_normalize(feats)
        feats["label"]   = df["label"].values
        feats["dataset"] = ds
        frames[ds] = feats
    return frames


def cross_dataset_eval_bda_with_depmap(
    bda_data: dict[str, pd.DataFrame],
    depmap_data: dict[str, pd.DataFrame],
    bda_frames: dict[str, pd.DataFrame],
    depmap_frames: dict[str, pd.DataFrame],
) -> dict:
    """LOO on BDA datasets, with ALL DepMap datasets as permanent training data."""
    depmap_concat = (
        pd.concat(depmap_frames.values(), ignore_index=True)
        if depmap_frames else pd.DataFrame()
    )
    results = {}
    for test_ds in bda_data:
        train_parts = [
            bda_frames[ds]
            for ds in bda_data
            if ds != test_ds
        ]
        if not depmap_concat.empty:
            train_parts.append(depmap_concat)

        train_df = pd.concat(train_parts, ignore_index=True)
        model    = train_model(train_df)

        test_feats          = bda_frames[test_ds].copy()
        test_feats["label"] = bda_data[test_ds]["label"].values
        bs                  = BATCH_SIZES[test_ds]
        results[test_ds]    = evaluate_model(model, test_feats, test_ds, bs)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(datasets_to_train: list[str] | None = None, eval_only: bool = False,
        use_rank_norm: bool = False, include_depmap: bool = False):
    print("Loading essential genes...")
    essential = load_essential_genes()
    print(f"  CEGv2: {len(essential)} essential genes")

    print("Loading universal features (pLI + STRING degree)...")
    univ_features = load_universal_features()

    print("Loading pathway features (KEGG count + Reactome count)...")
    pathway_features = load_pathway_features()

    # Load BDA datasets
    print("\nLoading BDA datasets...")
    bda_data: dict[str, pd.DataFrame] = {}
    for ds in (datasets_to_train or list(DATASETS.keys())):
        df = load_dataset_labeled(ds, essential)
        bda_data[ds] = df
        print(f"  {ds}: {len(df)} genes, {df['label'].sum()} hits ({100*df['label'].mean():.1f}%)")

    # Optionally load DepMap datasets (training-only, not saved as per-dataset models)
    depmap_data: dict[str, pd.DataFrame] = {}
    depmap_anchors: dict[str, list[str]] = {}
    if include_depmap:
        print("\nLoading DepMap cell-line datasets...")
        depmap_data, depmap_anchors = load_depmap_datasets(essential)
        print(f"  Loaded {len(depmap_data)} DepMap datasets")
        for ds, df in list(depmap_data.items())[:5]:
            print(f"  {ds}: {len(df)} genes, {df['label'].sum()} hits ({100*df['label'].mean():.1f}%)")
        if len(depmap_data) > 5:
            print(f"  ... ({len(depmap_data) - 5} more)")

    all_data = {**bda_data, **depmap_data}

    # Pre-fetch all g1 (PPI) scores first — this populates _ppi_cache for ALL datasets.
    # hub_scores and ppi_sum must be computed AFTER this loop so they reflect
    # the complete cache (including any newly added dataset anchors).
    print("\nPre-fetching STRING PPI for all dataset anchors...")
    g1_all: dict[str, dict[str, float]] = {}
    for ds in bda_data:
        g1_all[ds] = compute_g1_scores_for_dataset(ds)
    for ds, anchors in depmap_anchors.items():
        print(f"  {ds}: anchors={anchors[:4]}")
        g1_all[ds] = compute_g1_scores_with_anchors(anchors)

    print("Computing hub scores from PPI cache (post-fetch)...")
    hub_scores = compute_hub_scores()
    print(f"  Hub scores: {len(hub_scores)} genes scored")

    print("Computing PPI sum scores from cache (post-fetch)...")
    ppi_sum = compute_ppi_sum_from_cache()
    print(f"  PPI sum scores: {len(ppi_sum)} genes scored")

    # Build features for BDA datasets
    rn_tag = " + rank-norm" if use_rank_norm else ""
    print(f"\nBuilding features (ARCHS4, ppi_sum, KEGG from cache{rn_tag})...")
    bda_frames   = _precompute_features(
        bda_data, {}, hub_scores, ppi_sum, g1_all,
        univ_features=univ_features, use_rank_norm=use_rank_norm,
        pathway_features=pathway_features,
    )
    depmap_frames = _precompute_features(
        depmap_data, depmap_anchors, hub_scores, ppi_sum, g1_all,
        univ_features=univ_features, use_rank_norm=use_rank_norm,
        pathway_features=pathway_features,
    )
    feature_frames = {**bda_frames, **depmap_frames}

    # Save training data CSV (BDA only — DepMap would be huge)
    csv_path = DATA_DIR / "lgbm_training_data.csv"
    pd.concat(bda_frames.values(), ignore_index=True).to_csv(csv_path, index=False)
    print(f"\nBDA training data saved to {csv_path}")

    if eval_only:
        print("\n[eval-only] Skipping model training.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Per-dataset LightGBM training + evaluation (BDA only)")
    print("=" * 70)

    per_dataset_results: dict[str, dict] = {}
    for ds, df_feat in bda_frames.items():
        model  = train_model(df_feat)
        bs     = BATCH_SIZES[ds]
        result = evaluate_model(model, df_feat, ds, bs)
        per_dataset_results[ds] = result

        to_save    = LGBMWrapper(model, rank_norm=use_rank_norm) if use_rank_norm else model
        model_path = MODEL_DIR / f"lgbm_{ds}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(to_save, f)

        hr5 = result["hit_ratios"][-1] if result["hit_ratios"] else float("nan")
        print(f"  {ds:20s}  AUC-ROC={result['auc_roc']:.3f}  hit_ratio@R5={hr5:.3f}")

    # Cross-dataset model (trained on BDA + DepMap if available)
    tag = "+DepMap" if include_depmap else "BDA"
    print(f"\nTraining cross-dataset model ({tag})...")
    all_feats     = pd.concat(feature_frames.values(), ignore_index=True)
    cross_model   = train_model(all_feats)
    to_save_cross = LGBMWrapper(cross_model, rank_norm=use_rank_norm) if use_rank_norm else cross_model
    cross_model_path = MODEL_DIR / "lgbm_cross_dataset.pkl"
    with open(cross_model_path, "wb") as f:
        pickle.dump(to_save_cross, f)
    print(f"  Saved to {cross_model_path}")

    # Feature importances
    print("\nFeature importances (cross-dataset model):")
    for feat, imp in zip(FEATURE_COLS, cross_model.feature_importances_):
        print(f"  {feat:25s}  {imp}")

    # Leave-one-dataset-out on BDA datasets
    if len(bda_data) >= 3:
        # Standard BDA-only LOO
        print(f"\nLeave-one-dataset-out AUC — BDA only (generalization check{rn_tag}):")
        loo_bda = cross_dataset_eval(bda_data, hub_scores, ppi_sum, use_rank_norm,
                                     univ_features=univ_features,
                                     pathway_features=pathway_features)
        for ds, r in loo_bda.items():
            hr5 = r["hit_ratios"][-1] if r["hit_ratios"] else float("nan")
            print(f"  {ds:20s}  AUC-ROC={r['auc_roc']:.3f}  hit_ratio@R5={hr5:.3f}")

        if include_depmap and depmap_data:
            # LOO on BDA datasets with DepMap as permanent training data
            print(f"\nLeave-one-dataset-out AUC — BDA + DepMap (generalization check{rn_tag}):")
            loo_with_dm = cross_dataset_eval_bda_with_depmap(
                bda_data, depmap_data, bda_frames, depmap_frames
            )
            # Print comparison table
            print(f"\n  {'Dataset':22s}  {'BDA-only hit@R5':>18s}  {'BDA+DepMap hit@R5':>18s}  {'Delta':>8s}")
            print(f"  {'-'*72}")
            bda_avg    = 0.0
            dm_avg     = 0.0
            n          = len(loo_bda)
            for ds in loo_bda:
                hr5_bda = loo_bda[ds]["hit_ratios"][-1] if loo_bda[ds]["hit_ratios"] else 0.0
                hr5_dm  = loo_with_dm[ds]["hit_ratios"][-1] if loo_with_dm[ds]["hit_ratios"] else 0.0
                delta   = hr5_dm - hr5_bda
                bda_avg += hr5_bda
                dm_avg  += hr5_dm
                sign    = "+" if delta >= 0 else ""
                print(f"  {ds:22s}  {hr5_bda:18.3f}  {hr5_dm:18.3f}  {sign}{delta:.3f}")
            bda_avg /= n
            dm_avg  /= n
            delta_avg = dm_avg - bda_avg
            sign      = "+" if delta_avg >= 0 else ""
            print(f"  {'-'*72}")
            print(f"  {'Average':22s}  {bda_avg:18.3f}  {dm_avg:18.3f}  {sign}{delta_avg:.3f}")

    print("\nDone. Models saved to", MODEL_DIR)
    return per_dataset_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", choices=list(DATASETS), default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--rank-norm", dest="rank_norm", action="store_true",
                        help="Enable per-dataset rank normalisation of anchor features (10.3 experiment)")
    parser.add_argument("--include-depmap", dest="include_depmap", action="store_true",
                        help="Add 50 DepMap cell-line datasets as additional training data (10.3 LOO improvement)")
    parser.set_defaults(rank_norm=False, include_depmap=False)
    args = parser.parse_args()
    run(args.dataset, eval_only=args.eval_only, use_rank_norm=args.rank_norm,
        include_depmap=args.include_depmap)
