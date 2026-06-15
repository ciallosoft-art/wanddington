#!/usr/bin/env python3
"""
G1 GeneRanker — Phase 1 + Phase 2.

Phase 1 (rule-based): STRING PPI expansion from biological anchor genes.
Phase 2 (LightGBM):   Loaded from workspace/models/lgbm_{dataset}.pkl if available.
  - Features: g1_ppi_score, hub_score_norm, is_essential
  - Trained via bootstrap_lgbm.py on BioDiscoveryAgent benchmark data
  - Automatically activates when model file exists for the dataset

Integration:
  from gene_ranker import waddington_ranker
  ranker = waddington_ranker("IFNG", batch_size=128)
  batch  = ranker(universe, already_selected, round_num)
"""

import json
import pickle
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Optional

RankerFn = Callable[[list[str], list[str], int], list[str]]

CACHE_DIR        = Path(__file__).parent / "_ppi_cache"
ARCHS4_CACHE_DIR = Path(__file__).parent / "_archs4_cache"
MODEL_DIR        = Path(__file__).resolve().parents[2] / "workspace" / "models"
CEG_PATH   = Path(__file__).resolve().parents[2] / "workspace" / "benchmarks" / "CEGv2.txt"
BDA_CEG    = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")

STRING_BASE = "https://string-db.org/api/json"
DEFAULT_SPECIES = 9606  # Homo sapiens
STRING_LIMIT = 200
STRING_TIMEOUT = 15

# ---------------------------------------------------------------------------
# Biological anchor genes per dataset
#
# Anchor choice rationale:
#  IFNG / IL2: cytokine production screens → hits are TCR negative regulators
#              (CBLB, MAP4K1, CD5) and downstream effectors. Anchors = TCR
#              pathway components known from literature to be relevant.
#  Carnevale22: adenosine-mediated CAR-T suppression → hits are genes in
#               cAMP/PKA pathway that relay ADORA2A signal.
#  Sanchez21*:  neuronal choline recycling → hits cluster around vesicle
#               trafficking and transport.
#  Scharenberg22: T cell proliferation → hits include autophagy, lipid
#                metabolism, ER stress genes (SPNS1, ATG9A, EIF2AK3, LDLR).
#  Steinhart:   CRISPRa GD2 expression → ganglioside synthesis pathway.
# ---------------------------------------------------------------------------

DATASET_ANCHORS: dict[str, list[str]] = {
    "IFNG": [
        "ZAP70", "LCK", "LAT", "PLCG1", "VAV1",        # TCR proximal
        "CBLB", "MAP4K1", "PTPN6", "CD5", "NFKB2",     # negative regulators (known checkpoints)
        "RNF20", "RNF40",                                # ubiquitin ligases (top hits from screen)
    ],
    "IL2": [
        "ZAP70", "LCK", "LAT", "PLCG1", "VAV1",
        "CBLB", "MAP4K1", "PTPN6", "CD5",
        "IL2", "IL2RA", "IL2RB",
    ],
    "Sanchez21": [
        "SLC5A7", "CHAT", "SLC18A3",                    # choline transport/synthesis
        "VPS33A", "RAB7A", "RAB14",                     # vesicle trafficking (top hits)
        "STAU1", "UBA5",                                 # RNA processing / UBL
    ],
    "Sanchez21_down": [
        "SLC5A7", "CHAT", "SLC18A3",
        "VPS33A", "RAB7A", "RAB14",
        "STAU1", "UBA5",
    ],
    "Carnevale22": [
        "ADORA2A", "ENTPD1",                             # adenosine receptor + ecto-nucleotidase
        "PRKAR1A", "PRKAR2A", "PRKACA",                  # PKA regulatory/catalytic subunits
        "ATP6V0D1",                                      # ATPase (top hit)
    ],
    "Scharenberg22": [
        "SPNS1", "S1PR1",                                # sphingosine transport
        "ATG9A", "BECN1", "SQSTM1",                     # autophagy
        "EIF2AK3",                                       # ER stress kinase (top hit)
        "LDLR", "NPC1",                                  # cholesterol / lipid metabolism
    ],
    "Steinhart": [
        "B4GALNT1", "ST8SIA1", "B3GALT4",               # ganglioside synthesis
        "UGCG", "SMPD1",                                 # ceramide/glycolipid metabolism
        "AXIN2", "BATF",                                 # top hits from screen
    ],
}

# ---------------------------------------------------------------------------
# STRING PPI with disk cache
# ---------------------------------------------------------------------------

def _cache_path(gene: str) -> Path:
    return CACHE_DIR / f"{gene.upper()}.json"


def get_ppi_scores(gene: str, verbose: bool = False) -> dict[str, float]:
    """
    STRING interaction_partners for one gene.
    Returns {partner_symbol: combined_score, ...}. Results cached to disk.
    """
    cached = _cache_path(gene)
    if cached.exists():
        return json.loads(cached.read_text())

    if verbose:
        print(f"    [STRING] fetching PPI for {gene.upper()}", flush=True)

    params = urllib.parse.urlencode({
        "identifiers":     gene.upper(),
        "species":         DEFAULT_SPECIES,
        "limit":           STRING_LIMIT,
        "caller_identity": "waddington_gene_ranker",
    })
    url = f"{STRING_BASE}/interaction_partners?{params}"
    try:
        with urllib.request.urlopen(url, timeout=STRING_TIMEOUT) as resp:
            data = json.loads(resp.read())
        scores: dict[str, float] = {}
        for item in data:
            partner = item.get("preferredName_B") or item.get("preferredName_A", "")
            if partner and partner.upper() != gene.upper():
                scores[partner.upper()] = round(float(item.get("score", 0)), 3)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(scores))
        return scores
    except Exception as e:
        if verbose:
            print(f"    [STRING] error for {gene}: {e}", flush=True)
        return {}


def build_anchor_scores(anchor_genes: list[str], verbose: bool = False) -> dict[str, float]:
    """
    Expand anchor genes via STRING PPI.
    Returns {gene_symbol: max_ppi_score_to_any_anchor}.
    Anchor genes themselves receive score 1.0.
    """
    all_scores: dict[str, float] = {}
    for anchor in anchor_genes:
        a = anchor.upper()
        all_scores[a] = 1.0
        neighbors = get_ppi_scores(a, verbose=verbose)
        for gene, score in neighbors.items():
            if score > all_scores.get(gene, 0.0):
                all_scores[gene] = score
    return all_scores


# ---------------------------------------------------------------------------
# ARCHS4 co-expression (Feature 4)
# ---------------------------------------------------------------------------

def get_archs4_coexpr(gene: str, top_n: int = 200, verbose: bool = False) -> dict[str, float]:
    """
    ARCHS4 co-expression for one gene (top_n most correlated genes).
    Returns {correlated_gene: pearson_correlation}. Results cached to disk.
    """
    cached = ARCHS4_CACHE_DIR / f"{gene.upper()}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    if verbose:
        print(f"    [ARCHS4] fetching co-expression for {gene.upper()}", flush=True)

    try:
        import gget  # type: ignore
        df = gget.archs4(gene.upper(), gene_count=top_n, species="human",
                         which="correlation", verbose=False)
        result: dict[str, float] = {}
        if df is not None and len(df) > 0:
            for _, row in df.iterrows():
                sym = str(row.get("gene_symbol", "")).strip().upper()
                cor = float(row.get("pearson_correlation", 0.0))
                if sym:
                    result[sym] = round(cor, 4)
        ARCHS4_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(result))
        return result
    except Exception as e:
        if verbose:
            print(f"    [ARCHS4] error for {gene}: {e}", flush=True)
        return {}


def build_archs4_scores(anchor_genes: list[str], verbose: bool = False) -> dict[str, float]:
    """
    Co-expression signal anchored to target-phenotype genes.
    Returns {gene: max_pearson_coexpr_with_any_anchor}.
    """
    all_scores: dict[str, float] = {}
    for anchor in anchor_genes:
        a = anchor.upper()
        coexpr = get_archs4_coexpr(a, verbose=verbose)
        for gene, score in coexpr.items():
            if score > all_scores.get(gene, 0.0):
                all_scores[gene] = score
    return all_scores


# ---------------------------------------------------------------------------
# Phase 2 — LightGBM helpers
# ---------------------------------------------------------------------------

def _load_essential_genes() -> set[str]:
    ceg = CEG_PATH if CEG_PATH.exists() else BDA_CEG
    if not ceg.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(ceg, sep="\t")
        return set(df["GENE"].dropna().str.strip().str.upper())
    except Exception:
        return set()


def _compute_hub_scores() -> dict[str, float]:
    """Inverted PPI cache → normalised hub score per gene (count-based)."""
    counts: dict[str, int] = {}
    for f in CACHE_DIR.glob("*.json"):
        for g in json.loads(f.read_text()):
            counts[g] = counts.get(g, 0) + 1
    if not counts:
        return {}
    mx = max(counts.values())
    return {g: round(c / mx, 4) for g, c in counts.items()}


def _compute_ppi_sum_scores() -> dict[str, float]:
    """
    Weighted PPI hub: sum of STRING combined_scores across all anchor connections.
    Complements hub_score_norm (count) and g1_ppi_score (max) by capturing
    total interaction strength with the anchor network.
    """
    totals: dict[str, float] = {}
    for f in CACHE_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        for gene, score in d.items():
            totals[gene] = totals.get(gene, 0.0) + score
    if not totals:
        return {}
    mx = max(totals.values())
    return {g: round(s / mx, 4) for g, s in totals.items()}


# Public aliases used by coreset_ranker and external scripts
compute_hub_scores_from_cache = _compute_hub_scores
compute_ppi_sum_from_cache    = _compute_ppi_sum_scores


def _load_lgbm_model(dataset_name: str):
    """Load a per-dataset LightGBM model if available, else return None."""
    model_path = MODEL_DIR / f"lgbm_{dataset_name}.pkl"
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# Current feature set (4 features, is_essential removed — importance was 0)
FEATURE_COLS = ["g1_ppi_score", "hub_score_norm", "archs4_coexpr", "ppi_score_sum"]

# Legacy 3-feature columns (models trained before ARCHS4/ppi_sum expansion)
_FEATURE_COLS_3 = ["g1_ppi_score", "hub_score_norm", "is_essential"]


def _lgbm_scores(
    genes: list[str],
    model,
    anchor_scores: dict[str, float],
    hub_scores: dict[str, float],
    archs4_scores: Optional[dict[str, float]] = None,
    ppi_sum_scores: Optional[dict[str, float]] = None,
    essential: Optional[set[str]] = None,
) -> dict[str, float]:
    """Run LightGBM model over a gene list; returns {gene: prob}."""
    import pandas as pd
    _a4  = archs4_scores  or {}
    _ps  = ppi_sum_scores or {}
    n_feat = getattr(model, "n_features_in_", len(FEATURE_COLS))

    if n_feat == 3:
        # Legacy 3-feature models (g1_ppi_score, hub_score_norm, is_essential)
        _ess = essential or set()
        rows = [{"g1_ppi_score": anchor_scores.get(g, 0.0),
                 "hub_score_norm": hub_scores.get(g, 0.0),
                 "is_essential": float(g in _ess)} for g in genes]
        X = pd.DataFrame(rows, columns=_FEATURE_COLS_3)
    else:
        # Current 4-feature models
        rows = [{"g1_ppi_score":   anchor_scores.get(g, 0.0),
                 "hub_score_norm": hub_scores.get(g, 0.0),
                 "archs4_coexpr":  _a4.get(g, 0.0),
                 "ppi_score_sum":  _ps.get(g, 0.0)} for g in genes]
        X = pd.DataFrame(rows, columns=FEATURE_COLS)

    probs = model.predict_proba(X)[:, 1]
    return dict(zip(genes, probs.tolist()))


# ---------------------------------------------------------------------------
# G1 GeneRanker (Phase 1 + optional Phase 2)
# ---------------------------------------------------------------------------

def waddington_ranker(
    dataset_name: str,
    batch_size: int,
    verbose: bool = False,
) -> RankerFn:
    """
    G1 GeneRanker.

    Phase 1 (always active): STRING PPI expansion from biological anchor genes.
    Phase 2 (auto-activates): If workspace/models/lgbm_{dataset}.pkl exists,
      uses LightGBM probabilities to rerank — trained via bootstrap_lgbm.py.

    Ranking is stable across rounds.
    """
    anchors = DATASET_ANCHORS.get(dataset_name)
    if not anchors:
        if verbose:
            print(f"  [G1] No anchors for {dataset_name} — falling back to random")
        return _random_ranker(batch_size)

    if verbose:
        print(f"  [G1] Building anchor scores for {dataset_name}: {anchors}")
    anchor_scores = build_anchor_scores(anchors, verbose=verbose)

    # Phase 2: try to load LightGBM model
    lgbm_model   = _load_lgbm_model(dataset_name)
    hub_scores:   dict[str, float] = {}
    essential:    set[str] = set()
    archs4_scores: dict[str, float] = {}

    hub_scores:    dict[str, float] = {}
    archs4_scores: dict[str, float] = {}
    ppi_sum_scores: dict[str, float] = {}
    essential:     set[str] = set()

    if lgbm_model is not None:
        n_feat = getattr(lgbm_model, "n_features_in_", len(FEATURE_COLS))
        if verbose:
            print(f"  [G1 P2] LightGBM model loaded for {dataset_name} ({n_feat} features)")
        hub_scores = _compute_hub_scores()
        if n_feat == 3:
            essential = _load_essential_genes()
        else:
            if verbose:
                print(f"  [G1 P2] Building ARCHS4 co-expression scores for {dataset_name}")
            archs4_scores  = build_archs4_scores(anchors, verbose=verbose)
            ppi_sum_scores = _compute_ppi_sum_scores()

    def _ranker(universe: list[str], already_selected: list[str], round_num: int) -> list[str]:
        already = set(already_selected)
        pool = [g for g in universe if g not in already]

        if lgbm_model is not None:
            probs  = _lgbm_scores(pool, lgbm_model, anchor_scores, hub_scores,
                                   archs4_scores or None, ppi_sum_scores or None,
                                   essential or None)
            scored = [(g, probs[g]) for g in pool]
        else:
            scored = [(g, anchor_scores.get(g, 0.0)) for g in pool]

        scored.sort(key=lambda x: -x[1])
        return [g for g, _ in scored[:batch_size]]

    n_feat_loaded = getattr(lgbm_model, "n_features_in_", 0) if lgbm_model else 0
    phase = f"P2(LightGBM+ARCHS4+ppi_sum)" if n_feat_loaded == 4 \
        else "P2(LightGBM)" if lgbm_model is not None else "P1(PPI)"
    if verbose:
        print(f"  [G1 {phase}] Ranker ready for {dataset_name}")
    return _ranker


def _random_ranker(batch_size: int) -> RankerFn:
    def _r(universe: list[str], already_selected: list[str], round_num: int) -> list[str]:
        pool = [g for g in universe if g not in set(already_selected)]
        return random.sample(pool, min(batch_size, len(pool)))
    return _r


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_rank(dataset: str, n: int, exclude: list[str]) -> None:
    """
    JSON output mode for server subprocess calls.
    Outputs: [{"gene": "ZAP70", "score": 1.0, "signals": ["ppi_anchor"]}, ...]
    """
    anchors = DATASET_ANCHORS.get(dataset, [])
    if not anchors:
        print(json.dumps([]))
        return

    scores = build_anchor_scores(anchors, verbose=False)
    excl = set(g.upper() for g in exclude)

    ranked = [
        {"gene": g, "score": s, "signals": ["ppi_anchor"]}
        for g, s in sorted(scores.items(), key=lambda x: -x[1])
        if g not in excl
    ]
    print(json.dumps(ranked[:n]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="",
                        help="Dataset name (e.g. IFNG). If omitted, show debug info.")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON array for server subprocess use")
    parser.add_argument("--n", type=int, default=200,
                        help="Max number of candidates to return")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated gene symbols to exclude")
    args = parser.parse_args()

    if args.json:
        exclude = [g.strip() for g in args.exclude.split(",") if g.strip()]
        _cli_rank(args.dataset, args.n, exclude)
    else:
        # Debug mode
        dataset = args.dataset or "IFNG"
        anchors = DATASET_ANCHORS.get(dataset, [])
        print(f"Dataset: {dataset}")
        print(f"Anchors: {anchors}")
        if anchors:
            print("Building anchor scores (may call STRING API)...")
            scores = build_anchor_scores(anchors, verbose=True)
            top = sorted(scores.items(), key=lambda x: -x[1])[:20]
            print(f"\nTop 20 genes by PPI score to {dataset} anchors:")
            for g, s in top:
                print(f"  {g:12s}  {s:.3f}")
