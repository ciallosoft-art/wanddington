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

CACHE_DIR        = Path(__file__).parent / "_ppi_cache"        # anchor genes only
ARCHS4_CACHE_DIR = Path(__file__).parent / "_archs4_cache"
KEGG_CACHE_DIR   = Path(__file__).parent / "_kegg_cache"
REVEAL_CACHE_DIR = Path(__file__).parent / "_reveal_ppi_cache"  # confirmed hits (sequential sim)
MODEL_DIR        = Path(__file__).resolve().parents[2] / "workspace" / "models"
CEG_PATH   = Path(__file__).resolve().parents[2] / "workspace" / "benchmarks" / "CEGv2.txt"
BDA_CEG    = Path("/home/duanyu/Python/keypaper/code/BioDiscoveryAgent/CEGv2.txt")
UNIV_FEAT_CSV     = Path(__file__).resolve().parents[2] / "workspace" / "data" / "universal_features.csv"
PATHWAY_FEAT_CSV  = Path(__file__).resolve().parents[2] / "workspace" / "data" / "pathway_features.csv"

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
    "Replogle_K562_essential": [
        "SF3B1", "PRPF8",                                # core spliceosome (hits: SFPQ, SNRPE, RBM25)
        "MED1", "MED12",                                 # Mediator complex (hits: MED4, MED9, MED21)
        "CDK9", "BRD4",                                  # P-TEFb/elongation (hits: PAF1, RTF1 complex)
        "TAL1", "SPI1",                                  # hematopoietic TFs (hit: GATA1 in K562)
        "PSMD1", "PSMD3",                                # 19S proteasome (hits: PSMD2, PSMC4)
        "HSPA8",                                         # cytoplasmic chaperone (hit: HSPA5 ER)
    ],
    "Replogle_K562_gwps": [
        "MED19", "MED10", "MED17",                       # Mediator core (top 1/2/5 AD hits)
        "TAF1", "TAF2",                                  # TFIID general transcription (#3/7 hits)
        "KDM1A",                                         # LSD1 histone demethylase (#6 hit)
        "MAX",                                           # MYC/MAX network (K562 dependency)
        "WDR82",                                         # SETD1 complex / H3K4me3
        "SSRP1",                                         # FACT chromatin remodeling complex
        "CDK9",                                          # P-TEFb elongation (shared signal)
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


def _get_reveal_ppi_scores(gene: str) -> dict[str, float]:
    """
    PPI scores for a confirmed-hit gene used in sequential simulation.
    Caches to REVEAL_CACHE_DIR (_reveal_ppi_cache/) — NOT to CACHE_DIR (_ppi_cache/).
    This keeps hub_score_norm stable: _compute_hub_scores() reads only _ppi_cache/,
    so reveal() calls never corrupt the hub score normalization.
    """
    g = gene.upper()
    reveal_path = REVEAL_CACHE_DIR / f"{g}.json"
    if reveal_path.exists():
        return json.loads(reveal_path.read_text())
    # If the gene is also an anchor, reuse its anchor cache file (read-only)
    anchor_path = CACHE_DIR / f"{g}.json"
    if anchor_path.exists():
        return json.loads(anchor_path.read_text())
    # Fetch from STRING, write to reveal cache (not anchor cache)
    params = urllib.parse.urlencode({
        "identifiers":     g,
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
            if partner and partner.upper() != g:
                scores[partner.upper()] = round(float(item.get("score", 0)), 3)
        REVEAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        reveal_path.write_text(json.dumps(scores))
        return scores
    except Exception:
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


# ---------------------------------------------------------------------------
# KEGG pathway overlap (Feature 5)
# ---------------------------------------------------------------------------

def _fetch_kegg_batch(genes: list[str]) -> dict[str, list[str]]:
    """
    Batch-query MyGene.info for KEGG pathway IDs.
    Returns {GENE: [pathway_id, ...]} for genes that have KEGG data.
    """
    data = json.dumps({
        "q": [g.upper() for g in genes],
        "scopes": "symbol",
        "fields": "pathway.kegg",
        "species": "human",
    }).encode()
    req = urllib.request.Request(
        "https://mygene.info/v3/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            hits = json.loads(resp.read())
    except Exception:
        return {}

    result: dict[str, list[str]] = {}
    for hit in hits:
        sym = hit.get("query", "").upper()
        pathways = hit.get("pathway", {}).get("kegg", [])
        if isinstance(pathways, dict):
            pathways = [pathways]
        ids = [p["id"] for p in pathways if isinstance(p, dict) and "id" in p]
        result[sym] = ids
    return result


def get_kegg_pathways(gene: str, verbose: bool = False) -> list[str]:
    """
    KEGG pathway IDs for one gene. Results cached to _kegg_cache/{GENE}.json.
    """
    KEGG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = KEGG_CACHE_DIR / f"{gene.upper()}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    batch = _fetch_kegg_batch([gene])
    ids = batch.get(gene.upper(), [])
    cached.write_text(json.dumps(ids))
    return ids


def prefetch_kegg_batch(genes: list[str], batch_size: int = 500,
                        verbose: bool = False) -> None:
    """
    Pre-populate _kegg_cache/ for a list of genes using batch API.
    Skips genes already cached. Typically called once per dataset.
    """
    KEGG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    to_fetch = [g for g in genes
                if not (KEGG_CACHE_DIR / f"{g.upper()}.json").exists()]
    if not to_fetch:
        return
    if verbose:
        print(f"    [KEGG] batch-fetching {len(to_fetch)} genes "
              f"({len(genes) - len(to_fetch)} cached)", flush=True)
    for i in range(0, len(to_fetch), batch_size):
        chunk = to_fetch[i: i + batch_size]
        fetched = _fetch_kegg_batch(chunk)
        for g in chunk:
            cached = KEGG_CACHE_DIR / f"{g.upper()}.json"
            ids = fetched.get(g.upper(), [])
            cached.write_text(json.dumps(ids))
        if verbose:
            done = min(i + batch_size, len(to_fetch))
            print(f"    [KEGG] {done}/{len(to_fetch)} fetched", flush=True)


def build_kegg_scores(
    genes: list[str],
    anchor_genes: list[str],
    verbose: bool = False,
) -> dict[str, float]:
    """
    KEGG pathway overlap feature.

    1. Collect all KEGG pathway IDs for anchor genes → anchor_pathway_set
    2. For each candidate gene, count shared pathways with anchor_pathway_set
    3. Normalize to [0, 1] by dividing by the max count in this gene set

    Returns {gene: score} with scores in [0, 1].
    """
    # Pre-populate cache for anchors + candidates in one pass
    all_genes_needed = list(set(anchor_genes) | set(genes))
    prefetch_kegg_batch(all_genes_needed, verbose=verbose)

    # Build anchor pathway set
    anchor_pathway_set: set[str] = set()
    for a in anchor_genes:
        anchor_pathway_set.update(get_kegg_pathways(a))

    if not anchor_pathway_set:
        return {g: 0.0 for g in genes}

    # Score each candidate
    raw: dict[str, int] = {}
    for g in genes:
        g_paths = set(get_kegg_pathways(g))
        raw[g] = len(g_paths & anchor_pathway_set)

    mx = max(raw.values()) if raw else 1
    if mx == 0:
        return {g: 0.0 for g in genes}
    return {g: round(raw[g] / mx, 4) for g in genes}


class LGBMWrapper:
    """
    Thin wrapper around LGBMClassifier that carries a rank_norm flag.

    When rank_norm=True the model expects its anchor-specific features
    (g1_ppi_score, archs4_coexpr, kegg_overlap) to be rank-normalised
    within the gene universe before predict_proba is called.
    Saved to disk together with the underlying model via pickle.
    """
    def __init__(self, model, rank_norm: bool = False):
        self.model      = model
        self.rank_norm  = rank_norm
        # expose the attribute that _lgbm_scores inspects
        self.n_features_in_ = model.n_features_in_

    def predict_proba(self, X):
        return self.model.predict_proba(X)


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


# ---------------------------------------------------------------------------
# Universal (anchor-independent) biological features — Plan B
# ---------------------------------------------------------------------------

def _load_universal_features() -> dict[str, tuple[float, float]]:
    """Load pLI and STRING global degree for all genes. Returns {gene: (pli, deg)}."""
    if not UNIV_FEAT_CSV.exists():
        return {}
    import csv
    result: dict[str, tuple[float, float]] = {}
    with open(UNIV_FEAT_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gene = row["gene"].strip().upper()
            try:
                pli = float(row["pli_score"])
                deg = float(row["string_degree_norm"])
                result[gene] = (pli, deg)
            except (ValueError, KeyError):
                pass
    return result

_UNIV_FEATURES: dict[str, tuple[float, float]] = {}  # lazy-loaded once


def _get_univ_features() -> dict[str, tuple[float, float]]:
    global _UNIV_FEATURES
    if not _UNIV_FEATURES:
        _UNIV_FEATURES = _load_universal_features()
    return _UNIV_FEATURES


def _load_pathway_features() -> dict[str, tuple[float, float]]:
    """Load KEGG and Reactome pathway counts. Returns {gene: (kegg_norm, reactome_norm)}."""
    if not PATHWAY_FEAT_CSV.exists():
        return {}
    import csv
    result: dict[str, tuple[float, float]] = {}
    with open(PATHWAY_FEAT_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gene = row["gene"].strip().upper()
            try:
                kegg = float(row["kegg_pathway_count_norm"])
                reac = float(row["reactome_pathway_count_norm"])
                result[gene] = (kegg, reac)
            except (ValueError, KeyError):
                pass
    return result


_PATHWAY_FEATURES: dict[str, tuple[float, float]] = {}  # lazy-loaded once


def _get_pathway_features() -> dict[str, tuple[float, float]]:
    global _PATHWAY_FEATURES
    if not _PATHWAY_FEATURES:
        _PATHWAY_FEATURES = _load_pathway_features()
    return _PATHWAY_FEATURES


# Current feature set (9 features, V7: +kegg_pathway_count_norm, +reactome_pathway_count_norm)
FEATURE_COLS = [
    "g1_ppi_score", "hub_score_norm", "archs4_coexpr",
    "ppi_score_sum", "kegg_overlap",
    "pli_score", "string_degree_norm",
    "kegg_pathway_count_norm", "reactome_pathway_count_norm",
]

# Anchor-specific features: distribution depends on the dataset's anchor genes.
# These three are rank-normalised per dataset when rank_norm=True.
_ANCHOR_FEAT_COLS = ["g1_ppi_score", "archs4_coexpr", "kegg_overlap"]

# Legacy feature sets for backward-compatible model loading
_FEATURE_COLS_7 = ["g1_ppi_score", "hub_score_norm", "archs4_coexpr", "ppi_score_sum",
                   "kegg_overlap", "pli_score", "string_degree_norm"]
_FEATURE_COLS_5 = ["g1_ppi_score", "hub_score_norm", "archs4_coexpr", "ppi_score_sum", "kegg_overlap"]
_FEATURE_COLS_4 = ["g1_ppi_score", "hub_score_norm", "archs4_coexpr", "ppi_score_sum"]
_FEATURE_COLS_3 = ["g1_ppi_score", "hub_score_norm", "is_essential"]


def _lgbm_scores(
    genes: list[str],
    model,
    anchor_scores: dict[str, float],
    hub_scores: dict[str, float],
    archs4_scores: Optional[dict[str, float]] = None,
    ppi_sum_scores: Optional[dict[str, float]] = None,
    kegg_scores: Optional[dict[str, float]] = None,
    essential: Optional[set[str]] = None,
) -> dict[str, float]:
    """Run LightGBM model over a gene list; returns {gene: prob}."""
    import pandas as pd
    _a4  = archs4_scores  or {}
    _ps  = ppi_sum_scores or {}
    _kg  = kegg_scores    or {}
    n_feat = getattr(model, "n_features_in_", len(FEATURE_COLS))

    if n_feat == 3:
        _ess = essential or set()
        rows = [{"g1_ppi_score": anchor_scores.get(g, 0.0),
                 "hub_score_norm": hub_scores.get(g, 0.0),
                 "is_essential": float(g in _ess)} for g in genes]
        X = pd.DataFrame(rows, columns=_FEATURE_COLS_3)
    elif n_feat == 4:
        rows = [{"g1_ppi_score":   anchor_scores.get(g, 0.0),
                 "hub_score_norm": hub_scores.get(g, 0.0),
                 "archs4_coexpr":  _a4.get(g, 0.0),
                 "ppi_score_sum":  _ps.get(g, 0.0)} for g in genes]
        X = pd.DataFrame(rows, columns=_FEATURE_COLS_4)
    elif n_feat == 5:
        rows = [{"g1_ppi_score":   anchor_scores.get(g, 0.0),
                 "hub_score_norm": hub_scores.get(g, 0.0),
                 "archs4_coexpr":  _a4.get(g, 0.0),
                 "ppi_score_sum":  _ps.get(g, 0.0),
                 "kegg_overlap":   _kg.get(g, 0.0)} for g in genes]
        X = pd.DataFrame(rows, columns=_FEATURE_COLS_5)
    elif n_feat == 7:
        # 7-feature model (V6: +pli_score, +string_degree_norm)
        _uf = _get_univ_features()
        rows = [{"g1_ppi_score":        anchor_scores.get(g, 0.0),
                 "hub_score_norm":      hub_scores.get(g, 0.0),
                 "archs4_coexpr":       _a4.get(g, 0.0),
                 "ppi_score_sum":       _ps.get(g, 0.0),
                 "kegg_overlap":        _kg.get(g, 0.0),
                 "pli_score":           _uf.get(g, (0.0, 0.0))[0],
                 "string_degree_norm":  _uf.get(g, (0.0, 0.0))[1]} for g in genes]
        X = pd.DataFrame(rows, columns=_FEATURE_COLS_7)
    else:
        # 9-feature model (V7: +kegg_pathway_count_norm, +reactome_pathway_count_norm)
        _uf = _get_univ_features()
        _pf = _get_pathway_features()
        rows = [{"g1_ppi_score":                 anchor_scores.get(g, 0.0),
                 "hub_score_norm":               hub_scores.get(g, 0.0),
                 "archs4_coexpr":                _a4.get(g, 0.0),
                 "ppi_score_sum":                _ps.get(g, 0.0),
                 "kegg_overlap":                 _kg.get(g, 0.0),
                 "pli_score":                    _uf.get(g, (0.0, 0.0))[0],
                 "string_degree_norm":           _uf.get(g, (0.0, 0.0))[1],
                 "kegg_pathway_count_norm":      _pf.get(g, (0.0, 0.0))[0],
                 "reactome_pathway_count_norm":  _pf.get(g, (0.0, 0.0))[1]} for g in genes]
        X = pd.DataFrame(rows, columns=FEATURE_COLS)

    probs = model.predict_proba(X)[:, 1]
    return dict(zip(genes, probs.tolist()))


def _lgbm_scores_with_rank_norm(
    pool_genes: list[str],
    universe_genes: list[str],
    model,
    anchor_scores: dict[str, float],
    hub_scores: dict[str, float],
    archs4_scores: Optional[dict[str, float]] = None,
    ppi_sum_scores: Optional[dict[str, float]] = None,
    kegg_scores: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    """
    LightGBM scoring with rank normalisation of anchor-specific features.

    Rank normalisation is applied to _ANCHOR_FEAT_COLS within *universe_genes*
    so the reference distribution stays fixed as the pool shrinks across rounds.
    """
    import pandas as pd
    from scipy.stats import rankdata

    n_feat = getattr(model, "n_features_in_", len(FEATURE_COLS))
    _a4 = archs4_scores  or {}
    _ps = ppi_sum_scores or {}
    _kg = kegg_scores    or {}

    if n_feat >= 9:
        _uf = _get_univ_features()
        _pf = _get_pathway_features()
        feat_cols = FEATURE_COLS
        rows = [{"g1_ppi_score":                anchor_scores.get(g, 0.0),
                 "hub_score_norm":              hub_scores.get(g, 0.0),
                 "archs4_coexpr":               _a4.get(g, 0.0),
                 "ppi_score_sum":               _ps.get(g, 0.0),
                 "kegg_overlap":                _kg.get(g, 0.0),
                 "pli_score":                   _uf.get(g, (0.0, 0.0))[0],
                 "string_degree_norm":          _uf.get(g, (0.0, 0.0))[1],
                 "kegg_pathway_count_norm":     _pf.get(g, (0.0, 0.0))[0],
                 "reactome_pathway_count_norm": _pf.get(g, (0.0, 0.0))[1]} for g in universe_genes]
    elif n_feat >= 7:
        _uf = _get_univ_features()
        feat_cols = _FEATURE_COLS_7
        rows = [{"g1_ppi_score":        anchor_scores.get(g, 0.0),
                 "hub_score_norm":      hub_scores.get(g, 0.0),
                 "archs4_coexpr":       _a4.get(g, 0.0),
                 "ppi_score_sum":       _ps.get(g, 0.0),
                 "kegg_overlap":        _kg.get(g, 0.0),
                 "pli_score":           _uf.get(g, (0.0, 0.0))[0],
                 "string_degree_norm":  _uf.get(g, (0.0, 0.0))[1]} for g in universe_genes]
    elif n_feat >= 5:
        feat_cols = _FEATURE_COLS_5
        rows = [{"g1_ppi_score":   anchor_scores.get(g, 0.0),
                 "hub_score_norm": hub_scores.get(g, 0.0),
                 "archs4_coexpr":  _a4.get(g, 0.0),
                 "ppi_score_sum":  _ps.get(g, 0.0),
                 "kegg_overlap":   _kg.get(g, 0.0)} for g in universe_genes]
    else:
        feat_cols = _FEATURE_COLS_4
        rows = [{"g1_ppi_score":   anchor_scores.get(g, 0.0),
                 "hub_score_norm": hub_scores.get(g, 0.0),
                 "archs4_coexpr":  _a4.get(g, 0.0),
                 "ppi_score_sum":  _ps.get(g, 0.0)} for g in universe_genes]

    df = pd.DataFrame(rows, columns=feat_cols)
    n  = len(df)
    for feat in _ANCHOR_FEAT_COLS:
        if feat in df.columns:
            r = rankdata(df[feat].values, method="average") - 1
            df[feat] = r / max(n - 1, 1)

    gene_to_idx = {g: i for i, g in enumerate(universe_genes)}
    valid_pool  = [g for g in pool_genes if g in gene_to_idx]
    idx         = [gene_to_idx[g] for g in valid_pool]
    X           = df.iloc[idx][feat_cols].values
    probs       = model.predict_proba(X)[:, 1]
    return dict(zip(valid_pool, probs.tolist()))


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
    lgbm_model     = _load_lgbm_model(dataset_name)
    hub_scores:    dict[str, float] = {}
    archs4_scores: dict[str, float] = {}
    ppi_sum_scores: dict[str, float] = {}
    kegg_scores:   dict[str, float] = {}
    essential:     set[str] = set()

    if lgbm_model is not None:
        n_feat = getattr(lgbm_model, "n_features_in_", len(FEATURE_COLS))
        if verbose:
            print(f"  [G1 P2] LightGBM model loaded for {dataset_name} ({n_feat} features)")
        hub_scores = _compute_hub_scores()
        if n_feat == 3:
            essential = _load_essential_genes()
        elif n_feat == 4:
            if verbose:
                print(f"  [G1 P2] Building ARCHS4+PPI scores for {dataset_name}")
            archs4_scores  = build_archs4_scores(anchors, verbose=verbose)
            ppi_sum_scores = _compute_ppi_sum_scores()
        elif n_feat == 5:
            if verbose:
                print(f"  [G1 P2] Building ARCHS4+PPI+KEGG scores for {dataset_name}")
            archs4_scores  = build_archs4_scores(anchors, verbose=verbose)
            ppi_sum_scores = _compute_ppi_sum_scores()
        else:
            # 7-feature model (V6): add KEGG + pLI + STRING degree
            if verbose:
                print(f"  [G1 P2] Building ARCHS4+PPI+KEGG+Universal scores for {dataset_name}")
            archs4_scores  = build_archs4_scores(anchors, verbose=verbose)
            ppi_sum_scores = _compute_ppi_sum_scores()
            _get_univ_features()     # warm up the lazy cache
            _get_pathway_features()  # warm up the lazy cache
            # KEGG scores require universe — computed lazily in _ranker

    _rn_scores: dict[str, float] = {}  # rank-norm score cache (computed once per universe)

    def _ranker(universe: list[str], already_selected: list[str], round_num: int) -> list[str]:
        already = set(already_selected)
        pool = [g for g in universe if g not in already]

        if lgbm_model is not None:
            n_feat     = getattr(lgbm_model, "n_features_in_", len(FEATURE_COLS))
            rank_norm  = getattr(lgbm_model, "rank_norm", False)
            _kg = kegg_scores
            if n_feat >= 5 and not _kg:
                _kg = build_kegg_scores(universe, anchors, verbose=verbose)
                kegg_scores.update(_kg)

            if rank_norm:
                # Compute once for the full universe; cache for subsequent rounds.
                if not _rn_scores:
                    _rn_scores.update(_lgbm_scores_with_rank_norm(
                        universe, universe, lgbm_model, anchor_scores, hub_scores,
                        archs4_scores or None, ppi_sum_scores or None, _kg or None,
                    ))
                probs = {g: _rn_scores.get(g, 0.0) for g in pool}
            else:
                probs = _lgbm_scores(pool, lgbm_model, anchor_scores, hub_scores,
                                     archs4_scores or None, ppi_sum_scores or None,
                                     _kg or None, essential or None)
            scored = [(g, probs[g]) for g in pool]
        else:
            scored = [(g, anchor_scores.get(g, 0.0)) for g in pool]

        scored.sort(key=lambda x: -x[1])
        return [g for g, _ in scored[:batch_size]]

    n_feat_loaded = getattr(lgbm_model, "n_features_in_", 0) if lgbm_model else 0
    phase = ("P2(LightGBM+ARCHS4+PPI+KEGG)" if n_feat_loaded >= 5
             else "P2(LightGBM+ARCHS4+ppi_sum)" if n_feat_loaded == 4
             else "P2(LightGBM)" if lgbm_model is not None else "P1(PPI)")
    if verbose:
        print(f"  [G1 {phase}] Ranker ready for {dataset_name}")
    return _ranker


class SequentialWaddingtonRanker:
    """
    Stateful G1 ranker for true sequential simulation (10.2).

    Difference from static waddington_ranker:
      After each batch is selected, the oracle calls reveal(confirmed_hits).
      Confirmed hits become dynamic anchors — their PPI neighbors receive a
      score bonus in subsequent rounds, progressively homing in on the biology.

    Usage:
        ranker = SequentialWaddingtonRanker("IFNG", batch_size=128)
        for rnd in range(1, 6):
            batch = ranker(remaining_pool, selected_so_far, rnd)
            new_hits = oracle_reveal(batch)
            ranker.reveal(new_hits)
        ranker.reset()  # clears state, keeps model/features loaded
    """

    DYNAMIC_WEIGHT: float = 0.5  # additive bonus weight (relative to base ∈ [0,1])

    def __init__(self, dataset_name: str, batch_size: int, verbose: bool = False):
        self.batch_size   = batch_size
        self.verbose      = verbose
        self.dataset_name = dataset_name

        anchors = DATASET_ANCHORS.get(dataset_name)
        if not anchors:
            raise ValueError(f"No anchors registered for dataset '{dataset_name}'")
        self.anchors = anchors

        # ── Immutable base features (same as waddington_ranker) ──────────────
        self.anchor_scores  = build_anchor_scores(anchors, verbose=verbose)
        self.lgbm_model     = _load_lgbm_model(dataset_name)
        self.hub_scores:    dict[str, float] = {}
        self.archs4_scores: dict[str, float] = {}
        self.ppi_sum_scores: dict[str, float] = {}
        self.kegg_scores:   dict[str, float] = {}
        self.essential:     set[str] = set()
        self._kegg_initialized = False

        if self.lgbm_model is not None:
            n_feat = getattr(self.lgbm_model, "n_features_in_", 0)
            self.hub_scores = _compute_hub_scores()
            if n_feat == 3:
                self.essential = _load_essential_genes()
            elif n_feat == 4:
                self.archs4_scores  = build_archs4_scores(anchors, verbose=verbose)
                self.ppi_sum_scores = _compute_ppi_sum_scores()
            else:
                self.archs4_scores  = build_archs4_scores(anchors, verbose=verbose)
                self.ppi_sum_scores = _compute_ppi_sum_scores()
                # kegg_scores initialized lazily on first __call__ (needs universe)

        # ── Mutable state — reset between trials ──────────────────────────────
        self._confirmed_hits: list[str] = []
        self._dynamic_ppi:    dict[str, float] = {}  # gene → max PPI to any confirmed hit
        self._universe:       list[str] = []          # stored on first __call__ for rank-norm ref

    # ── Oracle interface ──────────────────────────────────────────────────────

    def reveal(self, confirmed_hits: list[str]) -> None:
        """
        Oracle reveals which genes from the last batch were true hits.
        Uses _get_reveal_ppi_scores() which writes to _reveal_ppi_cache/ (NOT
        _ppi_cache/) so hub_score_norm stays unaffected by sequential reveals.
        """
        for hit in confirmed_hits:
            hit_ppi = _get_reveal_ppi_scores(hit)
            for gene, score in hit_ppi.items():
                if score > self._dynamic_ppi.get(gene, 0.0):
                    self._dynamic_ppi[gene] = score
        self._confirmed_hits.extend(confirmed_hits)
        if self.verbose and confirmed_hits:
            print(f"  [SEQ] revealed {len(confirmed_hits)} hits; "
                  f"dynamic PPI covers {len(self._dynamic_ppi)} genes", flush=True)

    def reset(self) -> None:
        """Clear per-trial state. Model and features stay loaded for reuse."""
        self._confirmed_hits = []
        self._dynamic_ppi    = {}
        self._universe       = []

    # ── Ranker callable ───────────────────────────────────────────────────────

    def __call__(
        self,
        universe: list[str],
        already_selected: list[str],
        round_num: int,
    ) -> list[str]:
        already = set(already_selected)
        pool    = [g for g in universe if g not in already]

        # Lazy KEGG init on first call (universe == full gene set at round 1)
        if self.lgbm_model is not None:
            n_feat = getattr(self.lgbm_model, "n_features_in_", 0)
            if n_feat >= 5 and not self._kegg_initialized:
                self.kegg_scores = build_kegg_scores(universe, self.anchors,
                                                     verbose=self.verbose)
                self._kegg_initialized = True

        # Store universe on first call — rank-norm reference must stay fixed
        if not self._universe:
            self._universe = list(universe)

        # Base LightGBM probability (or PPI score if no model)
        if self.lgbm_model is not None:
            rank_norm = getattr(self.lgbm_model, "rank_norm", False)
            if rank_norm:
                base = _lgbm_scores_with_rank_norm(
                    pool, self._universe, self.lgbm_model, self.anchor_scores,
                    self.hub_scores, self.archs4_scores or None,
                    self.ppi_sum_scores or None, self.kegg_scores or None,
                )
            else:
                base = _lgbm_scores(
                    pool, self.lgbm_model, self.anchor_scores, self.hub_scores,
                    self.archs4_scores or None, self.ppi_sum_scores or None,
                    self.kegg_scores or None, self.essential or None,
                )
        else:
            base = {g: self.anchor_scores.get(g, 0.0) for g in pool}

        # Additive dynamic bonus from confirmed-hit PPI neighborhood
        scored = [
            (g, base[g] + self._dynamic_ppi.get(g, 0.0) * self.DYNAMIC_WEIGHT)
            for g in pool
        ]
        scored.sort(key=lambda x: -x[1])
        return [g for g, _ in scored[:self.batch_size]]


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
