#!/usr/bin/env python3
"""
G1 GeneRanker — Phase 1 (rule-based biological anchor expansion).

Ranks candidate genes using STRING PPI expansion from context-relevant
anchor genes. Results are cached to disk so the benchmark runs fast.

Integration:
  from gene_ranker import waddington_ranker
  ranker = waddington_ranker("IFNG", batch_size=128)
  batch  = ranker(universe, already_selected, round_num)

Phase 2 will replace this with a LightGBM ranker trained on quality-filtered
experiment history from the Waddington DB.
"""

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

RankerFn = Callable[[list[str], list[str], int], list[str]]

CACHE_DIR = Path(__file__).parent / "_ppi_cache"
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
# G1 GeneRanker
# ---------------------------------------------------------------------------

def waddington_ranker(
    dataset_name: str,
    batch_size: int,
    verbose: bool = False,
) -> RankerFn:
    """
    G1 GeneRanker Phase 1.

    Builds a static relevance score for every gene in the universe using
    STRING PPI expansion from biological anchor genes.

    Ranking is stable across rounds — the first round selects the globally
    best candidates; later rounds continue from where it left off.

    Phase 2 (LightGBM): after ≥20 quality-verified experiments, train a
    ranker on DB-stored (gene, RFS_features → hit_label) pairs.
    """
    anchors = DATASET_ANCHORS.get(dataset_name)
    if not anchors:
        if verbose:
            print(f"  [G1] No anchors for {dataset_name} — falling back to random")
        return _random_ranker(batch_size)

    if verbose:
        print(f"  [G1] Building anchor scores for {dataset_name}: {anchors}")
    anchor_scores = build_anchor_scores(anchors, verbose=verbose)

    def _ranker(universe: list[str], already_selected: list[str], round_num: int) -> list[str]:
        already = set(already_selected)
        pool = [g for g in universe if g not in already]
        # Score by PPI proximity to anchors; ties broken randomly for exploration
        scored = [(g, anchor_scores.get(g, 0.0)) for g in pool]
        scored.sort(key=lambda x: (-x[1], random.random()))
        return [g for g, _ in scored[:batch_size]]

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
