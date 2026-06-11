#!/usr/bin/env python3
"""
G4 PhenotypeMapper — external API queries for a single gene.

Queries:
  1. MyGene.info  → KEGG + Reactome pathway annotations
  2. STRING-DB    → top PPI interaction partners (gene similarity)

Uses only stdlib (urllib) — no extra dependencies.

Usage:
  phenotype_mapper.py <GENE_SYMBOL>
  phenotype_mapper.py --ping
"""
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

MYGENE_BASE = "https://mygene.info/v3"
STRING_BASE  = "https://string-db.org/api/json"
DEFAULT_SPECIES = 9606  # Homo sapiens
TIMEOUT = 12


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

def ping() -> bool:
    try:
        urllib.request.urlopen(f"{MYGENE_BASE}/metadata", timeout=5)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# MyGene.info — KEGG + Reactome pathways
# ---------------------------------------------------------------------------

def _query_mygene(gene: str) -> dict:
    params = urllib.parse.urlencode({
        "q":       gene,
        "scopes":  "symbol",
        "fields":  "name,symbol,pathway.kegg,pathway.reactome,summary",
        "species": "human",
        "size":    1,
    })
    url = f"{MYGENE_BASE}/query?{params}"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    hits = data.get("hits", [])
    return hits[0] if hits else {}


def _normalise_pathways(raw) -> list[dict]:
    """Handle both single dict and list-of-dicts from MyGene.info."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    return [{"id": p.get("id", ""), "name": p.get("name", "")} for p in raw if isinstance(p, dict)]


def get_pathways(gene: str) -> dict:
    """Return KEGG and Reactome pathway lists for a gene."""
    hit = _query_mygene(gene)
    pathways = hit.get("pathway", {})
    return {
        "gene_name":         hit.get("name", ""),
        "gene_symbol":       hit.get("symbol", gene.upper()),
        "gene_summary":      hit.get("summary", ""),
        "kegg_pathways":     _normalise_pathways(pathways.get("kegg")),
        "reactome_pathways": _normalise_pathways(pathways.get("reactome")),
    }


# ---------------------------------------------------------------------------
# STRING-DB — interaction partners (PPI neighbours = functionally similar genes)
# ---------------------------------------------------------------------------

def get_ppi_neighbors(gene: str, limit: int = 15, species: int = DEFAULT_SPECIES) -> list[dict]:
    """Return top PPI interaction partners from STRING."""
    params = urllib.parse.urlencode({
        "identifiers":     gene,
        "species":         species,
        "limit":           limit,
        "caller_identity": "waddington_phenotype_mapper",
    })
    url = f"{STRING_BASE}/interaction_partners?{params}"
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    neighbors = []
    for item in data:
        # The query gene appears as preferredName_A; partner as preferredName_B
        partner = item.get("preferredName_B") or item.get("preferredName_A")
        if partner and partner.upper() != gene.upper():
            neighbors.append({
                "gene":  partner.upper(),
                "score": round(float(item.get("score", 0)), 3),
            })
    # Sort by score descending, deduplicate
    seen = set()
    result = []
    for n in sorted(neighbors, key=lambda x: x["score"], reverse=True):
        if n["gene"] not in seen:
            seen.add(n["gene"])
            result.append(n)
    return result[:limit]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: phenotype_mapper.py <GENE_SYMBOL>"}))
        sys.exit(1)

    if sys.argv[1] == "--ping":
        print(json.dumps({"ok": ping()}))
        sys.exit(0)

    gene = sys.argv[1].strip().upper()

    try:
        pathway_info = get_pathways(gene)
    except Exception as e:
        pathway_info = {
            "gene_name": "", "gene_symbol": gene, "gene_summary": "",
            "kegg_pathways": [], "reactome_pathways": [],
            "error_pathways": str(e),
        }

    try:
        ppi = get_ppi_neighbors(gene)
    except Exception as e:
        ppi = []
        pathway_info["error_ppi"] = str(e)

    result = {**pathway_info, "ppi_neighbors": ppi}
    print(json.dumps(result))
