#!/usr/bin/env python3
"""
Build genuine INTERACTION features x(g_a, g_b) for the combinatorial tier.

Why this file exists
--------------------
The pair table built by build_horlbeck_pairs.py describes a pair only through symmetric functions of
its two genes -- the mean of their features and the absolute difference. A genetic interaction is
defined as the *deviation from the additive expectation of the two single perturbations*, so a
symmetric function of the two marginals is close to orthogonal to the label by construction. The
diagnosis was unambiguous: the leave-one-phenotype-out model scores AUC 0.4959 on Horlbeck, its top
640 pairs contain fewer hits than random, and no feature correlates with the label above |r| = 0.036.
Even each gene's *measured* solo effect, taken from the screen's own negative-control pairs, gives
AUC 0.537 -- additive magnitude genuinely does not predict interaction.

What actually predicts a genetic interaction is whether the two genes act together. These features
say that directly, and each is a property of the PAIR that cannot be recovered from either gene alone:

  pair_coessential      Pearson r between the two genes' DepMap Chronos profiles across 1178 cell
                        lines. Co-essentiality is the standard signal for shared complex membership.
                        Fully offline. Strongest single feature in the offline check: ranking the pool
                        by |r| puts 168 hits in the top 640, against 63 for random.
  pair_kegg_shared      number of KEGG pathways both genes belong to (and the Jaccard of their sets)
  pair_reactome_shared  the same over Reactome pathway IDs
  pair_string_edge      STRING combined score of the direct a--b edge, 0 when no edge is reported
  pair_archs4_coexpr    ARCHS4 co-expression of b with a, where the cached top-200 list covers it
  pair_complex_shared   number of EBI Complex Portal complexes both genes belong to. Very low coverage
                        (0.44% of pairs) but the highest precision of any single signal measured here:
                        those 445 pairs hit at 34.4%, 3.47x the 9.90% base rate and above even the
                        LLM's per-pick rate. Two subunits of one complex is the textbook buffering
                        interaction -- knocking out either already breaks the complex, so the second
                        knockout adds little. Because of that precision/coverage shape it is meant to
                        be consumed as a selection priority rather than as one more term in a rank
                        sum: ranking with it as an 8th equal-weight feature gives 196 hits in the top
                        640, while taking those 445 pairs first and filling the rest from the prior
                        gives 216 (against 182 without it).

A note on what this can and cannot fix
--------------------------------------
These features are dead in the *cross-experiment* prior and that is not a defect of this script: a
single-gene screen contains no pair interactions, so the nine training screens enter the pair table as
degenerate pairs (a = a) whose interaction features carry no information. They are set to 0 there
rather than to the self-value (r = 1, all pathways shared), which would be a constant the model could
only learn as "this row is training data". The consumer that CAN learn them is the online model, which
retrains on pairs actually revealed inside the screen. The prediction this sets up is explicit and
falsifiable: online_adaptive should improve, static_ranker should not.

Usage:
    conda run -n waddington-bio python3 workspace/evaluation/build_pair_interaction_features.py
    conda run -n waddington-bio python3 workspace/evaluation/build_pair_interaction_features.py --offline
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
from waddington_select.tables import pair_token  # noqa: E402

BDA_DIR = REPO_ROOT / "workspace" / "data" / "bda_benchmark"
DEPMAP_CSV = REPO_ROOT / "workspace" / "data" / "depmap" / "CRISPRGeneEffect.csv"
KEGG_CACHE = HERE / "_kegg_cache"
ARCHS4_CACHE = HERE / "_archs4_cache"
REACTOME_ID_CACHE = HERE / "_reactome_id_cache"      # pathway IDs (the old cache stores only counts)
STRING_CACHE_DIR = HERE / "_string_pair_cache"
OUT_CSV = HERE / "pair_interaction_features.csv"

FEATURES = ["pair_coessential", "pair_kegg_shared", "pair_kegg_jaccard",
            "pair_reactome_shared", "pair_reactome_jaccard", "pair_string_edge",
            "pair_archs4_coexpr", "pair_complex_shared"]

COMPLEX_TSV = REPO_ROOT / "workspace" / "data" / "complexes" / "complexportal_9606.tsv"
UNIPROT_MAP = REPO_ROOT / "workspace" / "data" / "complexes" / "uniprot_to_symbol.json"


def horlbeck_pairs() -> tuple[list[tuple[str, str]], list[str]]:
    gt = pd.read_csv(BDA_DIR / "ground_truth_Horlbeck.csv")
    raw = [ast.literal_eval(s) for s in gt["Gene_pairs"]]
    key = [tuple(sorted((a.upper(), b.upper()))) for a, b in raw]
    return key, sorted({g for k in key for g in k})


def pairs_from_csv(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    """Any screen with a `pair` column of "A|B" tokens -- used for the Norman transfer set."""
    df = pd.read_csv(path)
    key = [tuple(sorted(t.upper().split("|"))) for t in df["pair"]]
    return key, sorted({g for k in key for g in k})


# ── offline: DepMap co-essentiality ─────────────────────────────────────────────────────────────

def coessentiality(genes: list[str]) -> dict[tuple[str, str], float]:
    print(f"  reading {DEPMAP_CSV.name} ...")
    df = pd.read_csv(DEPMAP_CSV, index_col=0)
    sym = [re.match(r"^([A-Za-z0-9_\-\.]+)\s*\(", c).group(1).upper()
           if re.match(r"^([A-Za-z0-9_\-\.]+)\s*\(", c) else c.strip().upper() for c in df.columns]
    df.columns = sym
    df = df.loc[:, ~df.columns.duplicated()]
    avail = [g for g in genes if g in df.columns]
    print(f"  DepMap covers {len(avail)}/{len(genes)} genes over {len(df)} cell lines")
    M = df[avail]
    M = M.fillna(M.mean())
    C = np.corrcoef(((M - M.mean()) / M.std().replace(0, 1)).values.T)
    idx = {g: i for i, g in enumerate(avail)}
    return {"__matrix__": (C, idx)}


# ── pathway membership ──────────────────────────────────────────────────────────────────────────

def kegg_sets(genes: list[str], offline: bool) -> dict[str, set]:
    missing = [g for g in genes if not (KEGG_CACHE / f"{g}.json").exists()]
    if missing and not offline:
        print(f"  KEGG: fetching {len(missing)} uncached genes")
        sys.path.insert(0, str(HERE))
        from gene_ranker import prefetch_kegg_batch
        prefetch_kegg_batch(missing, verbose=True)
    out = {}
    for g in genes:
        try:
            out[g] = set(json.loads((KEGG_CACHE / f"{g}.json").read_text()))
        except Exception:
            out[g] = set()
    print(f"  KEGG: {sum(1 for v in out.values() if v)}/{len(genes)} genes annotated")
    return out


def reactome_sets(genes: list[str], offline: bool) -> dict[str, set]:
    REACTOME_ID_CACHE.mkdir(parents=True, exist_ok=True)
    missing = [g for g in genes if not (REACTOME_ID_CACHE / f"{g}.json").exists()]
    if missing and not offline:
        print(f"  Reactome: fetching pathway IDs for {len(missing)} genes")
        for i in range(0, len(missing), 200):
            chunk = missing[i: i + 200]
            body = json.dumps({"q": chunk, "scopes": "symbol",
                               "fields": "pathway.reactome", "species": "human"}).encode()
            req = urllib.request.Request("https://mygene.info/v3/query", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                hits = json.loads(urllib.request.urlopen(req, timeout=60).read())
            except Exception as exc:
                print(f"    [WARN] MyGene error: {exc}")
                hits = []
            got = {}
            for h in hits:
                pw = h.get("pathway", {}).get("reactome", [])
                if isinstance(pw, dict):
                    pw = [pw]
                got.setdefault(h.get("query", "").upper(), []).extend(
                    p.get("id") for p in pw if isinstance(p, dict) and p.get("id"))
            for g in chunk:
                (REACTOME_ID_CACHE / f"{g}.json").write_text(json.dumps(sorted(set(got.get(g, [])))))
            print(f"    {min(i + 200, len(missing))}/{len(missing)}", flush=True)
            time.sleep(0.5)
    out = {}
    for g in genes:
        try:
            out[g] = set(json.loads((REACTOME_ID_CACHE / f"{g}.json").read_text()))
        except Exception:
            out[g] = set()
    print(f"  Reactome: {sum(1 for v in out.values() if v)}/{len(genes)} genes annotated")
    return out


# ── STRING direct edges ─────────────────────────────────────────────────────────────────────────

def string_edges(genes: list[str], offline: bool, tag: str = "horlbeck") -> dict[tuple[str, str], float]:
    STRING_CACHE = STRING_CACHE_DIR / f"{tag}_network.tsv"
    if not STRING_CACHE.exists() and not offline:
        STRING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"  STRING: fetching the induced network over {len(genes)} genes")
        rows = []
        # STRING caps identifiers per request; ask for the sub-network of each chunk against all.
        for i in range(0, len(genes), 100):
            chunk = genes[i: i + 100]
            url = "https://string-db.org/api/tsv/network?" + urllib.parse.urlencode(
                {"identifiers": "%0d".join(chunk + genes), "species": 9606,
                 "caller_identity": "waddington"})
            try:
                txt = urllib.request.urlopen(url, timeout=120).read().decode()
                rows.extend(txt.strip().split("\n")[1:])
            except Exception as exc:
                print(f"    [WARN] STRING error on chunk {i}: {exc}")
            time.sleep(1.0)
            print(f"    {min(i + 100, len(genes))}/{len(genes)}", flush=True)
        STRING_CACHE.write_text("\n".join(sorted(set(rows))))
    edges: dict[tuple[str, str], float] = {}
    if STRING_CACHE.exists():
        for line in STRING_CACHE.read_text().split("\n"):
            f = line.split("\t")
            if len(f) < 6:
                continue
            try:
                edges[tuple(sorted((f[2].upper(), f[3].upper())))] = max(
                    float(f[5]), edges.get(tuple(sorted((f[2].upper(), f[3].upper()))), 0.0))
            except ValueError:
                continue
    print(f"  STRING: {len(edges)} distinct edges cached")
    return edges


def complex_membership(genes: list[str], offline: bool) -> dict[str, set[int]]:
    """{GENE: set of complex indices}, from EBI Complex Portal.

    CORUM would be the conventional source but its host currently serves a certificate its own chain
    cannot validate; Complex Portal is the same kind of curated assembly data over a working endpoint.
    Members are listed as UniProt accessions, so we map our gene symbols to UniProt once (449 symbols,
    one batched MyGene call) rather than mapping thousands of accessions the other way.
    """
    if not COMPLEX_TSV.exists():
        print(f"  [NOTE] {COMPLEX_TSV.name} missing — complex co-membership will be 0 for every pair")
        return {}
    # The map is keyed by accession, so a cache built for one screen silently covers none of another
    # screen's genes and every pair comes out 0. Top it up for whatever is missing instead.
    u2g = json.loads(UNIPROT_MAP.read_text()) if UNIPROT_MAP.exists() else {}
    have = set(u2g.values())
    missing = [g for g in genes if g not in have]
    if missing and offline:
        print(f"  [--offline] {len(missing)}/{len(genes)} genes unmapped — their pairs will read 0")
    elif missing:
        print(f"  UniProt: fetching {len(missing)} unmapped genes")
        for i in range(0, len(missing), 200):
            chunk = missing[i: i + 200]
            body = json.dumps({"q": chunk, "scopes": "symbol",
                               "fields": "uniprot.Swiss-Prot", "species": "human"}).encode()
            req = urllib.request.Request("https://mygene.info/v3/query", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            try:
                for h in json.loads(urllib.request.urlopen(req, timeout=60).read()):
                    up = h.get("uniprot", {}).get("Swiss-Prot")
                    if isinstance(up, str):
                        up = [up]
                    for acc in (up or []):
                        u2g[acc] = h.get("query", "").upper()
            except Exception as exc:
                print(f"    [WARN] UniProt mapping error: {exc}")
            time.sleep(0.5)
        UNIPROT_MAP.parent.mkdir(parents=True, exist_ok=True)
        UNIPROT_MAP.write_text(json.dumps(u2g))

    member: dict[str, set[int]] = {}
    n = 0
    for line in COMPLEX_TSV.read_text().split("\n")[1:]:
        f = line.split("\t")
        if len(f) < 5:
            continue
        sym = {u2g[m.split("(")[0]] for m in f[4].split("|") if m.split("(")[0] in u2g}
        if len(sym) < 2:
            continue
        for g in sym:
            member.setdefault(g, set()).add(n)
        n += 1
    covered = len(set(member) & set(genes))
    print(f"  Complex Portal: {n} complexes with >=2 annotated genes, covering {covered}/{len(genes)} "
          f"of this screen")
    return member


def archs4_pairs(genes: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for g in genes:
        p = ARCHS4_CACHE / f"{g}.json"
        if p.exists():
            try:
                out[g] = json.loads(p.read_text())
            except Exception:
                pass
    print(f"  ARCHS4: {len(out)}/{len(genes)} genes have a cached co-expression list")
    return out


def main(offline: bool = False, pairs_csv: str | None = None,
         out: str | None = None, tag: str = "horlbeck") -> None:
    if pairs_csv:
        key, genes = pairs_from_csv(Path(pairs_csv))
    else:
        key, genes = horlbeck_pairs()
    out_csv = Path(out) if out else OUT_CSV
    print(f"{tag}: {len(key)} pairs over {len(genes)} genes\n")

    print("DepMap co-essentiality")
    C, idx = coessentiality(genes)["__matrix__"]
    print("\nKEGG");     kegg = kegg_sets(genes, offline)
    print("\nReactome"); reac = reactome_sets(genes, offline)
    print("\nSTRING");   edges = string_edges(genes, offline, tag=tag)
    print("\nARCHS4");   a4 = archs4_pairs(genes)
    print("\nComplex Portal"); cx = complex_membership(genes, offline)

    print("\nAssembling ...")
    rows = []
    for a, b in key:
        co = float(C[idx[a], idx[b]]) if a in idx and b in idx else 0.0
        ka, kb = kegg.get(a, set()), kegg.get(b, set())
        ra, rb = reac.get(a, set()), reac.get(b, set())
        coexpr = 0.0
        if a in a4 and b in a4[a]:
            coexpr = float(a4[a][b])
        elif b in a4 and a in a4[b]:
            coexpr = float(a4[b][a])
        rows.append({
            "gene": pair_token(a, b),
            "pair_coessential": round(co, 5),
            "pair_kegg_shared": len(ka & kb),
            "pair_kegg_jaccard": round(len(ka & kb) / max(len(ka | kb), 1), 5),
            "pair_reactome_shared": len(ra & rb),
            "pair_reactome_jaccard": round(len(ra & rb) / max(len(ra | rb), 1), 5),
            "pair_string_edge": round(edges.get((a, b), 0.0), 4),
            "pair_archs4_coexpr": round(coexpr, 5),
            "pair_complex_shared": len(cx.get(a, set()) & cx.get(b, set())),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv.name}: {len(out_df)} pairs x {len(FEATURES)} features")
    print("\nnon-zero coverage:")
    for c in FEATURES:
        print(f"  {c:24s} {100 * (out_df[c] != 0).mean():6.2f}%   std={out_df[c].std():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use only what is already cached")
    ap.add_argument("--pairs-csv", default=None, help="CSV with a `pair` column of A|B tokens")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="horlbeck", help="names the STRING network cache")
    main(**vars(ap.parse_args()))
