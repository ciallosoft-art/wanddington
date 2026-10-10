#!/usr/bin/env python3
"""
Build Horlbeck et al. 2018's Jurkat genetic-interaction map as a second combinatorial benchmark, in the
shape BDA ships for the K562 map (protocol: docs/jurkat_protocol.md).

Source: Table S5 of the paper (workspace/data/horlbeck2018/cell2018_mmc5.xlsx, provenance.json), sheet
"gene GI scores and correlations". BDA's K562 benchmark is that sheet's K562 replicate-average gene-level
GI score, value for value, so the Jurkat benchmark takes the Jurkat replicate average.

  pool    every gene pair with a Jurkat replicate-average score: 388 genes, 75,078 pairs. The Jurkat map
          has no negative-control pairs.
  hits    the most extreme 5% in each tail of that score over the pool. BDA's K562 hits are a
          two-threshold set whose thresholds we could not recover; on K562 this rule reproduces them at
          99.1% (check_k562_rule below).
  prompt  BDA's Horlbeck task prompt with the cell line changed from K562 to Jurkat, nothing else.
  STRING  the interaction-feature builder fetches a STRING network per gene set. To give the pairs shared
          with K562 exactly the features they had there, the Jurkat cache starts from the K562 network
          and adds only the edges of the genes K562 lacks, fetched now.

Writes to workspace/data/horlbeck2018/: ground_truth_Horlbeck_Jurkat.csv, topmovers_Horlbeck_Jurkat.npy,
task_prompts/Horlbeck_Jurkat.json, pairs_Horlbeck_Jurkat.csv (pair tokens for the feature builder);
and workspace/evaluation/_string_pair_cache/horlbeck_jurkat_network.tsv.

Then:
    python3 workspace/evaluation/build_pair_interaction_features.py \
        --pairs-csv workspace/data/horlbeck2018/pairs_Horlbeck_Jurkat.csv \
        --out workspace/evaluation/pair_interaction_features_horlbeck_jurkat.csv --tag horlbeck_jurkat
    python3 workspace/evaluation/build_horlbeck_pairs.py --screen Horlbeck_Jurkat

Usage:
    python3 workspace/evaluation/build_horlbeck_jurkat.py
"""

from __future__ import annotations

import ast
import json
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

DATA = REPO_ROOT / "workspace" / "data" / "horlbeck2018"
XLSX = DATA / "cell2018_mmc5.xlsx"
SCORES = DATA / "extracted" / "gene_gi_scores.csv"
BDA_DIR = REPO_ROOT / "workspace" / "data" / "bda_benchmark"
STRING_DIR = HERE / "_string_pair_cache"

DATASET = "Horlbeck_Jurkat"
TAIL = 0.05
COLUMNS = ["gene_a", "gene_b", "k562_r1_gi", "k562_r1_corr", "k562_r2_gi", "k562_r2_corr", "k562_avg_gi",
           "k562_avg_corr", "jurkat_r1_gi", "jurkat_r1_corr", "jurkat_r2_gi", "jurkat_r2_corr",
           "jurkat_avg_gi", "jurkat_avg_corr"]


def gene_scores() -> pd.DataFrame:
    """Table S5's gene-level sheet, extracted once to CSV."""
    if not SCORES.exists():
        df = pd.read_excel(XLSX, sheet_name="gene GI scores and correlations", header=None, skiprows=4,
                           names=COLUMNS, engine="openpyxl")
        SCORES.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SCORES, index=False)
    df = pd.read_csv(SCORES)
    df["gene_a"] = df["gene_a"].astype(str).str.strip().str.upper()
    df["gene_b"] = df["gene_b"].astype(str).str.strip().str.upper()
    return df


def two_tail_hits(scores: pd.Series) -> tuple[pd.Series, float, float]:
    lo, hi = np.quantile(scores.to_numpy(), [TAIL, 1 - TAIL])
    return (scores <= lo) | (scores >= hi), float(lo), float(hi)


def check_k562_rule(df: pd.DataFrame) -> dict:
    """Apply the same rule to the K562 gene pairs and compare with BDA's K562 hits."""
    k = df[df["k562_avg_gi"].notna() & (df["gene_a"] != df["gene_b"])].copy()
    k["token"] = [pair_token(a, b) for a, b in zip(k["gene_a"], k["gene_b"])]
    hit, lo, hi = two_tail_hits(k["k562_avg_gi"])
    ours = set(k.loc[hit, "token"])
    bda = {pair_token(a, b) for a, b in np.load(BDA_DIR / "topmovers_Horlbeck.npy", allow_pickle=True)}
    inter = len(ours & bda)
    return {"pairs": len(k), "ours": len(ours), "bda": len(bda), "missed": len(bda - ours),
            "extra": len(ours - bda), "jaccard": inter / len(ours | bda), "thresholds": [lo, hi]}


def build_benchmark(df: pd.DataFrame) -> dict:
    j = df[df["jurkat_avg_gi"].notna() & (df["gene_a"] != df["gene_b"])].copy()
    j["token"] = [pair_token(a, b) for a, b in zip(j["gene_a"], j["gene_b"])]
    if j["token"].duplicated().any():
        raise RuntimeError("duplicate pair tokens in the Jurkat map")
    hit, lo, hi = two_tail_hits(j["jurkat_avg_gi"])
    pairs = [tuple(t.split("|")) for t in j["token"]]

    gt = pd.DataFrame({"Gene_pairs": [str(p) for p in pairs], "Score": j["jurkat_avg_gi"].to_numpy()})
    gt.to_csv(DATA / f"ground_truth_{DATASET}.csv", index=False)
    np.save(DATA / f"topmovers_{DATASET}.npy", np.array([p for p, h in zip(pairs, hit) if h], dtype=str))
    pd.DataFrame({"pair": j["token"]}).to_csv(DATA / f"pairs_{DATASET}.csv", index=False)

    prompt = json.loads((BDA_DIR / "task_prompts" / "Horlbeck.json").read_text())
    if prompt["Task"].count("K562") != 1 or any("K562" in v for k, v in prompt.items() if k != "Task"):
        raise RuntimeError("expected exactly one 'K562' in BDA's Horlbeck task prompt")
    prompt["Task"] = prompt["Task"].replace("K562", "Jurkat")
    (DATA / "task_prompts").mkdir(exist_ok=True)
    (DATA / "task_prompts" / f"{DATASET}.json").write_text(json.dumps(prompt, indent=2))

    genes = sorted({g for p in pairs for g in p})
    return {"pool": len(j), "genes": len(genes), "hits": int(hit.sum()), "hit_rate": float(hit.mean()),
            "thresholds": [lo, hi], "gene_list": genes}


def seed_string_cache(genes: list[str]) -> dict:
    """The K562 network, plus the STRING edges of the genes K562 lacks (fetched now, against all genes)."""
    k562 = (STRING_DIR / "horlbeck_network.tsv").read_text().split("\n")
    k562_genes = set()
    for line in k562:
        f = line.split("\t")
        if len(f) >= 4:
            k562_genes |= {f[2].upper(), f[3].upper()}
    gt = pd.read_csv(BDA_DIR / "ground_truth_Horlbeck.csv")
    k562_lib = {g.upper() for p in gt["Gene_pairs"] for g in ast.literal_eval(p)}
    new = sorted(set(genes) - k562_lib)
    rows = list(k562)
    url = "https://string-db.org/api/tsv/network?" + urllib.parse.urlencode(
        {"identifiers": "%0d".join(new + genes), "species": 9606, "caller_identity": "waddington"})
    txt = urllib.request.urlopen(url, timeout=120).read().decode()
    fetched = [r for r in txt.strip().split("\n")[1:]
               if len(r.split("\t")) >= 4 and ({r.split("\t")[2].upper(), r.split("\t")[3].upper()} & set(new))]
    time.sleep(1.0)
    version = json.loads(urllib.request.urlopen("https://string-db.org/api/json/version", timeout=60).read())
    out = STRING_DIR / "horlbeck_jurkat_network.tsv"
    out.write_text("\n".join(sorted(set(r for r in rows + fetched if r))))
    return {"new_genes": new, "new_edges_rows": len(fetched), "k562_rows": len([r for r in k562 if r]),
            "string_version_for_new_edges": version[0].get("string_version") if version else None}


def main() -> None:
    df = gene_scores()
    k = check_k562_rule(df)
    print(f"K562 check: the 5%/5% rule gives {k['ours']} hits against BDA's {k['bda']}: "
          f"missed {k['missed']}, extra {k['extra']}, Jaccard {k['jaccard']:.3f}")
    b = build_benchmark(df)
    print(f"Jurkat: {b['pool']} pairs over {b['genes']} genes, {b['hits']} hits ({b['hit_rate']:.2%}), "
          f"thresholds {b['thresholds'][0]:.3f} / {b['thresholds'][1]:.3f}")
    s = seed_string_cache(b["gene_list"])
    print(f"STRING: {s['k562_rows']} K562 rows + {s['new_edges_rows']} rows for {len(s['new_genes'])} new genes "
          f"({', '.join(s['new_genes'])}; STRING {s['string_version_for_new_edges']})")
    record = {"k562_rule_check": k, "jurkat": {x: v for x, v in b.items() if x != "gene_list"}, "string": s}
    (DATA / f"build_{DATASET}.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
