#!/usr/bin/env python3
"""
Build the *hardest* BDA difficulty tier: 2-gene combinatorial selection (Horlbeck 2018, K562).

BioDiscoveryAgent's third tier asks for gene PAIRS whose combined knockdown shows a genetic
interaction — synergy or buffering — rather than single genes. The benchmark files ship with this
repo unchanged (`ground_truth_Horlbeck.csv`, `topmovers_Horlbeck.npy`, `task_prompts/Horlbeck.json`).

Encoding
--------
A candidate is ONE opaque token "GENEA|GENEB", the two symbols sorted alphabetically (BDA's own
`tuple(sorted(...))` convention, just serialised). The candidate unit was already an opaque string
everywhere below the prompt, so the oracle, the round loop, hit@R5, the result JSON and the analysis
code all work unchanged; only the LLM prompt/parser had to learn the "GENE1 + GENE2" wire format.

Pair features
-------------
Each gene-intrinsic feature f enters twice:

    f            = (f(a) + f(b)) / 2      pair mean   -- keeps the canonical column name, so every
                                                        arm's hardcoded FEATURE_COLS keeps working
                                                        (LightGBM is scale-invariant, so the mean is
                                                        equivalent to the sum f(a)+f(b))
    f_absdiff    = |f(a) - f(b)|          within-pair spread

The nine single-gene screens are written into the same table as DEGENERATE pairs (a = b): canonical
columns hold the gene's own value and every _absdiff is 0. That is what makes the leave-one-out
cross-experiment prior transfer to pairs at all — the single-gene screens are the training half and
Horlbeck is the held-out half, in one shared feature space. The consequence is stated plainly: the
spread columns are constant in the LOO training data, so the *static* prior cannot use them; only the
online model, once in-experiment pair feedback arrives, can.

Anchor features
---------------
Horlbeck gets no anchor genes, so `g1_ppi_score`, `archs4_coexpr` and `kegg_overlap` are 0 for every
pair. This is deliberate: no phenotype-specific anchor prior is invented for this screen, and the
transfer rests only on the six genuinely gene-intrinsic features.

Gene-level source
-----------------
The Horlbeck GI map is a map over *essential* genes, so most of its 448 genes are CEGv2 core
essentials and are absent from the frozen benchmark tables. Per-gene features therefore come from the
full-pool (easier-tier) tables built by build_tier_a_unfiltered.py, which must be run first.

Usage:
    conda run -n waddington-bio python3 workspace/evaluation/build_tier_a_unfiltered.py
    conda run -n waddington-bio python3 workspace/evaluation/build_horlbeck_pairs.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from waddington_select.tables import ABSDIFF_SUFFIX, pair_token  # noqa: E402

BDA_DIR = REPO_ROOT / "workspace" / "data" / "bda_benchmark"
GROUND_TRUTH = BDA_DIR / "ground_truth_Horlbeck.csv"
TOPMOVERS = BDA_DIR / "topmovers_Horlbeck.npy"

TIER_A_V1 = HERE / "lgbm_training_data_tier_a.csv"
TIER_A_V3 = HERE / "lgbm_training_data_tier_a_v3.csv"
OUT_V1 = HERE / "lgbm_training_data_horlbeck_pairs.csv"
OUT_V3 = HERE / "lgbm_training_data_horlbeck_pairs_v3.csv"

DATASET = "Horlbeck"
ANCHOR_FEATS = ["g1_ppi_score", "archs4_coexpr", "kegg_overlap"]


def load_pairs() -> tuple[list[str], dict[str, int], list[tuple[str, str]]]:
    gt = pd.read_csv(GROUND_TRUTH)
    raw = [ast.literal_eval(s) for s in gt["Gene_pairs"]]
    tokens = [pair_token(a, b) for a, b in raw]
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("duplicate pair tokens in the Horlbeck ground truth")

    tm = np.load(TOPMOVERS, allow_pickle=True)
    hits = {pair_token(a, b) for a, b in tm}
    labels = {t: int(t in hits) for t in tokens}

    missing = hits - set(tokens)
    if missing:
        raise RuntimeError(f"{len(missing)} labelled pairs are not in the candidate pool")
    print(f"  pool  : {len(tokens)} pairs over "
          f"{len({g for p in raw for g in p})} genes")
    print(f"  labels: {sum(labels.values())} hits ({100 * np.mean(list(labels.values())):.2f}%) "
          f"— BDA's two-tailed rule, ~5% most buffering + ~5% most synergistic by signed GI score")
    return tokens, labels, [(a.strip().upper(), b.strip().upper()) for a, b in raw]


def gene_feature_map(tier_a_csv: Path, feature_cols: list[str]) -> dict[str, np.ndarray]:
    """{GENE: feature vector} from the full-pool single-gene table (gene-intrinsic features only)."""
    df = pd.read_csv(tier_a_csv, usecols=["gene"] + feature_cols)
    df["gene"] = df["gene"].str.strip().str.upper()
    # Anchor features are screen-specific; zero them so no screen's anchor prior leaks into the pairs.
    for c in ANCHOR_FEATS:
        if c in df.columns:
            df[c] = 0.0
    df = df.drop_duplicates("gene").set_index("gene")
    return {g: row.to_numpy(dtype=float) for g, row in df[feature_cols].iterrows()}


def build(tier_a_csv: Path, out_csv: Path) -> pd.DataFrame:
    base = pd.read_csv(tier_a_csv)
    base["gene"] = base["gene"].str.strip().str.upper()
    feature_cols = [c for c in base.columns if c not in ("gene", "label", "dataset")]
    print(f"  {len(feature_cols)} features: {', '.join(feature_cols)}")

    tokens, labels, raw_pairs = load_pairs()
    fmap = gene_feature_map(tier_a_csv, feature_cols)

    genes = {g for p in raw_pairs for g in p}
    absent = sorted(g for g in genes if g not in fmap)
    if absent:
        print(f"  [NOTE] {len(absent)} pair genes have no single-gene feature row "
              f"(zero-filled): {', '.join(absent[:8])}{'...' if len(absent) > 8 else ''}")
    zero = np.zeros(len(feature_cols))

    A = np.stack([fmap.get(a, zero) for a, _ in raw_pairs])
    Bm = np.stack([fmap.get(b, zero) for _, b in raw_pairs])
    mean = (A + Bm) / 2.0
    spread = np.abs(A - Bm)

    pair_df = pd.DataFrame(mean, columns=feature_cols)
    for i, c in enumerate(feature_cols):
        pair_df[c + ABSDIFF_SUFFIX] = spread[:, i]
    pair_df.insert(0, "gene", tokens)
    pair_df["label"] = [labels[t] for t in tokens]
    pair_df["dataset"] = DATASET

    # The nine single-gene screens as degenerate pairs (a = b): spread is 0 by construction.
    single = base.copy()
    for c in feature_cols:
        single[c + ABSDIFF_SUFFIX] = 0.0

    cols = ["gene"] + feature_cols + [c + ABSDIFF_SUFFIX for c in feature_cols] + ["label", "dataset"]
    out = pd.concat([single[cols], pair_df[cols]], ignore_index=True)
    if out.isna().any().any():
        raise RuntimeError(f"NaNs in the pair table: "
                           f"{ {k: int(v) for k, v in out.isna().sum().items() if v} }")
    out.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv.name}: {len(out)} rows "
          f"({len(single)} single-gene + {len(pair_df)} pairs), {len(cols)} columns, "
          f"{out_csv.stat().st_size / 1e6:.0f} MB")
    return out


def main(check_only: bool = False) -> None:
    for p in (TIER_A_V1, TIER_A_V3):
        if not p.exists():
            raise SystemExit(f"{p.name} missing — run build_tier_a_unfiltered.py first")

    print("v1 (9 gene-intrinsic features -> 18 pair features)")
    v1 = build(TIER_A_V1, OUT_V1)
    print("\nv3 (+ 4 DepMap columns -> 26 pair features)")
    v3 = build(TIER_A_V3, OUT_V3)

    print("\nSanity checks:")
    for name, df in (("v1", v1), ("v3", v3)):
        sub = df[df["dataset"] == DATASET]
        n, h = len(sub), int(sub["label"].sum())
        best = 5 * 128
        print(f"  {name}: pool={n}, hits={h} ({100 * h / n:.2f}%), "
              f"random hit@R5 = {best / n * h / h if h else 0:.4f} "
              f"({best} picks -> expected {best * h / n:.1f} hits, recall {best / n:.4f}), "
              f"ceiling recall {min(best, h) / h:.4f}")
        assert sub["gene"].str.contains(r"\|").all(), "pair rows must carry | tokens"
        assert not df[df["dataset"] != DATASET]["gene"].str.contains(r"\|").any()
        spread_cols = [c for c in df.columns if c.endswith(ABSDIFF_SUFFIX)]
        assert (df[df["dataset"] != DATASET][spread_cols] == 0).all().all(), \
            "single-gene rows must have zero spread"
        assert sub[spread_cols].to_numpy().max() > 0, "pair rows must have non-zero spread"
    print("  OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    main(**vars(ap.parse_args()))
