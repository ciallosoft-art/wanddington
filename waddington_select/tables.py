"""
tables.py — single source of truth for *which* feature/label table a run reads.

Every arm used to re-declare its own ``TRAINING_DATA_CSV`` constant pointing at the frozen v1 table,
so ``WADDINGTON_TRAINING_CSV`` (honoured only by the oracle) silently desynchronised them: the arms
selected from the frozen pool while the oracle scored a different one. Anything that needs a table
now imports it from here, so one env var moves the whole system together.

Env vars
--------
WADDINGTON_TRAINING_CSV      v1 table (9 gene-intrinsic features).  Default: lgbm_training_data.csv
WADDINGTON_TRAINING_CSV_V3   v3 table (v1 + 4 DepMap columns).      Default: lgbm_training_data_v3.csv
WADDINGTON_EXTRA_DATASETS    extra screens the oracle should serve, "Name:batch,Name:batch"
WADDINGTON_PAIR_DATASETS     screens whose candidates are gene PAIRS, comma-separated

Difficulty tiers
----------------
The paper's main results are BDA's *harder* single-gene tier (non-essential pool). The two other
BDA tiers are run by pointing these variables at purpose-built tables:

  easier   1-gene, full pool     WADDINGTON_TRAINING_CSV{,_V3}=lgbm_training_data_tier_a{,_v3}.csv
                                 (built by workspace/evaluation/build_tier_a_unfiltered.py)
  harder   2-gene combinations   WADDINGTON_TRAINING_CSV{,_V3}=…_horlbeck_pairs_mean{,_v3}.csv
                                 WADDINGTON_EXTRA_DATASETS=Horlbeck:128

A registered user phenotype (phenotype.py) carries its own single-gene tier instead of reading these
variables: ``non_essential`` (the default, the paper's headline tier: the 684 CEGv2 core-essential genes
are removed from its pool) or ``full_pool`` (they are kept, and its prior is trained on the full-pool
tables, the only ones whose screens include those genes). ``tier_tables`` maps a tier to its tables.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "workspace" / "evaluation"

DEFAULT_TRAINING_CSV = EVAL_DIR / "lgbm_training_data.csv"
DEFAULT_TRAINING_V3 = EVAL_DIR / "lgbm_training_data_v3.csv"
CRISPRA_VALIDATION_CSV = EVAL_DIR / "lgbm_training_data_crispra_validation.csv"

TRAINING_DATA_CSV = Path(os.environ.get("WADDINGTON_TRAINING_CSV", DEFAULT_TRAINING_CSV))
TRAINING_DATA_V3 = Path(os.environ.get("WADDINGTON_TRAINING_CSV_V3", DEFAULT_TRAINING_V3))

# ── Single-gene tiers for registered phenotypes ─────────────────────────────────────────────────
TIER_NON_ESSENTIAL = "non_essential"   # BDA's harder single-gene tier, the paper's headline
TIER_FULL_POOL = "full_pool"           # BDA's easier single-gene tier: core-essential genes kept
TIERS = (TIER_NON_ESSENTIAL, TIER_FULL_POOL)
FULL_POOL_CSV = EVAL_DIR / "lgbm_training_data_tier_a.csv"
FULL_POOL_V3 = EVAL_DIR / "lgbm_training_data_tier_a_v3.csv"
# Hart et al. (2017) CEGv2 core-essential genes, as distributed with BioDiscoveryAgent (684 genes,
# GENE / HGNC_ID / ENTREZ_ID); the list BDA and our benchmark tables remove from every screen's pool.
CEG_PATH = Path(__file__).resolve().parent / "data" / "CEGv2.txt"


def tier_tables(tier: str) -> tuple[Path, Path]:
    """(v1, v3) training tables for a single-gene tier."""
    if tier == TIER_NON_ESSENTIAL:
        return TRAINING_DATA_CSV, TRAINING_DATA_V3
    if tier == TIER_FULL_POOL:
        return FULL_POOL_CSV, FULL_POOL_V3
    raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")


def core_essential_genes() -> set[str]:
    """The CEGv2 core-essential gene symbols (upper-case)."""
    df = pd.read_csv(CEG_PATH, sep="\t", usecols=["GENE"])
    return set(df["GENE"].astype(str).str.strip().str.upper())

# ── Combinatorial (2-gene) screens ──────────────────────────────────────────────────────────────
# A pair candidate is ONE opaque token "GENEA|GENEB" with the two symbols sorted alphabetically, so
# every layer below the prompt (oracle, runner, hit@R5, result JSON) is unchanged — the candidate
# unit was already an opaque string end to end. Only the LLM prompt/parse needs to know.
PAIR_SEP = "|"
_DEFAULT_PAIR_DATASETS = {"Horlbeck"}
PAIR_DATASETS: set[str] = _DEFAULT_PAIR_DATASETS | {
    s.strip() for s in os.environ.get("WADDINGTON_PAIR_DATASETS", "").split(",") if s.strip()
}


def is_pair_dataset(dataset_name: str) -> bool:
    return dataset_name in PAIR_DATASETS


def pair_token(gene_a: str, gene_b: str) -> str:
    """Canonical pair token: alphabetically sorted, matching BDA's `tuple(sorted(...))` convention."""
    a, b = sorted((gene_a.strip().upper(), gene_b.strip().upper()))
    return f"{a}{PAIR_SEP}{b}"


def split_pair(token: str) -> tuple[str, str]:
    a, _, b = token.partition(PAIR_SEP)
    return a, b


# ── Feature columns ─────────────────────────────────────────────────────────────────────────────
# A pair table carries each gene-intrinsic feature twice: the canonical column name holds the pair
# MEAN (x(a)+x(b))/2 — scale-equivalent to the sum, and LightGBM is scale-invariant — and a
# "<feat>_absdiff" companion holds |x(a)-x(b)|, the within-pair spread. Single-gene tables have no
# _absdiff columns, so this helper is a strict no-op on every existing run.
ABSDIFF_SUFFIX = "_absdiff"
# True interaction features x(g_a, g_b) -- co-essentiality, co-pathway, the STRING a--b edge. Unlike
# the mean and the spread, these are NOT symmetric functions of the two marginals, which is what a
# genetic interaction (deviation from the additive expectation) actually requires. Built by
# workspace/evaluation/build_pair_interaction_features.py; absent from every single-gene table, so
# picking them up by prefix is a no-op there.
INTERACTION_PREFIX = "pair_"


def with_pair_features(feats: list[str], df: pd.DataFrame) -> list[str]:
    """Extend a single-gene feature list with everything a pair table additionally provides.

    Two kinds are added: the |x(a)-x(b)| companion of each base feature, and any interaction column.
    On a single-gene table neither exists, so the result is the input unchanged.
    """
    out = list(feats) + [
        f"{c}{ABSDIFF_SUFFIX}" for c in feats if f"{c}{ABSDIFF_SUFFIX}" in df.columns
    ]
    out += [c for c in df.columns
            if c.startswith(INTERACTION_PREFIX) and c not in out]
    return out


# Historical name, kept so nothing breaks mid-refactor.
with_pair_spread = with_pair_features


# ── Extra screens the oracle should serve ───────────────────────────────────────────────────────

def extra_datasets() -> dict[str, int]:
    """Parse WADDINGTON_EXTRA_DATASETS ("Name:batch,Name:batch") into {name: batch_size}."""
    spec = os.environ.get("WADDINGTON_EXTRA_DATASETS", "").strip()
    out: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, batch = item.partition(":")
        name = name.strip()
        if not name:
            continue
        try:
            out[name] = int(batch) if batch.strip() else 128
        except ValueError as exc:
            raise ValueError(
                f"WADDINGTON_EXTRA_DATASETS entry {item!r} is not 'Name:batch_size'"
            ) from exc
    # Back-compat: the CRISPRa reason-vs-recall validation used to be registered whenever ANY table
    # override was set, which would misfire for every other superset table. Bind it to its own file.
    if TRAINING_DATA_CSV == CRISPRA_VALIDATION_CSV:
        out.setdefault("IL2_crispra", 128)
        out.setdefault("IFNG_crispra", 128)
    return out
