"""
ingest.py — turn a CRISPR screen readout file into hit/no-hit labels for a tested batch.

This is the *real experiment* counterpart to the benchmark oracle (`tools.reveal`): instead of the
ground truth, the scientist provides their own screen readout each round, and we derive which of the
tested genes hit.

Accepted files:
  - MAGeCK `gene_summary.txt` (tab-delimited; score columns e.g. `pos|lfc` / `neg|lfc`, `pos|fdr` / `neg|fdr`)
  - a `Gene,Score` CSV (like BDA's `ground_truth_*.csv`)
  - any table with a gene-id column + a numeric score column, or an explicit hit/label column

Hit rules. A registered phenotype fixes one when it is registered (phenotype.register --hit-rule), so a
campaign calls hits the same way every round and the rule is decided before any result is seen:
  label         the readout carries a hit/label column (hit / is_hit / label ...), used as is
  threshold     an absolute cutoff on the score: |score| >= X (direction both), score >= X (up), score <= -X (down)
  fdr           an FDR cutoff, fdr < q: MAGeCK's pos|fdr (up), neg|fdr (down) or the smaller of the two (both),
                else one fdr / padj / qvalue column (with up/down, the score's sign picks the direction)
  top_fraction  the top fraction of the screen by |score| (both), score (up) or -score (down): BDA's
                convention for its benchmark labels. It ranks a gene against every other gene of the screen, so
                it needs the WHOLE screen's readout (a completed screen being replayed); on one round's batch
                it would call a fixed ~5% of whatever was tested a hit, whatever the effects.

Scope. With the candidate pool known (--dataset), a readout covering at least 90% of the pool is a full
screen: total_hits is its hit count within the pool, the hit@R5 denominator. Anything less is one batch:
total_hits is None, since the screen-wide hit count is unknown until the whole screen has been run, and the
ranking rules (top_fraction, and the legacy gaussian / castle) are refused. For a phenotype in the
non-essential tier the CEGv2 core-essential genes are removed from the hits after the call, as the benchmark
labels are (topmovers minus CEGv2).

Without a registered rule the legacy options apply: a hit/label column if present, else --method gaussian
(the top --top-ratio by |score|) / castle (> 95th percentile) / --threshold X (score > X).

Usage:
    python -m waddington_select.ingest --file screen.txt --genes STAT1 JAK2 ACTB --dataset MyScreen --json
    python -m waddington_select.ingest --file screen.txt --genes STAT1 JAK2 ACTB \\
        [--score-col "pos|lfc"] [--top-ratio 0.05] [--method gaussian|castle] [--threshold X] --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_CANDIDATES = ["gene", "id", "gene_symbol", "symbol", "genes"]
SCORE_CANDIDATES = ["score", "log-fold-change", "lfc", "pos|lfc", "neg|lfc", "logfc", "z", "zscore"]
HIT_CANDIDATES = ["hit", "is_hit", "hits", "label", "hit_label"]
FDR_CANDIDATES = ["fdr", "padj", "p_adj", "adj_p", "qvalue", "q_value", "q"]

HIT_RULE_TYPES = ("label", "threshold", "fdr", "top_fraction")
DIRECTIONS = ("both", "up", "down")
FULL_SCREEN_COVERAGE = 0.9   # share of the candidate pool a readout must cover to count as the whole screen

_TRUE = {"1", "true", "yes", "y", "hit", "t"}
_FALSE = {"0", "false", "no", "n", "miss", "f"}


def _norm(genes) -> list[str]:
    return list(dict.fromkeys(str(g).strip().upper() for g in (genes or []) if str(g).strip()))


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _read_table(path: str) -> pd.DataFrame:
    """Read a screen file (.csv → comma; .txt/.tsv/MAGeCK gene_summary → tab) and add _GENE."""
    if Path(path).suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")
    id_col = _pick(list(df.columns), ID_CANDIDATES) or df.columns[0]
    df["_GENE"] = df[id_col].astype("string").str.strip().str.upper()
    return df.loc[df["_GENE"].notna() & df["_GENE"].ne("")].copy()


def read_gene_pool(path: str) -> list[str]:
    """Every gene in a screen/library file — the candidate pool for a new phenotype."""
    df = _read_table(path)
    return sorted({g for g in df["_GENE"].tolist() if g and g != "NAN"})


# ── hit rules ─────────────────────────────────────────────────────────────────────────────────────

def make_hit_rule(kind: str, *, threshold: float | None = None, direction: str = "both",
                  score_col: str | None = None, fdr_col: str | None = None, max_fdr: float | None = None,
                  fraction: float | None = None) -> dict:
    """A validated hit rule (see the module docstring)."""
    if kind not in HIT_RULE_TYPES:
        raise ValueError(f"hit rule must be one of {', '.join(HIT_RULE_TYPES)}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {', '.join(DIRECTIONS)}")
    if kind == "label":
        return {"type": "label"}
    if kind == "threshold":
        if threshold is None or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("a threshold rule needs a positive cutoff X: |score| >= X (both), score >= X (up), "
                             "score <= -X (down)")
        return {"type": "threshold", "threshold": float(threshold), "direction": direction, "score_col": score_col}
    if kind == "fdr":
        if max_fdr is None or not np.isfinite(max_fdr) or not 0 < max_fdr <= 1:
            raise ValueError("an FDR rule needs a cutoff q in (0, 1]")
        return {"type": "fdr", "max_fdr": float(max_fdr), "direction": direction, "fdr_col": fdr_col}
    if fraction is None or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("a top_fraction rule needs a fraction in (0, 1]")
    return {"type": "top_fraction", "fraction": float(fraction), "direction": direction, "score_col": score_col}


def describe_hit_rule(rule: dict | None) -> str:
    if not rule:
        return "legacy options (no registered rule)"
    t, d = rule["type"], rule.get("direction", "both")
    if t == "label":
        return "the readout's hit/label column"
    if t == "threshold":
        x = rule["threshold"]
        return {"both": f"|score| >= {x:g}", "up": f"score >= {x:g}", "down": f"score <= -{x:g}"}[d]
    if t == "fdr":
        return f"FDR < {rule['max_fdr']:g}" + ("" if d == "both" else f" ({d})")
    by = {"both": "|score|", "up": "score", "down": "-score"}[d]
    return f"top {rule['fraction']:.0%} of the whole screen by {by}"


def _labels(df: pd.DataFrame, hit_col: str) -> tuple[set, set]:
    text = df[hit_col].astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(df[hit_col], errors="coerce")
    is_hit = (text.isin(_TRUE) | numeric.eq(1)).fillna(False)
    is_miss = (text.isin(_FALSE) | numeric.eq(0)).fillna(False)
    missing = df[hit_col].isna() | text.eq("").fillna(False)
    invalid = ~(is_hit | is_miss | missing)
    if invalid.any():
        raise ValueError(f"Unrecognised hit labels in {hit_col}: "
                         f"{df.loc[invalid, hit_col].astype(str).unique()[:5].tolist()}")
    hitset = set(df.loc[is_hit, "_GENE"])
    negative = set(df.loc[is_miss, "_GENE"])
    if hitset & negative:
        raise ValueError(f"Conflicting labels for genes: {', '.join(sorted(hitset & negative))}")
    return hitset, hitset | negative


def _scores(df: pd.DataFrame, score_col: str | None) -> tuple[str, np.ndarray, np.ndarray]:
    sc = score_col or _pick(list(df.columns), SCORE_CANDIDATES)
    if sc is None:
        raise ValueError(f"No score/hit column found. Columns: {list(df.columns)}. Pass --score-col explicitly.")
    if sc not in df.columns:
        raise ValueError(f"Score column {sc!r} is not in the readout")
    vals = pd.to_numeric(df[sc], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if not finite.any():
        raise ValueError(f"No finite scores found in {sc!r}")
    return sc, vals, finite


def _top(key: np.ndarray, finite: np.ndarray, fraction: float) -> np.ndarray:
    valid_idx = np.flatnonzero(finite)
    order = valid_idx[np.argsort(key[finite], kind="stable")[::-1]]
    return order[:max(1, int(finite.sum() * fraction))]


def _apply_rule(df: pd.DataFrame, rule: dict) -> tuple[set, set, str, str | None, bool]:
    """(hitset, observed genes, method, column used, whether the rule ranks against the whole screen)"""
    genes = df["_GENE"].to_numpy()
    t = rule["type"]
    if t == "label":
        hit_col = _pick(list(df.columns), HIT_CANDIDATES)
        if hit_col is None:
            raise ValueError("The registered hit rule reads a hit/label column (hit / is_hit / label); "
                             f"none in this readout. Columns: {list(df.columns)}")
        hitset, observed = _labels(df, hit_col)
        return hitset, observed, f"label:{hit_col}", None, False
    d = rule.get("direction", "both")
    if t == "threshold":
        sc, vals, finite = _scores(df, rule.get("score_col"))
        x = rule["threshold"]
        with np.errstate(invalid="ignore"):
            hit = {"both": np.abs(vals) >= x, "up": vals >= x, "down": vals <= -x}[d] & finite
        return set(genes[hit]), set(genes[finite]), describe_hit_rule(rule), sc, False
    if t == "fdr":
        cols = list(df.columns)
        col = rule.get("fdr_col")
        pos, neg = _pick(cols, ["pos|fdr"]), _pick(cols, ["neg|fdr"])
        if col is None and pos and neg:   # MAGeCK: one FDR per direction
            fp = pd.to_numeric(df[pos], errors="coerce").to_numpy(dtype=float)
            fn = pd.to_numeric(df[neg], errors="coerce").to_numpy(dtype=float)
            f = {"up": fp, "down": fn, "both": np.fmin(fp, fn)}[d]
            used, sign_ok = {"up": pos, "down": neg, "both": f"min({pos}, {neg})"}[d], np.ones(len(df), bool)
        else:
            col = col or _pick(cols, FDR_CANDIDATES)
            if col is None or col not in df.columns:
                raise ValueError(f"No FDR column found (fdr / padj / qvalue, or MAGeCK pos|fdr and neg|fdr). "
                                 f"Columns: {cols}")
            f = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            used = col
            if d == "both":
                sign_ok = np.ones(len(df), bool)
            else:  # a single FDR column: the score's sign gives the direction
                _, s, _ = _scores(df, None)
                with np.errstate(invalid="ignore"):
                    sign_ok = s > 0 if d == "up" else s < 0
        finite = np.isfinite(f)
        if not finite.any():
            raise ValueError(f"No finite FDR values in {used}")
        with np.errstate(invalid="ignore"):
            hit = finite & (f < rule["max_fdr"]) & sign_ok
        return set(genes[hit]), set(genes[finite]), describe_hit_rule(rule), used, False
    sc, vals, finite = _scores(df, rule.get("score_col"))
    key = {"both": np.abs(vals), "up": vals, "down": -vals}[d]
    return set(genes[_top(key, finite, rule["fraction"])]), set(genes[finite]), describe_hit_rule(rule), sc, True


def _apply_legacy(df: pd.DataFrame, score_col, method, top_ratio, threshold) -> tuple[set, set, str, str | None, bool]:
    hit_col = _pick(list(df.columns), HIT_CANDIDATES)
    if hit_col is not None:
        hitset, observed = _labels(df, hit_col)
        return hitset, observed, f"label:{hit_col}", None, False
    sc, vals, finite = _scores(df, score_col)
    genes = df["_GENE"].to_numpy()
    observed = set(genes[finite])
    if threshold is not None:
        return set(genes[finite & (vals > threshold)]), observed, f"threshold>{threshold}", sc, False
    if method == "castle":
        thr = np.nanpercentile(vals[finite], 95)
        return set(genes[finite & (vals > thr)]), observed, "castle(>p95)", sc, True
    return set(genes[_top(np.abs(vals), finite, top_ratio)]), observed, f"gaussian(top{top_ratio:g})", sc, True


def derive_hits(
    path: str,
    genes,
    *,
    score_col: str | None = None,
    method: str = "gaussian",
    top_ratio: float = 0.05,
    threshold: float | None = None,
    rule: dict | None = None,
    pool=None,
    exclude=None,
) -> dict:
    """Label the tested `genes` from a readout; absent genes and missing measurements stay unknown.

    `rule` is a registered hit rule (make_hit_rule) and replaces the legacy options. `pool` is the phenotype's
    candidate pool: with it the readout is classed as a full screen or one batch (see the module docstring).
    `exclude` (the core-essential genes, for a non-essential-tier phenotype) is removed from the hits."""
    if not np.isfinite(top_ratio) or not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1]")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if method not in {"gaussian", "castle"}:
        raise ValueError(f"Unknown hit derivation method: {method}")
    df = _read_table(path)
    if rule is not None:
        hitset, observed, used_method, used_col, ranks_screen = _apply_rule(df, rule)
    else:
        hitset, observed, used_method, used_col, ranks_screen = _apply_legacy(df, score_col, method, top_ratio,
                                                                               threshold)

    scope, coverage, pool_set = None, None, None
    if pool is not None:
        pool_set = set(_norm(pool))
        coverage = len(observed & pool_set) / len(pool_set) if pool_set else 0.0
        scope = "full_screen" if coverage >= FULL_SCREEN_COVERAGE else "batch"
        if ranks_screen and scope == "batch":
            raise ValueError(
                f"The hit call '{used_method}' ranks each gene against the whole screen, but this readout covers "
                f"only {coverage:.0%} of the {len(pool_set)}-gene candidate pool (one round's batch?). Upload the "
                f"full screen's readout, or register an absolute hit rule: a hit/label column, a score cutoff "
                f"(|score| >= X) or an FDR cutoff.")
    if exclude:
        ex = set(_norm(exclude))
        hitset, observed = hitset - ex, observed - ex
    if pool_set is None:
        total = len(hitset)                 # legacy: no pool known, the readout is taken as the screen
    else:
        total = len(hitset & pool_set) if scope == "full_screen" else None

    batch = _norm(genes)
    hits = [g for g in batch if g in hitset]
    misses = [g for g in batch if g in observed and g not in hitset]
    unknown = [g for g in batch if g not in observed]
    return {
        "tool": "ingest",
        "tested": len(batch),
        "hits": hits,
        "misses": misses,
        "n_hits": len(hits),
        "total_hits": total,
        "unknown_genes": unknown,
        "method": used_method,
        "score_col": used_col,
        "rule": describe_hit_rule(rule),
        "scope": scope,
        "pool_coverage": None if coverage is None else round(coverage, 4),
    }


def dataset_context(dataset: str) -> dict:
    """The registered hit rule, candidate pool and excluded genes of a phenotype (registered or benchmark)."""
    from .phenotype import features_frame, load_registry
    from .tables import TIER_NON_ESSENTIAL, core_essential_genes
    entry = load_registry().get(dataset)
    if entry is not None:
        feats = features_frame(dataset)
        pool = [] if feats is None else feats["gene"].astype(str).tolist()
        exclude = core_essential_genes() if entry.get("tier") == TIER_NON_ESSENTIAL else set()
        return {"rule": entry.get("hit_rule"), "pool": pool, "exclude": exclude}
    from .oracle import ALL_DATASETS
    if dataset not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}")
    from .features import load_training_frame
    from .tables import TRAINING_DATA_CSV
    df = load_training_frame(TRAINING_DATA_CSV, dataset)
    pool = df.loc[df["dataset"] == dataset, "gene"].astype(str).tolist()
    # the benchmark pools are the non-essential tier: labels are topmovers minus CEGv2
    return {"rule": None, "pool": pool, "exclude": core_essential_genes()}


def main() -> None:
    p = argparse.ArgumentParser(description="Derive hits for a tested batch from a screen readout file.")
    p.add_argument("--file", required=True, help="MAGeCK gene_summary.txt / Gene,Score CSV / labelled table")
    p.add_argument("--genes", nargs="+", required=True, help="the tested batch")
    p.add_argument("--dataset", default=None,
                   help="the phenotype: its registered hit rule, candidate pool and tier apply")
    p.add_argument("--score-col", default=None, help="score column (else auto: Score/log-fold-change/pos|lfc/…)")
    p.add_argument("--method", default=None, choices=["gaussian", "castle"], help="legacy (default gaussian)")
    p.add_argument("--top-ratio", type=float, default=None, help="legacy gaussian hit fraction (default 0.05)")
    p.add_argument("--threshold", type=float, default=None, help="legacy absolute cutoff, score > X")
    p.add_argument("--json", action="store_true", help="emit JSON only")
    args = p.parse_args()

    legacy = {"score_col": args.score_col, "method": args.method, "top_ratio": args.top_ratio,
              "threshold": args.threshold}
    try:
        ctx = dataset_context(args.dataset) if args.dataset else {"rule": None, "pool": None, "exclude": None}
        ignored = [k for k, v in legacy.items() if v is not None] if ctx["rule"] else []
        res = derive_hits(
            args.file, args.genes,
            score_col=args.score_col, method=args.method or "gaussian",
            top_ratio=0.05 if args.top_ratio is None else args.top_ratio, threshold=args.threshold,
            rule=ctx["rule"], pool=ctx["pool"], exclude=ctx["exclude"],
        )
        res["overrides_ignored"] = ignored   # the registered rule is fixed for the whole campaign
    except ValueError as exc:
        p.error(str(exc))
    if args.json:
        json.dump(res, sys.stdout)
        sys.stdout.write("\n")
        return
    print(f"Screen readout: {args.file}  (hit call: {res['method']}, column: {res['score_col']})")
    if res["scope"]:
        print(f"Readout covers {res['pool_coverage']:.0%} of the candidate pool → {res['scope'].replace('_', ' ')}")
    print(f"Screen-wide hits: {res['total_hits'] if res['total_hits'] is not None else 'unknown (one batch)'}")
    if res["overrides_ignored"]:
        print(f"[note] the registered hit rule applies; ignored: {', '.join(res['overrides_ignored'])}")
    if res["unknown_genes"]:
        print(f"[warn] missing gene or measurement (not labelled): {', '.join(res['unknown_genes'])}")
    print(f"\nTested {res['tested']} genes → {res['n_hits']} hits:")
    print("  hits:   " + (", ".join(res["hits"]) or "(none)"))
    print("  misses: " + (", ".join(res["misses"]) or "(none)"))


if __name__ == "__main__":
    main()
