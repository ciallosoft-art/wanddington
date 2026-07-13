"""
phenotype.py — onboard a genuinely NEW phenotype (a real screen that isn't one of the 9 benchmarks).

The benchmark datasets have precomputed ML features. A scientist's own screen does not — this builds
them, so the C-arm can rank their gene pool.

How the features split (verified against the training CSV):
  - gene-intrinsic (identical for a gene across every dataset) → looked up from the existing training
    data: hub_score_norm, ppi_score_sum, pli_score, string_degree_norm, kegg/reactome_pathway_count_norm,
    depmap_*.
  - anchor-relative (depend on the phenotype's *anchor* genes) → computed here for the new anchors:
    g1_ppi_score (STRING PPI expansion), archs4_coexpr (co-expression), kegg_overlap.
    The STRING/ARCHS4 work is cached per anchor, so this costs ~one call per anchor, not per gene.

The registered phenotype gets label=0 rows: it is never used as training data (see features.py), only
as the candidate pool. Round 1 therefore comes from zero-shot transfer (the model trained on the 9
benchmarks); from round 2 the online arm retrains on the hits the scientist uploads.

Usage:
    python -m waddington_select.phenotype register --name MyScreen \
        --genes-file screen.csv --anchors STAT1 JAK1 IFNGR1 \
        --task "..." --measurement "..." [--batch 32] [--expected-hit-rate 0.05] [--json]
    python -m waddington_select.phenotype list [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "workspace" / "evaluation"
BASE_CSV = EVAL_DIR / "lgbm_training_data_v3.csv"
USER_DIR = REPO_ROOT / "workspace" / "data" / "user_phenotypes"
REGISTRY_PATH = USER_DIR / "registry.json"

# Features that are identical for a gene in every dataset → reuse rather than recompute.
INTRINSIC_COLS = [
    "hub_score_norm", "ppi_score_sum", "pli_score", "string_degree_norm",
    "kegg_pathway_count_norm", "reactome_pathway_count_norm",
    "depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm", "depmap_K562_norm",
]
# Features that depend on this phenotype's anchor genes → computed per registration.
ANCHOR_COLS = ["g1_ppi_score", "archs4_coexpr", "kegg_overlap"]


# ── registry ────────────────────────────────────────────────────────────────

def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except Exception:
        return {}


def is_user_phenotype(name: str) -> bool:
    return name in load_registry()


def task_prompt(name: str) -> dict | None:
    """Task/Measurement for a registered phenotype (else None)."""
    entry = load_registry().get(name)
    if not entry:
        return None
    p = USER_DIR / name / "task_prompt.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"Task": name, "Measurement": "gene fitness"}


def features_frame(name: str) -> pd.DataFrame | None:
    """The registered phenotype's feature rows (the candidate pool), else None."""
    p = USER_DIR / name / "features.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ── feature construction ────────────────────────────────────────────────────

def _intrinsic_lookup() -> pd.DataFrame:
    """gene → the gene-intrinsic feature values, taken from the existing training data."""
    df = pd.read_csv(BASE_CSV, usecols=["gene"] + INTRINSIC_COLS)
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    return df.drop_duplicates("gene").set_index("gene")


def _anchor_features(genes: list[str], anchors: list[str]) -> dict[str, dict[str, float]]:
    """Compute the three anchor-relative features for this phenotype's anchors."""
    sys.path.insert(0, str(EVAL_DIR))
    import gene_ranker as gr  # noqa: E402  (lives in workspace/evaluation, not the package)

    g1 = gr.build_anchor_scores(anchors, verbose=False)
    a4 = gr.build_archs4_scores(anchors, verbose=False)
    kg = gr.build_kegg_scores(genes, anchors, verbose=False)
    return {"g1_ppi_score": g1, "archs4_coexpr": a4, "kegg_overlap": kg}


def register(
    name: str,
    genes: list[str],
    anchors: list[str],
    task: str,
    measurement: str,
    batch_size: int = 32,
    expected_hit_rate: float | None = None,
) -> dict:
    """Build features for a new phenotype's gene pool and register it."""
    if not name or "/" in name:
        raise SystemExit("--name must be a simple identifier")
    genes = sorted({g.strip().upper() for g in genes if str(g).strip()})
    anchors = [a.strip().upper() for a in anchors if str(a).strip()]
    if not genes:
        raise SystemExit("empty gene pool")
    if not anchors:
        raise SystemExit("at least one anchor gene is required (seed biology for this phenotype)")

    base_cols = pd.read_csv(BASE_CSV, nrows=0).columns.tolist()
    intrinsic = _intrinsic_lookup()
    anchor_feats = _anchor_features(genes, anchors)

    df = pd.DataFrame({"gene": genes})
    for col in ANCHOR_COLS:
        m = anchor_feats[col]
        df[col] = [float(m.get(g, 0.0)) for g in genes]
    joined = intrinsic.reindex(genes)
    for col in INTRINSIC_COLS:
        df[col] = joined[col].fillna(0.0).to_numpy()
    df["label"] = 0          # unknown — never used as training data (see features.py)
    df["dataset"] = name

    df = df.reindex(columns=base_cols, fill_value=0.0)

    out_dir = USER_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "features.csv", index=False)
    (out_dir / "task_prompt.json").write_text(
        json.dumps({"Task": task, "Measurement": measurement}, indent=2)
    )

    covered = int(intrinsic.reindex(genes).notna().all(axis=1).sum())
    entry = {
        "batch_size": int(batch_size),
        "anchors": anchors,
        "n_genes": len(genes),
        "n_genes_with_known_features": covered,
        "expected_hit_rate": expected_hit_rate,
        "created": pd.Timestamp.utcnow().isoformat(),
    }
    reg = load_registry()
    reg[name] = entry
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    return {"name": name, **entry, "features_csv": str(out_dir / "features.csv")}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Register a new phenotype (build its ML features).")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--genes-file", default=None, help="screen/library file — the gene pool")
    r.add_argument("--genes", nargs="*", default=None, help="explicit gene pool (alternative)")
    r.add_argument("--anchors", nargs="+", required=True, help="seed genes for this phenotype")
    r.add_argument("--task", required=True)
    r.add_argument("--measurement", required=True)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--expected-hit-rate", type=float, default=None)
    r.add_argument("--json", action="store_true")

    ls = sub.add_parser("list")
    ls.add_argument("--json", action="store_true")

    ds = sub.add_parser("datasets", help="every rankable phenotype (benchmarks + registered)")
    ds.add_argument("--json", action="store_true")

    args = p.parse_args()

    if args.cmd == "datasets":
        from .oracle import ALL_DATASETS
        if args.json:
            print(json.dumps(ALL_DATASETS))
        else:
            print("\n".join(ALL_DATASETS))
        return

    if args.cmd == "list":
        reg = load_registry()
        if args.json:
            print(json.dumps(reg))
            return
        if not reg:
            print("No user phenotypes registered.")
            return
        for n, e in reg.items():
            print(f"  {n:24s} genes={e['n_genes']:6d}  batch={e['batch_size']:4d}  anchors={', '.join(e['anchors'][:5])}")
        return

    genes = args.genes
    if args.genes_file:
        from .ingest import read_gene_pool
        genes = read_gene_pool(args.genes_file)
    if not genes:
        raise SystemExit("provide --genes-file or --genes")

    res = register(
        args.name, genes, args.anchors, args.task, args.measurement,
        batch_size=args.batch, expected_hit_rate=args.expected_hit_rate,
    )
    if args.json:
        print(json.dumps(res))
        return
    print(f"Registered phenotype '{res['name']}'")
    print(f"  gene pool:  {res['n_genes']} ({res['n_genes_with_known_features']} with known intrinsic features)")
    print(f"  anchors:    {', '.join(res['anchors'])}")
    print(f"  batch size: {res['batch_size']}")
    print(f"  features:   {res['features_csv']}")
    print(f"\nNow run:  python -m waddington_select.suggest --dataset {res['name']}")


if __name__ == "__main__":
    main()
