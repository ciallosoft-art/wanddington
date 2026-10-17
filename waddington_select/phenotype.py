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
    Anchors are OPTIONAL: the reported C-arm drops these three features (waddington_c_arm
    _drop_anchor_default), so they only matter to the legacy anchor-feature configuration
    (WADDINGTON_LEGACY_ROUTER=1 or WADDINGTON_DROP_ANCHOR_FEATS=0). Without anchors they are 0.

The registered phenotype gets label=0 rows: it is never used as training data (see features.py), only
as the candidate pool. Round 1 therefore comes from zero-shot transfer (the model trained on the 9
benchmarks); from round 2 the online arm retrains on the hits the scientist uploads.

Single-gene tier (tables.py): ``non_essential`` (default, the paper's headline tier) removes the 684
CEGv2 core-essential genes from the pool, as the benchmark tables do; ``full_pool`` keeps them, and the
prior is then trained on the full-pool tables -- the only tables whose screens contain those genes, so
also the only place their gene-intrinsic features can be looked up (the non-essential tables have none
of them, and a gene missing from the lookup gets all-zero features).

Design metadata (router_protocol.feature_policy_from): the perturbation (KO / CRISPRi / CRISPRa, or
unknown), the cell line and whether the library is a curated essential-gene panel -- known before the
screen is run. They set which DepMap features the prior uses, by the same rule as the benchmarks: none for
CRISPRa (DepMap is knockout-derived) or a curated essentiality panel (it would restate the selection),
pan-cancer only in K562, else pan-cancer + K562. An unknown modality is treated as loss-of-function.

Hit rule (ingest.py): how a tested gene is called a hit from the scientist's readout, fixed at
registration so every round is called the same way and the rule is chosen before any result is seen:
``label`` (the readouts carry a hit column), ``threshold`` (|score| >= X, or one direction), ``fdr``
(fdr < q) or ``top_fraction`` (the top x% of the whole screen -- only for replaying a completed screen;
ingest refuses it on one round's batch). Without one, ingest falls back to its legacy options.

Usage:
    python -m waddington_select.phenotype register --name MyScreen \
        --genes-file screen.csv --task "..." --measurement "..." \
        [--modality KO|CRISPRi|CRISPRa|unknown] [--cell-line K562] [--curated-essential] \
        [--anchors STAT1 JAK1 IFNGR1] [--tier non_essential|full_pool] \
        [--hit-rule label|threshold|fdr|top_fraction [--threshold X] [--max-fdr Q] [--top-fraction F]
         [--direction both|up|down] [--score-col C] [--fdr-col C]] \
        [--batch 32] [--expected-hit-rate 0.05] [--json]
    python -m waddington_select.phenotype list [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "workspace" / "evaluation"
from .tables import TIER_NON_ESSENTIAL, TIERS, core_essential_genes, tier_tables  # noqa: E402
from .ingest import DIRECTIONS, HIT_RULE_TYPES, describe_hit_rule, make_hit_rule  # noqa: E402
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

MODALITIES = ("KO", "CRISPRi", "CRISPRa", "unknown")
_MODALITY_WORDS = [  # checked in order: CRISPRa/CRISPRi before KO, so "dCas9-KRAB" is not a knockout
    ("CRISPRa", re.compile(r"crispr-?a\b|activation|over-?express|gain.of.function|vp64|\bvpr\b|\bsam\b", re.I)),
    ("CRISPRi", re.compile(r"crispr-?i\b|interference|knock-?down|krab", re.I)),
    ("KO", re.compile(r"\bko\b|knock-?out|crispr-?ko|\bcas9\b|deletion", re.I)),
    ("unknown", re.compile(r"^(?:unknown|unsure|not sure|don'?t know|n/?a|\?)$", re.I)),
]


def normalize_modality(value: str | None) -> str:
    """KO / CRISPRi / CRISPRa / unknown from a free-text perturbation ("CRISPR knockout", "dCas9-VP64" …);
    None or blank is unknown. Raises ValueError for anything else."""
    text = str(value or "").strip()
    if not text:
        return "unknown"
    for modality, pattern in _MODALITY_WORDS:
        if pattern.search(text):
            return modality
    raise ValueError(f"unrecognised modality {value!r}: use one of {', '.join(MODALITIES)}")


def normalize_cell_line(value: str | None) -> str | None:
    """'K562' for any spelling the rule must recognise (k562, K-562, "K562 cells"), else the text as given."""
    text = " ".join(str(value or "").split())
    if not text:
        return None
    return "K562" if re.search(r"(?<![A-Za-z0-9])K-?562(?![0-9])", text, re.I) else text[:80]


DEPMAP_POLICY_TEXT = {
    "none": "off (CRISPRa or a curated essential panel: a knockout essentiality prior would mislead)",
    "pan": "pan-cancer features only (K562: the K562 feature would restate the label on its own cell line)",
    "pan_k562": "pan-cancer + K562 features",
}


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


def phenotype_tier(name: str) -> str | None:
    """A registered phenotype's single-gene tier (entries made before tiers existed are non-essential
    in intent but were not filtered; they report ``None``), else None."""
    entry = load_registry().get(name)
    return entry.get("tier") if entry else None


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

def _intrinsic_lookup(base_csv: Path) -> pd.DataFrame:
    """gene → the gene-intrinsic feature values, taken from the existing training data."""
    df = pd.read_csv(base_csv, usecols=["gene"] + INTRINSIC_COLS)
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
    tier: str = TIER_NON_ESSENTIAL,
    hit_rule: dict | None = None,
    modality: str | None = None,
    cell_line: str | None = None,
    curated_essential: bool = False,
) -> dict:
    """Build features for a new phenotype's gene pool and register it. ``anchors`` may be empty."""
    from .oracle import BENCHMARK_DATASETS  # lazy: oracle imports this module
    from .router_protocol import feature_policy_from

    if not name or "/" in name:
        raise SystemExit("--name must be a simple identifier")
    if name in BENCHMARK_DATASETS:
        raise SystemExit(f"--name {name!r} is a benchmark screen; choose another name")
    if tier not in TIERS:
        raise SystemExit(f"--tier must be one of {', '.join(TIERS)}")
    try:
        metadata = {"modality": normalize_modality(modality), "cell_line": normalize_cell_line(cell_line),
                    "curated_ess": bool(curated_essential)}
    except ValueError as exc:
        raise SystemExit(str(exc))
    genes = sorted({g.strip().upper() for g in genes if str(g).strip()})
    anchors = [a.strip().upper() for a in (anchors or []) if str(a).strip()]
    if not genes:
        raise SystemExit("empty gene pool")
    ceg = core_essential_genes()
    n_core_essential = len(set(genes) & ceg)
    if tier == TIER_NON_ESSENTIAL:
        genes = [g for g in genes if g not in ceg]
        if not genes:
            raise SystemExit("every gene in the pool is a CEGv2 core-essential gene; use --tier full_pool")

    base_csv = tier_tables(tier)[1]
    base_cols = pd.read_csv(base_csv, nrows=0).columns.tolist()
    intrinsic = _intrinsic_lookup(base_csv)
    anchor_feats = _anchor_features(genes, anchors) if anchors else {col: {} for col in ANCHOR_COLS}

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
        "tier": tier,
        # design metadata -> which DepMap features the prior uses (router_protocol.feature_policy_from)
        "metadata": metadata,
        "depmap_policy": feature_policy_from(metadata)["depmap"],
        "hit_rule": hit_rule,   # ingest.make_hit_rule; None = ingest's legacy options
        # core-essential genes in the uploaded pool: removed under non_essential, kept under full_pool
        "n_core_essential_in_upload": n_core_essential,
        "n_core_essential_removed": n_core_essential if tier == TIER_NON_ESSENTIAL else 0,
        "n_genes": len(genes),
        "n_genes_with_known_features": covered,
        "expected_hit_rate": expected_hit_rate,
        "created": pd.Timestamp.utcnow().isoformat(),
    }
    reg = load_registry()
    reg[name] = entry
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    return {"name": name, **entry, "hit_rule_text": describe_hit_rule(hit_rule),
            "features_csv": str(out_dir / "features.csv")}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Register a new phenotype (build its ML features).")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--genes-file", default=None, help="screen/library file — the gene pool")
    r.add_argument("--genes", nargs="*", default=None, help="explicit gene pool (alternative)")
    r.add_argument("--anchors", nargs="*", default=[],
                   help="optional seed genes; only the legacy anchor-feature configuration uses them")
    r.add_argument("--modality", default=None,
                   help="the perturbation: KO, CRISPRi, CRISPRa or unknown (default; treated as loss-of-function). "
                        "CRISPRa switches the knockout-derived DepMap features off")
    r.add_argument("--cell-line", default=None, help="the cell line / type (K562 switches the K562 DepMap feature off)")
    r.add_argument("--curated-essential", action="store_true",
                   help="the library is a curated essential-gene panel (switches the DepMap features off)")
    r.add_argument("--task", required=True)
    r.add_argument("--measurement", required=True)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--expected-hit-rate", type=float, default=None)
    r.add_argument("--tier", choices=TIERS, default=TIER_NON_ESSENTIAL,
                   help="non_essential (default, the paper's headline tier) removes the 684 CEGv2 core-essential "
                        "genes from the pool; full_pool keeps them and trains the prior on the full-pool tables")
    r.add_argument("--hit-rule", choices=HIT_RULE_TYPES, default=None,
                   help="how a tested gene is called a hit from your readouts (fixed for the campaign): label = "
                        "they carry a hit column; threshold = |score| >= X; fdr = fdr < q; top_fraction = top "
                        "share of the WHOLE screen (replaying a completed screen only)")
    r.add_argument("--threshold", type=float, default=None, help="threshold rule: the cutoff X (> 0)")
    r.add_argument("--max-fdr", type=float, default=None, help="fdr rule: the cutoff q")
    r.add_argument("--top-fraction", type=float, default=None, help="top_fraction rule: the share, e.g. 0.05")
    r.add_argument("--direction", choices=DIRECTIONS, default="both",
                   help="both (|score|), up (score >= X / MAGeCK pos) or down (score <= -X / MAGeCK neg)")
    r.add_argument("--score-col", default=None, help="score column (else auto-detected)")
    r.add_argument("--fdr-col", default=None, help="FDR column (else auto: MAGeCK pos|fdr / neg|fdr, fdr, padj …)")
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
            m = e.get("metadata") or {}
            print(f"  {n:24s} genes={e['n_genes']:6d}  batch={e['batch_size']:4d}  tier={e.get('tier') or '-':13s} "
                  f"modality={m.get('modality') or '-':8s} cell_line={m.get('cell_line') or '-'}")
        return

    genes = args.genes
    if args.genes_file:
        from .ingest import read_gene_pool  # noqa: F811
        genes = read_gene_pool(args.genes_file)
    if not genes:
        raise SystemExit("provide --genes-file or --genes")

    hit_rule = None
    if args.hit_rule:
        try:
            hit_rule = make_hit_rule(args.hit_rule, threshold=args.threshold, direction=args.direction,
                                     score_col=args.score_col, fdr_col=args.fdr_col, max_fdr=args.max_fdr,
                                     fraction=args.top_fraction)
        except ValueError as exc:
            p.error(str(exc))
    try:
        normalize_modality(args.modality)
    except ValueError as exc:
        p.error(str(exc))
    res = register(
        args.name, genes, args.anchors, args.task, args.measurement,
        batch_size=args.batch, expected_hit_rate=args.expected_hit_rate, tier=args.tier, hit_rule=hit_rule,
        modality=args.modality, cell_line=args.cell_line, curated_essential=args.curated_essential,
    )
    if args.json:
        print(json.dumps(res))
        return
    print(f"Registered phenotype '{res['name']}'")
    print(f"  gene pool:  {res['n_genes']} ({res['n_genes_with_known_features']} with known intrinsic features)")
    print(f"  tier:       {res['tier']} ({res['n_core_essential_removed']} core-essential genes removed, "
          f"{res['n_core_essential_in_upload']} in the upload)")
    print(f"  hit rule:   {describe_hit_rule(res['hit_rule'])}")
    m = res["metadata"]
    print(f"  design:     {m['modality']}{' (treated as loss-of-function)' if m['modality'] == 'unknown' else ''}, "
          f"cell line {m['cell_line'] or 'unknown'}{', curated essential panel' if m['curated_ess'] else ''}")
    print(f"  DepMap:     {DEPMAP_POLICY_TEXT[res['depmap_policy']]}")
    if res["anchors"]:
        print(f"  anchors:    {', '.join(res['anchors'])}")
    print(f"  batch size: {res['batch_size']}")
    print(f"  features:   {res['features_csv']}")
    print(f"\nNow run:  python -m waddington_select.suggest --dataset {res['name']}")


if __name__ == "__main__":
    main()
