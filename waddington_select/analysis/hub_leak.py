"""
hub_leak.py — are hub_score_norm and ppi_score_sum gene-intrinsic? (2026-10-22; local, no LLM calls)

The paper lists both among the gene-intrinsic features ("STRING hubness, total PPI weight"). They are not
genome-wide network properties. Both are computed from a cache of STRING interaction_partners queries
(workspace/evaluation/_ppi_cache, top 200 partners per query gene; gene_ranker._compute_hub_scores):

  hub_score_norm(g) = number of query genes whose top-200 partner list contains g, / its maximum
  ppi_score_sum(g)  = sum of the combined scores of those listings, / its maximum

The frozen query set (build_tier_a_unfiltered.frozen_ppi_cache_files, 315 genes) is the nine screens' 64
anchor genes plus 251 genes written on 2026-06-16, before the reveal cache for confirmed hits was split off
(gene_ranker, 2026-06-20); half of those 251 are benchmark hits against a 21% base rate. Through that query
set, each screen's own anchors (51% own-screen hits) and hits shape its genes' values of both features.

Variants of the two columns, applied to every row of both headline tables (training rows and target rows):

  reported   as stored
  dropped    both set to 0 (constant, so no tree can split on them)
  own_out    rebuilt from the same query cache with each row's OWN screen's anchors and hits removed from the
             query set: no screen's features see its own labels. Other screens' anchors and hits stay, as
             cross-experiment information of the kind the LOO prior is allowed to use.
  genome     what the paper's wording describes: hub dropped (genome-wide STRING degree is already a feature)
             and ppi replaced by genome-wide total PPI weight, the sum of a gene's STRING v12 combined scores
             over all its partners (workspace/data/_universal_cache), normalised by the maximum

Each variant is evaluated, in a subprocess pointed at its tables (WADDINGTON_TRAINING_CSV{,_V3}), with
  matched prior  the final configuration's static LOO prior (final_system_stats._matched_loo; 0.231)
  online ML      the C-arm's ML branch alone over five rounds with retraining (no LLM)
and compared to `reported`, paired and screen-clustered. Also recorded: the single-feature rankings by
each of the two columns (as ml_engines.label_free_baselines: top 5B, ties in label-free order seed 0)
under `reported` and `own_out`, and which query files the repository tracks (only the 64 anchors).

Run:  python -m waddington_select.analysis.hub_leak          -> workspace/results/hub_leak.json
      python -m waddington_select.analysis.hub_leak --single-only   (adds the single-feature rankings to it)
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "workspace" / "evaluation"
V1 = EVAL / "lgbm_training_data.csv"
V3 = EVAL / "lgbm_training_data_v3.csv"
FULL = EVAL / "lgbm_training_data_tier_a.csv"
STRING_LINKS = REPO / "workspace" / "data" / "_universal_cache" / "string_links.txt.gz"
STRING_INFO = REPO / "workspace" / "data" / "_universal_cache" / "string_info.txt.gz"
OUT = REPO / "workspace" / "results" / "hub_leak.json"
VARIANTS = ("reported", "dropped", "own_out", "genome")
COLS = ("hub_score_norm", "ppi_score_sum")


def _evaluation_modules():
    sys.path.insert(0, str(EVAL))
    import build_tier_a_unfiltered as bt  # noqa: E402  (workspace/evaluation, not the package)
    import gene_ranker as gr  # noqa: E402
    return bt, gr


def _from_queries(files: dict[str, dict[str, float]], queries) -> tuple[dict, dict]:
    count: dict[str, int] = {}
    total: dict[str, float] = {}
    for q in queries:
        for g, s in files[q].items():
            count[g] = count.get(g, 0) + 1
            total[g] = total.get(g, 0.0) + s
    mc, mt = max(count.values()), max(total.values())
    return ({g: round(c / mc, 4) for g, c in count.items()}, {g: round(t / mt, 4) for g, t in total.items()})


def _genome_ppi_weight() -> dict[str, float]:
    """Sum of STRING v12 combined scores over all partners (the links file lists each edge both ways)."""
    sym = {}
    with gzip.open(STRING_INFO, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        pc = header.index("protein_external_id") if "protein_external_id" in header else 0
        sc = header.index("preferred_name") if "preferred_name" in header else 1
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            sym[parts[pc].strip()] = parts[sc].strip().upper()
    total: dict[str, float] = {}
    with gzip.open(STRING_LINKS, "rt") as fh:
        fh.readline()
        for line in fh:
            a, _, score = line.split()[:3]
            g = sym.get(a)
            if g:
                total[g] = total.get(g, 0.0) + int(score) / 1000.0
    mx = max(total.values())
    return {g: round(t / mx, 4) for g, t in total.items()}


def build_tables(out_dir: Path) -> dict:
    """Write each variant's two tables under out_dir/<variant>/; return provenance counts."""
    bt, gr = _evaluation_modules()
    files = {f.stem.upper(): json.loads(f.read_text()) for f in bt.frozen_ppi_cache_files()}
    anchors = {k: {a.upper() for a in v} for k, v in gr.DATASET_ANCHORS.items()}
    labels = pd.concat([pd.read_csv(p, usecols=["gene", "label", "dataset"]) for p in (V3, FULL)])
    labels["gene"] = labels["gene"].str.strip().str.upper()
    hits = labels[labels["label"] == 1].groupby("dataset")["gene"].apply(set).to_dict()

    hub_all, ppi_all = _from_queries(files, files)
    own = {}
    prov = {"n_query_genes": len(files), "n_anchor_queries": len(set().union(*anchors.values()) & set(files)),
            "own_queries_removed": {}}
    for ds in anchors:
        drop = (anchors[ds] | hits.get(ds, set())) & set(files)
        prov["own_queries_removed"][ds] = len(drop)
        own[ds] = _from_queries(files, [q for q in files if q not in drop])
    genome = _genome_ppi_weight()

    for src in (V1, V3):
        df = pd.read_csv(src)
        g = df["gene"].astype(str).str.strip().str.upper()
        # the stored values are exactly the full-query-set rebuild (checked, not assumed)
        assert (df["hub_score_norm"] - g.map(hub_all).fillna(0.0)).abs().max() < 1e-9, src
        assert (df["ppi_score_sum"] - g.map(ppi_all).fillna(0.0)).abs().max() < 1e-9, src
        for variant in VARIANTS:
            v = df.copy()
            if variant == "dropped":
                v[list(COLS)] = 0.0
            elif variant == "own_out":
                for ds, (hub, ppi) in own.items():
                    rows = v["dataset"] == ds
                    v.loc[rows, "hub_score_norm"] = g[rows].map(hub).fillna(0.0)
                    v.loc[rows, "ppi_score_sum"] = g[rows].map(ppi).fillna(0.0)
            elif variant == "genome":
                v["hub_score_norm"] = 0.0
                v["ppi_score_sum"] = g.map(genome).fillna(0.0)
            (out_dir / variant).mkdir(parents=True, exist_ok=True)
            v.to_csv(out_dir / variant / src.name, index=False)
    return prov


def single_feature(table_dir: Path) -> dict:
    """hit@R5 of ranking each screen's genes by one column, per variant (headline tier, v3 table)."""
    from ..arms.base import label_free_order
    from ..oracle import BATCH_SIZES
    from ..router_protocol import BENCH
    out = {}
    for variant in ("reported", "own_out"):
        df = pd.read_csv(table_dir / variant / V3.name)
        df["gene"] = df["gene"].astype(str).str.strip().str.upper()
        out[variant] = {}
        for col in (*COLS, "string_degree_norm"):
            per = {}
            for ds in BENCH:
                tf = df[df["dataset"] == ds]
                tf = tf.iloc[label_free_order(tf["gene"].tolist(), 0)]
                top = tf.sort_values(col, ascending=False, kind="stable").head(5 * BATCH_SIZES[ds])
                per[ds] = float(top["label"].sum() / tf["label"].sum())
            out[variant][col] = {"mean": sum(per.values()) / len(per), "per_screen": per}
    return out


def _tracked_queries() -> dict:
    _, gr = _evaluation_modules()
    ls = subprocess.run(["git", "ls-files", "workspace/evaluation/_ppi_cache"], cwd=REPO, capture_output=True,
                        text=True, check=True).stdout.split()
    tracked = {Path(f).stem.upper() for f in ls}
    anchors = {a.upper() for v in gr.DATASET_ANCHORS.values() for a in v}
    return {"n_tracked": len(tracked), "tracked_are_exactly_the_anchors": tracked == anchors}


def evaluate() -> dict:
    """Matched prior and online ML on the tables in $WADDINGTON_TRAINING_CSV{,_V3} (run in a subprocess)."""
    os.environ["WADDINGTON_FEATURE_POLICY"] = "metadata"
    os.environ["WADDINGTON_DROP_ANCHOR_FEATS"] = "1"
    from ..arms.online_adaptive_arm import OnlineAdaptiveArm
    from ..arms.waddington_c_arm import _get_feature_config
    from ..oracle import BATCH_SIZES
    from ..router_protocol import BENCH
    from ..sequential_runner import run_arm_on_dataset
    from .final_system_stats import _matched_loo

    online = {}
    for d in BENCH:
        csv, extra = _get_feature_config(d)
        arm = OnlineAdaptiveArm(d, BATCH_SIZES[d], training_csv=csv, extra_feature_cols=extra, drop_anchor_feats=True)
        online[d] = run_arm_on_dataset(arm, d, n_rounds=5).hit_ratio_per_round[-1]
    return {"matched_prior": _matched_loo(), "online_ml": online}


def main() -> None:
    if "--evaluate" in sys.argv:
        print("RESULT " + json.dumps(evaluate()))
        return
    if "--single-only" in sys.argv:
        out = json.loads(OUT.read_text())
        with tempfile.TemporaryDirectory(prefix="hub_leak_") as tmp:
            out["provenance"] = {**build_tables(Path(tmp)), **_tracked_queries()}
            out["single_feature"] = single_feature(Path(tmp))
        OUT.write_text(json.dumps(out, indent=1))
        for v, cols in out["single_feature"].items():
            print(v, {c: round(r["mean"], 4) for c, r in cols.items()})
        print(out["provenance"]["n_tracked"], out["provenance"]["tracked_are_exactly_the_anchors"])
        return
    from .final_system_stats import _paired
    import numpy as np
    out = {"variants": {}}
    with tempfile.TemporaryDirectory(prefix="hub_leak_") as tmp:
        out["provenance"] = {**build_tables(Path(tmp)), **_tracked_queries()}
        out["single_feature"] = single_feature(Path(tmp))
        for variant in VARIANTS:
            env = dict(os.environ, WADDINGTON_TRAINING_CSV=str(Path(tmp) / variant / V1.name),
                       WADDINGTON_TRAINING_CSV_V3=str(Path(tmp) / variant / V3.name))
            if variant == "reported":   # the committed tables themselves
                env.update(WADDINGTON_TRAINING_CSV=str(V1), WADDINGTON_TRAINING_CSV_V3=str(V3))
            res = subprocess.run([sys.executable, "-m", "waddington_select.analysis.hub_leak", "--evaluate"],
                                 cwd=REPO, env=env, capture_output=True, text=True, check=True).stdout
            out["variants"][variant] = json.loads(res[res.rindex("RESULT ") + 7:])
            r = out["variants"][variant]
            print(f"{variant:9s} matched prior {np.mean(list(r['matched_prior'].values())):.4f}  "
                  f"online ML {np.mean(list(r['online_ml'].values())):.4f}", flush=True)
    ref = out["variants"]["reported"]
    out["paired_vs_reported"] = {
        v: {k: {"mean": float(np.mean(list(out["variants"][v][k].values()))), **_paired(out["variants"][v][k], ref[k])}
            for k in ("matched_prior", "online_ml")}
        for v in VARIANTS if v != "reported"}
    OUT.write_text(json.dumps(out, indent=1))
    for v, ks in out["paired_vs_reported"].items():
        for k, p in ks.items():
            print(f"{v:9s} {k:13s} {p['mean']:.4f}  vs reported {p['delta']:+.4f} "
                  f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  {p['wins']}/{p['ties']}/{p['losses']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
