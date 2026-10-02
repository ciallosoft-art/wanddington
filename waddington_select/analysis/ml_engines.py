"""
ml_engines.py — compare the alternative ML engines against the reported LightGBM engine (step 5 of
docs/ml_engines_plan.md). Reads experiment 15's outputs; trains nothing except the label-shuffle control.

Pre-registered rule (from the plan): the primary metric is the 9-screen mean hit@R5 of the ONLINE ML-only
engine; an engine is "better"/"worse" than LightGBM only if the paired, screen-clustered bootstrap CI of
the difference excludes 0; the (at most two) engines sent into the full GPT C-arm are the best online
engines by that metric among those that output probabilities, LightGBM excluded.

    python -m waddington_select.analysis.ml_engines            # summary + figure
    python -m waddington_select.analysis.ml_engines --control  # also the label-shuffle leakage control
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from ..ml_models import ENGINES, engine_names, is_stochastic
from ..oracle import BATCH_SIZES
from .final_system_stats import BENCH, _paired

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "workspace" / "results" / "ml_engines"
OUT = RES / "summary.json"
FIG = REPO / "workspace" / "results" / "figures" / "09_ml_engines.png"
REF = "ml_engine_lightgbm"
# Engines of the first comparison (steps 2-4 of the plan); everything registered later is phase 2.
PHASE1 = {"logreg", "rf", "mlp", "lambdarank", "rwr", "gnn", "genept_lr", "genept_lgbm", "svm_linear", "svm_rbf"}


def _load() -> dict[str, dict[str, list[dict]]]:
    """arm -> screen -> runs, merged over every experiment-15 output file."""
    arms: dict[str, dict[str, list[dict]]] = {}
    # "<name>.rerun.json" files re-run arms after a fix (docs/ml_engines_plan.md) and load last, so their
    # entries supersede the original file's; the originals stay on disk untouched.
    for p in sorted(RES.glob("*.json"), key=lambda p: (".rerun." in p.name, p.name)):
        if p.name.endswith(".manifest.json") or p.name in {"summary.json", "control.json"}:
            continue
        for ds, by_arm in json.loads(p.read_text()).items():
            if not isinstance(by_arm, dict):
                continue
            for arm, runs in by_arm.items():
                if isinstance(runs, list):
                    arms.setdefault(arm, {})[ds] = runs
    return arms


def _model(arm: str) -> str:
    return arm.split("_", 2)[2]


def _per_screen(runs_by_ds: dict[str, list[dict]]) -> dict[str, float]:
    return {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in runs])) for ds, runs in runs_by_ds.items()}


def _determinism(runs_by_ds: dict[str, list[dict]]) -> bool:
    """Deterministic engines ran two replicates: their selections must be identical, round by round."""
    return all(all(r["selected_genes_per_round"] == runs[0]["selected_genes_per_round"] for r in runs[1:])
               for runs in runs_by_ds.values())


def _seed_spread(runs_by_ds: dict[str, list[dict]]) -> float | None:
    """SD across seeds of the 9-screen mean (seeded engines only)."""
    n = min(len(v) for v in runs_by_ds.values())
    if n < 2:
        return None
    means = [np.mean([runs_by_ds[ds][i]["hit_ratio_per_round"][-1] for ds in BENCH]) for i in range(n)]
    return float(np.std(means, ddof=1))


LABEL_FREE_FEATURES = ["depmap_frac_ess", "depmap_mean_norm", "depmap_min_norm", "hub_score_norm",
                       "ppi_score_sum", "string_degree_norm", "reactome_pathway_count_norm", "pli_score"]


def label_free_baselines() -> dict[str, dict[str, float]]:
    """No training at all: walk down a single gene-level feature from high to low (static, 5 rounds of B).
    Post-hoc reference rows (added after the label-permutation control showed how predictive the raw
    features are), not part of the pre-registered comparison. DepMap columns exist for every screen's
    genes in the v3 table, so these run on all nine screens even where the C-arm's policy drops DepMap."""
    import pandas as pd
    from ..oracle import DatasetOracle
    df = pd.read_csv(REPO / "workspace" / "evaluation" / "lgbm_training_data_v3.csv")
    df["gene"] = df["gene"].str.strip().str.upper()
    out = {}
    for f in LABEL_FREE_FEATURES:
        out[f] = {}
        for ds in BENCH:
            tf = df[df["dataset"] == ds]
            o = DatasetOracle(ds)
            top = tf.sort_values(f, ascending=False, kind="stable")["gene"].tolist()[: 5 * BATCH_SIZES[ds]]
            out[f][ds] = sum(o.reveal(top).values()) / o.total_hits
    return out


def _c_reference() -> dict[str, float]:
    """The reported full system (GPT C-arm, 20 replicates) per screen, for context only."""
    main = json.loads((REPO / "workspace" / "results" / "gpt" / "gpt_5_6_luna_none_20rep"
                       / "main_table_20rep.json").read_text())
    return {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in main[ds]["waddington_c"]])) for ds in BENCH}


def summarize() -> dict:
    arms = _load()
    complete = {a: v for a, v in arms.items() if all(ds in v for ds in BENCH)}
    acq = {a: complete.pop(a) for a in list(complete) if a.startswith("ml_acq_")}
    bnnacq = {a: complete.pop(a) for a in list(complete) if a.startswith("ml_bnnacq_")}
    gdf = {a: complete.pop(a) for a in list(complete) if a.startswith("gdf_")}
    if REF not in complete:
        raise SystemExit(f"{REF} missing or incomplete; run experiments/15_ml_engines.sh tabular first")
    ref = _per_screen(complete[REF])
    rows = {}
    for arm, by_ds in sorted(complete.items()):
        m = _model(arm)
        ps = _per_screen(by_ds)
        rows[arm] = {
            "model": m,
            "online": arm.startswith("ml_engine_"),
            "probabilistic": m == "lightgbm" or ENGINES[m].probabilistic,
            "replicates": min(len(v) for v in by_ds.values()),
            "per_screen": ps,
            "mean": float(np.mean([ps[ds] for ds in BENCH])),
            "vs_lightgbm_online": None if arm == REF else _paired(ps, ref),
            "deterministic_check": None if is_stochastic(m) else _determinism(by_ds),
            "seed_sd_of_mean": _seed_spread(by_ds) if is_stochastic(m) else None,
        }
    for m in {r["model"] for r in rows.values()}:  # what does online retraining add, per engine?
        on, st = rows.get(f"ml_engine_{m}"), rows.get(f"ml_static_{m}")
        if on and st:
            on["online_minus_static"] = _paired(on["per_screen"], st["per_screen"])
    online = [r for a, r in rows.items() if r["online"] and a != REF and r["probabilistic"]]
    ranked = sorted(online, key=lambda r: -r["mean"])
    # The pre-registered stage-B pick is over the phase-1 engines it was made for (GNN, MLP were run in
    # the C-arm); phase-2 engines are ranked separately so adding them never rewrites that record.
    candidates = [r["model"] for r in ranked if r["model"] in PHASE1][:2]
    phase2_top = [r["model"] for r in ranked if r["model"] not in PHASE1][:2]
    acq_rows = {}
    if "ml_acq_greedy" in acq:  # step 6: rules compared on the same LightGBM ensemble
        greedy = _per_screen(acq["ml_acq_greedy"])
        for arm, by_ds in sorted(acq.items()):
            ps = _per_screen(by_ds)
            acq_rows[arm] = {"rule": arm[len("ml_acq_"):], "per_screen": ps,
                             "mean": float(np.mean([ps[ds] for ds in BENCH])),
                             "replicates": min(len(v) for v in by_ds.values()),
                             "seed_sd_of_mean": _seed_spread(by_ds),
                             "vs_ensemble_greedy": None if arm == "ml_acq_greedy" else _paired(ps, greedy),
                             "vs_single_lightgbm": _paired(ps, ref)}
    bnn_rows = {}
    if bnnacq and "ml_engine_bnn" in complete:  # phase 2 item 3b: rules on the same Bayesian MLP
        bnn_greedy = _per_screen(complete["ml_engine_bnn"])
        for arm, by_ds in sorted(bnnacq.items()):
            ps = _per_screen(by_ds)
            bnn_rows[arm] = {"rule": arm[len("ml_bnnacq_"):], "per_screen": ps,
                             "mean": float(np.mean([ps[ds] for ds in BENCH])),
                             "replicates": min(len(v) for v in by_ds.values()),
                             "seed_sd_of_mean": _seed_spread(by_ds),
                             "vs_bnn_greedy": _paired(ps, bnn_greedy), "vs_lightgbm_online": _paired(ps, ref)}
    c_ref = _c_reference()
    free = {}
    for f, ps in label_free_baselines().items():
        free[f] = {"per_screen": ps, "mean": float(np.mean([ps[ds] for ds in BENCH])),
                   "ml_engine_minus_this": _paired(ref, ps), "c_minus_this": _paired(c_ref, ps)}
    res = {"rows": rows, "reference": REF, "stage_b_candidates": candidates,
           "phase2_top_probabilistic": phase2_top,
           "missing_engines": sorted(set(engine_names()) - {r["model"] for r in rows.values()}),
           "c_20rep_reference": {"per_screen": c_ref, "mean": float(np.mean(list(c_ref.values()))),
                                 "c_minus_ml_engine": _paired(c_ref, ref)},
           "label_free_post_hoc": free,
           "acquisition": acq_rows,
           "bnn_acquisition": bnn_rows,
           "genedisco_faithful": _genedisco_faithful(gdf, ref) if gdf else {}}
    OUT.write_text(json.dumps(res, indent=2))
    return res


PORTED = REPO / "workspace" / "results" / "sequential" / "genedisco_ported.json"


def _genedisco_faithful(gdf: dict, ref: dict) -> dict:
    """Phase 2 item 3a: GeneDisco with its own models, target, loop and (optionally) features, against
    random, the paper's port (its marginsample rule on our LightGBM, 0.169) and our ML engine."""
    from ..arms.genedisco_faithful_arm import parse_name
    ported = json.loads(PORTED.read_text())
    port = {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in ported[ds]["genedisco_marginsample"]]))
            for ds in BENCH}
    rand = _per_screen(gdf["gdf_random"]) if "gdf_random" in gdf else None
    rows = {}
    for arm, by_ds in sorted(gdf.items()):
        feats, model, rule = parse_name(arm)
        ps = _per_screen(by_ds)
        rows[arm] = {"features": feats, "model": model, "rule": rule, "per_screen": ps,
                     "mean": float(np.mean([ps[ds] for ds in BENCH])),
                     "replicates": min(len(v) for v in by_ds.values()), "seed_sd_of_mean": _seed_spread(by_ds),
                     "vs_random": None if rand is None or arm == "gdf_random" else _paired(ps, rand),
                     "vs_ported_marginsample": _paired(ps, port), "vs_lightgbm_online": _paired(ps, ref)}
    return {"rows": rows, "ported_marginsample": {"per_screen": port, "mean": float(np.mean(list(port.values())))}}


def control(models: list[str] | None = None, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Label-permutation controls on each engine's STATIC prior (top-5B recall vs the random 5B/N).

    global  — permute the labels across ALL training rows. No label information survives, so the
              recall must fall to the random expectation; staying high would mean target labels reach
              the model through the features or the pipeline (the leakage null).
    within  — permute labels inside each training screen only. Each screen keeps its hit RATE, and the
              screens differ in composition (K562-Essential is 623 essential genes at a 10% hit rate,
              the genome-wide screens are 4-5%), so a model can still learn "essential, central genes
              are hits more often" from between-screen differences. Not a leak — it uses other screens'
              labels only — but it measures how much of a prior is screen composition rather than
              gene-level pattern. (A first version of this control used only this mode and was misread
              as a leak; see docs/ml_engines_plan.md, step 5.)
    """
    from ..arms.ml_engine_arm import MLEngineArm
    from ..oracle import DatasetOracle
    models = models or engine_names()
    out = {}
    for m in models:
        out[m] = {}
        for mode in ("global", "within"):
            rec = {ds: [] for ds in BENCH}
            for seed in seeds:
                for ds in BENCH:
                    bs = BATCH_SIZES[ds]
                    arm = MLEngineArm(ds, bs, model=m, online=False, seed=seed)
                    rng = np.random.default_rng(seed)
                    lab = arm._loo_train["label"].to_numpy().copy()
                    if mode == "global":
                        lab = rng.permutation(lab)
                    else:
                        dsets = arm._loo_train["dataset"].to_numpy()
                        for d in np.unique(dsets):
                            idx = np.flatnonzero(dsets == d)
                            lab[idx] = rng.permutation(lab[idx])
                    arm._loo_train["label"] = lab
                    arm._rebuild_ranking()
                    oracle = DatasetOracle(ds)
                    found = sum(oracle.reveal(arm._ranking[: 5 * bs]).values())
                    rec[ds].append(found / oracle.total_hits)
            per = {ds: float(np.mean(v)) for ds, v in rec.items()}
            out[m][mode] = {"per_screen": per, "mean_recall": float(np.mean(list(per.values()))),
                            "seeds": list(seeds)}
        out[m]["random"] = float(np.mean([5 * BATCH_SIZES[ds] / DatasetOracle(ds).n_genes for ds in BENCH]))
        print(f"  {m:12s} static recall with labels permuted globally {out[m]['global']['mean_recall']:.3f} | "
              f"within screens {out[m]['within']['mean_recall']:.3f} | random {out[m]['random']:.3f}", flush=True)
    path = RES / "control.json"  # merge, so a later run for new engines keeps the earlier ones
    merged = json.loads(path.read_text()) if path.exists() else {}
    merged.update(out)
    path.write_text(json.dumps(merged, indent=2))
    return out


CARM_DIR = RES / "carm"
CARM_REF = REPO / "workspace" / "results" / "gpt" / "replogle_transcriptomics_v2_20rep" / "main_table_20rep.json"


def carm_summary() -> dict:
    """Step 7: the full GPT C-arm with an alternative engine vs the reported C-arm of the corrected-prompt
    20-replicate run (same code, memory and protocol), and vs the same engine without the LLM."""
    ref_runs = json.loads(CARM_REF.read_text())
    ref = {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in ref_runs[ds]["waddington_c"]])) for ds in BENCH}
    ml_rows = json.loads(OUT.read_text())["rows"] if OUT.exists() else {}
    out = {"reference": {"source": str(CARM_REF.relative_to(REPO)), "per_screen": ref,
                         "mean": float(np.mean(list(ref.values())))}, "arms": {}}
    arms: dict[str, dict[str, list[dict]]] = {}
    for p in sorted(CARM_DIR.glob("*.json")):
        if p.name.endswith(".manifest.json"):
            continue
        for ds, by_arm in json.loads(p.read_text()).items():
            if isinstance(by_arm, dict):
                for arm, runs in by_arm.items():
                    if isinstance(runs, list) and arm.startswith("waddington_c__"):
                        arms.setdefault(arm, {})[ds] = arms.get(arm, {}).get(ds, []) + runs
    for arm, by_ds in sorted(arms.items()):
        if not all(ds in by_ds for ds in BENCH):
            continue
        model = arm[len("waddington_c__"):]
        ps = _per_screen(by_ds)
        row = {"per_screen": ps, "mean": float(np.mean([ps[ds] for ds in BENCH])),
               "replicates": min(len(v) for v in by_ds.values()),
               "vs_reported_c": _paired(ps, ref)}
        ml = ml_rows.get(f"ml_engine_{model}")
        if ml:  # what the LLM fusion adds on top of this engine
            row["llm_contribution"] = _paired(ps, ml["per_screen"])
        out["arms"][arm] = row
    (RES / "carm_summary.json").write_text(json.dumps(out, indent=2))
    return out


DISPLAY = {"lightgbm": "LightGBM (reported)", "gnn": "GraphSAGE on STRING", "mlp": "MLP",
           "genept_lgbm": "LightGBM + GenePT", "logreg": "Logistic regression", "rf": "Random forest",
           "rwr": "RWR on STRING", "lambdarank": "LambdaMART", "genept_lr": "GenePT + logistic",
           "svm_linear": "Linear SVM", "svm_rbf": "RBF SVM (Nystroem)"}


def figure(res: dict) -> Path:
    import matplotlib.pyplot as plt
    from . import style as S
    rows = res["rows"]
    models = sorted({r["model"] for r in rows.values() if r["model"] in PHASE1 | {"lightgbm"}},  # the paper figure
                    key=lambda m: -rows.get(f"ml_engine_{m}", {"mean": 0})["mean"])
    S.apply()
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(models) + 1.8))
    y = np.arange(len(models))[::-1]
    means = [r["mean"] for r in rows.values()]
    span = max(means) - min(means)
    label_x = max(means) + 0.04 * span  # one column of labels right of every point, never on top of one
    ax.set_xlim(min(means) - 0.05 * span, max(means) + 0.62 * span)
    for yi, m in zip(y, models):
        st, on = rows.get(f"ml_static_{m}"), rows.get(f"ml_engine_{m}")
        if st and on:
            ax.plot([st["mean"], on["mean"]], [yi, yi], color=S.BASELINE, linewidth=1.2, zorder=2)
        if st:
            ax.scatter(st["mean"], yi, s=36, facecolor=S.SURFACE, edgecolor=S.SLOTS[0], zorder=3,
                       label="static prior" if yi == y[0] else None)
        if on:
            ax.scatter(on["mean"], yi, s=42, color=S.SLOTS[0], zorder=4,
                       label="online (retrained each round)" if yi == y[0] else None)
            d = on.get("vs_lightgbm_online")
            tag = "reference" if d is None else f"{d['delta']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
            ax.annotate(tag, (label_x, yi), va="center", fontsize=8, color=S.INK_2)
    ax.set_yticks(y, [DISPLAY.get(m, m) for m in models])
    ax.set_xlabel("hit@R5, mean over the nine screens (ML only, no LLM)")
    ax.set_title("Swapping the ML engine: online vs LightGBM, paired Δ [screen-clustered 95% CI]",
                 color=S.INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(fontsize=8, loc="upper center", ncols=2, bbox_to_anchor=(0.5, -0.16 - 0.5 / len(models)))
    ax.grid(axis="y", visible=False)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    return FIG


if __name__ == "__main__":
    r = summarize()
    for arm, row in sorted(r["rows"].items(), key=lambda kv: -kv[1]["mean"]):
        d = row["vs_lightgbm_online"]
        chk = row["deterministic_check"]
        sd = row["seed_sd_of_mean"]
        parts = [f"  {arm:24s} {row['mean']:.4f}  reps={row['replicates']}"]
        if chk is not None:
            parts.append("deterministic" if chk else "NOT DETERMINISTIC")
        if sd is not None:
            parts.append(f"seed sd {sd:.4f}")
        if d is not None:
            parts.append(f"vs LightGBM {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
                         f"{d['wins']}/{d['ties']}/{d['losses']}")
        print("  ".join(parts))
    print("stage-B candidates:", r["stage_b_candidates"], "| not yet run:", r["missing_engines"])
    c = r["c_20rep_reference"]
    print(f"full C (20-rep) {c['mean']:.4f}; C - ML engine {c['c_minus_ml_engine']['delta']:+.4f} "
          f"[{c['c_minus_ml_engine']['ci95'][0]:+.4f}, {c['c_minus_ml_engine']['ci95'][1]:+.4f}]")
    if r["acquisition"]:
        print("acquisition rules (LightGBM 5-member bootstrap ensemble, 5 seeds):")
        for arm, v in sorted(r["acquisition"].items(), key=lambda kv: -kv[1]["mean"]):
            g, sgl = v["vs_ensemble_greedy"], v["vs_single_lightgbm"]
            gtxt = "reference" if g is None else f"vs greedy {g['delta']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}] {g['wins']}/{g['ties']}/{g['losses']}"
            print(f"  {v['rule']:9s} {v['mean']:.4f}  seed sd {v['seed_sd_of_mean']:.4f}  {gtxt}  | vs single LightGBM {sgl['delta']:+.4f}")
    if r["bnn_acquisition"]:
        print("acquisition rules on the Bayesian MLP (5 seeds; reference = ml_engine_bnn, greedy):")
        for arm, v in sorted(r["bnn_acquisition"].items(), key=lambda kv: -kv[1]["mean"]):
            g, l = v["vs_bnn_greedy"], v["vs_lightgbm_online"]
            print(f"  {v['rule']:17s} {v['mean']:.4f}  vs bnn greedy {g['delta']:+.4f} [{g['ci95'][0]:+.4f}, "
                  f"{g['ci95'][1]:+.4f}] {g['wins']}/{g['ties']}/{g['losses']}  | vs LightGBM {l['delta']:+.4f}")
    if r["genedisco_faithful"]:
        gf = r["genedisco_faithful"]
        print(f"faithful GeneDisco (5 seeds; paper's port, marginsample = {gf['ported_marginsample']['mean']:.4f}):")
        for arm, v in sorted(gf["rows"].items(), key=lambda kv: -kv[1]["mean"]):
            p, l = v["vs_ported_marginsample"], v["vs_lightgbm_online"]
            print(f"  {arm:34s} {v['mean']:.4f}  vs port {p['delta']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
                  f"  | vs LightGBM {l['delta']:+.4f}")
    print("label-free single-feature rankings (post hoc, no training):")
    for f, v in sorted(r["label_free_post_hoc"].items(), key=lambda kv: -kv[1]["mean"]):
        d = v["c_minus_this"]
        print(f"  {f:28s} {v['mean']:.4f}   C - this {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
              f"{d['wins']}/{d['ties']}/{d['losses']}")
    print("figure ->", figure(r))
    if "--control" in sys.argv:
        control()
    if CARM_DIR.exists():
        cs = carm_summary()
        print(f"full GPT C-arm with a swapped engine (reference {cs['reference']['mean']:.4f}, "
              f"{cs['reference']['source']}):")
        for arm, v in cs["arms"].items():
            d, l = v["vs_reported_c"], v.get("llm_contribution")
            print(f"  {arm:20s} {v['mean']:.4f} (x{v['replicates']})  vs reported C {d['delta']:+.4f} "
                  f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] {d['wins']}/{d['ties']}/{d['losses']}"
                  + ("" if l is None else f"  | LLM adds {l['delta']:+.4f} [{l['ci95'][0]:+.4f}, {l['ci95'][1]:+.4f}]"))
