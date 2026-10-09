"""
pair_tier.py — the combinatorial tier's numbers, recomputed from the result files (paper: "All three of
the benchmark's difficulty tiers", Table pairs-bda, "Diagnosing and repairing the combinatorial failure").

  prior_alone    hits in the top 160 / 640 of the repaired arm's label-free interaction prior ranked on its
                 own (no LLM, no learning), with the seven original pair columns and with the eighth
                 (complex co-membership); also the offline complex-first ordering of that prior
  table_version  which feature table each repaired-arm run used. Runs with a round-1 cold start rank round
                 1 by the prior alone, so their round-1 batch identifies the table; the other runs are
                 placed by their creation time relative to those
  runs           every configuration at both budgets: replicates, mean +- s.d.
  steps          the changes the paper attributes, as Welch tests between consecutive configurations
  over_prior     each run that uses the prior, against that prior ranked on its own
  vs_bda         the 160-pick rows against BioDiscoveryAgent's agent at its own budget
  llm_supply     LLM-named valid pairs per round, distinct over five rounds, the share of the batch they
                 fill, and the hit rate of LLM-named and other picks
  complex        the complex co-membership pairs: count and hit rate
  llm_contribution  the final repair three ways: the prior alone, without its LLM (prior + online ML),
                 with the LLM's names first, and with the LLM as the reported 0.2 verifier
  coverage       genes touched by the selected pairs, and the exploration slope
  learning_curve a model trained on N randomly revealed pairs, scored on the rest (AUC, hits in the top
                 640), on the seven- and the eight-column table

    python -m waddington_select.analysis.pair_tier
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from ..arms.waddington_c_pair_arm import interaction_prior
from ..phenotype import ANCHOR_COLS
from ..tables import ABSDIFF_SUFFIX
from .sources import GPT, REPO

PAIR_TABLE_V3 = REPO / "workspace" / "evaluation" / "lgbm_training_data_horlbeck_pairs_v3.csv"
OUT = REPO / "workspace" / "results" / "pair_tier_summary.json"
COMPLEX_COL = "pair_complex_shared"
NEGATIVE = "NEGATIVE"  # the library's negative control, paired with each of the 448 genes
CURVE_SIZES = [32, 64, 128, 320, 640, 1280, 2560, 5000, 10000, 20000]
CURVE_DRAWS = 5
CURVE_LGBM = dict(objective="binary", learning_rate=0.05, n_estimators=300, num_leaves=31, verbose=-1,
                  n_jobs=4, random_state=42)
BUDGETS = {128: 640, 32: 160}

B128 = GPT / "tier_b_pairs_20rep"                      # pair-mean table (no interaction features)
I128 = GPT / "tier_b_pairs_interaction_20rep"          # interaction features, 128 pairs per round
B32 = GPT / "tier_b_pairs_b32_20rep"                   # pair-mean table, BDA's 32 pairs per round
I32 = GPT / "tier_b_pairs_interaction_b32_20rep"       # interaction features, 32 pairs per round

# label -> (file, arm). Pair-arm switches: experiments/13_tier_b_fusion_ablation.sh.
RUNS = {
    128: {
        "random": (B128 / "calibration_20rep.json", "random"),
        "coreset": (B128 / "calibration_20rep.json", "coreset"),
        "static_ranker": (B128 / "calibration_20rep.json", "static_ranker"),
        "online_ml": (B128 / "calibration_20rep.json", "online_adaptive"),
        "padded_llm": (B128 / "calibration_20rep.json", "llm_reasoning"),
        "c_arm": (B128 / "waddington_c_20rep.json", "waddington_c"),
        "c_arm_interaction": (I128 / "waddington_c_20rep.json", "waddington_c"),
        "online_ml_interaction_7col": (I128 / "deterministic_1rep.json", "online_adaptive"),
        "online_ml_interaction_8col": (I128 / "deterministic_8feat_1rep.json", "online_adaptive"),
        "repair_no_llm": (I128 / "pair_no_llm_1rep.json", "waddington_c_pair_no_llm"),
        "verifier_prior": (I128 / "pair_verifier_prior.json", "waddington_c_pair"),
        "namedfirst_only": (I128 / "pair_namedfirst_only.json", "waddington_c_pair"),
        "namedfirst_prior": (I128 / "pair_namedfirst_prior.json", "waddington_c_pair"),
        "coldstart_prior": (I128 / "pair_coldstart_prior.json", "waddington_c_pair"),
        "complex_first": (I128 / "pair_complex_first.json", "waddington_c_pair"),
        "coldstart_prior_v8": (I128 / "pair_coldstart_prior_v8.json", "waddington_c_pair"),
        "explore_12pct": (I128 / "pair_explore_12pct.json", "waddington_c_pair"),
        "explore_25pct": (I128 / "pair_explore_25pct.json", "waddington_c_pair"),
    },
    32: {
        "random": (B32 / "calibration_20rep.json", "random"),
        "coreset": (B32 / "calibration_20rep.json", "coreset"),
        "static_ranker": (B32 / "calibration_20rep.json", "static_ranker"),
        "online_ml": (B32 / "calibration_20rep.json", "online_adaptive"),
        "padded_llm": (B32 / "calibration_20rep.json", "llm_reasoning"),
        "c_arm": (B32 / "waddington_c_20rep.json", "waddington_c"),
        "bda": (B32 / "bda_controlled_real.json", "bda"),
        "c_arm_interaction": (I32 / "waddington_c_7col_20rep.json", "waddington_c"),
        "online_ml_interaction_7col": (I32 / "deterministic_7col_1rep.json", "online_adaptive"),
        "repair_no_llm": (I32 / "pair_no_llm_1rep.json", "waddington_c_pair_no_llm"),
        "verifier_prior": (I32 / "pair_verifier_prior.json", "waddington_c_pair"),
        "namedfirst_only": (I32 / "pair_namedfirst_only.json", "waddington_c_pair"),
        "namedfirst_prior": (I32 / "pair_namedfirst_prior.json", "waddington_c_pair"),
        "coldstart_prior": (I32 / "pair_coldstart_prior.json", "waddington_c_pair"),
        "complex_first": (I32 / "pair_complex_first.json", "waddington_c_pair"),
        "coldstart_prior_v8": (I32 / "pair_coldstart_prior_v8.json", "waddington_c_pair"),
        "explore_12pct": (I32 / "pair_explore_12pct.json", "waddington_c_pair"),
        "explore_25pct": (I32 / "pair_explore_25pct.json", "waddington_c_pair"),
    },
}
PRIOR_FILL = {"namedfirst_prior", "coldstart_prior", "complex_first", "coldstart_prior_v8",
              "explore_12pct", "explore_25pct", "verifier_prior"}
COLD_START_PLAIN = {"coldstart_prior", "coldstart_prior_v8", "repair_no_llm", "verifier_prior"}  # round 1 = the prior's top B
TABLE_LABELS = ("namedfirst", "coldstart", "complex", "explore", "repair_no_llm", "c_arm_interaction",
                "online_ml_interaction", "verifier")

# (budget, from, to): the steps the paper attributes
STEPS = [
    (128, "c_arm", "c_arm_interaction"),
    (128, "c_arm_interaction", "namedfirst_only"),
    (128, "namedfirst_only", "namedfirst_prior"),
    (128, "namedfirst_prior", "coldstart_prior"),
    (128, "coldstart_prior", "coldstart_prior_v8"),
    (128, "coldstart_prior_v8", "complex_first"),
    (128, "coldstart_prior_v8", "explore_12pct"),
    (128, "coldstart_prior_v8", "explore_25pct"),
    (128, "coldstart_prior_v8", "verifier_prior"),
    (32, "c_arm", "c_arm_interaction"),
    (32, "c_arm_interaction", "namedfirst_only"),
    (32, "c_arm", "namedfirst_only"),
    (32, "namedfirst_only", "namedfirst_prior"),
    (32, "namedfirst_prior", "coldstart_prior_v8"),
    (32, "coldstart_prior_v8", "complex_first"),
    (32, "coldstart_prior", "coldstart_prior_v8"),
    (32, "coldstart_prior_v8", "verifier_prior"),
]


def _horlbeck() -> pd.DataFrame:
    df = pd.read_csv(PAIR_TABLE_V3)
    df["gene"] = df["gene"].str.strip().str.upper()
    return df[df["dataset"] == "Horlbeck"].reset_index(drop=True)


def _priors(sub: pd.DataFrame) -> dict[int, dict[str, float]]:
    """The arm's prior on the eight-column table and on the seven-column table it replaced."""
    return {8: interaction_prior(sub), 7: interaction_prior(sub.drop(columns=[COMPLEX_COL]))}


def _order(score: dict[str, float], pool: list[str]) -> list[str]:
    return sorted(pool, key=lambda g: score.get(g, 0.0), reverse=True)  # stable, as in the arm


def _runs(budget: int) -> dict[str, list[dict]]:
    out = {}
    for label, (f, arm) in RUNS[budget].items():
        if f.exists():
            out[label] = json.loads(f.read_text())["Horlbeck"][arm]
    return out


def _hits(rs: list[dict]) -> np.ndarray:
    return np.array([r["cumulative_hits"][-1] for r in rs], dtype=float)


def _manifest(budget: int, label: str) -> dict:
    m = RUNS[budget][label][0].with_suffix(".manifest.json")
    return json.loads(m.read_text()) if m.exists() else {}


def _created(budget: int, label: str) -> datetime | None:
    c = _manifest(budget, label).get("created_at")
    return datetime.fromisoformat(c) if c else None


def _fingerprint(budget: int, label: str) -> str | None:
    return _manifest(budget, label).get("protocol", {}).get("runtime_fingerprint", {}).get("sha256")


def _welch(a: np.ndarray, b: np.ndarray) -> dict:
    t = stats.ttest_ind(b, a, equal_var=False)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dfree = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    h90 = stats.t.ppf(0.95, dfree) * se
    d = float(b.mean() - a.mean())
    return {"delta": d, "p": float(t.pvalue), "cohens_d": d / sd if sd > 0 else None, "ci90": [d - h90, d + h90]}


def _ci95(x: np.ndarray) -> list[float]:
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def learning_curve(sub: pd.DataFrame, drop: set[str]) -> dict:
    """The within-screen learning curve: train on N randomly revealed Horlbeck pairs, rank the rest, and
    count the hits among the top 640 (and the AUC). Five draws per N. The draws are those of the
    2026-09-12 diagnosis (seed 0), so `hits_top640_row_order` reproduces its numbers; that run broke score
    ties by row order, while `hits_top640` breaks them at random. The two differ only where the model has
    barely learned anything: at 32 pairs it cannot split at all and scores every pair the same."""
    import lightgbm as lgb
    anchors = set(ANCHOR_COLS) | {c + ABSDIFF_SUFFIX for c in ANCHOR_COLS}
    feats = [c for c in sub.columns
             if c not in ("gene", "label", "dataset") and c not in anchors and c not in drop]
    y_all = sub["label"].to_numpy()
    X_all = sub[feats].to_numpy(dtype=float)
    draw_rng, tie_rng = np.random.default_rng(0), np.random.default_rng(1)
    out = {"features": len(feats), "draws": CURVE_DRAWS, "points": {}}
    for n in CURVE_SIZES:
        aucs, hits, hits_row = [], [], []
        for _ in range(CURVE_DRAWS):
            idx = draw_rng.choice(len(sub), n, replace=False)
            mask = np.ones(len(sub), bool)
            mask[idx] = False
            ytr, yte = y_all[idx], y_all[mask]
            if ytr.sum() < 2:
                continue
            m = lgb.LGBMClassifier(**CURVE_LGBM, scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1))
            m.fit(X_all[idx], ytr)
            p = m.predict_proba(X_all[mask])[:, 1]
            aucs.append(roc_auc_score(yte, p))
            hits_row.append(int(yte[np.argsort(-p)[:640]].sum()))
            hits.append(int(yte[np.lexsort((tie_rng.random(len(p)), -p))[:640]].sum()))
        out["points"][str(n)] = {"auc": float(np.mean(aucs)), "hits_top640": float(np.mean(hits)),
                                 "hits_top640_row_order": float(np.mean(hits_row)), "draws_used": len(aucs)}
    return out


def run() -> dict:
    sub = _horlbeck()
    pool = sub["gene"].tolist()
    hits = set(sub.loc[sub["label"] == 1, "gene"])
    base_rate = len(hits) / len(pool)
    priors = _priors(sub)
    orders = {k: _order(p, pool) for k, p in priors.items()}
    cx = set(sub.loc[sub[COMPLEX_COL] > 0, "gene"])

    def top_hits(order: list[str], n: int) -> int:
        return sum(g in hits for g in order[:n])

    prior_alone = {f"{k}_columns": {str(n): top_hits(o, n) for n in BUDGETS.values()} for k, o in orders.items()}
    for k, o in orders.items():
        cxo = [g for g in o if g in cx] + [g for g in o if g not in cx]
        prior_alone[f"{k}_columns_complex_first"] = {str(n): top_hits(cxo, n) for n in BUDGETS.values()}
    for k, o in orders.items():  # the cold start's first batch
        prior_alone[f"{k}_columns"]["top128_hit_rate"] = top_hits(o, 128) / 128

    res: dict = {"n_pairs": len(pool), "n_hits": len(hits), "base_rate": base_rate,
                 "prior_alone": prior_alone, "complex": {"n_pairs": len(cx),
                                                         "hit_rate": sum(g in hits for g in cx) / len(cx)},
                 "table_version": {}, "runs": {}, "steps": [], "over_prior": {}, "vs_bda": {}, "llm_supply": {}}

    # Which table each pair-arm run used. Round 1 of a plain cold-start run is the prior's own top B, so
    # its round-1 batch identifies the table. Every other run is placed by a verified run with the same
    # runtime fingerprint, else by creation time: the column was added once.
    all_runs = {B: _runs(B) for B in BUDGETS}
    verified: dict[tuple[int, str], dict] = {}
    for B, runs in all_runs.items():
        for label in COLD_START_PLAIN & set(runs):
            r1 = [set(r["selected_genes_per_round"][0]) for r in runs[label]]
            overlap = {k: float(np.mean([len(s & set(o[:B])) / B for s in r1])) for k, o in orders.items()}
            verified[(B, label)] = {"columns": max(overlap, key=overlap.get), "how": "round-1 batch",
                                    "overlap": overlap, "created": _created(B, label),
                                    "fingerprint": _fingerprint(B, label)}
    by_fingerprint = {v["fingerprint"]: v["columns"] for v in verified.values() if v["fingerprint"]}
    last_seven = max((v["created"] for v in verified.values() if v["columns"] == 7), default=None)
    first_eight = min((v["created"] for v in verified.values() if v["columns"] == 8), default=None)
    for B, runs in all_runs.items():
        tv = {}
        for label in runs:
            if (B, label) in verified:
                tv[label] = {k: v for k, v in verified[(B, label)].items() if k not in ("created", "fingerprint")}
                continue
            if not label.startswith(TABLE_LABELS):
                continue
            c, fp = _created(B, label), _fingerprint(B, label)
            table = str(_manifest(B, label).get("protocol", {}).get("training_csv_v3_override") or "")
            if table.endswith("_7col.csv"):
                tv[label] = {"columns": 7, "how": "seven-column table copy named in the manifest"}
            elif fp in by_fingerprint:
                tv[label] = {"columns": by_fingerprint[fp], "how": "same runtime fingerprint as a verified run"}
            elif last_seven and c and c <= last_seven:
                tv[label] = {"columns": 7, "how": "created before a run verified on seven columns"}
            elif first_eight and c and c >= first_eight:
                tv[label] = {"columns": 8, "how": "created after a run verified on eight columns"}
        res["table_version"][str(BUDGETS[B])] = tv

    for B, budget in BUDGETS.items():
        runs = all_runs[B]
        tv = res["table_version"][str(budget)]

        res["runs"][str(budget)] = {
            label: {"n": len(rs), "mean": float(_hits(rs).mean()),
                    "sd": float(_hits(rs).std(ddof=1)) if len(rs) > 1 else 0.0,
                    "ci95": _ci95(_hits(rs)) if len(rs) > 1 else None}
            for label, rs in runs.items()}

        for b, a_label, b_label in STEPS:
            if b == B and a_label in runs and b_label in runs:
                res["steps"].append({"budget": budget, "from": a_label, "to": b_label,
                                     **_welch(_hits(runs[a_label]), _hits(runs[b_label]))})

        for label in PRIOR_FILL & set(runs):
            cols = tv.get(label, {}).get("columns")
            if cols is None:
                continue
            # complex_first is compared with the same ordering offline: complex pairs first, then the prior
            key = f"{cols}_columns" + ("_complex_first" if label == "complex_first" else "")
            ref = prior_alone[key][str(budget)]
            x = _hits(runs[label])
            if len(x) < 2:
                continue
            res["over_prior"][f"{budget}:{label}"] = {
                "columns": cols, "prior_ordering": key, "prior_alone": ref, "delta": float(x.mean() - ref),
                "p": float(stats.ttest_1samp(x, ref).pvalue), "replicates_above": int((x > ref).sum()),
                "n": len(x)}

        if "bda" in runs:
            bda = _hits(runs["bda"])
            for label in runs:
                if label != "bda" and len(runs[label]) > 1:
                    res["vs_bda"][label] = _welch(bda, _hits(runs[label]))
            for k in (7, 8):
                ref = prior_alone[f"{k}_columns"][str(budget)]
                res["vs_bda"][f"prior_alone_{k}_columns"] = {
                    "value": ref, "delta": float(ref - bda.mean()),
                    "p": float(stats.ttest_1samp(bda, ref).pvalue),
                    "bda_replicates_reaching": int((bda >= ref).sum())}

        for label, rs in runs.items():
            if "llm_named_proposals_per_round" not in rs[0]:
                continue
            named = np.array([[len(x) for x in r["llm_named_proposals_per_round"]] for r in rs], dtype=float)
            if not named.any():  # arms without an LLM record empty lists
                continue
            share = np.array([[len(x) / B for x in r["selected_llm_named_genes_per_round"]] for r in rs])
            sel_named, sel_other = [], []
            for r in rs:
                for i, batch in enumerate(r["selected_genes_per_round"]):
                    nm = set(r["selected_llm_named_genes_per_round"][i])
                    sel_named += [g in hits for g in batch if g in nm]
                    sel_other += [g in hits for g in batch if g not in nm]
            res["llm_supply"][f"{budget}:{label}"] = {
                "named_per_round": named.mean(0).round(1).tolist(),
                "distinct_named_over_five_rounds": float(np.mean(
                    [len(set().union(*map(set, r["llm_named_proposals_per_round"]))) for r in rs])),
                "llm_share_of_batch": float(share.mean()), "llm_share_per_round": share.mean(0).round(3).tolist(),
                "hit_rate_llm_named": float(np.mean(sel_named)) if sel_named else None,
                "hit_rate_other": float(np.mean(sel_other)) if sel_other else None}

    # What the LLM adds to the repair: prior alone -> + online ML (no LLM, deterministic) -> + LLM (full arm)
    res["llm_contribution"] = {}
    for B, budget in BUDGETS.items():
        runs = all_runs[B]
        if "repair_no_llm" not in runs or "coldstart_prior_v8" not in runs:
            continue
        cols = res["table_version"][str(budget)].get("repair_no_llm", {}).get("columns", 8)
        prior = prior_alone[f"{cols}_columns"][str(budget)]
        no_llm = float(_hits(runs["repair_no_llm"]).mean())
        full = _hits(runs["coldstart_prior_v8"])
        res["llm_contribution"][str(budget)] = {
            "columns": cols, "prior_alone": prior, "no_llm": no_llm, "full": float(full.mean()),
            "online_ml_over_prior": no_llm - prior, "llm_over_no_llm": float(full.mean() - no_llm),
            "p_llm": float(stats.ttest_1samp(full, no_llm).pvalue),
            "replicates_above_no_llm": int((full > no_llm).sum()), "n": len(full)}
        if "verifier_prior" in runs:  # the LLM as a 0.2 verifier on the same fill score
            ver = _hits(runs["verifier_prior"])
            res["llm_contribution"][str(budget)]["verifier"] = {
                "mean": float(ver.mean()), "sd": float(ver.std(ddof=1)), "n": len(ver),
                "over_no_llm": float(ver.mean() - no_llm), "p_vs_no_llm": float(stats.ttest_1samp(ver, no_llm).pvalue),
                "replicates_above_no_llm": int((ver > no_llm).sum()),
                "vs_named_first": _welch(full, ver)}

    # Coverage (the exploration table): real genes touched by the selected pairs, per replicate. The
    # library is 448 genes plus a negative control; pairing a gene with the control is a candidate too.
    n_genes = len({g for t in pool for g in t.split("|")} - {NEGATIVE})
    res["coverage"] = {"n_library_genes": n_genes, "n_negative_control_pairs": sum(NEGATIVE in t.split("|") for t in pool)}
    for B, budget in BUDGETS.items():
        for label in ("coldstart_prior_v8", "explore_12pct", "explore_25pct"):
            rs = all_runs[B].get(label)
            if not rs:
                continue
            from collections import Counter
            touched, round1, top_count, control_picks = [], [], [], []
            for r in rs:
                c = Counter(g for batch in r["selected_genes_per_round"] for t in batch for g in t.split("|")
                            if g != NEGATIVE)
                touched.append(len(c))
                top_count.append(max(c.values()))
                round1.append(len({g for t in r["selected_genes_per_round"][0] for g in t.split("|")} - {NEGATIVE}))
                control_picks.append(sum(NEGATIVE in t.split("|") for b in r["selected_genes_per_round"] for t in b))
            res["coverage"][f"{budget}:{label}"] = {
                "genes_touched": float(np.mean(touched)), "round1_genes": float(np.mean(round1)),
                "most_sampled_gene_count": float(np.mean(top_count)),
                "negative_control_picks": float(np.mean(control_picks)),
                "uniform_expectation_per_gene": 2 * budget / n_genes}
        xs, ys = [], []
        for label, share in (("coldstart_prior_v8", 0.0), ("explore_12pct", 0.125), ("explore_25pct", 0.25)):
            for h in _hits(all_runs[B].get(label, [])):
                xs.append(share)
                ys.append(h)
        if len(set(xs)) == 3:
            fit = stats.linregress(xs, ys)
            res["coverage"][f"{budget}:explore_slope"] = {"hits_per_unit_share": float(fit.slope),
                                                           "p": float(fit.pvalue)}

    # The within-screen learning curve, on the seven-column table of the original diagnosis and on today's
    res["learning_curve"] = {"7_columns": learning_curve(sub, {COMPLEX_COL}), "8_columns": learning_curve(sub, set())}

    OUT.write_text(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    r = run()
    print(f"Horlbeck {r['n_pairs']} pairs, {r['n_hits']} hits; complex pairs {r['complex']['n_pairs']} "
          f"at {r['complex']['hit_rate']:.3f}")
    print("prior alone:", json.dumps(r["prior_alone"]))
    for budget, tv in r["table_version"].items():
        print(f"[table version, {budget} picks]", {k: (v["columns"], v["how"]) for k, v in tv.items()})
    for budget, rows in r["runs"].items():
        print(f"[runs, {budget} picks]", ", ".join(f"{k} {v['mean']:.1f}±{v['sd']:.1f}" for k, v in rows.items()))
    for s in r["steps"]:
        print(f"  step {s['budget']}: {s['from']} -> {s['to']}: {s['delta']:+.1f} p={s['p']:.2g}")
    for k, v in r["over_prior"].items():
        print(f"  over prior {k}: {v['delta']:+.1f} vs {v['prior_alone']} ({v['columns']} cols) "
              f"p={v['p']:.2g} above {v['replicates_above']}/{v['n']}")
    for k, v in r["vs_bda"].items():
        print(f"  vs BDA {k}: {v['delta']:+.1f} p={v['p']:.2g}" +
              (f" d={v['cohens_d']:.2f} ci90=[{v['ci90'][0]:.1f},{v['ci90'][1]:.1f}]" if "cohens_d" in v else ""))
    for k, v in r.get("llm_contribution", {}).items():
        print(f"  LLM contribution, {k} picks ({v['columns']} cols): prior {v['prior_alone']} -> no LLM "
              f"{v['no_llm']:.0f} -> full {v['full']:.1f}; LLM {v['llm_over_no_llm']:+.1f} p={v['p_llm']:.2g} "
              f"above {v['replicates_above_no_llm']}/{v['n']}")
    for k, curve in r["learning_curve"].items():
        print(f"  learning curve, {k} ({curve['features']} features): " + ", ".join(
            f"{n}: AUC {v['auc']:.3f} top640 {v['hits_top640']:.0f} (row-order {v['hits_top640_row_order']:.0f})"
            for n, v in curve["points"].items()))
    for k, v in r["llm_supply"].items():
        print(f"  supply {k}: named/round {v['named_per_round']} distinct {v['distinct_named_over_five_rounds']:.0f} "
              f"share {v['llm_share_of_batch']:.2f} hit named {v['hit_rate_llm_named']} other {v['hit_rate_other']}")
