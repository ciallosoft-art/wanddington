"""
novel_hit_analysis.py — EXPLORATORY recurrent-vs-novel hit decomposition (reviewer v4).

The cross-screen prior is largely a recurrence prior (a plain hit-frequency ranking matches it). The
scientifically sharper question is therefore: can the within-screen components (online retraining + LLM
endorsement) recover hits that never occur in the training screens --- biology recurrence cannot see?

For each target screen we split its hits into
    recurrent  = also a hit in >= 1 other benchmark screen,
    novel      = never a hit in any other benchmark screen,
and report, for three methods, how many of each kind they find in their round-5 selection:
    feature-LOO   : the static LOO-LightGBM ranking (no online, no LLM),
    hit-frequency : rank by count of hits in other screens (pure recurrence),
    Waddington    : the final leakage-free system (from a dumped run with revealed_hits).

This is EXPLORATORY: nothing is tuned on it. Deterministic methods need no seeds; Waddington is read from
a WADDINGTON_DUMP_SELECTIONS run (per-seed recall, then averaged).

    python -m waddington_select.analysis.novel_hit_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import lightgbm as lgb

from ..oracle import BATCH_SIZES, BENCHMARK_DATASETS
from .prior_probes import _load, LGBM, ALLF

REPO = Path(__file__).resolve().parents[2]
WADD_RUN = REPO / "workspace" / "results" / "router" / "clean_headline_w0.2_5seed.json"
OUT = REPO / "workspace" / "results" / "novel_hit_analysis.json"


def _hitsets(df):
    return {d: set(df[(df.dataset == d) & (df.label == 1)].gene) for d in BENCHMARK_DATASETS}


def _split(target, hitset):
    others = [o for o in BENCHMARK_DATASETS if o != target]
    seen = set().union(*[hitset[o] for o in others])
    H = hitset[target]
    return (H & seen), (H - seen)  # recurrent, novel


def _loo_selection(df, target):
    """Top 5*batch genes on the target by an LOO-LightGBM trained on the other screens."""
    sub = df[df["dataset"] == target]
    train = df[df["dataset"] != target]
    pos, neg = (train.label == 1).sum(), (train.label == 0).sum()
    m = lgb.LGBMClassifier(**LGBM, scale_pos_weight=neg / max(pos, 1))
    m.fit(train[ALLF].values, train["label"].values)
    order = np.argsort(-m.predict_proba(sub[ALLF].values)[:, 1])
    genes = sub["gene"].tolist()
    return {genes[i] for i in order[: 5 * BATCH_SIZES[target]]}


def _hitfreq_selection(df, target, hitset):
    sub = df[df["dataset"] == target]
    others = [o for o in BENCHMARK_DATASETS if o != target]
    freq = {g: sum(g in hitset[o] for o in others) for g in sub["gene"]}
    top = sorted(sub["gene"].tolist(), key=lambda g: -freq[g])[: 5 * BATCH_SIZES[target]]
    return set(top)


def _recall(found: set, target_kind: set) -> float:
    return len(found & target_kind) / len(target_kind) if target_kind else float("nan")


def run() -> dict:
    df = _load()
    hitset = _hitsets(df)
    wadd = json.loads(WADD_RUN.read_text()) if WADD_RUN.exists() else None

    per = {}
    for d in BENCHMARK_DATASETS:
        rec, nov = _split(d, hitset)
        loo = _loo_selection(df, d)
        hf = _hitfreq_selection(df, d, hitset)
        row = {
            "n_recurrent": len(rec), "n_novel": len(nov),
            "feature_loo": {"recurrent_recall": _recall(loo, rec), "novel_recall": _recall(loo, nov)},
            "hit_frequency": {"recurrent_recall": _recall(hf, rec), "novel_recall": _recall(hf, nov)},
        }
        if wadd and d in wadd and wadd[d].get("waddington_c"):
            seeds = [set(s.get("revealed_hits", [])) for s in wadd[d]["waddington_c"]]
            if all("revealed_hits" in s for s in wadd[d]["waddington_c"]):
                row["waddington"] = {
                    "recurrent_recall": float(np.mean([_recall(s, rec) for s in seeds])),
                    "novel_recall": float(np.mean([_recall(s, nov) for s in seeds])),
                    # of the extra hits Waddington finds beyond the hit-frequency selection, how many novel?
                    "extra_over_hitfreq_novel_frac": float(np.mean([
                        (len((s - hf) & nov) / len(s - hf)) if (s - hf) else float("nan") for s in seeds])),
                }
        per[d] = row

    def agg(method, kind):
        vals = [per[d][method][kind] for d in BENCHMARK_DATASETS
                if method in per[d] and not np.isnan(per[d][method][kind])]
        return float(np.mean(vals)) if vals else float("nan")

    methods = ["feature_loo", "hit_frequency"] + (["waddington"] if wadd else [])
    res = {
        "per_screen": per,
        "totals": {"recurrent": sum(len(_split(d, hitset)[0]) for d in BENCHMARK_DATASETS),
                   "novel": sum(len(_split(d, hitset)[1]) for d in BENCHMARK_DATASETS)},
        "macro_avg": {m: {"recurrent_recall": agg(m, "recurrent_recall"),
                          "novel_recall": agg(m, "novel_recall")} for m in methods},
        "waddington_run_present": wadd is not None,
    }
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    t = r["totals"]
    print(f"hits: {t['recurrent']} recurrent, {t['novel']} novel "
          f"({100*t['novel']/(t['recurrent']+t['novel']):.0f}% novel)")
    print(f"\n{'method':14s} {'recurrent-recall':>16s} {'novel-recall':>13s}")
    for m, v in r["macro_avg"].items():
        nr = f"{v['novel_recall']:.3f}" if not np.isnan(v["novel_recall"]) else "  -  "
        print(f"{m:14s} {v['recurrent_recall']:16.3f} {nr:>13s}")
    if not r["waddington_run_present"]:
        print("\n[note] Waddington row pending a WADDINGTON_DUMP_SELECTIONS=1 5-seed run (auth required).")
    print(f"saved -> {OUT}")
