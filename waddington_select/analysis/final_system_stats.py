"""
final_system_stats.py — statistics for the FINAL 5-seed leakage-free Waddington (reviewer round 5).

Three deterministic re-computes (no LLM), so the paper's final-system numbers stop referring to the legacy
0.256 configuration:

  1. matched_loo   — the static LOO prior in the final config's OWN feature space (no anchor + metadata
                     DepMap policy), so the "improves its deployed prior" claim is matched. Compares
                     matched-LOO -> final Waddington with a paired screen-clustered CI.
  2. final_paired  — final Waddington (5 seeds) vs Online ML / LLM branch / Coreset / LOO: paired deltas,
                     95% CI, win/tie/loss (replaces the legacy paired table near tab:main).
  3. novel_hit     — macro averages + paired screen-level CI for Waddington vs feature-LOO novel-recall
                     (reads the per-screen recalls already in novel_hit_analysis.json).

    python -m waddington_select.analysis.final_system_stats
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEQ = REPO / "workspace" / "results" / "sequential"
ROUTER = REPO / "workspace" / "results" / "router"
OUT = REPO / "workspace" / "results" / "final_system_stats.json"

BENCH = ["IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22", "Scharenberg22",
         "Steinhart", "Replogle_K562_essential", "Replogle_K562_gwps"]


def _mean_last(runs):
    return float(np.mean([r["hit_ratio_per_round"][-1] for r in runs]))


def _paired(a: dict, b: dict, seed=0, n_boot=5000):
    """Paired screen-clustered bootstrap of mean(a-b) over the nine screens."""
    d = np.array([a[s] - b[s] for s in BENCH])
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(BENCH), size=(n_boot, len(BENCH)))].mean(1)
    return {"delta": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((d > 1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()),
            "losses": int((d < -1e-9).sum())}


def _matched_loo() -> dict:
    """Static no-anchor-metadata LOO prior: the final config's OnlineAdaptiveArm at round 1 (no reveals)."""
    os.environ["WADDINGTON_FEATURE_POLICY"] = "metadata"
    os.environ["WADDINGTON_DROP_ANCHOR_FEATS"] = "1"
    from ..arms.waddington_c_arm import _get_feature_config
    from ..arms.online_adaptive_arm import OnlineAdaptiveArm
    from ..oracle import BATCH_SIZES, DatasetOracle
    per = {}
    for d in BENCH:
        csv, extra = _get_feature_config(d)
        arm = OnlineAdaptiveArm(d, BATCH_SIZES[d], training_csv=csv, extra_feature_cols=extra)
        orc = DatasetOracle(d)
        scores = arm.all_scores()
        top = sorted(scores, key=lambda g: -scores[g])[: 5 * BATCH_SIZES[d]]
        rev = orc.reveal(top)
        per[d] = (sum(rev.values()) / orc.total_hits) if orc.total_hits else 0.0
    return per


def run() -> dict:
    # All six main-table arms come from the reported 20-seed run (main_table_20seed.json).
    m20 = json.loads((ROUTER / "main_table_20seed.json").read_text())

    FINAL = {s: _mean_last(m20[s]["waddington_c"]) for s in BENCH}
    LOO = {s: _mean_last(m20[s]["static_ranker"]) for s in BENCH}
    ONLINE = {s: _mean_last(m20[s]["online_adaptive"]) for s in BENCH}
    CORESET = {s: _mean_last(m20[s]["coreset"]) for s in BENCH}
    LLM = {s: _mean_last(m20[s]["llm_reasoning"]) for s in BENCH}
    MLOO = _matched_loo()

    mean = lambda x: float(np.mean([x[s] for s in BENCH]))
    res = {
        "final_mean": mean(FINAL),
        "matched_loo": {
            "per_screen": MLOO, "mean": mean(MLOO),
            "final_vs_matched_loo": _paired(FINAL, MLOO, seed=1),
            "note": "matched final-configuration LOO (no-anchor metadata feature policy, static round-1 prior).",
        },
        "final_paired": {
            "vs_LOO_baseline": _paired(FINAL, LOO, seed=2),
            "vs_online_ml": _paired(FINAL, ONLINE, seed=3),
            "vs_llm_branch": _paired(FINAL, LLM, seed=4),
            "vs_coreset": _paired(FINAL, CORESET, seed=5),
            "means": {"final": mean(FINAL), "LOO": mean(LOO), "online": mean(ONLINE),
                      "llm": mean(LLM), "coreset": mean(CORESET)},
        },
    }

    nh = json.loads((REPO / "workspace" / "results" / "novel_hit_analysis.json").read_text())["per_screen"]
    wad_nov = {s: nh[s]["waddington"]["novel_recall"] for s in BENCH}
    loo_nov = {s: nh[s]["feature_loo"]["novel_recall"] for s in BENCH}
    wad_rec = {s: nh[s]["waddington"]["recurrent_recall"] for s in BENCH}
    loo_rec = {s: nh[s]["feature_loo"]["recurrent_recall"] for s in BENCH}
    res["novel_hit"] = {
        "macro": {"waddington_novel": mean(wad_nov), "feature_loo_novel": mean(loo_nov),
                  "waddington_recurrent": mean(wad_rec), "feature_loo_recurrent": mean(loo_rec)},
        "paired_novel_wadd_minus_loo": _paired(wad_nov, loo_nov, seed=6),
        "paired_recurrent_wadd_minus_loo": _paired(wad_rec, loo_rec, seed=7),
        "per_screen": {s: {"wadd_novel": wad_nov[s], "loo_novel": loo_nov[s],
                           "wadd_recurrent": wad_rec[s], "loo_recurrent": loo_rec[s]} for s in BENCH},
    }
    OUT.write_text(json.dumps(res, indent=2))
    return res


def _fmt(p):
    return f"{p['delta']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] {p['wins']}/{p['ties']}/{p['losses']}"


if __name__ == "__main__":
    r = run()
    print(f"final 20-seed Waddington mean = {r['final_mean']:.3f}")
    ml = r["matched_loo"]
    print(f"\n[#7] matched no-anchor-metadata LOO = {ml['mean']:.3f}  (unmatched paper LOO = 0.217)")
    print(f"     final vs matched-LOO: {_fmt(ml['final_vs_matched_loo'])}")
    print(f"\n[#4] final Waddington paired vs baselines:")
    for k, v in r["final_paired"].items():
        if k != "means":
            print(f"     {k:16s} {_fmt(v)}")
    nh = r["novel_hit"]
    print(f"\n[#6] novel-recall macro: Waddington {nh['macro']['waddington_novel']:.3f} vs "
          f"feature-LOO {nh['macro']['feature_loo_novel']:.3f}")
    print(f"     paired (Wadd-LOO) novel: {_fmt(nh['paired_novel_wadd_minus_loo'])}")
    print(f"     paired (Wadd-LOO) recurrent: {_fmt(nh['paired_recurrent_wadd_minus_loo'])}")
    print(f"saved -> {OUT}")
