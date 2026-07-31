"""
clustered_ci.py — screen-clustered bootstrap CIs and paired win/tie/loss (reviewer: statistics).

The results are nine screens times five seeds. Treating each selected gene (or each screen-seed cell) as
an independent sample understates uncertainty: picks within a screen/round/seed are highly correlated,
and the real unit of replication is the screen. This probe resamples whole screens with replacement (the
cluster), averaging over a screen's five seeds first, to produce:

  1. a 95% bootstrap CI on each method's mean hit@R5;
  2. for the C-arm vs each baseline, the paired per-screen delta with a clustered CI and the
     win / tie / loss count over the nine screens.

Everything is deterministic (fixed seed) and reads the committed per-seed result files; no re-runs.

    python -m waddington_select.analysis.clustered_ci
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEQ = REPO / "workspace" / "results" / "sequential"
OUT = REPO / "workspace" / "results" / "clustered_ci.json"

BENCH = ["IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22",
         "Scharenberg22", "Steinhart", "Replogle_K562_essential", "Replogle_K562_gwps"]

# method label -> (results file, arm key)
SOURCES = {
    "Waddington C":    ("three_arm.json", "waddington_c"),
    "LOO prior":       ("baselines.json", "static_ranker"),
    "Online ML":       ("baselines.json", "online_adaptive"),
    "LLM reasoning":   ("three_arm.json", "llm_reasoning"),
    "Coreset":         ("baselines.json", "coreset"),
    "Random":          ("baselines.json", "random"),
}
N_BOOT = 5000
SEED = 0


def _per_screen_means(fname: str, arm: str) -> dict[str, float]:
    """Mean final-round hit ratio over seeds, per screen."""
    d = json.loads((SEQ / fname).read_text())
    out = {}
    for ds in BENCH:
        runs = d.get(ds, {}).get(arm)
        if runs:
            out[ds] = float(np.mean([r["hit_ratio_per_round"][-1] for r in runs]))
    return out


def _boot_ci(vals: np.ndarray, rng, n=N_BOOT):
    """Screen-clustered bootstrap CI on the mean of per-screen values."""
    idx = rng.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run() -> dict:
    rng = np.random.default_rng(SEED)
    means = {m: _per_screen_means(f, a) for m, (f, a) in SOURCES.items()}

    res: dict = {"methods": {}, "paired_vs_C": {}}
    for m, ps in means.items():
        v = np.array([ps[d] for d in BENCH if d in ps])
        lo, hi = _boot_ci(v, rng)
        res["methods"][m] = {"mean": float(v.mean()), "ci95": [lo, hi], "n_screens": int(len(v))}

    C = means["Waddington C"]
    for m, ps in means.items():
        if m == "Waddington C":
            continue
        shared = [d for d in BENCH if d in C and d in ps]
        delta = np.array([C[d] - ps[d] for d in shared])
        lo, hi = _boot_ci(delta, rng)
        res["paired_vs_C"][m] = {
            "mean_delta": float(delta.mean()), "ci95": [lo, hi],
            "wins": int((delta > 1e-9).sum()), "ties": int((np.abs(delta) <= 1e-9).sum()),
            "losses": int((delta < -1e-9).sum()), "n_screens": len(shared),
        }
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    print("Method              mean hit@R5   95% CI (screen-clustered)")
    for m, v in r["methods"].items():
        print(f"  {m:16s}  {v['mean']:.3f}       [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print("\nC vs baseline (paired over 9 screens):   mean Δ   95% CI            W/T/L")
    for m, v in r["paired_vs_C"].items():
        print(f"  {m:16s}  {v['mean_delta']:+.3f}   [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]   "
              f"{v['wins']}/{v['ties']}/{v['losses']}")
    print(f"saved -> {OUT}")
