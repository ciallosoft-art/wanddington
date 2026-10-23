"""
final_system_stats.py — statistics for the FINAL 20-seed leakage-free Waddington (reviewer round 5).

Three deterministic re-computes (no LLM). The headline and paired comparisons use the reported 20-seed
run, rather than the legacy 5-seed supporting configuration (0.256). The matched 20-seed legacy-router
control (0.259) is analyzed separately in the paper:

  1. matched_loo   — the static LOO prior in the final config's OWN feature space (no anchor + metadata
                     DepMap policy), so the "improves its deployed prior" claim is matched. Compares
                     matched-LOO -> final Waddington with a paired screen-clustered CI.
  2. final_paired  — final Waddington (20 seeds) vs Online ML / LLM branch / Coreset / LOO: paired deltas,
                     95% CI, win/tie/loss (replaces the legacy paired table near tab:main).
  3. novel_hit     — macro averages + paired screen-level CI for Waddington vs feature-LOO novel-recall
                     (exploratory: reads per-screen recalls from a selection-dump 5-seed run already in
                     novel_hit_analysis.json).

    python -m waddington_select.analysis.final_system_stats            # the reported GPT run
    python -m waddington_select.analysis.final_system_stats --claude   # the Claude predecessor (frozen)

The default reads the reported GPT run (analysis/sources.py) and adds what the paper's robustness and
cross-model paragraphs cite: screen-clustered marginal CIs, the five-replicate run, the legacy-consistent
routed GPT C-arm and the Claude predecessor, each as a paired comparison.
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


def tier_stats(run_file: Path) -> dict:
    """The headline comparisons on another tier's twenty-replicate run (its tables must be in the
    environment): the full system against the matched static LOO prior, the routed LOO baseline, online
    ML, the padded LLM baseline and coreset, paired over the nine screens; screen-clustered marginal CIs."""
    run = json.loads(Path(run_file).read_text())
    per = lambda arm: {d: float(np.mean([r["hit_ratio_per_round"][-1] for r in run[d][arm]])) for d in BENCH}
    P = {"final": per("waddington_c"), "LOO": per("static_ranker"), "online": per("online_adaptive"),
         "llm": per("llm_reasoning"), "coreset": per("coreset"), "matched_LOO": _matched_loo()}
    return {"source": str(run_file), "means": {k: float(np.mean(list(v.values()))) for k, v in P.items()},
            "per_screen": P,
            "paired": {f"vs_{k}": _paired(P["final"], P[k]) for k in ("matched_LOO", "LOO", "online", "llm", "coreset")},
            "marginal_ci": {k: _marginal(P[k]) for k in ("final", "matched_LOO", "online")}}


def _marginal(per: dict, seed=0, n_boot=5000) -> dict:
    """Screen-clustered bootstrap CI of a method's nine-screen mean (unpaired; deliberately wide)."""
    v = np.array([per[s] for s in BENCH])
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(BENCH), size=(n_boot, len(BENCH)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def run_gpt() -> dict:
    """The reported GPT run (analysis/sources.py) — the numbers the paper reports."""
    from . import novel_hit_analysis as N
    from .sources import (CLAUDE_FINAL_20SEED, LEGACY_GPT_ABLATIONS, REPORTED_RUN, REPORTED_RUN_5REP,
                          REPORTED_RUN_DIR)
    m20 = json.loads(REPORTED_RUN.read_text())
    arms = {"final": "waddington_c", "LOO": "static_ranker", "online": "online_adaptive",
            "llm": "llm_reasoning", "coreset": "coreset", "random": "random"}
    P = {k: {s: _mean_last(m20[s][a]) for s in BENCH} for k, a in arms.items()}
    MLOO = _matched_loo()
    mean = lambda x: float(np.mean([x[s] for s in BENCH]))
    res = {"source": str(REPORTED_RUN.relative_to(REPO)), "final_mean": mean(P["final"]),
           "per_screen": P,
           "matched_loo": {"per_screen": MLOO, "mean": mean(MLOO),
                           "final_vs_matched_loo": _paired(P["final"], MLOO, seed=1)},
           "final_paired": {"vs_LOO_baseline": _paired(P["final"], P["LOO"], seed=2),
                            "vs_online_ml": _paired(P["final"], P["online"], seed=3),
                            "vs_llm_branch": _paired(P["final"], P["llm"], seed=4),
                            "vs_coreset": _paired(P["final"], P["coreset"], seed=5),
                            "means": {k: mean(v) for k, v in P.items()}},
           "marginal_ci": {k: _marginal(P[k], seed=10) for k in ("final", "LOO", "online")}}
    leg = json.loads(LEGACY_GPT_ABLATIONS.read_text())
    LEG = {s: _mean_last(leg[s]["waddington_c"]) for s in BENCH}
    res["legacy_routed"] = {"mean": mean(LEG), "per_screen": LEG, "final_minus_legacy": _paired(P["final"], LEG, seed=0)}
    cl = json.loads(CLAUDE_FINAL_20SEED.read_text())
    CL = {s: _mean_last(cl[s]["waddington_c"]) for s in BENCH}
    res["claude_predecessor"] = {"mean": mean(CL), "gpt_minus_claude": _paired(P["final"], CL, seed=0),
                                 "claude_vs_matched_loo": _paired(CL, MLOO, seed=1)}
    m5 = json.loads(REPORTED_RUN_5REP.read_text())
    P5 = {k: {s: _mean_last(m5[s][a]) for s in BENCH} for k, a in arms.items()}
    res["five_replicates"] = {"per_screen": P5, "means": {k: mean(v) for k, v in P5.items()},
                              "final_vs_matched_loo": _paired(P5["final"], MLOO, seed=1),
                              "secondary": {k: _paired(P5["final"], P5[k], seed=i)
                                            for i, k in enumerate(("LOO", "online", "llm", "coreset"), 2)}}
    N.WADD_RUN, N.OUT = REPORTED_RUN, REPORTED_RUN_DIR / "novel_hit_analysis.json"
    nh = N.run()["per_screen"]
    res["novel_hit"] = _novel(nh, mean)
    (REPORTED_RUN_DIR / "final_system_stats.json").write_text(json.dumps(res, indent=2))
    return res


def _novel(nh: dict, mean) -> dict:
    wad_nov = {s: nh[s]["waddington"]["novel_recall"] for s in BENCH}
    loo_nov = {s: nh[s]["feature_loo"]["novel_recall"] for s in BENCH}
    wad_rec = {s: nh[s]["waddington"]["recurrent_recall"] for s in BENCH}
    loo_rec = {s: nh[s]["feature_loo"]["recurrent_recall"] for s in BENCH}
    hf_nov = {s: nh[s]["hit_frequency"]["novel_recall"] for s in BENCH}
    hf_rec = {s: nh[s]["hit_frequency"]["recurrent_recall"] for s in BENCH}
    return {"macro": {"waddington_novel": mean(wad_nov), "feature_loo_novel": mean(loo_nov),
                      "waddington_recurrent": mean(wad_rec), "feature_loo_recurrent": mean(loo_rec),
                      "hit_frequency_novel": mean(hf_nov), "hit_frequency_recurrent": mean(hf_rec),
                      "novel_share": sum(nh[s]["n_novel"] for s in BENCH)
                      / sum(nh[s]["n_novel"] + nh[s]["n_recurrent"] for s in BENCH)},
            "paired_novel_wadd_minus_loo": _paired(wad_nov, loo_nov, seed=6),
            "paired_recurrent_wadd_minus_loo": _paired(wad_rec, loo_rec, seed=7)}


def run() -> dict:
    """The Claude predecessor (frozen): its 20-seed run and its 5-seed novel-hit file."""
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
    import sys
    if "--tier-run" in sys.argv:  # another tier: python -m ... --tier-run RUN.json OUT.json (tables in the env)
        i = sys.argv.index("--tier-run")
        res = tier_stats(Path(sys.argv[i + 1]))
        Path(sys.argv[i + 2]).write_text(json.dumps(res, indent=2))
        print({k: round(v, 4) for k, v in res["means"].items()})
        for k, v in res["paired"].items():
            print(f"  {k:16s} {_fmt(v)}")
        sys.exit(0)
    if "--claude" not in sys.argv:
        r = run_gpt()
        f = _fmt
        print(f"reported GPT run ({r['source']}): C = {r['final_mean']:.4f}")
        ml = r["matched_loo"]
        print(f"  vs matched LOO ({ml['mean']:.4f}): {f(ml['final_vs_matched_loo'])}")
        for k, v in r["final_paired"].items():
            if k != "means":
                print(f"  {k:16s} {f(v)}")
        print("  means:", {k: round(v, 4) for k, v in r["final_paired"]["means"].items()})
        print("  marginal CIs:", {k: (round(v["mean"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in r["marginal_ci"].items()})
        lg = r["legacy_routed"]
        print(f"  legacy routed GPT C {lg['mean']:.4f}; final - legacy {f(lg['final_minus_legacy'])}")
        cp = r["claude_predecessor"]
        print(f"  Claude predecessor {cp['mean']:.4f}; GPT - Claude {f(cp['gpt_minus_claude'])}; Claude vs matched LOO {f(cp['claude_vs_matched_loo'])}")
        fv = r["five_replicates"]
        print("  five replicates:", {k: round(v, 4) for k, v in fv["means"].items()},
              "vs matched LOO", f(fv["final_vs_matched_loo"]))
        nh = r["novel_hit"]
        print("  novel:", {k: round(v, 3) for k, v in nh["macro"].items()})
        print(f"  paired novel {f(nh['paired_novel_wadd_minus_loo'])}; recurrent {f(nh['paired_recurrent_wadd_minus_loo'])}")
        raise SystemExit(0)
    r = run()
    print(f"final 20-seed Waddington mean = {r['final_mean']:.3f}")
    ml = r["matched_loo"]
    print(f"\n[#7] matched no-anchor-metadata LOO = {ml['mean']:.3f}  (unmatched paper LOO = 0.217)")
    print(f"     final vs matched-LOO: {_fmt(ml['final_vs_matched_loo'])}")
    print("\n[#4] final Waddington paired vs baselines:")
    for k, v in r["final_paired"].items():
        if k != "means":
            print(f"     {k:16s} {_fmt(v)}")
    nh = r["novel_hit"]
    print(f"\n[#6] novel-recall macro: Waddington {nh['macro']['waddington_novel']:.3f} vs "
          f"feature-LOO {nh['macro']['feature_loo_novel']:.3f}")
    print(f"     paired (Wadd-LOO) novel: {_fmt(nh['paired_novel_wadd_minus_loo'])}")
    print(f"     paired (Wadd-LOO) recurrent: {_fmt(nh['paired_recurrent_wadd_minus_loo'])}")
    print(f"saved -> {OUT}")
