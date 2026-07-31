"""
router_analysis.py — leakage-free routing: does the C-arm's edge survive an honest router?

Reads the frozen global-weight runs (workspace/results/router/global_w{0.2,0.3,0.4}.json — each a full
C-arm run at one global fusion weight with the metadata feature policy, no realized hit rate used) and
derives two honest routing variants plus a stricter leave-one-study-out, per the frozen protocol
(waddington_select/router_protocol.py). Every configuration is chosen using ONLY non-target screens and
pre-experiment metadata; the target's hit@R5 is never an input to a routing/weight decision.

Variants:
  global_fixed  — one global w_llm, chosen by argmax mean hit@R5 over the other screens.
  nested_router — rule  w_llm = LOW if n_pool > T else HIGH ; (T,LOW,HIGH) chosen over the other screens,
                  applied to the target via its n_pool only.
  *_study_out   — same, but the target's whole same-study group is removed from the selection set.

For each variant: per-screen and mean hit@R5; the config chosen in each outer fold; and paired
screen-clustered bootstrap deltas (95% CI) + win/tie/loss against (a) the static LOO-LightGBM prior and
(b) the current routed C-arm.

    python -m waddington_select.analysis.router_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..router_protocol import SCREEN_METADATA, BENCH, CANDIDATE_WLLM, study_group

REPO = Path(__file__).resolve().parents[2]
ROUTER = REPO / "workspace" / "results" / "router"
SEQ = REPO / "workspace" / "results" / "sequential"
OUT = ROUTER / "router_analysis.json"
N_BOOT = 5000
SEED = 0


def _mean_last(runs) -> float:
    return float(np.mean([r["hit_ratio_per_round"][-1] for r in runs]))


def _weight_means() -> dict:
    """(screen, w_llm) -> mean hit@R5 over seeds, from the frozen global-weight runs."""
    W = {}
    for w in CANDIDATE_WLLM:
        d = json.loads((ROUTER / f"global_w{w}.json").read_text())
        for s in BENCH:
            W[(s, w)] = _mean_last(d[s]["waddington_c"])
    return W


def _refs() -> tuple[dict, dict]:
    loo = json.loads((SEQ / "baselines.json").read_text())
    cc = json.loads((SEQ / "three_arm.json").read_text())
    LOO = {s: _mean_last(loo[s]["static_ranker"]) for s in BENCH}
    C = {s: _mean_last(cc[s]["waddington_c"]) for s in BENCH}
    return LOO, C


def _train_set(target: str, study_out: bool) -> list[str]:
    if study_out:
        g = study_group(target)
        return [d for d in BENCH if study_group(d) != g]
    return [d for d in BENCH if d != target]


def global_fixed(W, study_out=False) -> tuple[dict, dict]:
    per, choice = {}, {}
    for s in BENCH:
        train = _train_set(s, study_out)
        best_w = max(CANDIDATE_WLLM, key=lambda w: np.mean([W[(d, w)] for d in train]))
        per[s], choice[s] = W[(s, best_w)], {"w_llm": best_w}
    return per, choice


def nested_router(W, study_out=False) -> tuple[dict, dict]:
    ns = sorted({SCREEN_METADATA[d]["n_pool"] for d in BENCH})
    Tcands = [0] + [(ns[i] + ns[i + 1]) / 2 for i in range(len(ns) - 1)] + [ns[-1] + 1]
    per, choice = {}, {}
    for s in BENCH:
        train = _train_set(s, study_out)
        best = None
        for T in Tcands:
            for low in CANDIDATE_WLLM:
                for high in CANDIDATE_WLLM:
                    sc = np.mean([W[(d, low if SCREEN_METADATA[d]["n_pool"] > T else high)] for d in train])
                    if best is None or sc > best[0]:
                        best = (sc, T, low, high)
        _, T, low, high = best
        w_t = low if SCREEN_METADATA[s]["n_pool"] > T else high
        per[s], choice[s] = W[(s, w_t)], {"T": T, "low": low, "high": high, "w_llm": w_t}
    return per, choice


def _paired(varper: dict, ref: dict, rng) -> dict:
    d = np.array([varper[s] - ref[s] for s in BENCH])
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boot = d[idx].mean(axis=1)
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wins": int((d > 1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()),
        "losses": int((d < -1e-9).sum()),
    }


def run() -> dict:
    W = _weight_means()
    LOO, C = _refs()
    rng = np.random.default_rng(SEED)

    variants = {
        "global_fixed": global_fixed(W, study_out=False),
        "nested_router": nested_router(W, study_out=False),
        "global_fixed_study_out": global_fixed(W, study_out=True),
        "nested_router_study_out": nested_router(W, study_out=True),
    }
    res: dict = {"weight_means": {f"{s}|{w}": W[(s, w)] for s in BENCH for w in CANDIDATE_WLLM},
                 "refs": {"LOO": LOO, "C_current": C,
                          "LOO_mean": float(np.mean(list(LOO.values()))),
                          "C_current_mean": float(np.mean(list(C.values())))},
                 "variants": {}}
    for name, (per, choice) in variants.items():
        res["variants"][name] = {
            "per_screen": per,
            "mean": float(np.mean([per[s] for s in BENCH])),
            "config_per_fold": choice,
            "paired_vs_LOO": _paired(per, LOO, np.random.default_rng(SEED)),
            "paired_vs_C": _paired(per, C, np.random.default_rng(SEED + 1)),
        }

    # Strictest test: also remove the same-study sibling from the cross-experiment PRIOR (not just from
    # config selection). Compared like-for-like against a sibling-excluded LOO, so the numerator and the
    # baseline are handicapped the same way. Uses the honest global weight (0.2) with metadata features.
    sop_path = ROUTER / "study_out_prior_w0.2.json"
    pp_path = REPO / "workspace" / "results" / "prior_probes.json"
    if sop_path.exists() and pp_path.exists():
        from ..router_protocol import siblings
        sop = json.loads(sop_path.read_text())
        w02 = json.loads((ROUTER / "global_w0.2.json").read_text())
        c_strict = {s: (_mean_last(sop[s]["waddington_c"]) if siblings(s) else _mean_last(w02[s]["waddington_c"]))
                    for s in BENCH}
        pp = json.loads(pp_path.read_text())["per_screen"]
        loo_sibout = {s: pp[s]["loo_no_sibling"] for s in BENCH}
        res["strict_study_out_prior"] = {
            "per_screen": c_strict,
            "mean": float(np.mean([c_strict[s] for s in BENCH])),
            "loo_sibling_excluded_mean": float(np.mean([loo_sibout[s] for s in BENCH])),
            "paired_vs_sibling_excluded_LOO": _paired(c_strict, loo_sibout, np.random.default_rng(SEED + 2)),
            "paired_vs_standard_LOO": _paired(c_strict, LOO, np.random.default_rng(SEED + 3)),
            "note": "fair leave-one-study-out: both C and LOO deny the same-study sibling.",
        }
    OUT.write_text(json.dumps(res, indent=2))
    return res


def _fmt_choice(name, ch):
    if "T" in ch:
        return f"n>{int(ch['T']) if ch['T']!=0 else 0}? w={ch['low']}:{ch['high']} -> {ch['w_llm']}"
    return f"w={ch['w_llm']}"


if __name__ == "__main__":
    r = run()
    print(f"refs:  LOO mean={r['refs']['LOO_mean']:.3f}   current C mean={r['refs']['C_current_mean']:.3f}\n")
    print("per-screen weight means (hit@R5) at each global w_llm:")
    print(f"  {'screen':24s} " + "  ".join(f"w={w}" for w in CANDIDATE_WLLM))
    for s in BENCH:
        print(f"  {s:24s} " + "  ".join(f"{r['weight_means'][f'{s}|{w}']:.3f}" for w in CANDIDATE_WLLM))
    for name, v in r["variants"].items():
        print(f"\n=== {name}  (mean hit@R5 = {v['mean']:.3f}) ===")
        pl = v["paired_vs_LOO"]; pc = v["paired_vs_C"]
        print(f"  vs LOO:  Δ={pl['mean_delta']:+.3f} CI[{pl['ci95'][0]:+.3f},{pl['ci95'][1]:+.3f}] "
              f"W/T/L={pl['wins']}/{pl['ties']}/{pl['losses']}")
        print(f"  vs C  :  Δ={pc['mean_delta']:+.3f} CI[{pc['ci95'][0]:+.3f},{pc['ci95'][1]:+.3f}] "
              f"W/T/L={pc['wins']}/{pc['ties']}/{pc['losses']}")
        picks = {s: _fmt_choice(name, v["config_per_fold"][s]) for s in BENCH}
        uniq = set(picks.values())
        if len(uniq) == 1:
            print(f"  config (all folds): {next(iter(uniq))}")
        else:
            for s in BENCH:
                print(f"    {s:24s} {picks[s]}")
    print(f"\nsaved -> {OUT}")
