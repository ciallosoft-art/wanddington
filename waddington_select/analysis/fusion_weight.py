"""
fusion_weight.py — how sensitive is the full system to its ML-LLM fusion weight, and what would a weight
chosen without the target screen pick? (docs/fusion_weight_plan.md)

Grid w_llm in GRID, all on the reported GPT system:
  0.0  the C-arm's own online ML engine (identical selections at w_llm = 0; tests/test_ml_models.py)
  0.2  the reported 20-replicate run (and, as a sensitivity check, its first five replicates)
  rest experiment 17, five replicates each, same settings as the 0.2 run except WADDINGTON_FORCE_WLLM

Reports the sensitivity curve (paired delta vs 0.2), a nested leave-one-screen-out selection (choose w
on the other eight screens, apply it to the ninth; ties go to the smaller w) and the per-screen oracle
upper bound.

    python -m waddington_select.analysis.fusion_weight
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .final_system_stats import BENCH, _paired

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "workspace" / "results" / "fusion_weight"
REF = REPO / "workspace" / "results" / "gpt" / "replogle_transcriptomics_v2_20rep" / "main_table_20rep.json"
ML_SUMMARY = REPO / "workspace" / "results" / "ml_engines" / "summary.json"
FIG = REPO / "workspace" / "results" / "figures" / "10_fusion_weight.png"
GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
REFERENCE_W = 0.2


def _per_screen(runs_by_ds: dict, arm: str = "waddington_c", first: int | None = None) -> dict[str, float]:
    return {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in runs_by_ds[ds][arm][:first]]))
            for ds in BENCH}


def load(first_reference_reps: int | None = None) -> tuple[dict[float, dict[str, float]], dict[float, int]]:
    """w -> per-screen hit@R5, and w -> replicates, for every grid point whose results exist."""
    curves, reps = {}, {}
    ml = json.loads(ML_SUMMARY.read_text())["rows"]["ml_engine_lightgbm"]
    curves[0.0], reps[0.0] = ml["per_screen"], ml["replicates"]
    ref = json.loads(REF.read_text())
    curves[REFERENCE_W] = _per_screen(ref, first=first_reference_reps)
    reps[REFERENCE_W] = min(len(ref[ds]["waddington_c"][:first_reference_reps]) for ds in BENCH)
    for w in GRID:
        p = RES / f"carm_w{w:.1f}_5rep.json"
        if w in curves or not p.exists():
            continue
        runs = json.loads(p.read_text())
        if all(ds in runs and len(runs[ds].get("waddington_c", [])) for ds in BENCH):
            curves[w] = _per_screen(runs)
            reps[w] = min(len(runs[ds]["waddington_c"]) for ds in BENCH)
    return dict(sorted(curves.items())), reps


def analyse(curves: dict[float, dict[str, float]]) -> dict:
    grid = list(curves)
    ref = curves[REFERENCE_W]
    curve = {f"{w:.1f}": {"mean": float(np.mean(list(c.values()))), "per_screen": c,
                          "vs_reference": None if w == REFERENCE_W else _paired(c, ref)}
             for w, c in curves.items()}
    # nested leave-one-screen-out: pick w on the other eight screens, apply to the held-out one
    chosen, selected = {}, {}
    for ds in BENCH:
        others = [d for d in BENCH if d != ds]
        best_w, best = None, -np.inf
        for w in grid:  # ascending, strict '>' -> ties go to the smaller (more ML-weighted) w
            m = float(np.mean([curves[w][d] for d in others]))
            if m > best + 1e-12:
                best_w, best = w, m
        chosen[ds], selected[ds] = best_w, curves[best_w][ds]
    oracle = {ds: max(curves[w][ds] for w in grid) for ds in BENCH}
    oracle_w = {ds: max(grid, key=lambda w: (curves[w][ds], -w)) for ds in BENCH}
    return {"grid": grid, "curve": curve,
            "loo_selection": {"chosen_w": chosen, "per_screen": selected,
                              "mean": float(np.mean(list(selected.values()))),
                              "vs_reference": _paired(selected, ref)},
            "per_screen_oracle": {"best_w": oracle_w, "per_screen": oracle,
                                  "mean": float(np.mean(list(oracle.values()))),
                                  "vs_reference": _paired(oracle, ref)}}


def figure(res: dict) -> Path:
    import matplotlib.pyplot as plt
    from . import style as S
    S.apply()
    ws = [float(w) for w in res["curve"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for d in BENCH:  # each screen relative to its own reference value, thin and quiet
        ref = res["curve"][f"{REFERENCE_W:.1f}"]["per_screen"][d]
        ax.plot(ws, [res["curve"][f"{w:.1f}"]["per_screen"][d] - ref for w in ws], color=S.BASELINE,
                linewidth=0.8, zorder=1)
    deltas, lo, hi = [], [], []
    for w in ws:
        v = res["curve"][f"{w:.1f}"]["vs_reference"]
        deltas.append(0.0 if v is None else v["delta"])
        lo.append(0.0 if v is None else v["delta"] - v["ci95"][0])
        hi.append(0.0 if v is None else v["ci95"][1] - v["delta"])
    ax.errorbar(ws, deltas, yerr=[lo, hi], color=S.SLOTS[0], marker="o", linewidth=1.8, capsize=3, zorder=3,
                label="mean over nine screens, 95% CI (screen-clustered)")
    ax.axhline(0, color=S.INK_2, linewidth=0.8, zorder=2)
    ax.set_xticks(ws, [f"{w:.1f}" + ("\n(reported)" if w == REFERENCE_W else "\n(ML only)" if w == 0 else "")
                       for w in ws])
    ax.set_xlabel("LLM fusion weight $w_{LLM}$  ($w_{ML} = 1 - w_{LLM}$)")
    ax.set_ylabel("hit@R5 minus the reported $w_{LLM}=0.2$")
    ax.set_title("Fusion-weight sensitivity of the full system (grey: individual screens)",
                 color=S.INK, fontsize=10.5, loc="left", pad=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="x", visible=False)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    return FIG


def main() -> dict:
    curves, reps = load()
    out = {"replicates": {f"{w:.1f}": n for w, n in reps.items()},
           "missing": [f"{w:.1f}" for w in GRID if w not in curves], **analyse(curves)}
    if REFERENCE_W in curves:  # sensitivity: the reference at the same five replicates as the new points
        c5, _ = load(first_reference_reps=5)
        out["reference_first5"] = analyse(c5)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = main()
    f = lambda d: "reference" if d is None else (f"{d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] "
                                                 f"{d['wins']}/{d['ties']}/{d['losses']}")
    print("replicates:", r["replicates"], "| missing:", r["missing"])
    for w, v in r["curve"].items():
        print(f"  w_llm {w}: {v['mean']:.4f}  vs 0.2 {f(v['vs_reference'])}")
    s, o = r["loo_selection"], r["per_screen_oracle"]
    print(f"leave-one-screen-out selection: {s['mean']:.4f}  vs 0.2 {f(s['vs_reference'])}  chosen {s['chosen_w']}")
    print(f"per-screen oracle (upper bound): {o['mean']:.4f}  vs 0.2 {f(o['vs_reference'])}  best {o['best_w']}")
    if "reference_first5" in r:
        s5 = r["reference_first5"]["loo_selection"]
        print(f"[0.2 at 5 replicates] selection {s5['mean']:.4f} chosen {sorted(set(s5['chosen_w'].values()))}")
    if not r["missing"]:
        print("figure ->", figure(r))
