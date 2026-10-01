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


def _load() -> dict[str, dict[str, list[dict]]]:
    """arm -> screen -> runs, merged over every experiment-15 output file."""
    arms: dict[str, dict[str, list[dict]]] = {}
    for p in sorted(RES.glob("*.json")):
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


def summarize() -> dict:
    arms = _load()
    complete = {a: v for a, v in arms.items() if all(ds in v for ds in BENCH)}
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
    candidates = [r["model"] for r in sorted(online, key=lambda r: -r["mean"])[:2]]
    res = {"rows": rows, "reference": REF, "stage_b_candidates": candidates,
           "missing_engines": sorted(set(engine_names()) - {r["model"] for r in rows.values()})}
    OUT.write_text(json.dumps(res, indent=2))
    return res


def control(models: list[str] | None = None, seed: int = 0) -> dict:
    """Leakage control: train each engine's STATIC prior on labels permuted within every training screen
    and check that the top-5B recall falls to the random expectation 5B/N. A leak of target labels into
    the features or the pipeline would keep it high."""
    from ..arms.ml_engine_arm import MLEngineArm
    from ..oracle import DatasetOracle
    models = models or engine_names()
    out = {}
    for m in models:
        rec = {}
        for ds in BENCH:
            bs = BATCH_SIZES[ds]
            arm = MLEngineArm(ds, bs, model=m, online=False, seed=seed)
            rng = np.random.default_rng(seed)
            lab = arm._loo_train["label"].to_numpy().copy()
            for d in arm._loo_train["dataset"].unique():
                idx = np.flatnonzero(arm._loo_train["dataset"].to_numpy() == d)
                lab[idx] = rng.permutation(lab[idx])
            arm._loo_train["label"] = lab
            arm._rebuild_ranking()
            oracle = DatasetOracle(ds)
            top = arm._ranking[: 5 * bs]
            found = sum(oracle.reveal(top).values())
            rec[ds] = {"recall": found / oracle.total_hits, "random": 5 * bs / oracle.n_genes}
        out[m] = {"per_screen": rec,
                  "mean_recall": float(np.mean([v["recall"] for v in rec.values()])),
                  "mean_random": float(np.mean([v["random"] for v in rec.values()]))}
        print(f"  {m:12s} shuffled-label static recall {out[m]['mean_recall']:.3f}  "
              f"(random {out[m]['mean_random']:.3f})", flush=True)
    (RES / "control.json").write_text(json.dumps(out, indent=2))
    return out


def figure(res: dict) -> Path:
    import matplotlib.pyplot as plt
    from . import style as S
    rows = res["rows"]
    models = sorted({r["model"] for r in rows.values()},
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
    ax.set_yticks(y, models)
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
    print("figure ->", figure(r))
    if "--control" in sys.argv:
        control()
