"""
explain.py — figures that answer "why these genes?" from a decision trace.

The headline is `attribution`: it splits every pick into ML-only / LLM-only / both and then shows the
**hit rate of each source**. That is the mechanistic version of the ablation study — instead of "the
LLM is worth +0.008 on average", it says *where* that comes from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import style as S
from .trace import hit_rate_by_source

SOURCES = ["both", "ml_only", "llm_only"]


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p)
    plt.close(fig)
    return p


def attribution(trace: dict, out: Path, tag: str = "") -> Path:
    """Left: what each round's batch was made of. Right: which source actually finds hits."""
    rounds = trace["rounds"]
    S.apply()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 3.9),
        gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.42},  # room for ax2's tick labels
    )

    # ── composition per round (stacked; 2px surface gap between segments) ──
    x = [r["round"] for r in rounds]
    bottom = np.zeros(len(rounds))
    for s in SOURCES:
        vals = np.array([r["counts"][s] for r in rounds], dtype=float)
        ax1.bar(x, vals, bottom=bottom, width=0.62, color=S.SOURCE_COLOR[s],
                label=S.SOURCE_LABEL[s], zorder=3, linewidth=2, edgecolor=S.SURFACE)
        biggest = max(sum(r["counts"].values()) for r in rounds)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 0.05 * biggest:  # only label segments with room, never every mark
                ax1.text(xi, b + v / 2, f"{int(v)}", ha="center", va="center",
                         fontsize=8, color=S.SURFACE if s == "both" else S.INK)
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Genes in the batch")
    ax1.set_title("What each batch was made of", color=S.INK, fontsize=11, loc="left", pad=8)
    ax1.legend(fontsize=8.5, labelcolor=S.INK_2, loc="upper center", ncols=3,
               bbox_to_anchor=(0.5, -0.22))
    ax1.grid(axis="x", visible=False)

    # ── hit rate by source (the payoff) ──
    agg = hit_rate_by_source(trace)
    y = np.arange(len(SOURCES))[::-1]
    rates = [agg[s]["hit_rate"] * 100 for s in SOURCES]
    ax2.barh(y, rates, height=0.6, color=[S.SOURCE_COLOR[s] for s in SOURCES], zorder=3)
    for yi, s, v in zip(y, SOURCES, rates):
        ax2.annotate(f"{v:.0f}%   (n={agg[s]['picked']})", (v, yi), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9, color=S.INK_2)
    ax2.set_yticks(y, [S.SOURCE_LABEL[s] for s in SOURCES], fontsize=9.5)
    ax2.set_xlabel("Hit rate of the picks (%)")
    ax2.set_title("Agreement is the strongest signal", color=S.INK, fontsize=11, loc="left", pad=8)
    ax2.set_xlim(0, max(rates) * 1.42 if max(rates) else 1)
    ax2.grid(axis="y", visible=False)

    fig.suptitle("")
    return _save(fig, out, f"06_attribution{tag}.png")


def shap_features(trace: dict, out: Path, tag: str = "") -> Path:
    """Which features drove the ML score, averaged over the traced rounds (LightGBM native SHAP)."""
    agg: dict[str, list[float]] = {}
    for r in trace["rounds"]:
        for f, v in (r.get("shap") or {}).items():
            agg.setdefault(f, []).append(v)
    if not agg:
        return None
    means = {f: float(np.mean(v)) for f, v in agg.items()}
    top = sorted(means.items(), key=lambda kv: -kv[1])[:10][::-1]

    S.apply()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(top))
    ax.barh(y, [v for _f, v in top], height=0.62, color=S.SLOTS[0], zorder=3)
    for yi, (_f, v) in zip(y, top):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=S.INK_2)
    ax.set_yticks(y, [f for f, _v in top], fontsize=9)
    ax.set_xlabel("mean |SHAP| over the selected genes")
    ax.set_title("What the ML model is actually using", color=S.INK, fontsize=11.5, loc="left", pad=10)
    ax.set_xlim(0, max(v for _f, v in top) * 1.18)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, f"07_shap_features{tag}.png")


def calibration(trace: dict, out: Path, tag: str = "") -> Path:
    """Is the ML score trustworthy? Predicted-score bin vs the hit rate actually observed."""
    pts = [(g["ml_score"], bool(g.get("hit"))) for r in trace["rounds"] for g in r["genes"] if "hit" in g]
    if len(pts) < 20:
        return None
    scores = np.array([p[0] for p in pts])
    hits = np.array([p[1] for p in pts], dtype=float)

    qs = np.quantile(scores, np.linspace(0, 1, 6))
    qs = np.unique(qs)
    if len(qs) < 3:
        return None
    idx = np.clip(np.digitize(scores, qs[1:-1]), 0, len(qs) - 2)

    centers, rates, ns = [], [], []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        centers.append(scores[m].mean())
        rates.append(hits[m].mean() * 100)
        ns.append(int(m.sum()))

    S.apply()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(centers, rates, color=S.SLOTS[0], linewidth=2.2, marker="o", markersize=6,
            markeredgecolor=S.SURFACE, markeredgewidth=0.9, zorder=3)
    # equal-size quintiles → n is constant, so it goes in the axis label, not on every point
    for c, r in zip(centers, rates):
        ax.annotate(f"{r:.0f}%", (c, r), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.8, color=S.INK_2)

    per_bin = int(np.mean(ns))
    ax.set_xlabel(f"ML score of the gene — quintiles of the tested genes (n≈{per_bin} each)")
    ax.set_ylabel("Observed hit rate (%)")
    # Say what the data actually shows: the extremes separate, the middle is flat. And note the
    # selection effect — these are only genes the agent chose to test, so the score range is narrow.
    lo, hi = rates[0], rates[-1]
    ax.set_title(
        f"Calibration — top quintile hits {hi:.0f}% vs {lo:.0f}% at the bottom\n"
        "(monotone at the extremes, flat in the middle; only tested genes, so the score range is narrow)",
        color=S.INK, fontsize=10.5, loc="left", pad=10)
    ax.set_ylim(0, max(rates) * 1.30)
    return _save(fig, out, f"08_calibration{tag}.png")


def build_all(trace: dict, out: Path, tag: str = "") -> list[Path]:
    made = []
    for fn in (attribution, shap_features, calibration):
        p = fn(trace, out, tag)
        if p:
            made.append(p)
    return made
