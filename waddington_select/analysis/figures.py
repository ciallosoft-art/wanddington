"""
figures.py — the quantification figure set, drawn from the frozen benchmark results.

Design follows the dataviz method: form chosen by the data's job (change-over-time → lines;
magnitude → bars/heatmap; polarity → diverging), categorical hues assigned per entity in a fixed
order, one axis per chart, recessive chrome, and — because two categorical slots sit below 3:1 on the
light surface — every series is also direct-labelled or legended in ink, never colour alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from . import data as D
from . import style as S


def _save(fig, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    fig.savefig(p)
    plt.close(fig)
    return p


def discovery_curves(out: Path) -> Path:
    """Change-over-time: cumulative hit_ratio vs round, averaged over the 9 screens.

    No error band: the spread across screens is dominated by how hard each screen is (essential 0.59
    vs Carnevale 0.06), so a band here would encode screen difficulty, not uncertainty about a method
    — and six overlapping fills read as mud. Per-screen values live in the heatmap; exact endpoints
    live in the comparison bars. Endpoint labels are selective (hero + floor) to avoid a collision
    where three methods converge around 0.22.
    """
    df = D.load_methods(final=True)
    S.apply()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    labelled = {"waddington_c", "random"}
    for arm in S.METHOD_ORDER:
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue
        # average seeds within a screen first, then across the 9 screens
        per_ds = sub.groupby(["dataset", "round"])["hit_ratio"].mean().reset_index()
        mean = per_ds.groupby("round")["hit_ratio"].mean()
        c = S.METHOD_COLOR[arm]
        hero = arm == "waddington_c"
        ax.plot(mean.index, mean.values, color=c, linewidth=2.4 if hero else 1.8,
                marker="o", markersize=5 if hero else 4,
                markeredgecolor=S.SURFACE, markeredgewidth=0.8,
                label=S.METHOD_LABEL[arm], zorder=4 if hero else 3)
        if arm in labelled:
            ax.annotate(f"{mean.iloc[-1]:.3f}", (mean.index[-1], mean.iloc[-1]),
                        xytext=(7, 0), textcoords="offset points", va="center",
                        fontsize=9, color=S.INK_2)

    ax.set_xlabel("Round (batch of genes tested)")
    ax.set_ylabel("Cumulative hit ratio")
    ax.set_title("Discovery curves — hits found per round, averaged over 9 screens",
                 color=S.INK, fontsize=11.5, loc="left", pad=10)
    ax.set_xticks(sorted(df["round"].unique()))
    ax.set_xlim(0.9, df["round"].max() + 0.62)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=8.5, labelcolor=S.INK_2)
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "01_discovery_curves.png")


def method_comparison(out: Path) -> Path:
    """Magnitude, one entity per bar → a single hue + direct value labels."""
    means = D.method_means(D.load_methods(final=True))
    S.apply()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    labels = [S.METHOD_LABEL.get(a, a) for a in means.index]
    y = np.arange(len(means))[::-1]
    colors = [S.METHOD_COLOR.get(a, S.SLOTS[0]) for a in means.index]
    ax.barh(y, means.values, height=0.62, color=colors, zorder=3)
    for yi, v in zip(y, means.values):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=S.INK_2)

    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_xlabel("Hit ratio @ round 5  (mean over 9 screens × 5 seeds)")
    ax.set_title("The hybrid pipeline beats every baseline", color=S.INK, fontsize=11.5, loc="left", pad=10)
    ax.set_xlim(0, max(means.values) * 1.16)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "02_method_comparison.png")


def dataset_heatmap(out: Path) -> Path:
    """Magnitude across two keys → sequential single-hue ramp, with values written in."""
    df = D.load_methods(final=True)
    fr = D.final_round(df)
    m = fr.groupby(["arm", "dataset"])["hit_ratio"].mean().unstack()
    m = m.reindex([a for a in S.METHOD_ORDER if a in m.index])
    S.apply()
    fig, ax = plt.subplots(figsize=(8.6, 3.5))

    im = ax.imshow(m.values, cmap=S.cmap_sequential(), aspect="auto", vmin=0)
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=38, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(m.index)), [S.METHOD_LABEL.get(a, a) for a in m.index], fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            if np.isnan(v):
                continue
            # ink on light cells, surface on dark cells — the value is always legible
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color=S.SURFACE if v > 0.42 else S.INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.set_label("hit ratio @ R5", color=S.INK_2, fontsize=8.5)
    cb.outline.set_visible(False)
    ax.set_title("Per-screen hit ratio @ R5", color=S.INK, fontsize=11.5, loc="left", pad=10)
    ax.grid(visible=False)
    return _save(fig, out, "03_dataset_heatmap.png")


def ablations(out: Path) -> Path:
    """Polarity (helps / hurts) → diverging hues around a neutral zero."""
    df = D.ablation_deltas()
    if df.empty:
        return None
    S.apply()
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    y = np.arange(len(df))
    colors = [S.POS if d > 0 else S.NEG for d in df["delta"]]
    ax.barh(y, df["delta"], height=0.6, color=colors, zorder=3)
    ax.axvline(0, color=S.BASELINE, linewidth=1.0, zorder=2)
    for yi, d in zip(y, df["delta"]):
        off = 5 if d >= 0 else -5
        ax.annotate(f"{d:+.3f}", (d, yi), xytext=(off, 0), textcoords="offset points",
                    va="center", ha="left" if d >= 0 else "right", fontsize=8.5, color=S.INK_2)

    ax.set_yticks(y, df["label"], fontsize=9)
    # Colour states a fact about the VARIANT ("scored lower than the full arm"), not a verdict on the
    # component: a red "− online ML" means removing online ML costs 0.012, i.e. it is the most valuable part.
    ax.set_xlabel("Δ hit ratio @ R5   (variant − full C-arm)")
    ax.set_title("Ablations: each variant vs the full C-arm\nremoving (−) the online ML costs the most; memory adds nothing",
                 color=S.INK, fontsize=11.5, loc="left", pad=10)
    pad = max(abs(df["delta"])) * 0.42
    ax.set_xlim(min(df["delta"]) - pad, max(df["delta"]) + pad)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "04_ablations.png")


def agent_vs_pipeline(out: Path) -> Path:
    """The tool-using agent's per-screen delta — polarity again."""
    df = D.load_agent()
    if df is None or df.empty:
        return None
    S.apply()
    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    y = np.arange(len(df))
    colors = [S.POS if d > 0 else S.NEG for d in df["delta"]]
    ax.barh(y, df["delta"], height=0.6, color=colors, zorder=3)
    ax.axvline(0, color=S.BASELINE, linewidth=1.0, zorder=2)
    for yi, d in zip(y, df["delta"]):
        off = 5 if d >= 0 else -5
        ax.annotate(f"{d:+.3f}", (d, yi), xytext=(off, 0), textcoords="offset points",
                    va="center", ha="left" if d >= 0 else "right", fontsize=8.5, color=S.INK_2)

    mean_d = df["delta"].mean()
    ax.set_yticks(y, df["dataset"], fontsize=8.8)
    ax.set_xlabel("Δ hit ratio @ R5   (tool-using agent − pipeline)")
    ax.set_title(f"Giving the LLM tools makes it worse (mean {mean_d:+.3f})",
                 color=S.INK, fontsize=11.5, loc="left", pad=10)
    pad = max(abs(df["delta"])) * 0.45
    ax.set_xlim(min(df["delta"]) - pad, max(df["delta"]) + pad)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "05_agent_vs_pipeline.png")


def build_all(out: Path) -> list[Path]:
    made = []
    for fn in (discovery_curves, method_comparison, dataset_heatmap, ablations, agent_vs_pipeline):
        p = fn(out)
        if p:
            made.append(p)
    return made
