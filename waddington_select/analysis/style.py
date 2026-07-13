"""
style.py — one visual system for every figure.

Palette and rules follow the dataviz reference instance. The 6-slot categorical set below was run
through the validator (light surface #fcfcfb): lightness band PASS, chroma floor PASS, CVD separation
PASS (worst adjacent ΔE 24.2, target ≥12). Two slots (aqua 2.74:1, yellow 2.11:1) fall below 3:1
contrast on the light surface, so the **relief rule** applies — every figure using them ships visible
direct labels / a legend in ink, never colour alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── surfaces & ink (never wear a series colour) ─────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# ── categorical slots, in fixed order (colour follows the entity, never its rank) ──
SLOTS = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

# Methods get a FIXED slot each — a filter that drops a method never repaints the survivors.
METHOD_ORDER = ["waddington_c", "llm_reasoning", "online_adaptive", "static_ranker", "coreset", "random"]
METHOD_COLOR = dict(zip(METHOD_ORDER, SLOTS))
METHOD_LABEL = {
    "waddington_c": "Waddington C (ours)",
    "llm_reasoning": "LLM only",
    "online_adaptive": "Online ML",
    "static_ranker": "Static LOO-LGBM",
    "coreset": "Coreset",
    "random": "Random",
}

# Attribution sources (Phase B) — first three slots.
SOURCE_COLOR = {"both": SLOTS[0], "ml_only": SLOTS[1], "llm_only": SLOTS[2]}
SOURCE_LABEL = {"both": "ML + LLM agree", "ml_only": "ML only", "llm_only": "LLM only"}

# ── sequential (magnitude: heatmaps) — one hue, light→dark ──────────────────
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

# ── diverging (polarity: deltas) — two hues + a NEUTRAL GRAY midpoint ───────
DIVERGING = ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#a52220"]
POS = "#2a78d6"   # better
NEG = "#e34948"   # worse


def cmap_sequential():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wadd_seq", SEQ_BLUE)


def cmap_diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wadd_div", DIVERGING)


def apply() -> None:
    """Recessive chrome; text in ink tokens; hairline grid."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
    })
