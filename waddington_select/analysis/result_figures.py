"""
result_figures.py — result charts for the paper (English) and for the Chinese explainer, from one code path.

Every chart puts the key numbers on the marks, so a reader does not have to go to a table for them. Two kinds of
source:

  * result files under workspace/results/ (*.json), read directly;
  * numbers that exist only in a paper table, transcribed below with the table they come from. `paper()` checks
    each transcribed number against docs/results_tables.tex (local-only) and stops on any that is not there; on a
    clone without docs/ the check is skipped.

Visual system: analysis/style.py (validated categorical slots, sequential blue, recessive chrome). Colour follows
the entity (each method keeps its slot in every chart); emphasis charts grey everything but the subject. The
Chinese labels use the Simplified-Chinese face of the system Noto CJK collection, extracted once.

    python -m waddington_select.analysis.result_figures --lang en          -> workspace/results/figures/1x_*.png
    python -m waddington_select.analysis.result_figures --lang zh --out D  -> D/fig_*.png  (the explainer)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from . import style as S  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "workspace" / "results"
PAPER_TEX = REPO / "docs" / "results_tables.tex"
FONT_CACHE = REPO / "workspace" / "cache" / "fonts"

LANG = "en"
OUT = RES / "figures"
TEX = PAPER_TEX.read_text() if PAPER_TEX.exists() else None

# paper file names (the paper's figures 01-10 come from analysis/figures.py)
PAPER_NAMES = {"paired": "11_paired.png", "robustness": "12_robustness.png", "reasoning": "13_reasoning.png",
               "tiers": "14_tiers.png", "pairs": "15_pairs.png", "jurkat": "16_jurkat.png",
               "feature_families": "17_feature_families.png", "benchmark_config": "18_benchmark_config.png",
               "engines": "19_engines_both_tiers.png", "acquisition": "20_acquisition.png",
               "engines_in_system": "21_engines_in_system.png", "ablation": "22_ablation_tiers.png",
               "pair_engines": "23_pair_engines.png", "per_screen_full": "24_per_screen_full.png",
               "clean_features": "25_clean_features.png"}

C = S.METHOD_COLOR
BLUE, PURPLE, RED = S.SLOTS[0], S.SLOTS[4], S.NEG
DEEMPH = "#cfcdc4"          # de-emphasised bars (the rest, in an emphasis chart)
REPAIR = "#104281"           # the repaired pair arm (darkest step of the sequential blue)
LIGHT_BLUE = "#86b6ef"

SCREENS = ["IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22", "Scharenberg22", "Steinhart",
           "Replogle_K562_essential", "Replogle_K562_gwps"]
SCREEN_LABEL = ["IFN-γ", "IL-2", "Sanchez21", "Sanchez21-dn", "Carnevale22", "Scharenberg22", "Steinhart",
                "K562-Ess.", "K562-GWPS"]


def L(zh: str, en: str) -> str:
    return zh if LANG == "zh" else en


def _setup(lang: str) -> None:
    global LANG
    LANG = lang
    S.apply()
    family = ["DejaVu Sans"]
    if lang == "zh":
        from fontTools.ttLib import TTCollection
        FONT_CACHE.mkdir(parents=True, exist_ok=True)
        for weight in ("Regular", "Bold"):
            p = FONT_CACHE / f"NotoSansCJKsc-{weight}.otf"
            if not p.exists():
                col = TTCollection(f"/usr/share/fonts/opentype/noto/NotoSansCJK-{weight}.ttc")
                [face] = [f for f in col.fonts if f["name"].getDebugName(1) == "Noto Sans CJK SC"]
                face.save(str(p))
            fm.fontManager.addfont(str(p))
        family = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({"font.family": family, "font.size": 9, "axes.titlesize": 10,
                         "figure.dpi": 200, "savefig.dpi": 200})


def paper(*vals, nd: int = 3) -> None:
    """Every transcribed number must appear in the paper source (as 0.123 or .123)."""
    if TEX is None:
        return
    for v in vals:
        s = str(abs(v)) if isinstance(v, int) else f"{abs(v):.{nd}f}"
        assert s in TEX or s.lstrip("0") in TEX, f"{v} (as {s}) is not in results_tables.tex"


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def save(fig, key: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"fig_{key}.png" if LANG == "zh" else PAPER_NAMES[key])
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


def title(ax, text: str, sub: str | None = None) -> None:
    ax.set_title(text, loc="left", color=S.INK, fontsize=10, fontweight="bold", pad=14 if sub else 8)
    if sub:
        ax.text(0, 1.01, sub, transform=ax.transAxes, fontsize=8, color=S.INK_2, va="bottom")


def hbars(ax, labels, values, colors, err=None, fmt="{:.3f}", ref=None, ref_label=None, xmax=None):
    """Horizontal bars, first label on top; value at the bar end (past the error bar); a value of None leaves
    an empty separator row; optional vertical reference line labelled above the top bar."""
    n = len(labels)
    y = np.arange(n)[::-1]
    err = err or [None] * n
    hi = max(v + (e or 0) for v, e in zip(values, err) if v is not None)
    xmax = xmax or hi * 1.18
    for yi, v, e, c in zip(y, values, err, colors):
        if v is None:
            continue
        ax.barh(yi, v, height=0.58, color=c, edgecolor=S.SURFACE, linewidth=1.5, zorder=3)
        if e:
            ax.errorbar(v, yi, xerr=e, fmt="none", ecolor=S.INK_2, elinewidth=0.8, capsize=2, zorder=4)
        ax.text(v + (e or 0) + 0.01 * xmax, yi, fmt.format(v), va="center", ha="left", fontsize=8, color=S.INK)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4 + (0.5 if ref_label else 0))
    if ref is not None:
        ax.axvline(ref, color=S.INK_2, lw=0.9, zorder=5)
        if ref_label:
            ax.text(ref, n - 0.45, ref_label, fontsize=7.4, color=S.INK_2, ha="center", va="bottom")
    return y


def forest(ax, rows, xlabel, fmt="{:+.3f}", colors=None, filled=None):
    """rows: (label, delta, lo, hi, note). One point + CI per row, first row on top, zero line, values right."""
    y = np.arange(len(rows))[::-1]
    for i, (yi, (lab, d, lo, hi, note)) in enumerate(zip(y, rows)):
        col = (colors or [BLUE] * len(rows))[i]
        fill = True if filled is None else filled[i]
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(d, yi, "o", ms=7, mfc=col if fill else S.SURFACE, mec=col, mew=1.6, zorder=4)
    ax.axvline(0, color=S.INK_2, lw=0.9, zorder=2)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    lo_all = min(r[2] for r in rows)
    hi_all = max(r[3] for r in rows)
    span = hi_all - min(lo_all, 0)
    ax.set_xlim(min(lo_all, 0) - 0.05 * span, hi_all + 0.05 * span)
    for yi, (lab, d, lo, hi, note) in zip(y, rows):
        ax.text(1.02, yi, f"{fmt.format(d)}  [{lo:+.3f}, {hi:+.3f}]  {note}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8, color=S.INK)
    return y


def heat(ax, matrix, row_labels, col_labels=None, avg=True, fmt="{:.3f}"):
    """Per-screen grid: colour = value / best value on that screen (who wins where); the cell shows the value."""
    m = np.asarray(matrix, float)
    if avg:
        m = np.column_stack([m, m.mean(axis=1)])
    rel = m / m.max(axis=0, keepdims=True)
    cmap = S.cmap_sequential()
    nr, nc = m.shape
    for i in range(nr):
        for j in range(nc):
            x = j + (0.35 if avg and j == nc - 1 else 0)
            ax.add_patch(Rectangle((x, nr - 1 - i), 0.94, 0.9, color=cmap(rel[i, j] ** 2), lw=0))
            best = m[i, j] == m[:, j].max()
            ax.text(x + 0.47, nr - 1 - i + 0.45, fmt.format(m[i, j]), ha="center", va="center", fontsize=7.4,
                    color="white" if rel[i, j] ** 2 > 0.55 else S.INK, fontweight="bold" if best else "normal")
            if best:
                ax.add_patch(Rectangle((x + 0.03, nr - 1 - i + 0.03), 0.88, 0.84, fill=False, ec=S.INK, lw=1.1))
    ax.set_xlim(0, nc + (0.35 if avg else 0))
    ax.set_ylim(0, nr)
    cols = (col_labels or SCREEN_LABEL) + ([L("平均", "Mean")] if avg else [])
    ax.set_xticks([j + 0.47 + (0.35 if avg and j == nc - 1 else 0) for j in range(nc)], cols,
                  rotation=35 if col_labels is None else 0, ha="right" if col_labels is None else "center", fontsize=8)
    ax.set_yticks([nr - 1 - i + 0.45 for i in range(nr)], row_labels, fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def legend(ax, handles, **kw):
    kw.setdefault("loc", "lower right")
    kw.setdefault("fontsize", 7.8)
    ax.legend(handles=handles, labelcolor=S.INK_2, handlelength=1.4, **kw)


def dot_handle(color, label, filled=True):
    return Line2D([], [], marker="o", ls="", ms=6, mfc=color if filled else S.SURFACE, mec=color, mew=1.5,
                  label=label)


def name_b():
    return L("B（LLM 补位）", "Padded LLM (B)")


def name_static():
    return L("静态先验", "Static prior")


def hit_axis(extra_zh="", extra_en=""):
    return L(f"9 屏平均 hit@R5{extra_zh}", f"hit@R5, mean of 9 screens{extra_en}")


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
def fig_overview():
    """Explainer section 1: the reported system against the strongest baseline on each of BDA's three tiers."""
    mb = load("matched_baselines.json")
    h, f = mb["headline"]["means"], mb["full_pool"]["means"]
    pairs = [("Waddington C", 112.6, C["waddington_c"]), (name_b(), 128.2, C["llm_reasoning"]),
             ("BDA", 160.0, C["bda"]), ("Random", 60.0, C["random"])]
    paper(112.6, 128.2, 160.0, 60.0, nd=1)
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 1.95), gridspec_kw=dict(wspace=0.95))
    for ax, t, m in ((axs[0], L("单扰动・非必需（主档）", "Single gene, non-essential"), h),
                     (axs[1], L("单扰动・全池", "Single gene, full pool"), f)):
        hbars(ax, ["Waddington C", "Online ML", "Random"], [m["C"], m["matched_online_ml"], m["benchmark_random"]],
              [C["waddington_c"], C["online_adaptive"], C["random"]], xmax=0.34)
        ax.set_title(t, loc="left", fontsize=9, color=S.INK, fontweight="bold")
        ax.set_xlabel(hit_axis(), fontsize=8)
    hbars(axs[2], [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs], fmt="{:.1f}", xmax=205)
    axs[2].set_title(L("双扰动（Horlbeck K562）", "Two-gene (Horlbeck K562)"), loc="left", fontsize=9, color=S.INK,
                     fontweight="bold")
    axs[2].set_xlabel(L("640 次挑选的命中数", "hits in 640 picks"), fontsize=8)
    save(fig, "overview")


def fig_feature_families():
    """The prior (no LLM) by feature family, and the hub / PPI leak in single-feature rankings."""
    ff = load("feature_families.json")
    m, pv = ff["means"], ff["paired_vs_clean"]
    rows = [(L("anchor 3 项", "anchor (3)"), "anchor", False), (L("hub / PPI 2 项", "hub / PPI (2)"), "pooled", False),
            (L("全局 4 项", "global (4)"), "global", False), (L("DepMap 4 项", "DepMap (4)"), "depmap", False),
            (None, None, None),
            (L("全局 + hub/PPI", "global + hub/PPI"), "global+pooled (prior_probes 'intrinsic')", True),
            (L("干净 8 项（全局 + DepMap）", "clean 8 (global + DepMap)"), "global+depmap (clean)", True),
            (L("干净 8 项 + anchor 3 项", "clean 8 + anchor"), "global+depmap+anchor", True),
            (L("干净 8 项 + hub/PPI", "clean 8 + hub/PPI"), "global+depmap+pooled", True),
            (L("全部 13 项", "all 13"), "all thirteen", True)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 3.0), gridspec_kw=dict(width_ratios=[1.55, 1], wspace=0.75))
    labels, vals, cols = [], [], []
    for lab, key, combo in rows:
        if lab is None:
            labels.append(""), vals.append(None), cols.append(S.SURFACE)
            continue
        labels.append(lab), vals.append(m[key])
        cols.append(BLUE if key == "global+depmap (clean)" else DEEMPH)
    y = hbars(a, labels, vals, cols, xmax=0.42)
    a.text(0, y[4], L("  组合（与干净 8 项配对）", "  combinations (paired with clean 8)"), fontsize=7.5,
           color=S.INK_2, va="center")
    for yi, (lab, key, combo) in zip(y, rows):
        if combo and key in pv:
            p = pv[key]
            sig = p["ci95"][1] < 0 or p["ci95"][0] > 0
            a.text(0.418, yi, f"{p['delta']:+.3f}{'*' if sig else ''}", fontsize=7.5, ha="right", va="center",
                   color=S.INK if sig else S.INK_2)
    a.set_xlabel(L("静态先验（不调 LLM）9 屏平均 hit@R5", "static prior (no LLM), hit@R5"))
    title(a, L("按特征来源分组的先验", "The prior by feature family"),
          L("右列：与干净 8 项的配对差，* = 置信区间不含 0", "right: Δ vs clean 8; * CI excludes 0"))
    sf = load("hub_leak.json")["single_feature"]
    deg = load("ml_engines/summary.json")["label_free_post_hoc"]["string_degree_norm"]["mean"]
    items = [("hub score", "hub_score_norm"), (L("PPI 总权重", "PPI weight"), "ppi_score_sum")]
    yy = np.arange(len(items))[::-1]
    for yi, (lab, k) in zip(yy, items):
        r, o = sf["reported"][k]["mean"], sf["own_out"][k]["mean"]
        b.plot([o, r], [yi, yi], color=DEEMPH, lw=3, zorder=2)
        b.plot(r, yi, "o", ms=8, color=S.MUTED, zorder=3)
        b.plot(o, yi, "o", ms=8, color=BLUE, zorder=3)
        b.text(r, yi - 0.3, f"{r:.3f}", fontsize=7.8, color=S.INK_2, ha="center", va="top")
        b.text(o, yi - 0.3, f"{o:.3f}", fontsize=7.8, color=S.INK, ha="center", va="top")
    b.axvline(deg, color=S.INK_2, lw=0.9)
    b.text(deg + 0.0005, 1.62, L(f"全基因组 degree {deg:.3f}", f"genome-wide degree {deg:.3f}"), fontsize=7.5,
           color=S.INK_2, ha="left")
    b.set_yticks(yy, [i[0] for i in items])
    b.set_ylim(-0.6, 1.8)
    b.set_xlim(0.155, 0.188)
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0)
    b.set_xlabel(L("单特征排序 hit@R5", "single-feature ranking, hit@R5"))
    title(b, L("单特征排序：去掉本屏 anchor/hit 后", "Own anchors and hits removed"))
    legend(b, [dot_handle(S.MUTED, L("原查询集", "stored query set")),
               dot_handle(BLUE, L("去掉本屏 anchor 与 hit", "target's anchors and hits removed"))],
           loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    save(fig, "feature_families")


def _main_rows(tier: str):
    mb = load("matched_baselines.json")[tier]
    rep, mat = mb["reported"], mb["matched"]
    return {
        "Waddington C": [rep[s]["waddington_c"][-1] for s in SCREENS],
        "Online ML": [mat[s]["online_ml"][-1] for s in SCREENS],
        name_static(): [mat[s]["static_prior"][-1] for s in SCREENS],
        name_b(): [rep[s]["llm_reasoning"][-1] for s in SCREENS],
        "Coreset": [mat[s]["coreset"][-1] for s in SCREENS],
        "Random": [rep[s]["random"][-1] for s in SCREENS],
    }


def fig_per_screen(tier: str, key: str, gd, bda, ttl: str):
    rows = _main_rows(tier)
    rows["GeneDisco"] = gd
    rows["BDA"] = bda
    order = ["Waddington C", "Online ML", name_static(), name_b(), "GeneDisco", "Coreset", "BDA", "Random"]
    fig, ax = plt.subplots(figsize=(6.6, 3.35))
    heat(ax, [rows[k] for k in order], order)
    title(ax, ttl, L("颜色 = 该屏内相对最高分的比例；加粗加框 = 该屏最高",
                     "colour = share of the best score on that screen; bold and boxed = best on the screen"))
    save(fig, key)




def fig_paired():
    """C minus each baseline, paired over the nine screens (paper Table 5's paired block)."""
    rows = [(L("静态先验（0.231）", "Static prior (0.231)"), 0.030, 0.016, 0.044, "8/0/1"),
            (L("Online ML（0.245，即 C 去掉 LLM）", "Online ML = C without LLM (0.245)"), 0.015, 0.004, 0.026, "7/0/2"),
            (L("B：LLM 补位（0.212）", "Padded LLM, B (0.212)"), 0.049, 0.019, 0.080, "8/0/1"),
            (L("Coreset（0.154）", "Coreset (0.154)"), 0.107, 0.060, 0.155, "9/0/0")]
    for r in rows:
        paper(*r[1:4])
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    forest(ax, rows, xlabel=L("C − 基线（hit@R5），95% CI 按屏聚类",
                              "C minus baseline (hit@R5), screen-clustered 95% CI"))
    ax.text(1.02, 1.04, L("差值  [95% CI]  胜/平/负", "Δ  [95% CI]  W/T/L"), transform=ax.transAxes, fontsize=7.8,
            color=S.INK_2)
    title(ax, L("C 相对每个基线的配对差（9 屏）", "C minus each baseline"))
    save(fig, "paired")


def fig_clean_features():
    """Experiment 22: the full system re-run without hub score and total PPI weight."""
    cc = load("clean_connectivity.json")["means"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.4, 1.9),
                               gridspec_kw=dict(width_ratios=[1, 1.25], wspace=0.6 if LANG == "zh" else 1.0))
    items = [("Waddington C", cc["c_reported"], cc["c_clean"]),
             (L("同配置静态先验", "matched static prior"), cc["prior_reported"], cc["prior_clean"])]
    yy = np.arange(2)[::-1]
    for yi, (lab, r, c) in zip(yy, items):
        a.plot([r, c], [yi, yi], color=DEEMPH, lw=3, zorder=2)
        a.plot(r, yi, "o", ms=8, color=S.MUTED, zorder=3)
        a.plot(c, yi, "o", ms=8, color=BLUE, zorder=3, mec=S.SURFACE, mew=1.2)
        a.text(max(r, c) + 0.002, yi, f"{r:.3f} → {c:.3f}", fontsize=8, va="center", color=S.INK)
    a.set_yticks(yy, [i[0] for i in items])
    a.set_ylim(-0.6, 1.6)
    a.set_xlim(0.225, 0.300)
    a.grid(axis="y", visible=False)
    a.tick_params(axis="y", length=0)
    a.set_xlabel(hit_axis())
    legend(a, [dot_handle(S.MUTED, L("原特征", "stored features")),
               dot_handle(BLUE, L("去掉 hub / PPI", "without hub / PPI"))], loc="upper center",
           bbox_to_anchor=(0.45, -0.32), ncol=2)
    title(a, L("两者的水平", "Levels"))
    rows = [(L("原特征", "stored features"), 0.030, 0.016, 0.044, "8/0/1"),
            (L("去掉 hub / PPI", "without hub / PPI"), 0.020, 0.005, 0.035, "7/0/2")]
    for r in rows:
        paper(*r[1:4])
    forest(b, rows, xlabel=L("C − 同配置静态先验", "C minus matched static prior"))
    title(b, L("系统相对先验的增益", "Gain over the prior"))
    save(fig, "clean_features")


def fig_ablation():
    """Component ablations on the three tiers (paper Table 19)."""
    comps = [L("去掉跨实验记忆", "no memory"), L("去掉在线重训", "no online retraining"),
             L("匿名基因名", "anonymous names"), L("去掉 LLM", "no LLM")]
    head = [(-0.005, -0.011, -0.000), (-0.015, -0.025, -0.006), (-0.026, -0.043, -0.010), (-0.016, -0.027, -0.005)]
    full = [(-0.005, -0.011, -0.000), (-0.006, -0.016, 0.004), (-0.021, -0.038, -0.007), (-0.014, -0.027, -0.004)]
    pair = [3.2, 2.6, -22.0, -22.0]
    pair_p = ["p = 0.52", "p = 0.49", "p < 1e-6", "p < 1e-6"]
    for t in head + full:
        paper(*t)
    paper(*pair, nd=1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.72))
    y = np.arange(len(comps))[::-1]
    for off, data, col in ((0.16, head, BLUE), (-0.16, full, PURPLE)):
        for yi, (d, lo, hi) in zip(y, data):
            sig = hi < 0
            a.plot([lo, hi], [yi + off] * 2, color=col, lw=2, solid_capstyle="round")
            a.plot(d, yi + off, "o", ms=6.5, mfc=col if sig else S.SURFACE, mec=col, mew=1.5)
    a.axvline(0, color=S.INK_2, lw=0.9)
    for yi, (d, *_), (d2, *_) in zip(y, head, full):
        a.text(1.02, yi, f"{d:+.3f} / {d2:+.3f}", transform=a.get_yaxis_transform(), fontsize=8, va="center")
    a.text(1.02, 1.03, L("主档 / 全池", "headline / full pool"), transform=a.transAxes, fontsize=7.8, color=S.INK_2)
    a.set_yticks(y, comps)
    a.tick_params(axis="y", length=0)
    a.grid(axis="y", visible=False)
    a.set_xlabel(L("变体 − 完整系统（hit@R5），95% CI", "variant minus full system (hit@R5), 95% CI"))
    legend(a, [dot_handle(BLUE, L("主档", "headline")), dot_handle(PURPLE, L("全池", "full pool")),
               dot_handle(S.MUTED, L("区间含 0", "CI includes 0"), filled=False)], loc="upper left")
    title(a, L("单扰动两档（各 20 次）", "Single-gene tiers"))
    cols = [RED if v < 0 else DEEMPH for v in pair]
    b.barh(y, pair, height=0.55, color=cols, edgecolor=S.SURFACE, zorder=3)
    for yi, v, p in zip(y, pair, pair_p):
        b.text(max(v, 0) + 1.5, yi, L(f"{v:+.1f}（{p}）", f"{v:+.1f} ({p})"), fontsize=7.5, va="center", ha="left",
               color=S.INK)
    b.axvline(0, color=S.INK_2, lw=0.9)
    b.set_yticks(y, [""] * len(y))
    b.set_xlim(-30, 40)
    b.grid(axis="y", visible=False)
    b.set_xlabel(L("命中数差（640 次挑选）", "difference in hits (640 picks)"))
    title(b, L("双扰动档", "Two-gene tier"))
    save(fig, "ablation")


LADDER = [("A0：只给匿名 ID", "A0: anonymous IDs", 0.235), ("A0.5：+ ML 候选池", "A0.5: + ML shortlist", 0.194),
          ("A1：+ 每个候选的结构特征", "A1: + structural features", 0.241),
          ("A2：真实基因名（= C）", "A2: real gene names (= C)", 0.261),
          ("参照：ML 引擎单独", "reference: ML engine alone", 0.245), ("参照：质心规则", "reference: centroid rule", 0.250)]


def _ladder_panel(a):
    paper(*[v for *_, v in LADDER])
    paper(0.020, 0.005, 0.037)
    cols = [DEEMPH, DEEMPH, DEEMPH, BLUE, "#e8e6df", "#e8e6df"]
    y = hbars(a, [L(zh, en) for zh, en, _ in LADDER], [v for *_, v in LADDER], cols, xmax=0.40)
    a.annotate("", xy=(0.318, y[3]), xytext=(0.318, y[2]), arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1))
    a.text(0.326, (y[2] + y[3]) / 2, L("名字\n+0.020\n[.005, .037]", "name\n+0.020\n[.005, .037]"), fontsize=7.2,
           va="center", color=S.INK)
    a.set_xlabel(hit_axis())


def fig_reasoning():
    """Paper: reasoning vs recall on the reported system (paper Table 8)."""
    fig, a = plt.subplots(figsize=(4.6, 2.5))
    _ladder_panel(a)
    title(a, L("推理还是记忆：逐级给 LLM 信息", "Reasoning or recall: information given to the LLM, step by step"))
    save(fig, "reasoning")




ENGINE_ROUND1 = [("gnn", "GraphSAGE（STRING 图神经网络）", "GraphSAGE on STRING"), ("mlp", "MLP（2×64）", "MLP (2×64)"),
                 ("genept_lgbm", "LightGBM + GenePT 嵌入", "LightGBM + GenePT"), ("logreg", "逻辑回归", "Logistic regression"),
                 ("svm_linear", "线性 SVM", "Linear SVM"), ("rf", "随机森林", "Random forest"),
                 ("rwr", "RWR 网络传播", "RWR on STRING"), ("svm_rbf", "RBF 核 SVM", "RBF SVM (Nyström)"),
                 ("lambdarank", "LambdaMART（排序学习）", "LambdaMART"), ("genept_lr", "GenePT 嵌入 + 逻辑回归", "GenePT + logistic")]
ENGINE_ROUND2 = [("lgbm_score", "学连续分数：回归百分位", "regression on score percentile"),
                 ("lgbm_score_xent", "学连续分数：交叉熵软目标", "cross-entropy, soft target"),
                 ("lgbm_score_hard", "回归，已测基因用 0/1 †", "regression, revealed genes 0/1 †"),
                 ("lgbm_score_xent_hard", "交叉熵，已测基因用 0/1 †", "cross-entropy, revealed genes 0/1 †"),
                 ("lgbm_depmap_pca", "+ DepMap 依赖谱 PCA-32", "+ DepMap profile, PCA-32"),
                 ("lgbm_string_emb", "+ STRING 拉普拉斯 32 维", "+ STRING Laplacian eigenmap, 32"),
                 ("lgbm_desc", "+ 两种描述都加", "+ both descriptors"),
                 ("bnn", "贝叶斯 MLP（GeneDisco 的模型）", "Bayesian MLP (MC dropout)"),
                 ("mf", "多任务矩阵分解", "Multi-task matrix factorisation"),
                 ("svgp", "稀疏变分高斯过程", "Sparse variational GP"), ("lgbm_rwr", "LightGBM + RWR 特征 †", "LightGBM + RWR feature †")]
# paper Table 17 (full-pool tier): online hit@R5, difference from online LightGBM and its 95% CI
ENGINE_FULL = {"gnn": (0.242, -0.003, -0.020, 0.016), "mlp": (0.241, -0.004, -0.016, 0.009),
               "genept_lgbm": (0.236, -0.009, -0.019, 0.001), "logreg": (0.233, -0.012, -0.025, 0.004),
               "svm_linear": (0.230, -0.015, -0.026, -0.003), "rf": (0.229, -0.015, -0.039, 0.003),
               "svm_rbf": (0.218, -0.027, -0.039, -0.016), "rwr": (0.217, -0.028, -0.058, -0.003),
               "lambdarank": (0.203, -0.041, -0.078, -0.012), "genept_lr": (0.175, -0.069, -0.130, -0.027),
               "lgbm_score": (0.247, 0.003, -0.011, 0.017), "lgbm_score_xent": (0.250, 0.005, -0.006, 0.021),
               "lgbm_score_hard": (0.253, 0.008, -0.004, 0.024), "lgbm_score_xent_hard": (0.253, 0.008, -0.008, 0.025),
               "lgbm_depmap_pca": (0.240, -0.005, -0.015, 0.004), "lgbm_string_emb": (0.248, 0.003, -0.008, 0.018),
               "lgbm_desc": (0.234, -0.011, -0.025, 0.000), "bnn": (0.238, -0.007, -0.017, 0.006),
               "mf": (0.155, -0.090, -0.141, -0.043), "svgp": (0.239, -0.006, -0.015, 0.003),
               "lgbm_rwr": (0.236, -0.009, -0.026, 0.008)}


def fig_engines():
    """Every engine's online hit@R5 minus online LightGBM's, headline (repo summary) and full pool (paper
    Table 17)."""
    rows = load("ml_engines/summary.json")["rows"]
    for v in ENGINE_FULL.values():
        paper(*v)
    groups = [(L("第一轮：换模型族", "Round 1: model family"), ENGINE_ROUND1),
              (L("第二轮：换学习目标、加描述、换代理模型", "Round 2: target, descriptors, surrogate"), ENGINE_ROUND2)]
    labels, ypos, data, heads = [], [], [], []
    yv = 0
    for gname, items in reversed(groups):
        items = sorted(items, key=lambda kv: rows[f"ml_engine_{kv[0]}"]["vs_lightgbm_online"]["delta"])
        for key, zh, en in items:
            r = rows[f"ml_engine_{key}"]
            data.append((yv, r["mean"], r["vs_lightgbm_online"], ENGINE_FULL[key]))
            labels.append(L(zh, en))
            ypos.append(yv)
            yv += 1
        labels.append(gname)
        heads.append(gname)
        ypos.append(yv)
        yv += 1.3
    fig, ax = plt.subplots(figsize=(5.0, 6.4))
    for yi, mean, hv, fv in data:
        for off, (d, lo, hi), col in ((0.17, (hv["delta"], *hv["ci95"]), BLUE), (-0.17, fv[1:], PURPLE)):
            sig = hi < 0 or lo > 0
            ax.plot([lo, hi], [yi + off] * 2, color=col, lw=1.8, solid_capstyle="round")
            ax.plot(d, yi + off, "o", ms=5.8, mfc=col if sig else S.SURFACE, mec=col, mew=1.4)
        ax.text(1.02, yi, f"{mean:.3f} / {fv[0]:.3f}", transform=ax.get_yaxis_transform(), fontsize=7.8,
                va="center")
    ax.axvline(0, color=S.INK_2, lw=0.9)
    ax.set_yticks(ypos, labels, fontsize=8)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab in heads:
            t.set_fontweight("bold")
            t.set_color(S.INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, yv - 0.6)
    ax.set_xlabel(L("在线引擎 − 在线 LightGBM（hit@R5），95% CI 按屏聚类",
                    "online engine minus online LightGBM (hit@R5), screen-clustered 95% CI"))
    ax.text(1.02, 1.005, L("hit@R5 主档 / 全池", "hit@R5 headline / full"), transform=ax.transAxes, fontsize=7.8,
            color=S.INK_2)
    legend(ax, [dot_handle(BLUE, L("主档", "headline tier")), dot_handle(PURPLE, L("全池", "full-pool tier")),
                dot_handle(S.MUTED, L("区间含 0（无显著差异）", "CI includes 0"), filled=False)], loc="upper left")
    title(ax, L("只换 ML 引擎：没有一个超过 LightGBM（主档与全池都是 0.245）",
                "Swapping the ML engine: none beats LightGBM (0.245 on both tiers)"),
          L("只跑 ML；实心 = 区间不含 0；† = 事后加的变体", "ML only; filled = CI excludes 0; † = post hoc"))
    save(fig, "engines")


def fig_acquisition():
    """Acquisition rules on a LightGBM ensemble and on GeneDisco's Bayesian MLP (repo summary)."""
    s = load("ml_engines/summary.json")
    single = s["rows"]["ml_engine_lightgbm"]["mean"]
    acq = s["acquisition"]
    lg = [(L("贪心（现行规则）", "greedy (reported rule)"), acq["ml_acq_greedy"], None),
          ("UCB, β = 0.5" if LANG == "en" else "UCB，β = 0.5", acq["ml_acq_ucb05"], "vs_ensemble_greedy"),
          (L("Thompson 采样", "batch Thompson"), acq["ml_acq_thompson"], "vs_ensemble_greedy"),
          ("UCB, β = 1" if LANG == "en" else "UCB，β = 1", acq["ml_acq_ucb1"], "vs_ensemble_greedy"),
          (L("margin（不确定性采样）", "margin (uncertainty)"), acq["ml_acq_margin"], "vs_ensemble_greedy")]
    bn = s["bnn_acquisition"]
    bnn_greedy = {"mean": s["rows"]["ml_engine_bnn"]["mean"]}
    bb = [(L("贪心", "greedy"), bnn_greedy, None), ("UCB", bn["ml_bnnacq_ucb1"], "vs_bnn_greedy"),
          ("BALD", bn["ml_bnnacq_bald"], "vs_bnn_greedy"),
          (L("k-means（嵌入）", "k-means (embedding)"), bn["ml_bnnacq_kmeans_embedding"], "vs_bnn_greedy"),
          ("BADGE", bn["ml_bnnacq_badge"], "vs_bnn_greedy")]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(wspace=0.85))

    def col(r, k):
        if k is None:
            return BLUE
        return RED if r[k]["ci95"][1] < 0 else DEEMPH

    for ax, rows, t in ((axs[0], lg, L("LightGBM（5 成员集成）", "LightGBM, 5-member ensemble")),
                        (axs[1], bb, L("贝叶斯 MLP（GeneDisco）", "Bayesian MLP (GeneDisco)"))):
        hbars(ax, [r[0] for r in rows], [r[1]["mean"] for r in rows], [col(r[1], r[2]) for r in rows],
              ref=single, ref_label=L(f"单个 LightGBM {single:.3f}", f"single LightGBM {single:.3f}"), xmax=0.31)
        ax.set_xlabel(L("9 屏平均 hit@R5（5 个 seed）", "hit@R5, mean of 9 screens (5 seeds)"))
        title(ax, t)
    fig.legend(handles=[Patch(color=BLUE, label=L("贪心（按命中概率取前 B）", "greedy: top-B by P(hit)")),
                        Patch(color=DEEMPH, label=L("与贪心无显著差异", "no detected difference from greedy")),
                        Patch(color=RED, label=L("显著差于贪心", "worse than greedy (CI below 0)"))],
               loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.8, labelcolor=S.INK_2)
    save(fig, "acquisition")


def fig_engines_in_system():
    """The two closest engines inside the full GPT system (repo carm_summary)."""
    c = load("ml_engines/carm_summary.json")
    rows = load("ml_engines/summary.json")["rows"]
    items = [(L("LightGBM（报告）", "LightGBM (reported)"), rows["ml_engine_lightgbm"]["mean"], c["reference"]["mean"], 20),
             ("GraphSAGE", rows["ml_engine_gnn"]["mean"], c["arms"]["waddington_c__gnn"]["mean"], 5),
             ("MLP", rows["ml_engine_mlp"]["mean"], c["arms"]["waddington_c__mlp"]["mean"], 5)]
    fig, ax = plt.subplots(figsize=(4.8, 1.7))
    yy = np.arange(3)[::-1]
    for yi, (lab, ml, full, n) in zip(yy, items):
        ax.annotate("", xy=(full, yi), xytext=(ml, yi), arrowprops=dict(arrowstyle="-|>", color=DEEMPH, lw=2.2))
        ax.plot(ml, yi, "o", ms=8, color=S.MUTED, zorder=3)
        ax.plot(full, yi, "o", ms=8, color=BLUE, zorder=3)
        ax.text(1.02, yi, L(f"{ml:.3f} → {full:.3f}（LLM +{full - ml:.3f}，{n} 次）",
                            f"{ml:.3f} → {full:.3f} (LLM +{full - ml:.3f}, {n} replicates)"),
                transform=ax.get_yaxis_transform(), fontsize=8, va="center")
    ax.set_yticks(yy, [i[0] for i in items])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(0.232, 0.264)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(hit_axis())
    legend(ax, [dot_handle(S.MUTED, L("只跑 ML 引擎", "ML engine alone")),
                dot_handle(BLUE, L("放进完整系统（+ LLM）", "in the full system (+ LLM)"))], loc="lower left")
    title(ax, L("换引擎后的完整系统：都不比 LightGBM 好", "Engines inside the full system: none beats LightGBM"))
    save(fig, "engines_in_system")


def fig_robustness():
    """The within-system gain under every stricter variant (paper Tables 6-7 and the robustness text)."""
    rows = [(L("报告系统（20 次）", "reported system (20 replicates)"), 0.030, 0.016, 0.044, "8/0/1"),
            (L("只看 BDA 原有 7 屏", "BDA's original 7 screens"), 0.024, 0.009, 0.040, "6/0/1"),
            (L("只用 5 次重复", "5 replicates"), 0.027, 0.014, 0.040, "8/0/1"),
            (L("每折重新选权重", "weight re-selected per fold"), 0.028, 0.012, 0.043, "7/0/2"),
            (L("嵌套路由", "nested router"), 0.030, 0.016, 0.043, "8/0/1"),
            (L("按研究排除选择（固定权重）", "study-excluded selection, fixed"), 0.026, 0.009, 0.042, "7/0/2"),
            (L("按研究排除选择（嵌套）", "study-excluded selection, nested"), 0.028, 0.012, 0.042, "8/0/1"),
            (L("完整留一研究 ¹", "full leave-one-study-out ¹"), 0.031, 0.011, 0.053, "8/0/1"),
            (L("去掉 hub / PPI 后重跑 ²", "re-run without hub / PPI ²"), 0.020, 0.005, 0.035, "7/0/2")]
    for r in rows:
        paper(*r[1:4])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    forest(ax, rows, xlabel=L("C − 同配置静态先验（hit@R5），95% CI 按屏聚类",
                              "C minus matched static prior (hit@R5), screen-clustered 95% CI"))
    ax.text(1.02, 1.02, L("差值  [95% CI]  胜/平/负", "Δ  [95% CI]  W/T/L"), transform=ax.transAxes, fontsize=7.8,
            color=S.INK_2)
    ax.text(-0.55, -0.3, L("¹ 先验训练也排除同研究的姊妹屏，参照是同样排除姊妹屏的先验（0.211）。\n"
                           "² 参照是同样去掉这两项的先验（0.240）。",
                           "¹ the prior is also denied same-study siblings; reference = sibling-excluded prior (0.211).\n"
                           "² reference = the prior without the same two features (0.240)."),
            transform=ax.transAxes, fontsize=7.4, color=S.INK_2, va="top")
    title(ax, L("更严格的设置下，系统相对先验的增益都不含 0", "Gain over the matched prior"))
    save(fig, "robustness")


def fig_tiers():
    """Every method on the two single-gene tiers."""
    mb = load("matched_baselines.json")
    h, f = mb["headline"]["means"], mb["full_pool"]["means"]
    paper(0.146, 0.166, 0.171, 0.170)
    items = [("Waddington C", h["C"], f["C"]), ("Online ML", h["matched_online_ml"], f["matched_online_ml"]),
             (name_static(), h["matched_static_prior"], f["matched_static_prior"]),
             (name_b(), h["benchmark_llm_reasoning"], f["benchmark_llm_reasoning"]),
             ("GeneDisco", 0.171, 0.170), ("BDA", 0.146, 0.166), ("Coreset", h["matched_coreset"], f["matched_coreset"]),
             ("Random", h["benchmark_random"], f["benchmark_random"])]
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    yy = np.arange(len(items))[::-1]
    for yi, (lab, a, b) in zip(yy, items):
        ax.plot([a, b], [yi + 0.13, yi - 0.13], color=DEEMPH, lw=2, zorder=2)
        ax.plot(a, yi + 0.13, "o", ms=7, color=BLUE, zorder=3, mec=S.SURFACE, mew=1)
        ax.plot(b, yi - 0.13, "s", ms=6.5, color=PURPLE, zorder=3, mec=S.SURFACE, mew=1)
        ax.text(1.02, yi, f"{a:.3f} / {b:.3f}", transform=ax.get_yaxis_transform(), fontsize=8, va="center")
    ax.text(1.02, 1.02, L("主档 / 全池", "headline / full pool"), transform=ax.transAxes, fontsize=7.8, color=S.INK_2)
    ax.set_yticks(yy, [i[0] for i in items])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.05, 0.29)
    ax.set_xlabel(hit_axis())
    legend(ax, [Line2D([], [], marker="o", ls="", ms=7, color=BLUE, label=L("非必需（主档）", "non-essential (headline)")),
                Line2D([], [], marker="s", ls="", ms=7, color=PURPLE, label=L("全池", "full pool"))], loc="upper left")
    title(ax, L("两个单扰动档：排序基本不变", "The two single-gene tiers: the ordering barely moves"),
          L("只有 BDA 在全池档明显上升（必需基因正是 LLM 容易点名的）",
            "only BDA rises clearly on the full pool"))
    save(fig, "tiers")


def fig_pairs():
    """The two-gene tier at 640 picks (paper Tables 11-12, 21 and the pair text)."""
    rows = [("Random", 60.0, 7.3, C["random"]), (name_static(), 40, None, C["static_ranker"]),
            ("Coreset", 77, None, C["coreset"]), ("Online ML", 83, None, C["online_adaptive"]),
            (L("Waddington C（报告配置）", "Waddington C (reported)"), 112.6, 24.7, C["waddington_c"]),
            (name_b(), 128.2, 22.1, C["llm_reasoning"]),
            (L("C + 随机森林引擎（仅 K562）", "C with random-forest engine"), 157.5, 25.5, LIGHT_BLUE),
            (L("BDA（GPT-5.6 Luna）", "BDA (our run, GPT-5.6 Luna)"), 160.0, 15.7, C["bda"]),
            ("", None, None, S.SURFACE),
            (L("交互先验单独（不用 LLM、不学习）", "interaction prior alone"), 196, None, REPAIR),
            (L("修复：不用 LLM", "repair without its LLM"), 239, None, REPAIR),
            (L("修复：LLM 名字优先", "repair, LLM names first"), 220.9, 10.9, REPAIR),
            (L("修复：LLM 作 0.2 验证者", "repair, LLM as 0.2 verifier"), 232.6, 6.0, REPAIR)]
    paper(*[r[1] for r in rows if r[1] is not None], nd=1)
    fig, ax = plt.subplots(figsize=(5.75, 3.6))
    y = hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows],
              err=[r[2] for r in rows], fmt="{:g}", ref=60, ref_label=L("随机 60", "random 60"), xmax=300)
    ax.text(3, y[8], L("诊断后新增的修复臂（单独实验，报告配置不变）",
                       "the repaired arm, added after the diagnosis (separate experiment)"),
            fontsize=7.6, color=S.INK_2, va="center")
    ax.set_xlabel(L("640 次挑选的累计命中数（误差线 = 20 次重复的标准差）",
                    "cumulative hits in 640 picks (error bars: s.d. over 20 replicates)"))
    title(ax, L("双扰动档：报告系统输给 BDA；修复臂在 K562 上反超",
                "Two-gene tier (Horlbeck K562): the reported system loses to BDA"),
          L("静态先验 40 低于随机：单基因特征描述不了相互作用",
            "the static prior (40) is below random: single-gene features cannot describe an interaction"))
    save(fig, "pairs")


def fig_pair_engines():
    """ML engines on the two pair tables (paper Table 21)."""
    names = [L("LightGBM（报告）", "LightGBM (reported)"), L("随机森林", "Random forest"), L("逻辑回归", "Logistic regression"),
             L("线性 SVM", "Linear SVM"), L("稀疏高斯过程", "Sparse variational GP"), L("贝叶斯 MLP", "Bayesian MLP"),
             "MLP", "LambdaMART"]
    mean_tab = [(83, None), (117.8, 15.5), (99, None), (84, None), (79.8, 29.4), (73.2, 21.6), (71.0, 14.4), (70, None)]
    inter_tab = [(158, None), (154.4, 17.5), (97, None), (98, None), (71.0, 9.0), (100.2, 57.7), (125.6, 40.7), (70, None)]
    paper(*[v for v, _ in mean_tab + inter_tab], *[e for _, e in mean_tab + inter_tab if e], nd=1)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(wspace=0.15))
    for ax, tab, t in ((axs[0], mean_tab, L("报告用的配对表（均值 + 差）", "Pair-mean table (reported)")),
                       (axs[1], inter_tab, L("加入 8 个交互特征的表", "With the interaction features"))):
        cols = [BLUE] + [DEEMPH] * (len(names) - 1)
        hbars(ax, names, [v for v, _ in tab], cols, err=[e for _, e in tab], fmt="{:g}", ref=60,
              ref_label=L("随机 60", "random 60"), xmax=235)
        ax.set_xlabel(L("640 次挑选的命中数（只跑 ML）", "hits in 640 picks (ML only)"))
        title(ax, t)
    axs[1].set_yticklabels([])
    save(fig, "pair_engines")


def fig_jurkat():
    """The pre-registered held-out test, both maps at both budgets (paper Tables 12 and 32)."""
    names = ["Random", L("Waddington C（报告配置）", "Waddington C (reported)"), "BDA",
             L("交互先验单独", "interaction prior alone"), L("修复：不用 LLM", "repair without its LLM"),
             L("修复：LLM 名字优先", "repair, LLM names first"), L("修复：LLM 作 0.2 验证者", "repair, LLM as 0.2 verifier")]
    b640 = {"K562": [60.0, 112.6, 160.0, 196, 239, 220.9, 232.6], "Jurkat": [63.5, 104.0, 109.2, 96, 88, 123.5, 110.8]}
    b160 = {"K562": [16.3, 31.9, 55.9, 65, 72, 68.5, 76.2], "Jurkat": [15.6, 28.4, 33.8, 30, 22, 31.9, 29.2]}
    for d in (b640, b160):
        paper(*d["K562"], *d["Jurkat"], nd=1)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(wspace=0.12))
    for ax, d, t, xm in ((axs[0], b640, L("640 次挑选（次要预算）", "640 picks (secondary budget)"), 290),
                         (axs[1], b160, L("160 次挑选（事先登记的主要预算）", "160 picks (pre-registered primary)"), 92)):
        yy = np.arange(len(names))[::-1]
        for off, mp, col in ((0.18, "K562", BLUE), (-0.18, "Jurkat", PURPLE)):
            ax.barh(yy + off, d[mp], height=0.34, color=col, edgecolor=S.SURFACE, zorder=3)
            for yi, v in zip(yy, d[mp]):
                ax.text(v + xm * 0.012, yi + off, f"{v:g}", fontsize=7, va="center")
        ax.set_yticks(yy, names, fontsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, xm)
        ax.set_xlabel(L("累计命中数", "cumulative hits"))
        title(ax, t)
    axs[1].set_yticklabels([])
    fig.legend(handles=[Patch(color=BLUE, label=L("K562（诊断用的图谱）", "K562 (the map the repair was built on)")),
                        Patch(color=PURPLE, label=L("Jurkat（留出）", "Jurkat (held out)"))],
               loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=7.8, labelcolor=S.INK_2)
    save(fig, "jurkat")


def fig_benchmark_config():
    """Paper appendix: the three ML baselines in the original benchmark configuration (nine features with the
    anchor-relative ones, no DepMap) against the versions in C's own features, per screen (repo
    matched_baselines.json)."""
    mb = load("matched_baselines.json")["headline"]
    rep, mat = mb["reported"], mb["matched"]
    arms = [("Static prior", "static_ranker", "static_prior"), ("Online ML", "online_adaptive", "online_ml"),
            ("Coreset", "coreset", "coreset")]
    labels = SCREEN_LABEL + ["Mean"]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.0), sharey=True, gridspec_kw=dict(wspace=0.12))
    yy = np.arange(len(labels))[::-1]
    yy = np.where(yy == 0, -0.4, yy)               # a little air above the mean row
    for ax, (name, orig_key, mat_key) in zip(axs, arms):
        o = [rep[s][orig_key][-1] for s in SCREENS]
        m = [mat[s][mat_key][-1] for s in SCREENS]
        o.append(float(np.mean(o)))
        m.append(float(np.mean(m)))
        for yi, a, b in zip(yy, o, m):
            ax.plot([a, b], [yi, yi], color=DEEMPH, lw=2.5, zorder=2)
            ax.plot(a, yi, "o", ms=6.5, color=S.MUTED, zorder=3)
            ax.plot(b, yi, "o", ms=6.5, color=BLUE, zorder=3, mec=S.SURFACE, mew=0.8)
        ax.axhline(0.3, color=S.GRID, lw=0.8)
        ax.set_xlim(0, 0.66)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("hit@R5")
        title(ax, f"{name}\n{o[-1]:.3f} → {m[-1]:.3f}")
    axs[0].set_yticks(yy, labels)
    axs[0].get_yticklabels()[-1].set_fontweight("bold")
    legend(axs[2], [dot_handle(S.MUTED, "original configuration"), dot_handle(BLUE, "C's own features (matched)")],
           loc="upper center", bbox_to_anchor=(-0.6, -0.2), ncol=2)
    save(fig, "benchmark_config")


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════
# Explainer: the same nine charts for each of BDA's three tiers (headline, full pool, two-gene), in the same order,
# so a reader can lay the tiers side by side. Single-gene tiers are hit@R5, paired over the nine screens; the
# two-gene tier is cumulative hits at 640 picks on Horlbeck K562 (Welch over replicates), with the held-out Jurkat
# map in the place of a second screen.
TIERS = ("headline", "full_pool", "pairs")
TIER_ZH = {"headline": "主档", "full_pool": "全池档", "pairs": "双扰动档"}
GPT = RES / "gpt"
PAIR_RUNS = {   # map -> (dataset, matched-baseline key, {method: (run file under gpt/, arm)})
    "K562": ("Horlbeck", "pairs", {"Random": ("tier_b_pairs_20rep/calibration_20rep.json", "random"),
                                   "B": ("tier_b_pairs_20rep/calibration_20rep.json", "llm_reasoning"),
                                   "C": ("tier_b_pairs_20rep/waddington_c_20rep.json", "waddington_c"),
                                   "BDA": ("tier_b_pairs_20rep/bda_controlled_real.json", "bda")}),
    "Jurkat": ("Horlbeck_Jurkat", "jurkat_b128", {"Random": ("jurkat_b128_20rep/random_20rep.json", "random"),
                                                  "B": ("jurkat_b128_20rep/llm_reasoning_20rep.json", "llm_reasoning"),
                                                  "C": ("jurkat_b128_20rep/waddington_c_20rep.json", "waddington_c"),
                                                  "BDA": ("jurkat_b128_20rep/bda_20rep.json", "bda")}),
    # K562 at BDA's own protocol (32 pairs per round, 160 picks; paper Table pairs-bda)
    "K562_b32": ("Horlbeck", "pairs_b32", {"Random": ("tier_b_pairs_b32_20rep/calibration_20rep.json", "random"),
                                           "B": ("tier_b_pairs_b32_20rep/calibration_20rep.json", "llm_reasoning"),
                                           "C": ("tier_b_pairs_b32_20rep/waddington_c_20rep.json", "waddington_c"),
                                           "BDA": ("tier_b_pairs_b32_20rep/bda_controlled_real.json", "bda")}),
}
GD_PAIR_PROBE = 77.7     # the best of GeneDisco's acquisition rules run on pairs as a mechanism probe (paper text)


def _tcolor(name):
    return {"Waddington C": C["waddington_c"], "Online ML": C["online_adaptive"], name_static(): C["static_ranker"],
            name_b(): C["llm_reasoning"], "GeneDisco": C["genedisco"], "Coreset": C["coreset"], "BDA": C["bda"],
            "Random": C["random"]}[name]


def _gd_curves(folder: Path, arm: str = "gdfb_achilles_bnn_coreset") -> dict:
    """GeneDisco's best equal-information rule: per screen, mean hit ratio per round over its seeds."""
    out = {}
    for p in sorted(folder.glob("gdfb_seeded_*.json")):
        if p.name.endswith(".manifest.json"):
            continue
        for ds, arms in json.loads(p.read_text()).items():
            if isinstance(arms, dict) and arm in arms:
                out[ds] = np.mean([r["hit_ratio_per_round"] for r in arms[arm]], axis=0).tolist()
    return out


def _single_tier(tier: str) -> dict:
    """Single-gene tier: method -> screen -> hit ratio per round (C, B, Random: replicate means)."""
    mb = load("matched_baselines.json")[tier]
    rep, mat = mb["reported"], mb["matched"]
    m = {"Waddington C": {s: rep[s]["waddington_c"] for s in SCREENS},
         "Online ML": {s: mat[s]["online_ml"] for s in SCREENS},
         name_static(): {s: mat[s]["static_prior"] for s in SCREENS},
         name_b(): {s: rep[s]["llm_reasoning"] for s in SCREENS},
         "Coreset": {s: mat[s]["coreset"] for s in SCREENS},
         "Random": {s: rep[s]["random"] for s in SCREENS}}
    if tier == "headline":
        bda = load("bda_controlled/bda_gpt.json")["results"]
        m["BDA"] = {s: bda[s]["hit_ratio_per_round"] for s in SCREENS}
        m["GeneDisco"] = _gd_curves(RES / "ml_engines")
    else:
        bda = json.loads((GPT / "tier_a_full_pool_20rep" / "bda_controlled.json").read_text())
        m["BDA"] = {s: bda[s]["bda"][0]["hit_ratio_per_round"] for s in SCREENS}
        m["GeneDisco"] = _gd_curves(RES / "ml_engines" / "tier_a")
    return m


def _pair_map(mp: str) -> dict:
    """Two-gene tier, one map: method -> array (replicates x rounds) of cumulative hits; the matched ML baselines
    are deterministic (one row; the runner stores 4-decimal ratios, so hits are the nearest integer)."""
    ds, key, runs = PAIR_RUNS[mp]
    names = {"Random": "Random", "B": name_b(), "C": "Waddington C", "BDA": "BDA"}
    out = {}
    for short, (path, arm) in runs.items():
        reps = json.loads((GPT / path).read_text())[ds][arm]
        out[names[short]] = np.array([r["cumulative_hits"] for r in reps], float)
    mat = load("matched_baselines.json")[key]["matched"][ds]
    tot = mat["total_hits"]
    for name, k in (("Online ML", "online_ml"), (name_static(), "static_prior"), ("Coreset", "coreset")):
        out[name] = np.array([[round(v * tot) for v in mat[k]]], float)
    return out


def _tier_means(tier: str) -> dict:
    if tier == "pairs":
        out = {n: float(a[:, -1].mean()) for n, a in _pair_map("K562").items()}
        paper(GD_PAIR_PROBE, nd=1)
        out["GeneDisco"] = GD_PAIR_PROBE
        return out
    return {n: float(np.mean([v[s][-1] for s in SCREENS])) for n, v in _single_tier(tier).items()}


def _unit(tier, short=False):
    if tier == "pairs":
        return "累计命中数" if short else "640 次挑选的累计命中数（Horlbeck K562）"
    return "hit@R5" if short else "9 屏平均 hit@R5"


def fig_tier_overall(tier: str):
    """① every method, in the order of the information it uses; the padded LLM baseline B is set apart."""
    m = _tier_means(tier)
    gd_label = "GeneDisco 规则（机制探针）\n前人：只在本屏内主动学习" if tier == "pairs" else "GeneDisco\n前人：只在本屏内主动学习"
    rows = [("Random\n什么都不用", "Random"), ("Coreset\n不用标签", "Coreset"), ("BDA\n前人：LLM agent，只用本屏", "BDA"),
            (gd_label, "GeneDisco"), (f"{name_static()}\n+ 其它屏的标签", name_static()),
            ("Online ML\n+ 本屏在线重训", "Online ML"), ("Waddington C\n+ LLM 点名（0.2 权重）", "Waddington C"),
            ("", None), (f"{name_b()}\n对照：LLM 主导 + 静态排序补位", name_b())]
    fmt = "{:.1f}" if tier == "pairs" else "{:.3f}"
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = hbars(ax, [r[0] for r in rows], [m[r[1]] if r[1] else None for r in rows],
              [_tcolor(r[1]) if r[1] else S.SURFACE for r in rows], fmt=fmt, xmax=205 if tier == "pairs" else 0.33)
    ax.tick_params(axis="y", labelsize=7.8)
    prev = max(m["BDA"], m["GeneDisco"])
    step = m[name_static()] - prev
    ax.text(m[name_static()] + (40 if tier == "pairs" else 0.045), y[4],
            f"跨实验先验这一步：{fmt.format(step) if step < 0 else '+' + fmt.format(step)}\n（相对前人最好的方法）",
            fontsize=7.6, color=S.INK, va="center")
    ax.set_xlabel(_unit(tier))
    title(ax, f"{TIER_ZH[tier]}：各方法的总体成绩", "按所用信息由少到多排列；灰色 = 前人方法")
    save(fig, f"tier_overall_{tier}")


def fig_tier_screens(tier: str):
    """② per screen (single-gene tiers) or per map (two-gene tier: K562 and the held-out Jurkat map)."""
    if tier == "pairs":
        k, j = _pair_map("K562"), _pair_map("Jurkat")
        order = ["Waddington C", "Online ML", name_static(), name_b(), "Coreset", "BDA", "Random"]
        mat = [[k[n][:, -1].mean(), j[n][:, -1].mean()] for n in order]
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        heat(ax, mat, order, col_labels=["K562", "Jurkat（留出）"], avg=False, fmt="{:.1f}")
        title(ax, f"{TIER_ZH[tier]}：逐图谱命中数（640 次挑选）",
              "颜色 = 该图谱内相对最高分的比例；加粗加框 = 该图谱最高")
    else:
        m = _single_tier(tier)
        order = ["Waddington C", "Online ML", name_static(), name_b(), "GeneDisco", "Coreset", "BDA", "Random"]
        fig, ax = plt.subplots(figsize=(6.6, 3.35))
        heat(ax, [[m[n][s][-1] for s in SCREENS] for n in order], order)
        title(ax, f"{TIER_ZH[tier]}：逐屏 hit@R5", "颜色 = 该屏内相对最高分的比例；加粗加框 = 该屏最高")
    save(fig, f"tier_screens_{tier}")


def fig_tier_curves(tier: str):
    """③ cumulative discovery per round."""
    if tier == "pairs":
        curves = {n: a.mean(axis=0) for n, a in _pair_map("K562").items()}
    else:
        curves = {n: np.mean([v[s] for s in SCREENS], axis=0) for n, v in _single_tier(tier).items()}
    order = ["Waddington C", name_b(), "Online ML", name_static(), "Coreset", "Random", "GeneDisco", "BDA"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for n in order:
        if n not in curves:
            continue
        ext = n in ("GeneDisco", "BDA")
        hero = n == "Waddington C"
        cv = curves[n]
        ax.plot(range(1, len(cv) + 1), cv, color=_tcolor(n), lw=2.4 if hero else 1.7, ls="--" if ext else "-",
                marker="s" if ext else "o", ms=4.5 if hero else 3.5, mec=S.SURFACE, mew=0.6, label=n,
                zorder=4 if hero else (2 if ext else 3))
        if n in ("Waddington C", "Random", "BDA"):
            ax.annotate(("{:.1f}" if tier == "pairs" else "{:.3f}").format(cv[-1]), (len(cv), cv[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.8, color=S.INK_2)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.85, 5.6)
    ax.set_ylim(0, None)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("轮次")
    ax.set_ylabel("累计命中数" if tier == "pairs" else "累计 hit 率（9 屏平均）")
    ax.legend(fontsize=7.4, labelcolor=S.INK_2, loc="upper left", ncol=2)
    title(ax, f"{TIER_ZH[tier]}：逐轮累计发现", "虚线 = 前人方法" + ("（双扰动档没有 GeneDisco 的组合实验）" if tier == "pairs" else ""))
    save(fig, f"tier_curves_{tier}")


def _welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    if len(b) == 1 or b.std(ddof=1) == 0:        # a deterministic comparator: one-sample interval on a
        se, dof = a.std(ddof=1) / np.sqrt(len(a)), len(a) - 1
        p = stats.ttest_1samp(a, b.mean()).pvalue
    else:
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    t = stats.t.ppf(0.975, dof)
    return d, d - t * se, d + t * se, p


def _welch_stats(a: dict, b: dict):
    """a minus b from {n, mean, sd}: Welch t interval and p; a constant b gives the one-sample interval of a."""
    from scipy import stats
    d = a["mean"] - b["mean"]
    va = a["sd"] ** 2 / a["n"] if a["n"] > 1 else 0.0
    vb = b["sd"] ** 2 / b["n"] if b["n"] > 1 else 0.0
    se = np.sqrt(va + vb)
    if se == 0:
        return d, None, None, None
    parts = [(v, n) for v, n in ((va, a["n"]), (vb, b["n"])) if v > 0]
    dof = (va + vb) ** 2 / sum(v ** 2 / (n - 1) for v, n in parts)
    t = stats.t.ppf(0.975, dof)
    return d, d - t * se, d + t * se, float(2 * stats.t.sf(abs(d) / se, dof))


def _pfmt(p):
    return "p < 1e-6" if p < 1e-6 else f"p = {p:.2g}"


def fig_tier_paired(tier: str):
    """④ C minus each baseline: paired over screens (single-gene) or Welch / one-sample over replicates (pairs)."""
    names = [name_static(), "Online ML", name_b(), "Coreset", "BDA"]
    rows = []
    if tier == "pairs":
        k = _pair_map("K562")
        c = k["Waddington C"][:, -1]
        for n in names:
            d, lo, hi, p = _welch(c, k[n][:, -1])
            rows.append((f"{n}（{k[n][:, -1].mean():.1f}）", d, lo, hi, _pfmt(p)))
        fmt, xl = "{:+.1f}", "C − 基线（命中数），95% CI：Welch（B、BDA）或单样本（确定性基线）"
    else:
        from .final_system_stats import _paired
        m = _single_tier(tier)
        cfin = {s: m["Waddington C"][s][-1] for s in SCREENS}
        if tier == "headline":      # the paper's paired block (Table 5), as printed there
            given = {name_static(): (0.030, 0.016, 0.044, "8/0/1"), "Online ML": (0.015, 0.004, 0.026, "7/0/2"),
                     name_b(): (0.049, 0.019, 0.080, "8/0/1"), "Coreset": (0.107, 0.060, 0.155, "9/0/0")}
            for v in given.values():
                paper(*v[:3])
        else:
            pc = load("matched_baselines.json")["full_pool"]["paired_C_vs"]
            src = {name_static(): "matched_static_prior", "Online ML": "matched_online_ml",
                   name_b(): "benchmark_llm_reasoning", "Coreset": "matched_coreset"}
            given = {n: (pc[k]["delta"], *pc[k]["ci95"], f"{pc[k]['wins']}/{pc[k]['ties']}/{pc[k]['losses']}")
                     for n, k in src.items()}
        bda = _paired(cfin, {s: m["BDA"][s][-1] for s in SCREENS})
        given["BDA"] = (bda["delta"], *bda["ci95"], f"{bda['wins']}/{bda['ties']}/{bda['losses']}")
        means = {n: np.mean([m[n][s][-1] for s in SCREENS]) for n in names}
        rows = [(f"{n}（{means[n]:.3f}）", *given[n]) for n in names]
        fmt, xl = "{:+.3f}", "C − 基线（hit@R5），95% CI 按屏聚类"
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    filled = [lo > 0 or hi < 0 for _, _, lo, hi, _ in rows]
    cols = [BLUE if d > 0 else RED for _, d, *_ in rows]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, lo, hi, note), col, f in zip(y, rows, cols, filled):
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(d, yi, "o", ms=7, mfc=col if f else S.SURFACE, mec=col, mew=1.6)
        ax.text(1.02, yi, f"{fmt.format(d)}  [{fmt.format(lo)}, {fmt.format(hi)}]  {note}",
                transform=ax.get_yaxis_transform(), fontsize=7.8, va="center")
    ax.axvline(0, color=S.INK_2, lw=0.9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xl, fontsize=8)
    ax.text(1.02, 1.04, "差值  [95% CI]  " + ("p" if tier == "pairs" else "胜/平/负"), transform=ax.transAxes,
            fontsize=7.6, color=S.INK_2)
    title(ax, f"{TIER_ZH[tier]}：C 相对各基线的差")
    save(fig, f"tier_paired_{tier}")


ABL = [("waddington_c_nomem", "去掉跨实验记忆"), ("waddington_c_static_ml", "去掉在线重训"),
       ("waddington_c_anon_names", "匿名基因名"), ("no_llm", "去掉 LLM")]


def fig_tier_ablation(tier: str):
    """⑤ component ablation of the reported system, paired with a re-run of the full system (paper Table 19)."""
    a = load("tier_extension_summary.json")["ablations"][tier]
    ref = a["waddington_c"]["mean"]
    rows = []
    for key, lab in ABL:
        if tier == "pairs":
            d, lo, hi, pv = _welch_stats(a[key], a["waddington_c"])
            rows.append((lab, d, lo, hi, _pfmt(pv)))
        else:
            v = a[key]["vs_c"]
            rows.append((lab, v["delta"], *v["ci95"], f"{v['wins']}/{v['ties']}/{v['losses']}"))
    fmt = "{:+.1f}" if tier == "pairs" else "{:+.3f}"
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, lo, hi, note) in zip(y, rows):
        sig = hi < 0 or lo > 0
        col = RED if d < 0 else BLUE
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(d, yi, "o", ms=7, mfc=col if sig else S.SURFACE, mec=col, mew=1.6)
        ax.text(1.02, yi, f"{fmt.format(d)}  [{fmt.format(lo)}, {fmt.format(hi)}]  {note}",
                transform=ax.get_yaxis_transform(), fontsize=7.8, va="center")
    ax.axvline(0, color=S.INK_2, lw=0.9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"变体 − 完整系统（{_unit(tier, True)}），95% CI；空心 = 区间含 0", fontsize=8)
    ax.text(1.02, 1.04, "差值  [95% CI]  " + ("p" if tier == "pairs" else "胜/平/负"), transform=ax.transAxes,
            fontsize=7.6, color=S.INK_2)
    title(ax, f"{TIER_ZH[tier]}：组件消融（完整系统同批重跑 {('{:.1f}' if tier == 'pairs' else '{:.3f}').format(ref)}）")
    save(fig, f"tier_ablation_{tier}")


def fig_tier_fusion(tier: str):
    """⑥ the full GPT system at six fusion weights; the reported 0.2 is the twenty-replicate run (paper Tables 18
    and 20)."""
    if tier == "headline":      # paper Table 18
        pts = [(0.0, 0.245, -0.015, -0.026, -0.004), (0.1, 0.256, -0.005, -0.010, 0.001), (0.2, 0.260, 0, 0, 0),
               (0.3, 0.259, -0.002, -0.008, 0.005), (0.4, 0.247, -0.014, -0.024, -0.004),
               (0.5, 0.249, -0.011, -0.025, 0.003)]
        for p in pts:
            paper(*p[1:])
    else:
        f = load("tier_extension_summary.json")["fusion"][tier]
        pts = []
        for w in ("0.0", "0.1", "0.2", "0.3", "0.4", "0.5"):
            v = f[w].get("vs_0.2")
            if not v:
                pts.append((float(w), f[w]["mean"], 0, 0, 0))
            elif tier == "pairs":
                d, lo, hi, _ = _welch_stats(f[w], f["0.2"])
                pts.append((float(w), f[w]["mean"], d, lo, hi))
            else:
                pts.append((float(w), f[w]["mean"], v["delta"], *v["ci95"]))
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    ws = [p[0] for p in pts]
    ax.plot(ws, [p[1] for p in pts], color=DEEMPH, lw=2, zorder=2)
    for w, lvl, d, lo, hi in pts:
        if w == 0.2:
            col, fill, mk = BLUE, True, "*"
        elif hi < 0:
            col, fill, mk = RED, True, "o"
        elif lo > 0:
            col, fill, mk = BLUE, True, "o"
        else:
            col, fill, mk = S.MUTED, False, "o"
        ax.plot(w, lvl, mk, ms=12 if mk == "*" else 7, mfc=col if fill else S.SURFACE, mec=col, mew=1.5, zorder=3)
        ax.annotate(("{:.1f}" if tier == "pairs" else "{:.3f}").format(lvl), (w, lvl), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=7.6, color=S.INK)
    ax.set_xticks(ws, ["0\n（只用 ML）", "0.1", "0.2\n（报告）", "0.3", "0.4", "0.5"], fontsize=8)
    lv = [p[1] for p in pts]
    pad = (max(lv) - min(lv)) * 0.35
    ax.set_ylim(min(lv) - pad, max(lv) + pad)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(_unit(tier, True))
    legend(ax, [Line2D([], [], marker="*", ls="", ms=10, color=BLUE, label="报告的 0.2"),
                dot_handle(RED, "显著差于 0.2"), dot_handle(BLUE, "显著好于 0.2"),
                dot_handle(S.MUTED, "与 0.2 无显著差异", filled=False)],
           loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=7.4)
    title(ax, f"{TIER_ZH[tier]}：融合权重 w_LLM")
    save(fig, f"tier_fusion_{tier}")


def fig_tier_attribution(tier: str):
    """⑦ the hit rate of the picked genes by source, and the endorsement odds ratio adjusted for the ML score and the
    screen (paper Table 9 and the tier follow-up text)."""
    if tier == "headline":      # paper Table 9 and the conditional attribution
        src = {"both": (42.1, 1900), "ml_only": (18.3, 77020), "llm_only": (21.0, 17080)}
        orv = (2.51, 1.82, 3.43)
        paper(42.1, 18.3, 21.0, nd=1)
        paper(*orv, nd=2)
    else:
        a = load("tier_extension_summary.json")["attribution"][tier]
        src = {k: (100 * v["hit_rate"], v["picked"]) for k, v in a["by_source"].items()}
        c = a["conditional"]
        c = c.get("logit", c)
        orv = (c["endorsed_odds_ratio"], *c["or_ci95"])
    rows = [("ML 与 LLM 都选", "both"), ("只有 ML 选", "ml_only"), ("只有 LLM 选", "llm_only")]
    fig, ax = plt.subplots(figsize=(4.4, 1.8))
    hbars(ax, [f"{lab}\n（{src[k][1]:,} 个）" for lab, k in rows], [src[k][0] for _, k in rows],
          [BLUE, DEEMPH, C["llm_reasoning"]], fmt="{:.1f}%", xmax=60)
    ax.set_xlabel("被选基因的 hit 率（%）")
    title(ax, f"{TIER_ZH[tier]}：LLM 的作用（按来源的命中率）",
          f"调整 ML 分数与屏后，LLM 点名的 OR = {orv[0]:.2f} [{orv[1]:.2f}, {orv[2]:.2f}]")
    save(fig, f"tier_attribution_{tier}")


PAIR_ENGINES = [("rf", "随机森林"), ("logreg", "逻辑回归"), ("svm_linear", "线性 SVM"), ("svgp", "稀疏变分高斯过程"),
                ("bnn", "贝叶斯 MLP（GeneDisco 的模型）"), ("mlp", "MLP（2×64）"), ("lambdarank", "LambdaMART（排序学习）")]


def fig_tier_engines(tier: str):
    """⑧ the ML engine swapped, ML only: online engine minus online LightGBM."""
    if tier == "pairs":
        pe = load("tier_extension_summary.json")["pair_engines"]["pairs"]
        lg = pe["ml_engine_lightgbm"]["mean"]
        data = []
        for key, lab in PAIR_ENGINES:
            v = pe[f"ml_engine_{key}"]
            d, lo, hi, pv = _welch_stats(v, pe["ml_engine_lightgbm"])
            data.append((lab, d, lo, hi, v["mean"], pv))
        data.sort(key=lambda r: r[1])
        labels = [r[0] for r in data]
        heads = []
        ypos = list(range(len(data)))
        figsize = (4.6, 2.4)
    else:
        rows = load("ml_engines/summary.json" if tier == "headline" else "ml_engines/tier_a/summary.json")["rows"]
        lg = rows["ml_engine_lightgbm"]["mean"]
        data, labels, ypos, heads = [], [], [], []
        yv = 0
        for gname, items in reversed([("第一轮：换模型族", ENGINE_ROUND1), ("第二轮：换学习目标、加描述、换代理模型", ENGINE_ROUND2)]):
            items = sorted(items, key=lambda kv: rows[f"ml_engine_{kv[0]}"]["vs_lightgbm_online"]["delta"])
            for key, zh, _ in items:
                r = rows[f"ml_engine_{key}"]
                vs = r["vs_lightgbm_online"]
                data.append((zh, vs["delta"], *vs["ci95"], r["mean"], None))
                labels.append(zh)
                ypos.append(yv)
                yv += 1
            labels.append(gname)
            heads.append(gname)
            ypos.append(yv)
            yv += 1.3
        figsize = (4.6, 6.0)
    fig, ax = plt.subplots(figsize=figsize)
    dy = [p for p, lab in zip(ypos, labels) if lab not in heads]
    for yi, (lab, d, lo, hi, lvl, p) in zip(dy, data):
        sig = lo is not None and (lo > 0 or hi < 0)
        col = BLUE if d >= 0 else RED
        if lo is not None:
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.8, solid_capstyle="round")
        ax.plot(d, yi, "o" if lo is not None else "D", ms=6, mfc=col if sig or lo is None else S.SURFACE,
                mec=col, mew=1.4)
        tail = ("" if p is None else f"  {_pfmt(p)}") if tier == "pairs" else ""
        ax.text(1.02, yi, (f"{lvl:.1f}" if tier == "pairs" else f"{lvl:.3f}") + tail,
                transform=ax.get_yaxis_transform(), fontsize=7.6, va="center")
    ax.axvline(0, color=S.INK_2, lw=0.9)
    ax.set_yticks(ypos, labels, fontsize=8)
    for t, lab in zip(ax.get_yticklabels(), labels):
        if lab in heads:
            t.set_fontweight("bold")
            t.set_color(S.INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"在线引擎 − 在线 LightGBM（{_unit(tier, True)}）", fontsize=8)
    ax.text(1.02, 1.01, "该引擎的成绩" + ("  p" if tier == "pairs" else ""), transform=ax.transAxes, fontsize=7.6,
            color=S.INK_2)
    sub = (f"LightGBM = {lg:.1f}；菱形 = 确定性引擎（无区间）；实心 = 区间不含 0" if tier == "pairs"
           else f"LightGBM = {lg:.3f}；实心 = 区间不含 0；† = 事后加的变体")
    title(ax, f"{TIER_ZH[tier]}：只换 ML 引擎（只跑 ML）", sub)
    save(fig, f"tier_engines_{tier}")


def fig_tier_acquisition(tier: str):
    """⑨ acquisition rules on a five-member LightGBM ensemble and on GeneDisco's Bayesian MLP."""
    if tier == "headline":
        s = load("ml_engines/summary.json")
        single = s["rows"]["ml_engine_lightgbm"]["mean"]
        acq = s["acquisition"]
        lg = [("贪心（现行规则）", acq["ml_acq_greedy"]["mean"], None)] + \
             [(lab, acq[k]["mean"], acq[k]["vs_ensemble_greedy"]["ci95"]) for k, lab in
              (("ml_acq_ucb05", "UCB，β = 0.5"), ("ml_acq_thompson", "Thompson 采样"), ("ml_acq_ucb1", "UCB，β = 1"),
               ("ml_acq_margin", "margin（不确定性采样）"))]
        bn = s["bnn_acquisition"]
        bb = [("贪心", s["rows"]["ml_engine_bnn"]["mean"], None)] + \
             [(lab, bn[k]["mean"], bn[k]["vs_bnn_greedy"]["ci95"]) for k, lab in
              (("ml_bnnacq_ucb1", "UCB"), ("ml_bnnacq_bald", "BALD"), ("ml_bnnacq_kmeans_embedding", "k-means（嵌入）"),
               ("ml_bnnacq_badge", "BADGE"))]
    else:
        t = load("tier_extension_summary.json")
        acq, rules = t["acquisition"][tier], t["rules_on_engines"][tier]
        single = (load("ml_engines/tier_a/summary.json")["rows"]["ml_engine_lightgbm"]["mean"] if tier == "full_pool"
                  else t["pair_engines"]["pairs"]["ml_engine_lightgbm"]["mean"])
        if tier == "pairs":
            bnn_ref = t["pair_engines"]["pairs"]["ml_engine_bnn"]

            def ci(x, ref):
                return _welch_stats(x, ref)[1:3]
        else:
            def ci(x, ref):
                v = x.get("vs_greedy") or x.get("vs_reference")
                return v["ci95"]
            bnn_ref = None
        lg = [("贪心（现行规则）", acq["ml_acq_greedy"]["mean"], None)] + \
             [(lab, acq[k]["mean"], ci(acq[k], acq["ml_acq_greedy"])) for k, lab in
              (("ml_acq_ucb05", "UCB，β = 0.5"), ("ml_acq_thompson", "Thompson 采样"), ("ml_acq_ucb1", "UCB，β = 1"),
               ("ml_acq_margin", "margin（不确定性采样）"))]
        bb = [("贪心", rules["ml_bnnacq_bald"]["reference_mean"], None)] + \
             [(lab, rules[k]["mean"], ci(rules[k], bnn_ref)) for k, lab in
              (("ml_bnnacq_ucb1", "UCB"), ("ml_bnnacq_bald", "BALD"), ("ml_bnnacq_kmeans_embedding", "k-means（嵌入）"),
               ("ml_bnnacq_badge", "BADGE"))]

    def col(ci):
        if ci is None:
            return BLUE
        return RED if ci[1] < 0 else (LIGHT_BLUE if ci[0] > 0 else DEEMPH)

    fmt = "{:.1f}" if tier == "pairs" else "{:.3f}"
    xmax = 150 if tier == "pairs" else 0.31
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw=dict(wspace=0.85))
    for ax, rows, t_ in ((axs[0], lg, "LightGBM（5 成员集成）"), (axs[1], bb, "贝叶斯 MLP（GeneDisco）")):
        hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [col(r[2]) for r in rows], fmt=fmt,
              ref=single, ref_label=f"单个 LightGBM {fmt.format(single)}", xmax=xmax)
        ax.set_xlabel(f"{_unit(tier, True)}（5 个 seed）")
        title(ax, t_)
    fig.suptitle(f"{TIER_ZH[tier]}：采集规则", x=0.02, ha="left", fontsize=10, fontweight="bold", color=S.INK, y=1.08)
    fig.legend(handles=[Patch(color=BLUE, label="贪心（按命中概率取前 B）"), Patch(color=DEEMPH, label="与贪心无显著差异"),
                        Patch(color=RED, label="显著差于贪心"), Patch(color=LIGHT_BLUE, label="显著好于贪心")],
               loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=7.6, labelcolor=S.INK_2)
    save(fig, f"tier_acquisition_{tier}")


def fig_pairs_repair():
    """Two-gene tier only: the reported system, the random-forest variant, BDA and the repaired arm (640 picks)."""
    rows = [("Waddington C（报告配置）", 112.6, 24.7, C["waddington_c"]),
            ("C + 随机森林引擎（仅 K562）", 157.5, 25.5, LIGHT_BLUE), ("BDA（GPT-5.6 Luna）", 160.0, 15.7, C["bda"]),
            ("交互先验单独（不用 LLM、不学习）", 196, None, REPAIR), ("修复：不用 LLM", 239, None, REPAIR),
            ("修复：LLM 名字优先", 220.9, 10.9, REPAIR), ("修复：LLM 作 0.2 验证者", 232.6, 6.0, REPAIR)]
    paper(*[r[1] for r in rows], nd=1)
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[3] for r in rows], err=[r[2] for r in rows],
          fmt="{:g}", ref=60, ref_label="随机 60", xmax=290)
    ax.set_xlabel("640 次挑选的累计命中数（误差线 = 20 次重复的标准差）")
    title(ax, "双扰动档特有：诊断后的修复臂（K562）", "修复臂是单独的实验，报告配置不变")
    save(fig, "pairs_repair")


def fig_pair_engines_interaction():
    """Two-gene tier only: the engines on the table with the eight interaction features (paper Table 21, right)."""
    names = ["LightGBM（报告）", "随机森林", "逻辑回归", "线性 SVM", "稀疏高斯过程", "贝叶斯 MLP", "MLP", "LambdaMART"]
    tab = [(158, None), (154.4, 17.5), (97, None), (98, None), (71.0, 9.0), (100.2, 57.7), (125.6, 40.7), (70, None)]
    paper(*[v for v, _ in tab], *[e for _, e in tab if e], nd=1)
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    hbars(ax, names, [v for v, _ in tab], [BLUE] + [DEEMPH] * 7, err=[e for _, e in tab], fmt="{:g}", ref=60,
          ref_label="随机 60", xmax=235)
    ax.set_xlabel("640 次挑选的命中数（只跑 ML）")
    title(ax, "双扰动档特有：加入 8 个交互特征后的引擎", "这时 LightGBM 最好（158）")
    save(fig, "pair_engines_interaction")


# BDA's published two-gene results, Roohani et al. 2025 Table 2 (10 runs, 32 pairs per round): cumulative hits per
# round, mean and 1 SD. Only the endpoints and the Random row are quoted in our paper, so only those are checked.
BDA_TABLE2 = {"Random": ([2.6, 5.7, 8.9, 12.8, 16.4], [1.43, 2.83, 3.67, 3.74, 3.8]),
              "Claude 3 Haiku": ([5.1, 14.8, 23.6, 34.1, 45.1], [2.2, 2.9, 4.1, 5.8, 5.3]),
              "Claude 3.5 Sonnet": ([8.4, 15.7, 23.1, 31.7, 40.0], [1.36, 1.1, 2.59, 2.9, 3.77])}


def fig_bda_protocol():
    """Two-gene tier only: BDA at its own protocol (32 pairs per round, 160 picks). Left: our runs against its
    published Table 2, round by round; right: every method at that budget (paper Table pairs-bda)."""
    paper(*BDA_TABLE2["Random"][0], 45.1, 40.0, nd=1)
    k = _pair_map("K562_b32")
    s = load("pair_tier_summary.json")
    run, prior = s["runs"]["160"], s["prior_alone"]["8_columns"]["160"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 1.05})
    x = np.arange(1, 6)
    lines = [("Random（本文）", k["Random"].mean(0), C["random"], "-", "o", True),
             ("Random（BDA 论文）", BDA_TABLE2["Random"][0], C["random"], ":", "o", False),
             ("BDA·GPT-5.6（本文）", k["BDA"].mean(0), C["bda"], "--", "s", True),
             ("BDA·Claude 3 Haiku（论文）", BDA_TABLE2["Claude 3 Haiku"][0], C["bda"], ":", "^", False),
             ("BDA·Claude 3.5 Sonnet（论文）", BDA_TABLE2["Claude 3.5 Sonnet"][0], C["bda"], ":", "D", False)]
    for lab, y, col, ls, mk, filled in lines:
        a.plot(x, y, color=col, ls=ls, lw=2.0 if filled else 1.4, marker=mk, ms=4.5, mfc=col if filled else S.SURFACE,
               mec=col, mew=1.2, label=lab, zorder=3 if filled else 2)
    for v, dy in ((k["BDA"].mean(0)[-1], 0), (45.1, 2), (40.0, -2), (k["Random"].mean(0)[-1], 0)):
        a.annotate(f"{v:.1f}", (5, v), xytext=(7, dy), textcoords="offset points", va="center", fontsize=7.8,
                   color=S.INK_2)
    a.set_xticks(x)
    a.set_xlim(0.8, 5.7)
    a.set_ylim(0, 90)                                # headroom so the legend sits above the curves
    a.grid(axis="x", visible=False)
    a.set_xlabel("轮次（每轮 32 对）")
    a.set_ylabel("累计命中数")
    a.legend(fontsize=7.2, labelcolor=S.INK_2, loc="upper left")
    title(a, "与 BDA 论文表 2 逐轮对照", "实心 = 本文 20 次；空心点线 = 论文 10 次")

    def m(arr):
        return round(float(arr[:, -1].mean()), 1), round(float(arr[:, -1].std(ddof=1)), 1) if len(arr) > 1 else None
    rows = [("Random", *m(k["Random"]), C["random"]), ("Coreset", *m(k["Coreset"]), C["coreset"]),
            (name_static(), *m(k[name_static()]), C["static_ranker"]), ("Online ML", *m(k["Online ML"]), C["online_adaptive"]),
            ("Waddington C（报告配置）", *m(k["Waddington C"]), C["waddington_c"]), (name_b(), *m(k[name_b()]), C["llm_reasoning"]),
            ("", None, None, None),
            ("BDA 论文：Claude 3.5 Sonnet", 40.0, 3.8, DEEMPH), ("BDA 论文：Claude 3 Haiku", 45.1, 5.3, DEEMPH),
            ("BDA 本文：GPT-5.6 Luna", *m(k["BDA"]), C["bda"]),
            ("", None, None, None),
            ("交互先验单独（不用 LLM、不学习）", prior, None, REPAIR),
            ("修复：不用 LLM", int(run["repair_no_llm"]["mean"]), None, REPAIR),
            ("修复：LLM 名字优先", round(run["coldstart_prior_v8"]["mean"], 1), round(run["coldstart_prior_v8"]["sd"], 1), REPAIR),
            ("修复：LLM 作 0.2 验证者", round(run["verifier_prior"]["mean"], 1), round(run["verifier_prior"]["sd"], 1), REPAIR)]
    paper(*[r[1] for r in rows if r[1] is not None], *[r[2] for r in rows if r[2]], nd=1)
    hbars(b, [r[0] for r in rows], [r[1] for r in rows], [r[3] or S.SURFACE for r in rows], err=[r[2] for r in rows],
          fmt="{:g}", xmax=95)
    b.tick_params(axis="y", labelsize=7.8)
    b.set_xlabel("160 次挑选的累计命中数（误差线 = 重复间标准差）")
    title(b, "160 次挑选下的全部方法", "论文 Notes 中的表；浅灰两行转录自 BDA 论文")
    save(fig, "bda_protocol")


LINKS = [("同一蛋白复合体", "pair_complex_shared"), ("STRING 中直接相连", "pair_string_edge"),
         ("共享 KEGG 通路", "pair_kegg_shared"), ("共享 Reactome 通路", "pair_reactome_shared")]


def _bda_pick_mix() -> dict:
    """What BDA proposes on K562: for each kind of functional link between the two genes (the label-free pair
    features of the repaired arm), its share of the library and of BDA's picks (all 20 replicates pooled), and the
    hit rate inside it."""
    import pandas as pd
    ev = REPO / "workspace" / "evaluation"
    lab = pd.read_csv(ev / "lgbm_training_data_horlbeck_pairs_v3.csv", usecols=["gene", "dataset", "label"])
    f = pd.read_csv(ev / "pair_interaction_features.csv").set_index("gene")
    f = f.join(lab[lab["dataset"] == "Horlbeck"].set_index("gene")["label"], how="inner")
    cats = {name: f[col] > 0 for name, col in LINKS}
    cats["以上任一"] = (f[[col for _, col in LINKS]] > 0).any(axis=1)
    cats["都没有"] = ~cats["以上任一"]
    sets = {"Horlbeck 库内全部 100,576 对": f.index}
    for path, name in (("tier_b_pairs_b32_20rep", "BDA 选的对（160 次挑选）"), ("tier_b_pairs_20rep", "BDA 选的对（640 次挑选）")):
        reps = json.loads((GPT / path / "bda_controlled_real.json").read_text())["Horlbeck"]["bda"]
        sets[name] = pd.Index([p for r in reps for rnd in r["selected_genes_per_round"] for p in rnd])
    out = {}
    for name, idx in sets.items():
        y = f["label"].reindex(idx).to_numpy()
        out[name] = {c: (float(m.reindex(idx).mean()), float(y[m.reindex(idx).to_numpy()].mean())) for c, m in cats.items()}
        out[name]["_all"] = float(y.mean())
    return out


def fig_bda_picks():
    """Two-gene tier only: the kinds of pairs BDA proposes, against the library (share, and hit rate inside)."""
    mix = _bda_pick_mix()
    cats = [c for c, _ in LINKS] + ["以上任一", "都没有"]
    groups = list(mix)
    cols = [DEEMPH, C["bda"], S.INK_2]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, gridspec_kw={"wspace": 0.14})
    yc = np.arange(len(cats))[::-1]
    h = 0.26
    for ax, j, ttl, sub in ((axes[0], 0, "这类对的占比", "库内 vs BDA 选的对"),
                            (axes[1], 1, "这类对的 hit 率", "竖线 = 库内总 hit 率")):
        for g, (grp, col) in enumerate(zip(groups, cols)):
            vals = [mix[grp][c][j] for c in cats]
            yy = yc + (1 - g) * h
            ax.barh(yy, vals, height=h * 0.92, color=col, zorder=3, label=grp)
            for yi, v in zip(yy, vals):
                ax.text(v + 0.012, yi, f"{v:.1%}" if v < 0.1 else f"{v:.0%}", va="center", fontsize=6.4, color=S.INK)
        ax.set_xlim(0, 1.0 if j == 0 else 0.62)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        title(ax, ttl, sub)
    axes[1].axvline(mix[groups[0]]["_all"], color=S.INK_2, lw=0.9, zorder=5)
    axes[0].set_yticks(yc, cats)
    for ax in axes:                                  # the four links above the line, their union and its complement below
        ax.axhline(yc[cats.index("以上任一")] + 0.5, color=S.BASELINE, lw=0.8)
    fig.legend(handles=[Patch(color=c, label=g) for g, c in zip(groups, cols)], loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7.6, labelcolor=S.INK_2)
    save(fig, "bda_picks")
    return mix


TIER_FIGS = (fig_tier_overall, fig_tier_screens, fig_tier_curves, fig_tier_paired, fig_tier_ablation, fig_tier_fusion,
             fig_tier_attribution, fig_tier_engines, fig_tier_acquisition)


# BDA and GeneDisco per screen: paper Table 4 (headline) and Table 22 (full pool)
GD_H = [0.104, 0.168, 0.071, 0.088, 0.052, 0.433, 0.043, 0.352, 0.228]
BDA_H = [0.110, 0.133, 0.038, 0.073, 0.050, 0.327, 0.131, 0.254, 0.199]
GD_F = [0.098, 0.174, 0.058, 0.071, 0.061, 0.449, 0.042, 0.352, 0.228]
BDA_F = [0.098, 0.099, 0.039, 0.097, 0.042, 0.358, 0.121, 0.492, 0.147]


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("en", "zh"), default="en")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    _setup(args.lang)
    OUT = args.out or RES / "figures"
    paper(*GD_H, *BDA_H, *GD_F, *BDA_F)
    if args.lang == "zh":
        fig_overview()
        fig_feature_families()
        for tier in TIERS:                 # the same nine charts for each tier
            for f in TIER_FIGS:
                f(tier)
        fig_reasoning()                    # headline only
        fig_clean_features()
        fig_engines_in_system()
        fig_robustness()
        fig_pairs_repair()                 # two-gene tier only
        fig_pair_engines_interaction()
        fig_jurkat()
        fig_bda_protocol()                 # the two-gene tier's state of the art, explained
        fig_bda_picks()
        fig_tiers()                        # across the two single-gene tiers
    else:
        fig_paired()
        fig_robustness()
        fig_reasoning()
        fig_tiers()
        fig_pairs()
        fig_jurkat()
        fig_feature_families()
        fig_benchmark_config()
        fig_engines()
        fig_acquisition()
        fig_engines_in_system()
        fig_ablation()
        fig_pair_engines()
        fig_per_screen("full_pool", "per_screen_full", GD_F, BDA_F, "hit@R5 per screen, full-pool tier")
        fig_clean_features()


if __name__ == "__main__":
    main()
