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


def heat(ax, matrix, row_labels):
    """Per-screen grid: colour = value / best value on that screen (who wins where); the cell shows the value."""
    m = np.asarray(matrix, float)
    m = np.column_stack([m, m.mean(axis=1)])
    rel = m / m.max(axis=0, keepdims=True)
    cmap = S.cmap_sequential()
    nr, nc = m.shape
    for i in range(nr):
        for j in range(nc):
            x = j + (0.35 if j == nc - 1 else 0)
            ax.add_patch(Rectangle((x, nr - 1 - i), 0.94, 0.9, color=cmap(rel[i, j] ** 2), lw=0))
            best = m[i, j] == m[:, j].max()
            ax.text(x + 0.47, nr - 1 - i + 0.45, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=7.4,
                    color="white" if rel[i, j] ** 2 > 0.55 else S.INK, fontweight="bold" if best else "normal")
            if best:
                ax.add_patch(Rectangle((x + 0.03, nr - 1 - i + 0.03), 0.88, 0.84, fill=False, ec=S.INK, lw=1.1))
    ax.set_xlim(0, nc + 0.35)
    ax.set_ylim(0, nr)
    ax.set_xticks([j + 0.47 + (0.35 if j == nc - 1 else 0) for j in range(nc)], SCREEN_LABEL + [L("平均", "Mean")],
                  rotation=35, ha="right", fontsize=8)
    ax.set_yticks([nr - 1 - i + 0.45 for i in range(nr)], row_labels, fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def legend(ax, handles, **kw):
    kw.setdefault("loc", "lower right")
    ax.legend(handles=handles, fontsize=7.8, labelcolor=S.INK_2, handlelength=1.4, **kw)


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


def fig_ladder():
    """Explainer 4.1.2: the information ladder, top to bottom as in the text."""
    mb = load("matched_baselines.json")["headline"]["means"]
    paper(0.146, 0.171)
    rows = [("Random\n什么都不用", mb["benchmark_random"], C["random"]),
            ("Coreset\n不用标签，偏向高连接基因", mb["matched_coreset"], C["coreset"]),
            ("BDA（前人）\n只用本屏信息", 0.146, C["bda"]),
            ("GeneDisco（前人）\n只用本屏信息", 0.171, C["genedisco"]),
            ("静态先验\n+ 其它屏的标签", mb["matched_static_prior"], C["static_ranker"]),
            ("Online ML\n+ 本屏在线重训", mb["matched_online_ml"], C["online_adaptive"]),
            ("Waddington C\n+ LLM 点名（0.2 权重）", mb["C"], C["waddington_c"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    y = hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], xmax=0.33)
    ax.tick_params(axis="y", labelsize=8)
    jump = rows[4][1] - rows[3][1]
    ax.annotate("", xy=(rows[4][1], y[4] + 0.32), xytext=(rows[3][1], y[4] + 0.32),
                arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.1))
    ax.text(rows[4][1] + 0.004, y[4] + 0.55, f"最大一跳 +{jump:.3f}", fontsize=8, color=S.INK, va="center")
    ax.set_xlabel("9 屏平均 hit@R5（主档）")
    title(ax, "提升从哪来：每一级只多一种信息", "灰色 = 前人方法；三个 ML 基线与 C 同一特征配置")
    save(fig, "ladder")


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
    title(a, "Reasoning or recall: information given to the LLM, step by step")
    save(fig, "reasoning")


def fig_llm_role():
    """Explainer 4.1.6: reasoning vs recall (paper Table 8) and hit rate by attribution source (paper Table 9)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.95))
    _ladder_panel(a)
    title(a, "推理还是记忆：逐级给 LLM 信息")
    src = ["ML 与 LLM 都选\n（1,900 个）", "只有 ML 选\n（77,020 个）", "只有 LLM 选\n（17,080 个）"]
    allv, nogw = [42.1, 18.3, 21.0], [21.6, 13.8, 17.0]
    paper(*allv, *nogw, nd=1)
    yy = np.arange(3)[::-1]
    b.barh(yy + 0.17, allv, height=0.32, color=BLUE, edgecolor=S.SURFACE, zorder=3)
    b.barh(yy - 0.17, nogw, height=0.32, color=PURPLE, edgecolor=S.SURFACE, zorder=3)
    for yi, v1, v2 in zip(yy, allv, nogw):
        b.text(v1 + 0.8, yi + 0.17, f"{v1:.1f}%", fontsize=7.6, va="center")
        b.text(v2 + 0.8, yi - 0.17, f"{v2:.1f}%", fontsize=7.6, va="center")
    b.set_yticks(yy, src, fontsize=8)
    b.tick_params(axis="y", length=0)
    b.grid(axis="y", visible=False)
    b.set_xlim(0, 72)
    b.set_xlabel("被选基因的 hit 率")
    legend(b, [Patch(color=BLUE, label="9 屏合并"), Patch(color=PURPLE, label="去掉 K562-GWPS")], loc="upper center",
           bbox_to_anchor=(0.45, -0.24), ncol=2)
    title(b, "按来源分的命中率")
    save(fig, "llm_role")


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
        fig_per_screen("headline", "per_screen_main", GD_H, BDA_H, "逐屏 hit@R5：非必需（主档）")
        fig_ladder()
        fig_paired()
        fig_clean_features()
        fig_ablation()
        fig_llm_role()
        fig_engines()
        fig_acquisition()
        fig_engines_in_system()
        fig_robustness()
        fig_tiers()
        fig_per_screen("full_pool", "per_screen_full", GD_F, BDA_F, "逐屏 hit@R5：全池档")
        fig_pairs()
        fig_pair_engines()
        fig_jurkat()
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
