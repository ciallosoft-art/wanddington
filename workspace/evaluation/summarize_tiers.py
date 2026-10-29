#!/usr/bin/env python3
"""
Summarise the three BDA difficulty tiers side by side and emit the paper's LaTeX table.

  harder (headline)  1-gene, non-essential   workspace/results/gpt/gpt_5_6_luna_none_20rep/
  easier             1-gene, full pool       workspace/results/gpt/tier_a_full_pool_20rep/
  hardest            2-gene combinations     workspace/results/gpt/tier_b_pairs_20rep/

hit@R5 is the metric used throughout: cumulative hits after round 5 divided by the screen's total
hits, averaged over the twenty evaluation replicates and then over screens. For the combinatorial
tier the pool is 100 576 pairs against 9 959 hits, so 640 picks cap recall at 6.43%; the raw hit
count and batch precision are reported alongside because they are the readable numbers there.

Usage:
    conda run -n waddington-bio python3 workspace/evaluation/summarize_tiers.py
    conda run -n waddington-bio python3 workspace/evaluation/summarize_tiers.py --latex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GPT = REPO_ROOT / "workspace" / "results" / "gpt"

TIERS = {
    "harder": GPT / "gpt_5_6_luna_none_20rep",
    "easier": GPT / "tier_a_full_pool_20rep",
    "pairs": GPT / "tier_b_pairs_20rep",
    "pairs_b32": GPT / "tier_b_pairs_b32_20rep",
}
# GeneDisco's acquisition functions, ported onto our benchmark (three replicates, no LLM). The
# headline tier's run predates the tier work and lives in the frozen results tree.
GENEDISCO_FILES = {
    "harder": REPO_ROOT / "workspace" / "results" / "sequential" / "genedisco_ported.json",
    "easier": GPT / "tier_a_full_pool_20rep" / "genedisco_3rep.json",
    "pairs": GPT / "tier_b_pairs_20rep" / "genedisco_3rep.json",
}
# BioDiscoveryAgent, run on our benchmark with our oracle (see workspace/results/bda_controlled/).
# The two Replogle screens re-run under the corrected task prompts (experiment 25, analysis/bda_replogle_v2.py).
BDA_FILE = REPO_ROOT / "workspace" / "results" / "bda_controlled" / "bda_gpt_v2.json"
ARMS = ["random", "coreset", "static_ranker", "online_adaptive", "llm_reasoning", "waddington_c"]
ARM_LABEL = {
    "random": "Random",
    "coreset": "Coreset (GeneDisco A-arm)",
    "static_ranker": "Static LOO ranker",
    "online_adaptive": "Online ML",
    "llm_reasoning": "Padded LLM (BDA-style)",
    "waddington_c": "Waddington C",
}
SINGLE_GENE = [
    "IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22", "Scharenberg22",
    "Steinhart", "Replogle_K562_essential", "Replogle_K562_gwps",
]


def load_tier(directory: Path) -> dict:
    """Merge whatever result files a tier directory has into {dataset: {arm: [runs]}}."""
    merged: dict[str, dict[str, list]] = {}
    for name in ("main_table_20rep.json", "calibration_20rep.json", "waddington_c_20rep.json",
                 "genedisco_3rep.json"):
        path = directory / name
        if not path.exists():
            continue
        for dataset, arms in json.loads(path.read_text()).items():
            for arm, runs in arms.items():
                merged.setdefault(dataset, {}).setdefault(arm, runs)
    return merged


def final_ratios(runs: list) -> np.ndarray:
    return np.array([r["hit_ratio_per_round"][-1] for r in runs], dtype=float)


def final_hits(runs: list) -> np.ndarray:
    return np.array([r["cumulative_hits"][-1] for r in runs], dtype=float)


def ci95(x: np.ndarray) -> float:
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def genedisco_summary(tier: str, datasets: list[str]) -> tuple[dict[str, float], float] | None:
    """Per-acquisition means plus the best-per-screen oracle, as the paper reports GeneDisco."""
    path = GENEDISCO_FILES.get(tier)
    if not path or not path.exists():
        return None
    d = json.loads(path.read_text())
    present = [ds for ds in datasets if ds in d]
    if not present:
        return None
    arms = sorted({a for ds in present for a in d[ds]})
    per_arm = {a: float(np.mean([final_ratios(d[ds][a]).mean() for ds in present if a in d[ds]]))
               for a in arms}
    best = float(np.mean([max(final_ratios(d[ds][a]).mean() for a in arms if a in d[ds])
                          for ds in present]))
    return per_arm, best


def report_genedisco(tier: str, datasets: list[str]) -> None:
    res = genedisco_summary(tier, datasets)
    if res is None:
        print("    GeneDisco: no results yet")
        return
    per_arm, best = res
    for a, v in sorted(per_arm.items(), key=lambda kv: kv[1]):
        print(f"    {a.replace('genedisco_', 'GeneDisco '):26s}{v:>10.3f}")
    print(f"    {'GeneDisco best-per-screen':26s}{best:>10.3f}   (oracle over acquisitions)")


def report(name: str, data: dict, datasets: list[str]) -> dict[str, np.ndarray]:
    present = [d for d in datasets if d in data]
    if not present:
        print(f"\n### {name}: no results yet\n")
        return {}
    n_rep = {len(runs) for d in present for runs in data[d].values()}
    print(f"\n### {name}  ({len(present)} screen(s), {sorted(n_rep)} replicate(s))")
    header = f"{'arm':28s}" + "".join(f"{d[:14]:>16s}" for d in present) + f"{'mean':>10s}"
    print(header)
    print("-" * len(header))
    out: dict[str, np.ndarray] = {}
    for arm in ARMS:
        cells, per_ds = [], []
        for d in present:
            runs = data[d].get(arm)
            if not runs:
                cells.append(f"{'—':>16s}")
                continue
            r = final_ratios(runs)
            cells.append(f"{r.mean():>16.3f}")
            per_ds.append(r.mean())
        if not per_ds:
            continue
        out[arm] = np.array(per_ds)
        print(f"{ARM_LABEL[arm]:28s}" + "".join(cells) + f"{np.mean(per_ds):>10.3f}")
    return out


def report_pairs(data: dict) -> None:
    if "Horlbeck" not in data:
        print("\n### 2-gene combinations (Horlbeck): no results yet\n")
        return
    runs_by_arm = data["Horlbeck"]
    any_runs = next(iter(runs_by_arm.values()))
    total, pool = any_runs[0]["total_hits"], any_runs[0]["n_genes"]
    picks = any_runs[0]["batch_size"] * any_runs[0]["n_rounds"]
    print(f"\n### 2-gene combinations — Horlbeck (K562 GI map)")
    print(f"    pool {pool} pairs, {total} hits ({100 * total / pool:.2f}%), "
          f"{picks} picks -> recall ceiling {picks / total:.4f}, random expectation {picks / pool:.4f}")
    print(f"{'arm':28s}{'hits/640':>12s}{'precision':>12s}{'hit@R5':>12s}{'95% CI':>18s}{'x random':>10s}")
    print("-" * 92)
    base = None
    for arm in ARMS:
        runs = runs_by_arm.get(arm)
        if not runs:
            continue
        h, r = final_hits(runs), final_ratios(runs)
        if arm == "random":
            base = h.mean()
        lift = f"{h.mean() / base:>10.2f}" if base else f"{'—':>10s}"
        print(f"{ARM_LABEL[arm]:28s}{h.mean():>12.1f}{h.mean() / picks:>12.3f}"
              f"{r.mean():>12.4f}{f'[{r.mean()-ci95(r):.4f}, {r.mean()+ci95(r):.4f}]':>18s}{lift}")


def report_pairs_b32(data: dict) -> None:
    """Horlbeck at BioDiscoveryAgent's own Table 2 protocol: 32 pairs/round, 160 picks."""
    if "Horlbeck" not in data:
        print("\n### 2-gene at BDA's 32/round protocol: no results yet\n")
        return
    runs_by_arm = data["Horlbeck"]
    any_runs = next(iter(runs_by_arm.values()))
    picks = any_runs[0]["batch_size"] * any_runs[0]["n_rounds"]
    print(f"\n### 2-gene at BDA's own Table 2 protocol "
          f"({any_runs[0]['batch_size']} pairs/round x {any_runs[0]['n_rounds']} = {picks} picks)")
    print(f"{'arm':28s}{'cum hits @R5':>16s}{'x random':>10s}")
    print("-" * 54)
    base = None
    for arm in ARMS:
        runs = runs_by_arm.get(arm)
        if not runs:
            continue
        h = final_hits(runs)
        if arm == "random":
            base = h.mean()
        print(f"{ARM_LABEL[arm]:28s}{h.mean():>9.1f} +- {h.std(ddof=1):<4.1f}"
              f"{(h.mean() / base if base else float('nan')):>10.2f}")
    print("  BDA's published Table 2 (10 runs, Horlbeck): "
          "Random 16.4+-3.8 | Claude 3 Haiku 45.1+-5.3 | Claude 3.5 Sonnet 40.0+-3.8")


def latex(tiers: dict[str, dict]) -> None:
    print("\n% ---- generated by workspace/evaluation/summarize_tiers.py ----")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{single-gene, hit@R5} & 2-gene \\")
    print(r"\cmidrule(lr){2-3}")
    print(r"Method & full pool & non-essential & Horlbeck \\")
    print(r"\midrule")
    easy = report_quiet(tiers.get("easier", {}), SINGLE_GENE)
    hard = report_quiet(tiers.get("harder", {}), SINGLE_GENE)
    pair = tiers.get("pairs", {}).get("Horlbeck", {})
    for arm in ARMS:
        def cell(d):
            return f"{np.mean(d[arm]):.3f}" if arm in d else "--"
        p = f"{final_ratios(pair[arm]).mean():.4f}" if arm in pair else "--"
        print(f"{ARM_LABEL[arm]} & {cell(easy)} & {cell(hard)} & {p} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def report_quiet(data: dict, datasets: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for arm in ARMS:
        vals = [final_ratios(data[d][arm]).mean() for d in datasets
                if d in data and arm in data[d]]
        if vals:
            out[arm] = np.array(vals)
    return out


def report_bda(datasets: list[str]) -> None:
    if not BDA_FILE.exists():
        return
    d = json.loads(BDA_FILE.read_text())
    res = d.get("results", {})
    vals = [res[ds]["hit_at_r5"] for ds in datasets if ds in res]
    if vals:
        print(f"    {'BioDiscoveryAgent (ours)':26s}{np.mean(vals):>10.3f}   "
              f"(GPT-5.6 Luna, our oracle; headline tier only)")


def main(as_latex: bool = False) -> None:
    tiers = {k: load_tier(v) for k, v in TIERS.items()}
    report("1-gene, FULL pool (BDA 'easier')", tiers["easier"], SINGLE_GENE)
    print("  external prior methods:")
    report_genedisco("easier", SINGLE_GENE)
    report("1-gene, NON-ESSENTIAL pool (BDA 'harder' — the paper's headline tier)",
           tiers["harder"], SINGLE_GENE)
    print("  external prior methods:")
    report_genedisco("harder", SINGLE_GENE)
    report_bda(SINGLE_GENE)
    report_pairs(tiers["pairs"])
    print("  external prior methods:")
    report_genedisco("pairs", ["Horlbeck"])
    report_pairs_b32(tiers["pairs_b32"])
    if as_latex:
        latex(tiers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", dest="as_latex", action="store_true")
    main(**vars(ap.parse_args()))
