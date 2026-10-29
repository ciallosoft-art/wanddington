"""
matched_b.py — the padded LLM baseline B with its padding on the C-arm's features (experiment 23), against the
reported B (padding on the original benchmark configuration: six intrinsic + three anchor, no DepMap) and C.

Single-gene tiers: per-screen means over twenty replicates, then paired screen-clustered bootstrap over the nine
screens (final_system_stats._paired, seed 4 as its vs_llm_branch, so C - reported B reproduces the paper's
+0.049 [0.019, 0.080]). Two-gene tiers (one map each): Welch t over replicates.

  C - matched B       the comparison the main tables carry: only who leads, the LLM or the ML, differs
  C - reported B      the original-configuration B (Appendix, tab:benchmark_config)
  matched B - reported B   what the padding features alone did to B (anchors out, DepMap in)
  BDA - matched B     two-gene tiers: BioDiscoveryAgent's agent at the same budget

Run:  python -m waddington_select.analysis.matched_b      -> workspace/results/matched_b.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .final_system_stats import _paired
from .sources import GPT, MATCHED_B_ARM, REPO, REPORTED_RUN, REPORTED_TIER_A_RUN, matched_b_run

OUT = REPO / "workspace" / "results" / "matched_b.json"
SINGLE = {"headline": REPORTED_RUN, "full_pool": REPORTED_TIER_A_RUN}
EVAL = REPO / "workspace" / "evaluation"
LABELS = {"headline": EVAL / "lgbm_training_data.csv", "full_pool": EVAL / "lgbm_training_data_tier_a.csv",
          "pairs128": EVAL / "lgbm_training_data_horlbeck_pairs_mean.csv",
          "pairs32": EVAL / "lgbm_training_data_horlbeck_pairs_mean.csv",
          "jurkat128": EVAL / "lgbm_training_data_horlbeck_jurkat_pairs.csv",
          "jurkat32": EVAL / "lgbm_training_data_horlbeck_jurkat_pairs.csv"}
PAIRS = {  # tier -> (dataset, reported B file, C file, BDA file)
    "pairs128": ("Horlbeck", GPT / "tier_b_pairs_20rep" / "calibration_20rep.json",
                 GPT / "tier_b_pairs_20rep" / "waddington_c_20rep.json",
                 GPT / "tier_b_pairs_20rep" / "bda_controlled_real.json"),
    "pairs32": ("Horlbeck", GPT / "tier_b_pairs_b32_20rep" / "calibration_20rep.json",
                GPT / "tier_b_pairs_b32_20rep" / "waddington_c_20rep.json",
                GPT / "tier_b_pairs_b32_20rep" / "bda_controlled_real.json"),
    "jurkat128": ("Horlbeck_Jurkat", GPT / "jurkat_b128_20rep" / "llm_reasoning_20rep.json",
                  GPT / "jurkat_b128_20rep" / "waddington_c_20rep.json",
                  GPT / "jurkat_b128_20rep" / "bda_20rep.json"),
    "jurkat32": ("Horlbeck_Jurkat", GPT / "jurkat_b32_20rep" / "llm_reasoning_20rep.json",
                 GPT / "jurkat_b32_20rep" / "waddington_c_20rep.json",
                 GPT / "jurkat_b32_20rep" / "bda_20rep.json"),
}
SEED = 4  # final_system_stats.run_gpt's seed for C vs the LLM branch


def _reps(path: Path, ds: str, arm: str) -> list[dict]:
    return json.loads(path.read_text())[ds][arm]


def _welch(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, dof)
    d = a.mean() - b.mean()
    return {"delta": float(d), "ci95": [float(d - t * se), float(d + t * se)],
            "p": float(stats.ttest_ind(a, b, equal_var=False).pvalue)}


def _padding_share(reps: list[dict]) -> float:
    """Share of B's selected genes that came from the static padding, over rounds and replicates."""
    named = sum(len(x) for r in reps for x in r["llm_named_proposals_per_round"])
    pad = sum(len(x) for r in reps for x in r["llm_static_padding_proposals_per_round"])
    return pad / (named + pad)


def _hits(tier: str) -> dict:
    """dataset -> set of hit tokens (genes, or GENEA|GENEB pairs), from the tier's label table."""
    import pandas as pd
    df = pd.read_csv(LABELS[tier], usecols=["gene", "dataset", "label"])
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    return {d: set(g.loc[g["label"] == 1, "gene"]) for d, g in df.groupby("dataset")}


def _split(reps: list[dict], hits: set, total: int) -> dict:
    """Mean per replicate: LLM-named picks, hits among them, hits among the padding (as a share of all hits
    for a single-gene screen: total > 0 divides)."""
    named = [sum(len(x) for x in r["llm_named_proposals_per_round"]) for r in reps]
    hn = [sum(len(set(x) & hits) for x in r["llm_named_proposals_per_round"]) for r in reps]
    hp = [sum(len(set(x) & hits) for x in r["llm_static_padding_proposals_per_round"]) for r in reps]
    return {"named_picks": float(np.mean(named)), "named_hits": float(np.mean(hn)) / total,
            "padding_hits": float(np.mean(hp)) / total}


def main() -> None:
    out = {}
    for tier, run in SINGLE.items():
        f = matched_b_run(tier)
        if not f.exists():
            continue
        new, rep = json.loads(f.read_text()), json.loads(run.read_text())
        screens = list(rep)
        mean = lambda reps: float(np.mean([r["hit_ratio_per_round"][-1] for r in reps]))
        b_new = {s: mean(new[s][MATCHED_B_ARM]) for s in screens}
        b_old = {s: mean(rep[s]["llm_reasoning"]) for s in screens}
        c = {s: mean(rep[s]["waddington_c"]) for s in screens}
        out[tier] = {
            "unit": "hit@R5, mean of nine screens",
            "means": {"matched_B": float(np.mean(list(b_new.values()))), "reported_B": float(np.mean(list(b_old.values()))),
                      "C": float(np.mean(list(c.values())))},
            "per_screen": {"matched_B": b_new, "reported_B": b_old, "C": c},
            "C_minus_matched_B": _paired(c, b_new, seed=SEED), "C_minus_reported_B": _paired(c, b_old, seed=SEED),
            "matched_B_minus_reported_B": _paired(b_new, b_old, seed=SEED),
            "padding_share": {"matched_B": _padding_share([r for s in screens for r in new[s][MATCHED_B_ARM]]),
                              "reported_B": _padding_share([r for s in screens for r in rep[s]["llm_reasoning"]])},
            "padding_share_per_screen": {
                "matched_B": {s: _padding_share(new[s][MATCHED_B_ARM]) for s in screens},
                "reported_B": {s: _padding_share(rep[s]["llm_reasoning"]) for s in screens}},
            "n_replicates": {s: len(new[s][MATCHED_B_ARM]) for s in screens},
        }
        H = _hits(tier)
        out[tier]["source_of_hits"] = {
            name: {k: float(np.mean([_split(reps[s], H[s], len(H[s]))[k] for s in screens]))
                   for k in ("named_picks", "named_hits", "padding_hits")}
            for name, reps in (("matched_B", {s: new[s][MATCHED_B_ARM] for s in screens}),
                               ("reported_B", {s: rep[s]["llm_reasoning"] for s in screens}))}
    for tier, (ds, b_file, c_file, bda_file) in PAIRS.items():
        f = matched_b_run(tier)
        if not f.exists():
            continue
        new = _reps(f, ds, MATCHED_B_ARM)
        old, c = _reps(b_file, ds, "llm_reasoning"), _reps(c_file, ds, "waddington_c")
        hits = lambda reps: [r["cumulative_hits"][-1] for r in reps]
        hn, ho, hc, hb = hits(new), hits(old), hits(c), hits(_reps(bda_file, ds, "bda"))
        out[tier] = {
            "unit": "cumulative hits, " + ds,
            "means": {"matched_B": float(np.mean(hn)), "reported_B": float(np.mean(ho)), "C": float(np.mean(hc)),
                      "BDA": float(np.mean(hb))},
            "sd": {"matched_B": float(np.std(hn, ddof=1)), "reported_B": float(np.std(ho, ddof=1)),
                   "C": float(np.std(hc, ddof=1)), "BDA": float(np.std(hb, ddof=1))},
            "C_minus_matched_B": _welch(hc, hn), "C_minus_reported_B": _welch(hc, ho),
            "matched_B_minus_reported_B": _welch(hn, ho), "BDA_minus_matched_B": _welch(hb, hn),
            "padding_share": {"matched_B": _padding_share(new), "reported_B": _padding_share(old)},
            "n_replicates": len(new),
            "source_of_hits": {"matched_B": _split(new, _hits(tier)[ds], 1),
                               "reported_B": _split(old, _hits(tier)[ds], 1)},
        }
    OUT.write_text(json.dumps(out, indent=1))
    for tier, r in out.items():
        m = r["means"]
        print(f"{tier:10s} matched B {m['matched_B']:.4g}  reported B {m['reported_B']:.4g}  C {m['C']:.4g}")
        for k in ("C_minus_matched_B", "C_minus_reported_B", "matched_B_minus_reported_B", "BDA_minus_matched_B"):
            if k not in r:
                continue
            p = r[k]
            tail = f"{p['wins']}/{p['ties']}/{p['losses']}" if "wins" in p else f"p={p['p']:.2g}"
            print(f"    {k:28s} {p['delta']:+.4g} [{p['ci95'][0]:+.4g}, {p['ci95'][1]:+.4g}] {tail}")
        print(f"    padding share: matched {r['padding_share']['matched_B']:.2f}, reported {r['padding_share']['reported_B']:.2f}")
        for k, v in r["source_of_hits"].items():
            print(f"    {k:10s} named picks {v['named_picks']:.1f}  hits from named {v['named_hits']:.4g}  from padding {v['padding_hits']:.4g}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
