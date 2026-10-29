"""
jurkat_no_interaction.py — the held-out Jurkat map in the reported C-arm's own pair encoding (experiment 24).

Experiment 18 ran every arm on Jurkat from the pair table that also carries the eight pair_* interaction columns,
and tables.with_pair_features adds any such column to the features an arm reads. The K562 C-arm it is compared
with ran before those columns existed (its setting is the *_pairs_mean table). A static ranker cannot use them:
its training rows are the nine single-gene screens as degenerate pairs, where the columns carry nothing (the
static ranker reaches 66 / 21 Jurkat hits either way, and the matched B's padding ranking is identical on both
tables). Every model retrained on revealed Jurkat pairs can: experiment 18's online ML reaches 87 hits at 640
picks with them and 72 without. Experiment 18's "reported C-arm" is therefore K562's C-arm with the interaction
features in its table, not its reported one. Experiment 24 re-runs C and the deterministic baselines on the
Jurkat *_pairs_mean tables. Experiment 18 stays the frozen, pre-registered analysis; this is a supplement.

Per budget (640 picks = 128 per round, 160 = 32), cumulative hits, Welch t over replicates (one-sample against a
deterministic arm):

  C without the interaction features  vs C of experiment 18 (with them), BDA's agent, the matched B (its padding
                                      is the same on both tables, so experiment 23's run serves both), and the
                                      matched online ML of each table
  K562 analogues                      C 112.6 / 31.9 without interaction features, 137.3 / 44.9 with the seven
                                      original columns (pair_tier.py; Jurkat's table has the eighth, complex
                                      co-membership, too)

    python -m waddington_select.analysis.jurkat_no_interaction   -> workspace/results/jurkat_no_interaction.json
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from .sources import GPT, MATCHED_B_ARM, REPO, matched_b_run

RES = REPO / "workspace" / "results"
OUT = RES / "jurkat_no_interaction.json"
DS = "Horlbeck_Jurkat"
BUDGETS = {128: 640, 32: 160}


def _hits(path, arm: str) -> np.ndarray:
    return np.array([r["cumulative_hits"][-1] for r in json.loads(path.read_text())[DS][arm]], float)


def _cmp(a: np.ndarray, b: np.ndarray) -> dict:
    """a minus b: Welch t interval and p, or the one-sample interval of a when b is a single deterministic run."""
    d = float(a.mean() - b.mean())
    if len(b) == 1:
        se, dof = a.std(ddof=1) / np.sqrt(len(a)), len(a) - 1
        p = stats.ttest_1samp(a, b[0]).pvalue
    else:
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        se = np.sqrt(va + vb)
        dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    h = stats.t.ppf(0.975, dof) * se
    return {"delta": d, "ci95": [d - h, d + h], "p": float(p)}


def main() -> None:
    mb = json.loads((RES / "matched_baselines.json").read_text())
    k562 = json.loads((RES / "pair_tier_summary.json").read_text())["runs"]
    out = {}
    for b, picks in BUDGETS.items():
        new, old = GPT / f"jurkat_mean_b{b}_20rep", GPT / f"jurkat_b{b}_20rep"
        if not (new / "waddington_c_20rep.json").exists():
            continue
        arms = {"C_no_interaction": _hits(new / "waddington_c_20rep.json", "waddington_c"),
                "C_exp18_with_interaction": _hits(old / "waddington_c_20rep.json", "waddington_c"),
                "BDA": _hits(old / "bda_20rep.json", "bda"),
                "matched_B": _hits(matched_b_run(f"jurkat{b}"), MATCHED_B_ARM),
                "random": _hits(old / "random_20rep.json", "random")}
        det = {}
        for tag, folder in (("no_interaction", new), ("with_interaction", old)):
            run = json.loads((folder / "deterministic_1rep.json").read_text())[DS]
            det[f"original_config_{tag}"] = {a: run[a][0]["cumulative_hits"][-1] for a in run}
            det[f"matched_{tag}"] = mb[f"jurkat_mean_b{b}" if tag == "no_interaction" else f"jurkat_b{b}"]["hits"]
        mo_new = np.array([det["matched_no_interaction"]["matched_online_ml"]])
        mo_old = np.array([det["matched_with_interaction"]["matched_online_ml"]])
        c, c18 = arms["C_no_interaction"], arms["C_exp18_with_interaction"]
        out[str(picks)] = {
            "means": {k: float(v.mean()) for k, v in arms.items()},
            "sd": {k: float(v.std(ddof=1)) for k, v in arms.items()},
            "n": {k: len(v) for k, v in arms.items()},
            "deterministic": det,
            "C_no_interaction_minus": {"C_exp18_with_interaction": _cmp(c, c18), "BDA": _cmp(c, arms["BDA"]),
                                       "matched_B": _cmp(c, arms["matched_B"]),
                                       "matched_online_ml_no_interaction": _cmp(c, mo_new)},
            "C_exp18_minus_matched_online_ml_with_interaction": _cmp(c18, mo_old),
            "k562": {"C_no_interaction": k562[str(picks)]["c_arm"]["mean"],
                     "C_with_7_interaction_columns": k562[str(picks)]["c_arm_interaction"]["mean"]},
        }
    OUT.write_text(json.dumps(out, indent=1))
    for picks, r in out.items():
        m, s = r["means"], r["sd"]
        print(f"== Jurkat, {picks} picks: " + ", ".join(f"{k} {m[k]:.1f}±{s[k]:.1f}" for k in m))
        for k, v in r["deterministic"].items():
            print(f"   {k:32s} {v}")
        for k, v in r["C_no_interaction_minus"].items():
            print(f"   C (no interaction) - {k:34s} {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] p={v['p']:.2g}")
        v = r["C_exp18_minus_matched_online_ml_with_interaction"]
        print(f"   C (exp 18) - matched online ML (with interaction)  {v['delta']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] p={v['p']:.2g}")
        print(f"   K562: {r['k562']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
