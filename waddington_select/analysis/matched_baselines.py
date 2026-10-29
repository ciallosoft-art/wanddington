"""
matched_baselines.py — the main table's ML baselines re-run in Waddington C's own feature configuration
(2026-10-23; local, no LLM).

The benchmark baselines (static LOO ranker, Online ML, Coreset) were defined before the leakage audit and read a
fixed nine features: the six gene-intrinsic ones and the three anchor-relative ones, with no DepMap. The reported
C-arm reads no anchor-relative feature and selects DepMap by the metadata rule (6 / 9 / 10 features per screen).
Comparing C with those baselines therefore mixes two changes. Here each baseline is re-run with exactly the
features C's ML branch reads (waddington_c_arm._get_feature_config, anchors dropped), on all three tiers:

  matched static prior   C's ML engine at round 0, its ranking walked down (= final_system_stats._matched_loo)
  matched online ML      C's ML engine over five rounds with retraining, no LLM (= the "no LLM" ablation)
  matched Coreset        greedy k-center over C's features instead of the nine benchmark ones

The padded LLM baseline (B) is re-run separately, with its padding on the same features (experiment 23,
analysis/matched_b.py); the B read here is the original-configuration one. Reported C (twenty replicates) and the original benchmark-configuration baselines are read from the
reported runs for each tier, and C is paired against both versions of each baseline.

Run:  python -m waddington_select.analysis.matched_baselines     -> workspace/results/matched_baselines.json
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "workspace" / "evaluation"
GPT = REPO / "workspace" / "results" / "gpt"
OUT = REPO / "workspace" / "results" / "matched_baselines.json"
TIERS = {
    "headline": {"env": {}, "runs": [GPT / "replogle_transcriptomics_v2_20rep" / "main_table_20rep.json"]},
    "full_pool": {"env": {"WADDINGTON_TRAINING_CSV": str(EVAL / "lgbm_training_data_tier_a.csv"),
                          "WADDINGTON_TRAINING_CSV_V3": str(EVAL / "lgbm_training_data_tier_a_v3.csv")},
                  "runs": [GPT / "tier_a_full_pool_v2_20rep" / "main_table_20rep.json"]},
    "pairs": {"env": {"WADDINGTON_TRAINING_CSV": str(EVAL / "lgbm_training_data_horlbeck_pairs_mean.csv"),
                      "WADDINGTON_TRAINING_CSV_V3": str(EVAL / "lgbm_training_data_horlbeck_pairs_mean_v3.csv"),
                      "WADDINGTON_EXTRA_DATASETS": "Horlbeck:128"},
              "runs": [GPT / "tier_b_pairs_20rep" / "calibration_20rep.json", GPT / "tier_b_pairs_20rep" / "waddington_c_20rep.json"]},
    # BDA's own 32-pairs-per-round protocol (160 picks), and the held-out Jurkat map at both budgets
    "pairs_b32": {"env": {"WADDINGTON_TRAINING_CSV": str(EVAL / "lgbm_training_data_horlbeck_pairs_mean.csv"),
                          "WADDINGTON_TRAINING_CSV_V3": str(EVAL / "lgbm_training_data_horlbeck_pairs_mean_v3.csv"),
                          "WADDINGTON_EXTRA_DATASETS": "Horlbeck:32"},
                  "runs": [GPT / "tier_b_pairs_b32_20rep" / "calibration_20rep.json"]},
    **{f"jurkat_b{b}": {"env": {"WADDINGTON_TRAINING_CSV": str(EVAL / "lgbm_training_data_horlbeck_jurkat_pairs.csv"),
                                "WADDINGTON_TRAINING_CSV_V3": str(EVAL / "lgbm_training_data_horlbeck_jurkat_pairs_v3.csv"),
                                "WADDINGTON_PAIR_DATASETS": "Horlbeck_Jurkat",
                                "WADDINGTON_EXTRA_DATASETS": f"Horlbeck_Jurkat:{b}"},
                        "runs": [GPT / f"jurkat_b{b}_20rep" / f for f in
                                 ("deterministic_1rep.json", "random_20rep.json", "llm_reasoning_20rep.json",
                                  "waddington_c_20rep.json")]}
       for b in (128, 32)},
    # Jurkat from the table without the eight pair_* interaction columns, K562's reported pair encoding
    # (experiment 24, analysis/jurkat_no_interaction.py); random is the same run as above
    **{f"jurkat_mean_b{b}": {"env": {"WADDINGTON_TRAINING_CSV": str(EVAL / "lgbm_training_data_horlbeck_jurkat_pairs_mean.csv"),
                                     "WADDINGTON_TRAINING_CSV_V3": str(EVAL / "lgbm_training_data_horlbeck_jurkat_pairs_mean_v3.csv"),
                                     "WADDINGTON_PAIR_DATASETS": "Horlbeck_Jurkat",
                                     "WADDINGTON_EXTRA_DATASETS": f"Horlbeck_Jurkat:{b}"},
                             "runs": [GPT / f"jurkat_mean_b{b}_20rep" / "deterministic_1rep.json",
                                      GPT / f"jurkat_b{b}_20rep" / "random_20rep.json",
                                      GPT / f"jurkat_mean_b{b}_20rep" / "waddington_c_20rep.json"]}
       for b in (128, 32)},
}
PAIR_TIERS = {"pairs": "Horlbeck", "pairs_b32": "Horlbeck", "jurkat_b128": "Horlbeck_Jurkat", "jurkat_b32": "Horlbeck_Jurkat",
              "jurkat_mean_b128": "Horlbeck_Jurkat", "jurkat_mean_b32": "Horlbeck_Jurkat"}
ARMS = ("static_ranker", "online_adaptive", "coreset", "llm_reasoning", "waddington_c", "random")


def evaluate(tier: str) -> dict:
    """Run in a subprocess whose environment selects the tier's tables."""
    os.environ["WADDINGTON_FEATURE_POLICY"] = "metadata"
    os.environ["WADDINGTON_DROP_ANCHOR_FEATS"] = "1"
    from ..arms import coreset_arm as CA
    from ..arms.online_adaptive_arm import OnlineAdaptiveArm
    from ..arms.waddington_c_arm import _get_feature_config
    from ..oracle import BATCH_SIZES, DatasetOracle
    from ..router_protocol import BENCH
    from ..sequential_runner import run_arm_on_dataset
    from ..tables import with_pair_spread

    datasets = [PAIR_TIERS[tier]] if tier in PAIR_TIERS else list(BENCH)

    def engine(d):
        csv, extra = _get_feature_config(d)
        return OnlineAdaptiveArm(d, BATCH_SIZES[d], training_csv=csv, extra_feature_cols=extra, drop_anchor_feats=True)

    class MatchedCoreset(CA.CoresetArm):
        """Greedy k-center over the features C's ML branch reads, from C's table."""
        def _load_features(self, dataset_name):
            import pandas as pd
            csv, _ = _get_feature_config(dataset_name)
            feats = engine(dataset_name)._available_feats
            df = pd.read_csv(csv)
            df["gene"] = df["gene"].astype(str).str.strip().str.upper()
            sub = df[df["dataset"] == dataset_name].reset_index(drop=True)
            cols = with_pair_spread([c for c in feats if c in sub.columns], sub)
            assert set(cols) == set(feats), (dataset_name, sorted(set(feats) - set(cols)))
            return sub["gene"].tolist(), sub[cols].fillna(0.0).values.astype(np.float32)

    out = {}
    for d in datasets:
        orc = DatasetOracle(d)
        b, total = BATCH_SIZES[d], orc.total_hits
        e = engine(d)
        scores = e.all_scores()
        ranking = sorted(scores, key=lambda g: -scores[g])          # as final_system_stats._matched_loo
        static = []
        for r in range(1, 6):
            static.append(sum(orc.reveal(ranking[: r * b]).values()) / total)
        online = run_arm_on_dataset(engine(d), d, n_rounds=5).hit_ratio_per_round
        core = run_arm_on_dataset(MatchedCoreset(d, b), d, n_rounds=5).hit_ratio_per_round
        out[d] = {"total_hits": int(total), "n_features": len(e._available_feats), "features": e._available_feats,
                  "static_prior": static, "online_ml": list(online), "coreset": list(core)}
    return out


def _reported(tier: str) -> dict:
    """Per screen, per arm: mean hit ratio per round over the reported run's replicates."""
    res = {}
    for f in TIERS[tier]["runs"]:
        if not Path(f).exists():
            continue
        run = json.loads(Path(f).read_text())
        for d, arms in run.items():
            for a, reps in arms.items():
                if a in ARMS:
                    res.setdefault(d, {})[a] = np.mean([r["hit_ratio_per_round"] for r in reps], axis=0).tolist()
    return res


def main() -> None:
    if "--evaluate" in sys.argv:
        print("RESULT " + json.dumps(evaluate(sys.argv[sys.argv.index("--evaluate") + 1])))
        return
    from .final_system_stats import _paired
    out = {}
    for tier, cfg in TIERS.items():
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("WADDINGTON_TRAINING_CSV", "WADDINGTON_EXTRA_DATASETS", "WADDINGTON_PAIR_DATASETS"))}
        env.update(cfg["env"])
        res = subprocess.run([sys.executable, "-m", "waddington_select.analysis.matched_baselines", "--evaluate", tier],
                             cwd=REPO, env=env, capture_output=True, text=True, check=True).stdout
        matched = json.loads(res[res.rindex("RESULT ") + 7:])
        rep = _reported(tier)
        t = {"matched": matched, "reported": {d: rep[d] for d in matched}}
        final = lambda per_round_by_ds: {d: v[-1] for d, v in per_round_by_ds.items()}
        C = final({d: rep[d]["waddington_c"] for d in matched if "waddington_c" in rep[d]})
        series = {"C": C}
        for a in ("static_prior", "online_ml", "coreset"):
            series[f"matched_{a}"] = final({d: matched[d][a] for d in matched})
        for a in ("static_ranker", "online_adaptive", "coreset", "llm_reasoning", "random"):
            series[f"benchmark_{a}"] = final({d: rep[d][a] for d in matched if a in rep[d]})
        t["means"] = {k: float(np.mean(list(v.values()))) for k, v in series.items() if v}
        if tier in PAIR_TIERS:
            tot = matched[PAIR_TIERS[tier]]["total_hits"]
            t["hits"] = {k: round(list(v.values())[0] * tot, 1) for k, v in series.items() if v}
        else:
            t["paired_C_vs"] = {k: _paired(C, v) for k, v in series.items() if k != "C" and v}
        out[tier] = t
        print(f"== {tier}: " + ", ".join(f"{k} {v:.4f}" for k, v in t["means"].items()), flush=True)
        if tier in PAIR_TIERS:
            print("   hits:", t["hits"])
        else:
            for k, p in t["paired_C_vs"].items():
                print(f"   C vs {k:28s} {p['delta']:+.4f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] {p['wins']}/{p['ties']}/{p['losses']}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
