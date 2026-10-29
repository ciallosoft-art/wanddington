"""
bda_replogle_v2.py — BioDiscoveryAgent's two Replogle screens re-run with the corrected task prompts
(experiment 25), merged into copies of the original controlled runs. Local, no LLM calls.

BDA ships no task prompt for the two Replogle Perturb-seq screens; the controlled runs read ours, written before
the 2026-09-18 correction (fitness / survival wording) and never refreshed. Experiment 25 re-ran those two
screens on both single-gene tiers with the prompt our own LLM arms read. The seven other screens are copied
unchanged:

  headline   bda_controlled/bda_gpt.json (2026-08-30)              -> bda_controlled/bda_gpt_v2.json
  full_pool  gpt/tier_a_full_pool_20rep/bda_controlled.json (09-11) -> gpt/tier_a_full_pool_20rep/bda_controlled_v2.json

The headline file keeps its older record shape: selected_genes are the proposed names that are in the pool (on
the original K562-Essential run 64 of 160), n_selected their number; all proposed names go under
selected_genes_all. Also written, workspace/results/bda_replogle_v2.json: per screen old against new, the
nine-screen means, and C minus BDA paired over the screens (final_system_stats._paired) with either BDA.

    python -m waddington_select.analysis.bda_replogle_v2
"""

from __future__ import annotations

import json

import numpy as np

from .final_system_stats import BENCH, _paired
from .sources import GPT, REPO, REPORTED_RUN, REPORTED_TIER_A_RUN

RES = REPO / "workspace" / "results"
NEW = GPT / "bda_replogle_v2"
SCREENS = ("Replogle_K562_essential", "Replogle_K562_gwps")
FILES = {"headline": (RES / "bda_controlled" / "bda_gpt.json", RES / "bda_controlled" / "bda_gpt_v2.json"),
         "full_pool": (GPT / "tier_a_full_pool_20rep" / "bda_controlled.json",
                       GPT / "tier_a_full_pool_20rep" / "bda_controlled_v2.json")}
C_RUNS = {"headline": REPORTED_RUN, "full_pool": REPORTED_TIER_A_RUN}


def _pool(tier: str, ds: str) -> set[str]:
    """The tier's candidate pool for one screen, read from its table (not from the environment)."""
    import pandas as pd
    from ..tables import DEFAULT_TRAINING_CSV, FULL_POOL_CSV
    df = pd.read_csv(DEFAULT_TRAINING_CSV if tier == "headline" else FULL_POOL_CSV, usecols=["gene", "dataset"])
    return set(df.loc[df["dataset"] == ds, "gene"].astype(str).str.strip().str.upper())


def _final(rec: dict) -> float:
    return float(rec["hit_ratio_per_round"][-1])


def main() -> None:
    out = {}
    for tier, (orig_path, v2_path) in FILES.items():
        new_path = NEW / f"{tier}.json"
        if not new_path.exists():
            continue
        orig, new = json.loads(orig_path.read_text()), json.loads(new_path.read_text())
        merged = json.loads(orig_path.read_text())
        per_old, per_new, change = {}, {}, {}
        for ds in BENCH:
            if tier == "headline":
                per_old[ds] = _final(orig["results"][ds])
            else:
                per_old[ds] = _final(orig[ds]["bda"][0])
            per_new[ds] = per_old[ds]
        for ds in SCREENS:
            rec = new[ds]["bda"][0]
            picks = [g for r in rec["selected_genes_per_round"] for g in r]
            pool = _pool(tier, ds)
            in_pool = [g for g in picks if g in pool]
            if tier == "headline":
                merged["results"][ds] = {"rounds": rec["n_rounds"], "n_selected": len(in_pool),
                                         "hits": rec["cumulative_hits"][-1], "total_hits": rec["total_hits"],
                                         "hit_at_r5": _final(rec), "hit_ratio_per_round": rec["hit_ratio_per_round"],
                                         "selected_genes": in_pool, "selected_genes_all": picks,
                                         "task_prompt": "corrected (experiment 25)"}
            else:
                merged[ds]["bda"] = [dict(rec, task_prompt="corrected (experiment 25)")]
            per_new[ds] = _final(rec)
            old_rec = orig["results"][ds] if tier == "headline" else orig[ds]["bda"][0]
            old_in_pool = (old_rec["n_selected"] if tier == "headline" else
                           len([g for r in old_rec["selected_genes_per_round"] for g in r if g in pool]))
            change[ds] = {"old": per_old[ds], "new": per_new[ds], "picks": len(picks),
                          "in_pool_new": len(in_pool), "in_pool_old": old_in_pool,
                          "hits_old": old_rec.get("hits", old_rec.get("cumulative_hits", [None])[-1]),
                          "hits_new": rec["cumulative_hits"][-1]}
        note = ("Replogle_K562_essential and Replogle_K562_gwps re-run 2026-10-29 with the corrected task prompts "
                "(experiment 25, gpt/bda_replogle_v2); the other seven screens as in " + orig_path.name)
        if tier == "headline":
            merged["_method"] = merged.get("_method", "") + " | " + note
            merged["mean_hit_at_r5"] = round(float(np.mean(list(per_new.values()))), 4)
        else:
            merged["_note"] = note
        v2_path.write_text(json.dumps(merged, indent=1))
        run = json.loads(C_RUNS[tier].read_text())
        C = {s: float(np.mean([r["hit_ratio_per_round"][-1] for r in run[s]["waddington_c"]])) for s in BENCH}
        out[tier] = {"screens": change,
                     "mean_old": float(np.mean(list(per_old.values()))), "mean_new": float(np.mean(list(per_new.values()))),
                     "per_screen_new": per_new,
                     "C_minus_BDA_old": _paired(C, per_old), "C_minus_BDA_new": _paired(C, per_new),
                     "written": str(v2_path.relative_to(REPO))}
    (RES / "bda_replogle_v2.json").write_text(json.dumps(out, indent=1))
    for tier, r in out.items():
        print(f"== {tier}: BDA nine-screen mean {r['mean_old']:.4f} -> {r['mean_new']:.4f}")
        for ds, c in r["screens"].items():
            print(f"   {ds:26s} {c['old']:.3f} -> {c['new']:.3f}   in-pool picks {c['in_pool_old']} -> {c['in_pool_new']}"
                  f" (of {c['picks']} proposed)   hits {c['hits_old']} -> {c['hits_new']}")
        for k in ("C_minus_BDA_old", "C_minus_BDA_new"):
            p = r[k]
            print(f"   {k:16s} {p['delta']:+.4f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] {p['wins']}/{p['ties']}/{p['losses']}")
        print(f"   wrote {r['written']}")


if __name__ == "__main__":
    main()
