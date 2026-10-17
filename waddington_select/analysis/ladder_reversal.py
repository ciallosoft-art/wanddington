"""
ladder_reversal.py — why did the reasoning-vs-recall ladder reverse? (2026-10-17, no LLM calls)

The earlier GPT ladder (gpt/router/reasoning_noanchor.json, 3 replicates) put the anonymous-features arm A1
level with the real-name arm A2 (0.260 vs 0.254); on the reported system (tier_ablations/headline, 20
replicates) A2 leads by 0.020 (A1 0.241, A2 0.261). Between the two runs:

  A1  fusion weights   legacy target-aware router (0.4 on Scharenberg22, Steinhart, K562-Essential; two-stage
                       on K562-GWPS; 0.2 elsewhere)  ->  0.2 everywhere
      anchor features  kept, in the ML branch and in the LLM's feature view (main_reruns --old-anchor)  ->  dropped
      memory / prompts the first GPT run's memory, uncorrected Replogle task prompts  ->  the reported ones
  A2  memory / prompts only (same fixed weight, anchors dropped in both)
  both                 3 replicates  ->  20

What can be separated without calling the LLM:

  noise     each arm's replicate spread per screen, and the standard error it implies for the nine-screen
            mean and for the change in A2 - A1 between the runs
  replay    each run's recorded trajectories re-fused from the recorded LLM names under another fusion-weight
            setting or another ML feature set. Approximate: the ML branch keeps learning from the recorded
            reveals and the LLM's later proposals are held fixed; each alternative keeps its own picks (a gene
            it picked is not picked or counted again) and never re-picks a gene the recorded trajectory has
            tested (the ML has seen its label). Round 1 is
            reported separately: its LLM proposals do not depend on any earlier batch, so for the fusion
            weight it is an exact counterfactual. Under its own setting the replay must reproduce the recorded
            batches (verified rounds are reported). On K562-GWPS
            the legacy route is two-stage (the LLM picks the batch from the ML shortlist); the old A1's own
            batches there are taken as recorded, and the other runs' "legacy" alternative there is
            approximated by the weighted 0.6/0.4 fusion (flagged)
  llm_side  what each LLM named: the precision of its named genes, how many the ML's own top batch already
            held (from the replay), and the anchor-feature percentile of its named genes within the screen
            (did the old A1, which saw the anchor features, name genes by them?)

The rest of the A1 change --- what the LLM names when it sees the anchor features, and under the old memory
and prompts --- needs LLM calls (see docs/tier_extension_plan.md, part 戊).

    python -m waddington_select.analysis.ladder_reversal
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("WADDINGTON_FEATURE_POLICY", "metadata")  # all four runs' policy (their manifests)

import numpy as np

from ..arms.online_adaptive_arm import OnlineAdaptiveArm
from ..arms.waddington_c_arm import (W_LLM_DEFAULT, W_LLM_LARGE, W_ML_DEFAULT, W_ML_LARGE, _classify,
                                     _get_dataset_stats, _get_feature_config)
from ..features import load_training_frame
from ..oracle import BATCH_SIZES, DatasetOracle
from .final_system_stats import BENCH, _paired
from .replay_attribution import W_LLM, W_ML
from .sources import GPT_ROUTER
from .tier_extension import _arms

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "workspace" / "results"
HEAD = RES / "tier_ablations" / "headline"
OUT = RES / "ladder_reversal.json"
ANCHOR = ("g1_ppi_score", "archs4_coexpr", "kegg_overlap")

# run -> (file, arm, its own weight setting, its own ML anchors dropped)
RUNS = {
    "A1_old": (GPT_ROUTER / "reasoning_noanchor.json", "waddington_c_feature_reasoning", "legacy", False),
    "A2_old": (GPT_ROUTER / "reasoning_noanchor.json", "waddington_c", "fixed", True),
    "A1_new": (HEAD / "ladder_20rep.json", "waddington_c_anon_features", "fixed", True),
    "A2_new": (HEAD / "ablations_20rep.json", "waddington_c", "fixed", True),
}
# the alternative settings replayed for each run: name -> (weights, ML anchors dropped)
ALTS = {
    "A1_old": {"fixed_weights": ("fixed", False), "ml_no_anchor": ("legacy", True), "both": ("fixed", True)},
    "A2_old": {"legacy_weights": ("legacy", True)},
    "A1_new": {"legacy_weights": ("legacy", True), "ml_anchor": ("fixed", False), "both": ("legacy", False)},
    "A2_new": {"legacy_weights": ("legacy", True)},
}


def _weights(ds: str, kind: str) -> tuple[str, float, float]:
    if kind == "fixed":
        return "baseline", W_ML, W_LLM
    route = _classify(*_get_dataset_stats(ds))
    if route == "ml_heavy":
        return route, W_ML_LARGE, W_LLM_LARGE
    return route, W_ML_DEFAULT, W_LLM_DEFAULT


def _load(run: str) -> dict[str, list[dict]]:
    f, arm, _, _ = RUNS[run]
    return _arms(f)[arm]


def _replay_screen(ds: str, runs: list[dict], own: tuple[str, bool], alts: dict) -> dict:
    bs = BATCH_SIZES[ds]
    csv, extra = _get_feature_config(ds)
    oracle = DatasetOracle(ds)
    settings = {"own": own} | alts
    bases = {d: OnlineAdaptiveArm(ds, bs, training_csv=csv, extra_feature_cols=extra, drop_anchor_feats=d)
             for d in {s[1] for s in settings.values()}}
    hits = {k: [] for k in settings}
    round1 = {k: [] for k in settings}
    verified = rounds = in_top = named_total = 0
    approx = False
    for rec in runs:
        onl = {d: copy.deepcopy(b) for d, b in bases.items()}
        selected: set[str] = set()
        own_picks = {k: set() for k in settings}  # each setting's own earlier picks
        h = dict.fromkeys(settings, 0)
        n = len(rec["selected_genes_per_round"])
        for r, recorded in enumerate(rec["selected_genes_per_round"]):
            named = set(rec["llm_named_proposals_per_round"][r])
            for k, (wkind, drop) in settings.items():
                route, w_ml, w_llm = _weights(ds, wkind)
                ml = onl[drop].all_scores()
                # tested genes are out for every setting (the ML has their labels), and so are the setting's
                # own earlier picks
                pool = [g for g in onl[drop]._genes if g not in selected and g not in own_picks[k]]
                if route == "two_stage" and k == "own":
                    batch = list(recorded)  # the LLM's own pick from the shortlist; not replayable
                else:
                    if route == "two_stage":
                        approx = True  # another run under the legacy route on K562-GWPS: weighted 0.6/0.4
                    comb = {g: w_ml * ml.get(g, 0.0) + w_llm * (1.0 if g in named else 0.0) for g in pool}
                    batch = sorted(comb, key=comb.__getitem__, reverse=True)[:bs]
                if k == "own":
                    rounds += 1
                    verified += batch == list(recorded)
                    ml_top = set(sorted(pool, key=lambda g: -ml.get(g, 0.0))[:bs])
                    in_top += len(named & ml_top)
                    named_total += len(named)
                got = sum(oracle.reveal(batch).values())
                h[k] += got
                own_picks[k] |= set(batch)
                if r == 0:
                    round1[k].append(got / oracle.total_hits)
            rev = oracle.reveal(list(recorded))
            selected |= set(recorded)
            if r < n - 1:
                for o in onl.values():
                    o.update(r, rev)
        for k in settings:
            hits[k].append(h[k] / oracle.total_hits)
    rec_hr = float(np.mean([r["hit_ratio_per_round"][-1] for r in runs]))
    return {"recorded": rec_hr, "rounds": rounds, "verified": verified,
            "named_in_ml_top": in_top / max(1, named_total), "two_stage_approximated": approx,
            **{f"first_order_{k}": float(np.mean(v)) for k, v in hits.items()},
            **{f"round1_{k}": float(np.mean(v)) for k, v in round1.items()}}


def replay() -> dict:
    out = {}
    for run, (_, _, wkind, drop) in RUNS.items():
        data = _load(run)
        per = {}
        for ds in BENCH:
            per[ds] = _replay_screen(ds, data[ds], (wkind, drop), ALTS[run])
            print(run, ds, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in per[ds].items()}, flush=True)
        effects = {}
        for alt in ALTS[run]:
            d = {ds: per[ds][f"first_order_{alt}"] - per[ds]["first_order_own"] for ds in BENCH}
            d1 = {ds: per[ds][f"round1_{alt}"] - per[ds]["round1_own"] for ds in BENCH}
            effects[alt] = {"per_screen": d, "mean": float(np.mean(list(d.values()))),
                            "round1_per_screen": d1, "round1_mean": float(np.mean(list(d1.values())))}
        out[run] = {"per_screen": per, "effects_vs_own": effects,
                    "verified": f"{sum(p['verified'] for p in per.values())}/{sum(p['rounds'] for p in per.values())}",
                    "named_in_ml_top": float(np.mean([p["named_in_ml_top"] for p in per.values()]))}
    return out


def noise() -> dict:
    out = {}
    for run in RUNS:
        data = _load(run)
        sd = {ds: float(np.std([r["hit_ratio_per_round"][-1] for r in data[ds]], ddof=1)) for ds in BENCH}
        n = {ds: len(data[ds]) for ds in BENCH}
        mean = {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in data[ds]])) for ds in BENCH}
        se = float(np.sqrt(sum(sd[ds] ** 2 / n[ds] for ds in BENCH)) / len(BENCH))
        out[run] = {"mean": float(np.mean(list(mean.values()))), "per_screen": mean, "sd": sd,
                    "replicates": n["IFNG"], "se_of_nine_screen_mean": se}
    se_change = float(np.sqrt(sum(out[r]["se_of_nine_screen_mean"] ** 2 for r in RUNS)))  # (A2-A1)_new - (A2-A1)_old
    gap_old = out["A2_old"]["mean"] - out["A1_old"]["mean"]
    gap_new = out["A2_new"]["mean"] - out["A1_new"]["mean"]
    a1_change = out["A1_new"]["mean"] - out["A1_old"]["mean"]
    a2_change = out["A2_new"]["mean"] - out["A2_old"]["mean"]
    se_a1 = float(np.hypot(out["A1_new"]["se_of_nine_screen_mean"], out["A1_old"]["se_of_nine_screen_mean"]))
    se_a2 = float(np.hypot(out["A2_new"]["se_of_nine_screen_mean"], out["A2_old"]["se_of_nine_screen_mean"]))
    return {"runs": out, "name_value_old": gap_old, "name_value_new": gap_new,
            "change_in_name_value": gap_new - gap_old, "se_change_in_name_value": se_change,
            "a1_change": a1_change, "se_a1_change": se_a1, "a2_change": a2_change, "se_a2_change": se_a2,
            "a1_change_paired": _paired(out["A1_new"]["per_screen"], out["A1_old"]["per_screen"]),
            "a2_change_paired": _paired(out["A2_new"]["per_screen"], out["A2_old"]["per_screen"])}


def _anchor_pct(ds: str) -> dict[str, dict[str, float]]:
    """Each candidate's percentile within the screen on each anchor feature (higher = closer to the anchors)."""
    csv, _ = _get_feature_config(ds)
    df = load_training_frame(csv, ds)
    df = df[df["dataset"] == ds].copy()
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    return {f: dict(zip(df["gene"], df[f].rank(pct=True))) for f in ANCHOR if f in df.columns}


def llm_side() -> dict:
    out = {}
    pct = {ds: _anchor_pct(ds) for ds in BENCH}
    for run in RUNS:
        data = _load(run)
        per = {}
        for ds in BENCH:
            oracle = DatasetOracle(ds)
            names = [g for rec in data[ds] for rnd in rec["llm_named_proposals_per_round"] for g in rnd]
            prec = float(np.mean(list(oracle.reveal(names).values()))) if names else float("nan")
            anc = {f: float(np.mean([p[g] for g in names if g in p])) for f, p in pct[ds].items()}
            per[ds] = {"precision": prec, "anchor_percentile": anc, "named_per_round":
                       len(names) / sum(len(r["llm_named_proposals_per_round"]) for r in data[ds])}
        out[run] = {"per_screen": per,
                    "precision": float(np.mean([per[ds]["precision"] for ds in BENCH])),
                    "anchor_percentile": {f: float(np.mean([per[ds]["anchor_percentile"][f] for ds in BENCH]))
                                          for f in ANCHOR}}
    return out


def _cell(name: str, arm: str) -> dict[str, float]:
    return {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in runs]))
            for ds, runs in _arms(HEAD / f"{name}_10rep.json")[arm].items()}


def gpt_cells(replay_res: dict | None = None) -> dict:
    """Part 戊 (docs/tier_extension_plan.md): the GPT cells D0-D2 against the four runs, paired over screens."""
    if not (HEAD / "reversal_d1_10rep.json").exists():
        return {}
    ps = lambda run: {ds: float(np.mean([r["hit_ratio_per_round"][-1] for r in rs])) for ds, rs in _load(run).items()}
    a1_old, a2_old, a1_new, a2_new = ps("A1_old"), ps("A2_old"), ps("A1_new"), ps("A2_new")
    d0_a1, d0_a2 = _cell("reversal_d0", "waddington_c_feature_reasoning"), _cell("reversal_d0", "waddington_c")
    d1 = _cell("reversal_d1", "waddington_c_anon_features")
    d2 = _cell("reversal_d2", "waddington_c_anon_features") if (HEAD / "reversal_d2_10rep.json").exists() else {}
    d2_done = set(BENCH) <= set(d2)
    m = lambda x: float(np.mean([x[s] for s in BENCH]))
    weights = None
    if replay_res:
        # A1 at the legacy weights minus A1 at 0.2: minus the old A1's fixed-weight replay effect
        weights = -replay_res["A1_old"]["effects_vs_own"]["fixed_weights"]["mean"]
    a1_total = _paired(d0_a1, a1_new)
    anchors = _paired(d1, a1_new)
    memory = _paired(d2, a1_new) if d2_done else None
    out = {
        "means": {"d0_old_a1": m(d0_a1), "d0_a2": m(d0_a2), "d1_a1_anchors": m(d1), "d2_a1_old_memory": m(d2) if d2_done else None,
                  "a1_old_aug": m(a1_old), "a2_old_aug": m(a2_old), "a1_new": m(a1_new), "a2_new": m(a2_new)},
        "per_screen": {"d0_old_a1": d0_a1, "d0_a2": d0_a2, "d1": d1, "d2": d2},
        "drift": {"old_a1_now_minus_aug": _paired(d0_a1, a1_old), "a2_old_memory_now_minus_aug": _paired(d0_a2, a2_old)},
        "name_value": {"aug": m(a2_old) - m(a1_old), "d0_now": _paired(d0_a2, d0_a1), "new": _paired(a2_new, a1_new)},
        "a1": {"total_old_condition_minus_new": a1_total, "anchors": anchors, "memory_prompts": memory,
               "weights_replay": weights,
               "remainder": (a1_total["delta"] - anchors["delta"] - memory["delta"] - (weights or 0.0)) if memory else None,
               "d2_screens_done": sorted(d2)},
        "a2": {"memory_prompts": _paired(d0_a2, a2_new)},
    }
    return out


def run(with_replay: bool = True) -> dict:
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    rep = replay() if with_replay or "replay" not in prev else prev["replay"]
    res = {"noise": noise(), "llm_side": llm_side(), "replay": rep}
    res["gpt_cells"] = gpt_cells(rep)
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys
    r = run(with_replay="--no-replay" not in sys.argv)
    n = r["noise"]
    print(f"[noise] name value old {n['name_value_old']:+.4f} -> new {n['name_value_new']:+.4f} "
          f"(change {n['change_in_name_value']:+.4f}, s.e. {n['se_change_in_name_value']:.4f}); "
          f"A1 {n['a1_change']:+.4f} (s.e. {n['se_a1_change']:.4f}), A2 {n['a2_change']:+.4f} (s.e. {n['se_a2_change']:.4f})")
    for run_, v in r["llm_side"].items():
        print(f"[llm] {run_}: precision {v['precision']:.3f}; anchor percentile of named genes "
              + ", ".join(f"{f} {x:.2f}" for f, x in v["anchor_percentile"].items()))
    for run_, v in r["replay"].items():
        print(f"[replay] {run_}: verified {v['verified']}; named in ML top {v['named_in_ml_top']:.2f}; "
              + "; ".join(f"{k} {e['mean']:+.4f} (round 1 {e['round1_mean']:+.4f})" for k, e in v["effects_vs_own"].items()))
    g = r["gpt_cells"]
    if g:
        fmt = lambda v: f"{v['delta']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['wins']}/{v['ties']}/{v['losses']}"
        print("[gpt] means:", {k: (round(v, 4) if v is not None else None) for k, v in g["means"].items()})
        print(f"[gpt] drift: old A1 now - Aug {fmt(g['drift']['old_a1_now_minus_aug'])}; "
              f"A2 (old memory) now - Aug {fmt(g['drift']['a2_old_memory_now_minus_aug'])}")
        print(f"[gpt] name value: Aug {g['name_value']['aug']:+.4f}; D0 now {fmt(g['name_value']['d0_now'])}; "
              f"reported {fmt(g['name_value']['new'])}")
        a = g["a1"]
        print(f"[gpt] A1: old condition - new {fmt(a['total_old_condition_minus_new'])}; anchors {fmt(a['anchors'])}; "
              f"memory/prompts {fmt(a['memory_prompts']) if a['memory_prompts'] else 'pending (D2 ' + str(len(a['d2_screens_done'])) + '/9)'}; "
              f"weights (replay) {a['weights_replay']:+.4f}; remainder {a['remainder']}")
        print(f"[gpt] A2: memory/prompts {fmt(g['a2']['memory_prompts'])}")
    print(f"saved -> {OUT}")
