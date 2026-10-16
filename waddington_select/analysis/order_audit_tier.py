"""
order_audit_tier.py — does the reported system, or any baseline, depend on the stored row order of the target
screen on a given tier's tables? (2026-10-16; the audit of 2026-10-03, order_audit.py, used the headline tables.)

The single-gene tables store Scharenberg22's, K562-Essential's and K562-GWPS's hits first (IL2 partly), so a
method that breaks score ties by row order reads labels. Everything is run twice, with the target screen's rows
as stored and in alphabetical order (training rows untouched), and compared gene for gene. No LLM calls:

  c_arm_replay   the reported C-arm's own run (the tier's 20-replicate run), replayed with the recorded LLM
                 names under each order; each round's replayed batch is compared with the recorded one gene
                 for gene and in order, as a set, and by hits (first-order: the replay stays on the recorded
                 trajectory). The batch order is what the LLM's next prompt shows (its history lists
                 each round's hits and non-hits in batch order); the prompt has no other input that depends on
                 row order (task, memory, alphabetically sorted list of tested genes), so identical batches
                 mean identical prompts.
  padded_llm     the padding the padded LLM baseline draws from its static ranking when the LLM names too few
                 genes, recomputed on its recorded run under each order (the C-arm's named-only vote ignores
                 padding); the stored-order recomputation must reproduce the recorded padding
  baselines      static ranker, online ML, coreset and the C-arm's ML engine, full five-round campaigns
                 (static ranker and coreset read the table directly; a pandas shim reorders it for them)

The tier's tables must be in the environment before import (experiments/21_tier_offline.sh, TIER=...).

    python -m waddington_select.analysis.order_audit_tier --run RUN.json --out OUT.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

import waddington_select.features as F
from ..arms import llm_reasoning_arm, online_adaptive_arm, static_ranker_arm, waddington_c_arm
from ..arms.online_adaptive_arm import OnlineAdaptiveArm
from ..arms.waddington_c_arm import _drop_anchor_default, _get_feature_config
from ..oracle import BATCH_SIZES, DatasetOracle
from .final_system_stats import BENCH
from .replay_attribution import W_LLM, W_ML

from ..arms import coreset_arm

MODULES = [F, llm_reasoning_arm, online_adaptive_arm, static_ranker_arm, waddington_c_arm]
ORIG = F.load_training_frame
_CURRENT: list[str] = []  # the screen whose rows the alphabetical mode reorders


class _PandasShim:
    """pandas for the two baselines that read the table directly (static ranker, coreset): read_csv returns
    the table with the current target screen's rows in alphabetical order. The headline audit of 2026-10-03
    reordered only what goes through load_training_frame, so it never reached these two."""

    def __getattr__(self, name):
        return getattr(pd, name)

    def read_csv(self, *a, **k):
        df = pd.read_csv(*a, **k)
        if _CURRENT and "dataset" in df.columns:
            ds = _CURRENT[0]
            tgt = df["dataset"] == ds
            key = df.loc[tgt, "gene"].astype(str).str.strip().str.upper()
            df = pd.concat([df[~tgt], df[tgt].iloc[np.argsort(key.to_numpy(), kind="stable")]], ignore_index=True)
        return df


def _alpha(csv, dataset):
    df = ORIG(csv, dataset)
    tgt = df["dataset"] == dataset
    key = df.loc[tgt, "gene"].astype(str).str.strip().str.upper()
    return pd.concat([df[~tgt], df[tgt].iloc[np.argsort(key.to_numpy(), kind="stable")]], ignore_index=True)


def _mode(mode: str, ds: str | None = None) -> None:
    for m in MODULES:
        if hasattr(m, "load_training_frame"):
            m.load_training_frame = ORIG if mode == "stored" else _alpha
    for m in (static_ranker_arm, coreset_arm):
        m.pd = pd if mode == "stored" else _PandasShim()
    _CURRENT[:] = [] if mode == "stored" or ds is None else [ds]


def _c_online(ds: str) -> OnlineAdaptiveArm:
    csv, extra = _get_feature_config(ds)
    return OnlineAdaptiveArm(ds, BATCH_SIZES[ds], training_csv=csv, extra_feature_cols=extra,
                             drop_anchor_feats=_drop_anchor_default())


def _c_replay(base: OnlineAdaptiveArm, oracle: DatasetOracle, rec: dict, bs: int) -> list[dict]:
    """The recorded C-arm trajectory, its fused ranking recomputed from the recorded LLM names (as
    replay_attribution), comparing each round's replayed batch with the recorded one: exactly, as a set, and
    by hits. The replay stays on the recorded trajectory, so a hit difference is the first-order effect."""
    online = copy.deepcopy(base)
    selected: set[str] = set()
    out = []
    for r, recorded in enumerate(rec["selected_genes_per_round"]):
        ml = online.all_scores()
        named = set(rec["llm_named_proposals_per_round"][r])
        combined = {g: W_ML * ml.get(g, 0.0) + W_LLM * (1.0 if g in named else 0.0)
                    for g in online._genes if g not in selected}
        replayed = sorted(combined, key=combined.__getitem__, reverse=True)[:bs]
        rev = oracle.reveal(list(recorded))
        out.append({"exact": replayed == list(recorded), "same_set": set(replayed) == set(recorded),
                    "hits_recorded": int(sum(rev.values())), "hits_replayed": int(sum(oracle.reveal(replayed).values()))})
        selected |= set(recorded)
        if r < len(rec["selected_genes_per_round"]) - 1:
            online.update(r, rev)
    return out


def _padding(ranking: list[str], rec: dict) -> list[list[str]]:
    """The padding the padded LLM baseline would draw from `ranking`, round by round, on the recorded run."""
    selected: set[str] = set()
    out = []
    for r, sel in enumerate(rec["selected_genes_per_round"]):
        n = len(rec["llm_static_padding_proposals_per_round"][r])
        named = set(rec["llm_named_proposals_per_round"][r])
        pad = []
        for g in ranking:
            if len(pad) >= n:
                break
            if g not in selected and g not in named:
                pad.append(g)
        out.append(pad)
        selected |= set(sel)
    return out


def audit(run_file: Path) -> dict:
    from ..run_sequential import _run_one_repeat
    data = json.loads(run_file.read_text())
    res = {"source_run": str(run_file), "screens": {}}
    for ds in BENCH:
        oracle = DatasetOracle(ds)
        total = oracle.total_hits
        row = {}
        # 1. the reported C-arm's recorded trajectory under each order
        runs = data[ds]["waddington_c"]
        for mode in ("stored", "alpha"):
            _mode(mode, ds)
            base = _c_online(ds)
            rounds = [x for rec in runs for x in _c_replay(base, oracle, rec, BATCH_SIZES[ds])]
            row[f"c_{mode}"] = {
                "rounds": len(rounds), "exact": sum(x["exact"] for x in rounds),
                "same_set": sum(x["same_set"] for x in rounds),
                "hit_ratio_change": sum(x["hits_replayed"] - x["hits_recorded"] for x in rounds) / (len(runs) * total)}
        # 2. the padded LLM baseline's padding under each order (on its own recorded run)
        if "llm_reasoning" in data[ds]:
            pads = {}
            for mode in ("stored", "alpha"):
                _mode(mode, ds)
                ranking = llm_reasoning_arm.LLMReasoningArm(ds, BATCH_SIZES[ds])._static_ranking
                pads[mode] = [_padding(ranking, rec) for rec in data[ds]["llm_reasoning"]]
            recs = data[ds]["llm_reasoning"]
            reconstructed = all(pads["stored"][i][r] == list(rec["llm_static_padding_proposals_per_round"][r])
                                for i, rec in enumerate(recs) for r in range(len(rec["selected_genes_per_round"])))
            hits = {m: sum(sum(oracle.reveal(p).values()) for run in pads[m] for p in run if p) for m in pads}
            row["padded_llm"] = {"stored_reconstructs_recorded": reconstructed,
                                 "padded_genes": sum(len(p) for run in pads["stored"] for p in run),
                                 "hit_ratio_change": (hits["alpha"] - hits["stored"]) / (len(recs) * total)}
        # 3. deterministic baselines and the C-arm's ML engine, whole campaigns
        for arm in ("static_ranker", "online_adaptive", "coreset", "ml_engine_lightgbm"):
            sel = {}
            for mode in ("stored", "alpha"):
                _mode(mode, ds)
                r = _run_one_repeat(ds, arm, 0, oracle, 5, "local")
                sel[mode] = (r["selected_genes_per_round"], r["hit_ratio_per_round"][-1])
            row[arm] = {"same_order": sel["stored"][0] == sel["alpha"][0],
                        "same_sets": [set(a) for a in sel["stored"][0]] == [set(b) for b in sel["alpha"][0]],
                        "hit_at_r5": [sel["stored"][1], sel["alpha"][1]]}
        _mode("stored")
        res["screens"][ds] = row
        print(ds, json.dumps(row), flush=True)
    s = list(res["screens"].values())
    res["summary"] = {
        "c_rounds": sum(r["c_alpha"]["rounds"] for r in s),
        "c_exact_stored": sum(r["c_stored"]["exact"] for r in s),
        "c_exact_alpha": sum(r["c_alpha"]["exact"] for r in s),
        "c_same_set_alpha": sum(r["c_alpha"]["same_set"] for r in s),
        "c_mean_hit_ratio_change_alpha": sum(r["c_alpha"]["hit_ratio_change"] for r in s) / len(s),
        "padded_llm_mean_hit_ratio_change_alpha": (sum(r["padded_llm"]["hit_ratio_change"] for r in s if "padded_llm" in r)
                                                   / max(1, sum("padded_llm" in r for r in s))),
        "baselines_same_hit_at_r5": {a: all(abs(r[a]["hit_at_r5"][0] - r[a]["hit_at_r5"][1]) < 1e-12 for r in s)
                                     for a in ("static_ranker", "online_adaptive", "coreset", "ml_engine_lightgbm")},
        "baselines_same_sets": {a: all(r[a]["same_sets"] for r in s)
                                for a in ("static_ranker", "online_adaptive", "coreset", "ml_engine_lightgbm")},
    }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r = audit(a.run)
    a.out.write_text(json.dumps(r, indent=1))
    print("summary:", json.dumps(r["summary"]))
