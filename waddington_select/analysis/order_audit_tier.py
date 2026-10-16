"""
order_audit_tier.py — does the reported system, or any baseline, depend on the stored row order of the target
screen on a given tier's tables? (2026-10-16; the audit of 2026-10-03, order_audit.py, used the headline tables.)

The single-gene tables store Scharenberg22's, K562-Essential's and K562-GWPS's hits first (IL2 partly), so a
method that breaks score ties by row order reads labels. Everything is run twice, with the target screen's rows
as stored and in alphabetical order (training rows untouched), and compared gene for gene. No LLM calls:

  c_arm_replay   the reported C-arm's own run (the tier's 20-replicate run), replayed with the recorded LLM
                 names under each order; a round passes if the replayed batch equals the recorded batch gene
                 for gene and in order. The batch order is what the LLM's next prompt shows (its history lists
                 each round's hits and non-hits in batch order); the prompt has no other input that depends on
                 row order (task, memory, alphabetically sorted list of tested genes), so identical batches
                 mean identical prompts.
  llm_padding    the LLM arm's static ranking, which pads a batch when the LLM names too few genes (the padded
                 LLM baseline uses the padding; the C-arm's named-only vote ignores it)
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
from .replay_attribution import _replay_run

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


def audit(run_file: Path) -> dict:
    from ..run_sequential import _run_one_repeat
    data = json.loads(run_file.read_text())
    res = {"source_run": str(run_file), "screens": {}}
    for ds in BENCH:
        oracle = DatasetOracle(ds)
        row = {}
        # 1. the C-arm's recorded trajectory, replayed under each order
        runs = data[ds]["waddington_c"]
        for mode in ("stored", "alpha"):
            _mode(mode, ds)
            base = _c_online(ds)
            rounds = [rr for rec in runs for rr in _replay_run(copy.deepcopy(base), oracle, rec, BATCH_SIZES[ds])]
            row[f"c_replay_{mode}"] = {"verified": sum(r["verified"] for r in rounds), "rounds": len(rounds)}
        # 2. the LLM arm's padding order
        rank = {}
        for mode in ("stored", "alpha"):
            _mode(mode, ds)
            rank[mode] = llm_reasoning_arm.LLMReasoningArm(ds, BATCH_SIZES[ds])._static_ranking
        k = 5 * BATCH_SIZES[ds]
        row["llm_padding"] = {"top_5B_identical": rank["stored"][:k] == rank["alpha"][:k],
                              "full_identical": rank["stored"] == rank["alpha"]}
        # 3. deterministic baselines and the C-arm's ML engine, whole campaigns
        for arm in ("static_ranker", "online_adaptive", "coreset", "ml_engine_lightgbm"):
            sel = {}
            for mode in ("stored", "alpha"):
                _mode(mode, ds)
                r = _run_one_repeat(ds, arm, 0, oracle, 5, "local")
                sel[mode] = (r["selected_genes_per_round"], r["hit_ratio_per_round"][-1])
            row[arm] = {"selections_identical": sel["stored"][0] == sel["alpha"][0],
                        "hit_at_r5": [sel["stored"][1], sel["alpha"][1]]}
        _mode("stored")
        res["screens"][ds] = row
        print(ds, json.dumps(row), flush=True)
    s = res["screens"].values()
    res["summary"] = {
        "c_replay_verified_alpha": sum(r["c_replay_alpha"]["verified"] for r in s),
        "c_replay_rounds": sum(r["c_replay_alpha"]["rounds"] for r in s),
        "c_replay_verified_stored": sum(r["c_replay_stored"]["verified"] for r in s),
        "llm_padding_top_identical": all(r["llm_padding"]["top_5B_identical"] for r in s),
        "baselines_identical": {a: all(r[a]["selections_identical"] for r in s)
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
