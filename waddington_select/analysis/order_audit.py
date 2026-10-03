"""order_audit.py — does a method's result depend on the order of the target screen's candidate rows?

Order audit: does a method's result depend on the order of the target screen's candidate rows?

The training CSV lists some screens' hits first (Scharenberg22, K562-Essential, K562-GWPS; IL2 partly). A
method that breaks score ties by row order then reads labels. Each (arm, screen, replicate) is run twice
with the same seed: rows as stored, and the target screen's rows sorted alphabetically (label-free). The
other screens' rows (training data) are untouched, so a model trained on them is unchanged.

    python -m waddington_select.analysis.order_audit ARM1,ARM2 SCREEN1,SCREEN2 N_REPS OUT.json
"""
import json
import os
import sys
from pathlib import Path

for k in list(os.environ):
    if k.startswith("WADDINGTON_"):
        del os.environ[k]

import numpy as np
import pandas as pd

import waddington_select.features as F
from waddington_select.arms import (genedisco_arm, genedisco_faithful_arm, online_adaptive_arm,
                                    waddington_c_arm, waddington_c_feature_reasoning_arm)
from waddington_select.oracle import DatasetOracle
from waddington_select.run_sequential import _run_one_repeat

MODULES = [genedisco_arm, genedisco_faithful_arm, online_adaptive_arm, waddington_c_arm,
           waddington_c_feature_reasoning_arm]
ORIG = F.load_training_frame


def alpha_frame(csv_path, dataset_name):
    df = ORIG(csv_path, dataset_name)
    tgt = df["dataset"] == dataset_name
    key = df.loc[tgt, "gene"].astype(str).str.strip().str.upper()
    return pd.concat([df[~tgt], df[tgt].iloc[np.argsort(key.to_numpy(), kind="stable")]], ignore_index=True)


def set_mode(mode):
    for m in MODULES:
        m.load_training_frame = ORIG if mode == "orig" else alpha_frame


ARMS = sys.argv[1].split(",")
SCREENS = sys.argv[2].split(",")
REPS = int(sys.argv[3])
OUT = Path(sys.argv[4])
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for ds in SCREENS:
    oracle = DatasetOracle(ds)
    for arm in ARMS:
        for mode in ("orig", "alpha"):
            key = f"{ds}|{arm}|{mode}"
            if key in res:
                continue
            set_mode(mode)
            vals = []
            for rep in range(REPS):
                r = _run_one_repeat(ds, arm, rep, oracle, 5, "local")
                vals.append(r["hit_ratio_per_round"][-1])
            res[key] = vals
            OUT.write_text(json.dumps(res, indent=1))
            print(key, np.round(vals, 4), flush=True)
set_mode("orig")
