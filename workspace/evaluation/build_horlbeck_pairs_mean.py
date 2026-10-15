#!/usr/bin/env python3
"""
build_horlbeck_pairs_mean.py — the Horlbeck pair tables WITHOUT the interaction features.

The reported combinatorial run (experiment 09, 2026-09-10) read lgbm_training_data_horlbeck_pairs(_v3).csv
before experiment 12 added the eight pair_* interaction columns to those same files. This rebuilds the
pair-mean tables by dropping those columns, into separate files, so the reported C-arm's setting can be run
again (docs/tier_extension_plan.md, deviation log of 2026-10-16).

Check (2026-10-16): on the rebuilt tables the deterministic arms of experiment 09's calibration (static
ranker, online ML, coreset) select the same pairs as the September run in every round (58 / 93 / 72 hits).
The bytes differ from the September files (CSV float formatting), the content the arms read does not.

    python3 workspace/evaluation/build_horlbeck_pairs_mean.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

for v in ("", "_v3"):
    src = HERE / f"lgbm_training_data_horlbeck_pairs{v}.csv"
    out = HERE / f"lgbm_training_data_horlbeck_pairs_mean{v}.csv"
    df = pd.read_csv(src)
    cols = [c for c in df.columns if c.startswith("pair_")]
    df.drop(columns=cols).to_csv(out, index=False)
    print(f"{out.name}: {len(df)} rows, dropped {len(cols)} interaction columns")
