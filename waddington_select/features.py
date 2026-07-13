"""
features.py — the training frame the arms see.

The arms read one table and split it into `df[dataset == X]` (the candidate pool) and
`df[dataset != X]` (leave-one-out training data). To make a *registered user phenotype* rankable we
append its feature rows — but ONLY when it is the target dataset.

That guard matters: a user phenotype's rows are all `label=0` (its hits are unknown until the
scientist uploads results). If they were always concatenated, they would silently become negative
training data for the benchmark datasets and corrupt the benchmark. Appending only when the user
phenotype is the target gives exactly the right semantics:

    target = a benchmark dataset  → base table only            (benchmark unchanged)
    target = a user phenotype     → base table + its pool rows → LOO training = the 9 benchmarks
                                                                 (zero-shot transfer for round 1)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .phenotype import features_frame, is_user_phenotype


def load_training_frame(csv_path: str | Path, dataset_name: str) -> pd.DataFrame:
    """Base training data, plus `dataset_name`'s feature rows if it is a registered user phenotype."""
    df = pd.read_csv(csv_path)
    if not is_user_phenotype(dataset_name):
        return df
    user = features_frame(dataset_name)
    if user is None or user.empty:
        return df
    # Align to whichever feature set this CSV uses (v1 has no DepMap columns; v3 does).
    user = user.reindex(columns=df.columns, fill_value=0.0)
    return pd.concat([df, user], ignore_index=True)
