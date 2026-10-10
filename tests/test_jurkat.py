"""The Jurkat held-out benchmark (docs/jurkat_protocol.md): integrity of the data and tables, no outcomes."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "workspace" / "data" / "horlbeck2018"
EVAL = REPO / "workspace" / "evaluation"
GT = DATA / "ground_truth_Horlbeck_Jurkat.csv"
TABLE = EVAL / "lgbm_training_data_horlbeck_jurkat_pairs_v3.csv"


@unittest.skipUnless(GT.exists(), "Jurkat benchmark not built")
class JurkatBenchmarkTests(unittest.TestCase):
    def test_pool_and_two_tailed_hits(self):
        gt = pd.read_csv(GT)
        hits = np.load(DATA / "topmovers_Horlbeck_Jurkat.npy")
        self.assertEqual((len(gt), len(hits)), (75078, 7508))
        lo, hi = np.quantile(gt["Score"], [0.05, 0.95])
        self.assertEqual(int(((gt["Score"] <= lo) | (gt["Score"] >= hi)).sum()), 7508)

    def test_prompt_changes_only_the_cell_line(self):
        from waddington_select.arms.llm_reasoning_arm import _load_task
        k, j = _load_task("Horlbeck"), _load_task("Horlbeck_Jurkat")
        self.assertEqual(k["Task"].replace("K562", "Jurkat"), j["Task"])
        self.assertEqual(k["Measurement"], j["Measurement"])

    def test_shared_pairs_keep_their_k562_interaction_features(self):
        k = pd.read_csv(EVAL / "pair_interaction_features.csv").set_index("gene")
        j = pd.read_csv(EVAL / "pair_interaction_features_horlbeck_jurkat.csv").set_index("gene")
        shared = j.index.intersection(k.index)
        self.assertGreater(len(shared), 70000)
        pd.testing.assert_frame_equal(j.loc[shared, k.columns], k.loc[shared])


@unittest.skipUnless(TABLE.exists(), "Jurkat pair tables not built")
class JurkatTableTests(unittest.TestCase):
    def test_training_half_holds_no_k562_pair_labels(self):
        df = pd.read_csv(TABLE, usecols=["gene", "dataset", "label"])
        self.assertNotIn("Horlbeck", set(df["dataset"]))
        jur = df[df["dataset"] == "Horlbeck_Jurkat"]
        self.assertEqual((len(jur), int(jur["label"].sum())), (75078, 7508))
        self.assertFalse(df[df["dataset"] != "Horlbeck_Jurkat"]["gene"].str.contains("|", regex=False).any())

    def test_oracle_serves_the_screen(self):
        env = dict(os.environ,
                   WADDINGTON_TRAINING_CSV=str(EVAL / "lgbm_training_data_horlbeck_jurkat_pairs.csv"),
                   WADDINGTON_TRAINING_CSV_V3=str(TABLE), WADDINGTON_PAIR_DATASETS="Horlbeck_Jurkat",
                   WADDINGTON_EXTRA_DATASETS="Horlbeck_Jurkat:32")
        code = ("from waddington_select.oracle import DatasetOracle; o = DatasetOracle('Horlbeck_Jurkat'); "
                "print(len(o.all_genes()), o.batch_size)")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                             check=True)
        self.assertEqual(out.stdout.split()[-2:], ["75078", "32"])


if __name__ == "__main__":
    unittest.main()
