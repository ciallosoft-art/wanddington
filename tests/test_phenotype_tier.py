"""A registered phenotype's single-gene tier (phenotype.register --tier; tables.tier_tables).

non_essential (the default, the paper's headline tier) removes the CEGv2 core-essential genes from the pool;
full_pool keeps them, looks their gene-intrinsic features up in the full-pool tables (the non-essential
tables contain none of them, so they used to get all-zero features) and trains the prior on those tables.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


import waddington_select.phenotype as P
from waddington_select.arms import waddington_c_arm as C
from waddington_select.features import load_training_frame
from waddington_select.tables import (TIER_FULL_POOL, TIER_NON_ESSENTIAL, core_essential_genes,
                                      tier_tables)

CORE = ["ABCE1", "ACTB"]           # CEGv2 genes with DepMap values in the full-pool table
OTHER = ["STAT1", "JAK1", "IFNGR1"]


def _no_anchor_features(genes, anchors):
    return {col: {} for col in P.ANCHOR_COLS}


class PhenotypeTierTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        user_dir = Path(tmp.name)
        for patch in (mock.patch.object(P, "USER_DIR", user_dir),
                      mock.patch.object(P, "REGISTRY_PATH", user_dir / "registry.json"),
                      mock.patch.object(P, "_anchor_features", _no_anchor_features)):
            patch.start()
            self.addCleanup(patch.stop)

    def register(self, name, tier=None):
        kwargs = {} if tier is None else {"tier": tier}
        return P.register(name, CORE + OTHER, ["IFNG"], "task", "measurement", **kwargs)

    def features(self, name):
        return P.features_frame(name).set_index("gene")

    def test_ceg_list_ships_with_the_package(self):
        ceg = core_essential_genes()
        self.assertEqual(len(ceg), 684)
        self.assertTrue(set(CORE) <= ceg)
        self.assertFalse(set(OTHER) & ceg)

    def test_default_tier_removes_core_essential_genes(self):
        res = self.register("NonEss")
        self.assertEqual(res["tier"], TIER_NON_ESSENTIAL)
        self.assertEqual(res["n_genes"], len(OTHER))
        self.assertEqual((res["n_core_essential_in_upload"], res["n_core_essential_removed"]), (2, 2))
        self.assertEqual(sorted(self.features("NonEss").index), sorted(OTHER))
        self.assertEqual(json.loads(P.REGISTRY_PATH.read_text())["NonEss"]["tier"], TIER_NON_ESSENTIAL)

    def test_full_pool_keeps_them_with_their_real_features(self):
        res = self.register("Full", TIER_FULL_POOL)
        self.assertEqual(res["n_genes"], len(CORE + OTHER))
        self.assertEqual((res["n_core_essential_in_upload"], res["n_core_essential_removed"]), (2, 0))
        self.assertEqual(res["n_genes_with_known_features"], len(CORE + OTHER))
        f = self.features("Full")
        self.assertAlmostEqual(f.loc["ABCE1", "depmap_frac_ess"], 1.0)
        self.assertGreater(f.loc["ACTB", "depmap_frac_ess"], 0.5)

    def test_prior_reads_the_tier_tables(self):
        self.register("NonEss")
        self.register("Full", TIER_FULL_POOL)
        csv, extra = C._get_feature_config("Full")
        self.assertEqual(csv, tier_tables(TIER_FULL_POOL)[1])
        self.assertEqual(extra, C._feature_config_by_policy("Full")[1])  # the DepMap policy is unchanged
        self.assertEqual(C._get_feature_config("NonEss")[0], tier_tables(TIER_NON_ESSENTIAL)[1])
        # the full-pool frame holds the nine screens' core-essential genes and this phenotype's pool
        frame = load_training_frame(csv, "Full")
        self.assertTrue(set(CORE) <= set(frame.loc[frame["dataset"] != "Full", "gene"].str.upper()))
        self.assertEqual(sorted(frame.loc[frame["dataset"] == "Full", "gene"]), sorted(CORE + OTHER))

    def test_c_arm_ranks_a_full_pool_phenotype(self):
        self.register("Full", TIER_FULL_POOL)
        arm = C.WaddingtonCArm("Full", 32)  # builds the ML branch; no LLM call until select()
        scores = arm._online.all_scores()
        self.assertEqual(sorted(scores), sorted(CORE + OTHER))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))
        self.assertEqual(sorted(arm._llm._gene_set), sorted(CORE + OTHER))

    def test_recommendations_read_the_tiers_reported_memory(self):
        from waddington_select.analysis.sources import REPORTED_RUN_DIR, REPORTED_TIER_A_RUN
        from waddington_select.suggest import deployment_memory_path
        self.register("NonEss")
        self.register("Full", TIER_FULL_POOL)
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("WADDINGTON_MEMORY_PATH", None)
            self.assertEqual(deployment_memory_path("IFNG"), REPORTED_RUN_DIR / "experience_memory.json")
            self.assertEqual(deployment_memory_path("NonEss"), REPORTED_RUN_DIR / "experience_memory.json")
            full = deployment_memory_path("Full")
            self.assertEqual(full, REPORTED_TIER_A_RUN.parent / "experience_memory.json")
            self.assertTrue(full.exists())
            arm = C.WaddingtonCArm("Full", 32, memory_path=full)
            self.assertEqual(len(arm._llm._memory), 4)  # the top-4 relevant entries, not an empty memory
            env["WADDINGTON_MEMORY_PATH"] = "/x/memory.json"
            self.assertEqual(deployment_memory_path("Full"), Path("/x/memory.json"))

    def test_benchmark_screens_are_unchanged(self):
        for ds in ("IFNG", "Steinhart", "Replogle_K562_essential"):
            self.assertEqual(C._get_feature_config(ds), C._feature_config_by_policy(ds))

    def test_bad_tier_and_all_essential_pool_are_rejected(self):
        with self.assertRaises(SystemExit):
            self.register("Bad", "easier")
        with self.assertRaises(SystemExit):
            P.register("OnlyCore", CORE, ["IFNG"], "task", "measurement")


if __name__ == "__main__":
    unittest.main()
