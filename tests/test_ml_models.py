"""Alternative ML engines: interface contract, weighting, determinism, and the LightGBM reference path."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from waddington_select.ml_models import ENGINES, engine_names, is_stochastic, make_engine

_TABULAR = [n for n, c in ENGINES.items()
            if not getattr(c, "needs_graph", False) and not n.startswith("genept")]


def _toy(n=1200, d=6, seed=0):
    """Separable-ish binary data with a ~5% positive rate, like a screen."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    logit = 2.0 * X[:, 0] - 1.0 * X[:, 1] - 3.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    genes = [f"G{i}" for i in range(n)]
    groups = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    return X, y, genes, groups


class EngineContractTests(unittest.TestCase):
    def test_every_engine_scores_in_unit_interval_and_learns(self):
        X, y, genes, groups = _toy()
        pos_scale = (y == 0).sum() / (y == 1).sum()
        for name in _TABULAR:
            with self.subTest(engine=name):
                e = make_engine(name, seed=1).fit(X, y, np.ones(len(y)), pos_scale, genes=genes, groups=groups)
                s = e.score(X, genes)
                self.assertEqual(s.shape, (len(y),))
                self.assertTrue(np.all(np.isfinite(s)) and s.min() >= 0.0 and s.max() <= 1.0)
                # learned something: positives score higher than negatives on average
                self.assertGreater(s[y == 1].mean(), s[y == 0].mean())

    def test_protocol_weights_change_the_fit(self):
        X, y, genes, groups = _toy()
        w1 = np.ones(len(y))
        w10 = w1.copy()
        w10[:100] = 10.0  # like revealed in-experiment rows
        for name in ("logreg", "lambdarank", "svm_linear", "svm_rbf"):
            with self.subTest(engine=name):
                a = make_engine(name).fit(X, y, w1, 5.0, genes=genes, groups=groups).score(X, genes)
                b = make_engine(name).fit(X, y, w10, 5.0, genes=genes, groups=groups).score(X, genes)
                self.assertFalse(np.allclose(a, b))

    def test_determinism_and_seed_dependence(self):
        X, y, genes, groups = _toy()
        for name in _TABULAR:
            with self.subTest(engine=name):
                a = make_engine(name, seed=3).fit(X, y, np.ones(len(y)), 5.0, genes, groups).score(X, genes)
                b = make_engine(name, seed=3).fit(X, y, np.ones(len(y)), 5.0, genes, groups).score(X, genes)
                np.testing.assert_array_equal(a, b)
                if is_stochastic(name):
                    c = make_engine(name, seed=4).fit(X, y, np.ones(len(y)), 5.0, genes, groups).score(X, genes)
                    self.assertFalse(np.array_equal(a, c))

    def test_registry(self):
        self.assertEqual(engine_names()[0], "lightgbm")
        self.assertFalse(is_stochastic("lightgbm"))
        with self.assertRaises(ValueError):
            make_engine("no_such_engine")


class MLEngineArmTests(unittest.TestCase):
    """On a real (small) screen: the lightgbm engine IS the C-arm's ML, and static never retrains."""

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))

    def test_lightgbm_engine_equals_the_c_arm_ranker(self):
        from waddington_select.arms.ml_engine_arm import MLEngineArm
        from waddington_select.arms.online_adaptive_arm import OnlineAdaptiveArm
        from waddington_select.arms.waddington_c_arm import _drop_anchor_default, _get_feature_config
        ds, bs = "Scharenberg22", 32
        csv, extra = _get_feature_config(ds)
        ref = OnlineAdaptiveArm(ds, bs, training_csv=csv, extra_feature_cols=extra,
                                drop_anchor_feats=_drop_anchor_default())
        arm = MLEngineArm(ds, bs, model="lightgbm")
        self.assertEqual(arm.name, "ml_engine_lightgbm")
        self.assertEqual(arm._available_feats, ref._available_feats)
        self.assertEqual(arm.all_scores(), ref.all_scores())

    def test_static_variant_never_retrains(self):
        from waddington_select.arms.ml_engine_arm import MLEngineArm
        arm = MLEngineArm("Scharenberg22", 32, model="logreg", online=False)
        before = list(arm._ranking)
        batch = arm.select(0, {})
        arm.update(0, {g: (i % 3 == 0) for i, g in enumerate(batch)})
        self.assertEqual(arm._ranking, before)
        self.assertTrue(set(batch).isdisjoint(arm.select(1, {})))

    def test_arm_names_parse(self):
        from waddington_select.arms.ml_engine_arm import parse_arm_name
        self.assertEqual(parse_arm_name("ml_engine_rf"), ("rf", True))
        self.assertEqual(parse_arm_name("ml_static_logreg"), ("logreg", False))
        self.assertIsNone(parse_arm_name("online_adaptive"))


class GraphEngineTests(unittest.TestCase):
    """RWR and GraphSAGE on the real STRING graph (cached under workspace/cache/)."""

    @classmethod
    def setUpClass(cls):
        from waddington_select.string_graph import string_graph
        cls.genes, cls.A = string_graph()

    def test_rwr_ranks_seed_neighbours_first_and_is_deterministic(self):
        rng = np.random.default_rng(0)
        seeds = list(rng.choice(len(self.genes), size=5, replace=False))
        nbrs = set(np.concatenate([self.A[i].indices for i in seeds])) - set(seeds)
        train_genes = [self.genes[i] for i in seeds]
        e = make_engine("rwr").fit(None, np.ones(5, int), np.ones(5), 1.0, genes=train_genes)
        rest = [i for i in range(len(self.genes)) if i not in seeds]
        cand = [self.genes[i] for i in rest]
        s = e.score(None, cand)
        is_nbr = np.array([i in nbrs for i in rest])
        self.assertGreater(s[is_nbr].mean(), s[~is_nbr].mean())
        s2 = make_engine("rwr").fit(None, np.ones(5, int), np.ones(5), 1.0, genes=train_genes).score(None, cand)
        np.testing.assert_array_equal(s, s2)
        absent = e.score(None, ["NOT_A_GENE_XYZ", cand[0]])
        self.assertEqual(absent[0], 0.0)  # absent from STRING ranks last

    def test_gnn_learns_and_is_seed_reproducible(self):
        from waddington_select.ml_models import GraphSAGEEngine

        class Short(GraphSAGEEngine):
            EPOCHS = 30

        rng = np.random.default_rng(1)
        genes = list(rng.choice(self.genes, size=3000, replace=False)) + ["ONLY_IN_TRAINING_1"]
        X = rng.normal(size=(len(genes), 4)).astype(np.float32)
        y = (X[:, 0] > 1.2).astype(int)
        fit = lambda seed: Short(seed=seed, device="cpu").fit(X, y, np.ones(len(y)), 5.0, genes=genes)
        a = fit(0).score(X, genes)
        self.assertGreater(a[y == 1].mean(), a[y == 0].mean())
        np.testing.assert_array_equal(a, fit(0).score(X, genes))
        self.assertFalse(np.array_equal(a, fit(1).score(X, genes)))
        unseen = fit(0).score(X[:1], ["NEVER_SEEN_GENE"])  # isolated node scored from its features alone
        self.assertTrue(0.0 <= unseen[0] <= 1.0)


class GenePTEngineTests(unittest.TestCase):
    """GenePT engines on real gene symbols; skipped when the (gitignored) embeddings are absent."""

    @classmethod
    def setUpClass(cls):
        from waddington_select.ml_models import GENEPT_DIR, genept_embedding
        if not any(GENEPT_DIR.rglob("*.pickle")) and not (GENEPT_DIR / "genept_pca64.npz").exists():
            raise unittest.SkipTest("GenePT embeddings not downloaded")
        cls.index, cls.Z = genept_embedding()

    def test_contract_learning_and_determinism(self):
        rng = np.random.default_rng(0)
        genes = list(rng.choice(sorted(self.index), size=3000, replace=False)) + ["NOT_IN_GENEPT_1"]
        emb = np.vstack([self.Z[self.index[g]] if g in self.index else np.zeros(self.Z.shape[1])
                         for g in genes])
        y = (emb[:, 0] > np.quantile(emb[:, 0], 0.9)).astype(int)  # hits defined by the embedding
        X = rng.normal(size=(len(genes), 3))                       # uninformative tabular features
        for name in ("genept_lr", "genept_lgbm"):
            with self.subTest(engine=name):
                a = make_engine(name).fit(X, y, np.ones(len(y)), 9.0, genes=genes).score(X, genes)
                self.assertTrue(np.all((a >= 0) & (a <= 1)))
                self.assertGreater(a[y == 1].mean(), a[y == 0].mean())
                b = make_engine(name).fit(X, y, np.ones(len(y)), 9.0, genes=genes).score(X, genes)
                np.testing.assert_array_equal(a, b)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))

    def test_rules_run_reproducibly_and_thompson_orders_untested_genes(self):
        from waddington_select.arms.ml_engine_arm import ACQUISITIONS, AcquisitionArm, parse_acq_name
        ds, bs = "Scharenberg22", 32
        for a in ACQUISITIONS:
            with self.subTest(rule=a):
                arm = AcquisitionArm(ds, bs, acquisition=a, seed=0)
                b0 = arm.select(0, {})
                self.assertEqual(len(b0), bs)
                arm.update(0, {g: (i % 3 == 0) for i, g in enumerate(b0)})
                b1 = arm.select(1, {})
                self.assertTrue(set(b0).isdisjoint(b1))
                again = AcquisitionArm(ds, bs, acquisition=a, seed=0)
                self.assertEqual(again.select(0, {}), b0)
                if a == "thompson":  # after the update the ordering covers exactly the untested rows
                    # (row-wise: Scharenberg22's pool lists five genes twice)
                    self.assertEqual(sorted(arm._ranking), sorted(g for g in arm._genes if g not in set(b0)))
                self.assertEqual(parse_acq_name(f"ml_acq_{a}"), a)
        self.assertIsNone(parse_acq_name("ml_acq_nonsense"))


class CArmEngineTests(unittest.TestCase):
    """The full C-arm with a swapped engine: built without provider calls, default path untouched."""

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))
        self.enterContext(patch("waddington_select.llm_client.LLMClient.complete",
                                side_effect=AssertionError("provider calls forbidden")))

    def test_default_engine_is_the_reported_lightgbm(self):
        from waddington_select.arms.ml_engine_arm import MLEngineArm
        from waddington_select.arms.waddington_c_arm import WaddingtonCArm
        c = WaddingtonCArm("Scharenberg22", 32)
        self.assertEqual(c._online._model_name, "lightgbm")
        self.assertEqual(c._online.all_scores(), MLEngineArm("Scharenberg22", 32, model="lightgbm").all_scores())

    def test_swapped_engine_arms(self):
        from waddington_select import run_sequential as R
        self.assertIn("waddington_c__mlp", R._PAPER_ARMS)
        self.assertNotIn("waddington_c__rwr", R._PAPER_ARMS)  # rank scores cannot take the 0.2 bonus
        arm = R.make_arms("Scharenberg22", ["waddington_c__mlp"], local_seed=3)[0]
        self.assertEqual((arm.name, arm._online._model_name, arm._online._seed), ("waddington_c__mlp", "mlp", 3))


class LeakageTests(unittest.TestCase):
    """Every engine is fitted only on other screens' rows plus the target's REVEALED rows."""

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))

    def test_engines_never_see_unrevealed_target_labels(self):
        from waddington_select import ml_models
        from waddington_select.arms.ml_engine_arm import MLEngineArm
        seen = []

        class Spy:
            probabilistic, stochastic = True, False

            def __init__(self, seed=42):
                pass

            def fit(self, X, y, sample_weight, pos_scale, genes=None, groups=None):
                seen.append((list(groups), list(genes), np.asarray(sample_weight)))
                return self

            def score(self, X, genes=None):
                return np.linspace(1, 0, len(genes))

        ds = "Scharenberg22"
        with patch.dict(ml_models.ENGINES, {"spy": Spy}):
            arm = MLEngineArm(ds, 32, model="spy")
            groups, _, _ = seen[-1]
            self.assertNotIn(ds, groups)
            self.assertNotIn("__target__", groups)
            batch = arm.select(0, {})
            arm.update(0, {g: (i % 4 == 0) for i, g in enumerate(batch)})
            groups, genes, w = seen[-1]
            target_rows = [i for i, g in enumerate(groups) if g == "__target__"]
            self.assertNotIn(ds, groups)
            self.assertEqual(sorted(genes[i] for i in target_rows), sorted(batch))
            self.assertTrue(np.all(w[target_rows] == 10.0))


if __name__ == "__main__":
    unittest.main()
