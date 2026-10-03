"""Alternative ML engines: interface contract, weighting, determinism, and the LightGBM reference path."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from waddington_select.ml_models import ENGINES, engine_names, is_stochastic, make_engine

_TABULAR = [n for n, c in ENGINES.items() if not getattr(c, "needs_scores", False)
            and not getattr(c, "needs_graph", False) and not n.startswith("genept")]


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


class ScoreTargetTests(unittest.TestCase):
    """Phase 2 item 1: score targets are hit-aligned, read only training screens, and keep revealed rows binary."""

    def test_transform_follows_each_screens_hits(self):
        from waddington_select.features import load_training_frame
        from waddington_select.screen_scores import aligned_target
        from waddington_select.tables import TRAINING_DATA_CSV
        df = load_training_frame(TRAINING_DATA_CSV, "IFNG")
        df["gene"] = df["gene"].str.strip().str.upper()
        want = {"IFNG": "|s|", "Sanchez21_down": "-s", "Steinhart": "s", "Scharenberg22": "s"}
        for ds, tf in want.items():
            with self.subTest(screen=ds):
                rows = df[df["dataset"] == ds]
                t, info = aligned_target(ds, rows["gene"], rows["label"])
                self.assertEqual(info["transform"], tf)
                self.assertEqual(info["coverage"], 1.0)
                self.assertTrue(t.min() >= 0.0 and t.max() <= 1.0)
                # every hit sits above every non-hit: the target reproduces the screen's threshold
                self.assertGreater(t[rows["label"].values == 1].min(), t[rows["label"].values == 0].max())
        rows = df[df["dataset"] == "Sanchez21"]  # hit list unrelated to the shipped score
        t, info = aligned_target("Sanchez21", rows["gene"], rows["label"])
        self.assertIsNone(t)
        self.assertLess(info["auc"], 0.9)

    def test_target_and_revealed_rows_use_labels_only(self):
        from waddington_select import screen_scores
        y = np.array([1, 0, 0, 1, 0, 1, 0])
        groups = ["Steinhart"] * 4 + ["__target__"] * 3
        genes = ["A1BG", "A1CF", "A2M", "A2ML1", "X1", "X2", "X3"]
        read = []
        real = screen_scores.load_scores.__wrapped__

        def spy(ds):
            read.append(ds)
            return real(ds)

        with patch.object(screen_scores, "load_scores", spy):
            t, info = screen_scores.score_targets(y, groups, genes)
        self.assertEqual(read, ["Steinhart"])  # never the target screen
        np.testing.assert_allclose(t[4:], [info["imputed"][0], info["imputed"][1], info["imputed"][0]])
        self.assertEqual(info["n_imputed"], 3)
        h, _ = screen_scores.score_targets(y, groups, genes, revealed="hard")
        np.testing.assert_array_equal(h[4:], [0.0, 1.0, 0.0])
        np.testing.assert_array_equal(h[:4], t[:4])  # training-screen rows are unchanged

    def test_score_engine_fits_on_real_screens(self):
        from waddington_select.arms.ml_engine_arm import MLEngineArm
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        with patch.dict(os.environ, env, clear=True):
            arm = MLEngineArm("Scharenberg22", 32, model="lgbm_score")
        info = arm._model.target_info
        self.assertNotIn("Scharenberg22", info["screens"])
        self.assertLess(info["screens"]["Sanchez21"]["auc"], 0.9)  # its rows fall back to the label
        self.assertGreaterEqual(info["n_imputed"], 17807)
        s = list(arm.all_scores().values())
        self.assertTrue(min(s) >= 0.0 and max(s) <= 1.0)


class DescriptorEngineTests(unittest.TestCase):
    """Phase 2 item 2: descriptor engines follow the DepMap policy and reduce to LightGBM where it is off."""

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))

    def test_descriptors_are_label_free_lookups(self):
        from waddington_select import descriptors as D
        for name in ("depmap_pca", "string_spectral"):
            with self.subTest(descriptor=name):
                r = D.rows(name, ["TP53", "tp53", "NOT_A_GENE"])
                self.assertEqual(r.shape, (3, D.DIM))
                np.testing.assert_array_equal(r[0], r[1])  # case-insensitive
                self.assertTrue(np.isnan(r[2]).all())

    def test_depmap_descriptor_follows_policy(self):
        from waddington_select.arms.ml_engine_arm import MLEngineArm
        off = MLEngineArm("Replogle_K562_essential", 32, model="lgbm_depmap_pca", online=False)
        ref = MLEngineArm("Replogle_K562_essential", 32, model="lightgbm", online=False)
        self.assertFalse(off._model.depmap_allowed)
        self.assertEqual(off._ranking, ref._ranking)  # policy "none": exactly the reported engine
        on = MLEngineArm("Scharenberg22", 32, model="lgbm_depmap_pca", online=False)
        self.assertTrue(on._model.depmap_allowed)
        self.assertEqual(on._model._m.n_features_in_, len(on._available_feats) + 32)


class GeneDiscoFaithfulTests(unittest.TestCase):
    """Phase 2 item 3: GeneDisco's Bayesian MLP, the faithful GeneDisco arms, and BNN acquisition arms."""

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))

    def test_bnn_is_seed_reproducible_and_consistent(self):
        from waddington_select import bnn
        X, y, _, _ = _toy(n=400)
        a = bnn.mc_predict(bnn.fit(X, y.astype(float), seed=1, epochs=5), X, seed=7)
        b = bnn.mc_predict(bnn.fit(X, y.astype(float), seed=1, epochs=5), X, seed=7)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (bnn.MC_SAMPLES, 400))
        self.assertGreater(a.std(0).mean(), 0.0)  # dropout makes the samples differ

    def test_our_features_match_the_c_arm_engine(self):
        from waddington_select.arms.genedisco_faithful_arm import our_features
        from waddington_select.arms.ml_engine_arm import MLEngineArm
        for ds in ("Scharenberg22", "Steinhart", "Replogle_K562_gwps"):
            with self.subTest(screen=ds):
                _, feats, _ = our_features(ds)
                self.assertEqual(feats, MLEngineArm(ds, 32, model="logreg", online=False)._available_feats)

    def test_faithful_arms_run_reproducibly_on_revealed_scores_only(self):
        from waddington_select import bnn
        from waddington_select.arms.genedisco_faithful_arm import GeneDiscoFaithfulArm, arm_names
        from waddington_select.oracle import DatasetOracle
        ds, B = "Scharenberg22", 32
        oracle = DatasetOracle(ds)
        fitted_rows = []
        real_fit = bnn.fit

        def spy(X, target, *a, **k):
            fitted_rows.append(len(X))
            return real_fit(X, target, *a, **{**k, "epochs": 3})

        names = [n for n in arm_names() if "_rf_" not in n or n.endswith("topuncertain")]
        with patch.object(bnn, "fit", spy):
            for name in names:
                with self.subTest(arm=name):
                    picks = []
                    for _ in range(2):
                        arm = GeneDiscoFaithfulArm(ds, B, name=name, seed=3)
                        b1 = arm.select(0, {})
                        arm.update(0, oracle.reveal(b1))
                        b2 = arm.select(1, {})
                        picks.append((b1, b2))
                        self.assertEqual(len(set(b1) | set(b2)), 2 * B)  # full, disjoint batches
                    self.assertEqual(picks[0], picks[1])
        self.assertTrue(fitted_rows and all(n == B for n in fitted_rows))  # only the revealed genes

    def test_softuncertain_survives_count_scale_scores(self):
        """Regression: float32 MC samples underflowed the probability floor on count-scale scores."""
        from waddington_select.arms.genedisco_faithful_arm import GeneDiscoFaithfulArm
        from waddington_select.oracle import DatasetOracle
        ds = "Scharenberg22"
        arm = GeneDiscoFaithfulArm(ds, 32, name="gdf_achilles_bnn_softuncertain", seed=0)
        arm._score = arm._score * 1000.0  # spread the MLP's samples far beyond float32's softmax range
        b1 = arm.select(0, {})
        arm.update(0, DatasetOracle(ds).reveal(b1))
        self.assertEqual(len(set(arm.select(1, {})) - set(b1)), 32)

    def test_equal_information_arms_learn_from_labels_only(self):
        from waddington_select import bnn
        from waddington_select.arms import genedisco_faithful_arm as gdf
        from waddington_select.oracle import DatasetOracle
        ds, B = "Scharenberg22", 32
        oracle = DatasetOracle(ds)
        self.assertEqual(len(gdf.binary_arm_names()), 20)  # every arm with a model, both feature sets
        targets = []
        real_fit = bnn.fit

        def spy(X, target, *a, **k):
            targets.append(np.asarray(target))
            return real_fit(X, target, *a, **{**k, "epochs": 3})

        def no_scores(*a, **k):
            raise AssertionError("an equal-information arm read the target screen's scores")

        with patch.object(gdf, "load_scores", no_scores), patch.object(bnn, "fit", spy):
            for name in ("gdfb_ours_bnn_topuncertain", "gdfb_achilles_rf_marginsample"):
                with self.subTest(arm=name):
                    arm = gdf.GeneDiscoFaithfulArm(ds, B, name=name, seed=0)
                    b1 = arm.select(0, {})
                    rev = oracle.reveal(b1)
                    arm.update(0, rev)
                    b2 = arm.select(1, {})
                    self.assertEqual(len(set(b1) | set(b2)), 2 * B)
                    if len(set(rev.values())) == 1:
                        continue  # single-class fallback, covered below
        for t in targets:
            self.assertTrue(set(np.unique(t)) <= {0.0, 1.0})

    def test_equal_information_single_class_falls_back_to_random(self):
        from waddington_select.arms.genedisco_faithful_arm import GeneDiscoFaithfulArm
        arm = GeneDiscoFaithfulArm("Scharenberg22", 32, name="gdfb_ours_bnn_badge", seed=0)
        b1 = arm.select(0, {})
        arm.update(0, {g: False for g in b1})

        def no_fit():
            raise AssertionError("fitted a model on single-class labels")

        arm._fit = no_fit
        self.assertEqual(len(set(arm.select(1, {})) - set(b1)), 32)

    def test_bnn_acquisition_arms(self):
        from waddington_select.arms.ml_engine_arm import BNN_ACQUISITIONS, BNNAcquisitionArm
        for rule in BNN_ACQUISITIONS:
            with self.subTest(rule=rule):
                a = BNNAcquisitionArm("Scharenberg22", 32, acquisition=rule, seed=2).select(0, {})
                b = BNNAcquisitionArm("Scharenberg22", 32, acquisition=rule, seed=2).select(0, {})
                self.assertEqual(a, b)
                self.assertEqual(len(set(a)), 32)


class BeyondGeneDiscoTests(unittest.TestCase):
    """Phase 2 item 4: multi-task MF, sparse GP (+UCB arm), LightGBM + RWR feature."""

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))

    def test_rwr_feature_never_carries_a_rows_own_label(self):
        from waddington_select.string_graph import string_graph
        genes, _ = string_graph()
        rng = np.random.default_rng(0)
        g = list(np.array(genes)[rng.choice(len(genes), 1500, replace=False)])
        groups = ["S1"] * 500 + ["S2"] * 500 + ["S3"] * 400 + ["__target__"] * 100
        y = (rng.random(1500) < 0.1).astype(int)
        X = rng.normal(size=(1500, 3))
        w = np.where(np.array(groups) == "__target__", 10.0, 1.0)
        fit = lambda labels: make_engine("lgbm_rwr").fit(X, labels, w, 9.0, genes=g, groups=groups)
        a = fit(y)
        y_s1 = y.copy()
        y_s1[:500] = 1 - y_s1[:500]  # flip S1's own labels
        b = fit(y_s1)
        np.testing.assert_array_equal(a.train_feature[:500], b.train_feature[:500])  # S1 rows: unchanged
        self.assertFalse(np.array_equal(a.train_feature[500:1000], b.train_feature[500:1000]))  # S2 sees S1
        y_t = y.copy()
        y_t[1400:] = 1 - y_t[1400:]  # flip the revealed target labels
        c = fit(y_t)
        np.testing.assert_array_equal(a.train_feature, c.train_feature)  # no training row sees them...
        self.assertFalse(np.array_equal(a._score_vec, c._score_vec))  # ...only the genes being scored do

    def test_svgp_ucb_arm(self):
        from waddington_select.arms.ml_engine_arm import SVGPUCBArm
        a = SVGPUCBArm("Scharenberg22", 32, seed=1).select(0, {})
        b = SVGPUCBArm("Scharenberg22", 32, seed=1).select(0, {})
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 32)


class RowOrderTests(unittest.TestCase):
    """GeneDisco-style arms must not depend on how the target screen's rows are stored: the training table
    lists some screens' hits first, and tie-breaking by row order would read them."""

    def setUp(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("WADDINGTON_")}
        self.enterContext(patch.dict(os.environ, env, clear=True))

    def test_selections_identical_under_a_label_free_reordering(self):
        import pandas as pd
        import waddington_select.features as F
        from waddington_select.arms import genedisco_arm, genedisco_faithful_arm
        from waddington_select.oracle import DatasetOracle
        orig = F.load_training_frame

        def alpha(csv_path, ds):
            df = orig(csv_path, ds)
            tgt = df["dataset"] == ds
            key = df.loc[tgt, "gene"].astype(str).str.strip().str.upper().to_numpy()
            return pd.concat([df[~tgt], df[tgt].iloc[np.argsort(key, kind="stable")]], ignore_index=True)

        def run(make, ds, B):
            arm, oracle, picks = make(), DatasetOracle(ds), []
            for r in range(3):
                b = arm.select(r, {})
                arm.update(r, oracle.reveal(b))
                picks.append(b)
            return picks

        cases = [("Replogle_K562_essential", 32, lambda ds, B: genedisco_arm.GeneDiscoArm(ds, B, "marginsample", seed=1)),
                 ("Scharenberg22", 32, lambda ds, B: genedisco_arm.GeneDiscoArm(ds, B, "topuncertain", seed=1)),
                 ("Scharenberg22", 32, lambda ds, B: genedisco_faithful_arm.GeneDiscoFaithfulArm(
                     ds, B, name="gdfb_ours_rf_marginsample", seed=1)),
                 ("Replogle_K562_essential", 32, lambda ds, B: genedisco_faithful_arm.GeneDiscoFaithfulArm(
                     ds, B, name="gdf_ours_rf_marginsample", seed=1))]
        for ds, B, make in cases:
            with self.subTest(screen=ds, arm=make(ds, B).name):
                a = run(lambda: make(ds, B), ds, B)
                with patch.object(genedisco_arm, "load_training_frame", alpha), \
                     patch.object(genedisco_faithful_arm, "load_training_frame", alpha):
                    b = run(lambda: make(ds, B), ds, B)
                self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
