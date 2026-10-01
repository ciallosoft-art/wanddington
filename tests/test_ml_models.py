"""Alternative ML engines: interface contract, weighting, determinism, and the LightGBM reference path."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from waddington_select.ml_models import ENGINES, engine_names, is_stochastic, make_engine


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
        for name in ENGINES:
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
        for name in ("logreg", "lambdarank"):
            with self.subTest(engine=name):
                a = make_engine(name).fit(X, y, w1, 5.0, genes=genes, groups=groups).score(X, genes)
                b = make_engine(name).fit(X, y, w10, 5.0, genes=genes, groups=groups).score(X, genes)
                self.assertFalse(np.allclose(a, b))

    def test_determinism_and_seed_dependence(self):
        X, y, genes, groups = _toy()
        for name in ENGINES:
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


if __name__ == "__main__":
    unittest.main()
