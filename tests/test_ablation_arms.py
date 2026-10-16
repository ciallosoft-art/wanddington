"""The final-config component ablations (arms/waddington_c_ablation_arms.py) change exactly one thing each."""

import unittest

import numpy as np

from waddington_select.arms.llm_reasoning_arm import LLMReasoningArm
from waddington_select.arms.waddington_c_ablation_arms import (WaddingtonCAnonFeaturesArm,
                                                               WaddingtonCAnonLinearArm,
                                                               WaddingtonCAnonNamesArm, WaddingtonCAnonPoolArm,
                                                               WaddingtonCNoMemoryFinalArm,
                                                               WaddingtonCStaticMLArm)
from waddington_select.arms.waddington_c_feature_reasoning_arm import (_AnonymousFeaturesLLMArm,
                                                                       _AnonymousPoolOnlyLLMArm)
from waddington_select.arms.waddington_c_arm import WaddingtonCArm
from waddington_select.arms.waddington_c_shuffled_names_arm import _AnonymousNamesLLMArm
from waddington_select.oracle import BATCH_SIZES, DatasetOracle
from waddington_select.run_sequential import _PAPER_ARMS, make_arms

DS = "Replogle_K562_essential"  # the smallest screen (623 genes)


def _scores(arm) -> np.ndarray:
    s = arm._online.all_scores()
    return np.array([s[g] for g in sorted(s)])


class AblationArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bs = BATCH_SIZES[DS]
        cls.c = WaddingtonCArm(DS, bs)
        cls.static = WaddingtonCStaticMLArm(DS, bs)
        cls.nomem = WaddingtonCNoMemoryFinalArm(DS, bs)
        cls.anon = WaddingtonCAnonNamesArm(DS, bs)
        cls.pool = WaddingtonCAnonPoolArm(DS, bs)
        cls.feats = WaddingtonCAnonFeaturesArm(DS, bs)
        cls.linear = WaddingtonCAnonLinearArm(DS, bs)
        oracle = DatasetOracle(DS)
        genes = cls.c._online._ranking[:bs]
        cls.revealed = oracle.reveal(genes)

    def test_registered(self):
        names = ["waddington_c_nomem", "waddington_c_static_ml", "waddington_c_anon_names",
                 "waddington_c_anon_pool", "waddington_c_anon_features", "waddington_c_anon_linear"]
        self.assertTrue(set(names) <= _PAPER_ARMS)
        self.assertEqual([a.name for a in make_arms(DS, names)], names)

    def test_reported_arm_unchanged(self):
        self.assertIsNone(WaddingtonCArm._LLM_CLASS)
        self.assertIs(type(self.c._llm), LLMReasoningArm)

    def test_same_fusion_as_reported(self):
        for arm in (self.static, self.nomem, self.anon, self.pool, self.feats, self.linear):
            self.assertEqual((arm._route, arm._w_ml, arm._w_llm), (self.c._route, self.c._w_ml, self.c._w_llm))
            np.testing.assert_allclose(_scores(arm), _scores(self.c))  # same round-0 prior

    def test_static_ml_never_retrains(self):
        before = _scores(self.static)
        self.static.update(0, self.revealed)
        np.testing.assert_array_equal(_scores(self.static), before)
        # the reported arm does retrain on the same feedback, so the check above is not vacuous
        c = WaddingtonCArm(DS, BATCH_SIZES[DS])
        before_c = _scores(c)
        c.update(0, self.revealed)
        self.assertFalse(np.allclose(_scores(c), before_c))

    def test_no_memory(self):
        self.assertEqual(self.nomem._llm._memory, [])

    def test_ladder_rungs(self):
        self.assertIs(type(self.pool._llm), _AnonymousPoolOnlyLLMArm)
        self.assertIs(type(self.feats._llm), _AnonymousFeaturesLLMArm)
        self.assertEqual(self.feats._llm._memory, self.c._llm._memory)

    def test_linear_rule_counts_as_named(self):
        # round 1 has no feedback: the rule falls back to the static ranking and names nothing
        arm = WaddingtonCAnonLinearArm(DS, BATCH_SIZES[DS])
        b0 = arm.select(0, {})
        self.assertEqual(arm._last_llm_named_proposals, set())
        self.assertEqual(set(b0), set(self.c._online._ranking[:len(b0)]))  # ML alone
        arm.update(0, DatasetOracle(DS).reveal(b0))
        arm.select(1, {})
        named = arm._last_llm_named_proposals
        self.assertEqual(len(named), BATCH_SIZES[DS])  # with feedback every projection pick is named
        self.assertFalse(named & set(b0))

    def test_anonymous_names(self):
        self.assertIsInstance(self.anon._llm, _AnonymousNamesLLMArm)
        self.assertEqual(self.anon._llm._memory, self.c._llm._memory)


if __name__ == "__main__":
    unittest.main()
