"""The combinatorial tier's label-free interaction prior: the numbers the paper reports for it."""

from __future__ import annotations

import unittest

from waddington_select.analysis import pair_tier


@unittest.skipUnless(pair_tier.PAIR_TABLE_V3.exists(), "Horlbeck pair table not built")
class InteractionPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sub = pair_tier._horlbeck()
        cls.hits = set(cls.sub.loc[cls.sub["label"] == 1, "gene"])
        cls.orders = {k: pair_tier._order(p, cls.sub["gene"].tolist())
                      for k, p in pair_tier._priors(cls.sub).items()}

    def _top(self, columns: int, n: int) -> int:
        return sum(g in self.hits for g in self.orders[columns][:n])

    def test_prior_alone_at_both_budgets(self):
        # seven original pair columns; the eighth is complex co-membership
        self.assertEqual((self._top(7, 640), self._top(7, 160)), (182, 54))
        self.assertEqual((self._top(8, 640), self._top(8, 160)), (196, 65))

    def test_prior_reads_no_label(self):
        p = pair_tier.interaction_prior(self.sub.drop(columns=["label"]))
        self.assertEqual(p, pair_tier.interaction_prior(self.sub))

    def test_complex_pairs(self):
        cx = set(self.sub.loc[self.sub[pair_tier.COMPLEX_COL] > 0, "gene"])
        self.assertEqual(len(cx), 445)
        self.assertAlmostEqual(sum(g in self.hits for g in cx) / len(cx), 0.344, places=3)


if __name__ == "__main__":
    unittest.main()
