"""$WADDINGTON_EXCLUDE_FEATS removes named feature columns from the C-arm's ML branch and padding ranker,
and is recorded in the run manifest. Unset (the default), nothing changes."""

import os
import unittest
from unittest import mock

from waddington_select.arms import waddington_c_arm as C
from waddington_select.tables import excluded_feature_cols

PAIR = {"hub_score_norm", "ppi_score_sum"}


class ExcludeFeatsTests(unittest.TestCase):
    def arm(self):
        return C.WaddingtonCArm("Scharenberg22", 32)   # builds both branches; no LLM call until select()

    def test_default_keeps_every_feature(self):
        with mock.patch.dict(os.environ, {}, clear=False) as env:
            env.pop("WADDINGTON_EXCLUDE_FEATS", None)
            self.assertEqual(excluded_feature_cols(), set())
            arm = self.arm()
            self.assertTrue(PAIR <= set(arm._online._available_feats))
            self.assertTrue(PAIR <= set(arm._llm._all_feats))

    def test_named_columns_are_removed_from_both_branches(self):
        with mock.patch.dict(os.environ, {"WADDINGTON_EXCLUDE_FEATS": " hub_score_norm , ppi_score_sum "}):
            self.assertEqual(excluded_feature_cols(),
                             PAIR | {"hub_score_norm_absdiff", "ppi_score_sum_absdiff"})
            arm = self.arm()
            self.assertFalse(PAIR & set(arm._online._available_feats))
            self.assertFalse(PAIR & set(arm._llm._all_feats))
            self.assertIn("string_degree_norm", arm._online._available_feats)
            scores = arm._online.all_scores()
            self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))

    def test_manifest_records_the_exclusion(self):
        from waddington_select.run_sequential import _protocol_manifest
        with mock.patch.dict(os.environ, {"WADDINGTON_EXCLUDE_FEATS": "ppi_score_sum,hub_score_norm"}):
            m = _protocol_manifest(["Scharenberg22"], ["waddington_c"], 5, 1, "local")
        flat = repr(m)
        self.assertIn("'exclude_feature_cols': ['hub_score_norm', 'hub_score_norm_absdiff', 'ppi_score_sum', "
                      "'ppi_score_sum_absdiff']", flat)


if __name__ == "__main__":
    unittest.main()
