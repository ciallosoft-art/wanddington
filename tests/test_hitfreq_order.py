"""The hit-frequency probes must not depend on the stored row order (which puts some screens' hits first)."""

import os
import unittest
from unittest import mock

import pandas as pd

from waddington_select.analysis import prior_probes
from waddington_select.arms import online_adaptive_arm
from waddington_select.arms.waddington_c_arm import _drop_anchor_default, _get_feature_config
from waddington_select.oracle import BATCH_SIZES

DS = "Scharenberg22"  # its hits are stored first, and the hit-frequency cut-off falls inside a tie


def _reversed_target(load):
    def wrapped(csv, ds):
        df = load(csv, ds)
        tgt = df["dataset"] == ds
        return pd.concat([df[~tgt], df[tgt].iloc[::-1]], ignore_index=True)
    return wrapped


class HitFrequencyOrderTests(unittest.TestCase):
    def test_static_ranking_is_order_free(self):
        df = prior_probes._load()
        tgt = df["dataset"] == DS
        flipped = pd.concat([df[~tgt], df[tgt].iloc[::-1]], ignore_index=True)
        self.assertAlmostEqual(prior_probes._hit_frequency(df, DS), prior_probes._hit_frequency(flipped, DS))

    def test_online_arm_is_order_free_under_hitfreq_only(self):
        csv, extra = _get_feature_config(DS)
        kw = dict(training_csv=csv, extra_feature_cols=extra, drop_anchor_feats=_drop_anchor_default())
        with mock.patch.dict(os.environ, {"WADDINGTON_HITFREQ_ONLY": "1"}):
            a = online_adaptive_arm.OnlineAdaptiveArm(DS, BATCH_SIZES[DS], **kw)
            load = online_adaptive_arm.load_training_frame
            with mock.patch.object(online_adaptive_arm, "load_training_frame", _reversed_target(load)):
                b = online_adaptive_arm.OnlineAdaptiveArm(DS, BATCH_SIZES[DS], **kw)
        self.assertEqual(a._ranking, b._ranking)


if __name__ == "__main__":
    unittest.main()
