"""Registered hit rules and the full-screen / one-batch scope of a readout (ingest.py, phenotype.register)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import waddington_select.phenotype as P
from waddington_select.ingest import dataset_context, derive_hits, describe_hit_rule, make_hit_rule

REPO = Path(__file__).resolve().parents[1]

# a 20-gene "screen": scores from -2 to +1.8, MAGeCK-style FDRs
GENES = [f"G{i:02d}" for i in range(20)]
SCORES = [round(-2.0 + 0.2 * i, 2) for i in range(20)]          # G00 = -2.0 ... G19 = +1.8


def mageck(genes=GENES):
    rows = ["id\tpos|lfc\tpos|fdr\tneg|fdr"]
    for g in genes:
        s = SCORES[GENES.index(g)]
        rows.append(f"{g}\t{s}\t{0.01 if s >= 1.5 else 0.5}\t{0.01 if s <= -1.5 else 0.5}")
    return "\n".join(rows) + "\n"


class HitRuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, content, suffix=".txt"):
        path = Path(self.tmp.name) / f"readout{suffix}"
        path.write_text(content)
        return str(path)

    def hits(self, rule, genes=GENES, content=None, **kwargs):
        return derive_hits(self.write(content or mageck(genes)), genes, rule=rule, **kwargs)

    def test_threshold_directions(self):
        both = self.hits(make_hit_rule("threshold", threshold=1.5))["hits"]
        up = self.hits(make_hit_rule("threshold", threshold=1.5, direction="up"))["hits"]
        down = self.hits(make_hit_rule("threshold", threshold=1.5, direction="down"))["hits"]
        self.assertEqual(up, ["G18", "G19"])                     # 1.6, 1.8
        self.assertEqual(down, ["G00", "G01", "G02"])            # -2.0, -1.8, -1.6
        self.assertEqual(sorted(both), sorted(up + down))

    def test_mageck_fdr(self):
        self.assertEqual(self.hits(make_hit_rule("fdr", max_fdr=0.05, direction="up"))["hits"], ["G18", "G19"])
        self.assertEqual(sorted(self.hits(make_hit_rule("fdr", max_fdr=0.05))["hits"]),
                         ["G00", "G01", "G02", "G18", "G19"])

    def test_single_fdr_column_takes_the_direction_from_the_score_sign(self):
        content = "Gene,Score,padj\nA,2.0,0.01\nB,-2.0,0.01\nC,0.1,0.9\n"
        r = derive_hits(self.write(content, ".csv"), ["A", "B", "C"], rule=make_hit_rule("fdr", max_fdr=0.05,
                                                                                         direction="down"))
        self.assertEqual((r["hits"], r["misses"]), (["B"], ["A", "C"]))

    def test_label_rule_requires_a_label_column(self):
        r = derive_hits(self.write("Gene,hit\nA,1\nB,0\n", ".csv"), ["A", "B"], rule=make_hit_rule("label"))
        self.assertEqual((r["hits"], r["misses"]), (["A"], ["B"]))
        with self.assertRaisesRegex(ValueError, "hit/label column"):
            derive_hits(self.write("Gene,Score\nA,1\n", ".csv"), ["A"], rule=make_hit_rule("label"))

    def test_one_batch_is_not_ranked_and_has_no_screen_total(self):
        batch = GENES[:6]                                        # 30% of a 20-gene pool
        for rule in (make_hit_rule("top_fraction", fraction=0.1), None):   # None = the legacy gaussian default
            with self.subTest(rule=rule), self.assertRaisesRegex(ValueError, "whole screen"):
                self.hits(rule, genes=batch, pool=GENES)
        r = self.hits(make_hit_rule("threshold", threshold=1.5), genes=batch, pool=GENES)
        self.assertEqual((r["scope"], r["total_hits"], r["hits"]), ("batch", None, ["G00", "G01", "G02"]))

    def test_full_screen_replay_counts_the_screen_and_drops_excluded_genes(self):
        rule = make_hit_rule("top_fraction", fraction=0.25)     # 5 of 20 by |score|: G00 G01 G02 G18 G19
        r = self.hits(rule, pool=GENES)
        self.assertEqual((r["scope"], r["total_hits"]), ("full_screen", 5))
        r = self.hits(rule, pool=GENES, exclude=["G19"])        # e.g. a core-essential gene, non-essential tier
        self.assertEqual(r["total_hits"], 4)
        self.assertNotIn("G19", r["hits"] + r["misses"])

    def test_invalid_rules(self):
        for kind, kw in (("threshold", {}), ("threshold", {"threshold": -1}), ("fdr", {"max_fdr": 0}),
                         ("top_fraction", {"fraction": 1.5}), ("magic", {}), ("label", {"direction": "sideways"})):
            with self.subTest(kind=kind, kw=kw), self.assertRaises(ValueError):
                make_hit_rule(kind, **kw)
        self.assertEqual(describe_hit_rule(make_hit_rule("threshold", threshold=1.5)), "|score| >= 1.5")


class RegisteredRuleTests(unittest.TestCase):
    """The rule registered with the phenotype is what ingest applies (CLI --dataset)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for patch in (mock.patch.object(P, "USER_DIR", self.dir),
                      mock.patch.object(P, "REGISTRY_PATH", self.dir / "registry.json"),
                      mock.patch.object(P, "_anchor_features", lambda g, a: {c: {} for c in P.ANCHOR_COLS})):
            patch.start()
            self.addCleanup(patch.stop)
        pool = ["STAT1", "JAK1", "IFNGR1", "ABCE1"]               # ABCE1 is CEGv2
        P.register("Screen", pool, ["IFNG"], "t", "m", hit_rule=make_hit_rule("threshold", threshold=1.0))

    def test_context(self):
        ctx = dataset_context("Screen")
        self.assertEqual(ctx["rule"]["type"], "threshold")
        self.assertEqual(sorted(ctx["pool"]), ["IFNGR1", "JAK1", "STAT1"])   # non-essential tier: ABCE1 removed
        self.assertIn("ABCE1", ctx["exclude"])
        self.assertEqual(json.loads(P.REGISTRY_PATH.read_text())["Screen"]["hit_rule"]["threshold"], 1.0)

    def test_cli_applies_the_registered_rule_and_reports_ignored_overrides(self):
        readout = self.dir / "round1.csv"
        readout.write_text("Gene,Score\nSTAT1,2.5\nJAK1,0.2\n")
        code = (
            "import sys, runpy; from pathlib import Path; import waddington_select.phenotype as P; "
            f"P.USER_DIR = Path({str(self.dir)!r}); P.REGISTRY_PATH = P.USER_DIR / 'registry.json'; "
            "sys.argv = ['ingest', '--file', sys.argv[1], '--genes', 'STAT1', 'JAK1', '--dataset', 'Screen', "
            "'--top-ratio', '0.5', '--json']; runpy.run_module('waddington_select.ingest', run_name='__main__')"
        )
        out = subprocess.run([sys.executable, "-c", code, str(readout)], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout
        res = json.loads(out[out.index("{"):])
        self.assertEqual((res["hits"], res["misses"]), (["STAT1"], ["JAK1"]))
        self.assertEqual((res["scope"], res["total_hits"]), ("batch", None))
        self.assertEqual(res["overrides_ignored"], ["top_ratio"])


if __name__ == "__main__":
    unittest.main()
