import tempfile
import unittest
from pathlib import Path

from waddington_select.ingest import derive_hits


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def readout(self, content, genes=("STAT1", "JAK2"), suffix=".csv", **kwargs):
        path = Path(self.tmp.name) / f"screen{suffix}"
        path.write_text(content)
        return derive_hits(str(path), list(genes), **kwargs)

    def test_float_labels_are_binary_labels(self):
        result = self.readout("Gene,label\nSTAT1,1.0\nJAK2,0.0\n")
        self.assertEqual(result["hits"], ["STAT1"])
        self.assertEqual(result["misses"], ["JAK2"])

    def test_missing_label_does_not_turn_other_positives_into_negatives(self):
        result = self.readout("Gene,label\nSTAT1,1\nJAK2,\n")
        self.assertEqual(result["hits"], ["STAT1"])
        self.assertEqual(result["misses"], [])
        self.assertEqual(result["unknown_genes"], ["JAK2"])

    def test_absent_gene_is_unknown_not_negative(self):
        result = self.readout("Gene,label\nSTAT1,yes\n")
        self.assertEqual(result["hits"], ["STAT1"])
        self.assertEqual(result["misses"], [])
        self.assertEqual(result["unknown_genes"], ["JAK2"])

    def test_boolean_labels(self):
        result = self.readout("Gene,is_hit\nSTAT1,True\nJAK2,False\n")
        self.assertEqual(result["hits"], ["STAT1"])
        self.assertEqual(result["misses"], ["JAK2"])

    def test_unrecognised_or_conflicting_labels_are_rejected(self):
        for rows in ("STAT1,2\nJAK2,0", "STAT1,maybe\nJAK2,0", "STAT1,1\nSTAT1,0"):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.readout(f"Gene,label\n{rows}\n")

    def test_invalid_scores_do_not_create_a_hit(self):
        for method in ("gaussian", "castle"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "No finite scores"):
                self.readout("Gene,Score\nSTAT1,NaN\nJAK2,NaN\n", method=method)

    def test_missing_score_remains_unknown(self):
        result = self.readout("Gene,Score\nSTAT1,2\nJAK2,NaN\n")
        self.assertEqual(result["hits"], ["STAT1"])
        self.assertEqual(result["misses"], [])
        self.assertEqual(result["unknown_genes"], ["JAK2"])

    def test_invalid_ratio_is_rejected(self):
        for ratio in (0, -0.1, 1.1, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.readout("Gene,Score\nSTAT1,2\nJAK2,0\n", top_ratio=ratio)

    def test_mageck_and_uppercase_extensions(self):
        for suffix, content in (
            (".TXT", "id\tpos|lfc\nSTAT1\t2\nJAK2\t0\n"),
            (".TSV", "Gene\tScore\nSTAT1\t2\nJAK2\t0\n"),
            (".CSV", "Gene,Score\nSTAT1,2\nJAK2,0\n"),
        ):
            with self.subTest(suffix=suffix):
                result = self.readout(content, suffix=suffix)
                self.assertEqual(result["hits"], ["STAT1"])
                self.assertEqual(result["misses"], ["JAK2"])

    def test_missing_gene_identifiers_do_not_count_as_hits(self):
        result = self.readout("Gene,label\n,1\nSTAT1,0\n")
        self.assertEqual(result["total_hits"], 0)
        self.assertEqual(result["misses"], ["STAT1"])


if __name__ == "__main__":
    unittest.main()
