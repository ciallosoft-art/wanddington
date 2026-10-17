"""A registered phenotype's design metadata sets its DepMap features by the benchmarks' rule; anchors are optional.

router_protocol.feature_policy_from: DepMap off for CRISPRa or a curated essential panel, pan-cancer only in
K562, else pan-cancer + K562. Registered screens used to fall through to all four DepMap features whatever
their design, which is the wrong prior for a CRISPRa screen.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import waddington_select.phenotype as P
from waddington_select.arms import waddington_c_arm as C
from waddington_select.router_protocol import BENCH, feature_policy_metadata

REPO = Path(__file__).resolve().parents[1]
POOL = ["STAT1", "JAK1", "IFNGR1", "IRF1", "SOCS1"]
DEPMAP = set(C.DEPMAP_K562_FEATS)
REAL_REGISTRY = P.REGISTRY_PATH


def _no_network(genes, anchors):
    raise AssertionError("anchor features computed without anchors")


class MetadataTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        for patch in (mock.patch.object(P, "USER_DIR", self.dir),
                      mock.patch.object(P, "REGISTRY_PATH", self.dir / "registry.json"),
                      mock.patch.object(P, "_anchor_features", _no_network)):
            patch.start()
            self.addCleanup(patch.stop)

    def register(self, name, **kw):
        return P.register(name, POOL, [], "task", "measurement", **kw)

    def test_modality_and_cell_line_spellings(self):
        for text, want in (("KO", "KO"), ("CRISPR knockout", "KO"), ("Cas9", "KO"), ("crispri", "CRISPRi"),
                           ("dCas9-KRAB", "CRISPRi"), ("knockdown", "CRISPRi"), ("CRISPRa", "CRISPRa"),
                           ("dCas9-VP64 activation", "CRISPRa"), ("over-expression", "CRISPRa"),
                           ("", "unknown"), (None, "unknown"), ("unknown", "unknown"), ("don't know", "unknown")):
            with self.subTest(text=text):
                self.assertEqual(P.normalize_modality(text), want)
        for bad in ("CRISPR", "dCas9", "magic"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                P.normalize_modality(bad)
        for text, want in (("K562", "K562"), ("k562 cells", "K562"), ("K-562", "K562"), ("K562_dCas9", "K562"),
                           ("primary  T cells", "primary T cells"), ("HEK293T", "HEK293T"), ("", None), (None, None)):
            with self.subTest(text=text):
                self.assertEqual(P.normalize_cell_line(text), want)

    def test_the_rule_reads_registered_metadata(self):
        cases = {
            "Act": ({"modality": "CRISPRa", "cell_line": "primary T cells"}, "none", (C.TRAINING_DATA_V1, [])),
            "Panel": ({"modality": "CRISPRi", "curated_essential": True}, "none", (C.TRAINING_DATA_V1, [])),
            "K562i": ({"modality": "CRISPRi", "cell_line": "K562 cells"}, "pan",
                      (C.TRAINING_DATA_V3, C.DEPMAP_PAN_FEATS)),
            "Neuro": ({"modality": "KO", "cell_line": "iPSC neurons"}, "pan_k562",
                      (C.TRAINING_DATA_V3, C.DEPMAP_K562_FEATS)),
            "Unknown": ({}, "pan_k562", (C.TRAINING_DATA_V3, C.DEPMAP_K562_FEATS)),
        }
        for name, (kw, policy, config) in cases.items():
            with self.subTest(name=name):
                res = self.register(name, **kw)
                self.assertEqual(res["depmap_policy"], policy)
                self.assertEqual(C._get_feature_config(name), config)
        entry = json.loads(P.REGISTRY_PATH.read_text())["Act"]
        self.assertEqual(entry["metadata"], {"modality": "CRISPRa", "cell_line": "primary T cells",
                                             "curated_ess": False})
        self.assertEqual(json.loads(P.REGISTRY_PATH.read_text())["Unknown"]["metadata"]["modality"], "unknown")

    def test_an_entry_registered_before_metadata_keeps_its_features(self):
        self.register("Old")
        reg = json.loads(P.REGISTRY_PATH.read_text())
        del reg["Old"]["metadata"], reg["Old"]["depmap_policy"]
        P.REGISTRY_PATH.write_text(json.dumps(reg))
        self.assertEqual(C._get_feature_config("Old"), (C.TRAINING_DATA_V3, C.DEPMAP_K562_FEATS))

    def test_benchmarks_and_validation_screens_are_unchanged(self):
        want = {"Steinhart": "none", "Replogle_K562_essential": "none", "Replogle_K562_gwps": "pan"}
        for ds in BENCH:
            with self.subTest(ds=ds):
                self.assertEqual(feature_policy_metadata(ds)["depmap"], want.get(ds, "pan_k562"))
        self.assertEqual(C._get_feature_config("Steinhart"), (C.TRAINING_DATA_V1, []))
        self.assertEqual(C._get_feature_config("Replogle_K562_gwps"), (C.TRAINING_DATA_V3, C.DEPMAP_PAN_FEATS))
        self.assertEqual(C._get_feature_config("IFNG"), (C.TRAINING_DATA_V3, C.DEPMAP_K562_FEATS))
        self.assertEqual(C._get_feature_config("IL2_crispra"), (C.TRAINING_DATA_V1, []))  # committed set

    def test_a_crispra_phenotype_is_ranked_without_depmap(self):
        self.register("Act", modality="CRISPRa")
        self.register("Lof", modality="KO")
        act = C.WaddingtonCArm("Act", 4)._online._available_feats   # no LLM call until select()
        lof = C.WaddingtonCArm("Lof", 4)._online._available_feats
        self.assertFalse(DEPMAP & set(act))
        self.assertEqual(DEPMAP & set(lof), DEPMAP)
        self.assertFalse({"g1_ppi_score", "archs4_coexpr", "kegg_overlap"} & set(act + lof))  # dropped anyway

    def test_no_anchors_gives_zero_anchor_features(self):
        res = self.register("NoSeeds")
        self.assertEqual(res["anchors"], [])
        f = P.features_frame("NoSeeds")
        self.assertTrue((f[P.ANCHOR_COLS] == 0).all().all())

    def test_bad_metadata_and_benchmark_names_are_refused(self):
        with self.assertRaises(SystemExit):
            self.register("Bad", modality="CRISPR")
        with self.assertRaises(SystemExit):
            self.register("IFNG")

    def test_cli_registers_without_anchors(self):
        code = (
            "import sys; from pathlib import Path; import waddington_select.phenotype as P; "
            f"P.USER_DIR = Path({str(self.dir)!r}); P.REGISTRY_PATH = P.USER_DIR / 'registry.json'; "
            "sys.argv = ['phenotype', 'register', '--name', 'Cli', '--genes', *sys.argv[1:], '--task', 't', "
            "'--measurement', 'm', '--modality', 'CRISPR activation', '--cell-line', 'Jurkat', '--json']; "
            "P.main()"   # not runpy: a fresh __main__ copy of the module would write the real registry
        )
        out = subprocess.run([sys.executable, "-c", code, *POOL], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout
        res = json.loads(out[out.index("{"):])
        self.assertEqual((res["metadata"]["modality"], res["metadata"]["cell_line"]), ("CRISPRa", "Jurkat"))
        self.assertEqual((res["depmap_policy"], res["anchors"]), ("none", []))
        bad = subprocess.run([sys.executable, "-c", code.replace("CRISPR activation", "CRISPR"), *POOL], cwd=REPO,
                             capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("unrecognised modality", bad.stderr)
        self.assertIn("Cli", json.loads((self.dir / "registry.json").read_text()))   # the temporary registry
        self.assertNotIn("Cli", json.loads(REAL_REGISTRY.read_text()) if REAL_REGISTRY.exists() else {})


if __name__ == "__main__":
    unittest.main()
