# Data

Feature tables and benchmark labels for the gene-selection benchmark. Large binary inputs
(`.h5ad` / `.h5` / `.loom`) are gitignored; the derived feature CSVs are committed and are what the C-arm
actually reads (machine-independent).

## Contents

| Path | What it is |
|------|-----------|
| `bda_benchmark/` | Task prompts + hit labels for the nine benchmark screens (taken unchanged from the BioDiscoveryAgent release). |
| `universal_features.csv` | Gene-intrinsic features (gnomAD pLI, STRING degree / hubness) for ~20.8k genes. |
| `pathway_features.csv` | KEGG / Reactome pathway-count features. |
| `pathway_features_tier_a.csv` | Same, extended to the unfiltered (essential-inclusive) gene universe for BDA's *easier* tier. Normalisers are pinned to the committed file's, so shared genes keep identical values. |
| `depmap/` | DepMap CRISPR essentiality features (processed). |
| `raw_h5ad/` | Replogle Perturb-seq raw inputs (gitignored) — needed only to *re-derive* the Replogle feature rows via `workspace/evaluation/prep_replogle*.py`. |
| `user_phenotypes/` | Phenotypes a scientist registers at deploy time (gitignored). |

Rebuild any feature table from raw sources with the `prep_*.py` scripts in `workspace/evaluation/`.
The two BDA difficulty-tier tables are built by `build_tier_a_unfiltered.py` (1-gene, full pool)
and `build_horlbeck_pairs.py` (2-gene combinations); both verify themselves against the frozen
benchmark table and refuse to write a table that does not reproduce it. See `experiments/README.md`.
