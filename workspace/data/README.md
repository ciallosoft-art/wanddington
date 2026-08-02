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
| `depmap/` | DepMap CRISPR essentiality features (processed). |
| `raw_h5ad/` | Replogle Perturb-seq raw inputs (gitignored) — needed only to *re-derive* the Replogle feature rows via `workspace/evaluation/prep_replogle*.py`. |
| `user_phenotypes/` | Phenotypes a scientist registers at deploy time (gitignored). |

Rebuild any feature table from raw sources with the `prep_*.py` scripts in `workspace/evaluation/`.
