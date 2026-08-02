# Models

Per-dataset **LightGBM** gene-selection priors used by the C-arm's online adaptive arm.

| File | What it is |
|------|-----------|
| `lgbm_<dataset>.pkl` | Leave-one-out static ranker for one benchmark screen, trained on the *other* screens' hit labels (gene-intrinsic + anchor-relative + DepMap features). |
| `lgbm_cross_dataset.pkl` | A single model trained across all screens. |

These are the frozen priors behind the `static_ranker` and the online arm's round-1 ranking. Rebuild them
from the feature pipeline in `workspace/evaluation/` (`bootstrap_lgbm.py`; features from
`prep_*.py` / `gene_ranker.py`).

> The large single-cell perturbation-prediction model workspaces from the earlier paper-reproduction
> project (GEARS, scGPT, CPA, …) were split out to `../waddington-repro-archive` and are not part of this
> repository (gitignored here).
