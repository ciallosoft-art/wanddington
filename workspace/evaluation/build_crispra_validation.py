"""
build_crispra_validation.py — build the isolated CRISPRa validation table.

Purpose: test the reason-vs-recall router's gain-of-function side, which the paper validates on a single
screen (Steinhart). We add the Schmidt CRISPRa arm for IL-2 and IFN-γ as two more gain-of-function
screens, matched to the existing CRISPRi benchmark screens of the SAME phenotype.

The trick that makes the comparison clean: the anchor-relative features (PPI / co-expression / pathway
overlap to the phenotype's core genes) are modality-agnostic — activating vs. knocking down IL-2 biology
uses the same anchors. So we CLONE the existing IL2 / IFNG feature rows and only swap the label to the
CRISPRa hit set. The CRISPRa and CRISPRi versions of a phenotype then have identical features and differ
only in labels, isolating perturbation modality.

Output goes to a SEPARATE csv (a superset of the v1 benchmark table). Nothing overwrites the frozen
benchmark file. Point runs at it with:

    WADDINGTON_TRAINING_CSV=.../lgbm_training_data_crispra_validation.csv \
    python -m waddington_select.run_sequential --arms ... --datasets IL2_crispra IFNG_crispra ...

Source data: Schmidt et al. 2022 Science (abj4008), bioRxiv 2021.05.11.443701 supplementary media-1.xlsx,
sheets CRISPRa.IL2screen.gene_summary / CRISPRa.IFNGscreen.gene_summary (MAGeCK gene_summary). Hits are
our standard convention: gaussian top-5% by |pos|lfc| on the feature pool, minus CEGv2 (non-essential).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V1 = HERE / "lgbm_training_data.csv"
OUT = HERE / "lgbm_training_data_crispra_validation.csv"
# External inputs (not in the repo): the user's download + BDA's essential-gene list.
MEDIA1 = Path.home() / "下载" / "media-1.xlsx"
CEGV2 = Path.home() / "Python" / "keypaper" / "code" / "BioDiscoveryAgent" / "CEGv2.txt"

SHEETS = {"IL2": "CRISPRa.IL2screen.gene_summary", "IFNG": "CRISPRa.IFNGscreen.gene_summary"}
TOP_RATIO = 0.05


def crispra_hits(sheet: str, pool: set[str], ceg: set[str]) -> set[str]:
    df = pd.read_excel(MEDIA1, sheet_name=sheet)
    df["id"] = df["id"].astype(str).str.upper()
    df = df[df["id"].isin(pool)]
    v = pd.to_numeric(df["pos|lfc"], errors="coerce").to_numpy(float)
    n = max(1, int(np.isfinite(v).sum() * TOP_RATIO))
    idx = np.argsort(np.where(np.isfinite(v), np.abs(v), -np.inf))[::-1][:n]
    return set(df["id"].to_numpy()[idx]) - ceg


def main() -> None:
    ceg = {x.strip().upper() for x in CEGV2.read_text().splitlines() if x.strip()}
    v1 = pd.read_csv(V1)
    v1["gene"] = v1["gene"].str.upper()

    parts = [v1]
    for cyto, sheet in SHEETS.items():
        src = v1[v1["dataset"] == cyto].copy()
        hits = crispra_hits(sheet, set(src["gene"]), ceg)
        clone = src.copy()
        clone["dataset"] = f"{cyto}_crispra"
        clone["label"] = clone["gene"].isin(hits).astype(int)
        parts.append(clone)
        print(f"{cyto}_crispra: {len(clone)} rows, {int(clone['label'].sum())} hits "
              f"({clone['label'].mean():.1%}); features identical to {cyto} (CRISPRi)")

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(out)} rows, datasets={sorted(out['dataset'].unique())})")


if __name__ == "__main__":
    main()
