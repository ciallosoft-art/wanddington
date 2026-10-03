# Superseded: faithful GeneDisco runs with row-order tie-breaking

These `gdf_*` / `gdfb_*` results (2026-10-02/03) were produced before `label_free_order` (arms/base.py).
The training table stores some screens' hits first, and these arms broke score ties by row order, so on
Scharenberg22, K562-Essential, K562-GWPS (and partly IL2) their results read labels — see
`../order_audit.json` and docs/ml_engines_plan.md (P2-3c, P2-6). Kept for the record; not used by the
analysis (which reads only the top-level directory). Re-runs with the fix replace them one level up.
