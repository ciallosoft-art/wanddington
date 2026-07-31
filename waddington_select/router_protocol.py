"""
router_protocol.py — FROZEN protocol for the honest-router confirmatory experiments.

This file is committed BEFORE any target-screen result is inspected. It fixes, for the two
leakage-free routing variants, everything that a router is allowed to know: only quantities available
*before* the target screen is run. The target screen's realized hit rate and hit@R5 are NOT inputs to
any routing / weight / feature decision here.

Pre-experiment metadata (per screen) — all knowable from the experimental design, none from the results:
  - n_pool         : number of genes in the (CEGv2-filtered) library — a design choice, known up front.
  - modality       : CRISPRa (gain of function) / CRISPRi / KO — the perturbation you chose to run.
  - cell_line      : the cell system.
  - curated_ess    : True iff the screen is a curated essentiality panel (design metadata, not a result).
  - study          : same-study / same-source group, for leave-one-study-out.

Candidate configurations the variants may select among (fixed here):
  - fusion weight w_llm in CANDIDATE_WLLM (w_ml = 1 - w_llm), applied as a global weighted fusion
    (no two-stage, no hit-rate-triggered ml_heavy).
  - feature policy = FEATURE_POLICY_METADATA, a pure function of metadata (below).

Selection rules (how each variant chooses, using ONLY non-target screens):
  * global_fixed : one (w_llm) chosen by argmax of mean hit@R5 over the OTHER screens; applied to the
    held-out screen. Feature policy is the fixed metadata rule.
  * nested_router: a threshold-on-n rule family  w_llm = LOW if n_pool > T else HIGH ;  the pair
    (T, LOW, HIGH) is chosen by argmax mean hit@R5 over the OTHER screens; applied to the held-out
    screen via its n_pool only. Feature policy is the fixed metadata rule.
  * leave_one_study_out : as nested_router, but the whole same-study group is removed from the
    training screens when selecting (and from the cross-experiment prior).

Feature policy (metadata rule, label-free):
  - DepMap OFF entirely  iff modality == CRISPRa OR curated_ess  (knockout-derived prior misleads /
    a pan-cancer essentiality prior would restate a curated essentiality label).
  - K562-specific DepMap OFF iff cell_line == K562 (a cell-line-matched essentiality feature would be a
    proxy label on its own cell line).
  - otherwise pan-cancer + K562 DepMap.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOCOL_JSON = REPO / "workspace" / "results" / "router" / "protocol.json"

CANDIDATE_WLLM = [0.2, 0.3, 0.4]

# All fields are pre-experiment design metadata; none is derived from the screen's outcome.
SCREEN_METADATA: dict[str, dict] = {
    "IFNG":                    {"n_pool": 17785, "modality": "CRISPRi", "cell_line": "primary_T",  "curated_ess": False, "study": "schmidt"},
    "IL2":                     {"n_pool": 18273, "modality": "CRISPRi", "cell_line": "primary_T",  "curated_ess": False, "study": "schmidt"},
    "Sanchez21":               {"n_pool": 17807, "modality": "KO",      "cell_line": "neuronal",   "curated_ess": False, "study": "sanchez"},
    "Sanchez21_down":          {"n_pool": 17807, "modality": "KO",      "cell_line": "neuronal",   "curated_ess": False, "study": "sanchez"},
    "Carnevale22":             {"n_pool": 18224, "modality": "KO",      "cell_line": "primary_T",  "curated_ess": False, "study": "carnevale"},
    "Scharenberg22":           {"n_pool":  1029, "modality": "KO",      "cell_line": "primary_T",  "curated_ess": False, "study": "scharenberg"},
    "Steinhart":               {"n_pool": 18144, "modality": "CRISPRa", "cell_line": "CAR_T",      "curated_ess": False, "study": "steinhart"},
    "Replogle_K562_essential": {"n_pool":   623, "modality": "CRISPRi", "cell_line": "K562",       "curated_ess": True,  "study": "replogle"},
    "Replogle_K562_gwps":      {"n_pool":  9193, "modality": "CRISPRi", "cell_line": "K562",       "curated_ess": False, "study": "replogle"},
}

BENCH = list(SCREEN_METADATA.keys())


def study_group(dataset: str) -> str:
    return SCREEN_METADATA[dataset]["study"]


def siblings(dataset: str) -> list[str]:
    g = study_group(dataset)
    return [d for d in BENCH if d != dataset and SCREEN_METADATA[d]["study"] == g]


def feature_policy_metadata(dataset: str) -> dict:
    """Return {'depmap': 'none'|'pan'|'pan_k562'} from metadata only."""
    m = SCREEN_METADATA[dataset]
    if m["modality"] == "CRISPRa" or m["curated_ess"]:
        return {"depmap": "none"}
    if m["cell_line"] == "K562":
        return {"depmap": "pan"}
    return {"depmap": "pan_k562"}


def freeze_protocol() -> dict:
    doc = {
        "description": "Frozen honest-router protocol. Committed before inspecting target results.",
        "candidate_wllm": CANDIDATE_WLLM,
        "feature_policy": "metadata_rule (depmap off iff CRISPRa or curated; k562-depmap off iff cell==K562)",
        "screen_metadata": SCREEN_METADATA,
        "feature_policy_per_screen": {d: feature_policy_metadata(d) for d in BENCH},
        "study_groups": {g: [d for d in BENCH if SCREEN_METADATA[d]["study"] == g]
                         for g in sorted({m["study"] for m in SCREEN_METADATA.values()})},
        "selection_rules": {
            "global_fixed": "argmax mean hit@R5 over non-target screens over a single global w_llm",
            "nested_router": "w_llm = LOW if n_pool > T else HIGH; (T,LOW,HIGH) argmax over non-target screens; applied via target n_pool only",
            "leave_one_study_out": "nested_router with the whole same-study group removed from training + prior",
        },
        "forbidden_inputs": ["target realized hit rate", "target hit@R5", "any target-screen outcome"],
    }
    PROTOCOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL_JSON.write_text(json.dumps(doc, indent=2))
    return doc


if __name__ == "__main__":
    d = freeze_protocol()
    print("Frozen protocol ->", PROTOCOL_JSON)
    print("candidate w_llm:", d["candidate_wllm"])
    print("feature policy per screen:")
    for k, v in d["feature_policy_per_screen"].items():
        print(f"  {k:24s} {v['depmap']}")
    print("study groups:", d["study_groups"])
