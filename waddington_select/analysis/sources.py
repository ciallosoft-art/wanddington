"""
sources.py — which result files the paper's numbers come from (docs/paper_code_audit.md).

The reported system is the GPT-5.6 Luna C-arm under the corrected task prompts. The first twenty-replicate
GPT run (gpt_5_6_luna_none_20rep, 2026-08-24) described the two Replogle Perturb-seq screens as fitness /
survival screens, which their labels are not (transcriptomic L2 distance / A-D counts); the prompts were
corrected (`replogle_transcriptomics_v2`, docs/fix_impact_map.md) and the whole three-stage protocol was
re-run (replogle_transcriptomics_v2_20rep, 2026-09-18). That re-run is what the current code reproduces,
so it is the reported run. The first run, and the supporting GPT analyses run before the correction
(the legacy-consistent ablations, the router-variant and reasoning grids under gpt/router/, the tiers'
memory used by the combinatorial experiments), are kept on disk and cited as such.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GPT = REPO / "workspace" / "results" / "gpt"

# the reported run: calibration (five arms) + verified memory + C-arm, twenty replicates each
REPORTED_RUN_DIR = GPT / "replogle_transcriptomics_v2_20rep"
REPORTED_RUN = REPORTED_RUN_DIR / "main_table_20rep.json"
# the same protocol at five replicates (the repeat-count robustness table)
REPORTED_RUN_5REP = GPT / "replogle_transcriptomics_v2" / "main_table_5rep.json"

# before the prompt correction (kept, cited as such)
FIRST_GPT_RUN_DIR = GPT / "gpt_5_6_luna_none_20rep"
LEGACY_GPT_ABLATIONS = FIRST_GPT_RUN_DIR / "ablations_20rep.json"  # legacy-consistent routed C + ablations
GPT_ROUTER = GPT / "router"  # with-anchor global-weight grid, nested router, study-out, hit-frequency

# the Claude predecessor (cross-model check)
CLAUDE_FINAL_20SEED = REPO / "workspace" / "results" / "router" / "main_table_20seed.json"
