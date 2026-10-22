"""
clean_connectivity.py — the full system without hub_score_norm and ppi_score_sum (2026-10-22; reads experiment 22).

Both columns are counted over a STRING query set holding the screens' anchors and hits (analysis/hub_leak.py).
Experiment 22 re-runs the reported GPT C-arm (twenty replicates, the reported run's verified memory, every other
setting unchanged) with them excluded. Paired over the nine screens, screen-clustered:

  C_clean  vs  matched prior without the two columns   the within-system gain on clean features (the question)
  C_reported vs matched prior with them                 the paper's +0.030, for reference
  C_clean  vs  C_reported                               what removing them does to the system (includes any
                                                        between-session drift, ~0.005 in this project)
  matched prior without vs with                         cross-check against hub_leak's "dropped" (0.240)

Run:  python -m waddington_select.analysis.clean_connectivity     -> workspace/results/clean_connectivity.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..router_protocol import BENCH
from .final_system_stats import _marginal, _matched_loo, _paired
from .sources import GPT, REPORTED_RUN

RUN = GPT / "clean_connectivity" / "waddington_c_20rep.json"
OUT = Path(__file__).resolve().parents[2] / "workspace" / "results" / "clean_connectivity.json"
EXCLUDED = "hub_score_norm,ppi_score_sum"


def _per_screen(path: Path) -> tuple[dict, dict]:
    run = json.loads(path.read_text())
    reps = {d: [r["hit_ratio_per_round"][-1] for r in run[d]["waddington_c"]] for d in BENCH}
    return {d: float(np.mean(v)) for d, v in reps.items()}, {d: len(v) for d, v in reps.items()}


def main() -> None:
    c_clean, n_clean = _per_screen(RUN)
    c_rep, n_rep = _per_screen(REPORTED_RUN)
    os.environ.pop("WADDINGTON_EXCLUDE_FEATS", None)
    prior_rep = _matched_loo()
    os.environ["WADDINGTON_EXCLUDE_FEATS"] = EXCLUDED
    prior_clean = _matched_loo()
    os.environ.pop("WADDINGTON_EXCLUDE_FEATS", None)

    out = {
        "excluded": EXCLUDED, "run": str(RUN), "replicates": {"clean": n_clean, "reported": n_rep},
        "per_screen": {"c_clean": c_clean, "c_reported": c_rep, "prior_clean": prior_clean, "prior_reported": prior_rep},
        "means": {k: float(np.mean(list(v.values()))) for k, v in
                  (("c_clean", c_clean), ("c_reported", c_rep), ("prior_clean", prior_clean), ("prior_reported", prior_rep))},
        "paired": {
            "c_clean_vs_prior_clean": _paired(c_clean, prior_clean),
            "c_reported_vs_prior_reported": _paired(c_rep, prior_rep),
            "c_clean_vs_c_reported": _paired(c_clean, c_rep),
            "prior_clean_vs_prior_reported": _paired(prior_clean, prior_rep),
        },
        "marginal_ci": {"c_clean": _marginal(c_clean), "prior_clean": _marginal(prior_clean)},
    }
    OUT.write_text(json.dumps(out, indent=1))
    print("replicates per screen:", n_clean)
    print({k: round(v, 4) for k, v in out["means"].items()})
    for k, p in out["paired"].items():
        print(f"{k:32s} {p['delta']:+.4f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]  {p['wins']}/{p['ties']}/{p['losses']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
