#!/usr/bin/env python3
"""Exit 0 when every BDA tier run is complete; otherwise print what is missing and exit 1."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GPT = REPO / "workspace" / "results" / "gpt"
SINGLE = ["IFNG", "IL2", "Sanchez21", "Sanchez21_down", "Carnevale22", "Scharenberg22",
          "Steinhart", "Replogle_K562_essential", "Replogle_K562_gwps"]

WANT = [
    (GPT / "tier_a_full_pool_20rep" / "bda_controlled.json", SINGLE, 1),
    (GPT / "tier_b_pairs_b32_20rep" / "bda_controlled_real.json", ["Horlbeck"], 3),
    # One complete replicate: the control's claim is qualitative and it costs ~2 h per run.
    (GPT / "tier_b_pairs_b32_20rep" / "bda_controlled_shipped.json", ["Horlbeck"], 1),
]

missing = []
for path, datasets, reps in WANT:
    data = json.loads(path.read_text()) if path.exists() else {}
    for ds in datasets:
        have = sum(1 for r in data.get(ds, {}).get("bda", []) if r.get("n_rounds") == 5)
        if have < reps:
            missing.append(f"{path.name}:{ds} {have}/{reps}")

if missing:
    print("MISSING: " + "; ".join(missing))
    sys.exit(1)
print("COMPLETE: all BDA tier runs present")
