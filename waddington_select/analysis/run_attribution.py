"""Pool a traced campaign over all 9 benchmark screens → the attribution result for the paper."""
import json
import sys
from pathlib import Path

from waddington_select.oracle import BENCHMARK_DATASETS
from waddington_select.analysis.trace import traced_campaign, hit_rate_by_source

OUT = Path(__file__).resolve().parents[2] / "workspace" / "results" / "attribution_9ds.json"

pooled = {"rounds": []}
per_ds = {}
for ds in BENCHMARK_DATASETS:
    try:
        res = traced_campaign(ds, rounds=5)
    except Exception as e:  # keep going; report honestly at the end
        print(f"  {ds:26s} FAILED: {e}", flush=True)
        continue
    rounds = res["trace"]["rounds"]
    for r in rounds:
        r["dataset"] = ds
    pooled["rounds"].extend(rounds)
    per_ds[ds] = hit_rate_by_source(res["trace"])
    agg = per_ds[ds]
    print(f"  {ds:26s} hit_ratio={res['hit_ratio']:.3f} | "
          f"both {agg['both']['hits']}/{agg['both']['picked']} "
          f"ml {agg['ml_only']['hits']}/{agg['ml_only']['picked']} "
          f"llm {agg['llm_only']['hits']}/{agg['llm_only']['picked']}", flush=True)

overall = hit_rate_by_source(pooled)
json.dump({"pooled": pooled, "per_dataset": per_ds, "overall": overall}, open(OUT, "w"))
print("\nPOOLED OVER ALL SCREENS")
for s, v in overall.items():
    print(f"  {s:9s} picked={v['picked']:5d} hits={v['hits']:5d} hit_rate={v['hit_rate']*100:5.1f}%")
print(f"\nwrote {OUT}")
