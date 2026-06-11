"""
View aggregated benchmark results from workspace/results/.

Usage:
  python results_summary.py                   # all runs, sorted by timestamp
  python results_summary.py --ranker waddington
  python results_summary.py --compare-scramble
"""

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "workspace" / "results"
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
RUNS_DIR    = RESULTS_DIR / "runs"


def load_summary() -> list[dict]:
    if not SUMMARY_CSV.exists():
        return []
    with open(SUMMARY_CSV, newline="") as f:
        return list(csv.DictReader(f))


def print_table(rows: list[dict], title: str = "") -> None:
    if not rows:
        print("  (no data)")
        return
    if title:
        print(f"\n{title}")
        print("-" * len(title))

    # Dynamic columns: include dataset columns that have data
    ds_cols = [c for c in rows[0] if c.endswith("_r5_mean")]
    ds_labels = [c.replace("_r5_mean", "") for c in ds_cols]

    hdr = f"{'run_id':30s}  {'ranker':12s}  {'scr':3s}  {'avg_r5':7s}"
    for lbl in ds_labels:
        hdr += f"  {lbl[:10]:10s}"
    print(hdr)
    print("-" * len(hdr))

    for row in rows:
        scr = "Y" if row.get("scramble_genes", "").lower() == "true" else "N"
        avg = row.get("avg_r5", "")
        line = f"{row['run_id']:30s}  {row.get('ranker',''):12s}  {scr:3s}  {avg:7s}"
        for col in ds_cols:
            v = row.get(col, "")
            line += f"  {v[:10]:10s}"
        print(line)


def compare_scramble(rows: list[dict]) -> None:
    """Side-by-side normal vs scrambled for same ranker."""
    from collections import defaultdict

    by_ranker: dict[str, dict[str, list]] = defaultdict(lambda: {"normal": [], "scrambled": []})
    for row in rows:
        key = "scrambled" if row.get("scramble_genes", "").lower() == "true" else "normal"
        by_ranker[row["ranker"]][key].append(row)

    print("\nScramble ablation comparison  (avg_r5)")
    print("-" * 60)
    print(f"{'ranker':15s}  {'normal (last)':14s}  {'scrambled (last)':16s}  {'drop':6s}")
    print("-" * 60)

    for ranker, grp in sorted(by_ranker.items()):
        normals    = grp["normal"]
        scrambleds = grp["scrambled"]
        n_val = float(normals[-1]["avg_r5"])    if normals    and normals[-1]["avg_r5"]    else None
        s_val = float(scrambleds[-1]["avg_r5"]) if scrambleds and scrambleds[-1]["avg_r5"] else None
        n_str = f"{n_val:.4f}" if n_val is not None else "  —  "
        s_str = f"{s_val:.4f}" if s_val is not None else "  —  "
        drop  = f"{n_val - s_val:.4f}" if (n_val and s_val) else "  —  "
        print(f"{ranker:15s}  {n_str:14s}  {s_str:16s}  {drop}")


def show_run_detail(run_id_prefix: str) -> None:
    """Print full JSON for a matching run file."""
    matches = list(RUNS_DIR.glob(f"{run_id_prefix}*.json")) if RUNS_DIR.exists() else []
    if not matches:
        print(f"No run files matching '{run_id_prefix}'")
        return
    for path in sorted(matches):
        payload = json.loads(path.read_text())
        print(f"\n=== {path.name} ===")
        print(json.dumps(payload["config"], indent=2))
        for ranker, ds_dict in payload["results"].items():
            if ranker.startswith("_"):
                continue
            avg = ds_dict.get("_avg_ratio")
            print(f"  {ranker}  avg_r5={avg:.4f}" if avg else f"  {ranker}  avg_r5=—")


def main():
    parser = argparse.ArgumentParser(description="View benchmark results")
    parser.add_argument("--ranker",          default=None, help="Filter by ranker name")
    parser.add_argument("--compare-scramble",action="store_true",
                        help="Side-by-side normal vs scrambled")
    parser.add_argument("--detail",          default=None, metavar="RUN_ID_PREFIX",
                        help="Print full detail for a specific run")
    args = parser.parse_args()

    rows = load_summary()

    if not rows:
        print("No results yet. Run benchmark.py --auto-save to record results.")
        return

    if args.detail:
        show_run_detail(args.detail)
        return

    if args.ranker:
        rows = [r for r in rows if r["ranker"] == args.ranker]

    if args.compare_scramble:
        compare_scramble(rows)
    else:
        print_table(rows, title=f"All runs ({len(rows)} rows)")


if __name__ == "__main__":
    main()
