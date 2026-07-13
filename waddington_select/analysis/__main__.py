"""
CLI for the analysis package.

    python -m waddington_select.analysis figures [--out workspace/results/figures]
    python -m waddington_select.analysis report --campaign <trace.json> [--out report.html]
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "workspace" / "results" / "figures"


def main() -> None:
    p = argparse.ArgumentParser(description="Figures and reports for the Waddington gene selector.")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("figures", help="quantification figures from the frozen benchmark results")
    f.add_argument("--out", type=Path, default=DEFAULT_OUT)

    r = sub.add_parser("report", help="HTML experiment report from a campaign trace")
    r.add_argument("--campaign", type=Path, required=True)
    r.add_argument("--out", type=Path, default=None)

    o = sub.add_parser("overview", help="shareable one-page overview of the whole project")
    o.add_argument("--out", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "overview":
        from .overview import build_overview
        print(f"Overview: {build_overview(args.out)}")
        return

    if args.cmd == "figures":
        from .figures import build_all
        made = build_all(args.out)
        print(f"Wrote {len(made)} figures to {args.out}:")
        for m in made:
            print(f"  {m.name}")
        return

    if args.cmd == "report":
        from .report import build_report
        out = args.out or args.campaign.with_suffix(".html")
        path = build_report(args.campaign, out)
        print(f"Report: {path}")


if __name__ == "__main__":
    main()
