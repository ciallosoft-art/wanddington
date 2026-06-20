#!/usr/bin/env python3
"""
Parallel prefetch of ARCHS4 co-expression data for DepMap anchor genes.
Run this before bootstrap_lgbm.py --include-depmap to speed up the first run.

Usage:
  python3 prefetch_archs4.py [--workers N]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPMAP_PROCESSED = REPO_ROOT / "workspace" / "data" / "depmap" / "processed"

sys.path.insert(0, str(Path(__file__).parent))
from gene_ranker import get_archs4_coexpr, ARCHS4_CACHE_DIR, build_anchor_scores, CACHE_DIR


def prefetch_archs4(genes: list[str], workers: int = 4) -> None:
    missing = [g for g in genes
               if not (ARCHS4_CACHE_DIR / f"{g.upper()}.json").exists()]
    print(f"ARCHS4: {len(genes)} total anchors, {len(missing)} not yet cached")
    if not missing:
        return

    done = 0

    def fetch(gene: str) -> tuple[str, int]:
        result = get_archs4_coexpr(gene, verbose=False)
        return gene, len(result)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch, g): g for g in missing}
        for fut in as_completed(futs):
            gene, n = fut.result()
            done += 1
            print(f"  [{done}/{len(missing)}] {gene}: {n} co-expressed genes", flush=True)

    print(f"ARCHS4 done — {done} genes fetched.\n")


def prefetch_ppi(genes: list[str], workers: int = 4) -> None:
    missing = [g for g in genes
               if not (CACHE_DIR / f"{g.upper()}.json").exists()]
    print(f"STRING PPI: {len(genes)} total anchors, {len(missing)} not yet cached")
    if not missing:
        return

    done = 0

    def fetch(gene: str) -> tuple[str, int]:
        scores = build_anchor_scores([gene], verbose=False)
        return gene, len(scores)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch, g): g for g in missing}
        for fut in as_completed(futs):
            gene, n = fut.result()
            done += 1
            print(f"  [{done}/{len(missing)}] {gene}: {n} PPI neighbors", flush=True)

    print(f"STRING PPI done — {done} genes fetched.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel fetch threads (default 4)")
    args = parser.parse_args()

    # Collect all unique DepMap anchor genes
    unique_anchors: set[str] = set()
    for anc_f in DEPMAP_PROCESSED.glob("anchors_DepMap_*.json"):
        unique_anchors.update(json.loads(anc_f.read_text()))

    print(f"Found {len(unique_anchors)} unique DepMap anchor genes\n")
    anchors = sorted(unique_anchors)

    prefetch_ppi(anchors, workers=args.workers)
    prefetch_archs4(anchors, workers=args.workers)

    print("All done — ready to run bootstrap_lgbm.py --include-depmap")


if __name__ == "__main__":
    main()
