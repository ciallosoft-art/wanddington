"""
agent_benchmark.py — Evaluate the tool-using agent (agent_loop) against the pipeline baseline.

The agent is nondeterministic (it plans with an LLM), so — like PerTurboAgent — we run it several
times per dataset and report the mean hit_ratio@R(final), matched to the benchmark setup (batch =
the dataset's batch size, 5 rounds). We compare to the pipeline's waddington_c (three_arm.json).

Checkpoints after every dataset (survives rate limits / crashes; --resume continues).

Usage:
    conda run -n waddington-bio python3 -m waddington_select.agent_benchmark \
        --datasets Scharenberg22 Steinhart Carnevale22 --runs 2 --rounds 5 --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .oracle import ALL_DATASETS, BATCH_SIZES
from .agent_loop import run_campaign, LLM_MODEL
from .llm_client import LLMClient

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "workspace" / "results" / "sequential"
PIPELINE_BASELINE = RESULTS_DIR / "three_arm.json"


def _pipeline_r5(dataset: str) -> float | None:
    """waddington_c mean hit_ratio@R5 for a dataset, from the pipeline benchmark."""
    if not PIPELINE_BASELINE.exists():
        return None
    data = json.load(open(PIPELINE_BASELINE))
    arm = data.get(dataset, {}).get("waddington_c")
    if not arm:
        return None
    return sum(s["hit_ratio_per_round"][-1] for s in arm) / len(arm)


def run_benchmark(datasets, runs, rounds, model, out_path: Path, resume: bool) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    if resume and out_path.exists():
        results = json.load(open(out_path))
        done = [d for d in datasets if d in results]
        if done:
            print(f"[resume] skipping {len(done)} done: {', '.join(done)}")

    for ds in datasets:
        if ds in results:
            continue
        ratios = []
        for run in range(runs):
            llm = LLMClient(model=model, temperature=0.2, max_tokens=1200)
            res = run_campaign(ds, rounds=rounds, batch_size=BATCH_SIZES[ds],
                               llm=llm, verbose=False)
            ratios.append(res["hit_ratio"])
            print(f"  {ds}  run {run + 1}/{runs}: hit_ratio@R{rounds} = {res['hit_ratio']:.3f}")
        mean = sum(ratios) / len(ratios)
        results[ds] = {"runs": ratios, "mean": round(mean, 4)}
        json.dump(results, open(out_path, "w"), indent=2)
        base = _pipeline_r5(ds)
        base_str = f"{base:.3f}" if base is not None else "  n/a"
        print(f"  {ds:26s}  agent(mean)={mean:.3f}  pipeline={base_str}  "
              f"Δ={mean - base:+.3f}" if base is not None else "")

    # Summary
    print(f"\n{'='*60}\n{'Dataset':26s} {'agent':>7s} {'pipeline':>9s} {'Δ':>7s}\n{'-'*60}")
    ta = tp = 0.0
    n = 0
    for ds in datasets:
        if ds not in results:
            continue
        a = results[ds]["mean"]
        b = _pipeline_r5(ds)
        if b is None:
            print(f"{ds:26s} {a:7.3f} {'n/a':>9s}")
            continue
        ta += a
        tp += b
        n += 1
        print(f"{ds:26s} {a:7.3f} {b:9.3f} {a - b:+7.3f}")
    if n:
        print(f"{'-'*60}\n{'AVERAGE':26s} {ta/n:7.3f} {tp/n:9.3f} {(ta-tp)/n:+7.3f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the agent vs the pipeline.")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "agent_benchmark.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.datasets, args.runs, args.rounds, args.model, args.out, args.resume)


if __name__ == "__main__":
    main()
