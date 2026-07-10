"""
agent_loop.py — Task-specialized gene-selection agent harness (no agent framework).

A thin, fully-controlled loop: each round the LLM picks an action (ml_rank / enrich / finish) via a
simple JSON protocol, we dispatch to the native Python tools, and feed the result back until it
commits a batch; then `reveal_fn` returns the results and the next round adapts.

Design goals (why no LangChain/Agno): exact prompt control, reproducibility, provider-agnostic
(LLM via llm_client — anthropic for the benchmark, pi/codex for deployment, mock for tests), and
ONE harness that serves both the benchmark and the scientist entry:
  - benchmark:  reveal_fn = the ground-truth oracle
  - deployment: reveal_fn = the scientist's wet-lab results

Usage:
    conda run -n waddington-bio python3 -m waddington_select.agent_loop --dataset Scharenberg22 \
        --rounds 3 --batch 15
"""

from __future__ import annotations

import argparse
import json
import re

from .oracle import ALL_DATASETS, BATCH_SIZES, DatasetOracle
from .llm_client import LLMClient
from . import tools

LLM_MODEL = "claude-haiku-4-5-20251001"
MAX_STEPS_PER_ROUND = 6

_ACTIONS = """\
Available actions (respond with exactly ONE, as JSON, nothing else):
  {"action": "ml_rank"}                 → see the ML model's current top candidate genes
  {"action": "enrich"}                  → GO/pathway enrichment of the hits found so far
  {"action": "finish", "batch": [...]}  → commit this round's batch of gene symbols to test
"""


def _parse_action(text: str) -> dict:
    """Extract the first JSON object from the model's reply; {} on failure."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return {}
    return {}


def _summarize(result: dict) -> str:
    if result.get("tool") == "ml_rank":
        cands = result.get("candidates", [])[:25]
        return "ML top candidates: " + ", ".join(
            f"{c['gene']}({c['score']:.2f})" for c in cands) or "none"
    if result.get("tool") == "enrich":
        if result.get("error"):
            return f"enrichment error: {result['error']}"
        terms = result.get("terms", [])[:8]
        return "Enriched pathways: " + "; ".join(
            f"{t['term']} (genes: {', '.join(t['overlap_genes'][:4])})" for t in terms) or "none"
    return json.dumps(result)[:300]


def _build_prompt(dataset: str, batch_size: int, round_idx: int, n_rounds: int,
                  hits: list[str], misses: list[str], action_log: list[dict]) -> str:
    task = DatasetOracle(dataset)
    hit_str = ", ".join(hits[:40]) + ("…" if len(hits) > 40 else "") if hits else "none yet"
    log_str = "\n".join(f"  - {a['action']}: {a['summary']}" for a in action_log) or "  (none yet)"
    return f"""You are running a sequential CRISPR screen for the phenotype '{dataset}'.
Goal: find as many hit genes as possible. This is round {round_idx + 1} of {n_rounds}.
You must commit a batch of exactly {batch_size} gene symbols this round (not yet tested).

State:
  hits found so far ({len(hits)}): {hit_str}
  genes tested so far: {len(hits) + len(misses)}

Actions you have taken THIS round:
{log_str}

{_ACTIONS}
Strategy: use ml_rank for the ML signal; once you have a few hits, use enrich to see which pathways
are active and prioritise untested genes in them; blend that with your biological knowledge of the
phenotype; then finish with the strongest batch. Do not repeat already-tested genes. Reply with ONE
JSON action only."""


def run_campaign(dataset: str, rounds: int = 5, batch_size: int | None = None,
                 llm: LLMClient | None = None, reveal_fn=None,
                 max_steps: int = MAX_STEPS_PER_ROUND, verbose: bool = True) -> dict:
    if dataset not in ALL_DATASETS:
        raise SystemExit(f"Unknown dataset '{dataset}'. Available: {', '.join(ALL_DATASETS)}")
    batch_size = batch_size or BATCH_SIZES[dataset]
    llm = llm or LLMClient(model=LLM_MODEL, temperature=0.2, max_tokens=1200)
    oracle = DatasetOracle(dataset)
    if reveal_fn is None:
        reveal_fn = lambda genes: oracle.reveal(genes)  # benchmark: ground-truth oracle

    hits: list[str] = []
    misses: list[str] = []
    tested: set[str] = set()
    trajectory = []

    for r in range(rounds):
        action_log: list[dict] = []
        batch: list[str] = []
        for _ in range(max_steps):
            prompt = _build_prompt(dataset, batch_size, r, rounds, hits, misses, action_log)
            step = _parse_action(llm.complete(prompt))
            action = step.get("action")

            if action == "finish":
                batch = tools._norm(step.get("batch", []))
                break
            elif action == "ml_rank":
                res = tools.ml_rank(dataset, n=max(batch_size * 3, 40),
                                    tested_hits=hits, tested_misses=misses, exclude=list(tested))
            elif action == "enrich":
                res = tools.enrich(hits, top=8)
            else:
                res = {"tool": "note", "msg": f"unrecognised action {step!r}; please finish."}
            action_log.append({"action": action or "?", "summary": _summarize(res)})

        # Commit: clean the batch (dedupe, drop already-tested), backfill from ML if short.
        batch = [g for g in dict.fromkeys(batch) if g not in tested]
        llm_proposed = len(batch)
        if len(batch) < batch_size:
            ml = tools.ml_rank(dataset, n=batch_size * 4, tested_hits=hits,
                               tested_misses=misses, exclude=list(tested))
            for c in ml["candidates"]:
                if len(batch) >= batch_size:
                    break
                if c["gene"] not in tested and c["gene"] not in batch:
                    batch.append(c["gene"])
        batch = batch[:batch_size]
        backfilled = max(0, len(batch) - llm_proposed)

        outcome = reveal_fn(batch)
        round_hits = [g for g, is_hit in outcome.items() if is_hit]
        hits += round_hits
        misses += [g for g, is_hit in outcome.items() if not is_hit]
        tested.update(batch)
        ratio = len(hits) / oracle.total_hits if oracle.total_hits else 0.0
        trajectory.append({"round": r + 1, "tested": len(batch), "hits": len(round_hits),
                           "cumulative": len(hits), "hit_ratio": round(ratio, 4),
                           "steps": len(action_log),
                           "actions": [a["action"] for a in action_log],
                           "llm_proposed": llm_proposed, "backfilled": backfilled})
        if verbose:
            print(f"  Round {r + 1}: actions={[a['action'] for a in action_log]} "
                  f"| batch {len(batch)} (LLM {llm_proposed} + ML backfill {backfilled}) "
                  f"→ {len(round_hits)} hits [{', '.join(round_hits[:8])}] "
                  f"| cumulative {len(hits)}/{oracle.total_hits} ({ratio:.1%})")
            for a in action_log:
                print(f"       · {a['action']}: {a['summary'][:160]}")

    return {"dataset": dataset, "rounds": rounds, "batch_size": batch_size,
            "cumulative_hits": len(hits), "total_hits": oracle.total_hits,
            "hit_ratio": trajectory[-1]["hit_ratio"] if trajectory else 0.0,
            "trajectory": trajectory}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-specialized gene-selection agent (no framework).")
    parser.add_argument("--dataset", required=True, choices=ALL_DATASETS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args()

    print(f"\nAgent campaign on '{args.dataset}' ({args.rounds} rounds):\n")
    result = run_campaign(args.dataset, rounds=args.rounds, batch_size=args.batch,
                          llm=LLMClient(model=args.model, temperature=0.2, max_tokens=1200))
    print(f"\n  Final: {result['cumulative_hits']}/{result['total_hits']} hits "
          f"(hit_ratio = {result['hit_ratio']:.3f})\n")


if __name__ == "__main__":
    main()
