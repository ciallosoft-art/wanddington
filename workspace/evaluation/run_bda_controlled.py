#!/usr/bin/env python3
"""
Run BioDiscoveryAgent's agent on our benchmark and score it with OUR oracle.

This rebuilds the controlled head-to-head harness behind
`workspace/results/bda_controlled/bda_gpt.json`, and extends it to BDA's other two difficulty tiers.
BDA's agent logic is untouched; three things are adapted, each for a stated reason:

  * LLM routing (`LLM.py:complete_text`) goes through our LLMClient, so BDA runs on the SAME model as
    our C-arm and reuses our OAuth. Patched at the dispatcher because several call sites hardcode
    models that no longer exist ("claude-1" in summarize_remaining_genes, "claude-v1" as FAST_MODEL).
  * The response parser tolerates a missing literal "Solution:" header on top-up re-prompts.
  * A round the agent could not fill is credited with the genes it did propose, rather than discarded.

Scoring is entirely ours: the agent's saved selections are replayed through `DatasetOracle`, so BDA
and every Waddington arm are measured against an identical hit definition and pool. That also means
the tier is chosen the same way as everywhere else — by WADDINGTON_TRAINING_CSV{,_V3}.

Two-gene note: BDA's shipped pair prompt hardcodes the wrong gene library (see the patch comment in
research_assistant.py). `--pair-pool real` substitutes Horlbeck's actual library (448 genes plus the
negative control); `--pair-pool shipped` leaves it as released. Run both — the difference is the cost
of that packaging bug.

Usage:
    conda run -n waddington-bio python3 workspace/evaluation/run_bda_controlled.py \
        --datasets IFNG IL2 --replicates 1 --out /tmp/bda_tier_a.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

BDA_DIR = Path(os.environ.get("WADDINGTON_BDA_REPO",
                              "/home/duanyu/Python/keypaper/code/BioDiscoveryAgent"))

from waddington_select.oracle import BATCH_SIZES, DatasetOracle  # noqa: E402
from waddington_select.tables import TRAINING_DATA_CSV as TRAINING_CSV  # noqa: E402
from waddington_select.tables import TRAINING_DATA_V3 as TRAINING_V3  # noqa: E402
from waddington_select.arms.llm_reasoning_arm import HARDCODED_TASKS  # noqa: E402

# our dataset name -> BDA's file stem
class QuotaExhausted(RuntimeError):
    """The provider refused the call because the account's usage limit was reached."""


BDA_NAME = {
    "IFNG": "IFNG",
    "IL2": "IL2",
    "Sanchez21": "Sanchez21",
    "Sanchez21_down": "Sanchez21_down",
    "Carnevale22": "Carnevale22_Adenosine",
    "Scharenberg22": "Scharenberg22",
    "Steinhart": "Steinhart_crispra_GD2_D22",
    "Replogle_K562_essential": "Replogle_K562_essential",
    "Replogle_K562_gwps": "Replogle_K562_gwps",
    "Horlbeck": "Horlbeck",
    # Horlbeck's Jurkat map (docs/jurkat_protocol.md). Ours, not BDA's: built by build_horlbeck_jurkat.py
    # in BDA's format, and copied into BDA's datasets/ by ensure_derived_benchmark before a run.
    "Horlbeck_Jurkat": "Horlbeck_Jurkat",
}
DERIVED_DIR = REPO_ROOT / "workspace" / "data" / "horlbeck2018"
DERIVED = {"Horlbeck_Jurkat"}


def ensure_derived_benchmark(dataset: str) -> None:
    """Put a benchmark we built (ground truth, hits, task prompt) where BDA's code reads its own."""
    if dataset not in DERIVED:
        return
    stem = BDA_NAME[dataset]
    for src, dst in ((DERIVED_DIR / f"ground_truth_{stem}.csv", BDA_DIR / "datasets" / f"ground_truth_{stem}.csv"),
                     (DERIVED_DIR / f"topmovers_{stem}.npy", BDA_DIR / "datasets" / f"topmovers_{stem}.npy"),
                     (DERIVED_DIR / "task_prompts" / f"{stem}.json",
                      BDA_DIR / "datasets" / "task_prompts" / f"{stem}.json")):
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            dst.write_bytes(src.read_bytes())
            print(f"    [benchmark] copied {src.name} into BDA's datasets/")


def ensure_task_prompt(dataset: str) -> None:
    """BDA ships no task prompt for the two Replogle screens; write ours so both see identical text."""
    stem = BDA_NAME[dataset]
    path = BDA_DIR / "datasets" / "task_prompts" / f"{stem}.json"
    if path.exists():
        return
    task = HARDCODED_TASKS.get(dataset)
    if task is None:
        raise SystemExit(f"no task prompt for {dataset} in BDA or in our HARDCODED_TASKS")
    path.write_text(json.dumps({"Task": task["Task"], "Measurement": task["Measurement"]}, indent=2))
    print(f"    [task prompt] wrote {path.name} from our HARDCODED_TASKS")


def horlbeck_pool(stem: str = "Horlbeck") -> str:
    gt = BDA_DIR / "datasets" / f"ground_truth_{stem}.csv"
    import pandas as pd
    pairs = [ast.literal_eval(s) for s in pd.read_csv(gt)["Gene_pairs"]]
    genes = sorted({g.upper() for p in pairs for g in p})
    return repr(genes)


def run_one(dataset: str, replicate: int, rounds: int, combinatorial: bool,
            pair_pool: str, model: str) -> dict:
    """One agent campaign; returns a result record in our standard shape."""
    oracle = DatasetOracle(dataset)
    batch = oracle.batch_size
    stem = BDA_NAME[dataset]
    ensure_derived_benchmark(dataset)
    ensure_task_prompt(dataset)

    workdir = Path(tempfile.mkdtemp(prefix=f"bda_{dataset}_{replicate}_"))
    log_root = workdir / "logs"
    task = "perturb-genes-brief-Horlbeck" if combinatorial else "perturb-genes-brief"

    cmd = [
        sys.executable, "research_assistant.py",
        "--task", task,
        "--data_name", stem,
        "--steps", str(rounds),
        "--num_genes", str(batch),
        "--model", model,
        "--log_dir", str(log_root),
        "--run_name", "test",
        "--python", sys.executable,
    ]
    if combinatorial:
        cmd += ["--combinatorial", "True"]

    env = dict(os.environ)
    env["WADDINGTON_BDA_MODEL"] = model
    # BDA imports its optional tool dependencies at module load, so the process will not start without
    # them even in the no-tools configuration we evaluate. _bda_stubs/ satisfies those imports and
    # raises loudly if one is ever actually called (see its README).
    stubs = Path(__file__).resolve().parent / "_bda_stubs"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stubs), str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if combinatorial and pair_pool == "real":
        env["WADDINGTON_BDA_PAIR_POOL"] = horlbeck_pool(stem)
    else:
        env.pop("WADDINGTON_BDA_PAIR_POOL", None)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=BDA_DIR, env=env, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    log_dir = Path(f"{log_root}_{stem}") / "test"
    cumulative: list[int] = []
    ratios: list[float] = []
    selected_per_round: list[list[str]] = []
    seen: set[str] = set()
    total = oracle.total_hits

    for r in range(1, rounds + 1):
        f = log_dir / f"sampled_genes_{r}.npy"
        if not f.exists():
            break
        arr = np.load(f, allow_pickle=True)
        picks = ([_token(row) for row in arr] if arr.ndim == 2 else
                 [str(x).strip().upper() for x in arr])
        selected_per_round.append(picks[len(seen):])   # BDA saves cumulatively
        seen = set(picks)
        revealed = oracle.reveal(list(seen))
        hits = sum(1 for v in revealed.values() if v)
        cumulative.append(hits)
        ratios.append(round(hits / total, 6) if total else 0.0)

    if not cumulative:
        blob = (proc.stdout or "") + (proc.stderr or "")
        # Surface an exhausted API quota as itself. It arrives inside the agent subprocess, so without
        # this it reaches the caller as a generic "no sampled_genes files" and gets truncated out of the
        # failure line — which reads like a harness bug and sends you debugging the wrong thing.
        if "usage limit" in blob.lower() or "rate limit" in blob.lower():
            raise QuotaExhausted(f"{dataset} rep{replicate}: API quota exhausted; nothing was recorded")
        raise RuntimeError(
            f"{dataset} rep{replicate}: BDA produced no sampled_genes files in {log_dir}\n"
            f"--- stdout tail ---\n{proc.stdout[-2000:]}\n--- stderr tail ---\n{proc.stderr[-2000:]}"
        )
    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "dataset": dataset,
        "arm_name": "bda",
        "batch_size": batch,
        "n_rounds": len(cumulative),
        "total_hits": total,
        "n_genes": oracle.n_genes,
        "hits_per_round": [cumulative[0]] + [b - a for a, b in zip(cumulative, cumulative[1:])],
        "cumulative_hits": cumulative,
        "hit_ratio_per_round": ratios,
        "auc_normalized": round(float(np.mean(ratios)), 6),
        "selected_genes_per_round": selected_per_round,
        "repeat_index": replicate,
        "provider_seed": None,
        "model": model,
        "combinatorial": combinatorial,
        "pair_pool": pair_pool if combinatorial else None,
        "seconds": round(elapsed, 1),
    }


def _token(row) -> str:
    """A saved pair row -> our canonical 'A|B' candidate token."""
    from waddington_select.tables import pair_token
    return pair_token(str(row[0]), str(row[1]))


def _sha256(path: Path) -> str | None:
    import hashlib
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(out_path: Path, args, datasets: list[str]) -> None:
    """Provenance sidecar, matching what run_sequential writes for our own arms.

    The external agent is the part a reader cannot inspect from this repo, so record exactly which
    revision of it ran and which of its files we modified — an unrecorded patch to BDA would make the
    comparison unauditable.
    """
    import datetime
    import subprocess

    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=BDA_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return None

    stubs = Path(__file__).resolve().parent / "_bda_stubs"
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol": {
            "agent": "BioDiscoveryAgent (no tools)",
            "datasets": datasets,
            "replicates": args.replicates,
            "rounds": args.rounds,
            "combinatorial": args.combinatorial,
            "pair_pool": args.pair_pool if args.combinatorial else None,
            "llm_model": args.model,
            "scored_by": "waddington_select.oracle.DatasetOracle",
            "batch_sizes": {ds: BATCH_SIZES[ds] for ds in datasets if ds in BATCH_SIZES},
        },
        "scientific_inputs": {
            "training_csv": str(TRAINING_CSV),
            "training_csv_sha256": _sha256(TRAINING_CSV),
            "training_csv_v3": str(TRAINING_V3),
            "training_csv_v3_sha256": _sha256(TRAINING_V3),
        },
        "external_agent": {
            "repo": str(BDA_DIR),
            "git_head": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "patched_files": {
                name: _sha256(BDA_DIR / name)
                for name in ("LLM.py", "tools.py", "research_assistant.py")
            },
            "patches": [
                "LLM.py:complete_text — all completions routed through our LLMClient at a fixed model",
                "tools.py:parse_entries — tolerate a missing 'Solution:' header on top-up re-prompts",
                "tools.py:agent_loop — credit an unfilled round instead of discarding it",
                "research_assistant.py — opt-in WADDINGTON_BDA_PAIR_POOL override for the pair library",
            ],
            "import_stubs": sorted(d.name for d in stubs.iterdir() if d.is_dir()) if stubs.exists() else [],
        },
    }
    mpath = out_path.with_name(f"{out_path.stem}.manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] {mpath}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--combinatorial", action="store_true")
    ap.add_argument("--pair-pool", choices=["real", "shipped"], default="real",
                    help="'shipped' keeps BDA's hardcoded 124-gene list (see module docstring)")
    ap.add_argument("--model", default="openai-codex/gpt-5.6-luna")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    results: dict[str, dict[str, list]] = {}
    if out_path.exists():                        # resume
        results = json.loads(out_path.read_text())

    failures: list[str] = []
    for ds in args.datasets:
        if ds not in BATCH_SIZES:
            raise SystemExit(f"{ds} is not a registered dataset; check WADDINGTON_EXTRA_DATASETS")
        # Resume on COMPLETE records only. A run cut short mid-campaign (an exhausted API quota is the
        # usual cause) leaves a record with fewer than `rounds` rounds; counting it as done would freeze
        # that screen at a partial trajectory forever, and quietly average a 2-round result against
        # 5-round ones. Drop the incomplete tail and redo it.
        kept = [r for r in results.get(ds, {}).get("bda", []) if r.get("n_rounds") == args.rounds]
        dropped = len(results.get(ds, {}).get("bda", [])) - len(kept)
        if dropped:
            print(f"  {ds}: discarding {dropped} incomplete record(s) and redoing them", flush=True)
            results.setdefault(ds, {})["bda"] = kept
        for rep in range(len(kept), args.replicates):
            print(f"  {ds} replicate {rep + 1}/{args.replicates} ...", flush=True)
            try:
                rec = run_one(ds, rep, args.rounds, args.combinatorial, args.pair_pool, args.model)
            except QuotaExhausted as exc:
                # No point walking the remaining screens: they will all hit the same wall.
                print(f"    QUOTA EXHAUSTED: {exc}", flush=True)
                failures.append(f"{ds} rep{rep} (quota)")
                print(f"\nSaved to {out_path}")
                print(f"INCOMPLETE: stopped early on API quota; {len(failures)} run(s) outstanding")
                raise SystemExit(4)
            except Exception as exc:                       # one screen failing must not abort the rest
                print(f"    FAILED: {type(exc).__name__}: {str(exc)[:1200]}", flush=True)
                failures.append(f"{ds} rep{rep}")
                break
            if rec["n_rounds"] != args.rounds:
                print(f"    INCOMPLETE ({rec['n_rounds']}/{args.rounds} rounds) — not recorded",
                      flush=True)
                failures.append(f"{ds} rep{rep} (incomplete)")
                break
            results.setdefault(ds, {}).setdefault("bda", []).append(rec)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(results, indent=2))
            tmp.replace(out_path)
            print(f"    hits {rec['cumulative_hits']}  hit@R5={rec['hit_ratio_per_round'][-1]:.4f} "
                  f"({rec['seconds']:.0f}s)", flush=True)

    write_manifest(out_path, args, args.datasets)
    print(f"\nSaved to {out_path}")
    if failures:
        print(f"INCOMPLETE: {len(failures)} run(s) still missing: {', '.join(failures)}")
        raise SystemExit(3)          # non-zero so a retry wrapper knows there is work left


if __name__ == "__main__":
    main()
