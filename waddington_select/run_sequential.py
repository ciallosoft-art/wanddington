#!/usr/bin/env python3
"""
Sequential oracle evaluation — paper experiment runner.

Runs one or more arms across all BDA benchmark datasets for N evaluation replicates
and reports hit_ratio@R5 per dataset plus averages. New GPT work should use
``--replicates N --seed-mode local``; ``--seeds`` is a historical compatibility alias.

The LLM call is the wall-clock bottleneck (~1800 sequential calls x ~19s for a 20-repeat main run).
Set ``WADDINGTON_PARALLEL=N`` to run N repeats of the same arm concurrently (default 1 = sequential):
the Codex/OpenAI endpoint tolerates concurrency with flat per-call latency, so N=6-8 gives a ~6-8x
speedup. Results, checkpointing and ``--resume`` are unchanged (repeats still flush in index order);
deterministic arms are byte-identical regardless of N, and N is not part of the provenance manifest.

Usage:
    conda run -n waddington-bio python3 run_sequential.py \\
        --arms waddington_c llm_reasoning coreset \\
        --replicates 5 --seed-mode local

Arm names
---------
  random                  Random selection baseline
  coreset                 Coreset diversity-maximising (A arm)
  static_ranker           LOO LightGBM static prior
  online_adaptive         PerTurboAgent-style online ML
  llm_reasoning           GPT/LLM proposals with static LOO-ML padding (B arm)
  waddington_c            Waddington C (active GPT protocol; paper scripts pin legacy vote semantics)

Ablation arms:
  waddington_c_no_memory      C minus cross-experiment memory
  waddington_c_no_llm         C minus LLM (online ML only)
  waddington_c_no_ml          C minus online retraining (static LOO + LLM)
  waddington_c_shuffled_names   C with anonymous gene identifiers (A0: no info to reason over)
  waddington_c_feature_reasoning C with anonymous IDs + structural feature vectors (A1: reason, not recall)
"""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from .oracle import BENCHMARK_DATASETS, BATCH_SIZES, TRAINING_DATA_CSV, DatasetOracle
from .sequential_runner import RunResult, SequentialRunner

# ── Paper arms ──────────────────────────────────────────────────────────────
from .arms.random_arm import RandomArm
from .arms.static_ranker_arm import StaticRankerArm
from .arms.coreset_arm import CoresetArm
from .arms.online_adaptive_arm import OnlineAdaptiveArm
from .arms.llm_reasoning_arm import LLMReasoningArm
from .arms.waddington_c_arm import (
    DEFAULT_HONEST_WLLM,
    MEMORY_PATH,
    TRAINING_DATA_V1,
    TRAINING_DATA_V3,
    WaddingtonCArm,
    WaddingtonCSkillsArm,
    WaddingtonCMemSkillsArm,
    WaddingtonCEnrichArm,
    _drop_anchor_default,
    _feature_policy,
)
from .llm_client import (
    DEFAULT_BACKEND,
    DEFAULT_PI_CMD,
    DEFAULT_PI_MODEL,
    DEFAULT_PI_REASONING,
    default_model_for_backend,
)

# ── Ablation arms ────────────────────────────────────────────────────────────
from .arms.waddington_c_no_memory_arm import WaddingtonCNoMemoryArm
from .arms.waddington_c_no_llm_arm import WaddingtonCNoLLMArm
from .arms.waddington_c_no_ml_arm import WaddingtonCNoMLArm
from .arms.waddington_c_shuffled_names_arm import WaddingtonCShuffledNamesArm
from .arms.waddington_c_feature_reasoning_arm import WaddingtonCFeatureReasoningArm, WaddingtonCPoolOnlyArm, WaddingtonCLinearArm
from .arms.waddington_c_ensemble_arm import WaddingtonCEnsembleArm
from .arms.genedisco_arm import GeneDiscoArm, ACQUISITIONS as _GD_ACQ

RESULTS_DIR = REPO_ROOT / "workspace" / "results" / "sequential"

# Arms available via --arms flag
_PAPER_ARMS = {
    "random", "coreset", "static_ranker", "online_adaptive",
    "llm_reasoning", "waddington_c", "waddington_c_skills", "waddington_c_memskills", "waddington_c_enrich",
    "waddington_c_no_memory", "waddington_c_no_llm",
    "waddington_c_no_ml", "waddington_c_shuffled_names",
    "waddington_c_feature_reasoning", "waddington_c_pool_only", "waddington_c_ensemble", "waddington_c_linear",
} | {f"genedisco_{a}" for a in _GD_ACQ}


def make_arms(dataset_name: str, arm_names: list[str], local_seed: int = 42) -> list:
    oracle = DatasetOracle(dataset_name)
    pool = oracle.all_genes()
    bs = BATCH_SIZES[dataset_name]
    arms = []
    for name in arm_names:
        if name == "random":
            arms.append(RandomArm(dataset_name, bs, pool, seed=local_seed))
        elif name == "static_ranker":
            arms.append(StaticRankerArm(dataset_name, bs))
        elif name == "coreset":
            arms.append(CoresetArm(dataset_name, bs, seed=local_seed))
        elif name == "online_adaptive":
            arms.append(OnlineAdaptiveArm(dataset_name, bs))
        elif name == "llm_reasoning":
            arms.append(LLMReasoningArm(dataset_name, bs, seed=local_seed))
        elif name == "waddington_c":
            arms.append(WaddingtonCArm(dataset_name, bs))
        elif name == "waddington_c_skills":
            arms.append(WaddingtonCSkillsArm(dataset_name, bs))
        elif name == "waddington_c_memskills":
            arms.append(WaddingtonCMemSkillsArm(dataset_name, bs))
        elif name == "waddington_c_enrich":
            arms.append(WaddingtonCEnrichArm(dataset_name, bs))
        elif name == "waddington_c_no_memory":
            arms.append(WaddingtonCNoMemoryArm(dataset_name, bs))
        elif name == "waddington_c_no_llm":
            arms.append(WaddingtonCNoLLMArm(dataset_name, bs))
        elif name == "waddington_c_no_ml":
            arms.append(WaddingtonCNoMLArm(dataset_name, bs))
        elif name == "waddington_c_shuffled_names":
            arms.append(WaddingtonCShuffledNamesArm(dataset_name, bs))
        elif name == "waddington_c_feature_reasoning":
            arms.append(WaddingtonCFeatureReasoningArm(dataset_name, bs))
        elif name == "waddington_c_pool_only":
            arms.append(WaddingtonCPoolOnlyArm(dataset_name, bs))
        elif name == "waddington_c_linear":
            arms.append(WaddingtonCLinearArm(dataset_name, bs))
        elif name == "waddington_c_ensemble":
            arms.append(WaddingtonCEnsembleArm(dataset_name, bs))
        elif name.startswith("genedisco_"):
            arms.append(GeneDiscoArm(
                dataset_name, bs, acquisition=name[len("genedisco_"):], seed=local_seed,
            ))
        else:
            print(f"    [WARN] Unknown arm '{name}', skipping")
    return arms


def print_result(r: RunResult) -> None:
    ratios = "  ".join(f"R{i+1}={v:.3f}" for i, v in enumerate(r.hit_ratio_per_round))
    print(
        f"  {r.arm_name:30s}  {ratios}  "
        f"AUC_norm={r.auc_normalized:.3f}  hits={r.cumulative_hits[-1]}/{r.total_hits}"
    )


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write a checkpoint atomically so interruption cannot leave truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip())
        return head, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _runtime_fingerprint() -> dict:
    """Hash the executable closure and scientific inputs used by a run.

    This is intentionally narrower than the whole dirty worktree: unrelated documentation or router
    scratch JSON must not block resume, while any Python/bridge/data change that can alter selection
    must. Paths are stable workspace-relative labels even when task_prompts is a symlink.
    """
    candidates: list[tuple[str, Path]] = []
    for path in sorted((REPO_ROOT / "waddington_select").rglob("*.py")):
        if "__pycache__" not in path.parts:
            candidates.append((str(path.relative_to(REPO_ROOT)), path))
    frontend = REPO_ROOT / "frontend"
    for pattern in ("src/**/*.mjs", "bin/*.js", "package.json", "package-lock.json"):
        for path in sorted(frontend.glob(pattern)):
            if path.is_file():
                candidates.append((str(path.relative_to(REPO_ROOT)), path))
    # Hash the actually installed Codex adapter as well as package-lock. This closes the gap where
    # node_modules differs from the lockfile while every GPT request imports the installed code.
    pi_ai = frontend / "node_modules" / "@earendil-works" / "pi-ai"
    package_json = pi_ai / "package.json"
    if package_json.is_file():
        candidates.append(("installed:pi-ai/package.json", package_json))
    for path in sorted((pi_ai / "dist").rglob("*.js")):
        relative = path.relative_to(pi_ai)
        candidates.append((f"installed:pi-ai/{relative}", path))
    for relative in ("pyproject.toml", "environment.yml"):
        path = REPO_ROOT / relative
        if path.is_file():
            candidates.append((relative, path))

    scientific_inputs = {
        "oracle_training_csv": TRAINING_DATA_CSV,
        "carm_training_v1": TRAINING_DATA_V1,
        "carm_training_v3": TRAINING_DATA_V3,
        "phenotype_registry": REPO_ROOT / "workspace" / "data" / "user_phenotypes" / "registry.json",
    }
    for label, path in scientific_inputs.items():
        if path.exists():
            candidates.append((f"input:{label}:{path}", path))
    task_dir = REPO_ROOT / "workspace" / "data" / "bda_benchmark" / "task_prompts"
    for path in sorted(task_dir.glob("*.json")):
        candidates.append((f"input:task_prompt:{path.name}", path))

    hashes = {label: _file_sha256(path) for label, path in candidates}
    versions: dict[str, str | None] = {"python": sys.version}
    for package in ("lightgbm", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        versions["node"] = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        versions["node"] = None
    payload = {"files": hashes, "environment": versions}
    aggregate = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"sha256": aggregate, **payload}


def _protocol_manifest(
    datasets: list[str], arm_names: list[str], n_rounds: int, n_repeats: int, seed_mode: str,
) -> dict:
    backend = os.environ.get("WADDINGTON_LLM_BACKEND", DEFAULT_BACKEND)
    if backend == "pi":
        model = os.environ.get("WADDINGTON_PI_MODEL", DEFAULT_PI_MODEL)
        reasoning = os.environ.get("WADDINGTON_PI_REASONING", DEFAULT_PI_REASONING)
    else:
        model = default_model_for_backend(backend)
        reasoning = None
    uses_memory = any(
        arm.startswith("waddington") and arm not in {"waddington_c_no_memory", "waddington_c_no_llm"}
        for arm in arm_names
    )
    memory_hash = None
    if uses_memory and MEMORY_PATH.exists():
        memory_hash = hashlib.sha256(MEMORY_PATH.read_bytes()).hexdigest()
    legacy_router = os.environ.get("WADDINGTON_LEGACY_ROUTER") == "1"
    force_wllm = os.environ.get("WADDINGTON_FORCE_WLLM")
    return {
        "datasets": datasets,
        "arms": arm_names,
        "rounds": n_rounds,
        "repeats": n_repeats,
        "cli_option": "--replicates (legacy alias: --seeds)",
        "seed_mode": seed_mode,
        "seed_semantics": (
            "repeat index seeds only arms that actually consume a local RNG; deterministic arms are "
            "repeated evaluations. The remote GPT endpoint exposes no sampling seed, so LLM-backed "
            "arms are repeated API invocations and independence is not assumed"
            if seed_mode == "local" else
            "historical protocol: local stochastic arms are reconstructed with fixed seed 42; "
            "LLM-backed arms are repeated API calls"
        ),
        "llm_backend": backend,
        "llm_model": model,
        "llm_reasoning": reasoning,
        "codex_reasoning_effective": (
            "none is injected explicitly into the Codex Responses payload"
            if backend == "pi" and model.startswith("openai-codex/") and reasoning == "none"
            else "pi-ai maps requested minimal to Codex low"
            if backend == "pi" and model.startswith("openai-codex/") and reasoning == "minimal"
            else reasoning
        ),
        "codex_temperature_policy": (
            "requested arm temperatures are omitted because the openai-codex endpoint does not "
            "support temperature"
            if backend == "pi" and model.startswith("openai-codex/") else None
        ),
        "codex_max_tokens_policy": (
            "the ChatGPT/Codex OAuth endpoint rejects max_output_tokens; strict prompt contracts bound "
            "the response, while requested max_tokens is not enforced server-side"
            if backend == "pi" and model.startswith("openai-codex/") else None
        ),
        "pi_command": os.environ.get("WADDINGTON_PI_CMD", DEFAULT_PI_CMD),
        "memory_path": str(MEMORY_PATH) if uses_memory else None,
        "memory_sha256": memory_hash,
        "legacy_router": legacy_router,
        "force_route": os.environ.get("WADDINGTON_FORCE_ROUTE"),
        "force_wllm": force_wllm,
        "effective_wllm": (
            float(force_wllm) if force_wllm is not None else
            DEFAULT_HONEST_WLLM if not legacy_router else "dataset-dependent legacy route"
        ),
        "feature_policy": _feature_policy(),
        "drop_anchor_features": _drop_anchor_default(),
        "hit_frequency_only": os.environ.get("WADDINGTON_HITFREQ_ONLY") == "1",
        "exclude_study": os.environ.get("WADDINGTON_EXCLUDE_STUDY") == "1",
        "training_csv_override": os.environ.get("WADDINGTON_TRAINING_CSV"),
        # v3 (with-DepMap) is overridable too, and most screens route through it, so record it
        # alongside v1 rather than leaving half the table provenance implicit.
        "training_csv_v3_override": os.environ.get("WADDINGTON_TRAINING_CSV_V3"),
        "reason_from_structure": os.environ.get("WADDINGTON_REASON_FROM_STRUCTURE") == "1",
        "gene_level_audit": True,
        "dump_revealed_hit_genes": os.environ.get("WADDINGTON_DUMP_SELECTIONS") == "1",
        "full_selected_context": os.environ.get("WADDINGTON_FULL_SELECTED_CONTEXT", "1") == "1",
        "llm_vote_policy": (
            "gpt_named_only" if os.environ.get("WADDINGTON_LLM_NAMED_ONLY", "1") == "1"
            else "gpt_named_plus_static_padding_legacy"
        ),
        "cross_experiment_split": (
            "leave_one_study_out" if os.environ.get("WADDINGTON_EXCLUDE_STUDY") == "1"
            else "leave_one_phenotype_out"
        ),
        "same_study_siblings_allowed": os.environ.get("WADDINGTON_EXCLUDE_STUDY") != "1",
        "process_determinism": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "runtime_fingerprint": _runtime_fingerprint(),
        "remote_model_alias_mutable": backend == "pi" and model.startswith("openai-codex/"),
    }


def _write_or_validate_manifest(save_path: Path, protocol: dict, resume: bool) -> Path:
    manifest_path = save_path.with_name(f"{save_path.stem}.manifest.json")
    now = datetime.now(timezone.utc).isoformat()
    head, dirty = _git_state()
    session = {"started_at": now, "git_commit": head, "git_dirty": dirty}
    if resume and save_path.exists() and not manifest_path.exists():
        raise ValueError(
            f"refusing to resume {save_path}: checkpoint exists without a provenance manifest"
        )
    if resume and manifest_path.exists():
        with open(manifest_path) as f:
            previous = json.load(f)
        if previous.get("protocol") != protocol:
            raise ValueError(
                f"refusing to resume {save_path}: configuration differs from {manifest_path}"
            )
        previous.setdefault("sessions", []).append(session)
        _atomic_write_json(manifest_path, previous)
        return manifest_path
    _atomic_write_json(manifest_path, {
        "created_at": now,
        "git_commit": head,
        "git_dirty": dirty,
        "sessions": [session],
        "protocol": protocol,
    })
    return manifest_path


def _parallel_workers() -> int:
    """WADDINGTON_PARALLEL=N runs N repeats of the same arm concurrently (default 1 = sequential).
    The LLM call is the bottleneck and the provider tolerates concurrency with flat per-call latency;
    correctness is unaffected because each repeat builds a fresh arm and the oracle is a read-only lookup.
    Results are still checkpointed in repeat-index order, so resume is identical to the sequential path."""
    try:
        n = int(os.environ.get("WADDINGTON_PARALLEL", "1"))
    except ValueError:
        n = 1
    return max(1, n)


def _run_one_repeat(ds: str, arm_name: str, repeat_idx: int, oracle, n_rounds: int, seed_mode: str):
    """One independent (dataset, arm, repeat) campaign -> its result dict, or None if the arm is
    unavailable. Builds a fresh arm and runner, so it is safe to call concurrently for distinct repeats."""
    local_seed = repeat_idx if seed_mode == "local" else 42
    arms = make_arms(ds, [arm_name], local_seed=local_seed)
    if not arms:
        return None
    arm = arms[0]
    runner = SequentialRunner(arm, oracle, n_rounds=n_rounds)
    r = runner.run(seed=repeat_idx)
    item = _result_to_dict(r)
    item["repeat_index"] = repeat_idx
    consumes_local_rng = arm_name == "random" or arm_name.startswith("genedisco_")
    item["local_seed"] = local_seed if consumes_local_rng else None
    item["provider_seed"] = None if (
        arm_name == "llm_reasoning" or arm_name.startswith("waddington")
    ) else "not_applicable"
    item["replicate_semantics"] = (
        "local_rng_seeded" if consumes_local_rng else
        "remote_api_invocation; independence_not_assumed" if (
            arm_name == "llm_reasoning" or arm_name.startswith("waddington")
        ) else "deterministic_re_evaluation"
    )
    item["completed_at"] = datetime.now(timezone.utc).isoformat()
    return item


def run_all(
    datasets: list[str],
    arm_names: list[str],
    n_rounds: int,
    n_seeds: int,
    out_path: Path | None = None,
    resume: bool = False,
    seed_mode: str = "legacy",
) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = out_path or (RESULTS_DIR / "results.json")

    unknown_arms = sorted(set(arm_names) - _PAPER_ARMS)
    if unknown_arms:
        raise ValueError(f"unknown arm(s): {', '.join(unknown_arms)}")
    unknown_datasets = sorted(set(datasets) - set(BENCHMARK_DATASETS))
    if unknown_datasets:
        raise ValueError(f"dataset(s) have no benchmark oracle: {', '.join(unknown_datasets)}")
    if len(set(arm_names)) != len(arm_names) or len(set(datasets)) != len(datasets):
        raise ValueError("duplicate datasets/arms are not allowed")
    if n_rounds < 1 or n_seeds < 1:
        raise ValueError("rounds and replicates must both be positive")

    protocol = _protocol_manifest(datasets, arm_names, n_rounds, n_seeds, seed_mode)
    # A fresh run must never leave an old result beside a newly written manifest. Initialise the
    # checkpoint first; if manifest creation then fails, resume will reject the mismatched/old manifest.
    if not resume:
        _atomic_write_json(save_path, {})
    manifest_path = _write_or_validate_manifest(save_path, protocol, resume)
    print(f"[manifest] {manifest_path}")

    # --- Resume: load existing per-repeat checkpoints ---
    all_results: dict[str, dict[str, list]] = {}
    if resume and save_path.exists():
        with open(save_path) as f:
            all_results = json.load(f)
        if not isinstance(all_results, dict) or not set(all_results).issubset(datasets):
            raise ValueError("checkpoint contains unexpected dataset keys")
        for dataset, dataset_results in all_results.items():
            if not isinstance(dataset_results, dict) or not set(dataset_results).issubset(arm_names):
                raise ValueError(f"checkpoint contains unexpected arm keys for {dataset}")

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds}  (batch={BATCH_SIZES[ds]}, rounds={n_rounds})")
        print(f"{'='*60}")

        oracle = DatasetOracle(ds)
        print(f"  Genes: {oracle.n_genes}, Hits: {oracle.total_hits}")

        ds_results = all_results.setdefault(ds, {})

        for arm_name in arm_names:
            stored_results = ds_results.setdefault(arm_name, [])
            if len(stored_results) > n_seeds:
                raise ValueError(
                    f"checkpoint has {len(stored_results)} repeats for {ds}/{arm_name}, "
                    f"but this run requests {n_seeds}"
                )
            for expected_idx, prior in enumerate(stored_results):
                if prior.get("n_rounds") != n_rounds:
                    raise ValueError(f"checkpoint round mismatch for {ds}/{arm_name}")
                if prior.get("dataset") != ds or prior.get("arm_name") != arm_name:
                    raise ValueError(f"checkpoint identity mismatch for {ds}/{arm_name}")
                if prior.get("repeat_index") != expected_idx:
                    raise ValueError(f"checkpoint repeat_index mismatch for {ds}/{arm_name}")
                for field in ("hits_per_round", "cumulative_hits", "hit_ratio_per_round"):
                    values = prior.get(field)
                    if not isinstance(values, list) or len(values) != n_rounds:
                        raise ValueError(f"checkpoint {field} is incomplete for {ds}/{arm_name}")
                for field in (
                    "selected_genes_per_round", "llm_named_proposals_per_round",
                    "llm_static_padding_proposals_per_round", "selected_llm_named_genes_per_round",
                    "selected_without_llm_vote_per_round",
                ):
                    values = prior.get(field)
                    if not isinstance(values, list) or len(values) != n_rounds:
                        raise ValueError(f"checkpoint {field} is incomplete for {ds}/{arm_name}")

            if stored_results:
                print(f"  [resume] {ds}/{arm_name}: {len(stored_results)}/{n_seeds} repeats complete")

            remaining = list(range(len(stored_results), n_seeds))
            workers = _parallel_workers()
            if workers <= 1 or len(remaining) <= 1:
                for repeat_idx in remaining:
                    item = _run_one_repeat(ds, arm_name, repeat_idx, oracle, n_rounds, seed_mode)
                    if item is None:
                        continue
                    stored_results.append(item)
                    _atomic_write_json(save_path, all_results)
                    print(f"    [checkpoint] {arm_name} repeat {repeat_idx + 1}/{n_seeds}", flush=True)
            else:
                # Run repeats concurrently but flush to the checkpoint strictly in repeat-index order,
                # so an interrupted parallel run resumes exactly like the sequential path.
                buffer: dict[int, dict] = {}
                next_flush = len(stored_results)
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(_run_one_repeat, ds, arm_name, i, oracle, n_rounds, seed_mode): i
                        for i in remaining
                    }
                    for fut in concurrent.futures.as_completed(futures):
                        buffer[futures[fut]] = fut.result()
                        while next_flush in buffer:
                            done = buffer.pop(next_flush)
                            if done is not None:
                                stored_results.append(done)
                                _atomic_write_json(save_path, all_results)
                                print(f"    [checkpoint] {arm_name} repeat {next_flush + 1}/{n_seeds} (parallel x{workers})", flush=True)
                            next_flush += 1

            if not stored_results:
                continue
            avg_result = _average_results([_result_from_dict(r) for r in stored_results])
            print_result(avg_result)
        print(f"  [dataset complete] {save_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY  (hit_ratio @ final round, avg across repeated runs)")
    print(f"{'='*60}")
    header = f"{'Dataset':28s}" + "".join(f"  {a:>22s}" for a in arm_names)
    print(header)
    for ds in datasets:
        row = f"{ds:28s}"
        for arm_name in arm_names:
            if arm_name in all_results.get(ds, {}):
                seed_results = all_results[ds][arm_name]
                avg_ratio = sum(r["hit_ratio_per_round"][-1] for r in seed_results) / len(seed_results)
                row += f"  {avg_ratio:>22.3f}"
            else:
                row += f"  {'N/A':>22s}"
        print(row)

    print(f"\nResults saved to {save_path}")
    return all_results


def _average_results(results: list[RunResult]) -> RunResult:
    n = len(results)
    r0 = results[0]
    avg = RunResult(
        dataset=r0.dataset,
        arm_name=r0.arm_name,
        batch_size=r0.batch_size,
        n_rounds=r0.n_rounds,
        total_hits=r0.total_hits,
        n_genes=r0.n_genes,
    )
    n_rounds = r0.n_rounds
    avg.hits_per_round = [
        round(sum(r.hits_per_round[i] for r in results) / n)
        for i in range(n_rounds)
    ]
    avg.cumulative_hits = [
        round(sum(r.cumulative_hits[i] for r in results) / n)
        for i in range(n_rounds)
    ]
    avg.hit_ratio_per_round = [
        round(sum(r.hit_ratio_per_round[i] for r in results) / n, 4)
        for i in range(n_rounds)
    ]
    avg.auc_normalized = round(sum(r.auc_normalized for r in results) / n, 4)
    if any(r.llm_fallback_per_round for r in results):
        avg.llm_fallback_per_round = [
            round(sum(r.llm_fallback_per_round[i] for r in results) / n)
            for i in range(n_rounds)
        ]
    # Per-response diagnostics are not meaningfully averaged; retain them only in individual runs.
    return avg


def _result_from_dict(data: dict) -> RunResult:
    return RunResult(
        dataset=data["dataset"],
        arm_name=data["arm_name"],
        batch_size=data["batch_size"],
        n_rounds=data["n_rounds"],
        total_hits=data["total_hits"],
        n_genes=data["n_genes"],
        hits_per_round=data.get("hits_per_round", []),
        cumulative_hits=data.get("cumulative_hits", []),
        hit_ratio_per_round=data.get("hit_ratio_per_round", []),
        auc_normalized=data.get("auc_normalized", 0.0),
        revealed_hits=data.get("revealed_hits", []),
        selected_genes_per_round=data.get("selected_genes_per_round", []),
        llm_named_proposals_per_round=data.get("llm_named_proposals_per_round", []),
        llm_static_padding_proposals_per_round=data.get("llm_static_padding_proposals_per_round", []),
        selected_llm_named_genes_per_round=data.get("selected_llm_named_genes_per_round", []),
        selected_without_llm_vote_per_round=data.get("selected_without_llm_vote_per_round", []),
        llm_fallback_per_round=data.get("llm_fallback_per_round", []),
        llm_diagnostics_per_round=data.get("llm_diagnostics_per_round", []),
    )


def _result_to_dict(r: RunResult) -> dict:
    return {
        "dataset": r.dataset,
        "arm_name": r.arm_name,
        "batch_size": r.batch_size,
        "n_rounds": r.n_rounds,
        "total_hits": r.total_hits,
        "n_genes": r.n_genes,
        "hits_per_round": r.hits_per_round,
        "cumulative_hits": r.cumulative_hits,
        "hit_ratio_per_round": r.hit_ratio_per_round,
        "auc_normalized": r.auc_normalized,
        "selected_genes_per_round": r.selected_genes_per_round,
        "llm_named_proposals_per_round": r.llm_named_proposals_per_round,
        "llm_static_padding_proposals_per_round": r.llm_static_padding_proposals_per_round,
        "selected_llm_named_genes_per_round": r.selected_llm_named_genes_per_round,
        "selected_without_llm_vote_per_round": r.selected_without_llm_vote_per_round,
        **({"llm_fallback_per_round": r.llm_fallback_per_round}
           if r.llm_fallback_per_round else {}),
        **({"llm_diagnostics_per_round": r.llm_diagnostics_per_round}
           if r.llm_diagnostics_per_round else {}),
        # Gene identities of the hits found, dumped only when requested (for the novel-hit analysis);
        # off by default so ordinary result files stay compact.
        **({"revealed_hits": r.revealed_hits}
           if os.environ.get("WADDINGTON_DUMP_SELECTIONS") == "1" else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequential oracle evaluation for paper experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Paper arms:  " + "  ".join(sorted(_PAPER_ARMS)),
    )
    parser.add_argument("--datasets", nargs="+", default=BENCHMARK_DATASETS,
                        help="Datasets to evaluate (default: the 9 benchmark screens with ground "
                             "truth; registered user phenotypes have no oracle and are excluded)")
    parser.add_argument("--arms", nargs="+",
                        default=["coreset", "llm_reasoning", "waddington_c"],
                        help="Arms to run")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of selection rounds (default: 5)")
    parser.add_argument("--replicates", "--seeds", dest="replicates", type=int, default=5,
                        help="Number of evaluation replicates (--seeds is a legacy alias; default: 5)")
    parser.add_argument(
        "--seed-mode", choices=("legacy", "local"), default="legacy",
        help="legacy=fixed local seed 42 (paper compatibility); local=seed supported local arms by replicate index",
    )
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (default: workspace/results/sequential/results.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing checkpoint: skip already-completed datasets")
    args = parser.parse_args()

    unknown_arms = sorted(set(args.arms) - _PAPER_ARMS)
    if unknown_arms:
        parser.error(f"unknown arm(s): {', '.join(unknown_arms)}")
    unknown_datasets = sorted(set(args.datasets) - set(BENCHMARK_DATASETS))
    if unknown_datasets:
        parser.error(f"dataset(s) have no benchmark oracle: {', '.join(unknown_datasets)}")
    if args.rounds < 1 or args.replicates < 1:
        parser.error("--rounds and --replicates must both be positive")
    if len(set(args.arms)) != len(args.arms) or len(set(args.datasets)) != len(args.datasets):
        parser.error("duplicate datasets/arms are not allowed")

    run_all(
        args.datasets, args.arms, args.rounds, args.replicates, args.out, args.resume,
        seed_mode=args.seed_mode,
    )


if __name__ == "__main__":
    main()
