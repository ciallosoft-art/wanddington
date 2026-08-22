"""Fail-closed validation and merge for the staged GPT experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .oracle import BENCHMARK_DATASETS
from .run_sequential import _runtime_fingerprint

CALIBRATION_ARMS = (
    "random", "coreset", "static_ranker", "online_adaptive", "llm_reasoning",
)
C_ARM = "waddington_c"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _manifest_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}.manifest.json")


def _load_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_run_manifest(result_path: Path, label: str) -> tuple[Path, dict, dict]:
    manifest_path = _manifest_path(result_path)
    manifest = _load_json_object(manifest_path, f"{label} manifest")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{label} manifest has no protocol object: {manifest_path}")
    return manifest_path, manifest, protocol


def _validated_repeats(data: dict, dataset: str, arm: str, n: int = 5) -> list[dict]:
    repeats = data.get(dataset, {}).get(arm)
    if not isinstance(repeats, list) or len(repeats) != n:
        got = len(repeats) if isinstance(repeats, list) else 0
        raise ValueError(f"{dataset}/{arm}: expected {n} repeats, found {got}")
    for idx, result in enumerate(repeats):
        if not isinstance(result, dict):
            raise ValueError(f"{dataset}/{arm}: repeat {idx} is not an object")
        if result.get("dataset") != dataset or result.get("arm_name") != arm:
            raise ValueError(f"{dataset}/{arm}: embedded identity mismatch at repeat {idx}")
        if result.get("repeat_index") != idx or result.get("n_rounds") != 5:
            raise ValueError(f"{dataset}/{arm}: invalid repeat {idx}")
        if len(result.get("hit_ratio_per_round", [])) != 5:
            raise ValueError(f"{dataset}/{arm}: incomplete trajectory at repeat {idx}")
        for field in (
            "selected_genes_per_round", "llm_named_proposals_per_round",
            "llm_static_padding_proposals_per_round", "selected_llm_named_genes_per_round",
            "selected_without_llm_vote_per_round",
        ):
            rounds = result.get(field)
            if not isinstance(rounds, list) or len(rounds) != 5:
                raise ValueError(f"{dataset}/{arm}: missing gene-level {field} at repeat {idx}")
        for round_idx, selected in enumerate(result["selected_genes_per_round"]):
            if not isinstance(selected, list) or len(selected) != result.get("batch_size"):
                raise ValueError(f"{dataset}/{arm}: invalid selected batch at repeat {idx}, round {round_idx}")
            named = result["llm_named_proposals_per_round"][round_idx]
            padding = result["llm_static_padding_proposals_per_round"][round_idx]
            selected_named = result["selected_llm_named_genes_per_round"][round_idx]
            selected_without_vote = result["selected_without_llm_vote_per_round"][round_idx]
            if any(not isinstance(value, list) for value in (named, padding, selected_named, selected_without_vote)):
                raise ValueError(f"{dataset}/{arm}: invalid gene audit lists at repeat {idx}")
            if arm in {"llm_reasoning", C_ARM}:
                selected_set = set(selected)
                if set(selected_named) != selected_set & set(named):
                    raise ValueError(f"{dataset}/{arm}: inconsistent selected named genes at repeat {idx}")
                if set(selected_without_vote) != selected_set - set(named):
                    raise ValueError(f"{dataset}/{arm}: inconsistent non-voted selected genes at repeat {idx}")
                if arm == "llm_reasoning" and set(selected) != set(named) | set(padding):
                    raise ValueError(f"{dataset}/{arm}: B batch differs from named+padding proposals")
        if arm in {"llm_reasoning", C_ARM} and result.get("provider_seed", "missing") is not None:
            raise ValueError(f"{dataset}/{arm}: provider_seed must be explicit null at repeat {idx}")
    return repeats


def _validate_protocol(protocol: dict, expected_arms: tuple[str, ...], label: str,
                       expected_repeats: int = 5) -> None:
    expected = {
        "datasets": list(BENCHMARK_DATASETS),
        "arms": list(expected_arms),
        "rounds": 5,
        "repeats": expected_repeats,
        "seed_mode": "local",
        "llm_backend": "pi",
        "llm_model": "openai-codex/gpt-5.6-luna",
        "llm_reasoning": "none",
        "full_selected_context": True,
        "llm_vote_policy": "gpt_named_only",
        "legacy_router": False,
        "force_route": None,
        "force_wllm": None,
        "effective_wllm": 0.2,
        "feature_policy": "metadata",
        "drop_anchor_features": True,
        "hit_frequency_only": False,
        "exclude_study": False,
        "training_csv_override": None,
        "reason_from_structure": False,
        "gene_level_audit": True,
        "dump_revealed_hit_genes": True,
        "cross_experiment_split": "leave_one_phenotype_out",
        "same_study_siblings_allowed": True,
        "process_determinism": {
            "pythonhashseed": "0",
            "omp_num_threads": "1",
            "openblas_num_threads": "1",
            "mkl_num_threads": "1",
        },
    }
    mismatches = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected.items() if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} protocol mismatch: {json.dumps(mismatches, sort_keys=True)}")


def _validate_memory(
    memory: object,
    calibration_sha: str,
    calibration_protocol_sha: str,
    runtime_fingerprint: dict,
    model: str,
    reasoning: str,
) -> list[dict]:
    if not isinstance(memory, list) or len(memory) != len(BENCHMARK_DATASETS):
        raise ValueError("memory must contain exactly nine entries")
    datasets = [entry.get("dataset") for entry in memory if isinstance(entry, dict)]
    if len(datasets) != len(memory) or len(set(datasets)) != len(datasets):
        raise ValueError("memory contains a non-object or duplicate dataset entry")
    if set(datasets) != set(BENCHMARK_DATASETS):
        raise ValueError("memory dataset set does not match the nine benchmark screens")
    for entry in memory:
        dataset = entry["dataset"]
        generator = entry.get("generator", {})
        verifier = entry.get("verification", {})
        source = generator.get("performance_source", {})
        if entry.get("verified") is not True or verifier.get("status") != "passed":
            raise ValueError(f"memory entry is not strictly verified: {dataset}")
        if (
            generator.get("backend") != "pi"
            or generator.get("model") != model
            or generator.get("reasoning") != reasoning
            or verifier.get("backend") != "pi"
            or verifier.get("model") != model
            or verifier.get("reasoning") != reasoning
        ):
            raise ValueError(f"memory generator/verifier provenance mismatch: {dataset}")
        if (
            source.get("kind") != "gpt_calibration_json"
            or source.get("sha256") != calibration_sha
            or source.get("protocol_sha256") != calibration_protocol_sha
            or source.get("runtime_fingerprint") != runtime_fingerprint
        ):
            raise ValueError(f"memory is not conditioned on the supplied calibration: {dataset}")
    return memory


def main() -> None:
    p = argparse.ArgumentParser(description="Validate and merge completed staged GPT results")
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--carm", type=Path, required=True)
    p.add_argument("--memory", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5,
                   help="Evaluation replicates the run used (protocol check; default 5)")
    args = p.parse_args()

    resolved_inputs = {args.calibration.resolve(), args.carm.resolve(), args.memory.resolve()}
    if args.out.resolve() in resolved_inputs:
        raise ValueError("merged output must not overwrite an input component")

    calibration = _load_json_object(args.calibration, "calibration result")
    carm = _load_json_object(args.carm, "C-arm result")
    memory_raw = json.loads(args.memory.read_text()) if args.memory.is_file() else None
    calibration_manifest_path, calibration_manifest, calibration_protocol = _load_run_manifest(
        args.calibration, "calibration",
    )
    carm_manifest_path, carm_manifest, carm_protocol = _load_run_manifest(args.carm, "C-arm")

    _validate_protocol(calibration_protocol, CALIBRATION_ARMS, "calibration", args.repeats)
    _validate_protocol(carm_protocol, (C_ARM,), "C-arm", args.repeats)
    for key in (
        "llm_backend", "llm_model", "llm_reasoning", "codex_reasoning_effective",
        "runtime_fingerprint", "full_selected_context", "llm_vote_policy",
        "cross_experiment_split", "same_study_siblings_allowed",
    ):
        if calibration_protocol.get(key) != carm_protocol.get(key):
            raise ValueError(f"cross-stage protocol mismatch for {key}")

    current_runtime = _runtime_fingerprint()
    if calibration_protocol.get("runtime_fingerprint") != current_runtime:
        raise ValueError("runtime/input fingerprint changed after calibration; refusing mixed-code merge")

    calibration_sha = _sha(args.calibration)
    calibration_protocol_sha = hashlib.sha256(
        json.dumps(calibration_protocol, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    memory_sha = _sha(args.memory) if args.memory.is_file() else ""
    if carm_protocol.get("memory_sha256") != memory_sha:
        raise ValueError("C-arm manifest memory hash does not match the supplied memory")
    model = calibration_protocol["llm_model"]
    reasoning = calibration_protocol["llm_reasoning"]
    _validate_memory(
        memory_raw,
        calibration_sha,
        calibration_protocol_sha,
        current_runtime,
        model,
        reasoning,
    )

    if set(calibration) != set(BENCHMARK_DATASETS) or set(carm) != set(BENCHMARK_DATASETS):
        raise ValueError("result dataset keys do not exactly match the nine-screen protocol")
    for dataset in BENCHMARK_DATASETS:
        if set(calibration.get(dataset, {})) != set(CALIBRATION_ARMS):
            raise ValueError(f"calibration arm keys are incomplete or unexpected for {dataset}")
        if set(carm.get(dataset, {})) != {C_ARM}:
            raise ValueError(f"C-arm result keys are incomplete or unexpected for {dataset}")

    merged: dict[str, dict[str, list[dict]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for dataset in BENCHMARK_DATASETS:
        merged[dataset] = {}
        summary[dataset] = {}
        for arm in (*CALIBRATION_ARMS, C_ARM):
            source = carm if arm == C_ARM else calibration
            repeats = _validated_repeats(source, dataset, arm)
            merged[dataset][arm] = repeats
            summary[dataset][arm] = round(
                sum(float(result["hit_ratio_per_round"][-1]) for result in repeats) / len(repeats), 6
            )

    _atomic(args.out, merged)
    manifest_path = _manifest_path(args.out)
    component_paths = {
        "calibration": args.calibration,
        "carm": args.carm,
        "memory": args.memory,
        "calibration_manifest": calibration_manifest_path,
        "carm_manifest": carm_manifest_path,
    }
    _atomic(manifest_path, {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "backend": calibration_protocol["llm_backend"],
        "model": model,
        "reasoning": reasoning,
        "runtime_fingerprint": current_runtime,
        "cross_experiment_split": calibration_protocol["cross_experiment_split"],
        "same_study_siblings_allowed": calibration_protocol["same_study_siblings_allowed"],
        "llm_vote_policy": calibration_protocol["llm_vote_policy"],
        "replicates": 5,
        "provider_seed": None,
        "provider_seed_note": (
            "Codex OAuth does not expose a sampling seed; these are repeated API invocations and "
            "independence is not assumed"
        ),
        "components": {
            label: {"path": str(path), "sha256": _sha(path)}
            for label, path in component_paths.items()
        },
        "source_manifests": {
            "calibration_created_at": calibration_manifest.get("created_at"),
            "carm_created_at": carm_manifest.get("created_at"),
        },
        "final_hit_ratio_means": summary,
    })
    print(f"Merged result: {args.out}")
    print(f"Manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
