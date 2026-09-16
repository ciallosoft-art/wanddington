"""Offline protocol regressions: no provider calls or LightGBM training."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from waddington_select import memory_builder, merge_results, run_sequential
from waddington_select.arms.llm_reasoning_arm import _load_task, LEGACY_REPLOGLE_TASKS
from waddington_select.arms.waddington_arm import _load_memory
from waddington_select.config import ACTIVE_TASK_PROMPT_PROTOCOL, LEGACY_TASK_PROMPT_PROTOCOL


class ProtocolRegressionTests(unittest.TestCase):
    def setUp(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith("WADDINGTON_")}
        env.update({
            "WADDINGTON_LLM_BACKEND": "pi",
            "WADDINGTON_PI_MODEL": "openai-codex/gpt-5.6-luna",
            "WADDINGTON_PI_REASONING": "none",
            "WADDINGTON_DUMP_SELECTIONS": "1",
            "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        self.enterContext(patch.dict(os.environ, env, clear=True))
        self.runtime = {"sha256": "offline-fixture", "files": {}, "environment": {}}
        self.enterContext(patch.object(run_sequential, "_runtime_fingerprint", return_value=self.runtime))
        self.enterContext(patch.object(merge_results, "_runtime_fingerprint", return_value=self.runtime))
        self.enterContext(patch.object(run_sequential, "_git_state", return_value=("fixture", False)))
        self.enterContext(patch.object(memory_builder, "DEFAULT_BACKEND", "pi"))
        self.enterContext(patch.object(memory_builder, "LLM_MODEL", "openai-codex/gpt-5.6-luna"))
        self.enterContext(patch("waddington_select.llm_client.LLMClient.complete",
                                side_effect=AssertionError("provider calls forbidden")))
        self.enterContext(patch("lightgbm.LGBMClassifier.fit",
                                side_effect=AssertionError("training forbidden")))
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def manifest(self, arms=None, datasets=None):
        return run_sequential._protocol_manifest(
            datasets or ["IFNG"], arms or ["waddington_c_pair"], 5, 1, "local",
        )

    def write_manifest(self, path, protocol):
        return run_sequential._write_or_validate_manifest(path, protocol, resume=False)

    def assert_resume_rejected(self, before, after):
        path = self.tmp / "run.json"
        self.write_manifest(path, before)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            run_sequential._write_or_validate_manifest(path, after, resume=True)

    def test_replogle_active_prompts_match_label_construction(self):
        essential = _load_task("Replogle_K562_essential")
        gwps = _load_task("Replogle_K562_gwps")
        for task in (essential, gwps):
            self.assertIn("CRISPRi", task["Task"])
            self.assertIn("transcriptomic", task["Task"])
            self.assertIn("top 10%", task["Measurement"])
            self.assertNotIn("sgRNA", task["Measurement"])
        self.assertIn("L2 distance", essential["Measurement"])
        self.assertIn("anderson_darling_counts", gwps["Measurement"])
        self.assertIn("maximum", gwps["Measurement"])
        self.assertIn("transcription start sites", gwps["Measurement"])

    def test_frozen_prompts_require_explicit_legacy_switch(self):
        active = self.manifest()
        with patch.dict(os.environ, {"WADDINGTON_LEGACY_TASK_PROMPTS": "1"}):
            for dataset, task in LEGACY_REPLOGLE_TASKS.items():
                self.assertEqual(_load_task(dataset), task)
            legacy = self.manifest()
        self.assertEqual(active["task_prompt_protocol"], ACTIVE_TASK_PROMPT_PROTOCOL)
        self.assertEqual(legacy["task_prompt_protocol"], LEGACY_TASK_PROMPT_PROTOCOL)
        self.assert_resume_rejected(active, legacy)

    def test_active_memory_rejects_old_prompts_but_keeps_versioned_entries(self):
        memory = self.tmp / "memory.json"
        old = {"dataset": "Replogle_K562_gwps", "verified": True}
        current = {"dataset": "IL2", "verified": True,
                   "generator": {"task_prompt_protocol": ACTIVE_TASK_PROMPT_PROTOCOL}}
        memory.write_text(json.dumps([old, current]))
        with self.assertWarnsRegex(RuntimeWarning, "incompatible task-prompt protocol"):
            self.assertEqual(_load_memory(memory, "IFNG"), [current])
        with patch.dict(os.environ, {"WADDINGTON_LEGACY_TASK_PROMPTS": "1"}):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(_load_memory(memory, "IFNG"), [old])

    def test_resume_rejects_changed_effective_batch_size(self):
        before = self.manifest()
        with patch.dict(run_sequential.BATCH_SIZES, {"IFNG": 64}):
            after = self.manifest()
        self.assertEqual(before["batch_sizes"]["IFNG"], run_sequential.BATCH_SIZES["IFNG"])
        self.assertEqual(after["batch_sizes"]["IFNG"], 64)
        self.assert_resume_rejected(before, after)

    def test_active_skills_reject_old_prompt_protocol(self):
        from waddington_select.skills import SkillLibrary
        path = self.tmp / "skills.json"
        old = {"id": "old", "verified": True}
        current = {"id": "current", "verified": True,
                   "generator": {"task_prompt_protocol": ACTIVE_TASK_PROMPT_PROTOCOL}}
        path.write_text(json.dumps([old, current]))
        with self.assertWarnsRegex(RuntimeWarning, "incompatible task-prompt protocol"):
            self.assertEqual(SkillLibrary.load(path)._skills, [current])
        with patch.dict(os.environ, {"WADDINGTON_LEGACY_TASK_PROMPTS": "1"}):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(SkillLibrary.load(path)._skills, [old])

    def test_skill_builder_refuses_old_memory_before_calling_provider(self):
        from waddington_select import skill_builder
        memory = self.tmp / "memory.json"
        memory.write_text(json.dumps([{"dataset": "IL2", "verified": True}]))
        with patch.object(skill_builder, "MEMORY_PATH", memory):
            with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(ValueError, "no verified memory"):
                skill_builder.build_skills()

    def test_resume_rejects_each_changed_pair_selection_setting(self):
        for name, value in (
            ("NAMED_FIRST", "0"), ("PRIOR_FILL", "0"), ("COLD_START", "1"),
            ("COMPLEX_FIRST", "1"), ("EXPLORE_FRAC", "0.25"),
        ):
            with self.subTest(setting=name):
                before = self.manifest()
                with patch.dict(os.environ, {f"WADDINGTON_PAIR_{name}": value}):
                    after = self.manifest()
                self.assert_resume_rejected(before, after)

    def test_effective_pair_settings_allow_equivalent_environment_values(self):
        before = self.manifest()
        with patch.dict(os.environ, {"WADDINGTON_PAIR_EXPLORE_FRAC": "0.0"}):
            self.assertEqual(self.manifest(), before)

    def test_resume_rejects_changed_pair_dataset_membership(self):
        before = self.manifest()
        with patch.object(run_sequential, "PAIR_DATASETS", run_sequential.PAIR_DATASETS | {"IFNG"}):
            after = self.manifest()
        self.assert_resume_rejected(before, after)

    def test_resume_rejects_skill_file_replaced_at_same_path(self):
        skills = self.tmp / "skills.json"
        skills.write_text("[]")
        with patch.object(run_sequential, "SKILL_LIBRARY_PATH", skills):
            before = self.manifest(["waddington_c_skills"])
            skills.write_text('[{"id":"new-skill","verified":true}]')
            after = self.manifest(["waddington_c_skills"])
        self.assertNotEqual(before["skill_library_sha256"], after["skill_library_sha256"])
        self.assert_resume_rejected(before, after)

    def test_skill_creation_does_not_change_standard_c_protocol(self):
        skills = self.tmp / "skills.json"
        with patch.object(run_sequential, "SKILL_LIBRARY_PATH", skills):
            before = self.manifest(["waddington_c"])
            skills.write_text("[]")
            self.assertEqual(before, self.manifest(["waddington_c"]))
        self.assertIsNone(before["skill_library_sha256"])

    def test_unchanged_manifest_can_resume(self):
        path = self.tmp / "run.json"
        protocol = self.manifest()
        manifest_path = self.write_manifest(path, protocol)
        run_sequential._write_or_validate_manifest(path, protocol, resume=True)
        self.assertEqual(len(json.loads(manifest_path.read_text())["sessions"]), 2)

    @staticmethod
    def result(dataset, arm):
        batch_size = run_sequential.BATCH_SIZES[dataset]
        selected = [[f"GENE_{r}_{i}" for i in range(batch_size)] for r in range(5)]
        llm = arm in {"llm_reasoning", "waddington_c"}
        return {
            "dataset": dataset, "arm_name": arm, "repeat_index": 0,
            "n_rounds": 5, "batch_size": batch_size, "provider_seed": None,
            "hit_ratio_per_round": [0.01, 0.02, 0.03, 0.04, 0.05],
            "llm_fallback_per_round": [0] * 5,
            "selected_genes_per_round": selected,
            "llm_named_proposals_per_round": selected if llm else [[] for _ in range(5)],
            "llm_static_padding_proposals_per_round": [[] for _ in range(5)],
            "selected_llm_named_genes_per_round": selected if llm else [[] for _ in range(5)],
            "selected_without_llm_vote_per_round": [[] for _ in range(5)] if llm else selected,
        }

    def calibration(self, datasets):
        arms = list(memory_builder.REQUIRED_PERFORMANCE_ARMS)
        data = {dataset: {arm: [self.result(dataset, arm)] for arm in arms} for dataset in datasets}
        path = self.tmp / "calibration.json"
        path.write_text(json.dumps(data))
        protocol = self.manifest(arms, datasets)
        self.write_manifest(path, protocol)
        return path, protocol

    def test_calibration_rejects_changed_effective_batch_size(self):
        path, _ = self.calibration(["IFNG"])
        memory_builder._load_calibration_performance(path, ["IFNG"], 1)
        with patch.dict(run_sequential.BATCH_SIZES, {"IFNG": 64}):
            with self.assertRaisesRegex(ValueError, "batch_sizes"):
                memory_builder._load_calibration_performance(path, ["IFNG"], 1)

    def test_calibration_rejects_wrong_result_batch_size(self):
        path, _ = self.calibration(["IFNG"])
        data = json.loads(path.read_text())
        data["IFNG"]["random"][0]["batch_size"] = 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "invalid batch_size"):
            memory_builder._load_calibration_performance(path, ["IFNG"], 1)

    def test_calibration_rejects_missing_new_protocol_fields(self):
        path, protocol = self.calibration(["IFNG"])
        for field in ("batch_sizes", "pair_datasets", "pair_selection", "task_prompt_protocol"):
            with self.subTest(field=field):
                incomplete = copy.deepcopy(protocol)
                del incomplete[field]
                self.write_manifest(path, incomplete)
                with self.assertRaisesRegex(ValueError, field):
                    memory_builder._load_calibration_performance(path, ["IFNG"], 1)

    def test_formal_calibration_memory_and_merge_accept_new_protocol(self):
        datasets = list(merge_results.BENCHMARK_DATASETS)
        calibration_path, calibration_protocol = self.calibration(datasets)
        _, source = memory_builder._load_calibration_performance(calibration_path, datasets, 1)
        memory = [{
            "dataset": dataset, "verified": True,
            "generator": {
                "backend": "pi", "model": "openai-codex/gpt-5.6-luna", "reasoning": "none",
                "task_prompt_protocol": ACTIVE_TASK_PROMPT_PROTOCOL, "performance_source": source,
            },
            "verification": {"status": "passed", "backend": "pi",
                             "model": "openai-codex/gpt-5.6-luna", "reasoning": "none"},
        } for dataset in datasets]
        memory_path = self.tmp / "memory.json"
        memory_path.write_text(json.dumps(memory))
        carm_path = self.tmp / "carm.json"
        carm_path.write_text(json.dumps({
            dataset: {"waddington_c": [self.result(dataset, "waddington_c")]} for dataset in datasets
        }))
        with patch.object(run_sequential, "MEMORY_PATH", memory_path):
            carm_protocol = self.manifest(["waddington_c"], datasets)
        self.write_manifest(carm_path, carm_protocol)
        output_path = self.tmp / "merged.json"
        with patch("sys.argv", ["merge", "--calibration", str(calibration_path),
                                "--carm", str(carm_path), "--memory", str(memory_path),
                                "--out", str(output_path), "--repeats", "1"]):
            merge_results.main()
        merged = json.loads(output_path.read_text())
        self.assertEqual(set(merged), set(datasets))
        self.assertEqual(len(merged[datasets[0]]), 6)
        self.assertEqual(source["protocol_sha256"], hashlib.sha256(
            json.dumps(calibration_protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest())

    def test_merge_rejects_wrong_result_batch_size(self):
        result = self.result("IFNG", "random")
        result["batch_size"] = 1
        with self.assertRaisesRegex(ValueError, "batch_size differs"):
            merge_results._validated_repeats({"IFNG": {"random": [result]}}, "IFNG", "random", 1)


if __name__ == "__main__":
    unittest.main()
