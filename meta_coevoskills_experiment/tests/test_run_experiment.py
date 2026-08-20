from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_experiment.py"
CONFIG = ROOT / "configs" / "experiment.yaml"


def load_module():
    spec = importlib.util.spec_from_file_location("meta_coevoskills_runner", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    def test_preflight_accepts_frozen_l123_manifests_and_identical_empty_v0(self):
        runner = load_module()
        config = runner.load_config(CONFIG)
        report = runner.preflight(config)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["levels"], [1, 2, 3])
        self.assertEqual(report["counts"], {"A": 3, "B": 3, "holdout": 9})
        self.assertTrue(report["all_ids_disjoint"])
        self.assertTrue(report["all_episodes_disjoint"])
        self.assertTrue(report["skill_v0_byte_identical"])
        self.assertEqual(report["skill_v0_sha256"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        self.assertEqual(report["frozen_m1_hashes_verified"], 9)

    def test_preflight_rejects_level_four_before_api_client_creation(self):
        import tempfile
        runner = load_module()
        config = runner.load_config(CONFIG)
        bad_manifest = json.loads((ROOT / "manifests" / "dev_fold_a.json").read_text())
        bad_manifest["items"][0]["level"] = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(bad_manifest))
            config["manifests"]["A"]["path"] = str(path)
            config["manifests"]["A"]["sha256"] = runner.sha256_file(path)
            with self.assertRaisesRegex(ValueError, "Levels 1, 2, and 3"):
                runner.preflight(config)

    def test_runtime_guard_rejects_sol_and_aliases(self):
        runner = load_module()
        for model in ("gpt-5.6-sol", "openai/gpt-5.6-sol", "gpt_5_6_sol", "GPT-5.6-SOL-PRO", "gpt-5.6-luna-pro"):
            with self.assertRaisesRegex(ValueError, "Sol models are forbidden"):
                runner.assert_no_sol(model)
        runner.assert_no_sol("gpt-4o-mini")
        runner.assert_no_sol("gpt-5.6-luna")

    def test_full_both_plan_has_controlled_schedule_and_102_max_calls(self):
        runner = load_module()
        config = runner.load_config(CONFIG)
        plan = runner.build_run_plan(config, arm="both", rounds=3, smoke=False)
        self.assertEqual(plan["fold_schedule"], [
            {"round": 1, "optimization": "A", "validation": "B"},
            {"round": 2, "optimization": "B", "validation": "A"},
            {"round": 3, "optimization": "A", "validation": "B"},
        ])
        self.assertEqual(plan["calls"], {"target": 87, "verifier": 9, "rewriter": 6, "maximum_total": 102})
        self.assertEqual(plan["models"], {
            "target": "gpt-4o-mini", "verifier": "gpt-5.6-luna", "rewriter": "gpt-5.6-luna"
        })
        self.assertEqual(plan["inference_settings"]["target"]["seed"], 0)
        self.assertEqual(plan["inference_settings"]["target"]["temperature"], 0)
        self.assertEqual(plan["inference_settings"]["verifier"]["reasoning_mode"], "standard")
        self.assertEqual(plan["inference_settings"]["verifier"]["reasoning_effort"], "high")
        self.assertEqual(plan["inference_settings"]["rewriter"]["reasoning_mode"], "pro")
        self.assertEqual(plan["inference_settings"]["rewriter"]["reasoning_effort"], "xhigh")
        self.assertEqual(plan["frozen_artifacts"]["skill_v0_sha256"], runner.EMPTY_SHA256)
        self.assertEqual(len(plan["frozen_artifacts"]["config_sha256"]), 64)
        self.assertEqual(len(plan["frozen_artifacts"]["runner_sha256"]), 64)
        self.assertEqual(plan["http_policy"], {"max_retries": 0, "timeout_seconds": 180})

    def test_smoke_surrogate_plan_uses_three_levels_and_ten_calls(self):
        runner = load_module()
        config = runner.load_config(CONFIG)
        plan = runner.build_run_plan(config, arm="surrogate", rounds=1, smoke=True)
        self.assertEqual(plan["sample_levels"], [1, 2, 3])
        self.assertEqual(plan["calls"], {"target": 6, "verifier": 3, "rewriter": 1, "maximum_total": 10})
        self.assertEqual(plan["holdout_calls"], 0)

    def test_responses_api_reasoning_config_is_explicit_and_role_specific(self):
        runner = load_module()
        config = runner.load_config(CONFIG)

        class Responses:
            def __init__(self, response):
                self.response = response
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return self.response

        verifier_responses = Responses(SimpleNamespace(
            model="gpt-5.6-luna", id="verifier-response", output_text="{}", usage=None,
        ))
        verifier_client = SimpleNamespace(responses=verifier_responses)
        runner.verifier_call(verifier_client, config, {"sample_id": "x"})
        self.assertEqual(verifier_responses.kwargs["model"], "gpt-5.6-luna")
        self.assertEqual(verifier_responses.kwargs["reasoning"], {"mode": "standard", "effort": "high"})

        rewriter_responses = Responses(SimpleNamespace(
            model="gpt-5.6-luna", id="rewriter-response", output_text="{}", usage=None,
        ))
        rewriter_client = SimpleNamespace(responses=rewriter_responses)
        runner.rewriter_call(rewriter_client, config, {"current_skill": ""})
        self.assertEqual(rewriter_responses.kwargs["model"], "gpt-5.6-luna")
        self.assertEqual(rewriter_responses.kwargs["reasoning"], {"mode": "pro", "effort": "xhigh"})

    def test_cache_identity_includes_reasoning_mode_and_effort(self):
        runner = load_module()
        base = {"role": "verifier", "model": "gpt-5.6-luna", "input_sha256": "a" * 64}
        standard_high = runner.request_identity({**base, "reasoning_mode": "standard", "reasoning_effort": "high"})
        pro_high = runner.request_identity({**base, "reasoning_mode": "pro", "reasoning_effort": "high"})
        pro_xhigh = runner.request_identity({**base, "reasoning_mode": "pro", "reasoning_effort": "xhigh"})
        self.assertNotEqual(standard_high, pro_high)
        self.assertNotEqual(pro_high, pro_xhigh)

    def test_config_uses_base_luna_id_for_both_arms_and_explicit_reasoning(self):
        runner = load_module()
        config = runner.load_config(CONFIG)
        runner.assert_model_roles(config)
        self.assertEqual(config["models"]["target"]["model"], "gpt-4o-mini")
        self.assertEqual(config["models"]["verifier"], {
            "model": "gpt-5.6-luna", "reasoning_mode": "standard", "reasoning_effort": "high",
            "max_output_tokens": 4096,
        })
        self.assertEqual(config["models"]["rewriter"], {
            "model": "gpt-5.6-luna", "reasoning_mode": "pro", "reasoning_effort": "xhigh",
            "max_output_tokens": 8192,
        })

    def test_control_feedback_is_coarse_and_contains_no_answers_or_raw_outputs(self):
        runner = load_module()
        evaluation = {
            "canonical_score": 0.2, "fixed_cardinality_score": 0.1, "parse_failures": 1,
            "items": [
                {"id": "sample-1", "level": 3, "answer_format": "four_letter_tf", "score": 0.0,
                 "raw_output": "TTTT", "parsed": "TTTT", "reference_answer": "TFTF"},
                {"id": "sample-2", "level": 2, "answer_format": "scalar_range", "score": 1.0,
                 "raw_output": "7", "parsed": 7.0, "reference_answer": 7},
            ],
        }
        feedback = runner.build_coarse_feedback(evaluation)
        encoded = json.dumps(feedback)
        self.assertEqual(feedback["failed_samples"], [
            {"sample_id": "sample-1", "level": 3, "answer_format": "four_letter_tf"}
        ])
        for forbidden in ("raw_output", "parsed", "reference", "TTTT", "TFTF"):
            self.assertNotIn(forbidden, encoded)

    def test_verifier_payload_contains_only_public_input_contract_and_target_answer(self):
        runner = load_module()
        payload = runner.build_verifier_payload("id-1", 2, "public telemetry and question", "scalar_range", "42")
        self.assertEqual(set(payload), {
            "sample_id", "level", "public_rendered_task_input", "output_format_requirement", "target_model_answer"
        })
        encoded = json.dumps(payload).lower()
        for forbidden in ("skill", "reference", "holdout"):
            self.assertNotIn(forbidden, encoded)

    def test_aggregation_counts_types_and_separates_uncertainty(self):
        runner = load_module()
        diagnoses = [
            {"sample_id": "a", "level": 1, "surrogate_pass": False, "failed_assertions": [],
             "failure_types": ["FORMAT_ERROR"], "root_cause": "extra prose", "skill_gap": "strict finalization",
             "actionable_revision": "reserve final answer", "confidence": 0.9},
            {"sample_id": "b", "level": 3, "surrogate_pass": False, "failed_assertions": [],
             "failure_types": ["FORMAT_ERROR", "UNKNOWN"], "root_cause": "unclear",
             "skill_gap": "strict finalization", "actionable_revision": "reserve final answer", "confidence": 0.3},
        ]
        aggregate = runner.aggregate_diagnoses(diagnoses, round_number=1, optimization_fold="A")
        self.assertEqual(aggregate["failure_type_counts"], {"FORMAT_ERROR": 2, "UNKNOWN": 1})
        self.assertTrue(aggregate["cross_level_patterns"])
        self.assertEqual(len(aggregate["uncertain_diagnoses"]), 1)
        self.assertEqual(set(aggregate["level_specific_patterns"]), {"1", "2", "3"})
        encoded = json.dumps(aggregate)
        for item_level_text in ("extra prose", "strict finalization", "reserve final answer", "unclear"):
            self.assertNotIn(item_level_text, encoded)

    def test_atomic_checkpoint_resume_reuses_completed_call(self):
        import tempfile
        runner = load_module()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "call.json"
            calls = []
            def paid_call():
                calls.append("called")
                return {"raw_output": "A", "usage": {"input_tokens": 1, "output_tokens": 1}}
            first = runner.checkpointed_call(checkpoint, paid_call, resume=False, request_sha256="a" * 64)
            second = runner.checkpointed_call(checkpoint, paid_call, resume=True, request_sha256="a" * 64)
            self.assertEqual(first, second)
            self.assertEqual(calls, ["called"])
            self.assertEqual(json.loads(checkpoint.read_text()), first)
            self.assertEqual(first["_request_sha256"], "a" * 64)

    def test_resume_rejects_checkpoint_from_different_request(self):
        import tempfile
        runner = load_module()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "call.json"
            runner.checkpointed_call(checkpoint, lambda: {"status": "complete"}, resume=False, request_sha256="a" * 64)
            with self.assertRaisesRegex(ValueError, "request identity"):
                runner.checkpointed_call(checkpoint, lambda: {"status": "complete"}, resume=True, request_sha256="b" * 64)

    def test_invalid_checkpoint_is_not_reused(self):
        import tempfile
        runner = load_module()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "call.json"
            checkpoint.write_text(json.dumps({"status": "invalid", "_request_sha256": "a" * 64, "raw_output": "bad"}))
            calls = []
            result = runner.checkpointed_call(
                checkpoint,
                lambda: calls.append(1) or {"status": "complete", "raw_output": "good"},
                resume=True,
                request_sha256="a" * 64,
            )
            self.assertEqual(calls, [1])
            self.assertEqual(result["raw_output"], "good")

    def test_paid_execution_requires_matching_plan_hash(self):
        runner = load_module()
        config = runner.load_config(CONFIG)
        plan = runner.build_run_plan(config, arm="surrogate", rounds=1, smoke=True)
        digest = runner.plan_sha256(plan)
        runner.authorize_paid_execution(plan, execute_paid=True, supplied_plan_sha256=digest)
        with self.assertRaisesRegex(ValueError, "plan SHA256"):
            runner.authorize_paid_execution(plan, execute_paid=True, supplied_plan_sha256="0" * 64)
        with self.assertRaisesRegex(SystemExit, "RUN PLAN only"):
            runner.authorize_paid_execution(plan, execute_paid=False, supplied_plan_sha256=None)

    def test_returned_model_identity_must_match_exact_requested_model(self):
        runner = load_module()
        runner.validate_returned_model("gpt-4o-mini", "gpt-4o-mini")
        with self.assertRaisesRegex(ValueError, "Sol models are forbidden"):
            runner.validate_returned_model("gpt-4o-mini", "gpt-5.6-sol")
        with self.assertRaisesRegex(ValueError, "returned model mismatch"):
            runner.validate_returned_model("gpt-5.6-luna", "gpt-5.6-luna-aliased")

    def test_generation_metrics_records_target_verifier_rewriter_usage_and_wall_time(self):
        import tempfile
        runner = load_module()
        evaluation = {
            "canonical_score": 1.0,
            "fixed_cardinality_score": 1.0,
            "parse_failures": 0,
            "format_validity": 1.0,
            "by_level": {"1": 1.0},
            "by_answer_format": {"scalar_range": 1.0},
            "target_usage": {"input_tokens": 10, "output_tokens": 2},
            "target_estimated_cost_usd": 0.001,
            "target_wall_time_seconds": 1.5,
            "items": [{"id": "x", "score": 1.0, "parse_error": None}],
        }
        rewrite = {"_trace": {
            "usage": {"input_tokens": 20, "output_tokens": 4},
            "estimated_cost_usd": 0.002,
            "wall_time_seconds": 2.5,
        }}
        verifier = {
            "usage": {"input_tokens": 30, "output_tokens": 6},
            "estimated_cost_usd": 0.003,
            "wall_time_seconds": 3.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            skill = Path(directory) / "skill.txt"
            skill.write_text("shared instructions")
            metrics = runner.generation_metrics(
                "surrogate", 1, skill, evaluation, evaluation, evaluation, evaluation, rewrite,
                verifier_usage=verifier,
            )
        self.assertEqual(metrics["target_usage"]["optimization"]["input_tokens"], 10)
        self.assertEqual(metrics["verifier_usage"]["usage"]["input_tokens"], 30)
        self.assertEqual(metrics["rewriter_usage"]["input_tokens"], 20)
        self.assertEqual(metrics["wall_time_seconds"], 1.5 * 3 + 2.5 + 3.5)

    def test_direct_item_comparison_is_paired_by_identical_ids(self):
        runner = load_module()
        control = {"items": [
            {"id": "a", "level": 1, "answer_format": "scalar_range", "score": 1.0, "parse_error": None},
            {"id": "b", "level": 3, "answer_format": "four_letter_tf", "score": None, "parse_error": "empty"},
        ]}
        surrogate = {"items": [
            {"id": "a", "level": 1, "answer_format": "scalar_range", "score": 0.0, "parse_error": None},
            {"id": "b", "level": 3, "answer_format": "four_letter_tf", "score": 0.5, "parse_error": None},
        ]}
        rows = runner.paired_item_comparison(control, surrogate)
        self.assertEqual([row["id"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["surrogate_minus_control"], -1.0)
        self.assertEqual(rows[1]["surrogate_minus_control"], 0.5)
        self.assertTrue(rows[1]["control_parse_failure"])
        with self.assertRaisesRegex(ValueError, "identical ordered IDs"):
            runner.paired_item_comparison(control, {"items": list(reversed(surrogate["items"]))})

    def test_resume_rejects_stale_evaluation_for_different_skill(self):
        import tempfile
        runner = load_module()
        with tempfile.TemporaryDirectory() as directory:
            skill = Path(directory) / "skill.txt"
            skill.write_text("current")
            stale = {
                "skill_sha256": "0" * 64,
                "model": "gpt-4o-mini",
                "ordered_ids": ["a"],
                "evaluation_request_sha256": "a" * 64,
            }
            with self.assertRaisesRegex(ValueError, "stale evaluation checkpoint"):
                runner.validate_cached_evaluation(stale, skill, ["a"], "gpt-4o-mini", "a" * 64)

    def test_resume_rejects_stale_evaluation_with_same_skill_model_and_ids(self):
        import tempfile
        runner = load_module()
        with tempfile.TemporaryDirectory() as directory:
            skill = Path(directory) / "skill.txt"
            skill.write_text("current")
            stale = {
                "skill_sha256": runner.sha256_file(skill),
                "model": "gpt-4o-mini",
                "ordered_ids": ["a"],
                "evaluation_request_sha256": "a" * 64,
            }
            with self.assertRaisesRegex(ValueError, "stale evaluation checkpoint"):
                runner.validate_cached_evaluation(stale, skill, ["a"], "gpt-4o-mini", "b" * 64)


if __name__ == "__main__":
    unittest.main()
