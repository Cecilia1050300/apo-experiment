from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

try:
    import factorybench_gpt55_context_control as base
except ImportError as exc:
    raise SystemExit(
        "factorybench_gpt55_context_control.py must be in the same directory."
    ) from exc

try:
    import factorybench_evaluator_smoke_test as evaluator
except ImportError as exc:
    raise SystemExit(
        "factorybench_evaluator_smoke_test.py must be in the same directory."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "data" / "factorybench_l4_subset.json"
DEFAULT_SKILL = SCRIPT_DIR / "skills" / "skill_v2.md"
RESULTS_DIR = SCRIPT_DIR / "results" / "json_contract_v2"

DEFAULT_CONTRACT_AUDIT = RESULTS_DIR / "contract_audit.json"
DEFAULT_GPT55_GENERATION = RESULTS_DIR / "gpt55_generation.json"
DEFAULT_MINI_GENERATION = RESULTS_DIR / "mini_generation.json"
DEFAULT_GPT55_JUDGED = RESULTS_DIR / "gpt55_judged.json"
DEFAULT_MINI_JUDGED = RESULTS_DIR / "mini_judged.json"
DEFAULT_COMPARISON = RESULTS_DIR / "mini_vs_gpt55_comparison.json"

CONTRACT_VERSION = "factorybench_diagnostic_json_v2"
TARGET_KEYS = {"root_cause", "evidence", "corrective_actions"}
QUESTION_HEADING = re.compile(r"^[ \t]*Question[ \t]*:[ \t]*", re.I | re.M)
LETTER_ONLY = re.compile(r"^[A-Z][.)]?$", re.I)
LETTER_TASK_PHRASES = (
    "answer only with the letter",
    "single best-matching option",
    "single best matching option",
)

STANDARD_QUESTION = """Question:
Determine whether the supplied telemetry shows anomalous behavior. Return the
most defensible underlying physical or operational root cause, evidence grounded
only in the supplied input, and corrective actions with a verification step.
If the behavior is normal, use \"normal\" as root_cause and return an empty
corrective_actions array. Follow the diagnostic JSON output contract in the
system instructions; do not answer with an option letter.""".strip()

TARGET_SCHEMA = """Return JSON only with exactly these keys:
{
  "root_cause": "most defensible underlying cause, or normal",
  "evidence": ["specific signal or task-phase evidence from the input"],
  "corrective_actions": ["action and verification step"]
}
For normal behavior, corrective_actions must be an empty array. Do not return
Markdown, headings, an option letter, or a dataset label by itself.""".strip()

COMMON_INSTRUCTIONS = f"""You are solving a FactoryBench Level 4 industrial
troubleshooting case under contract {CONTRACT_VERSION}.

Use only the machine description, signal mapping, time series, and standardized
diagnostic question in the user input. Distinguish a symptom from its underlying
physical or operational cause. Do not invent signals, normal ranges, machine
specifications, fault documents, candidate options, or SOPs. When the telemetry
does not uniquely distinguish among physical causes, state that limitation in
the evidence and give only the narrowest diagnosis supported by the input.

{TARGET_SCHEMA}""".strip()

JUDGE_INSTRUCTIONS = """You are evaluating FactoryBench diagnostic JSON answers
under contract factorybench_diagnostic_json_v2.

You receive:
1. question: the complete standardized case input.
2. canonical_root_cause: the diagnosis gold from the dataset.
3. corrective_action_reference: semantic procedure text or null.
4. corrective_actions_evaluable: whether item 3 contains semantic procedure gold.
5. model_raw_answer: the complete unmodified target answer.

Judge semantic equivalence; do not require exact snake_case wording. Do not use
or reconstruct multiple-choice options. Do not treat a signal anomaly alone as
an underlying root cause. Evaluate evidence_grounded only from the question.

Scoring:
- diagnosis_score is 1.0 only when the underlying root cause is correct or
  clearly equivalent; otherwise 0.0.
- If corrective_actions_evaluable is false, corrective_actions_correct and
  full_protocol_score must both be null. Never invent a remediation reference.
- If corrective_actions_evaluable is true: full_protocol_score is 0.0 when the
  root cause is wrong, 0.5 when the root cause is correct but the actions are
  missing/materially wrong, and 1.0 when both are broadly correct.
- For a normal case whose reference says no remediation is required, an empty
  corrective_actions array is correct.

Return JSON only with exactly these keys:
{
  "diagnosis_score": 0.0,
  "root_cause_correct": false,
  "evidence_grounded": false,
  "corrective_actions_evaluable": false,
  "corrective_actions_correct": null,
  "full_protocol_score": null,
  "reason": "brief evidence-based explanation"
}""".strip()


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list.")
    return value


def is_letter_only(value: Any) -> bool:
    return isinstance(value, str) and bool(LETTER_ONLY.fullmatch(value.strip()))


def rewrite_target_input(original: str) -> str:
    """Remove the original question and append the single v2 diagnostic task."""
    matches = list(QUESTION_HEADING.finditer(original))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Question heading; found {len(matches)}")
    context = original[: matches[0].start()].rstrip()
    if not context:
        raise ValueError("Case context before Question is empty.")
    return f"{context}\n\n{STANDARD_QUESTION}"


def normalize_reference(case: dict[str, Any]) -> dict[str, Any]:
    """Build the v2 gold without fabricating content absent from the subset."""
    canonical = str(case.get("root_cause", "")).strip()
    if not canonical:
        raise ValueError(f"Case {case.get('id')} has no canonical root_cause.")
    original_reference = case.get("reference_answer")
    reference_text = (
        original_reference.strip() if isinstance(original_reference, str) else ""
    )
    semantic_reference = None if is_letter_only(reference_text) else reference_text or None
    return {
        "contract_version": CONTRACT_VERSION,
        "canonical_root_cause": canonical,
        "corrective_action_reference": semantic_reference,
        "corrective_actions_evaluable": semantic_reference is not None,
        "original_reference_answer": original_reference,
        "original_reference_type": (
            "option_letter" if is_letter_only(reference_text) else "semantic_text"
        ),
        "boundary": (
            "An option letter is provenance only. No option text or remediation "
            "is reconstructed when it is absent from the prepared subset."
        ),
    }


def build_instructions(role: str, skill_text: str) -> str:
    if role == "no_skill":
        supplement = """No reusable diagnostic Skill, RAG passage, fault catalog,
signal-analysis tool, candidate options, gold root cause, or reference procedure
is supplied in this condition.""".strip()
    elif role == "skill_v2":
        supplement = f"""Use the following reusable diagnostic reasoning procedure.
It is not a machine-specific fault catalog. The user input remains the only
source of case-specific evidence.

--- BEGIN DIAGNOSTIC SKILL V2 ---
{skill_text}
--- END DIAGNOSTIC SKILL V2 ---

No RAG passage, fault catalog, signal-analysis tool, candidate options, gold
root cause, or reference procedure is supplied in this condition.""".strip()
    else:
        raise ValueError(f"Unknown role: {role}")
    return f"{COMMON_INSTRUCTIONS}\n\n{supplement}"


def audit_contract(
    dataset: Path,
    split: str,
    expected_cases: int,
) -> dict[str, Any]:
    cases = base.load_cases(dataset, split, expected_cases)
    records: list[dict[str, Any]] = []
    for case in cases:
        errors: list[str] = []
        warnings: list[str] = []
        original = str(case["input"])
        try:
            target = rewrite_target_input(original)
        except ValueError as exc:
            target = ""
            errors.append(str(exc))
        try:
            reference = normalize_reference(case)
        except ValueError as exc:
            reference = {}
            errors.append(str(exc))

        context_audit: dict[str, Any] = {}
        if target:
            audit_case = dict(case)
            audit_case["input"] = target
            # This sentinel is used only by the deterministic context checker;
            # it is not a diagnosis or a reference supplied to the target.
            audit_case["reference_answer"] = "__REFERENCE_NOT_SUPPLIED_TO_TARGET__"
            context_audit = base.audit_case_context(audit_case)
            errors.extend(context_audit.get("errors", []))
            warnings.extend(context_audit.get("warnings", []))

        lowered = target.lower()
        for phrase in LETTER_TASK_PHRASES:
            if phrase in lowered:
                errors.append(f"letter_task_phrase_remains: {phrase}")
        if target and target.count("Question:") != 1:
            errors.append("standardized_target_must_have_one_question_heading")
        if target and STANDARD_QUESTION not in target:
            errors.append("standardized_question_missing")
        if reference.get("original_reference_type") == "option_letter":
            warnings.append("protocol_gold_unavailable_option_letter_retained_as_provenance")

        canonical = str(reference.get("canonical_root_cause", ""))
        if canonical and canonical.lower() in lowered:
            warnings.append("canonical_root_label_present_in_nongold_context_review")
        semantic = reference.get("corrective_action_reference")
        if isinstance(semantic, str) and semantic and semantic in target:
            errors.append("semantic_reference_leaked_into_target")

        records.append(
            {
                "id": str(case["id"]),
                "valid": not errors,
                "original_input_sha256": base.sha256_text(original),
                "target_input_sha256": base.sha256_text(target),
                "reference_contract_sha256": base.sha256_text(
                    json.dumps(reference, ensure_ascii=False, sort_keys=True)
                ),
                "original_question_replaced": bool(target),
                "original_reference_type": reference.get("original_reference_type"),
                "corrective_actions_evaluable": reference.get(
                    "corrective_actions_evaluable"
                ),
                "signal_mapping_count": context_audit.get("signal_mapping_count"),
                "time_series_rows": context_audit.get("time_series_rows"),
                "unmapped_time_series_signals": context_audit.get(
                    "unmapped_time_series_signals", []
                ),
                "errors": errors,
                "warnings": warnings,
            }
        )
    return {
        "status": "passed" if all(row["valid"] for row in records) else "failed",
        "contract_version": CONTRACT_VERSION,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": base.sha256_file(dataset),
        "split": split,
        "case_count": len(records),
        "all_cases_valid": all(row["valid"] for row in records),
        "diagnosis_gold_count": len(records),
        "protocol_gold_count": sum(
            bool(row["corrective_actions_evaluable"]) for row in records
        ),
        "records": records,
        "boundary": (
            "Diagnosis metrics cover every case. Full-protocol metrics cover only "
            "cases with semantic corrective-action reference text."
        ),
    }


def print_contract_audit(audit: dict[str, Any]) -> None:
    for row in audit["records"]:
        state = "PASS" if row["valid"] else "FAIL"
        print(
            f"{row['id']}: {state}; reference={row['original_reference_type']}; "
            f"protocol_evaluable={row['corrective_actions_evaluable']}"
        )
        for error in row["errors"]:
            print(f"  ERROR: {error}")
        for warning in row["warnings"]:
            print(f"  WARNING: {warning}")
    print(
        f"Contract audit: valid={audit['all_cases_valid']}; "
        f"diagnosis_gold={audit['diagnosis_gold_count']}; "
        f"protocol_gold={audit['protocol_gold_count']}"
    )


def command_check_contract(args: argparse.Namespace) -> None:
    audit = audit_contract(args.dataset, args.split, args.expected_cases)
    base.atomic_save(args.output, audit)
    print_contract_audit(audit)
    print(f"Saved: {args.output}")
    if not audit["all_cases_valid"]:
        raise SystemExit(1)


def generation_conditions(prefix: str) -> tuple[str, str]:
    return (f"{prefix}_no_skill", f"{prefix}_skill_v2")


def condition_role(condition: str) -> str:
    if condition.endswith("_no_skill"):
        return "no_skill"
    if condition.endswith("_skill_v2"):
        return "skill_v2"
    raise ValueError(f"Unknown condition: {condition}")


def generation_summary(
    records: list[dict[str, Any]], conditions: tuple[str, str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in conditions:
        rows = [row for row in records if row.get("condition") == condition]
        result[condition] = {
            "n": len(rows),
            "completed": sum(row.get("status") == "ok" for row in rows),
            "format_valid": sum(
                bool(row.get("deterministic_validation", {}).get("valid"))
                for row in rows
            ),
        }
    return result


def same_inputs(records: list[dict[str, Any]], conditions: tuple[str, str]) -> bool:
    grouped: dict[str, dict[str, str]] = defaultdict(dict)
    for row in records:
        grouped[str(row.get("id"))][str(row.get("condition"))] = str(
            row.get("target_input_sha256")
        )
    return bool(grouped) and all(
        set(by_condition) == set(conditions)
        and len(set(by_condition.values())) == 1
        for by_condition in grouped.values()
    )


def command_run(args: argparse.Namespace) -> None:
    conditions = generation_conditions(args.prefix)
    cases = base.load_cases(args.dataset, args.split, args.expected_cases)
    if not args.skill.exists():
        raise FileNotFoundError(f"Skill not found: {args.skill}")
    skill_text = args.skill.read_text(encoding="utf-8")
    if not skill_text.strip():
        raise ValueError("Skill v2 file is empty.")

    audit = audit_contract(args.dataset, args.split, args.expected_cases)
    base.atomic_save(args.contract_audit, audit)
    print_contract_audit(audit)
    if not audit["all_cases_valid"]:
        raise SystemExit("Contract audit failed; no target API calls were made.")

    configuration = {
        "contract_version": CONTRACT_VERSION,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": base.sha256_file(args.dataset),
        "split": args.split,
        "expected_cases": args.expected_cases,
        "target_model": args.model,
        "prefix": args.prefix,
        "conditions": list(conditions),
        "skill_path": str(args.skill.resolve()),
        "skill_sha256": base.sha256_text(skill_text),
        "seed": args.seed,
        "retries": args.retries,
        "target_schema_keys": sorted(TARGET_KEYS),
        "question_transform": "replace_original_question_with_standard_diagnostic_task",
        "diagnosis_gold_source": "case.root_cause",
        "option_letter_reference_policy": "provenance_only_not_scored_as_protocol",
        "held_out_gold_supplied_to_target": False,
        "domain_rag": None,
        "signal_tool": None,
    }
    if args.output.exists() and not args.overwrite:
        output = base.load_json(args.output)
        if output.get("configuration") != configuration:
            raise SystemExit(
                "Existing v2 output has a different configuration. Use another "
                "--output or pass --overwrite."
            )
        print(f"Resuming: {args.output}")
    else:
        output = {
            "status": "running",
            "contract_version": CONTRACT_VERSION,
            "experiment_type": f"factorybench_{args.prefix}_json_contract_v2",
            "configuration": configuration,
            "records": [],
            "summary": {},
            "same_target_input_across_conditions": None,
        }
        base.atomic_save(args.output, output)

    existing = {
        (str(row.get("condition")), str(row.get("id"))): row
        for row in output.get("records", [])
    }
    jobs = [
        (condition, index, case)
        for condition in conditions
        for index, case in enumerate(cases, start=1)
    ]
    random.Random(args.seed).shuffle(jobs)
    client = base.require_api_client()

    for job_index, (condition, case_index, case) in enumerate(jobs, start=1):
        case_id = str(case.get("id", f"case_{case_index}"))
        key = (condition, case_id)
        record = existing.get(key)
        if record is not None and record.get("status") == "ok":
            print(f"[{job_index}/{len(jobs)}] {condition} / {case_id}: complete")
            continue
        if record is not None and record.get("status") == "raw_response_saved":
            print(f"[{job_index}/{len(jobs)}] {condition} / {case_id}: parse saved raw")
            try:
                base.ensure_saved_raw_is_parsed(record, evaluator)
            except Exception as exc:  # noqa: BLE001
                record["status"] = "parser_error"
                record["parser_error"] = repr(exc)
            base.atomic_save(args.output, output)
            continue

        role = condition_role(condition)
        instructions = build_instructions(role, skill_text)
        target_input = rewrite_target_input(str(case["input"]))
        reference = normalize_reference(case)
        initial = {
            "condition": condition,
            "condition_role": role,
            "id": case_id,
            "status": "calling_model",
            "contract_version": CONTRACT_VERSION,
            "target_model": args.model,
            "skill_version": "skill_v2" if role == "skill_v2" else None,
            "target_input": target_input,
            "target_input_sha256": base.sha256_text(target_input),
            "instructions_sha256": base.sha256_text(instructions),
            "reference_contract": reference,
            "reference_contract_sha256": base.sha256_text(
                json.dumps(reference, ensure_ascii=False, sort_keys=True)
            ),
            "metadata": case.get("metadata"),
            "raw_response": None,
            "raw_response_sha256_at_save": None,
            "deterministic_validation": None,
            "parsed_response": None,
            "latency_seconds": None,
            "token_usage": None,
        }
        if record is None:
            record = initial
            output["records"].append(record)
            existing[key] = record
        else:
            record.clear()
            record.update(initial)
        base.atomic_save(args.output, output)

        print(f"[{job_index}/{len(jobs)}] {condition} / {case_id} / {args.model}")
        started = time.time()
        try:
            response = base.call_target_model(
                client, args.model, instructions, target_input, args.retries
            )
            raw = base.response_output_text(response)
            record.update(
                {
                    "status": "raw_response_saved",
                    "response_id": getattr(response, "id", None),
                    "raw_response": raw,
                    "raw_response_sha256_at_save": base.sha256_text(raw),
                    "latency_seconds": time.time() - started,
                    "token_usage": base.response_usage(response),
                }
            )
            base.atomic_save(args.output, output)
            base.ensure_saved_raw_is_parsed(record, evaluator)
        except Exception as exc:  # noqa: BLE001
            if record.get("status") != "raw_response_saved":
                record["status"] = "target_model_error"
                record["target_model_error"] = repr(exc)
                record["latency_seconds"] = time.time() - started
            else:
                record["status"] = "parser_error"
                record["parser_error"] = repr(exc)
            print(f"  ERROR: {exc}")
        output["summary"] = generation_summary(output["records"], conditions)
        base.atomic_save(args.output, output)

    output["summary"] = generation_summary(output["records"], conditions)
    output["same_target_input_across_conditions"] = same_inputs(
        output["records"], conditions
    )
    expected_record_count = len(cases) * len(conditions)
    completed_record_count = sum(
        row.get("status") == "ok" for row in output["records"]
    )
    output["status"] = (
        "completed"
        if completed_record_count == expected_record_count
        and output["same_target_input_across_conditions"] is True
        else "partial"
    )
    base.atomic_save(args.output, output)
    print(f"Saved: {args.output}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print("Same target input:", output["same_target_input_across_conditions"])
    if output["status"] != "completed":
        raise SystemExit(
            "Generation is partial. Re-run the same command to resume failed calls."
        )


def parse_judge_json(raw: str) -> dict[str, Any]:
    return evaluator.parse_json_object(raw)


def validate_judge_output(
    value: dict[str, Any], expected_protocol_evaluable: bool
) -> dict[str, Any]:
    keys = {
        "diagnosis_score",
        "root_cause_correct",
        "evidence_grounded",
        "corrective_actions_evaluable",
        "corrective_actions_correct",
        "full_protocol_score",
        "reason",
    }
    if set(value) != keys:
        raise ValueError(f"Judge keys mismatch: {sorted(value)}")
    diagnosis = float(value["diagnosis_score"])
    if diagnosis not in {0.0, 1.0}:
        raise ValueError("diagnosis_score must be 0.0 or 1.0")
    if not isinstance(value["root_cause_correct"], bool):
        raise ValueError("root_cause_correct must be boolean")
    if diagnosis != (1.0 if value["root_cause_correct"] else 0.0):
        raise ValueError("diagnosis_score conflicts with root_cause_correct")
    if not isinstance(value["evidence_grounded"], bool):
        raise ValueError("evidence_grounded must be boolean")
    if value["corrective_actions_evaluable"] is not expected_protocol_evaluable:
        raise ValueError("corrective_actions_evaluable conflicts with reference contract")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be non-empty")

    action_correct = value["corrective_actions_correct"]
    protocol = value["full_protocol_score"]
    if not expected_protocol_evaluable:
        if action_correct is not None or protocol is not None:
            raise ValueError("protocol fields must be null without semantic gold")
        normalized_protocol = None
    else:
        if not isinstance(action_correct, bool):
            raise ValueError("corrective_actions_correct must be boolean")
        normalized_protocol = float(protocol)
        if normalized_protocol not in {0.0, 0.5, 1.0}:
            raise ValueError("full_protocol_score must be 0.0, 0.5, or 1.0")
        expected_score = (
            0.0
            if not value["root_cause_correct"]
            else 1.0
            if action_correct
            else 0.5
        )
        if normalized_protocol != expected_score:
            raise ValueError("full_protocol_score conflicts with correctness flags")
    return {
        "diagnosis_score": diagnosis,
        "root_cause_correct": value["root_cause_correct"],
        "evidence_grounded": value["evidence_grounded"],
        "corrective_actions_evaluable": expected_protocol_evaluable,
        "corrective_actions_correct": action_correct,
        "full_protocol_score": normalized_protocol,
        "reason": value["reason"].strip(),
    }


def call_judge(
    client: Any,
    model: str,
    payload: dict[str, Any],
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    expected = bool(payload["corrective_actions_evaluable"])
    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=JUDGE_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False, indent=2),
            )
            raw = base.response_output_text(response).strip()
            parsed = validate_judge_output(parse_judge_json(raw), expected)
            return {
                "status": "ok",
                "judge_model": model,
                "raw_response": raw,
                "parsed": parsed,
                "attempt": attempt,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1)
                print(f"    Judge error; retrying in {wait_seconds}s: {exc}")
                time.sleep(wait_seconds)
    raise RuntimeError(f"Judge failed after {retries} attempts: {last_error}")


def judge_models(args: argparse.Namespace) -> list[str]:
    models = args.judge_models or [
        item.strip()
        for item in os.getenv("JUDGE_MODELS", "gpt-5.5").split(",")
        if item.strip()
    ]
    if not models or len(models) != len(set(models)):
        raise SystemExit("Judge model names must be non-empty and distinct.")
    return models


def judged_summary(
    records: list[dict[str, Any]], conditions: tuple[str, str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in conditions:
        rows = [row for row in records if row.get("condition") == condition]
        diagnosis = [
            float(row["median_diagnosis_score"])
            for row in rows
            if row.get("median_diagnosis_score") is not None
        ]
        protocol = [
            float(row["median_full_protocol_score"])
            for row in rows
            if row.get("median_full_protocol_score") is not None
        ]
        result[condition] = {
            "n_total": len(rows),
            "n_diagnosis_scored": len(diagnosis),
            "mean_diagnosis_score": mean(diagnosis) if diagnosis else None,
            "n_protocol_scored": len(protocol),
            "mean_full_protocol_score": mean(protocol) if protocol else None,
            "format_valid_count": sum(
                bool(row.get("deterministic_validation", {}).get("valid"))
                for row in rows
            ),
        }
    return result


def command_judge(args: argparse.Namespace) -> None:
    source = base.load_json(args.input)
    models = judge_models(args)
    if source.get("status") != "completed":
        raise SystemExit("Generation is incomplete.")
    if source.get("contract_version") != CONTRACT_VERSION:
        raise SystemExit("Input is not a JSON-contract v2 generation file.")
    if source.get("same_target_input_across_conditions") is not True:
        raise SystemExit("Generation did not verify target-input parity.")
    source_records = require_list(source.get("records"), "generation records")
    conditions = generation_conditions(args.prefix)
    source_digest = base.sha256_file(args.input)

    if args.output.exists() and not args.overwrite:
        output = base.load_json(args.output)
        if (
            output.get("source_sha256") != source_digest
            or output.get("judge_models") != models
        ):
            raise SystemExit(
                "Existing judged output has a different source or Judge set."
            )
        print(f"Resuming: {args.output}")
    else:
        output = {
            "status": "running",
            "contract_version": CONTRACT_VERSION,
            "experiment_type": f"factorybench_{args.prefix}_json_contract_v2_judged",
            "source_file": str(args.input.resolve()),
            "source_sha256": source_digest,
            "judge_models": models,
            "formal_three_independent_judge_result": len(models) >= 3,
            "records": [],
            "summary": {},
        }
        base.atomic_save(args.output, output)

    existing = {
        (str(row.get("condition")), str(row.get("id"))): row
        for row in output.get("records", [])
    }
    client = base.require_api_client()
    for index, source_record in enumerate(source_records, start=1):
        key = (str(source_record["condition"]), str(source_record["id"]))
        current = existing.get(key)
        if current is not None:
            vote_models = {
                vote.get("judge_model") for vote in current.get("judge_votes", [])
            }
            if current.get("judge_status") == "invalid_target_contract" or (
                current.get("judge_status") == "ok" and vote_models == set(models)
            ):
                print(f"[{index}/{len(source_records)}] {key}: complete")
                continue

        row = dict(source_record)
        if current is None:
            output["records"].append(row)
            existing[key] = row
        else:
            current.clear()
            current.update(row)
            row = current
        row["judge_votes"] = []
        row["judge_errors"] = []
        row["median_diagnosis_score"] = None
        row["median_full_protocol_score"] = None

        raw = str(row.get("raw_response") or "")
        digest_ok = row.get("raw_response_sha256_at_save") == base.sha256_text(raw)
        row["raw_response_still_matches_saved_digest"] = digest_ok
        format_valid = bool(row.get("deterministic_validation", {}).get("valid"))
        reference = row.get("reference_contract") or {}
        protocol_evaluable = bool(reference.get("corrective_actions_evaluable"))
        print(f"[{index}/{len(source_records)}] {key[0]} / {key[1]}")

        if not digest_ok:
            row["judge_status"] = "raw_digest_mismatch"
        elif not format_valid:
            row["judge_status"] = "invalid_target_contract"
            row["score_origin"] = "deterministic_contract_failure"
            row["median_diagnosis_score"] = 0.0
            row["median_full_protocol_score"] = 0.0 if protocol_evaluable else None
            print("    score=0.0 (invalid target JSON contract)")
        else:
            payload = {
                "question": row["target_input"],
                "canonical_root_cause": reference["canonical_root_cause"],
                "corrective_action_reference": reference[
                    "corrective_action_reference"
                ],
                "corrective_actions_evaluable": protocol_evaluable,
                "model_raw_answer": raw,
            }
            for model in models:
                try:
                    vote = call_judge(client, model, payload, args.retries)
                    row["judge_votes"].append(vote)
                    parsed = vote["parsed"]
                    print(
                        f"    {model}: diagnosis={parsed['diagnosis_score']}; "
                        f"protocol={parsed['full_protocol_score']}"
                    )
                except Exception as exc:  # noqa: BLE001
                    row["judge_errors"].append(
                        {"judge_model": model, "error": repr(exc)}
                    )
                    print(f"    {model}: ERROR {exc}")
            if row["judge_votes"]:
                parsed_votes = [vote["parsed"] for vote in row["judge_votes"]]
                row["median_diagnosis_score"] = float(
                    median(vote["diagnosis_score"] for vote in parsed_votes)
                )
                protocol_scores = [
                    vote["full_protocol_score"]
                    for vote in parsed_votes
                    if vote["full_protocol_score"] is not None
                ]
                row["median_full_protocol_score"] = (
                    float(median(protocol_scores)) if protocol_scores else None
                )
                row["judge_status"] = "ok"
                row["score_origin"] = "semantic_judge"
            else:
                row["judge_status"] = "error"
        output["summary"] = judged_summary(output["records"], conditions)
        base.atomic_save(args.output, output)

    output["summary"] = judged_summary(output["records"], conditions)
    expected_record_count = len(source_records)
    scored_record_count = sum(
        row.get("median_diagnosis_score") is not None for row in output["records"]
    )
    output["status"] = (
        "completed" if scored_record_count == expected_record_count else "partial"
    )
    output["metric_boundary"] = {
        "diagnosis": "All cases, including strict-contract failures scored as unusable zero.",
        "full_protocol": "Only cases with semantic corrective-action gold.",
    }
    base.atomic_save(args.output, output)
    print(f"Saved: {args.output}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    if output["status"] != "completed":
        raise SystemExit(
            "Judging is partial. Re-run the same command to resume failed Judge calls."
        )


def records_by_case(
    payload: dict[str, Any], conditions: tuple[str, str]
) -> dict[str, dict[str, dict[str, Any]]]:
    if payload.get("status") != "completed":
        raise ValueError("Judged file is incomplete.")
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in require_list(payload.get("records"), "judged records"):
        condition = str(row.get("condition"))
        if condition not in conditions:
            continue
        case_id = str(row.get("id"))
        if condition in result[case_id]:
            raise ValueError(f"Duplicate record: {condition}/{case_id}")
        result[case_id][condition] = row
    return dict(result)


def compare_payloads(
    mini: dict[str, Any], gpt55: dict[str, Any]
) -> dict[str, Any]:
    if mini.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("Mini result is not JSON-contract v2.")
    if gpt55.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("GPT-5.5 result is not JSON-contract v2.")
    mini_conditions = generation_conditions("mini")
    gpt_conditions = generation_conditions("gpt55")
    mini_rows = records_by_case(mini, mini_conditions)
    gpt_rows = records_by_case(gpt55, gpt_conditions)
    errors: list[str] = []
    if set(mini_rows) != set(gpt_rows):
        errors.append("case ids differ")
    if mini.get("judge_models") != gpt55.get("judge_models"):
        errors.append("Judge models differ")

    per_case: list[dict[str, Any]] = []
    for case_id in sorted(set(mini_rows) & set(gpt_rows)):
        mr = mini_rows[case_id]
        gr = gpt_rows[case_id]
        if set(mr) != set(mini_conditions) or set(gr) != set(gpt_conditions):
            errors.append(f"{case_id}: incomplete condition pair")
            continue
        roles = {
            "mini_no_skill": mr["mini_no_skill"],
            "mini_skill_v2": mr["mini_skill_v2"],
            "gpt55_no_skill": gr["gpt55_no_skill"],
            "gpt55_skill_v2": gr["gpt55_skill_v2"],
        }
        hashes = {row.get("target_input_sha256") for row in roles.values()}
        refs = {row.get("reference_contract_sha256") for row in roles.values()}
        if len(hashes) != 1:
            errors.append(f"{case_id}: target-input hashes differ")
        if len(refs) != 1:
            errors.append(f"{case_id}: reference-contract hashes differ")
        if (
            roles["mini_no_skill"].get("instructions_sha256")
            != roles["gpt55_no_skill"].get("instructions_sha256")
        ):
            errors.append(f"{case_id}: no-Skill instructions differ")
        if (
            roles["mini_skill_v2"].get("instructions_sha256")
            != roles["gpt55_skill_v2"].get("instructions_sha256")
        ):
            errors.append(f"{case_id}: Skill-v2 instructions differ")
        diagnosis = {
            name: row.get("median_diagnosis_score") for name, row in roles.items()
        }
        if any(value is None for value in diagnosis.values()):
            errors.append(f"{case_id}: missing diagnosis score")
            continue
        protocol = {
            name: row.get("median_full_protocol_score")
            for name, row in roles.items()
        }
        reference = roles["mini_no_skill"]["reference_contract"]
        per_case.append(
            {
                "id": case_id,
                "protocol_evaluable": reference["corrective_actions_evaluable"],
                **{f"diagnosis_{k}": float(v) for k, v in diagnosis.items()},
                **{f"protocol_{k}": v for k, v in protocol.items()},
            }
        )
    if errors:
        raise ValueError("Experimental parity failed:\n- " + "\n- ".join(errors))
    if not per_case:
        raise ValueError("No complete comparable cases.")

    condition_names = (
        "mini_no_skill",
        "mini_skill_v2",
        "gpt55_no_skill",
        "gpt55_skill_v2",
    )
    diagnosis_means = {
        name: mean(row[f"diagnosis_{name}"] for row in per_case)
        for name in condition_names
    }
    protocol_rows = [row for row in per_case if row["protocol_evaluable"]]
    protocol_means = {
        name: (
            mean(float(row[f"protocol_{name}"]) for row in protocol_rows)
            if protocol_rows
            else None
        )
        for name in condition_names
    }
    diagnosis_skill_mini = (
        diagnosis_means["mini_skill_v2"] - diagnosis_means["mini_no_skill"]
    )
    diagnosis_skill_gpt = (
        diagnosis_means["gpt55_skill_v2"] - diagnosis_means["gpt55_no_skill"]
    )
    return {
        "status": "completed",
        "contract_version": CONTRACT_VERSION,
        "comparison_type": "descriptive_paired_2x2_model_by_skill_json_contract_v2",
        "n_diagnosis_cases": len(per_case),
        "n_full_protocol_cases": len(protocol_rows),
        "judge_models": mini.get("judge_models"),
        "experimental_parity": {
            "same_case_ids": True,
            "same_target_input_hashes": True,
            "same_reference_contract_hashes": True,
            "same_instruction_hashes_by_role": True,
            "same_judge_models": True,
        },
        "diagnosis_mean_scores_all_cases": diagnosis_means,
        "diagnosis_skill_effect": {
            "mini": diagnosis_skill_mini,
            "gpt55": diagnosis_skill_gpt,
        },
        "diagnosis_model_effect_gpt55_minus_mini": {
            "no_skill": diagnosis_means["gpt55_no_skill"]
            - diagnosis_means["mini_no_skill"],
            "skill_v2": diagnosis_means["gpt55_skill_v2"]
            - diagnosis_means["mini_skill_v2"],
        },
        "diagnosis_difference_in_differences": diagnosis_skill_gpt
        - diagnosis_skill_mini,
        "full_protocol_mean_scores_evaluable_subset": protocol_means,
        "per_case": per_case,
        "limitations": [
            "The comparison is descriptive for a small held-out subset.",
            "Full-protocol scores exclude cases whose prepared reference was only an option letter.",
            "A single same-family Judge is preliminary unless multiple distinct Judge models are configured.",
            "JSON-contract v1 results are not comparable and must not be pooled with v2.",
        ],
    }


def command_compare(args: argparse.Namespace) -> None:
    mini = base.load_json(args.mini)
    gpt55 = base.load_json(args.gpt55)
    comparison = compare_payloads(mini, gpt55)
    comparison["mini_source_file"] = str(args.mini.resolve())
    comparison["mini_source_sha256"] = base.sha256_file(args.mini)
    comparison["gpt55_source_file"] = str(args.gpt55.resolve())
    comparison["gpt55_source_sha256"] = base.sha256_file(args.gpt55)
    base.atomic_save(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


def synthetic_reference(letter: bool) -> dict[str, Any]:
    return normalize_reference(
        {
            "id": "fixture",
            "root_cause": "collision_foam_object",
            "reference_answer": "B" if letter else "Stop, inspect, and verify.",
        }
    )


def synthetic_judged(prefix: str) -> dict[str, Any]:
    records = []
    conditions = generation_conditions(prefix)
    for index in range(2):
        reference = synthetic_reference(letter=index == 1)
        ref_hash = base.sha256_text(
            json.dumps(reference, ensure_ascii=False, sort_keys=True)
        )
        for condition, score in zip(conditions, (0.0, 1.0)):
            records.append(
                {
                    "id": f"case_{index}",
                    "condition": condition,
                    "target_input_sha256": f"input_{index}",
                    "instructions_sha256": f"instructions_{condition_role(condition)}",
                    "reference_contract_sha256": ref_hash,
                    "reference_contract": reference,
                    "median_diagnosis_score": score,
                    "median_full_protocol_score": (
                        score if reference["corrective_actions_evaluable"] else None
                    ),
                }
            )
    return {
        "status": "completed",
        "contract_version": CONTRACT_VERSION,
        "judge_models": ["judge"],
        "records": records,
    }


def command_self_test(_: argparse.Namespace) -> None:
    original = "Machine.\n\nSignal mapping:\n- x: signal\n\nTime series:\nt=1: x=0\n\nQuestion:\nAnswer only with the letter (e.g. A)."
    rewritten = rewrite_target_input(original)
    if "answer only with the letter" in rewritten.lower():
        raise SystemExit("Question rewrite regression failed.")
    if rewritten.count("Question:") != 1 or STANDARD_QUESTION not in rewritten:
        raise SystemExit("Standard question regression failed.")

    letter = synthetic_reference(letter=True)
    semantic = synthetic_reference(letter=False)
    if letter["corrective_actions_evaluable"] or letter["corrective_action_reference"]:
        raise SystemExit("Letter-reference boundary regression failed.")
    if not semantic["corrective_actions_evaluable"]:
        raise SystemExit("Semantic-reference regression failed.")

    valid_target = json.dumps(
        {
            "root_cause": "collision with foam object",
            "evidence": ["force changed"],
            "corrective_actions": ["stop, inspect, and verify"],
        }
    )
    if not evaluator.validate_deterministically(valid_target)["valid"]:
        raise SystemExit("Target JSON validator regression failed.")

    judge_no_protocol = validate_judge_output(
        {
            "diagnosis_score": 1.0,
            "root_cause_correct": True,
            "evidence_grounded": True,
            "corrective_actions_evaluable": False,
            "corrective_actions_correct": None,
            "full_protocol_score": None,
            "reason": "fixture",
        },
        False,
    )
    if judge_no_protocol["full_protocol_score"] is not None:
        raise SystemExit("Judge unavailable-protocol regression failed.")

    comparison = compare_payloads(synthetic_judged("mini"), synthetic_judged("gpt55"))
    if comparison["n_diagnosis_cases"] != 2:
        raise SystemExit("Comparison diagnosis coverage regression failed.")
    if comparison["n_full_protocol_cases"] != 1:
        raise SystemExit("Comparison protocol coverage regression failed.")
    print("FactoryBench JSON-contract v2 offline self-test PASSED.")


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("FACTORYBENCH_DATASET", str(DEFAULT_DATASET))),
    )
    parser.add_argument("--split", default=os.getenv("EVAL_SPLIT", "final_test"))
    parser.add_argument("--expected-cases", type=int, default=5)


def add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    prefix: str,
    model: str,
    output: Path,
) -> None:
    add_dataset_arguments(parser)
    parser.add_argument("--model", default=model)
    parser.add_argument("--skill", type=Path, default=DEFAULT_SKILL)
    parser.add_argument("--contract-audit", type=Path, default=DEFAULT_CONTRACT_AUDIT)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=command_run, prefix=prefix)


def add_judge_arguments(
    parser: argparse.ArgumentParser,
    *,
    prefix: str,
    input_path: Path,
    output: Path,
) -> None:
    parser.add_argument("--input", type=Path, default=input_path)
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        help="Repeat for distinct Judge models. Default: JUDGE_MODELS or gpt-5.5.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=command_judge, prefix=prefix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FactoryBench diagnostic JSON-contract v2 control experiment."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    test = commands.add_parser("self-test", help="Run offline regression tests.")
    test.set_defaults(func=command_self_test)

    check = commands.add_parser("check-contract", help="Audit v2 inputs and gold.")
    add_dataset_arguments(check)
    check.add_argument("--output", type=Path, default=DEFAULT_CONTRACT_AUDIT)
    check.set_defaults(func=command_check_contract)

    run_gpt = commands.add_parser("run-gpt55", help="Run GPT-5.5 v2 control.")
    add_run_arguments(
        run_gpt,
        prefix="gpt55",
        model=os.getenv("STRONG_MODEL", "gpt-5.5"),
        output=DEFAULT_GPT55_GENERATION,
    )
    run_mini = commands.add_parser("run-mini", help="Run gpt-4o-mini v2 control.")
    add_run_arguments(
        run_mini,
        prefix="mini",
        model=os.getenv("MINI_MODEL", "gpt-4o-mini"),
        output=DEFAULT_MINI_GENERATION,
    )

    judge_gpt = commands.add_parser("judge-gpt55", help="Judge GPT-5.5 v2 output.")
    add_judge_arguments(
        judge_gpt,
        prefix="gpt55",
        input_path=DEFAULT_GPT55_GENERATION,
        output=DEFAULT_GPT55_JUDGED,
    )
    judge_mini = commands.add_parser("judge-mini", help="Judge mini v2 output.")
    add_judge_arguments(
        judge_mini,
        prefix="mini",
        input_path=DEFAULT_MINI_GENERATION,
        output=DEFAULT_MINI_JUDGED,
    )

    compare = commands.add_parser("compare", help="Compare the complete v2 2x2.")
    compare.add_argument("--mini", type=Path, default=DEFAULT_MINI_JUDGED)
    compare.add_argument("--gpt55", type=Path, default=DEFAULT_GPT55_JUDGED)
    compare.add_argument("--output", type=Path, default=DEFAULT_COMPARISON)
    compare.set_defaults(func=command_compare)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "expected_cases") and args.expected_cases < 0:
        raise SystemExit("--expected-cases must be zero or positive.")
    if hasattr(args, "retries") and args.retries < 1:
        raise SystemExit("--retries must be at least 1.")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Saved API records can be resumed.", file=sys.stderr)
        raise SystemExit(130)