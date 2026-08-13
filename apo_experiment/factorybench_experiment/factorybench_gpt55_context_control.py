from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "data" / "factorybench_l4_subset.json"
DEFAULT_SKILL = SCRIPT_DIR / "skills" / "skill_v2.md"
DEFAULT_RESULTS = SCRIPT_DIR / "results"
DEFAULT_CONTEXT_AUDIT = DEFAULT_RESULTS / "gpt55_neutral_context_audit.json"
DEFAULT_GENERATION = DEFAULT_RESULTS / "gpt55_neutral_context_control.json"
DEFAULT_JUDGED = DEFAULT_RESULTS / "gpt55_neutral_context_control_judged_gpt-5.5.json"

CONDITIONS = ("gpt55_no_skill", "gpt55_skill_v2")
REQUIRED_TARGET_KEYS = {"root_cause", "evidence", "corrective_actions"}

TARGET_SCHEMA = """
Return JSON only, with exactly this structure:
{
  "root_cause": "the most likely underlying physical or operational cause",
  "evidence": ["specific signal or task-phase evidence from the input"],
  "corrective_actions": ["corrective action and a verification step"]
}
Do not return Markdown, headings, a bare option letter, or a dataset label.
""".strip()

COMMON_INSTRUCTIONS = f"""
You are solving a FactoryBench Level 4 industrial troubleshooting case.

Use the machine description, signal mapping, time series, and task question in
the user input. Distinguish an observed symptom from the underlying physical or
operational root cause. Ground every evidence item in the supplied input. Do
not invent signals, machine specifications, fault documentation, or SOPs. If
the supplied telemetry does not uniquely identify a cause, say so in the
evidence array and return the most defensible diagnosis supported by the input.

{TARGET_SCHEMA}
""".strip()

HEADING_PATTERNS = {
    "mapping": re.compile(
        r"^[ \t]*(?:Signal|Acronym)[ \t]+mapping[ \t]*:[ \t]*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "time_series": re.compile(
        r"^[ \t]*Time[ \t]+series[ \t]*:[ \t]*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "question": re.compile(
        r"^[ \t]*Question[ \t]*:[ \t]*(.*)$",
        re.IGNORECASE | re.MULTILINE,
    ),
}
MAPPING_ENTRY = re.compile(
    r"^[ \t]*(?:[-*][ \t]*)?([A-Za-z][A-Za-z0-9_]*)[ \t]*:[ \t]*(\S.*)$",
    re.MULTILINE,
)
SIGNAL_ASSIGNMENT = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)=")
TIME_ROW = re.compile(r"^[ \t]*t\s*=", re.IGNORECASE | re.MULTILINE)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def atomic_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def import_verified_evaluator() -> Any:
    try:
        import factorybench_evaluator_smoke_test as evaluator
    except ImportError as exc:
        raise SystemExit(
            "factorybench_evaluator_smoke_test.py must be in the same directory."
        ) from exc

    required = (
        "validate_deterministically",
        "call_judge",
        "run_parser_regression_tests",
    )
    missing = [name for name in required if not hasattr(evaluator, name)]
    if missing:
        raise SystemExit(
            "Verified evaluator is missing functions: " + ", ".join(missing)
        )
    return evaluator


def require_api_client() -> Any:
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_ADMIN_KEY"):
        raise SystemExit("OPENAI_API_KEY or OPENAI_ADMIN_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the openai Python package in this venv.") from exc
    return OpenAI()


def load_cases(
    dataset_path: Path,
    split: str,
    expected_cases: int,
) -> list[dict[str, Any]]:
    dataset = load_json(dataset_path)
    splits = dataset.get("splits") if isinstance(dataset, dict) else None
    if not isinstance(splits, dict):
        raise ValueError("Dataset must contain an object field named 'splits'.")
    if split not in splits:
        raise ValueError(
            f"Split {split!r} not found. Available splits: {list(splits)}"
        )
    cases = splits[split]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Split {split!r} is empty or not a list.")
    if expected_cases > 0 and len(cases) != expected_cases:
        raise ValueError(
            f"Expected exactly {expected_cases} cases in {split!r}; found {len(cases)}."
        )

    seen_ids: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} in {split!r} is not an object.")
        missing = [
            key
            for key in ("id", "input", "root_cause", "reference_answer")
            if case.get(key) in (None, "")
        ]
        if missing:
            raise ValueError(f"Case {index} is missing required fields: {missing}")
        if not isinstance(case["input"], str):
            raise ValueError(
                f"Case {case['id']!r} input must be the prepared prompt string; "
                f"got {type(case['input']).__name__}."
            )
        case_id = str(case["id"])
        if case_id in seen_ids:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen_ids.add(case_id)
    return cases


def _section_text(inline: str, following: str) -> str:
    return "\n".join(part for part in (inline.strip(), following.strip()) if part)


def extract_prompt_sections(prompt: str) -> dict[str, str]:
    """Extract the confirmed prepared-subset sections without rebuilding input."""
    matches = {name: pattern.search(prompt) for name, pattern in HEADING_PATTERNS.items()}
    if any(match is None for match in matches.values()):
        return {
            "machine_sentence": "",
            "acronym_mapping": "",
            "time_series": "",
            "question": "",
        }

    mapping = matches["mapping"]
    time_series = matches["time_series"]
    question = matches["question"]
    assert mapping is not None and time_series is not None and question is not None
    if not (mapping.start() < time_series.start() < question.start()):
        return {
            "machine_sentence": "",
            "acronym_mapping": "",
            "time_series": "",
            "question": "",
        }

    return {
        "machine_sentence": prompt[: mapping.start()].strip(),
        "acronym_mapping": _section_text(
            mapping.group(1), prompt[mapping.end() : time_series.start()]
        ),
        "time_series": _section_text(
            time_series.group(1), prompt[time_series.end() : question.start()]
        ),
        "question": _section_text(question.group(1), prompt[question.end() :]),
    }


def audit_case_context(case: dict[str, Any]) -> dict[str, Any]:
    prompt = str(case["input"])
    errors: list[str] = []
    warnings: list[str] = []

    heading_matches = {
        name: pattern.search(prompt) for name, pattern in HEADING_PATTERNS.items()
    }
    for name, match in heading_matches.items():
        if match is None:
            errors.append(f"missing_{name}_heading")

    if all(match is not None for match in heading_matches.values()):
        starts = [
            heading_matches["mapping"].start(),
            heading_matches["time_series"].start(),
            heading_matches["question"].start(),
        ]
        if starts != sorted(starts) or len(set(starts)) != 3:
            errors.append("prompt_sections_out_of_order")

    sections = extract_prompt_sections(prompt)
    for name, text in sections.items():
        if not text.strip():
            errors.append(f"empty_{name}")

    mapping_pairs = MAPPING_ENTRY.findall(sections["acronym_mapping"])
    mapping = {alias: description.strip() for alias, description in mapping_pairs}
    if not mapping:
        errors.append("no_signal_mapping_entries")
    if len(mapping) != len(mapping_pairs):
        errors.append("duplicate_signal_mapping_alias")
    empty_descriptions = sorted(alias for alias, description in mapping.items() if not description)
    if empty_descriptions:
        errors.append("empty_signal_descriptions")

    assignments = set(SIGNAL_ASSIGNMENT.findall(sections["time_series"]))
    assignments.discard("t")
    assignments.discard("T")
    unknown_signals = sorted(assignments - set(mapping))
    unused_mapping = sorted(set(mapping) - assignments)
    if not assignments:
        errors.append("no_time_series_signal_assignments")
    if unknown_signals:
        errors.append("time_series_uses_unmapped_signals")
    if unused_mapping:
        warnings.append("mapping_contains_unused_signals")

    time_rows = len(TIME_ROW.findall(sections["time_series"]))
    if time_rows == 0:
        errors.append("no_time_series_rows")

    reference = value_to_text(case.get("reference_answer", "")).strip()
    exact_reference_in_input = bool(reference and reference in prompt)
    if exact_reference_in_input:
        errors.append("exact_reference_answer_found_in_target_input")

    # A root-cause label may legitimately appear among multiple-choice options,
    # so it is reported for human review but is not automatically called leakage.
    root_label = str(case.get("root_cause", "")).strip()
    root_label_in_input = bool(root_label and root_label.lower() in prompt.lower())
    if root_label_in_input:
        warnings.append("canonical_root_label_present_review_options_or_leakage")

    return {
        "id": str(case["id"]),
        "valid": not errors,
        "input_sha256": sha256_text(prompt),
        "input_characters": len(prompt),
        "machine_sentence": sections["machine_sentence"],
        "machine_sentence_characters": len(sections["machine_sentence"]),
        "signal_mapping_count": len(mapping),
        "mapped_signals": sorted(mapping),
        "time_series_rows": time_rows,
        "time_series_signal_count": len(assignments),
        "unmapped_time_series_signals": unknown_signals,
        "unused_mapping_signals": unused_mapping,
        "question_characters": len(sections["question"]),
        "exact_reference_answer_found_in_target_input": exact_reference_in_input,
        "canonical_root_label_found_in_target_input": root_label_in_input,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def audit_contexts(
    dataset_path: Path,
    split: str,
    expected_cases: int,
) -> dict[str, Any]:
    cases = load_cases(dataset_path, split, expected_cases)
    records = [audit_case_context(case) for case in cases]
    return {
        "audit_type": "factorybench_neutral_target_context_contract",
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "split": split,
        "case_count": len(cases),
        "expected_case_count": expected_cases,
        "all_cases_valid": all(record["valid"] for record in records),
        "error_count": sum(len(record["errors"]) for record in records),
        "warning_count": sum(len(record["warnings"]) for record in records),
        "records": records,
        "boundary": (
            "This deterministic audit verifies the prepared prompt structure, "
            "signal-name coverage, and exact reference-answer absence. It cannot "
            "prove that every sentence is semantically neutral; warnings require "
            "human review."
        ),
    }


def print_context_audit(audit: dict[str, Any]) -> None:
    for record in audit["records"]:
        state = "PASS" if record["valid"] else "FAIL"
        print(
            f"{record['id']}: {state}; mappings={record['signal_mapping_count']}; "
            f"rows={record['time_series_rows']}; "
            f"unmapped={len(record['unmapped_time_series_signals'])}"
        )
        for error in record["errors"]:
            print(f"  ERROR: {error}")
        for warning in record["warnings"]:
            print(f"  WARNING: {warning}")
    print(
        f"Context audit: valid={audit['all_cases_valid']}; "
        f"errors={audit['error_count']}; warnings={audit['warning_count']}"
    )


def build_instructions(condition: str, skill_text: str) -> str:
    if condition == "gpt55_no_skill":
        supplement = """
No reusable diagnostic Skill, RAG passage, fault catalog, signal-analysis tool,
gold root cause, or reference answer is supplied in this condition.
""".strip()
    elif condition == "gpt55_skill_v2":
        supplement = f"""
Use the following reusable diagnostic procedure. It is a reasoning procedure,
not a machine-specific fault catalog. The user input remains the only source of
case-specific evidence.

--- BEGIN DIAGNOSTIC SKILL V2 ---
{skill_text}
--- END DIAGNOSTIC SKILL V2 ---

No RAG passage, fault catalog, signal-analysis tool, gold root cause, or
reference answer is supplied in this condition.
""".strip()
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return f"{COMMON_INSTRUCTIONS}\n\n{supplement}"


def call_target_model(
    client: Any,
    model: str,
    instructions: str,
    target_input: str,
    retries: int,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return client.responses.create(
                model=model,
                instructions=instructions,
                input=target_input,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                wait_seconds = 2 ** (attempt - 1)
                print(f"  API error; retrying in {wait_seconds}s: {exc}")
                time.sleep(wait_seconds)
    raise RuntimeError(f"Target call failed after {retries} attempts: {last_error}")


def response_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is None:
        raise ValueError("API response does not contain output_text.")
    return str(output_text)


def response_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    result: dict[str, Any] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, name, None)
        if value is not None:
            result[name] = value
    return result or {"repr": repr(usage)}


def generation_configuration(
    args: argparse.Namespace,
    skill_text: str,
    context_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "evaluation_split": args.split,
        "expected_cases": args.expected_cases,
        "target_model": args.model,
        "conditions": list(CONDITIONS),
        "skill_path": str(args.skill.resolve()),
        "skill_sha256": sha256_text(skill_text),
        "skill_text_snapshot": skill_text,
        "seed": args.seed,
        "retries": args.retries,
        "output_contract": sorted(REQUIRED_TARGET_KEYS),
        "domain_rag": None,
        "fault_catalog": None,
        "signal_tool": None,
        "held_out_gold_supplied_to_target": False,
        "context_audit_all_cases_valid": context_audit["all_cases_valid"],
        "context_audit_records": context_audit["records"],
    }


def initialize_or_resume_generation(
    path: Path,
    configuration: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        output = load_json(path)
        if output.get("configuration") != configuration:
            raise SystemExit(
                "Existing output has a different configuration. Use another "
                "--output or explicitly pass --overwrite."
            )
        print(f"Resuming: {path}")
        return output
    output = {
        "status": "running",
        "experiment_type": "factorybench_gpt55_neutral_context_skill_control",
        "configuration": configuration,
        "records": [],
        "generation_summary": {},
        "same_target_input_across_conditions": None,
    }
    atomic_save(path, output)
    return output


def ensure_saved_raw_is_parsed(
    record: dict[str, Any],
    evaluator: Any,
) -> None:
    raw = record.get("raw_response")
    digest = record.get("raw_response_sha256_at_save")
    if not isinstance(raw, str) or not isinstance(digest, str):
        raise ValueError("Saved raw response or digest is missing.")
    if sha256_text(raw) != digest:
        raise ValueError("Saved raw response no longer matches its original digest.")
    validation = evaluator.validate_deterministically(raw)
    record["deterministic_validation"] = validation
    record["parsed_response"] = validation.get("parsed")
    record["raw_response_still_matches_saved_digest"] = sha256_text(raw) == digest
    record["status"] = "ok"


def summarize_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [record for record in records if record.get("condition") == condition]
        result[condition] = {
            "n": len(rows),
            "successful": sum(record.get("status") == "ok" for record in rows),
            "format_valid": sum(
                bool(record.get("deterministic_validation", {}).get("valid"))
                for record in rows
            ),
        }
    return result


def same_inputs_across_conditions(records: list[dict[str, Any]]) -> bool:
    by_case: dict[str, set[str]] = defaultdict(set)
    condition_sets: dict[str, set[str]] = defaultdict(set)
    for record in records:
        case_id = str(record.get("id"))
        condition = str(record.get("condition"))
        digest = str(record.get("target_input_sha256"))
        by_case[case_id].add(digest)
        condition_sets[case_id].add(condition)
    return bool(by_case) and all(
        len(digests) == 1 and condition_sets[case_id] == set(CONDITIONS)
        for case_id, digests in by_case.items()
    )


def command_check_context(args: argparse.Namespace) -> None:
    audit = audit_contexts(args.dataset, args.split, args.expected_cases)
    atomic_save(args.output, audit)
    print_context_audit(audit)
    print(f"Saved: {args.output}")
    if not audit["all_cases_valid"]:
        raise SystemExit(1)


def command_run(args: argparse.Namespace) -> None:
    evaluator = import_verified_evaluator()
    cases = load_cases(args.dataset, args.split, args.expected_cases)
    if not args.skill.exists():
        raise FileNotFoundError(f"Skill not found: {args.skill}")
    skill_text = args.skill.read_text(encoding="utf-8")
    if not skill_text.strip():
        raise ValueError("Skill v2 file is empty.")

    context_audit = audit_contexts(args.dataset, args.split, args.expected_cases)
    atomic_save(args.context_audit, context_audit)
    print_context_audit(context_audit)
    if not context_audit["all_cases_valid"]:
        raise SystemExit(
            f"Context audit failed. Inspect {args.context_audit}; no API calls were made."
        )

    configuration = generation_configuration(args, skill_text, context_audit)
    output = initialize_or_resume_generation(args.output, configuration, args.overwrite)
    client = require_api_client()

    existing_by_key = {
        (str(record.get("condition")), str(record.get("id"))): record
        for record in output.get("records", [])
    }
    jobs = [
        (condition, index, case)
        for condition in CONDITIONS
        for index, case in enumerate(cases, start=1)
    ]
    random.Random(args.seed).shuffle(jobs)

    for job_index, (condition, case_index, case) in enumerate(jobs, start=1):
        case_id = str(case.get("id", f"case_{case_index}"))
        key = (condition, case_id)
        record = existing_by_key.get(key)

        if record is not None and record.get("status") == "ok":
            print(f"[{job_index}/{len(jobs)}] {condition} / {case_id}: already complete")
            continue

        # If the process stopped after the immutable raw save, parse it locally
        # instead of paying for a duplicate target-model call.
        if record is not None and record.get("status") == "raw_response_saved":
            print(f"[{job_index}/{len(jobs)}] {condition} / {case_id}: parsing saved raw")
            try:
                ensure_saved_raw_is_parsed(record, evaluator)
            except Exception as exc:  # noqa: BLE001
                record["status"] = "parser_error"
                record["parser_error"] = repr(exc)
            atomic_save(args.output, output)
            continue

        instructions = build_instructions(condition, skill_text)
        prompt = case["input"]
        initial_record: dict[str, Any] = {
            "condition": condition,
            "id": case_id,
            "status": "calling_model",
            "target_model": args.model,
            "skill_version": "skill_v2" if condition == "gpt55_skill_v2" else None,
            "rag_version": None,
            "tool_version": None,
            "retrieved_documents": [],
            "tool_output": {},
            "target_input": prompt,
            "target_input_sha256": sha256_text(prompt),
            "expected_root_cause": case["root_cause"],
            "reference_answer": case["reference_answer"],
            "metadata": case.get("metadata"),
            "instructions_sha256": sha256_text(instructions),
            "raw_response": None,
            "raw_response_sha256_at_save": None,
            "deterministic_validation": None,
            "parsed_response": None,
            "latency_seconds": None,
            "token_usage": None,
        }
        if record is None:
            record = initial_record
            output["records"].append(record)
            existing_by_key[key] = record
        else:
            record.clear()
            record.update(initial_record)
        atomic_save(args.output, output)

        print(f"[{job_index}/{len(jobs)}] {condition} / {case_id} / {args.model}")
        started = time.time()
        try:
            response = call_target_model(
                client,
                args.model,
                instructions,
                prompt,
                args.retries,
            )
            raw = response_output_text(response)
            record.update(
                {
                    "status": "raw_response_saved",
                    "response_id": getattr(response, "id", None),
                    "raw_response": raw,
                    "raw_response_sha256_at_save": sha256_text(raw),
                    "latency_seconds": time.time() - started,
                    "token_usage": response_usage(response),
                }
            )
            # Required boundary: persist exact output_text before any parser runs.
            atomic_save(args.output, output)
            ensure_saved_raw_is_parsed(record, evaluator)
        except Exception as exc:  # noqa: BLE001
            if record.get("status") != "raw_response_saved":
                record["status"] = "target_model_error"
                record["target_model_error"] = repr(exc)
                record["latency_seconds"] = time.time() - started
            else:
                record["status"] = "parser_error"
                record["parser_error"] = repr(exc)
            print(f"  ERROR: {exc}")
        output["generation_summary"] = summarize_generation(output["records"])
        atomic_save(args.output, output)

    output["status"] = "completed"
    output["generation_summary"] = summarize_generation(output["records"])
    output["same_target_input_across_conditions"] = same_inputs_across_conditions(
        output["records"]
    )
    atomic_save(args.output, output)
    print(f"\nSaved: {args.output}")
    print(json.dumps(output["generation_summary"], ensure_ascii=False, indent=2))
    print(
        "Same target input across conditions:",
        output["same_target_input_across_conditions"],
    )


def judge_models(args: argparse.Namespace) -> list[str]:
    models = args.judge_models or [
        part.strip()
        for part in os.getenv("JUDGE_MODELS", "gpt-5.5").split(",")
        if part.strip()
    ]
    if not models:
        raise SystemExit("At least one Judge model is required.")
    if len(models) != len(set(models)):
        raise SystemExit(
            "Judge model names must be distinct; repeating one model is not an "
            "independent-Judge experiment."
        )
    return models


def target_answer_usable(raw: str) -> tuple[bool, str | None]:
    stripped = raw.strip()
    if not stripped:
        return False, "empty_response"
    if re.fullmatch(r"[ABCD][.)]?", stripped, flags=re.IGNORECASE):
        return False, "option_letter_only"
    return True, None


def judge_payload(record: dict[str, Any]) -> dict[str, str]:
    return {
        "question": value_to_text(record.get("target_input")),
        "reference_answer": value_to_text(record.get("reference_answer")),
        "known_root_cause": value_to_text(record.get("expected_root_cause")),
        "model_raw_answer": str(record.get("raw_response", "")),
    }


def score_record(
    client: Any,
    evaluator: Any,
    record: dict[str, Any],
    models: list[str],
) -> None:
    raw = str(record.get("raw_response") or "")
    usable, reason = target_answer_usable(raw)
    record["semantic_judge_eligible"] = usable
    record["semantic_judge_ineligible_reason"] = reason
    record["judge_votes"] = []
    record["judge_errors"] = []
    record["median_score"] = None
    if not usable:
        record["judge_status"] = "skipped_unusable_answer"
        return

    payload = judge_payload(record)
    for model in models:
        try:
            vote = evaluator.call_judge(client, model, payload)
            vote["judge_model"] = model
            record["judge_votes"].append(vote)
            print(f"    {model}: {vote['parsed']['score']}")
        except Exception as exc:  # noqa: BLE001
            record["judge_errors"].append(
                {"judge_model": model, "error": repr(exc)}
            )
            print(f"    {model}: ERROR {exc}")
    if record["judge_votes"]:
        record["median_score"] = float(
            median(vote["parsed"]["score"] for vote in record["judge_votes"])
        )
        record["judge_status"] = "ok"
    else:
        record["judge_status"] = "error"


def summarize_judged(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        rows = [record for record in records if record.get("condition") == condition]
        scored = [record for record in rows if record.get("median_score") is not None]
        scores = [float(record["median_score"]) for record in scored]
        result[condition] = {
            "n_total": len(rows),
            "n_scored": len(scored),
            "mean_l4_score": mean(scores) if scores else None,
            "zero_count": sum(score == 0.0 for score in scores),
            "partial_count": sum(score == 0.5 for score in scores),
            "full_count": sum(score == 1.0 for score in scores),
            "format_valid_count": sum(
                bool(record.get("deterministic_validation", {}).get("valid"))
                for record in rows
            ),
        }
    return result


def compute_paired_skill_effect(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("median_score") is not None:
            by_case[str(record["id"])][str(record["condition"])] = float(
                record["median_score"]
            )
    per_case = []
    for case_id, scores in sorted(by_case.items()):
        if set(CONDITIONS) <= set(scores):
            effect = scores["gpt55_skill_v2"] - scores["gpt55_no_skill"]
            per_case.append(
                {
                    "id": case_id,
                    "gpt55_no_skill": scores["gpt55_no_skill"],
                    "gpt55_skill_v2": scores["gpt55_skill_v2"],
                    "skill_effect": effect,
                }
            )
    effects = [row["skill_effect"] for row in per_case]
    return {
        "n_complete_pairs": len(per_case),
        "mean_paired_skill_effect": mean(effects) if effects else None,
        "improved_cases": sum(effect > 0 for effect in effects),
        "unchanged_cases": sum(effect == 0 for effect in effects),
        "worse_cases": sum(effect < 0 for effect in effects),
        "per_case": per_case,
        "interpretation_boundary": (
            "With five held-out cases, this is a descriptive paired comparison. "
            "A positive effect is consistent with a Skill contribution; zero does "
            "not by itself prove that the Skill is generally ineffective."
        ),
    }


def command_judge(args: argparse.Namespace) -> None:
    evaluator = import_verified_evaluator()
    models = judge_models(args)
    source = load_json(args.input)
    if source.get("status") != "completed":
        raise SystemExit("Generation file is not completed; finish or resume run first.")
    if source.get("same_target_input_across_conditions") is not True:
        raise SystemExit("Generation did not verify identical target inputs by case.")
    source_records = source.get("records")
    if not isinstance(source_records, list) or not source_records:
        raise ValueError("Generation file contains no records.")

    source_digest = sha256_file(args.input)
    if args.output.exists() and not args.overwrite:
        output = load_json(args.output)
        if (
            output.get("source_sha256") != source_digest
            or output.get("judge_models") != models
        ):
            raise SystemExit(
                "Existing judged output has a different source or Judge set. "
                "Use another --output or explicitly pass --overwrite."
            )
        print(f"Resuming: {args.output}")
    else:
        output = {
            "status": "running",
            "experiment_type": "factorybench_gpt55_context_control_judged",
            "source_file": str(args.input.resolve()),
            "source_sha256": source_digest,
            "judge_models": models,
            "formal_three_independent_judge_result": (
                len(models) >= 3 and len(set(models)) >= 3
            ),
            "records": [],
            "summary": {},
            "paired_skill_effect": {},
        }
        atomic_save(args.output, output)

    existing_by_key = {
        (str(record.get("condition")), str(record.get("id"))): record
        for record in output.get("records", [])
    }
    client = require_api_client()

    for index, source_record in enumerate(source_records, start=1):
        key = (str(source_record.get("condition")), str(source_record.get("id")))
        existing = existing_by_key.get(key)
        if existing is not None:
            vote_models = {
                vote.get("judge_model") for vote in existing.get("judge_votes", [])
            }
            if existing.get("judge_status") == "skipped_unusable_answer" or (
                existing.get("judge_status") == "ok" and vote_models == set(models)
            ):
                print(f"[{index}/{len(source_records)}] {key}: already judged")
                continue

        print(f"[{index}/{len(source_records)}] {key[0]} / {key[1]}")
        record = dict(source_record)
        if existing is None:
            output["records"].append(record)
            existing_by_key[key] = record
        else:
            existing.clear()
            existing.update(record)
            record = existing

        raw = str(record.get("raw_response") or "")
        record["raw_response_still_matches_saved_digest"] = (
            record.get("raw_response_sha256_at_save") == sha256_text(raw)
        )
        if not record["raw_response_still_matches_saved_digest"]:
            record["judge_status"] = "skipped_raw_digest_mismatch"
            record["median_score"] = None
        elif record.get("status") != "ok":
            record["judge_status"] = "skipped_target_or_parser_error"
            record["median_score"] = None
        else:
            score_record(client, evaluator, record, models)
        output["summary"] = summarize_judged(output["records"])
        output["paired_skill_effect"] = compute_paired_skill_effect(output["records"])
        atomic_save(args.output, output)

    output["status"] = "completed"
    output["summary"] = summarize_judged(output["records"])
    output["paired_skill_effect"] = compute_paired_skill_effect(output["records"])
    atomic_save(args.output, output)
    print(f"\nSaved: {args.output}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(output["paired_skill_effect"], ensure_ascii=False, indent=2))


def synthetic_case() -> dict[str, Any]:
    return {
        "id": "synthetic_case",
        "input": (
            "The sensor data comes from a test robot performing a test task.\n\n"
            "Signal mapping:\n"
            "- tq0: target_torque_0\n"
            "- cf0: contact_force_0\n\n"
            "Time series:\n"
            "t=100: tq0=1.0, cf0=0.1\n"
            "t=200: tq0=1.2, cf0=0.3\n\n"
            "Question:\n"
            "Does the machine show anomalous behavior? Diagnose and recover."
        ),
        "root_cause": "synthetic_fault",
        "reference_answer": "Stop the synthetic test and verify the fixture.",
        "metadata": {"synthetic": True},
    }


def command_self_test(_: argparse.Namespace) -> None:
    evaluator = import_verified_evaluator()
    parser_results = evaluator.run_parser_regression_tests()
    if not parser_results or not all(result.get("passed") for result in parser_results):
        raise SystemExit("Verified evaluator parser regression failed.")

    case = synthetic_case()
    audit = audit_case_context(case)
    if not audit["valid"]:
        raise SystemExit(f"Valid context fixture failed: {audit['errors']}")
    if audit["signal_mapping_count"] != 2 or audit["time_series_rows"] != 2:
        raise SystemExit("Context counter regression failed.")

    missing_mapping = dict(case)
    missing_mapping["input"] = str(case["input"]).replace("Signal mapping:", "Signals:")
    invalid_audit = audit_case_context(missing_mapping)
    if invalid_audit["valid"] or "missing_mapping_heading" not in invalid_audit["errors"]:
        raise SystemExit("Missing-mapping regression failed.")

    leaked = dict(case)
    leaked["input"] = str(case["input"]) + "\n" + str(case["reference_answer"])
    leak_audit = audit_case_context(leaked)
    if "exact_reference_answer_found_in_target_input" not in leak_audit["errors"]:
        raise SystemExit("Reference-leak regression failed.")

    no_skill = build_instructions("gpt55_no_skill", "example skill")
    with_skill = build_instructions("gpt55_skill_v2", "example skill")
    if "example skill" in no_skill or "example skill" not in with_skill:
        raise SystemExit("Condition-isolation regression failed.")

    raw = json.dumps(
        {
            "root_cause": "synthetic fault",
            "evidence": ["cf0 increased"],
            "corrective_actions": ["stop and inspect"],
        }
    )
    validation = evaluator.validate_deterministically(raw)
    if not validation.get("valid"):
        raise SystemExit(f"Deterministic validator regression failed: {validation}")

    print("GPT-5.5 neutral-context control offline self-test PASSED.")


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("FACTORYBENCH_DATASET", str(DEFAULT_DATASET))),
    )
    parser.add_argument("--split", default=os.getenv("EVAL_SPLIT", "final_test"))
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=5,
        help="Expected held-out case count; use 0 only to disable the count check.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPT-5.5 no-Skill versus Skill-v2 FactoryBench control."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test = subparsers.add_parser("self-test", help="Offline regression tests.")
    self_test.set_defaults(func=command_self_test)

    context = subparsers.add_parser(
        "check-context", help="Audit neutral prompt context; no API calls."
    )
    add_dataset_arguments(context)
    context.add_argument("--output", type=Path, default=DEFAULT_CONTEXT_AUDIT)
    context.set_defaults(func=command_check_context)

    run = subparsers.add_parser(
        "run", help="Run GPT-5.5 with no Skill and Skill v2."
    )
    add_dataset_arguments(run)
    run.add_argument("--model", default=os.getenv("TARGET_MODEL", "gpt-5.5"))
    run.add_argument("--skill", type=Path, default=DEFAULT_SKILL)
    run.add_argument("--context-audit", type=Path, default=DEFAULT_CONTEXT_AUDIT)
    run.add_argument("--output", type=Path, default=DEFAULT_GENERATION)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--retries", type=int, default=3)
    run.add_argument("--overwrite", action="store_true")
    run.set_defaults(func=command_run)

    judge = subparsers.add_parser(
        "judge", help="Apply verified FactoryBench L4 semantic Judge."
    )
    judge.add_argument("--input", type=Path, default=DEFAULT_GENERATION)
    judge.add_argument("--output", type=Path, default=DEFAULT_JUDGED)
    judge.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        help=(
            "Judge model. Repeat with three distinct models for a formal median. "
            "Default: JUDGE_MODELS or gpt-5.5."
        ),
    )
    judge.add_argument("--overwrite", action="store_true")
    judge.set_defaults(func=command_judge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "expected_cases") and args.expected_cases < 0:
        raise SystemExit("--expected-cases must be 0 or a positive integer.")
    if hasattr(args, "retries") and args.retries < 1:
        raise SystemExit("--retries must be at least 1.")
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Saved records can be resumed.", file=sys.stderr)
        raise SystemExit(130)