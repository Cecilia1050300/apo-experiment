from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

try:
    import factorybench_gpt55_context_control as base
except ImportError as exc:
    raise SystemExit(
        "factorybench_gpt55_context_control.py must be in the same directory."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

DEFAULT_DATASET = SCRIPT_DIR / "data" / "factorybench_l4_subset.json"
DEFAULT_SKILL = SCRIPT_DIR / "skills" / "skill_v2.md"
DEFAULT_GPT55_JUDGED = (
    RESULTS_DIR / "gpt55_neutral_context_control_judged_gpt-5.5.json"
)
DEFAULT_MINI_GENERATION = RESULTS_DIR / "mini_neutral_context_control.json"
DEFAULT_MINI_JUDGED = RESULTS_DIR / "mini_neutral_context_control_judged_gpt-5.5.json"
DEFAULT_MINI_CONTEXT_AUDIT = RESULTS_DIR / "mini_neutral_context_audit.json"
DEFAULT_AUDIT_JSON = RESULTS_DIR / "failure_audit_source.json"
DEFAULT_AUDIT_BLIND = RESULTS_DIR / "failure_audit_blind.csv"
DEFAULT_AUDIT_ADJUDICATION = RESULTS_DIR / "failure_audit_adjudication.csv"
DEFAULT_AUDIT_SUMMARY = RESULTS_DIR / "failure_audit_summary.json"
DEFAULT_COMPARISON_JSON = RESULTS_DIR / "mini_vs_gpt55_comparison.json"
DEFAULT_COMPARISON_CSV = RESULTS_DIR / "mini_vs_gpt55_per_case.csv"
DEFAULT_COMPARISON_MD = RESULTS_DIR / "mini_vs_gpt55_summary.md"

MINI_CONDITIONS = ("mini_no_skill", "mini_skill_v2")
GPT55_CONDITIONS = ("gpt55_no_skill", "gpt55_skill_v2")
ROLE_TO_MINI = {"no_skill": "mini_no_skill", "skill_v2": "mini_skill_v2"}
ROLE_TO_GPT55 = {"no_skill": "gpt55_no_skill", "skill_v2": "gpt55_skill_v2"}
CONDITION_TO_ROLE = {
    **{value: key for key, value in ROLE_TO_MINI.items()},
    **{value: key for key, value in ROLE_TO_GPT55.items()},
}
INSTRUCTION_SOURCE_CONDITION = {
    "mini_no_skill": "gpt55_no_skill",
    "mini_skill_v2": "gpt55_skill_v2",
}

FAILURE_CATEGORIES = (
    "DOMAIN_KNOWLEDGE_MISSING",
    "SIGNAL_INTERPRETATION_ERROR",
    "SYMPTOM_AS_ROOT_CAUSE",
    "TASK_PHASE_ERROR",
    "ROOT_CAUSE_ERROR",
    "REMEDIATION_ERROR",
    "INSUFFICIENT_EVIDENCE",
    "EVALUATOR_OR_REFERENCE_REVIEW",
    "NO_FAILURE",
)

INTERVENTIONS = (
    "DOMAIN_KNOWLEDGE_OR_RAG",
    "SIGNAL_ANALYSIS_TOOL",
    "SKILL_REVISION",
    "INPUT_DATA_OR_EVALUATOR_REVIEW",
    "NO_ACTION",
)


def require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return value


def score_of(record: dict[str, Any]) -> float | None:
    value = record.get("median_score")
    return None if value is None else float(value)


def condition_records(
    records: Iterable[dict[str, Any]],
    expected_conditions: tuple[str, str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return case_id -> condition -> record and reject duplicate keys."""
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        condition = str(record.get("condition"))
        if condition not in expected_conditions:
            continue
        case_id = str(record.get("id"))
        if condition in result[case_id]:
            raise ValueError(f"Duplicate record: {condition} / {case_id}")
        result[case_id][condition] = record
    return dict(result)


def summarize_generation(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in MINI_CONDITIONS:
        rows = [row for row in records if row.get("condition") == condition]
        summary[condition] = {
            "n": len(rows),
            "successful": sum(row.get("status") == "ok" for row in rows),
            "format_valid": sum(
                bool(row.get("deterministic_validation", {}).get("valid"))
                for row in rows
            ),
        }
    return summary


def same_inputs_across_conditions(records: list[dict[str, Any]]) -> bool:
    grouped = condition_records(records, MINI_CONDITIONS)
    if not grouped:
        return False
    for rows in grouped.values():
        if set(rows) != set(MINI_CONDITIONS):
            return False
        digests = {str(row.get("target_input_sha256")) for row in rows.values()}
        if len(digests) != 1:
            return False
    return True


def build_mini_configuration(
    args: argparse.Namespace,
    skill_text: str,
    context_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": base.sha256_file(args.dataset),
        "evaluation_split": args.split,
        "expected_cases": args.expected_cases,
        "target_model": args.model,
        "conditions": list(MINI_CONDITIONS),
        "skill_path": str(args.skill.resolve()),
        "skill_sha256": base.sha256_text(skill_text),
        "skill_text_snapshot": skill_text,
        "seed": args.seed,
        "retries": args.retries,
        "output_contract": sorted(base.REQUIRED_TARGET_KEYS),
        "domain_rag": None,
        "fault_catalog": None,
        "signal_tool": None,
        "held_out_gold_supplied_to_target": False,
        "context_audit_all_cases_valid": context_audit["all_cases_valid"],
        "context_audit_records": context_audit["records"],
        "prompt_implementation": "factorybench_gpt55_context_control.py",
        "prompt_parity_note": (
            "The exact same build_instructions branches used by the GPT-5.5 "
            "control are reused for gpt-4o-mini."
        ),
    }


def initialize_or_resume_mini(
    path: Path,
    configuration: dict[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        output = base.load_json(path)
        if output.get("configuration") != configuration:
            raise SystemExit(
                "Existing mini output has a different configuration. Use a "
                "different --output or explicitly pass --overwrite."
            )
        print(f"Resuming: {path}")
        return output
    output = {
        "status": "running",
        "experiment_type": "factorybench_mini_neutral_context_skill_control",
        "configuration": configuration,
        "records": [],
        "generation_summary": {},
        "same_target_input_across_conditions": None,
    }
    base.atomic_save(path, output)
    return output


def command_run_mini(args: argparse.Namespace) -> None:
    evaluator = base.import_verified_evaluator()
    cases = base.load_cases(args.dataset, args.split, args.expected_cases)
    if not args.skill.exists():
        raise FileNotFoundError(f"Skill not found: {args.skill}")
    skill_text = args.skill.read_text(encoding="utf-8")
    if not skill_text.strip():
        raise ValueError("Skill v2 file is empty.")

    context_audit = base.audit_contexts(args.dataset, args.split, args.expected_cases)
    base.atomic_save(args.context_audit, context_audit)
    base.print_context_audit(context_audit)
    if not context_audit["all_cases_valid"]:
        raise SystemExit(
            f"Context audit failed. Inspect {args.context_audit}; no API calls were made."
        )

    configuration = build_mini_configuration(args, skill_text, context_audit)
    output = initialize_or_resume_mini(args.output, configuration, args.overwrite)
    client = base.require_api_client()

    existing_by_key = {
        (str(row.get("condition")), str(row.get("id"))): row
        for row in output.get("records", [])
    }
    jobs = [
        (condition, case_index, case)
        for condition in MINI_CONDITIONS
        for case_index, case in enumerate(cases, start=1)
    ]
    random.Random(args.seed).shuffle(jobs)

    for job_index, (condition, case_index, case) in enumerate(jobs, start=1):
        case_id = str(case.get("id", f"case_{case_index}"))
        key = (condition, case_id)
        record = existing_by_key.get(key)

        if record is not None and record.get("status") == "ok":
            print(f"[{job_index}/{len(jobs)}] {condition} / {case_id}: already complete")
            continue
        if record is not None and record.get("status") == "raw_response_saved":
            print(f"[{job_index}/{len(jobs)}] {condition} / {case_id}: parsing saved raw")
            try:
                base.ensure_saved_raw_is_parsed(record, evaluator)
            except Exception as exc:  # noqa: BLE001
                record["status"] = "parser_error"
                record["parser_error"] = repr(exc)
            base.atomic_save(args.output, output)
            continue

        source_condition = INSTRUCTION_SOURCE_CONDITION[condition]
        instructions = base.build_instructions(source_condition, skill_text)
        prompt = str(case["input"])
        initial_record: dict[str, Any] = {
            "condition": condition,
            "condition_role": CONDITION_TO_ROLE[condition],
            "id": case_id,
            "status": "calling_model",
            "target_model": args.model,
            "skill_version": "skill_v2" if condition == "mini_skill_v2" else None,
            "rag_version": None,
            "tool_version": None,
            "retrieved_documents": [],
            "tool_output": {},
            "target_input": prompt,
            "target_input_sha256": base.sha256_text(prompt),
            "expected_root_cause": case["root_cause"],
            "reference_answer": case["reference_answer"],
            "metadata": case.get("metadata"),
            "instructions_sha256": base.sha256_text(instructions),
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
        base.atomic_save(args.output, output)

        print(f"[{job_index}/{len(jobs)}] {condition} / {case_id} / {args.model}")
        started = time.time()
        try:
            response = base.call_target_model(
                client, args.model, instructions, prompt, args.retries
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
            # Preserve immutable raw output before running any parser.
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
        output["generation_summary"] = summarize_generation(output["records"])
        base.atomic_save(args.output, output)

    output["status"] = "completed"
    output["generation_summary"] = summarize_generation(output["records"])
    output["same_target_input_across_conditions"] = same_inputs_across_conditions(
        output["records"]
    )
    base.atomic_save(args.output, output)
    print(f"\nSaved: {args.output}")
    print(json.dumps(output["generation_summary"], ensure_ascii=False, indent=2))
    print(
        "Same target input across mini conditions:",
        output["same_target_input_across_conditions"],
    )


def summarize_judged_mini(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for condition in MINI_CONDITIONS:
        rows = [row for row in records if row.get("condition") == condition]
        scored = [row for row in rows if score_of(row) is not None]
        scores = [float(row["median_score"]) for row in scored]
        summary[condition] = {
            "n_total": len(rows),
            "n_scored": len(scored),
            "mean_l4_score": mean(scores) if scores else None,
            "zero_count": sum(value == 0.0 for value in scores),
            "partial_count": sum(value == 0.5 for value in scores),
            "full_count": sum(value == 1.0 for value in scores),
            "format_valid_count": sum(
                bool(row.get("deterministic_validation", {}).get("valid"))
                for row in rows
            ),
        }
    return summary


def paired_skill_effect(
    records: list[dict[str, Any]],
    conditions: tuple[str, str],
) -> dict[str, Any]:
    grouped = condition_records(records, conditions)
    rows: list[dict[str, Any]] = []
    effects: list[float] = []
    for case_id, by_condition in sorted(grouped.items()):
        if set(by_condition) != set(conditions):
            continue
        first = score_of(by_condition[conditions[0]])
        second = score_of(by_condition[conditions[1]])
        if first is None or second is None:
            continue
        effect = second - first
        effects.append(effect)
        rows.append(
            {
                "id": case_id,
                conditions[0]: first,
                conditions[1]: second,
                "skill_effect": effect,
            }
        )
    return {
        "n_complete_pairs": len(rows),
        "mean_paired_skill_effect": mean(effects) if effects else None,
        "improved_cases": sum(value > 0 for value in effects),
        "unchanged_cases": sum(value == 0 for value in effects),
        "worse_cases": sum(value < 0 for value in effects),
        "per_case": rows,
        "interpretation_boundary": (
            "This is a descriptive paired comparison over five held-out cases."
        ),
    }


def judge_model_names(args: argparse.Namespace) -> list[str]:
    models = args.judge_models or [
        part.strip()
        for part in os.getenv("JUDGE_MODELS", "gpt-5.5").split(",")
        if part.strip()
    ]
    if not models:
        raise SystemExit("At least one Judge model is required.")
    if len(models) != len(set(models)):
        raise SystemExit(
            "Judge model names must be distinct; repeated calls to one model are "
            "not independent Judges."
        )
    return models


def command_judge_mini(args: argparse.Namespace) -> None:
    evaluator = base.import_verified_evaluator()
    models = judge_model_names(args)
    source = base.load_json(args.input)
    if source.get("status") != "completed":
        raise SystemExit("Mini generation is incomplete; finish or resume run-mini.")
    if source.get("same_target_input_across_conditions") is not True:
        raise SystemExit("Mini generation did not verify identical inputs by case.")
    source_records = require_list(source.get("records"), "generation records")
    if not source_records:
        raise ValueError("Mini generation contains no records.")

    source_digest = base.sha256_file(args.input)
    if args.output.exists() and not args.overwrite:
        output = base.load_json(args.output)
        if (
            output.get("source_sha256") != source_digest
            or output.get("judge_models") != models
        ):
            raise SystemExit(
                "Existing judged output has a different source or Judge set. "
                "Use another --output or pass --overwrite."
            )
        print(f"Resuming: {args.output}")
    else:
        output = {
            "status": "running",
            "experiment_type": "factorybench_mini_context_control_judged",
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
        base.atomic_save(args.output, output)

    existing_by_key = {
        (str(row.get("condition")), str(row.get("id"))): row
        for row in output.get("records", [])
    }
    client = base.require_api_client()

    for index, source_record in enumerate(source_records, start=1):
        key = (str(source_record.get("condition")), str(source_record.get("id")))
        existing = existing_by_key.get(key)
        if existing is not None:
            vote_models = {
                vote.get("judge_model") for vote in existing.get("judge_votes", [])
            }
            complete = existing.get("judge_status") == "skipped_unusable_answer" or (
                existing.get("judge_status") == "ok" and vote_models == set(models)
            )
            if complete:
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
            record.get("raw_response_sha256_at_save") == base.sha256_text(raw)
        )
        if not record["raw_response_still_matches_saved_digest"]:
            record["judge_status"] = "skipped_raw_digest_mismatch"
            record["median_score"] = None
        elif record.get("status") != "ok":
            record["judge_status"] = "skipped_target_or_parser_error"
            record["median_score"] = None
        else:
            base.score_record(client, evaluator, record, models)
        output["summary"] = summarize_judged_mini(output["records"])
        output["paired_skill_effect"] = paired_skill_effect(
            output["records"], MINI_CONDITIONS
        )
        base.atomic_save(args.output, output)

    output["status"] = "completed"
    output["summary"] = summarize_judged_mini(output["records"])
    output["paired_skill_effect"] = paired_skill_effect(
        output["records"], MINI_CONDITIONS
    )
    base.atomic_save(args.output, output)
    print(f"\nSaved: {args.output}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(output["paired_skill_effect"], ensure_ascii=False, indent=2))


def judge_details(record: dict[str, Any]) -> dict[str, Any]:
    votes = record.get("judge_votes")
    if not isinstance(votes, list):
        votes = []
    parsed_votes = [
        vote.get("parsed")
        for vote in votes
        if isinstance(vote, dict) and isinstance(vote.get("parsed"), dict)
    ]
    return {
        "judge_models": [
            str(vote.get("judge_model"))
            for vote in votes
            if isinstance(vote, dict)
        ],
        "root_cause_correct_votes": [
            vote.get("root_cause_correct") for vote in parsed_votes
        ],
        "protocol_correct_votes": [vote.get("protocol_correct") for vote in parsed_votes],
        "reasons": [str(vote.get("reason", "")) for vote in parsed_votes],
    }


def select_failure_audit_cases(
    judged: dict[str, Any],
) -> list[dict[str, Any]]:
    records = require_list(judged.get("records"), "judged records")
    grouped = condition_records(records, GPT55_CONDITIONS)
    selected: list[dict[str, Any]] = []
    for case_id, rows in sorted(grouped.items()):
        if set(rows) != set(GPT55_CONDITIONS):
            continue
        no_skill = rows["gpt55_no_skill"]
        with_skill = rows["gpt55_skill_v2"]
        no_score = score_of(no_skill)
        skill_score = score_of(with_skill)
        if no_score is None or skill_score is None:
            continue
        stable_zero = no_score == 0.0 and skill_score == 0.0
        changed = no_score != skill_score
        if not (stable_zero or changed):
            continue
        selected.append(
            {
                "id": case_id,
                "selection_reason": (
                    "STABLE_ZERO_BOTH_CONDITIONS" if stable_zero else "SKILL_SCORE_CHANGED"
                ),
                "target_input": str(no_skill.get("target_input", "")),
                "target_input_sha256": str(no_skill.get("target_input_sha256", "")),
                "no_skill_raw_response": str(no_skill.get("raw_response", "")),
                "no_skill_parsed_response": no_skill.get("parsed_response"),
                "skill_v2_raw_response": str(with_skill.get("raw_response", "")),
                "skill_v2_parsed_response": with_skill.get("parsed_response"),
                "expected_root_cause": no_skill.get("expected_root_cause"),
                "reference_answer": no_skill.get("reference_answer"),
                "no_skill_score": no_score,
                "skill_v2_score": skill_score,
                "skill_effect": skill_score - no_score,
                "no_skill_judge": judge_details(no_skill),
                "skill_v2_judge": judge_details(with_skill),
                "deterministic_format_valid": {
                    "no_skill": bool(
                        no_skill.get("deterministic_validation", {}).get("valid")
                    ),
                    "skill_v2": bool(
                        with_skill.get("deterministic_validation", {}).get("valid")
                    ),
                },
            }
        )
    return selected


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def command_export_failure_audit(args: argparse.Namespace) -> None:
    judged = base.load_json(args.input)
    if judged.get("status") != "completed":
        raise SystemExit("GPT-5.5 judged result must be completed.")
    selected = select_failure_audit_cases(judged)
    if not selected:
        raise SystemExit("No stable-zero or Skill-changed complete pairs were found.")

    source = {
        "audit_type": "factorybench_failure_audit_source",
        "source_file": str(args.input.resolve()),
        "source_sha256": base.sha256_file(args.input),
        "selection_rule": (
            "Select complete GPT-5.5 pairs where both conditions scored zero or "
            "where Skill v2 changed the score."
        ),
        "selected_case_count": len(selected),
        "allowed_failure_categories": list(FAILURE_CATEGORIES),
        "allowed_recommended_interventions": list(INTERVENTIONS),
        "records": selected,
        "review_boundary": (
            "Scores alone cannot establish a domain-knowledge gap. Complete the "
            "blind review before viewing gold-aware adjudication fields."
        ),
    }
    base.atomic_save(args.json_output, source)

    blind_fields = [
        "id",
        "selection_reason",
        "target_input",
        "no_skill_raw_response",
        "skill_v2_raw_response",
        "blind_review_status",
        "blind_signal_interpretation_notes",
        "blind_task_phase_notes",
        "blind_symptom_vs_root_notes",
        "blind_difference_between_conditions",
        "blind_reviewer_notes",
    ]
    blind_rows = []
    for row in selected:
        blind_rows.append(
            {
                key: row.get(key, "")
                for key in blind_fields
            }
            | {
                "blind_review_status": "UNREVIEWED",
                "blind_signal_interpretation_notes": "",
                "blind_task_phase_notes": "",
                "blind_symptom_vs_root_notes": "",
                "blind_difference_between_conditions": "",
                "blind_reviewer_notes": "",
            }
        )
    write_csv(args.blind_output, blind_fields, blind_rows)

    adjudication_fields = [
        "id",
        "selection_reason",
        "expected_root_cause",
        "reference_answer",
        "no_skill_score",
        "skill_v2_score",
        "skill_effect",
        "no_skill_raw_response",
        "skill_v2_raw_response",
        "no_skill_judge_reason",
        "skill_v2_judge_reason",
        "adjudication_status",
        "primary_failure_category",
        "secondary_failure_category",
        "evidence_in_input_sufficient",
        "missing_knowledge_or_rule",
        "required_knowledge_source",
        "recommended_intervention",
        "reviewer_notes",
    ]
    adjudication_rows = []
    for row in selected:
        adjudication_rows.append(
            {
                "id": row["id"],
                "selection_reason": row["selection_reason"],
                "expected_root_cause": base.value_to_text(row["expected_root_cause"]),
                "reference_answer": base.value_to_text(row["reference_answer"]),
                "no_skill_score": row["no_skill_score"],
                "skill_v2_score": row["skill_v2_score"],
                "skill_effect": row["skill_effect"],
                "no_skill_raw_response": row["no_skill_raw_response"],
                "skill_v2_raw_response": row["skill_v2_raw_response"],
                "no_skill_judge_reason": " | ".join(
                    row["no_skill_judge"]["reasons"]
                ),
                "skill_v2_judge_reason": " | ".join(
                    row["skill_v2_judge"]["reasons"]
                ),
                "adjudication_status": "UNREVIEWED",
                "primary_failure_category": "",
                "secondary_failure_category": "",
                "evidence_in_input_sufficient": "",
                "missing_knowledge_or_rule": "",
                "required_knowledge_source": "",
                "recommended_intervention": "",
                "reviewer_notes": "",
            }
        )
    write_csv(args.adjudication_output, adjudication_fields, adjudication_rows)

    print(f"Selected audit cases: {len(selected)}")
    for row in selected:
        print(
            f"  {row['id']}: {row['selection_reason']}; "
            f"{row['no_skill_score']} -> {row['skill_v2_score']}"
        )
    print(f"Saved source: {args.json_output}")
    print(f"Saved blind review: {args.blind_output}")
    print(f"Saved adjudication: {args.adjudication_output}")
    print("Complete the blind CSV first; then open the adjudication CSV.")


def load_completed_adjudication(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Adjudication CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Adjudication CSV is empty.")

    errors: list[str] = []
    for index, row in enumerate(rows, start=2):
        status = row.get("adjudication_status", "").strip().upper()
        if status != "COMPLETED":
            errors.append(f"line {index}: adjudication_status must be COMPLETED")
        primary = row.get("primary_failure_category", "").strip().upper()
        if primary not in FAILURE_CATEGORIES:
            errors.append(
                f"line {index}: invalid primary_failure_category {primary!r}"
            )
        secondary = row.get("secondary_failure_category", "").strip().upper()
        if secondary and secondary not in FAILURE_CATEGORIES:
            errors.append(
                f"line {index}: invalid secondary_failure_category {secondary!r}"
            )
        sufficient = row.get("evidence_in_input_sufficient", "").strip().upper()
        if sufficient not in {"YES", "NO", "UNCLEAR"}:
            errors.append(
                f"line {index}: evidence_in_input_sufficient must be YES/NO/UNCLEAR"
            )
        intervention = row.get("recommended_intervention", "").strip().upper()
        if intervention not in INTERVENTIONS:
            errors.append(
                f"line {index}: invalid recommended_intervention {intervention!r}"
            )
    return rows, errors


def command_summarize_audit(args: argparse.Namespace) -> None:
    rows, errors = load_completed_adjudication(args.input)
    if errors:
        print("Audit is incomplete or invalid:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    primary_counts = Counter(
        row["primary_failure_category"].strip().upper() for row in rows
    )
    intervention_counts = Counter(
        row["recommended_intervention"].strip().upper() for row in rows
    )
    evidence_counts = Counter(
        row["evidence_in_input_sufficient"].strip().upper() for row in rows
    )
    ranked = intervention_counts.most_common()
    top_count = ranked[0][1]
    top_interventions = sorted(name for name, count in ranked if count == top_count)
    decision = top_interventions[0] if len(top_interventions) == 1 else "MIXED_OR_TIED"

    summary = {
        "audit_status": "completed",
        "source_file": str(args.input.resolve()),
        "source_sha256": base.sha256_file(args.input),
        "case_count": len(rows),
        "primary_failure_category_counts": dict(sorted(primary_counts.items())),
        "recommended_intervention_counts": dict(sorted(intervention_counts.items())),
        "evidence_in_input_sufficient_counts": dict(sorted(evidence_counts.items())),
        "descriptive_next_experiment": decision,
        "decision_basis": (
            "Plurality of reviewer-completed recommended_intervention labels; "
            "ties remain unresolved."
        ),
        "boundary": (
            "This summary reports human adjudication. It does not infer a "
            "knowledge gap from model scores alone."
        ),
    }
    base.atomic_save(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


def verify_judged_file(payload: dict[str, Any], name: str) -> list[dict[str, Any]]:
    if payload.get("status") != "completed":
        raise ValueError(f"{name} judged file is not completed.")
    records = require_list(payload.get("records"), f"{name} records")
    if not records:
        raise ValueError(f"{name} judged file contains no records.")
    return records


def compare_judged_results(
    mini: dict[str, Any],
    gpt55: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mini_records = verify_judged_file(mini, "mini")
    gpt_records = verify_judged_file(gpt55, "gpt55")
    mini_grouped = condition_records(mini_records, MINI_CONDITIONS)
    gpt_grouped = condition_records(gpt_records, GPT55_CONDITIONS)

    mini_ids = set(mini_grouped)
    gpt_ids = set(gpt_grouped)
    errors: list[str] = []
    if mini_ids != gpt_ids:
        errors.append(
            f"case id mismatch: mini_only={sorted(mini_ids-gpt_ids)}, "
            f"gpt55_only={sorted(gpt_ids-mini_ids)}"
        )

    mini_judges = mini.get("judge_models")
    gpt_judges = gpt55.get("judge_models")
    if mini_judges != gpt_judges:
        errors.append(f"Judge mismatch: mini={mini_judges}, gpt55={gpt_judges}")

    per_case: list[dict[str, Any]] = []
    for case_id in sorted(mini_ids & gpt_ids):
        mrows = mini_grouped[case_id]
        grows = gpt_grouped[case_id]
        if set(mrows) != set(MINI_CONDITIONS):
            errors.append(f"{case_id}: incomplete mini condition pair")
            continue
        if set(grows) != set(GPT55_CONDITIONS):
            errors.append(f"{case_id}: incomplete GPT-5.5 condition pair")
            continue

        scores: dict[str, float] = {}
        for role in ("no_skill", "skill_v2"):
            mini_row = mrows[ROLE_TO_MINI[role]]
            gpt_row = grows[ROLE_TO_GPT55[role]]
            mini_score = score_of(mini_row)
            gpt_score = score_of(gpt_row)
            if mini_score is None or gpt_score is None:
                errors.append(f"{case_id}/{role}: missing semantic score")
                continue
            scores[f"mini_{role}"] = mini_score
            scores[f"gpt55_{role}"] = gpt_score

            if mini_row.get("target_input_sha256") != gpt_row.get("target_input_sha256"):
                errors.append(f"{case_id}/{role}: target input hash mismatch")
            if mini_row.get("instructions_sha256") != gpt_row.get("instructions_sha256"):
                errors.append(f"{case_id}/{role}: instruction hash mismatch")
            if base.value_to_text(mini_row.get("reference_answer")) != base.value_to_text(
                gpt_row.get("reference_answer")
            ):
                errors.append(f"{case_id}/{role}: reference answer mismatch")
            if base.value_to_text(mini_row.get("expected_root_cause")) != base.value_to_text(
                gpt_row.get("expected_root_cause")
            ):
                errors.append(f"{case_id}/{role}: expected root-cause mismatch")

        if len(scores) != 4:
            continue
        per_case.append(
            {
                "id": case_id,
                **scores,
                "model_effect_no_skill": (
                    scores["gpt55_no_skill"] - scores["mini_no_skill"]
                ),
                "model_effect_skill_v2": (
                    scores["gpt55_skill_v2"] - scores["mini_skill_v2"]
                ),
                "skill_effect_mini": (
                    scores["mini_skill_v2"] - scores["mini_no_skill"]
                ),
                "skill_effect_gpt55": (
                    scores["gpt55_skill_v2"] - scores["gpt55_no_skill"]
                ),
            }
        )

    if errors:
        raise ValueError("Experimental parity check failed:\n- " + "\n- ".join(errors))
    if not per_case:
        raise ValueError("No complete comparable cases.")

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in per_case]

    means = {
        name: mean(values(name))
        for name in (
            "mini_no_skill",
            "mini_skill_v2",
            "gpt55_no_skill",
            "gpt55_skill_v2",
        )
    }
    model_no = values("model_effect_no_skill")
    model_skill = values("model_effect_skill_v2")
    skill_mini = values("skill_effect_mini")
    skill_gpt = values("skill_effect_gpt55")
    interaction = [gpt - mini_value for gpt, mini_value in zip(skill_gpt, skill_mini)]

    summary = {
        "comparison_type": "descriptive_paired_2x2_model_by_skill_control",
        "n_complete_cases": len(per_case),
        "judge_models": mini_judges,
        "experimental_parity": {
            "same_case_ids": True,
            "same_target_input_hash_by_case_and_role": True,
            "same_instruction_hash_by_case_and_role": True,
            "same_reference_and_root_cause": True,
            "same_judge_models": True,
        },
        "condition_mean_l4_scores": means,
        "model_effect_gpt55_minus_mini": {
            "without_skill": mean(model_no),
            "with_skill_v2": mean(model_skill),
            "without_skill_improved_cases": sum(value > 0 for value in model_no),
            "with_skill_improved_cases": sum(value > 0 for value in model_skill),
        },
        "skill_effect_within_model": {
            "mini": mean(skill_mini),
            "gpt55": mean(skill_gpt),
        },
        "difference_in_differences": mean(interaction),
        "interpretation": {
            "positive_model_effect": (
                "Consistent with greater target-model capability under the same prompt."
            ),
            "positive_skill_effect": (
                "Consistent with a Skill contribution for these held-out cases."
            ),
            "difference_in_differences": (
                "GPT-5.5 Skill effect minus mini Skill effect; descriptive only."
            ),
        },
        "limitations": [
            "Only five held-out cases are compared.",
            "A single GPT-5.5 Judge is preliminary unless the files contain three distinct Judges.",
            "GPT-5.5 judging GPT-5.5 may introduce same-model evaluation bias.",
            "Scores alone do not identify whether remaining failures require RAG, a signal tool, or Skill revision.",
        ],
    }
    return summary, per_case


def render_comparison_markdown(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    means = summary["condition_mean_l4_scores"]
    model_effect = summary["model_effect_gpt55_minus_mini"]
    skill_effect = summary["skill_effect_within_model"]
    lines = [
        "# FactoryBench mini vs GPT-5.5 comparison",
        "",
        "## Condition means",
        "",
        "| Target model | No Skill | Skill v2 | Skill effect |",
        "|---|---:|---:|---:|",
        (
            f"| gpt-4o-mini | {means['mini_no_skill']:.3f} | "
            f"{means['mini_skill_v2']:.3f} | {skill_effect['mini']:+.3f} |"
        ),
        (
            f"| GPT-5.5 | {means['gpt55_no_skill']:.3f} | "
            f"{means['gpt55_skill_v2']:.3f} | {skill_effect['gpt55']:+.3f} |"
        ),
        "",
        "## Paired effects",
        "",
        (
            f"- GPT-5.5 minus mini without Skill: "
            f"{model_effect['without_skill']:+.3f}"
        ),
        (
            f"- GPT-5.5 minus mini with Skill v2: "
            f"{model_effect['with_skill_v2']:+.3f}"
        ),
        (
            f"- Difference-in-differences: "
            f"{summary['difference_in_differences']:+.3f}"
        ),
        "",
        "## Per-case scores",
        "",
        "| Case | mini no Skill | mini Skill | GPT-5.5 no Skill | GPT-5.5 Skill |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['mini_no_skill']:.1f} | "
            f"{row['mini_skill_v2']:.1f} | {row['gpt55_no_skill']:.1f} | "
            f"{row['gpt55_skill_v2']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is a descriptive five-case control. Complete the failure audit "
            "before choosing Domain Knowledge/RAG, a Signal Tool, or another Skill revision.",
            "",
        ]
    )
    return "\n".join(lines)


def command_compare_models(args: argparse.Namespace) -> None:
    mini = base.load_json(args.mini)
    gpt55 = base.load_json(args.gpt55)
    summary, rows = compare_judged_results(mini, gpt55)
    summary["mini_source_file"] = str(args.mini.resolve())
    summary["mini_source_sha256"] = base.sha256_file(args.mini)
    summary["gpt55_source_file"] = str(args.gpt55.resolve())
    summary["gpt55_source_sha256"] = base.sha256_file(args.gpt55)

    base.atomic_save(args.json_output, {"summary": summary, "per_case": rows})
    csv_fields = [
        "id",
        "mini_no_skill",
        "mini_skill_v2",
        "gpt55_no_skill",
        "gpt55_skill_v2",
        "model_effect_no_skill",
        "model_effect_skill_v2",
        "skill_effect_mini",
        "skill_effect_gpt55",
    ]
    write_csv(args.csv_output, csv_fields, rows)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.md_output.with_name(args.md_output.name + ".tmp")
    temporary.write_text(render_comparison_markdown(summary, rows), encoding="utf-8")
    temporary.replace(args.md_output)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved JSON: {args.json_output}")
    print(f"Saved CSV: {args.csv_output}")
    print(f"Saved Markdown: {args.md_output}")


def synthetic_judged(
    model_prefix: str,
    conditions: tuple[str, str],
    scores: list[tuple[float, float]],
    judge_models: list[str] | None = None,
) -> dict[str, Any]:
    records = []
    for index, pair in enumerate(scores):
        case_id = f"case_{index}"
        for condition, score in zip(conditions, pair):
            role = CONDITION_TO_ROLE[condition]
            records.append(
                {
                    "condition": condition,
                    "id": case_id,
                    "target_model": model_prefix,
                    "target_input": f"input {case_id}",
                    "target_input_sha256": f"input_hash_{case_id}",
                    "instructions_sha256": f"instructions_{role}",
                    "expected_root_cause": f"root {case_id}",
                    "reference_answer": f"reference {case_id}",
                    "raw_response": json.dumps(
                        {
                            "root_cause": f"answer {case_id}",
                            "evidence": ["signal"],
                            "corrective_actions": ["inspect"],
                        }
                    ),
                    "parsed_response": {"root_cause": f"answer {case_id}"},
                    "deterministic_validation": {"valid": True},
                    "median_score": score,
                    "judge_votes": [
                        {
                            "judge_model": "judge",
                            "parsed": {
                                "score": score,
                                "root_cause_correct": score > 0,
                                "protocol_correct": score == 1,
                                "reason": "fixture",
                            },
                        }
                    ],
                }
            )
    return {
        "status": "completed",
        "judge_models": judge_models or ["judge"],
        "records": records,
    }


def command_self_test(_: argparse.Namespace) -> None:
    evaluator = base.import_verified_evaluator()
    parser_results = evaluator.run_parser_regression_tests()
    if not parser_results or not all(row.get("passed") for row in parser_results):
        raise SystemExit("Verified evaluator parser regression failed.")

    gpt_scores = [(0, 0), (0, 0), (0, 1), (0, 0), (1, 1)]
    mini_scores = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    gpt = synthetic_judged("gpt-5.5", GPT55_CONDITIONS, gpt_scores)
    mini = synthetic_judged("gpt-4o-mini", MINI_CONDITIONS, mini_scores)

    selected = select_failure_audit_cases(gpt)
    if len(selected) != 4:
        raise SystemExit(f"Failure-audit selection regression: {len(selected)} != 4")
    selection_counts = Counter(row["selection_reason"] for row in selected)
    if selection_counts != {
        "STABLE_ZERO_BOTH_CONDITIONS": 3,
        "SKILL_SCORE_CHANGED": 1,
    }:
        raise SystemExit(f"Failure-audit reason regression: {selection_counts}")

    summary, rows = compare_judged_results(mini, gpt)
    if len(rows) != 5:
        raise SystemExit("Model-comparison pair regression failed.")
    if summary["condition_mean_l4_scores"]["gpt55_no_skill"] != 0.2:
        raise SystemExit("GPT-5.5 no-Skill mean regression failed.")
    if summary["condition_mean_l4_scores"]["gpt55_skill_v2"] != 0.4:
        raise SystemExit("GPT-5.5 Skill mean regression failed.")
    if summary["model_effect_gpt55_minus_mini"]["without_skill"] != 0.2:
        raise SystemExit("No-Skill model-effect regression failed.")
    if summary["model_effect_gpt55_minus_mini"]["with_skill_v2"] != 0.4:
        raise SystemExit("Skill model-effect regression failed.")
    if summary["difference_in_differences"] != 0.2:
        raise SystemExit("Difference-in-differences regression failed.")

    raw = json.dumps(
        {
            "root_cause": "synthetic fault",
            "evidence": ["signal changed"],
            "corrective_actions": ["stop, inspect, and verify"],
        }
    )
    if not evaluator.validate_deterministically(raw).get("valid"):
        raise SystemExit("Deterministic validator regression failed.")

    print("FactoryBench model/failure follow-up offline self-test PASSED.")


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("FACTORYBENCH_DATASET", str(DEFAULT_DATASET))),
    )
    parser.add_argument("--split", default=os.getenv("EVAL_SPLIT", "final_test"))
    parser.add_argument("--expected-cases", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FactoryBench failure audit and mini-vs-GPT-5.5 follow-up."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test = subparsers.add_parser("self-test", help="Offline regression tests.")
    self_test.set_defaults(func=command_self_test)

    export = subparsers.add_parser(
        "export-failure-audit",
        help="Export blind and gold-aware audit worksheets from GPT-5.5 results.",
    )
    export.add_argument("--input", type=Path, default=DEFAULT_GPT55_JUDGED)
    export.add_argument("--json-output", type=Path, default=DEFAULT_AUDIT_JSON)
    export.add_argument("--blind-output", type=Path, default=DEFAULT_AUDIT_BLIND)
    export.add_argument(
        "--adjudication-output", type=Path, default=DEFAULT_AUDIT_ADJUDICATION
    )
    export.set_defaults(func=command_export_failure_audit)

    run_mini = subparsers.add_parser(
        "run-mini",
        help="Run gpt-4o-mini under the same no-Skill/Skill-v2 conditions.",
    )
    add_dataset_arguments(run_mini)
    run_mini.add_argument("--model", default=os.getenv("MINI_MODEL", "gpt-4o-mini"))
    run_mini.add_argument("--skill", type=Path, default=DEFAULT_SKILL)
    run_mini.add_argument("--context-audit", type=Path, default=DEFAULT_MINI_CONTEXT_AUDIT)
    run_mini.add_argument("--output", type=Path, default=DEFAULT_MINI_GENERATION)
    run_mini.add_argument("--seed", type=int, default=42)
    run_mini.add_argument("--retries", type=int, default=3)
    run_mini.add_argument("--overwrite", action="store_true")
    run_mini.set_defaults(func=command_run_mini)

    judge_mini = subparsers.add_parser(
        "judge-mini", help="Apply the verified L4 Judge to mini outputs."
    )
    judge_mini.add_argument("--input", type=Path, default=DEFAULT_MINI_GENERATION)
    judge_mini.add_argument("--output", type=Path, default=DEFAULT_MINI_JUDGED)
    judge_mini.add_argument(
        "--judge-model",
        dest="judge_models",
        action="append",
        help="Repeat for distinct Judge models. Default: JUDGE_MODELS or gpt-5.5.",
    )
    judge_mini.add_argument("--overwrite", action="store_true")
    judge_mini.set_defaults(func=command_judge_mini)

    compare = subparsers.add_parser(
        "compare-models", help="Verify parity and compare mini with GPT-5.5."
    )
    compare.add_argument("--mini", type=Path, default=DEFAULT_MINI_JUDGED)
    compare.add_argument("--gpt55", type=Path, default=DEFAULT_GPT55_JUDGED)
    compare.add_argument("--json-output", type=Path, default=DEFAULT_COMPARISON_JSON)
    compare.add_argument("--csv-output", type=Path, default=DEFAULT_COMPARISON_CSV)
    compare.add_argument("--md-output", type=Path, default=DEFAULT_COMPARISON_MD)
    compare.set_defaults(func=command_compare_models)

    summarize = subparsers.add_parser(
        "summarize-audit",
        help="Summarize a reviewer-completed adjudication CSV.",
    )
    summarize.add_argument("--input", type=Path, default=DEFAULT_AUDIT_ADJUDICATION)
    summarize.add_argument("--output", type=Path, default=DEFAULT_AUDIT_SUMMARY)
    summarize.set_defaults(func=command_summarize_audit)
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
        print("Interrupted. Saved API records can be resumed.", file=sys.stderr)
        raise SystemExit(130)