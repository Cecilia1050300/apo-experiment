from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from statistics import median
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent

# The script works either inside factorybench_experiment/ or one directory above it.
if (SCRIPT_DIR / "data" / "factorybench_l4_subset.json").exists():
    EXPERIMENT_DIR = SCRIPT_DIR
elif (
    SCRIPT_DIR / "factorybench_experiment" / "data" / "factorybench_l4_subset.json"
).exists():
    EXPERIMENT_DIR = SCRIPT_DIR / "factorybench_experiment"
else:
    EXPERIMENT_DIR = SCRIPT_DIR

DEFAULT_DATASET = EXPERIMENT_DIR / "data" / "factorybench_l4_subset.json"
DEFAULT_RESULT_DIR = EXPERIMENT_DIR / "results"

REQUIRED_KEYS = {"root_cause", "evidence", "corrective_actions"}
OPTION_ONLY = re.compile(r"^\s*[ABCD]\s*[.)]?\s*$", re.IGNORECASE)
SUSPICIOUS_TRUNCATION = {
    "root cause",
    "root_cause",
    "corrective action",
    "corrective actions",
    "corrective_actions",
}

JUDGE_INSTRUCTIONS = """
You are validating a FactoryBench Level 4 industrial troubleshooting evaluator.

Judge the COMPLETE model raw answer semantically. Do not use a parser-generated
summary and do not require exact snake_case matching. Accept a faithful semantic
paraphrase of the known root cause.

You will receive exactly these materials:
1. question: the complete FactoryBench case input.
2. reference_answer: the reference response/procedure.
3. known_root_cause: the canonical root cause.
4. model_raw_answer: the complete, unmodified answer being evaluated.

FactoryBench L4 scoring rubric:
- 1.0: The underlying root cause is correct or clearly equivalent, and the
  corrective actions are broadly correct according to the reference answer.
- 0.5: The underlying root cause is correct or clearly equivalent, but the
  corrective actions are missing, materially incorrect, or off-topic.
- 0.0: The root cause is incorrect, the answer only describes a symptom, or
  the answer is unusable.

Rules:
- A torque/force/position anomaly by itself is a symptom, not a root cause.
- Do not infer content that is absent from model_raw_answer.
- If the root cause is incorrect, score must be 0.0.
- Evidence quality is not a separate scoring dimension in this smoke test.

Return JSON only with exactly these keys:
{
  "score": 0.0,
  "root_cause_correct": false,
  "protocol_correct": false,
  "reason": "brief evidence-based explanation"
}
""".strip()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_target_json(raw_response: str) -> dict[str, Any]:
    """Strictly parse the full response; Markdown fences/trailing text are invalid."""
    value = json.loads(
        raw_response,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_json_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def validate_deterministically(raw_response: str) -> dict[str, Any]:
    """Validate transport/schema only; never decide semantic correctness."""
    before_digest = sha256_text(raw_response)
    errors: list[str] = []
    parsed: dict[str, Any] | None = None

    if not raw_response.strip():
        errors.append("empty_response")
    elif OPTION_ONLY.fullmatch(raw_response):
        errors.append("option_letter_only")
    else:
        try:
            parsed = parse_target_json(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid_or_incomplete_json: {exc}")

    if parsed is not None:
        keys = set(parsed)
        missing = sorted(REQUIRED_KEYS - keys)
        unexpected = sorted(keys - REQUIRED_KEYS)
        if missing:
            errors.append(f"missing_fields: {missing}")
        if unexpected:
            errors.append(f"unexpected_fields: {unexpected}")

        root_cause = parsed.get("root_cause")
        evidence = parsed.get("evidence")
        actions = parsed.get("corrective_actions")

        if not isinstance(root_cause, str):
            errors.append("root_cause_must_be_string")
        elif not root_cause.strip():
            errors.append("root_cause_is_empty")
        elif root_cause.strip().lower() in SUSPICIOUS_TRUNCATION:
            errors.append("suspected_truncated_heading")

        if not isinstance(evidence, list) or not all(
            isinstance(item, str) for item in evidence
        ):
            errors.append("evidence_must_be_string_array")

        if not isinstance(actions, list) or not all(
            isinstance(item, str) for item in actions
        ):
            errors.append("corrective_actions_must_be_string_array")

        has_answer_content = (
            bool(isinstance(root_cause, str) and root_cause.strip())
            or bool(
                isinstance(evidence, list)
                and any(str(item).strip() for item in evidence)
            )
            or bool(
                isinstance(actions, list)
                and any(str(item).strip() for item in actions)
            )
        )
        if not has_answer_content:
            errors.append("answer_has_no_content")

        # Verify a semantic JSON round trip. Duplicate keys were already rejected,
        # so any inequality here indicates parser/serialization content loss.
        round_trip = json.loads(
            json.dumps(parsed, ensure_ascii=False),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_json_constant,
        )
        if round_trip != parsed:
            errors.append("parser_content_loss")

    after_digest = sha256_text(raw_response)
    raw_unchanged = before_digest == after_digest
    if not raw_unchanged:
        errors.append("raw_response_was_modified")

    return {
        "valid": not errors,
        "errors": errors,
        "parsed": parsed,
        "raw_sha256_before_parse": before_digest,
        "raw_sha256_after_parse": after_digest,
        "raw_response_unchanged": raw_unchanged,
        "semantic_score": None,
    }


def run_parser_regression_tests() -> list[dict[str, Any]]:
    fixtures = [
        {
            "name": "complete_json",
            "raw": json.dumps(
                {
                    "root_cause": "collision with a rigid object",
                    "evidence": ["contact force changes during the task"],
                    "corrective_actions": ["stop and inspect the work area"],
                }
            ),
            "expected_valid": True,
            "expected_error": None,
        },
        {
            "name": "free_form_response",
            "raw": "The robot appears to have collided with an object.",
            "expected_valid": False,
            "expected_error": "invalid_or_incomplete_json",
        },
        {
            "name": "blank_response",
            "raw": "",
            "expected_valid": False,
            "expected_error": "empty_response",
        },
        {
            "name": "bare_option_letter",
            "raw": "B",
            "expected_valid": False,
            "expected_error": "option_letter_only",
        },
        {
            "name": "truncated_root_cause_heading",
            "raw": json.dumps(
                {
                    "root_cause": "root cause",
                    "evidence": [],
                    "corrective_actions": [],
                }
            ),
            "expected_valid": False,
            "expected_error": "suspected_truncated_heading",
        },
        {
            "name": "markdown_heading_response",
            "raw": "## Root cause\nCollision with an object\n\n## Corrective actions\nStop.",
            "expected_valid": False,
            "expected_error": "invalid_or_incomplete_json",
        },
    ]

    results = []
    for fixture in fixtures:
        validation = validate_deterministically(fixture["raw"])
        expected_error = fixture["expected_error"]
        error_matched = expected_error is None or any(
            error.startswith(expected_error) for error in validation["errors"]
        )
        passed = (
            validation["valid"] == fixture["expected_valid"]
            and error_matched
            and validation["raw_response_unchanged"]
        )
        results.append(
            {
                "name": fixture["name"],
                "expected_valid": fixture["expected_valid"],
                "expected_error": expected_error,
                "actual_validation": validation,
                "passed": passed,
            }
        )
    return results


def load_real_case(
    dataset_path: Path,
    split: str,
    case_id: str | None,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    splits = dataset.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Dataset must contain an object field named 'splits'.")
    if split not in splits:
        raise ValueError(
            f"Split {split!r} not found. Available splits: {list(splits)}"
        )

    cases = splits[split]
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Split {split!r} is empty or is not a list.")

    if case_id:
        candidates = [item for item in cases if str(item.get("id")) == case_id]
        if not candidates:
            raise ValueError(f"Case id {case_id!r} was not found in split {split!r}.")
    else:
        # A non-normal case is required so the intentionally wrong 'normal'
        # fixture is unambiguously incorrect.
        candidates = [
            item
            for item in cases
            if str(item.get("root_cause", "")).strip().lower() != "normal"
        ]

    for item in candidates:
        if (
            str(item.get("root_cause", "")).strip().lower() != "normal"
            and item.get("input") not in (None, "")
            and item.get("root_cause") not in (None, "")
            and item.get("reference_answer") not in (None, "")
        ):
            return item

    raise ValueError(
        "No suitable non-normal case contains all required fields: "
        "input, root_cause, reference_answer."
    )


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def semantic_paraphrase(canonical_root_cause: str) -> str:
    """Use confirmed FactoryBench labels when available; otherwise only humanize."""
    confirmed = {
        "collision_rigid_object": "The robot collided with a rigid object.",
        "collision_foam_object": "The robot collided with a compliant foam object.",
        "loosening_phase": "The fault occurred during the screw-loosening phase.",
        "extra_assembly_component": "An extra component is present in the assembly.",
        "unstable_mounting_platform": "The robot's mounting platform is unstable.",
        "gripper_activation_failure": "The gripper failed to activate.",
    }
    return confirmed.get(
        canonical_root_cause,
        canonical_root_cause.replace("_", " ").strip(),
    )


def as_target_json(
    root_cause: str,
    corrective_actions: list[str],
    evidence: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "root_cause": root_cause,
            "evidence": evidence or [],
            "corrective_actions": corrective_actions,
        },
        ensure_ascii=False,
        indent=2,
    )


def build_semantic_fixtures(case: dict[str, Any]) -> list[dict[str, Any]]:
    known_root_cause = str(case["root_cause"])
    reference_answer = value_to_text(case["reference_answer"])

    return [
        {
            "name": "correct_root_and_correct_action",
            "description": "Correct root cause plus reference corrective answer",
            "raw_response": as_target_json(
                known_root_cause,
                [reference_answer],
            ),
            "expected_deterministic_valid": True,
            "expected_judge_score": 1.0,
        },
        {
            "name": "correct_root_and_wrong_action",
            "description": "Correct root cause but materially wrong action",
            "raw_response": as_target_json(
                known_root_cause,
                ["Continue operating without inspection, recovery, or verification."],
            ),
            "expected_deterministic_valid": True,
            "expected_judge_score": 0.5,
        },
        {
            "name": "symptom_only",
            "description": "Only describes torque/force symptoms",
            "raw_response": as_target_json(
                "The torque and estimated contact force are abnormal.",
                ["Inspect the system."],
            ),
            "expected_deterministic_valid": True,
            "expected_judge_score": 0.0,
        },
        {
            "name": "wrong_root_cause",
            "description": "Incorrectly claims normal operation",
            "raw_response": as_target_json(
                "The machine is operating normally and no fault is present.",
                ["Continue normal operation."],
            ),
            "expected_deterministic_valid": True,
            "expected_judge_score": 0.0,
        },
        {
            "name": "bare_option_letter",
            "description": "Bare answer choice must be blocked before semantic judging",
            "raw_response": "B",
            "expected_deterministic_valid": False,
            "expected_judge_score": None,
        },
        {
            "name": "semantic_paraphrase",
            "description": "Correct semantic paraphrase plus reference corrective answer",
            "raw_response": as_target_json(
                semantic_paraphrase(known_root_cause),
                [reference_answer],
            ),
            "expected_deterministic_valid": True,
            "expected_judge_score": 1.0,
        },
    ]


def parse_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    start = raw.find("{")
    if start < 0:
        raise ValueError("Judge response does not contain a JSON object.")
    value, end = decoder.raw_decode(raw[start:])
    if not isinstance(value, dict):
        raise ValueError("Judge response is not a JSON object.")
    if raw[start + end :].strip() not in ("", "```"):
        raise ValueError("Judge response contains unexpected trailing content.")
    return value


def validate_judge_output(value: dict[str, Any]) -> dict[str, Any]:
    required = {"score", "root_cause_correct", "protocol_correct", "reason"}
    if set(value) != required:
        raise ValueError(
            f"Judge output keys must be exactly {sorted(required)}; got {sorted(value)}"
        )

    score = float(value["score"])
    if score not in {0.0, 0.5, 1.0}:
        raise ValueError(f"Judge score must be 0.0, 0.5, or 1.0; got {score}")
    if not isinstance(value["root_cause_correct"], bool):
        raise ValueError("root_cause_correct must be boolean")
    if not isinstance(value["protocol_correct"], bool):
        raise ValueError("protocol_correct must be boolean")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a non-empty string")

    root_correct = value["root_cause_correct"]
    protocol_correct = value["protocol_correct"]
    inconsistent = (
        (score == 0.0 and root_correct)
        or (score == 0.5 and not (root_correct and not protocol_correct))
        or (score == 1.0 and not (root_correct and protocol_correct))
    )
    if inconsistent:
        raise ValueError(
            "Judge correctness flags are inconsistent with the rubric score."
        )

    return {
        "score": score,
        "root_cause_correct": value["root_cause_correct"],
        "protocol_correct": value["protocol_correct"],
        "reason": value["reason"].strip(),
    }


def call_judge(
    client: Any,
    judge_model: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.responses.create(
                model=judge_model,
                instructions=JUDGE_INSTRUCTIONS,
                input=json.dumps(payload, ensure_ascii=False, indent=2),
            )
            raw = (getattr(response, "output_text", "") or "").strip()
            parsed = validate_judge_output(parse_json_object(raw))
            return {
                "status": "ok",
                "raw_response": raw,
                "parsed": parsed,
                "attempt": attempt,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 3:
                wait_seconds = 2 ** (attempt - 1)
                print(f"    Judge error; retrying in {wait_seconds}s: {exc}")
                time.sleep(wait_seconds)

    raise RuntimeError(f"Judge failed after 3 attempts: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the FactoryBench deterministic and semantic evaluator."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("FACTORYBENCH_DATASET", str(DEFAULT_DATASET))),
    )
    parser.add_argument(
        "--split",
        default=os.getenv("EVAL_SPLIT", "final_test"),
    )
    parser.add_argument(
        "--case-id",
        default=os.getenv("CASE_ID"),
        help="Optional real case id. It must be non-normal and contain required fields.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", "gpt-5.5"),
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=int(os.getenv("JUDGE_REPEATS", "1")),
        help="Repeated calls to one model; these are not independent judge models.",
    )
    parser.add_argument(
        "--validator-only",
        action="store_true",
        help="Run parser and deterministic checks without calling the API.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judge_repeats < 1:
        raise SystemExit("--judge-repeats must be at least 1.")

    parser_regression = run_parser_regression_tests()
    parser_passed = all(row["passed"] for row in parser_regression)
    print("Parser/validator regression:")
    for row in parser_regression:
        print(f"  {row['name']}: {'PASS' if row['passed'] else 'FAIL'}")

    case = load_real_case(args.dataset, args.split, args.case_id)
    fixtures = build_semantic_fixtures(case)
    output_path = args.output or (
        DEFAULT_RESULT_DIR
        / f"evaluator_smoke_test_{safe_name(args.judge_model)}.json"
    )

    output: dict[str, Any] = {
        "status": "raw_responses_saved",
        "experiment_type": "evaluator_smoke_test_not_formal_result",
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "case": {
            "id": case.get("id"),
            "question": case.get("input"),
            "known_root_cause": case.get("root_cause"),
            "reference_answer": case.get("reference_answer"),
        },
        "judge_model": None if args.validator_only else args.judge_model,
        "judge_repeats_same_model": 0 if args.validator_only else args.judge_repeats,
        "independent_judge_models": False,
        "parser_regression": parser_regression,
        "records": [
            {
                **fixture,
                # Persist the complete raw answer before any parsing.
                "raw_response_sha256_at_save": sha256_text(fixture["raw_response"]),
                "deterministic_validation": None,
                "judge_votes": [],
                "median_score": None,
                "passed": None,
            }
            for fixture in fixtures
        ],
        "summary": {},
    }
    atomic_save(output_path, output)

    for index, record in enumerate(output["records"], start=1):
        validation = validate_deterministically(record["raw_response"])
        record["deterministic_validation"] = validation
        record["raw_response_still_matches_saved_digest"] = (
            sha256_text(record["raw_response"])
            == record["raw_response_sha256_at_save"]
        )
        record["deterministic_expectation_met"] = (
            validation["valid"] == record["expected_deterministic_valid"]
        )
        print(
            f"[{index}/{len(output['records'])}] {record['name']}: "
            f"deterministic_valid={validation['valid']}"
        )
        atomic_save(output_path, output)

    deterministic_passed = all(
        record["deterministic_expectation_met"]
        and record["raw_response_still_matches_saved_digest"]
        for record in output["records"]
    )

    if not parser_passed or not deterministic_passed:
        output["status"] = "failed_deterministic_stage"
        output["summary"] = {
            "parser_regression_passed": parser_passed,
            "deterministic_fixtures_passed": deterministic_passed,
            "semantic_judge_smoke_test_passed": False,
            "overall_passed": False,
        }
        atomic_save(output_path, output)
        raise SystemExit(f"Deterministic stage failed. See: {output_path}")

    if args.validator_only:
        for record in output["records"]:
            record["passed"] = record["deterministic_expectation_met"]
        output["status"] = "completed_validator_only"
        output["summary"] = {
            "parser_regression_passed": True,
            "deterministic_fixtures_passed": True,
            "semantic_judge_smoke_test_run": False,
            "overall_passed": True,
        }
        atomic_save(output_path, output)
        print(f"\nValidator-only smoke test PASSED. Saved: {output_path}")
        return

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_ADMIN_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY or OPENAI_ADMIN_KEY is not set. "
            "The deterministic results were saved; set credentials and rerun."
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The openai package is required for the semantic Judge stage. "
            "Install it or run with --validator-only."
        ) from exc

    client = OpenAI()
    question = value_to_text(case["input"])
    reference_answer = value_to_text(case["reference_answer"])
    known_root_cause = value_to_text(case["root_cause"])

    for index, record in enumerate(output["records"], start=1):
        validation = record["deterministic_validation"]
        if not validation["valid"]:
            record["judge_status"] = "skipped_deterministic_invalid"
            record["passed"] = record["expected_judge_score"] is None
            print(f"[{index}/{len(output['records'])}] {record['name']}: Judge SKIPPED")
            atomic_save(output_path, output)
            continue

        payload = {
            "question": question,
            "reference_answer": reference_answer,
            "known_root_cause": known_root_cause,
            "model_raw_answer": record["raw_response"],
        }
        votes = []
        errors = []
        for repeat_index in range(1, args.judge_repeats + 1):
            try:
                vote = call_judge(client, args.judge_model, payload)
                vote["repeat_index"] = repeat_index
                votes.append(vote)
                print(
                    f"[{index}/{len(output['records'])}] {record['name']} "
                    f"judge {repeat_index}: score={vote['parsed']['score']}"
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
                print(
                    f"[{index}/{len(output['records'])}] {record['name']} "
                    f"judge {repeat_index}: ERROR {exc}"
                )

        record["judge_votes"] = votes
        record["judge_errors"] = errors
        if votes:
            scores = [vote["parsed"]["score"] for vote in votes]
            record["median_score"] = float(median(scores))
            record["judge_status"] = "ok"
            record["passed"] = (
                record["median_score"] == record["expected_judge_score"]
            )
        else:
            record["judge_status"] = "error"
            record["passed"] = False
        atomic_save(output_path, output)

    semantic_passed = all(record["passed"] for record in output["records"])
    output["status"] = "completed"
    output["summary"] = {
        "parser_regression_passed": parser_passed,
        "deterministic_fixtures_passed": deterministic_passed,
        "semantic_judge_smoke_test_passed": semantic_passed,
        "overall_passed": parser_passed and deterministic_passed and semantic_passed,
        "formal_three_independent_judge_result": False,
    }
    atomic_save(output_path, output)

    print(f"\nSaved: {output_path}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))

    if not output["summary"]["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()