from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import factorybench_json_contract_v2 as v2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_ID = "69d936dd-9ceb-4520-bb97-bf13ce264ec7"
DEFAULT_BASELINE = SCRIPT_DIR / "results" / "json_contract_v2" / "gpt55_judged.json"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "rag_smoke"
DEFAULT_CORPUS = DEFAULT_RESULTS_DIR / "oracle_fixture_corpus.json"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "rag_oracle_smoke.json"

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1}


def applicability_context(original_input: str) -> str:
    """Keep machine/signal context and omit the original question."""
    question = v2.QUESTION_HEADING.search(original_input)
    context = original_input[: question.start()] if question else original_input
    # Time-series values make the fixture unnecessarily large.  Retrieval only
    # needs the machine description and signal mapping for this sanity check.
    return re.split(r"^[ \t]*Time series[ \t]*:", context, maxsplit=1, flags=re.I | re.M)[0].strip()


def build_oracle_fixture(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for case in cases:
        reference = v2.normalize_reference(case)
        if not reference["corrective_actions_evaluable"]:
            continue
        docs.append(
            {
                "doc_id": f"oracle-fixture-{case['id']}",
                "source_case_id": str(case["id"]),
                "applicability_context": applicability_context(str(case["input"])),
                "canonical_root_cause": reference["canonical_root_cause"],
                "corrective_action_reference": reference["corrective_action_reference"],
                "fixture_only": True,
            }
        )
    if len(docs) < 2:
        raise SystemExit("Need at least two semantic-reference cases for the fixture corpus.")
    return docs[:2]


def retrieve(query: str, docs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_tokens = tokens(applicability_context(query))
    ranking: list[dict[str, Any]] = []
    for doc in docs:
        doc_tokens = tokens(str(doc["applicability_context"]))
        union = query_tokens | doc_tokens
        score = len(query_tokens & doc_tokens) / len(union) if union else 0.0
        ranking.append({"doc": doc, "score": score})
    ranking.sort(key=lambda item: (-item["score"], item["doc"]["doc_id"]))
    trace = [
        {
            "rank": index,
            "doc_id": item["doc"]["doc_id"],
            "source_case_id": item["doc"]["source_case_id"],
            "score": item["score"],
        }
        for index, item in enumerate(ranking, start=1)
    ]
    return ranking[0]["doc"], trace


def load_baseline_row(path: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = v2.base.load_json(path)
    if payload.get("status") != "completed":
        raise SystemExit(f"Baseline is not completed: {path}")
    matches = [
        row
        for row in payload.get("records", [])
        if str(row.get("id")) == case_id and row.get("condition") == "gpt55_skill_v2"
    ]
    if len(matches) != 1:
        raise SystemExit(f"Expected one gpt55_skill_v2 baseline row for {case_id}; found {len(matches)}")
    return payload, matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fast oracle RAG plumbing test.")
    parser.add_argument("--dataset", type=Path, default=v2.DEFAULT_DATASET)
    parser.add_argument("--split", default="final_test")
    parser.add_argument("--expected-cases", type=int, default=5)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--skill", type=Path, default=v2.DEFAULT_SKILL)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--corpus-output", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", default="gpt-5.5")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        existing = v2.base.load_json(args.output)
        print(json.dumps(existing.get("summary", existing), ensure_ascii=False, indent=2))
        print(f"Existing result: {args.output}")
        return

    cases = v2.base.load_cases(args.dataset, args.split, args.expected_cases)
    by_id = {str(case["id"]): case for case in cases}
    if args.case_id not in by_id:
        raise SystemExit(f"Case not found: {args.case_id}")
    case = by_id[args.case_id]
    reference = v2.normalize_reference(case)
    if not reference["corrective_actions_evaluable"]:
        raise SystemExit("The selected smoke-test case needs a semantic reference.")

    baseline_payload, baseline_row = load_baseline_row(args.baseline, args.case_id)
    corpus = build_oracle_fixture(cases)
    args.corpus_output.parent.mkdir(parents=True, exist_ok=True)
    v2.base.atomic_save(
        args.corpus_output,
        {
            "corpus_type": "oracle_fixture_for_plumbing_only",
            "dataset_sha256": v2.base.sha256_file(args.dataset),
            "documents": corpus,
            "warning": "Contains held-out gold. Never use this corpus for a formal RAG-effect claim.",
        },
    )

    retrieved, retrieval_trace = retrieve(str(case["input"]), corpus)
    retrieval_hit = retrieved["source_case_id"] == args.case_id
    if not retrieval_hit:
        raise SystemExit(
            "Fixture retrieval missed the selected case; no model call was made. "
            f"Top document: {retrieved['doc_id']}"
        )

    skill_text = args.skill.read_text(encoding="utf-8")
    model = args.model or str(
        baseline_payload.get("configuration", {}).get("target_model") or "gpt-5.5"
    )
    retrieved_block = json.dumps(
        {
            "source_id": retrieved["doc_id"],
            "applicability_context": retrieved["applicability_context"],
            "documented_root_cause": retrieved["canonical_root_cause"],
            "documented_corrective_action": retrieved["corrective_action_reference"],
        },
        ensure_ascii=False,
        indent=2,
    )
    instructions = f"""{v2.COMMON_INSTRUCTIONS}

Use the following reusable diagnostic reasoning procedure. It is not a
machine-specific fault catalog.

--- BEGIN DIAGNOSTIC SKILL V2 ---
{skill_text}
--- END DIAGNOSTIC SKILL V2 ---

A single retrieved knowledge passage is supplied for this oracle plumbing
test. Use it only when its applicability context matches the case. Include its
source_id verbatim in one evidence item. Do not cite any source not supplied.
No signal-analysis tool or candidate options are supplied.""".strip()
    target_input = (
        v2.rewrite_target_input(str(case["input"]))
        + "\n\nRetrieved knowledge passage (oracle fixture; external to telemetry):\n"
        + retrieved_block
    )

    output: dict[str, Any] = {
        "status": "running",
        "experiment_type": "one_case_oracle_rag_plumbing_smoke",
        "contract_version": v2.CONTRACT_VERSION,
        "case_id": args.case_id,
        "target_model": model,
        "judge_model": args.judge_model,
        "baseline_file": str(args.baseline.resolve()),
        "baseline_sha256": v2.base.sha256_file(args.baseline),
        "corpus_file": str(args.corpus_output.resolve()),
        "corpus_sha256": v2.base.sha256_file(args.corpus_output),
        "retrieval": {"hit": retrieval_hit, "top_k": 1, "trace": retrieval_trace},
        "baseline": {
            "condition": "gpt55_skill_v2_no_rag",
            "diagnosis_score": baseline_row.get("median_diagnosis_score"),
            "full_protocol_score": baseline_row.get("median_full_protocol_score"),
        },
        "rag": {"condition": "gpt55_skill_v2_oracle_rag"},
        "limitations": [
            "The fixture corpus contains held-out gold and is intentionally leaky.",
            "One case cannot estimate a stable RAG effect.",
            "Passing validates retrieval/injection/generation/judging plumbing only.",
        ],
    }
    v2.base.atomic_save(args.output, output)

    client = v2.base.require_api_client()
    started = time.time()
    response = v2.base.call_target_model(client, model, instructions, target_input, args.retries)
    raw = v2.base.response_output_text(response)
    validation = v2.evaluator.validate_deterministically(raw)
    output["rag"].update(
        {
            "raw_response": raw,
            "latency_seconds": time.time() - started,
            "token_usage": v2.base.response_usage(response),
            "deterministic_validation": validation,
        }
    )
    v2.base.atomic_save(args.output, output)

    if not validation["valid"]:
        output["status"] = "completed"
        output["rag"].update({"diagnosis_score": 0.0, "full_protocol_score": 0.0})
        output["summary"] = {
            "smoke_pass": False,
            "reason": "Retrieved the expected document, but target JSON violated the v2 contract.",
        }
    else:
        judge_payload = {
            "question": target_input,
            "canonical_root_cause": reference["canonical_root_cause"],
            "corrective_action_reference": reference["corrective_action_reference"],
            "corrective_actions_evaluable": True,
            "model_raw_answer": raw,
        }
        judged = v2.call_judge(client, args.judge_model, judge_payload, args.retries)
        parsed = judged["parsed"]
        output["rag"].update(
            {
                "judge": judged,
                "diagnosis_score": parsed["diagnosis_score"],
                "full_protocol_score": parsed["full_protocol_score"],
            }
        )
        output["status"] = "completed"
        output["summary"] = {
            "smoke_pass": bool(
                retrieval_hit
                and validation["valid"]
                and parsed["diagnosis_score"] == 1.0
            ),
            "retrieval_hit": retrieval_hit,
            "json_contract_valid": validation["valid"],
            "baseline_diagnosis_score": baseline_row.get("median_diagnosis_score"),
            "rag_diagnosis_score": parsed["diagnosis_score"],
            "baseline_full_protocol_score": baseline_row.get("median_full_protocol_score"),
            "rag_full_protocol_score": parsed["full_protocol_score"],
            "interpretation": (
                "Pipeline upper-bound smoke passed; replace oracle fixture with real manuals before formal evaluation."
                if parsed["diagnosis_score"] == 1.0
                else "Pipeline ran, but oracle knowledge did not yield a correct diagnosis; inspect the saved trace."
            ),
        }

    v2.base.atomic_save(args.output, output)
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()