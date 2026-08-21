#!/usr/bin/env python3
"""
Arm C: CoEvo for FactoryBench L1-L3.

- Agent: fixed GPT-4o-mini.
- Skill evolves on development only.
- Surrogate verifier evolves on development only.
- Hidden FactoryBench GT is never shown to the surrogate during item verification.
- Development GT/mismatch evidence may be used by the verifier optimizer.
- Holdout is frozen: no skill update, no verifier update, no feedback.

This script reuses:
    scripts/meta_coevo/run_static_surrogate.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO = Path("/home/training/automatic_prompt_engineer")
ROOT = REPO / "hermes_factorybench_skill_experiment"
BASE_RUNNER = ROOT / "scripts/meta_coevo/run_static_surrogate.py"

ARM = "coevo"
TASK = "m1_factorybench_l123"

RESULT_DIR = ROOT / "results/meta_coevo/coevo"
TRACE_DIR = ROOT / "traces/meta_coevo/coevo"
ADAPTER_ROOT = ROOT / "prompts/adapters/meta_coevo_coevo"
VERIFIER_ROOT = ROOT / "prompts/meta_coevo/verifiers/coevo"

VERIFIER_OPTIMIZER_MODEL = os.getenv(
    "VERIFIER_OPTIMIZER_MODEL",
    os.getenv("SURROGATE_MODEL", "gpt-5.6-luna"),
)
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "8192"))

VERIFIER_OPTIMIZER_SYSTEM = r"""
You revise a reusable surrogate verifier for FactoryBench Level 1-3 tasks.

Goal:
Improve agreement with the hidden deterministic GT oracle on DEVELOPMENT data,
especially reducing false PASS and false REJECT decisions, while remaining
general across unseen cases.

You are revising verifier RULES, not solving or memorizing individual cases.

Strict constraints:
1. Do not include case IDs, episode IDs, holdout references, gold answers,
   exact answer labels, or case-specific signal values in the verifier.
2. Do not encode dataset-specific lookup tables.
3. Do not write rules that reveal or reconstruct hidden GT.
4. The verifier itself must still judge using only:
   - task/question text
   - required answer format
   - agent answer
5. Prefer general procedural checks:
   temporal-window validity, operator semantics, threshold direction,
   complete-window requirements, ranking continuity, multi-element checking,
   format compliance, and unsupported-assumption detection.
6. If mismatch evidence is too weak to justify a general rule, preserve the
   previous verifier rather than inventing a brittle rule.
7. Return JSON only.

Required JSON schema:
{
  "decision": "VERIFIER" | "NO_CHANGE",
  "verifier_text": "full reusable verifier prompt, or empty string for NO_CHANGE",
  "changes": ["general change"],
  "mismatch_taxonomy": [
    {
      "category": "general mismatch category",
      "false_pass_count": 0,
      "false_reject_count": 0,
      "evidence_count": 0
    }
  ],
  "predicted_risks": ["generalization risk"],
  "evidence_limitations": ["limitation"]
}

The full verifier_text, when decision=VERIFIER, must instruct the verifier to
return exactly:
{
  "verdict": "PASS" | "FAIL",
  "confidence": 0.0,
  "diagnosis": ["brief reason"],
  "failed_checks": ["name of failed verification rule"]
}
""".strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, obj: Any) -> None:
    write_new(
        path,
        (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(),
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_base():
    if not BASE_RUNNER.exists():
        raise SystemExit(f"missing Arm-B runner: {BASE_RUNNER}")
    spec = importlib.util.spec_from_file_location("arm_b_static_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to import Arm-B runner")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    mod.ARM = ARM
    mod.RESULT_DIR = RESULT_DIR
    mod.TRACE_DIR = TRACE_DIR
    mod.ADAPTER_ROOT = ADAPTER_ROOT
    return mod


def verifier_dir(task: str) -> Path:
    return VERIFIER_ROOT / task


def verifier_path(task: str, version: int) -> Path:
    return verifier_dir(task) / f"verifier_v{version}.txt"


def verifier_trace_dir(task: str) -> Path:
    return TRACE_DIR / task / "verifier"


def item_lookup(base, manifest_path: Path):
    manifest, items = base.source_items(manifest_path)
    return manifest, {item.id: item for item in items}


def compact_for_verifier_revision(result: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for item in result["items"]:
        rows.append(
            {
                "id": item["id"],
                "level": item["level"],
                "dataset": item["dataset"],
                "answer_format": item["answer_format"],
                "agent_answer": item["raw_output"],
                "gt_score": item["score"],
                "gt_pass": item["gt_pass"],
                "surrogate_verdict": item["surrogate_verdict"],
                "false_pass": item["false_pass"],
                "false_reject": item["false_reject"],
                "surrogate_diagnosis": (item.get("surrogate") or {}).get(
                    "diagnosis", []
                ),
                "surrogate_failed_checks": (item.get("surrogate") or {}).get(
                    "failed_checks", []
                ),
                "development_evidence": item.get("development_evidence"),
            }
        )
    return {
        "condition": result["condition"],
        "item_count": result["item_count"],
        "surrogate_summary": result["surrogate_summary"],
        "items": rows,
    }


def validate_verifier_text(
    text: str,
    *,
    development_results: list[dict[str, Any]],
    holdout_ids: list[str],
) -> list[str]:
    errors: list[str] = []
    low = text.casefold()

    if not text.strip():
        return ["empty verifier"]

    for token in ["verdict", "PASS", "FAIL", "confidence", "diagnosis", "failed_checks"]:
        if token.casefold() not in low:
            errors.append(f"missing required verifier contract token: {token}")

    # Generic policy words such as "holdout" are NOT leakage by themselves.
    # The previous version rejected any occurrence of the word "holdout",
    # which caused a false positive when the optimizer wrote harmless policy
    # text such as "do not use holdout information".
    #
    # We instead block actual holdout identifiers below. Holdout examples,
    # scores, answers, and IDs are never included in the revision packet.
    if any(hid.casefold() in low for hid in holdout_ids):
        errors.append("holdout ID leak")

    for result in development_results:
        for item in result["items"]:
            if str(item["id"]).casefold() in low:
                errors.append("development ID leak")
                return errors

            evidence = item.get("development_evidence") or {}
            gold = str(evidence.get("reference_answer") or "")
            if len(gold) >= 20 and gold.casefold() in low:
                errors.append("copied gold answer")
                return errors

            rendered = str(evidence.get("rendered_input") or "")
            numbers = set(
                re.findall(
                    r"(?<![A-Za-z])(?:\d{4,}|-?\d+\.\d{3,})(?![A-Za-z])",
                    rendered,
                )
            )
            if any(number in text for number in numbers):
                errors.append("case-specific signal value")
                return errors

    return errors


def call_verifier_optimizer(
    base,
    client: OpenAI,
    *,
    task: str,
    roundn: int,
    previous_verifier_path: Path,
    development_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    trace_dir = verifier_trace_dir(task)
    input_path = trace_dir / f"verifier_round_{roundn}_input.json"
    raw_path = trace_dir / f"verifier_round_{roundn}_raw_output.txt"
    parsed_path = trace_dir / f"verifier_round_{roundn}_parsed_output.json"

    previous_text = previous_verifier_path.read_text(encoding="utf-8")
    mismatches = []
    for result in development_results:
        for row in result["items"]:
            if row.get("false_pass") or row.get("false_reject"):
                mismatches.append(
                    {
                        "condition": result["condition"],
                        "id": row["id"],
                        "level": row["level"],
                        "dataset": row["dataset"],
                        "answer_format": row["answer_format"],
                        "agent_answer": row["raw_output"],
                        "gt_score": row["score"],
                        "gt_pass": row["gt_pass"],
                        "surrogate_verdict": row["surrogate_verdict"],
                        "false_pass": row["false_pass"],
                        "false_reject": row["false_reject"],
                        "surrogate_diagnosis": (row.get("surrogate") or {}).get(
                            "diagnosis", []
                        ),
                        "surrogate_failed_checks": (row.get("surrogate") or {}).get(
                            "failed_checks", []
                        ),
                        "development_evidence": row.get("development_evidence"),
                    }
                )

    packet = {
        "arm": ARM,
        "task_name": task,
        "round": roundn,
        "purpose": "revise reusable surrogate verifier from development-only GT mismatches",
        "previous_verifier": {
            "sha256": sha(previous_verifier_path),
            "text": previous_text,
        },
        "development_results": [
            compact_for_verifier_revision(x) for x in development_results
        ],
        "mismatches_only": mismatches,
        "constraints": {
            "holdout_access": False,
            "case_memorization": False,
            "gt_visible_during_item_verification": False,
            "gt_visible_only_to_development_verifier_optimizer": True,
        },
        "model_roles": {
            "agent_model": base.AGENT_MODEL,
            "surrogate_model": base.SURROGATE_MODEL,
            "verifier_optimizer_model": VERIFIER_OPTIMIZER_MODEL,
            "skill_optimizer_model": base.OPTIMIZER_MODEL,
        },
    }
    write_json(input_path, packet)

    if raw_path.exists() or parsed_path.exists():
        raise RuntimeError(f"existing partial verifier trace for round {roundn}")

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=VERIFIER_OPTIMIZER_MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_OPTIMIZER_SYSTEM},
            {
                "role": "user",
                "content": (
                    "<VERIFIER_REVISION_INPUT>\n"
                    + json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n</VERIFIER_REVISION_INPUT>"
                ),
            },
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    wall = time.perf_counter() - started
    raw = response.choices[0].message.content or ""
    write_new(raw_path, raw.encode())

    try:
        parsed = json.loads(raw.strip())
        json_error = None
    except Exception as exc:
        parsed = {}
        json_error = f"{type(exc).__name__}: {exc}"

    errors = []
    if json_error:
        errors.append(json_error)

    if not isinstance(parsed, dict):
        errors.append("verifier optimizer output is not an object")
        parsed = {}

    decision = parsed.get("decision")
    if decision not in {"VERIFIER", "NO_CHANGE"}:
        errors.append("invalid verifier optimizer decision")

    text = parsed.get("verifier_text")
    if not isinstance(text, str):
        errors.append("verifier_text must be a string")
        text = ""

    if decision == "NO_CHANGE" and text != "":
        errors.append("NO_CHANGE must use empty verifier_text")
    if decision == "VERIFIER" and not text.strip():
        errors.append("VERIFIER requires non-empty verifier_text")

    holdout_ids = [x["id"] for x in load_json(base.manifests(task)["holdout"])["items"]]
    if decision == "VERIFIER":
        errors.extend(
            validate_verifier_text(
                text,
                development_results=development_results,
                holdout_ids=holdout_ids,
            )
        )

    role_usage = base.usage(response)
    envelope = {
        **parsed,
        "_trace_validation": {
            "valid": not errors,
            "errors": errors,
            "input_sha256": sha(input_path),
            "raw_output_sha256": sha(raw_path),
            "previous_verifier_sha256": sha(previous_verifier_path),
            "verifier_optimizer_model": VERIFIER_OPTIMIZER_MODEL,
            "usage": role_usage,
            "cost": base.safe_cost(
                VERIFIER_OPTIMIZER_MODEL, {**role_usage, "calls": 1}
            ),
            "wall_time_seconds": wall,
            "mismatch_count": len(mismatches),
        },
    }
    write_json(parsed_path, envelope)

    if errors:
        raise RuntimeError(f"invalid verifier revision output: {errors}")

    new_path = verifier_path(task, roundn)
    if decision == "VERIFIER":
        write_new(new_path, text.encode())
    else:
        write_new(new_path, previous_text.encode())

    return envelope, new_path


def reverify_existing_result(
    base,
    client: OpenAI,
    *,
    source_result: dict[str, Any],
    manifest_path: Path,
    verifier_prompt_path: Path,
    condition: str,
) -> dict[str, Any]:
    output_path = RESULT_DIR / "development" / f"{TASK}_{condition}.json"
    if output_path.exists():
        return load_json(output_path)

    old_surrogate_path = base.SURROGATE_PATH
    base.SURROGATE_PATH = verifier_prompt_path
    try:
        _, lookup = item_lookup(base, manifest_path)
        new_rows = []
        surrogate_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

        for row in source_result["items"]:
            item = lookup[row["id"]]
            surrogate = base.run_surrogate(client, item, row["raw_output"])
            su = surrogate.get("usage") or {}
            surrogate_usage["input_tokens"] += int(su.get("input_tokens", 0) or 0)
            surrogate_usage["output_tokens"] += int(su.get("output_tokens", 0) or 0)
            if su.get("input_tokens") or su.get("output_tokens"):
                surrogate_usage["calls"] += 1

            verdict = surrogate.get("verdict")
            gt_pass = bool(row["gt_pass"])
            updated = dict(row)
            updated["surrogate"] = surrogate
            updated["surrogate_verdict"] = verdict
            updated["false_pass"] = verdict == "PASS" and not gt_pass
            updated["false_reject"] = verdict == "FAIL" and gt_pass
            new_rows.append(updated)

        payload = dict(source_result)
        payload["arm"] = ARM
        payload["condition"] = condition
        payload["source_agent_condition"] = source_result["condition"]
        payload["agent_answers_reused"] = True
        payload["surrogate_prompt_path"] = str(verifier_prompt_path.relative_to(ROOT))
        payload["surrogate_prompt_sha256"] = sha(verifier_prompt_path)
        payload["surrogate_static"] = False
        payload["surrogate_version"] = verifier_prompt_path.stem
        payload["surrogate_summary"] = base.surrogate_summary(new_rows)
        payload["tokens_used"] = {
            "agent": {
                "model": base.AGENT_MODEL,
                "input_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
                "reused": True,
            },
            "surrogate": {"model": base.SURROGATE_MODEL, **surrogate_usage},
        }
        payload["cost"] = {
            "agent": 0.0,
            "surrogate": base.safe_cost(base.SURROGATE_MODEL, surrogate_usage),
        }
        payload["items"] = new_rows
        write_json(output_path, payload)
        return payload
    finally:
        base.SURROGATE_PATH = old_surrogate_path


def evaluate_with_verifier(
    base,
    client: OpenAI,
    *,
    manifest_path: Path,
    part: str,
    condition: str,
    adapter: str | None,
    verifier_prompt_path: Path,
):
    old_path = base.SURROGATE_PATH
    try:
        base.SURROGATE_PATH = verifier_prompt_path
        return base.evaluate(client, TASK, manifest_path, part, condition, adapter)
    finally:
        base.SURROGATE_PATH = old_path


def coevo_skill_input(
    base,
    *,
    mode: str,
    baseline_a: dict[str, Any],
    baseline_b: dict[str, Any],
    v1a: dict[str, Any] | None,
    v1b: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    verifier_prompt_path: Path,
) -> dict[str, Any]:
    packet = base.m1_input(
        TASK, mode, baseline_a, baseline_b, v1a, v1b, previous
    )
    packet["arm"] = ARM
    packet["surrogate_verifier"] = {
        "enabled": True,
        "mode": verifier_prompt_path.stem,
        "model": base.SURROGATE_MODEL,
        "prompt_sha256": sha(verifier_prompt_path),
        "gt_visible_to_surrogate": False,
        "diagnosis_available_to_optimizer": True,
        "co_evolving": True,
    }
    packet["constraints"]["surrogate_verifier_static"] = False
    packet["constraints"]["holdout_access"] = False
    packet["constraints"]["case_memorization"] = False
    return packet


def development(base, client: OpenAI):
    ms = base.manifests(TASK)

    v0_source = ROOT / "prompts/meta_coevo/surrogate_verifier_v0.txt"
    v0 = verifier_path(TASK, 0)
    write_new(v0, v0_source.read_bytes())

    baseline_a_v0 = evaluate_with_verifier(
        base, client,
        manifest_path=ms["fold_a"], part="development",
        condition="baseline_fold_a_verifier_v0",
        adapter=None, verifier_prompt_path=v0,
    )
    baseline_b_v0 = evaluate_with_verifier(
        base, client,
        manifest_path=ms["fold_b"], part="development",
        condition="baseline_fold_b_verifier_v0",
        adapter=None, verifier_prompt_path=v0,
    )

    verifier_out1, v1 = call_verifier_optimizer(
        base, client,
        task=TASK, roundn=1, previous_verifier_path=v0,
        development_results=[baseline_a_v0, baseline_b_v0],
    )

    baseline_a_v1 = reverify_existing_result(
        base, client,
        source_result=baseline_a_v0, manifest_path=ms["fold_a"],
        verifier_prompt_path=v1, condition="baseline_fold_a_reverified_v1",
    )
    baseline_b_v1 = reverify_existing_result(
        base, client,
        source_result=baseline_b_v0, manifest_path=ms["fold_b"],
        verifier_prompt_path=v1, condition="baseline_fold_b_reverified_v1",
    )

    skill_out1, adapter1, adapter1_path = base.call_m1(
        client, TASK, 1,
        coevo_skill_input(
            base,
            mode="generate",
            baseline_a=baseline_a_v1,
            baseline_b=baseline_b_v1,
            v1a=None, v1b=None, previous=None,
            verifier_prompt_path=v1,
        ),
    )

    adapter1_a_v1 = adapter1_b_v1 = None
    if adapter1:
        adapter1_a_v1 = evaluate_with_verifier(
            base, client,
            manifest_path=ms["fold_a"], part="development",
            condition="adapter_v1_fold_a_verifier_v1",
            adapter=adapter1, verifier_prompt_path=v1,
        )
        adapter1_b_v1 = evaluate_with_verifier(
            base, client,
            manifest_path=ms["fold_b"], part="development",
            condition="adapter_v1_fold_b_verifier_v1",
            adapter=adapter1, verifier_prompt_path=v1,
        )

    verifier_sources = (
        [adapter1_a_v1, adapter1_b_v1]
        if adapter1_a_v1 and adapter1_b_v1
        else [baseline_a_v1, baseline_b_v1]
    )

    verifier_out2, v2 = call_verifier_optimizer(
        base, client,
        task=TASK, roundn=2, previous_verifier_path=v1,
        development_results=verifier_sources,
    )

    adapter1_a_v2 = adapter1_b_v2 = None
    if adapter1_a_v1 and adapter1_b_v1:
        adapter1_a_v2 = reverify_existing_result(
            base, client,
            source_result=adapter1_a_v1, manifest_path=ms["fold_a"],
            verifier_prompt_path=v2, condition="adapter_v1_fold_a_reverified_v2",
        )
        adapter1_b_v2 = reverify_existing_result(
            base, client,
            source_result=adapter1_b_v1, manifest_path=ms["fold_b"],
            verifier_prompt_path=v2, condition="adapter_v1_fold_b_reverified_v2",
        )

    baseline_a_v2 = reverify_existing_result(
        base, client,
        source_result=baseline_a_v0, manifest_path=ms["fold_a"],
        verifier_prompt_path=v2, condition="baseline_fold_a_reverified_v2",
    )
    baseline_b_v2 = reverify_existing_result(
        base, client,
        source_result=baseline_b_v0, manifest_path=ms["fold_b"],
        verifier_prompt_path=v2, condition="baseline_fold_b_reverified_v2",
    )

    previous = (
        {"sha256": sha(adapter1_path), "text": adapter1}
        if adapter1_path and adapter1
        else {"sha256": None, "text": ""}
    )

    skill_out2, adapter2, adapter2_path = base.call_m1(
        client, TASK, 2,
        coevo_skill_input(
            base,
            mode="refine",
            baseline_a=baseline_a_v2,
            baseline_b=baseline_b_v2,
            v1a=adapter1_a_v2,
            v1b=adapter1_b_v2,
            previous=previous,
            verifier_prompt_path=v2,
        ),
    )

    adapter2_a_v2 = adapter2_b_v2 = None
    if adapter2:
        adapter2_a_v2 = evaluate_with_verifier(
            base, client,
            manifest_path=ms["fold_a"], part="development",
            condition="adapter_v2_fold_a_verifier_v2",
            adapter=adapter2, verifier_prompt_path=v2,
        )
        adapter2_b_v2 = evaluate_with_verifier(
            base, client,
            manifest_path=ms["fold_b"], part="development",
            condition="adapter_v2_fold_b_verifier_v2",
            adapter=adapter2, verifier_prompt_path=v2,
        )

    selection_candidates = []
    if adapter1_path and adapter1_a_v2 and adapter1_b_v2:
        selection_candidates.append(
            ("adapter_v1", adapter1_a_v2, adapter1_b_v2, adapter1_path)
        )
    if adapter2_path and adapter2_a_v2 and adapter2_b_v2:
        selection_candidates.append(
            ("adapter_v2", adapter2_a_v2, adapter2_b_v2, adapter2_path)
        )

    selection = base.select_candidate(
        TASK, baseline_a_v2, baseline_b_v2, selection_candidates
    )

    final_verifier = verifier_dir(TASK) / "selected_verifier.txt"
    write_new(final_verifier, v2.read_bytes())

    summary = {
        "arm": ARM,
        "task_name": TASK,
        "status": "DEVELOPMENT_COMPLETE",
        "model_roles": {
            "agent_model": base.AGENT_MODEL,
            "surrogate_model": base.SURROGATE_MODEL,
            "verifier_optimizer_model": VERIFIER_OPTIMIZER_MODEL,
            "skill_optimizer_model": base.OPTIMIZER_MODEL,
        },
        "verifier_evolution": {
            "v0_sha256": sha(v0),
            "round_1_decision": verifier_out1.get("decision"),
            "v1_sha256": sha(v1),
            "round_2_decision": verifier_out2.get("decision"),
            "v2_sha256": sha(v2),
            "selected_verifier_sha256": sha(final_verifier),
        },
        "baseline_fold_a_v0": base.compact(baseline_a_v0),
        "baseline_fold_b_v0": base.compact(baseline_b_v0),
        "baseline_fold_a_v1": base.compact(baseline_a_v1),
        "baseline_fold_b_v1": base.compact(baseline_b_v1),
        "baseline_fold_a_v2": base.compact(baseline_a_v2),
        "baseline_fold_b_v2": base.compact(baseline_b_v2),
        "skill_round_1_decision": skill_out1.get("decision"),
        "adapter_v1_fold_a_v1": base.compact(adapter1_a_v1) if adapter1_a_v1 else None,
        "adapter_v1_fold_b_v1": base.compact(adapter1_b_v1) if adapter1_b_v1 else None,
        "adapter_v1_fold_a_v2": base.compact(adapter1_a_v2) if adapter1_a_v2 else None,
        "adapter_v1_fold_b_v2": base.compact(adapter1_b_v2) if adapter1_b_v2 else None,
        "skill_round_2_decision": skill_out2.get("decision"),
        "adapter_v2_fold_a_v2": base.compact(adapter2_a_v2) if adapter2_a_v2 else None,
        "adapter_v2_fold_b_v2": base.compact(adapter2_b_v2) if adapter2_b_v2 else None,
        "selection": selection,
        "holdout_access": False,
    }
    write_json(RESULT_DIR / "development" / f"{TASK}_summary.json", summary)
    return summary


def holdout(base, client: OpenAI):
    ms = base.manifests(TASK)
    selection = load_json(base.adapter_dir(TASK) / "selection.json")
    final_verifier = verifier_dir(TASK) / "selected_verifier.txt"

    if not final_verifier.exists():
        raise SystemExit("missing selected_verifier.txt; run --phase development first")

    baseline = evaluate_with_verifier(
        base, client,
        manifest_path=ms["holdout"], part="holdout",
        condition="baseline",
        adapter=None, verifier_prompt_path=final_verifier,
    )

    selected_result = None
    if selection["decision"] == "ADAPTER":
        adapter = (base.adapter_dir(TASK) / "selected_adapter.txt").read_text(
            encoding="utf-8"
        )
        selected_result = evaluate_with_verifier(
            base, client,
            manifest_path=ms["holdout"], part="holdout",
            condition="selected_adapter",
            adapter=adapter, verifier_prompt_path=final_verifier,
        )

    summary = {
        "arm": ARM,
        "task_name": TASK,
        "status": "HOLDOUT_COMPLETE",
        "model_roles": {
            "agent_model": base.AGENT_MODEL,
            "surrogate_model": base.SURROGATE_MODEL,
            "verifier_optimizer_model": VERIFIER_OPTIMIZER_MODEL,
            "skill_optimizer_model": base.OPTIMIZER_MODEL,
        },
        "selection_decision": selection["decision"],
        "selection_sha256": sha(base.adapter_dir(TASK) / "selection.json"),
        "selected_verifier_sha256": sha(final_verifier),
        "baseline": base.compact(baseline),
        "selected_adapter": base.compact(selected_result) if selected_result else None,
        "no_holdout_feedback": True,
        "optimizer_called_on_holdout": False,
        "verifier_revision_on_holdout": False,
        "skill_revision_on_holdout": False,
    }
    write_json(RESULT_DIR / "holdout" / f"{TASK}_summary.json", summary)
    return summary


def preflight(base):
    v0 = ROOT / "prompts/meta_coevo/surrogate_verifier_v0.txt"
    if not v0.exists():
        raise SystemExit(f"missing verifier v0: {v0}")
    for path in base.manifests(TASK).values():
        if not path.exists():
            raise SystemExit(f"missing manifest: {path}")

    return {
        "arm": ARM,
        "task_name": TASK,
        "base_runner": str(BASE_RUNNER),
        "agent_model": base.AGENT_MODEL,
        "surrogate_model": base.SURROGATE_MODEL,
        "verifier_optimizer_model": VERIFIER_OPTIMIZER_MODEL,
        "skill_optimizer_model": base.OPTIMIZER_MODEL,
        "result_dir": str(RESULT_DIR),
        "trace_dir": str(TRACE_DIR),
        "adapter_dir": str(ADAPTER_ROOT),
        "verifier_dir": str(VERIFIER_ROOT),
        "gt_visible_to_surrogate": False,
        "gt_visible_to_development_verifier_optimizer": True,
        "holdout_feedback": False,
        "skill_evolution": True,
        "verifier_evolution": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["preflight", "development", "holdout"],
        required=True,
    )
    args = parser.parse_args()

    base = load_base()
    info = preflight(base)

    if args.phase == "preflight":
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in model process")

    client = OpenAI()
    result = development(base, client) if args.phase == "development" else holdout(base, client)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
