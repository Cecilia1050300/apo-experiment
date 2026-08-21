#!/usr/bin/env python3
"""
Arm D: Generalization-Gated Rule-Level Skill Evolution
======================================================

Goal
----
Evolve reusable Skill rules without allowing a rule to enter the final Skill
just because it improves the same development fold that generated it.

Protocol
--------
1) Evaluate baseline on Development Fold A and Fold B.
2) Fold A failures -> GPT-5.6 Luna proposes atomic candidate rules.
3) Each A-generated rule is tested on BOTH Fold A and Fold B.
   It is accepted only if it improves the opposite fold (Fold B) without
   regressing either fold or critical answer-format subgroups.
4) Mirror the process: Fold B failures -> candidate rules -> validate on Fold A.
5) Merge only cross-fold-accepted rules.
6) Final combined Skill must again pass a two-fold gate:
   - no Fold A regression
   - no Fold B regression
   - strict gain on at least one fold
   - no critical format regression
7) Frozen Holdout is evaluated only after development selection.
   Holdout never drives rule generation, rule selection, or Skill revision.

Model roles
-----------
- Agent: gpt-4o-mini (fixed model weights)
- Surrogate diagnosis: gpt-5.6-luna with the frozen v0 verifier
- Rule generator / merge optimizer: gpt-5.6-luna
- GT oracle: FactoryBench deterministic scorer (_score_one)

This script reuses the already validated Arm-B runner for FactoryBench
loading/scoring and model-role calls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths / experiment identity
# ---------------------------------------------------------------------------

REPO = Path("/home/training/automatic_prompt_engineer")
ROOT = REPO / "hermes_factorybench_skill_experiment"
BASE_RUNNER = ROOT / "scripts/meta_coevo/run_static_surrogate.py"

ARM = "generalization_gated"
TASK = "m1_factorybench_l123"

RESULT_DIR = ROOT / "results/meta_coevo/gated"
TRACE_DIR = ROOT / "traces/meta_coevo/gated"
SKILL_ROOT = ROOT / "prompts/adapters/meta_coevo_gated"
RULE_ROOT = ROOT / "prompts/meta_coevo/rules/gated"

RULE_MODEL = os.getenv("RULE_MODEL", "gpt-5.6-luna")
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "8192"))
MAX_RULES_PER_SOURCE_FOLD = int(os.getenv("MAX_RULES_PER_SOURCE_FOLD", "3"))

# To reduce unnecessary candidate explosion.
MAX_RULE_CHARS = int(os.getenv("MAX_RULE_CHARS", "900"))


RULE_GENERATOR_SYSTEM = r"""
You are a Skill Rule Generator for FactoryBench Level 1-3 manufacturing tasks.

You receive DEVELOPMENT-only failure evidence from exactly one source fold.

Your job is NOT to rewrite an entire prompt. Propose a small set of ATOMIC,
REUSABLE procedural rules that may generalize to different unseen cases.

A good rule:
- describes one general reasoning/checking procedure;
- is independent of case IDs, dataset IDs, timestamps, exact thresholds,
  exact option letters, gold labels, or memorized signal values;
- can be tested independently;
- is concise enough to append to an agent system prompt;
- addresses a recurring failure mechanism, not one specific answer.

Examples of acceptable abstractions:
- "For a fixed-length temporal window, verify that enough supplied observations
   remain from the proposed start to form the full requested window."
- "For multi-element Boolean outputs, evaluate each proposition independently
   and preserve the original proposition order."

Forbidden:
- copying a development answer;
- embedding a case ID;
- embedding a timestamp/threshold/value copied from a case;
- saying that a particular option is correct;
- mentioning holdout examples, holdout scores, or holdout IDs;
- dataset-specific lookup tables.

Return JSON only:
{
  "source_fold": "A" | "B",
  "rules": [
    {
      "rule_id": "R1",
      "category": "short general category",
      "rule_text": "one atomic reusable instruction",
      "rationale": "why this rule addresses a recurring source-fold failure",
      "source_failure_count": 1
    }
  ],
  "evidence_limitations": ["development evidence limitation"]
}

Return at most the requested maximum number of rules.
If the evidence does not support a reusable rule, return an empty rules list.
""".strip()


MERGE_SYSTEM = r"""
You are consolidating already cross-fold-validated Skill rules.

You must NOT invent new knowledge. You may only:
- remove duplicates,
- merge semantically redundant rules,
- make wording concise,
- preserve the meaning of every accepted rule.

Do not introduce case IDs, exact development answers, case-specific numbers,
dataset-specific lookup tables, or holdout information.

Return JSON only:
{
  "rules": [
    {
      "rule_id": "G1",
      "category": "short category",
      "rule_text": "atomic reusable instruction",
      "provenance_rule_ids": ["A_R1"]
    }
  ],
  "changes": ["deduplication or wording change"]
}
""".strip()


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def load_base():
    if not BASE_RUNNER.exists():
        raise SystemExit(f"missing base runner: {BASE_RUNNER}")

    spec = importlib.util.spec_from_file_location("arm_b_static_runner_for_gated", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to import Arm-B runner")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Isolate all Arm-D outputs.
    mod.ARM = ARM
    mod.RESULT_DIR = RESULT_DIR
    mod.TRACE_DIR = TRACE_DIR
    mod.ADAPTER_ROOT = SKILL_ROOT

    return mod


def rules_dir(task: str) -> Path:
    return RULE_ROOT / task


def skill_dir(task: str) -> Path:
    return SKILL_ROOT / task


def build_skill(rule_texts: list[str]) -> str:
    if not rule_texts:
        return ""

    lines = [
        "Reusable FactoryBench Skill Rules",
        "",
        "Apply the following general procedures only when relevant to the current task.",
        "Do not invent missing data or case-specific facts.",
        "",
    ]
    for i, rule in enumerate(rule_texts, 1):
        lines.append(f"{i}. {rule.strip()}")
    return "\n".join(lines).strip() + "\n"


def format_scores(base, result: dict[str, Any]) -> dict[str, float]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in result["items"]:
        buckets.setdefault(row["answer_format"], []).append(row)
    return {k: base.fixed(v) for k, v in buckets.items()}


def gate_compare(
    base,
    *,
    baseline_source: dict[str, Any],
    baseline_target: dict[str, Any],
    candidate_source: dict[str, Any],
    candidate_target: dict[str, Any],
    require_target_strict_gain: bool,
) -> dict[str, Any]:
    reasons = []

    if candidate_source["ordered_ids"] != baseline_source["ordered_ids"]:
        reasons.append("source ID mismatch")
    if candidate_target["ordered_ids"] != baseline_target["ordered_ids"]:
        reasons.append("target ID mismatch")

    if candidate_source["parse_failures"] > baseline_source["parse_failures"]:
        reasons.append("source parse regression")
    if candidate_target["parse_failures"] > baseline_target["parse_failures"]:
        reasons.append("target parse regression")

    src_base = baseline_source["fixed_cardinality_score"]
    tgt_base = baseline_target["fixed_cardinality_score"]
    src_cand = candidate_source["fixed_cardinality_score"]
    tgt_cand = candidate_target["fixed_cardinality_score"]

    if src_cand < src_base:
        reasons.append("source fold regression")
    if tgt_cand < tgt_base:
        reasons.append("target fold regression")

    if require_target_strict_gain and not (tgt_cand > tgt_base):
        reasons.append("no strict opposite-fold gain")

    base_fmt_source = format_scores(base, baseline_source)
    base_fmt_target = format_scores(base, baseline_target)
    cand_fmt_source = format_scores(base, candidate_source)
    cand_fmt_target = format_scores(base, candidate_target)

    for fmt, value in base_fmt_source.items():
        if cand_fmt_source.get(fmt, float("-inf")) < value:
            reasons.append(f"source critical format regression: {fmt}")
    for fmt, value in base_fmt_target.items():
        if cand_fmt_target.get(fmt, float("-inf")) < value:
            reasons.append(f"target critical format regression: {fmt}")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "source_baseline_score": src_base,
        "source_candidate_score": src_cand,
        "source_delta": src_cand - src_base,
        "target_baseline_score": tgt_base,
        "target_candidate_score": tgt_cand,
        "target_delta": tgt_cand - tgt_base,
        "macro_baseline": (src_base + tgt_base) / 2,
        "macro_candidate": (src_cand + tgt_cand) / 2,
        "macro_delta": ((src_cand + tgt_cand) - (src_base + tgt_base)) / 2,
        "candidate_bytes": 0,
    }


# ---------------------------------------------------------------------------
# Leakage / rule validation
# ---------------------------------------------------------------------------

def collect_case_numbers(result: dict[str, Any]) -> set[str]:
    out = set()
    for row in result["items"]:
        evidence = row.get("development_evidence") or {}
        rendered = str(evidence.get("rendered_input") or "")
        out.update(
            re.findall(
                r"(?<![A-Za-z])(?:\d{4,}|-?\d+\.\d{3,})(?![A-Za-z])",
                rendered,
            )
        )
    return out


def validate_rule_text(
    text: str,
    *,
    source_result: dict[str, Any],
    holdout_ids: list[str],
) -> list[str]:
    errors = []
    low = text.casefold()

    if not text.strip():
        return ["empty rule"]
    if len(text) > MAX_RULE_CHARS:
        errors.append("rule too long")

    # IDs must never appear.
    for row in source_result["items"]:
        if row["id"].casefold() in low:
            errors.append("development ID leak")
            break

    if any(h.casefold() in low for h in holdout_ids):
        errors.append("holdout ID leak")

    # Long gold answer copying is disallowed.
    for row in source_result["items"]:
        evidence = row.get("development_evidence") or {}
        gold = str(evidence.get("reference_answer") or "")
        if len(gold) >= 20 and gold.casefold() in low:
            errors.append("copied gold answer")
            break

    # Case-specific large / high precision values are disallowed.
    numbers = collect_case_numbers(source_result)
    if any(n in text for n in numbers):
        errors.append("case-specific signal value")

    # Explicit answer-key language is suspicious.
    if re.search(r"\b(option|answer)\s+[ABCD]\b", text, flags=re.I):
        errors.append("case-specific option label")

    return errors


# ---------------------------------------------------------------------------
# Rule generation / merge
# ---------------------------------------------------------------------------

def compact_failures(result: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for row in result["items"]:
        # Only source-fold GT failures drive rule generation.
        if row.get("gt_pass") is True:
            continue

        failures.append(
            {
                "id": row["id"],
                "level": row["level"],
                "dataset": row["dataset"],
                "answer_format": row["answer_format"],
                "agent_answer": row["raw_output"],
                "score": row["score"],
                "parse_error": row["parse_error"],
                "surrogate_verdict": row["surrogate_verdict"],
                "surrogate_diagnosis": (row.get("surrogate") or {}).get(
                    "diagnosis", []
                ),
                "surrogate_failed_checks": (row.get("surrogate") or {}).get(
                    "failed_checks", []
                ),
                "development_evidence": row.get("development_evidence"),
            }
        )

    return {
        "condition": result["condition"],
        "fixed_cardinality_score": result["fixed_cardinality_score"],
        "surrogate_summary": result.get("surrogate_summary"),
        "failures": failures,
    }


def call_rule_generator(
    base,
    client: OpenAI,
    *,
    source_fold: str,
    source_result: dict[str, Any],
) -> list[dict[str, Any]]:
    trace_dir = TRACE_DIR / TASK / "rule_generation"
    input_path = trace_dir / f"source_fold_{source_fold}_input.json"
    raw_path = trace_dir / f"source_fold_{source_fold}_raw_output.txt"
    parsed_path = trace_dir / f"source_fold_{source_fold}_parsed_output.json"

    packet = {
        "arm": ARM,
        "task_name": TASK,
        "source_fold": source_fold,
        "maximum_rules": MAX_RULES_PER_SOURCE_FOLD,
        "development_only": True,
        "source_result": compact_failures(source_result),
        "constraints": {
            "atomic_rules_only": True,
            "cross_fold_validation_required": True,
            "holdout_access": False,
            "case_memorization": False,
        },
    }
    write_json(input_path, packet)

    if raw_path.exists() or parsed_path.exists():
        raise RuntimeError(f"existing partial rule-generation trace for Fold {source_fold}")

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=RULE_MODEL,
        messages=[
            {"role": "system", "content": RULE_GENERATOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    "<RULE_GENERATION_INPUT>\n"
                    + json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n</RULE_GENERATION_INPUT>"
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
        errors.append("output is not object")
        parsed = {}

    if parsed.get("source_fold") != source_fold:
        errors.append("source_fold mismatch")

    rules = parsed.get("rules")
    if not isinstance(rules, list):
        errors.append("rules must be list")
        rules = []

    if len(rules) > MAX_RULES_PER_SOURCE_FOLD:
        errors.append("too many rules")

    holdout_ids = [x["id"] for x in load_json(base.manifests(TASK)["holdout"])["items"]]

    normalized_rules = []
    seen_ids = set()
    for i, rule in enumerate(rules, 1):
        if not isinstance(rule, dict):
            errors.append(f"rule {i} not object")
            continue

        rid = str(rule.get("rule_id") or f"R{i}")
        text = str(rule.get("rule_text") or "").strip()
        category = str(rule.get("category") or "uncategorized").strip()
        rationale = str(rule.get("rationale") or "").strip()
        count = rule.get("source_failure_count")

        if rid in seen_ids:
            errors.append(f"duplicate rule_id: {rid}")
        seen_ids.add(rid)

        rule_errors = validate_rule_text(
            text,
            source_result=source_result,
            holdout_ids=holdout_ids,
        )
        if rule_errors:
            errors.extend([f"{rid}: {e}" for e in rule_errors])

        normalized_rules.append(
            {
                "rule_id": f"{source_fold}_{rid}",
                "category": category,
                "rule_text": text,
                "rationale": rationale,
                "source_failure_count": count,
                "source_fold": source_fold,
            }
        )

    role_usage = base.usage(response)
    envelope = {
        **parsed,
        "_trace_validation": {
            "valid": not errors,
            "errors": errors,
            "model": RULE_MODEL,
            "usage": role_usage,
            "cost": base.safe_cost(RULE_MODEL, {**role_usage, "calls": 1}),
            "wall_time_seconds": wall,
            "input_sha256": sha(input_path),
            "raw_output_sha256": sha(raw_path),
        },
        "_normalized_rules": normalized_rules,
    }
    write_json(parsed_path, envelope)

    if errors:
        raise RuntimeError(f"invalid rule-generator output: {errors}")

    for rule in normalized_rules:
        p = rules_dir(TASK) / "candidates" / f"{safe_name(rule['rule_id'])}.json"
        write_json(p, rule)

    return normalized_rules


def merge_accepted_rules(
    base,
    client: OpenAI,
    *,
    accepted_rules: list[dict[str, Any]],
    all_source_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not accepted_rules:
        return []

    # If only one rule survived, no synthesis call is needed.
    if len(accepted_rules) == 1:
        return [
            {
                "rule_id": "G1",
                "category": accepted_rules[0]["category"],
                "rule_text": accepted_rules[0]["rule_text"],
                "provenance_rule_ids": [accepted_rules[0]["rule_id"]],
            }
        ]

    trace_dir = TRACE_DIR / TASK / "merge"
    input_path = trace_dir / "merge_input.json"
    raw_path = trace_dir / "merge_raw_output.txt"
    parsed_path = trace_dir / "merge_parsed_output.json"

    packet = {
        "accepted_rules": accepted_rules,
        "constraints": {
            "invent_new_rules": False,
            "holdout_access": False,
            "case_memorization": False,
        },
    }
    write_json(input_path, packet)

    if raw_path.exists() or parsed_path.exists():
        raise RuntimeError("existing partial merge trace")

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=RULE_MODEL,
        messages=[
            {"role": "system", "content": MERGE_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(packet, indent=2, ensure_ascii=False),
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

    rules = parsed.get("rules") if isinstance(parsed, dict) else None
    if not isinstance(rules, list):
        errors.append("merged rules must be list")
        rules = []

    holdout_ids = [x["id"] for x in load_json(base.manifests(TASK)["holdout"])["items"]]
    for i, rule in enumerate(rules, 1):
        if not isinstance(rule, dict):
            errors.append(f"merged rule {i} not object")
            continue
        text = str(rule.get("rule_text") or "")
        for source_result in all_source_results:
            errors.extend(
                [
                    f"merged rule {i}: {e}"
                    for e in validate_rule_text(
                        text,
                        source_result=source_result,
                        holdout_ids=holdout_ids,
                    )
                ]
            )

    role_usage = base.usage(response)
    envelope = {
        **parsed,
        "_trace_validation": {
            "valid": not errors,
            "errors": errors,
            "model": RULE_MODEL,
            "usage": role_usage,
            "cost": base.safe_cost(RULE_MODEL, {**role_usage, "calls": 1}),
            "wall_time_seconds": wall,
            "input_sha256": sha(input_path),
            "raw_output_sha256": sha(raw_path),
        },
    }
    write_json(parsed_path, envelope)

    if errors:
        raise RuntimeError(f"invalid merge output: {errors}")

    return rules


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(base, client: OpenAI, *, manifest_path: Path, part: str, condition: str, skill: str | None):
    return base.evaluate(
        client,
        TASK,
        manifest_path,
        part,
        condition,
        skill,
    )


def candidate_condition(rule_id: str, fold: str) -> str:
    return f"rule_{safe_name(rule_id)}_fold_{fold.lower()}"


def evaluate_rule_candidate(
    base,
    client: OpenAI,
    *,
    rule: dict[str, Any],
    source_fold: str,
    baseline_a: dict[str, Any],
    baseline_b: dict[str, Any],
) -> dict[str, Any]:
    ms = base.manifests(TASK)
    skill = build_skill([rule["rule_text"]])

    source_manifest = ms["fold_a"] if source_fold == "A" else ms["fold_b"]
    target_manifest = ms["fold_b"] if source_fold == "A" else ms["fold_a"]
    target_fold = "B" if source_fold == "A" else "A"

    source_result = evaluate(
        base,
        client,
        manifest_path=source_manifest,
        part="development",
        condition=candidate_condition(rule["rule_id"], source_fold),
        skill=skill,
    )
    target_result = evaluate(
        base,
        client,
        manifest_path=target_manifest,
        part="development",
        condition=candidate_condition(rule["rule_id"], target_fold),
        skill=skill,
    )

    baseline_source = baseline_a if source_fold == "A" else baseline_b
    baseline_target = baseline_b if source_fold == "A" else baseline_a

    gate = gate_compare(
        base,
        baseline_source=baseline_source,
        baseline_target=baseline_target,
        candidate_source=source_result,
        candidate_target=target_result,
        require_target_strict_gain=True,
    )
    gate["candidate_bytes"] = len(skill.encode())
    gate["rule_id"] = rule["rule_id"]
    gate["source_fold"] = source_fold
    gate["target_fold"] = target_fold
    gate["rule_text"] = rule["rule_text"]
    gate["source_result_condition"] = source_result["condition"]
    gate["target_result_condition"] = target_result["condition"]

    write_json(
        rules_dir(TASK) / "evaluations" / f"{safe_name(rule['rule_id'])}.json",
        gate,
    )
    return gate


def final_skill_gate(
    base,
    *,
    baseline_a: dict[str, Any],
    baseline_b: dict[str, Any],
    final_a: dict[str, Any],
    final_b: dict[str, Any],
    skill_bytes: int,
) -> dict[str, Any]:
    reasons = []

    a0 = baseline_a["fixed_cardinality_score"]
    b0 = baseline_b["fixed_cardinality_score"]
    a1 = final_a["fixed_cardinality_score"]
    b1 = final_b["fixed_cardinality_score"]

    if a1 < a0:
        reasons.append("Fold A regression")
    if b1 < b0:
        reasons.append("Fold B regression")
    if not (a1 > a0 or b1 > b0):
        reasons.append("no strict fold gain")

    base_fmt_a = format_scores(base, baseline_a)
    base_fmt_b = format_scores(base, baseline_b)
    final_fmt_a = format_scores(base, final_a)
    final_fmt_b = format_scores(base, final_b)

    for fmt, score in base_fmt_a.items():
        if final_fmt_a.get(fmt, float("-inf")) < score:
            reasons.append(f"Fold A critical format regression: {fmt}")
    for fmt, score in base_fmt_b.items():
        if final_fmt_b.get(fmt, float("-inf")) < score:
            reasons.append(f"Fold B critical format regression: {fmt}")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "fold_a_baseline": a0,
        "fold_a_final": a1,
        "fold_a_delta": a1 - a0,
        "fold_b_baseline": b0,
        "fold_b_final": b1,
        "fold_b_delta": b1 - b0,
        "macro_baseline": (a0 + b0) / 2,
        "macro_final": (a1 + b1) / 2,
        "macro_delta": ((a1 + b1) - (a0 + b0)) / 2,
        "skill_bytes": skill_bytes,
    }


# ---------------------------------------------------------------------------
# Development protocol
# ---------------------------------------------------------------------------

def development(base, client: OpenAI):
    ms = base.manifests(TASK)

    # Baselines use the same frozen static surrogate v0 as Arm B.
    baseline_a = evaluate(
        base,
        client,
        manifest_path=ms["fold_a"],
        part="development",
        condition="baseline_fold_a",
        skill=None,
    )
    baseline_b = evaluate(
        base,
        client,
        manifest_path=ms["fold_b"],
        part="development",
        condition="baseline_fold_b",
        skill=None,
    )

    # Generate atomic candidate rules independently from each source fold.
    rules_a = call_rule_generator(
        base,
        client,
        source_fold="A",
        source_result=baseline_a,
    )
    rules_b = call_rule_generator(
        base,
        client,
        source_fold="B",
        source_result=baseline_b,
    )

    # Cross-fold test every rule independently.
    evaluations = []
    for rule in rules_a:
        evaluations.append(
            evaluate_rule_candidate(
                base,
                client,
                rule=rule,
                source_fold="A",
                baseline_a=baseline_a,
                baseline_b=baseline_b,
            )
        )
    for rule in rules_b:
        evaluations.append(
            evaluate_rule_candidate(
                base,
                client,
                rule=rule,
                source_fold="B",
                baseline_a=baseline_a,
                baseline_b=baseline_b,
            )
        )

    accepted_ids = {x["rule_id"] for x in evaluations if x["eligible"]}
    accepted_rules = [
        r for r in (rules_a + rules_b) if r["rule_id"] in accepted_ids
    ]

    # Prefer fewer, shorter accepted rules when equivalent by not adding rejected
    # or redundant rules. Merge call only deduplicates wording.
    merged_rules = merge_accepted_rules(
        base,
        client,
        accepted_rules=accepted_rules,
        all_source_results=[baseline_a, baseline_b],
    )

    final_skill = build_skill([r["rule_text"] for r in merged_rules])
    final_skill_path = skill_dir(TASK) / "candidate_final_skill.txt"

    final_a = final_b = None
    final_gate = None

    if final_skill.strip():
        write_new(final_skill_path, final_skill.encode())

        final_a = evaluate(
            base,
            client,
            manifest_path=ms["fold_a"],
            part="development",
            condition="candidate_final_skill_fold_a",
            skill=final_skill,
        )
        final_b = evaluate(
            base,
            client,
            manifest_path=ms["fold_b"],
            part="development",
            condition="candidate_final_skill_fold_b",
            skill=final_skill,
        )

        final_gate = final_skill_gate(
            base,
            baseline_a=baseline_a,
            baseline_b=baseline_b,
            final_a=final_a,
            final_b=final_b,
            skill_bytes=len(final_skill.encode()),
        )
    else:
        final_gate = {
            "eligible": False,
            "reasons": ["no cross-fold-accepted rules"],
            "fold_a_baseline": baseline_a["fixed_cardinality_score"],
            "fold_b_baseline": baseline_b["fixed_cardinality_score"],
            "macro_baseline": (
                baseline_a["fixed_cardinality_score"]
                + baseline_b["fixed_cardinality_score"]
            )
            / 2,
            "skill_bytes": 0,
        }

    selection_path = skill_dir(TASK) / "selection.json"

    if final_gate["eligible"]:
        selected_skill_path = skill_dir(TASK) / "selected_adapter.txt"
        write_new(selected_skill_path, final_skill.encode())
        selection = {
            "arm": ARM,
            "task_name": TASK,
            "decision": "ADAPTER",
            "selected_candidate": "generalization_gated_skill",
            "selected_adapter_sha256": sha(selected_skill_path),
            "accepted_rule_ids": [r["rule_id"] for r in accepted_rules],
            "merged_rule_count": len(merged_rules),
            "final_gate": final_gate,
        }
    else:
        selection = {
            "arm": ARM,
            "task_name": TASK,
            "decision": "NO_ADAPTER",
            "selected_candidate": "baseline",
            "selected_adapter_sha256": None,
            "accepted_rule_ids": [r["rule_id"] for r in accepted_rules],
            "merged_rule_count": len(merged_rules),
            "final_gate": final_gate,
        }

    write_json(selection_path, selection)

    summary = {
        "arm": ARM,
        "task_name": TASK,
        "status": "DEVELOPMENT_COMPLETE",
        "model_roles": {
            "agent_model": base.AGENT_MODEL,
            "surrogate_model": base.SURROGATE_MODEL,
            "rule_model": RULE_MODEL,
        },
        "protocol": {
            "rule_level_evolution": True,
            "cross_fold_gate": True,
            "opposite_fold_strict_gain_required": True,
            "holdout_access": False,
            "surrogate_static": True,
        },
        "baseline_fold_a": base.compact(baseline_a),
        "baseline_fold_b": base.compact(baseline_b),
        "generated_rules": {
            "from_fold_a": rules_a,
            "from_fold_b": rules_b,
        },
        "rule_evaluations": evaluations,
        "accepted_rules": accepted_rules,
        "merged_rules": merged_rules,
        "candidate_final_skill_fold_a": base.compact(final_a) if final_a else None,
        "candidate_final_skill_fold_b": base.compact(final_b) if final_b else None,
        "selection": selection,
    }

    write_json(RESULT_DIR / "development" / f"{TASK}_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Frozen Holdout
# ---------------------------------------------------------------------------

def holdout(base, client: OpenAI):
    ms = base.manifests(TASK)
    selection_path = skill_dir(TASK) / "selection.json"
    if not selection_path.exists():
        raise SystemExit("missing development selection; run --phase development first")

    selection = load_json(selection_path)

    baseline = evaluate(
        base,
        client,
        manifest_path=ms["holdout"],
        part="holdout",
        condition="baseline",
        skill=None,
    )

    selected = None
    if selection["decision"] == "ADAPTER":
        selected_path = skill_dir(TASK) / "selected_adapter.txt"
        skill = selected_path.read_text(encoding="utf-8")
        selected = evaluate(
            base,
            client,
            manifest_path=ms["holdout"],
            part="holdout",
            condition="selected_adapter",
            skill=skill,
        )

    summary = {
        "arm": ARM,
        "task_name": TASK,
        "status": "HOLDOUT_COMPLETE",
        "selection_decision": selection["decision"],
        "selection_sha256": sha(selection_path),
        "baseline": base.compact(baseline),
        "selected_adapter": base.compact(selected) if selected else None,
        "selected_label": (
            "selected_adapter"
            if selected
            else "NO_ADAPTER (baseline reused; no duplicate Skill call)"
        ),
        "no_holdout_feedback": True,
        "rule_generation_on_holdout": False,
        "rule_selection_on_holdout": False,
        "skill_revision_on_holdout": False,
    }

    write_json(RESULT_DIR / "holdout" / f"{TASK}_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Preflight / CLI
# ---------------------------------------------------------------------------

def preflight(base):
    if not BASE_RUNNER.exists():
        raise SystemExit(f"missing base runner: {BASE_RUNNER}")

    for path in base.manifests(TASK).values():
        if not path.exists():
            raise SystemExit(f"missing manifest: {path}")

    if not base.SURROGATE_PATH.exists():
        raise SystemExit(f"missing static surrogate v0: {base.SURROGATE_PATH}")

    return {
        "arm": ARM,
        "task_name": TASK,
        "base_runner": str(BASE_RUNNER),
        "agent_model": base.AGENT_MODEL,
        "surrogate_model": base.SURROGATE_MODEL,
        "rule_model": RULE_MODEL,
        "max_rules_per_source_fold": MAX_RULES_PER_SOURCE_FOLD,
        "result_dir": str(RESULT_DIR),
        "trace_dir": str(TRACE_DIR),
        "skill_dir": str(SKILL_ROOT),
        "rule_dir": str(RULE_ROOT),
        "rule_level_evolution": True,
        "cross_fold_gate": True,
        "opposite_fold_strict_gain_required": True,
        "surrogate_static_for_ablation": True,
        "gt_visible_to_surrogate": False,
        "holdout_feedback": False,
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

    if args.phase == "development":
        result = development(base, client)
    else:
        result = holdout(base, client)

    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
