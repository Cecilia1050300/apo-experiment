#!/usr/bin/env python3
"""
Arm E: Conditioned + Repeated Cross-Fold Rule Evolution
========================================================

Purpose
-------
Improve Skill generalization by combining:

1) Atomic rule generation from one development fold.
2) Explicit applicability conditions (answer-format triggers).
3) Repeated agent-only evaluation to reduce single-run sampling noise.
4) Cross-fold gating on the rule's applicable subset.
5) A final repeated full-fold gate before a Skill is selected.
6) Frozen holdout evaluation only after development selection.

Important design choices
------------------------
- Agent weights are fixed (default: gpt-4o-mini).
- Rule generator is GPT-5.6 Luna.
- Static surrogate v0 is used ONLY for the initial diagnostic baseline that
  creates failure evidence for the Rule Generator.
- Repeated rule testing uses the deterministic FactoryBench GT scorer and does
  NOT call the surrogate, reducing cost and avoiding verifier-induced noise.
- Holdout is never used for rule generation, rule selection, or revision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Paths / identity
# ---------------------------------------------------------------------------

REPO = Path("/home/training/automatic_prompt_engineer")
ROOT = REPO / "hermes_factorybench_skill_experiment"
BASE_RUNNER = ROOT / "scripts/meta_coevo/run_static_surrogate.py"

ARM = "conditioned_repeated_gated"
TASK = "m1_factorybench_l123"

RESULT_DIR = ROOT / "results/meta_coevo/conditioned_repeated"
TRACE_DIR = ROOT / "traces/meta_coevo/conditioned_repeated"
SKILL_ROOT = ROOT / "prompts/adapters/meta_coevo_conditioned_repeated"
RULE_ROOT = ROOT / "prompts/meta_coevo/rules/conditioned_repeated"

RULE_MODEL = os.getenv("RULE_MODEL", "gpt-5.6-luna")
N_REPEATS = int(os.getenv("N_REPEATS", "5"))
MAX_RULES_PER_SOURCE_FOLD = int(os.getenv("MAX_RULES_PER_SOURCE_FOLD", "3"))
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "8192"))
AGENT_CONCURRENCY = int(os.getenv("AGENT_CONCURRENCY", "2"))

SOURCE_NONREG_RATE = float(os.getenv("SOURCE_NONREG_RATE", "0.8"))
TARGET_NONREG_RATE = float(os.getenv("TARGET_NONREG_RATE", "0.8"))
FINAL_NONREG_RATE = float(os.getenv("FINAL_NONREG_RATE", "0.8"))
EPS = 1e-12

ALLOWED_FORMATS = {
    "single_letter_mcq",
    "four_letter_tf",
    "four_letter_ranking",
    "scalar_range",
    "scalar_margin",
    "scalar_exact",
    "tensor_margin",
}


RULE_GENERATOR_SYSTEM = r"""
You are a Skill Rule Generator for FactoryBench Level 1-3 manufacturing tasks.

You receive DEVELOPMENT-only failure evidence from ONE source fold.

Your job is to propose a small number of ATOMIC, REUSABLE Skill rules with an
explicit applicability trigger.

Each rule must have:
- answer_formats: the exact FactoryBench answer format(s) where it applies.
- rule_text: one concise procedural instruction.

The rule must be general and must NOT memorize the source case.

Good example:
{
  "answer_formats": ["four_letter_tf"],
  "rule_text": "Evaluate each proposition independently using its own event interval, aggregation, comparison direction, and threshold, then preserve proposition order."
}

Another good example:
{
  "answer_formats": ["scalar_range"],
  "rule_text": "For a fixed-length forward window, verify that enough supplied observations remain from the proposed start to form the complete requested window."
}

Forbidden:
- case IDs, episode IDs, dataset-specific lookup tables;
- exact development answers;
- exact option labels claimed to be correct;
- timestamps, thresholds, or signal values copied from a source case;
- holdout examples, holdout scores, or holdout IDs;
- vague rules that apply to every task when the failure is format-specific.

Return JSON only:
{
  "source_fold": "A" | "B",
  "rules": [
    {
      "rule_id": "R1",
      "category": "short category",
      "answer_formats": ["four_letter_tf"],
      "rule_text": "atomic reusable instruction",
      "rationale": "why this should generalize beyond the source failure",
      "source_failure_count": 1
    }
  ],
  "evidence_limitations": ["limitation"]
}

Return at most the requested maximum number of rules.
Return an empty rules list if the evidence does not justify a reusable rule.
""".strip()


MERGE_SYSTEM = r"""
You are consolidating cross-fold-validated conditioned Skill rules.

You may ONLY deduplicate or shorten accepted rules.
Do NOT invent a new rule or broaden a rule's applicability trigger.

Preserve each rule's answer_formats trigger.

Return JSON only:
{
  "rules": [
    {
      "rule_id": "G1",
      "category": "short category",
      "answer_formats": ["four_letter_tf"],
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

    spec = importlib.util.spec_from_file_location("arm_b_static_runner_for_arm_e", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to import Arm-B runner")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    mod.ARM = ARM
    mod.RESULT_DIR = RESULT_DIR / "diagnostic"
    mod.TRACE_DIR = TRACE_DIR / "diagnostic"
    mod.ADAPTER_ROOT = SKILL_ROOT / "diagnostic"
    return mod


def rules_dir(task: str) -> Path:
    return RULE_ROOT / task


def skill_dir(task: str) -> Path:
    return SKILL_ROOT / task


def manifests(base) -> dict[str, Path]:
    return base.manifests(TASK)


def build_conditioned_system(rules: list[dict[str, Any]], answer_format: str) -> str | None:
    active = [r for r in rules if answer_format in r["answer_formats"]]
    if not active:
        return None

    lines = [
        "Reusable FactoryBench Skill Rules",
        "",
        f"These rules apply because the required answer format is: {answer_format}.",
        "Use only the procedures below. Do not invent missing facts.",
        "",
    ]
    for i, rule in enumerate(active, 1):
        lines.append(f"{i}. {rule['rule_text'].strip()}")
    return "\n".join(lines).strip() + "\n"


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs))


# ---------------------------------------------------------------------------
# Dataset helpers / scoring
# ---------------------------------------------------------------------------

def source_items(base, manifest_path: Path):
    return base.source_items(manifest_path)


def fixed_score(rows: list[dict[str, Any]]) -> float:
    scores = []
    chances = []
    for row in rows:
        clean = (
            row.get("parse_error") is None
            and isinstance(row.get("score"), (int, float))
            and math.isfinite(float(row["score"]))
        )
        scores.append(float(row["score"]) if clean else 0.0)
        chances.append(float(row.get("chance", 0.0)))

    if not rows:
        raise ValueError("cannot score empty row set")

    mean_chance = sum(chances) / len(chances)
    return (sum(scores) / len(scores) - mean_chance) / (1 - mean_chance)


def grouped_score(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(field) or "unknown"), []).append(row)
    return {k: fixed_score(v) for k, v in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Agent-only evaluation (no surrogate)
# ---------------------------------------------------------------------------

def call_agent(
    base,
    client: OpenAI,
    *,
    item: Any,
    system: str | None,
):
    return base.call_chat(
        client,
        model=base.AGENT_MODEL,
        system=system,
        prompt=base.render_prompt(item),
    )


def evaluate_items_once(
    base,
    client: OpenAI,
    *,
    manifest_path: Path,
    rules: list[dict[str, Any]],
    answer_formats: set[str] | None,
    run_label: str,
) -> dict[str, Any]:
    manifest, items = source_items(base, manifest_path)

    selected = []
    split_lookup = {x["id"]: x["split"] for x in manifest["items"]}
    for item in items:
        if answer_formats is None or item.answer_format.value in answer_formats:
            selected.append(item)

    if not selected:
        return {
            "run_label": run_label,
            "item_count": 0,
            "ordered_ids": [],
            "fixed_cardinality_score": None,
            "parse_failures": 0,
            "by_format": {},
            "items": [],
        }

    outputs = [None] * len(selected)
    with ThreadPoolExecutor(max_workers=AGENT_CONCURRENCY) as pool:
        futures = {}
        for idx, item in enumerate(selected):
            system = build_conditioned_system(rules, item.answer_format.value)
            fut = pool.submit(
                call_agent,
                base,
                client,
                item=item,
                system=system,
            )
            futures[fut] = idx

        for fut in as_completed(futures):
            outputs[futures[fut]] = fut.result()

    rows = []
    for item, (raw, role_usage, latency, transport_error) in zip(selected, outputs):
        scored = base._score_one(item, raw)
        finite = (
            float(scored.score)
            if isinstance(scored.score, (int, float))
            and math.isfinite(float(scored.score))
            else None
        )
        parse_error = scored.parse_error or (
            "non_finite_score" if finite is None else None
        )

        rows.append(
            {
                "id": item.id,
                "level": item.level,
                "dataset": item.dataset,
                "split": split_lookup[item.id],
                "answer_format": item.answer_format.value,
                "raw_output": raw,
                "parsed": scored.parsed,
                "score": finite,
                "chance": scored.chance,
                "parse_error": parse_error,
                "transport_error": transport_error,
                "usage": role_usage,
                "latency_seconds": latency,
                "skill_active": bool(
                    build_conditioned_system(rules, item.answer_format.value)
                ),
            }
        )

    return {
        "run_label": run_label,
        "item_count": len(rows),
        "ordered_ids": [r["id"] for r in rows],
        "fixed_cardinality_score": fixed_score(rows),
        "parse_failures": sum(r["parse_error"] is not None for r in rows),
        "by_format": grouped_score(rows, "answer_format"),
        "items": rows,
    }


def evaluate_repeated(
    base,
    client: OpenAI,
    *,
    manifest_path: Path,
    rules: list[dict[str, Any]],
    answer_formats: set[str] | None,
    condition: str,
) -> dict[str, Any]:
    out_path = RESULT_DIR / "repeated" / f"{safe_name(condition)}.json"
    if out_path.exists():
        return load_json(out_path)

    runs = []
    for i in range(N_REPEATS):
        runs.append(
            evaluate_items_once(
                base,
                client,
                manifest_path=manifest_path,
                rules=rules,
                answer_formats=answer_formats,
                run_label=f"{condition}_run_{i+1}",
            )
        )

    nonempty = [r for r in runs if r["item_count"] > 0]
    scores = [float(r["fixed_cardinality_score"]) for r in nonempty]

    # Mean by-format across repeats.
    all_formats = sorted(
        {
            fmt
            for run in nonempty
            for fmt in run.get("by_format", {}).keys()
        }
    )
    mean_by_format = {}
    for fmt in all_formats:
        vals = [
            run["by_format"][fmt]
            for run in nonempty
            if fmt in run.get("by_format", {})
        ]
        mean_by_format[fmt] = mean(vals)

    payload = {
        "condition": condition,
        "repeat_count": N_REPEATS,
        "answer_formats": sorted(answer_formats) if answer_formats is not None else None,
        "item_count_per_run": nonempty[0]["item_count"] if nonempty else 0,
        "ordered_ids": nonempty[0]["ordered_ids"] if nonempty else [],
        "scores": scores,
        "mean_score": mean(scores) if scores else None,
        "median_score": median(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "mean_by_format": mean_by_format,
        "runs": runs,
    }
    write_json(out_path, payload)
    return payload


# ---------------------------------------------------------------------------
# Repeated comparison / gates
# ---------------------------------------------------------------------------

def compare_repeated(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if baseline["item_count_per_run"] == 0 or candidate["item_count_per_run"] == 0:
        return {
            "valid": False,
            "reason": "no applicable items",
        }

    if baseline["ordered_ids"] != candidate["ordered_ids"]:
        return {
            "valid": False,
            "reason": "ordered IDs differ",
        }

    b = baseline["scores"]
    c = candidate["scores"]
    if len(b) != len(c):
        return {
            "valid": False,
            "reason": "repeat count mismatch",
        }

    deltas = [cv - bv for bv, cv in zip(b, c)]
    nonreg = sum(d >= -EPS for d in deltas) / len(deltas)
    wins = sum(d > EPS for d in deltas) / len(deltas)

    return {
        "valid": True,
        "baseline_scores": b,
        "candidate_scores": c,
        "paired_deltas": deltas,
        "baseline_mean": baseline["mean_score"],
        "candidate_mean": candidate["mean_score"],
        "mean_delta": candidate["mean_score"] - baseline["mean_score"],
        "baseline_median": baseline["median_score"],
        "candidate_median": candidate["median_score"],
        "median_delta": candidate["median_score"] - baseline["median_score"],
        "non_regression_rate": nonreg,
        "win_rate": wins,
    }


def rule_gate(
    *,
    source_cmp: dict[str, Any],
    target_cmp: dict[str, Any],
) -> dict[str, Any]:
    reasons = []

    if not source_cmp.get("valid"):
        reasons.append(f"source invalid: {source_cmp.get('reason')}")
    if not target_cmp.get("valid"):
        reasons.append(f"target invalid: {target_cmp.get('reason')}")

    if reasons:
        return {"eligible": False, "reasons": reasons}

    if source_cmp["mean_delta"] < -EPS:
        reasons.append("source mean regression")
    if source_cmp["non_regression_rate"] + EPS < SOURCE_NONREG_RATE:
        reasons.append("source non-regression rate too low")

    if target_cmp["mean_delta"] <= EPS:
        reasons.append("no positive target mean gain")
    if target_cmp["non_regression_rate"] + EPS < TARGET_NONREG_RATE:
        reasons.append("target non-regression rate too low")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "source": source_cmp,
        "target": target_cmp,
    }


def final_gate(
    *,
    fold_a_cmp: dict[str, Any],
    fold_b_cmp: dict[str, Any],
    baseline_a: dict[str, Any],
    baseline_b: dict[str, Any],
    candidate_a: dict[str, Any],
    candidate_b: dict[str, Any],
) -> dict[str, Any]:
    reasons = []

    for name, cmp in [("Fold A", fold_a_cmp), ("Fold B", fold_b_cmp)]:
        if not cmp.get("valid"):
            reasons.append(f"{name} comparison invalid")
            continue
        if cmp["mean_delta"] < -EPS:
            reasons.append(f"{name} mean regression")
        if cmp["non_regression_rate"] + EPS < FINAL_NONREG_RATE:
            reasons.append(f"{name} non-regression rate too low")

    if (
        fold_a_cmp.get("valid")
        and fold_b_cmp.get("valid")
        and fold_a_cmp["mean_delta"] <= EPS
        and fold_b_cmp["mean_delta"] <= EPS
    ):
        reasons.append("no strict mean gain on either fold")

    # Full-fold mean format regression guard.
    all_formats = set(baseline_a["mean_by_format"]) | set(baseline_b["mean_by_format"])
    for fmt in sorted(all_formats):
        a0 = baseline_a["mean_by_format"].get(fmt)
        a1 = candidate_a["mean_by_format"].get(fmt)
        b0 = baseline_b["mean_by_format"].get(fmt)
        b1 = candidate_b["mean_by_format"].get(fmt)

        if a0 is not None and a1 is not None and a1 < a0 - EPS:
            reasons.append(f"Fold A mean format regression: {fmt}")
        if b0 is not None and b1 is not None and b1 < b0 - EPS:
            reasons.append(f"Fold B mean format regression: {fmt}")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "fold_a": fold_a_cmp,
        "fold_b": fold_b_cmp,
    }


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------

def compact_failures(result: dict[str, Any]) -> dict[str, Any]:
    failures = []
    for row in result["items"]:
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
                "surrogate_verdict": row["surrogate_verdict"],
                "surrogate_diagnosis": (row.get("surrogate") or {}).get("diagnosis", []),
                "surrogate_failed_checks": (row.get("surrogate") or {}).get("failed_checks", []),
                "development_evidence": row.get("development_evidence"),
            }
        )
    return {
        "condition": result["condition"],
        "fixed_cardinality_score": result["fixed_cardinality_score"],
        "failures": failures,
    }


def validate_rule(
    rule: dict[str, Any],
    *,
    source_result: dict[str, Any],
    holdout_ids: list[str],
) -> list[str]:
    errors = []

    text = str(rule.get("rule_text") or "").strip()
    formats = rule.get("answer_formats")

    if not text:
        errors.append("empty rule_text")

    if not isinstance(formats, list) or not formats:
        errors.append("answer_formats must be non-empty list")
        formats = []
    elif any(fmt not in ALLOWED_FORMATS for fmt in formats):
        errors.append("invalid answer format trigger")

    # Require trigger to be grounded in at least one source failure format.
    source_failure_formats = {
        row["answer_format"]
        for row in source_result["items"]
        if row.get("gt_pass") is False
    }
    if formats and not set(formats).intersection(source_failure_formats):
        errors.append("trigger not grounded in source failure format")

    low = text.casefold()

    for row in source_result["items"]:
        if row["id"].casefold() in low:
            errors.append("development ID leak")
            break

        evidence = row.get("development_evidence") or {}
        gold = str(evidence.get("reference_answer") or "")
        if len(gold) >= 20 and gold.casefold() in low:
            errors.append("copied gold answer")
            break

        rendered = str(evidence.get("rendered_input") or "")
        numbers = set(
            re.findall(
                r"(?<![A-Za-z])(?:\d{4,}|-?\d+\.\d{3,})(?![A-Za-z])",
                rendered,
            )
        )
        if any(n in text for n in numbers):
            errors.append("case-specific signal value")
            break

    if any(h.casefold() in low for h in holdout_ids):
        errors.append("holdout ID leak")

    if re.search(r"\b(option|answer)\s+[ABCD]\b", text, flags=re.I):
        errors.append("case-specific option label")

    return errors


def call_rule_generator(
    base,
    client: OpenAI,
    *,
    source_fold: str,
    source_result: dict[str, Any],
) -> list[dict[str, Any]]:
    tdir = TRACE_DIR / TASK / "rule_generation"
    input_path = tdir / f"source_fold_{source_fold}_input.json"
    raw_path = tdir / f"source_fold_{source_fold}_raw_output.txt"
    parsed_path = tdir / f"source_fold_{source_fold}_parsed_output.json"

    packet = {
        "arm": ARM,
        "task_name": TASK,
        "source_fold": source_fold,
        "maximum_rules": MAX_RULES_PER_SOURCE_FOLD,
        "source_result": compact_failures(source_result),
        "constraints": {
            "conditioned_rules": True,
            "cross_fold_validation_required": True,
            "repeated_evaluation_required": True,
            "holdout_access": False,
            "case_memorization": False,
        },
    }
    write_json(input_path, packet)

    if raw_path.exists() or parsed_path.exists():
        raise RuntimeError(f"existing partial rule trace: Fold {source_fold}")

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
        errors.append("output not object")
        parsed = {}

    if parsed.get("source_fold") != source_fold:
        errors.append("source_fold mismatch")

    rules = parsed.get("rules")
    if not isinstance(rules, list):
        errors.append("rules must be list")
        rules = []

    if len(rules) > MAX_RULES_PER_SOURCE_FOLD:
        errors.append("too many rules")

    holdout_ids = [
        x["id"] for x in load_json(manifests(base)["holdout"])["items"]
    ]

    normalized = []
    for i, rule in enumerate(rules, 1):
        if not isinstance(rule, dict):
            errors.append(f"rule {i} not object")
            continue

        rid = str(rule.get("rule_id") or f"R{i}")
        norm = {
            "rule_id": f"{source_fold}_{rid}",
            "category": str(rule.get("category") or "uncategorized"),
            "answer_formats": list(dict.fromkeys(rule.get("answer_formats") or [])),
            "rule_text": str(rule.get("rule_text") or "").strip(),
            "rationale": str(rule.get("rationale") or "").strip(),
            "source_failure_count": rule.get("source_failure_count"),
            "source_fold": source_fold,
        }

        for err in validate_rule(
            norm,
            source_result=source_result,
            holdout_ids=holdout_ids,
        ):
            errors.append(f"{norm['rule_id']}: {err}")

        normalized.append(norm)

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
        "_normalized_rules": normalized,
    }
    write_json(parsed_path, envelope)

    if errors:
        raise RuntimeError(f"invalid rule-generator output: {errors}")

    for rule in normalized:
        write_json(
            rules_dir(TASK) / "candidates" / f"{safe_name(rule['rule_id'])}.json",
            rule,
        )

    return normalized


# ---------------------------------------------------------------------------
# Merge accepted rules
# ---------------------------------------------------------------------------

def merge_accepted_rules(
    base,
    client: OpenAI,
    *,
    accepted_rules: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not accepted_rules:
        return []
    if len(accepted_rules) == 1:
        r = accepted_rules[0]
        return [
            {
                "rule_id": "G1",
                "category": r["category"],
                "answer_formats": r["answer_formats"],
                "rule_text": r["rule_text"],
                "provenance_rule_ids": [r["rule_id"]],
            }
        ]

    tdir = TRACE_DIR / TASK / "merge"
    input_path = tdir / "merge_input.json"
    raw_path = tdir / "merge_raw_output.txt"
    parsed_path = tdir / "merge_parsed_output.json"

    packet = {
        "accepted_rules": accepted_rules,
        "constraints": {
            "do_not_broaden_triggers": True,
            "invent_new_rules": False,
            "holdout_access": False,
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
            {"role": "user", "content": json.dumps(packet, indent=2, ensure_ascii=False)},
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

    accepted_by_id = {r["rule_id"]: r for r in accepted_rules}

    for i, r in enumerate(rules, 1):
        if not isinstance(r, dict):
            errors.append(f"merged rule {i} not object")
            continue
        prov = r.get("provenance_rule_ids")
        fmts = r.get("answer_formats")
        if not isinstance(prov, list) or not prov:
            errors.append(f"merged rule {i}: missing provenance")
            continue

        source_formats = set()
        for pid in prov:
            if pid not in accepted_by_id:
                errors.append(f"merged rule {i}: unknown provenance {pid}")
                continue
            source_formats.update(accepted_by_id[pid]["answer_formats"])

        if set(fmts or []) != source_formats:
            errors.append(f"merged rule {i}: applicability trigger broadened/changed")

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
        },
    }
    write_json(parsed_path, envelope)

    if errors:
        raise RuntimeError(f"invalid merge output: {errors}")

    return rules


# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

def development(base, client: OpenAI):
    ms = manifests(base)

    # One diagnostic run per fold with static surrogate v0.
    diagnostic_a = base.evaluate(
        client,
        TASK,
        ms["fold_a"],
        "development",
        "diagnostic_baseline_fold_a",
        None,
    )
    diagnostic_b = base.evaluate(
        client,
        TASK,
        ms["fold_b"],
        "development",
        "diagnostic_baseline_fold_b",
        None,
    )

    rules_a = call_rule_generator(
        base, client,
        source_fold="A",
        source_result=diagnostic_a,
    )
    rules_b = call_rule_generator(
        base, client,
        source_fold="B",
        source_result=diagnostic_b,
    )
    all_rules = rules_a + rules_b

    # Cache repeated baseline by (fold, trigger formats).
    baseline_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}

    def get_baseline(fold: str, formats: list[str]):
        key = (fold, tuple(sorted(formats)))
        if key not in baseline_cache:
            baseline_cache[key] = evaluate_repeated(
                base,
                client,
                manifest_path=ms["fold_a"] if fold == "A" else ms["fold_b"],
                rules=[],
                answer_formats=set(formats),
                condition=f"baseline_fold_{fold.lower()}__formats__{'_'.join(sorted(formats))}",
            )
        return baseline_cache[key]

    evaluations = []
    accepted_rules = []

    for rule in all_rules:
        source_fold = rule["source_fold"]
        target_fold = "B" if source_fold == "A" else "A"
        formats = rule["answer_formats"]

        source_base = get_baseline(source_fold, formats)
        target_base = get_baseline(target_fold, formats)

        source_candidate = evaluate_repeated(
            base,
            client,
            manifest_path=ms["fold_a"] if source_fold == "A" else ms["fold_b"],
            rules=[rule],
            answer_formats=set(formats),
            condition=f"{rule['rule_id']}__source_{source_fold}",
        )
        target_candidate = evaluate_repeated(
            base,
            client,
            manifest_path=ms["fold_b"] if source_fold == "A" else ms["fold_a"],
            rules=[rule],
            answer_formats=set(formats),
            condition=f"{rule['rule_id']}__target_{target_fold}",
        )

        source_cmp = compare_repeated(source_base, source_candidate)
        target_cmp = compare_repeated(target_base, target_candidate)
        gate = rule_gate(source_cmp=source_cmp, target_cmp=target_cmp)

        record = {
            "rule_id": rule["rule_id"],
            "source_fold": source_fold,
            "target_fold": target_fold,
            "answer_formats": formats,
            "rule_text": rule["rule_text"],
            "eligible": gate["eligible"],
            "reasons": gate["reasons"],
            "source_comparison": gate.get("source"),
            "target_comparison": gate.get("target"),
        }
        evaluations.append(record)

        write_json(
            rules_dir(TASK) / "evaluations" / f"{safe_name(rule['rule_id'])}.json",
            record,
        )

        if gate["eligible"]:
            accepted_rules.append(rule)

    merged_rules = merge_accepted_rules(
        base,
        client,
        accepted_rules=accepted_rules,
    )

    # Final full-fold repeated gate.
    full_baseline_a = evaluate_repeated(
        base,
        client,
        manifest_path=ms["fold_a"],
        rules=[],
        answer_formats=None,
        condition="final_gate_baseline_fold_a",
    )
    full_baseline_b = evaluate_repeated(
        base,
        client,
        manifest_path=ms["fold_b"],
        rules=[],
        answer_formats=None,
        condition="final_gate_baseline_fold_b",
    )

    final_a = final_b = None
    final_selection_gate = {
        "eligible": False,
        "reasons": ["no cross-fold-accepted rules"],
    }

    if merged_rules:
        # Store a human-readable final conditioned Skill.
        skill_payload = {
            "arm": ARM,
            "task_name": TASK,
            "rules": merged_rules,
        }
        write_json(skill_dir(TASK) / "candidate_final_skill.json", skill_payload)

        final_a = evaluate_repeated(
            base,
            client,
            manifest_path=ms["fold_a"],
            rules=merged_rules,
            answer_formats=None,
            condition="candidate_final_skill_fold_a",
        )
        final_b = evaluate_repeated(
            base,
            client,
            manifest_path=ms["fold_b"],
            rules=merged_rules,
            answer_formats=None,
            condition="candidate_final_skill_fold_b",
        )

        cmp_a = compare_repeated(full_baseline_a, final_a)
        cmp_b = compare_repeated(full_baseline_b, final_b)

        final_selection_gate = final_gate(
            fold_a_cmp=cmp_a,
            fold_b_cmp=cmp_b,
            baseline_a=full_baseline_a,
            baseline_b=full_baseline_b,
            candidate_a=final_a,
            candidate_b=final_b,
        )

    selection_path = skill_dir(TASK) / "selection.json"

    if final_selection_gate["eligible"]:
        selected_path = skill_dir(TASK) / "selected_adapter.json"
        write_json(
            selected_path,
            {
                "arm": ARM,
                "task_name": TASK,
                "rules": merged_rules,
            },
        )
        selection = {
            "arm": ARM,
            "task_name": TASK,
            "decision": "ADAPTER",
            "selected_candidate": "conditioned_repeated_skill",
            "selected_adapter_sha256": sha(selected_path),
            "accepted_rule_ids": [r["rule_id"] for r in accepted_rules],
            "merged_rule_count": len(merged_rules),
            "final_gate": final_selection_gate,
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
            "final_gate": final_selection_gate,
        }

    write_json(selection_path, selection)

    summary = {
        "arm": ARM,
        "task_name": TASK,
        "status": "DEVELOPMENT_COMPLETE",
        "model_roles": {
            "agent_model": base.AGENT_MODEL,
            "diagnostic_surrogate_model": base.SURROGATE_MODEL,
            "rule_model": RULE_MODEL,
        },
        "protocol": {
            "conditioned_rules": True,
            "repeated_evaluation": True,
            "repeat_count": N_REPEATS,
            "source_nonreg_rate_threshold": SOURCE_NONREG_RATE,
            "target_nonreg_rate_threshold": TARGET_NONREG_RATE,
            "final_nonreg_rate_threshold": FINAL_NONREG_RATE,
            "surrogate_used_only_for_initial_diagnosis": True,
            "gt_visible_to_surrogate": False,
            "holdout_access": False,
        },
        "diagnostic_baseline_fold_a": base.compact(diagnostic_a),
        "diagnostic_baseline_fold_b": base.compact(diagnostic_b),
        "generated_rules": {
            "from_fold_a": rules_a,
            "from_fold_b": rules_b,
        },
        "rule_evaluations": evaluations,
        "accepted_rules": accepted_rules,
        "merged_rules": merged_rules,
        "final_gate_baseline_fold_a": full_baseline_a,
        "final_gate_baseline_fold_b": full_baseline_b,
        "candidate_final_skill_fold_a": final_a,
        "candidate_final_skill_fold_b": final_b,
        "selection": selection,
    }

    write_json(
        RESULT_DIR / "development" / f"{TASK}_summary.json",
        summary,
    )
    return summary


# ---------------------------------------------------------------------------
# Frozen holdout
# ---------------------------------------------------------------------------

def holdout(base, client: OpenAI):
    ms = manifests(base)
    selection_path = skill_dir(TASK) / "selection.json"
    if not selection_path.exists():
        raise SystemExit("missing selection; run --phase development first")

    selection = load_json(selection_path)

    baseline = evaluate_repeated(
        base,
        client,
        manifest_path=ms["holdout"],
        rules=[],
        answer_formats=None,
        condition="holdout_baseline",
    )

    selected = None
    holdout_cmp = None

    if selection["decision"] == "ADAPTER":
        selected_payload = load_json(skill_dir(TASK) / "selected_adapter.json")
        selected_rules = selected_payload["rules"]

        selected = evaluate_repeated(
            base,
            client,
            manifest_path=ms["holdout"],
            rules=selected_rules,
            answer_formats=None,
            condition="holdout_selected_adapter",
        )
        holdout_cmp = compare_repeated(baseline, selected)

    summary = {
        "arm": ARM,
        "task_name": TASK,
        "status": "HOLDOUT_COMPLETE",
        "selection_decision": selection["decision"],
        "selection_sha256": sha(selection_path),
        "repeat_count": N_REPEATS,
        "baseline": baseline,
        "selected_adapter": selected,
        "holdout_comparison": holdout_cmp,
        "no_holdout_feedback": True,
        "rule_generation_on_holdout": False,
        "rule_selection_on_holdout": False,
        "skill_revision_on_holdout": False,
        "surrogate_called_on_holdout": False,
    }

    write_json(
        RESULT_DIR / "holdout" / f"{TASK}_summary.json",
        summary,
    )
    return summary


# ---------------------------------------------------------------------------
# Preflight / CLI
# ---------------------------------------------------------------------------

def preflight(base):
    for path in manifests(base).values():
        if not path.exists():
            raise SystemExit(f"missing manifest: {path}")

    return {
        "arm": ARM,
        "task_name": TASK,
        "agent_model": base.AGENT_MODEL,
        "diagnostic_surrogate_model": base.SURROGATE_MODEL,
        "rule_model": RULE_MODEL,
        "repeat_count": N_REPEATS,
        "max_rules_per_source_fold": MAX_RULES_PER_SOURCE_FOLD,
        "source_nonreg_rate_threshold": SOURCE_NONREG_RATE,
        "target_nonreg_rate_threshold": TARGET_NONREG_RATE,
        "final_nonreg_rate_threshold": FINAL_NONREG_RATE,
        "conditioned_rules": True,
        "repeated_agent_only_rule_testing": True,
        "surrogate_used_only_for_initial_diagnosis": True,
        "gt_visible_to_surrogate": False,
        "holdout_feedback": False,
        "result_dir": str(RESULT_DIR),
        "trace_dir": str(TRACE_DIR),
        "skill_dir": str(SKILL_ROOT),
        "rule_dir": str(RULE_ROOT),
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

    result = (
        development(base, client)
        if args.phase == "development"
        else holdout(base, client)
    )

    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
