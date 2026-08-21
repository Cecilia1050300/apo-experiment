#!/usr/bin/env python3
"""
Arm F: Format-Stratified Leave-One-Case-Out Rule Evolution
===========================================================

Goal
----
Find reusable Skill rules that generalize across DIFFERENT CASES of the SAME
FactoryBench answer format, instead of relying on Fold A -> Fold B when the
target fold may contain no applicable task of that format.

Core protocol
-------------
1) Pool development cases from Fold A + Fold B.
2) Group cases by answer_format.
3) For each format with >= 2 development cases:
   - choose one GT-failed case as a source case;
   - generate up to K atomic rules from that single case;
   - each rule is conditioned on that answer_format only;
   - repeated-test the rule on the source case and on ALL other same-format
     development cases;
   - accept only if source does not regress and same-format validation cases
     improve on average with high non-regression rate.
4) Deduplicate accepted rules.
5) Repeated-test the combined conditioned Skill on ALL development cases.
6) Select the Skill only if the final development gate passes.
7) Frozen holdout is evaluated only after development selection.

This is a mechanism smoke test for cross-case generalization.
Holdout never participates in rule generation or selection.
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

ARM = "format_stratified_loocv"
TASK = "m1_factorybench_l123"

RESULT_DIR = ROOT / "results/meta_coevo/format_loocv"
TRACE_DIR = ROOT / "traces/meta_coevo/format_loocv"
SKILL_ROOT = ROOT / "prompts/adapters/meta_coevo_format_loocv"
RULE_ROOT = ROOT / "prompts/meta_coevo/rules/format_loocv"

RULE_MODEL = os.getenv("RULE_MODEL", "gpt-5.6-luna")
N_REPEATS = int(os.getenv("N_REPEATS", "5"))
MAX_RULES_PER_SOURCE_CASE = int(os.getenv("MAX_RULES_PER_SOURCE_CASE", "2"))
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

You receive exactly ONE DEVELOPMENT source case that the fixed agent answered
incorrectly. The generated rule will be evaluated on OTHER DEVELOPMENT cases of
the SAME answer format.

Propose a small number of ATOMIC, REUSABLE procedural rules.

Each rule MUST:
- apply only to the supplied answer_format;
- describe one general reasoning/checking procedure;
- be useful on other cases of the same answer format;
- avoid memorizing the source answer, case identity, timestamp, threshold, or
  signal value.

Good:
"For a fixed-length forward window, count observations from the proposed start
and verify the full requested window is available."

Bad:
"The correct start is 1630."
"Choose option C."
"For this FactoryWave case ..."

Forbidden:
- case IDs / dataset IDs as lookup rules;
- exact development answers;
- copied source timestamps / thresholds / high-precision signal values;
- answer-key option letters;
- holdout examples, scores, or IDs.

Return JSON only:
{
  "source_case_id": "the provided source case id",
  "answer_format": "the provided answer format",
  "rules": [
    {
      "rule_id": "R1",
      "category": "short category",
      "rule_text": "one atomic reusable instruction",
      "rationale": "why this may transfer to other same-format cases"
    }
  ],
  "evidence_limitations": ["limitation"]
}

Return at most the requested maximum number of rules.
Return an empty rules list if the single source case does not justify a
reusable rule.
""".strip()


MERGE_SYSTEM = r"""
You are consolidating already validated same-format Skill rules.

Do NOT invent new rules.
Do NOT broaden applicability to another answer format.
Only deduplicate or shorten semantically redundant rules.

Return JSON only:
{
  "rules": [
    {
      "rule_id": "G1",
      "answer_format": "four_letter_tf",
      "category": "short category",
      "rule_text": "atomic reusable instruction",
      "provenance_rule_ids": ["SRC_R1"]
    }
  ],
  "changes": ["deduplication or wording change"]
}
""".strip()


# ---------------------------------------------------------------------------
# Helpers
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


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs))


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def load_base():
    if not BASE_RUNNER.exists():
        raise SystemExit(f"missing base runner: {BASE_RUNNER}")

    spec = importlib.util.spec_from_file_location("arm_b_for_format_loocv", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to import base runner")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Keep Arm-F diagnostic outputs isolated.
    mod.ARM = ARM
    mod.RESULT_DIR = RESULT_DIR / "diagnostic"
    mod.TRACE_DIR = TRACE_DIR / "diagnostic"
    mod.ADAPTER_ROOT = SKILL_ROOT / "diagnostic"
    return mod


def manifests(base) -> dict[str, Path]:
    return base.manifests(TASK)


def rule_dir() -> Path:
    return RULE_ROOT / TASK


def skill_dir() -> Path:
    return SKILL_ROOT / TASK


def build_conditioned_system(rules: list[dict[str, Any]], answer_format: str) -> str | None:
    active = [r for r in rules if r["answer_format"] == answer_format]
    if not active:
        return None

    lines = [
        "Reusable FactoryBench Skill Rules",
        "",
        f"These procedures apply only because the answer format is {answer_format}.",
        "Do not invent missing facts or case-specific values.",
        "",
    ]
    for i, r in enumerate(active, 1):
        lines.append(f"{i}. {r['rule_text'].strip()}")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Development pool
# ---------------------------------------------------------------------------

def load_dev_pool(base):
    pool = []
    seen = set()

    for fold_name, manifest_path in [
        ("A", manifests(base)["fold_a"]),
        ("B", manifests(base)["fold_b"]),
    ]:
        manifest, items = base.source_items(manifest_path)
        split_lookup = {x["id"]: x["split"] for x in manifest["items"]}

        for item in items:
            if item.id in seen:
                raise RuntimeError(f"duplicate dev case across folds: {item.id}")
            seen.add(item.id)
            pool.append(
                {
                    "fold": fold_name,
                    "split": split_lookup[item.id],
                    "item": item,
                }
            )

    return pool


def by_format_pool(pool):
    out: dict[str, list[dict[str, Any]]] = {}
    for rec in pool:
        fmt = rec["item"].answer_format.value
        out.setdefault(fmt, []).append(rec)
    return out


# ---------------------------------------------------------------------------
# Scoring / agent-only repeated eval
# ---------------------------------------------------------------------------

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
        raise ValueError("empty row set")

    mean_chance = sum(chances) / len(chances)
    return (sum(scores) / len(scores) - mean_chance) / (1 - mean_chance)


def grouped_score(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row[field]), []).append(row)
    return {k: fixed_score(v) for k, v in sorted(buckets.items())}


def call_agent(base, client, *, item, system):
    return base.call_chat(
        client,
        model=base.AGENT_MODEL,
        system=system,
        prompt=base.render_prompt(item),
    )


def evaluate_records_once(
    base,
    client,
    *,
    records: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    run_label: str,
) -> dict[str, Any]:
    if not records:
        return {
            "run_label": run_label,
            "item_count": 0,
            "ordered_ids": [],
            "fixed_cardinality_score": None,
            "parse_failures": 0,
            "by_format": {},
            "items": [],
        }

    outputs = [None] * len(records)
    with ThreadPoolExecutor(max_workers=AGENT_CONCURRENCY) as pool:
        futures = {}
        for idx, rec in enumerate(records):
            item = rec["item"]
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
    for rec, out in zip(records, outputs):
        item = rec["item"]
        raw, usage, latency, transport_error = out
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
                "fold": rec["fold"],
                "split": rec["split"],
                "level": item.level,
                "dataset": item.dataset,
                "answer_format": item.answer_format.value,
                "raw_output": raw,
                "parsed": scored.parsed,
                "score": finite,
                "chance": scored.chance,
                "parse_error": parse_error,
                "transport_error": transport_error,
                "usage": usage,
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


def evaluate_repeated_records(
    base,
    client,
    *,
    records: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    condition: str,
) -> dict[str, Any]:
    path = RESULT_DIR / "repeated" / f"{safe_name(condition)}.json"
    if path.exists():
        return load_json(path)

    runs = [
        evaluate_records_once(
            base,
            client,
            records=records,
            rules=rules,
            run_label=f"{condition}_run_{i+1}",
        )
        for i in range(N_REPEATS)
    ]

    scores = [float(r["fixed_cardinality_score"]) for r in runs if r["item_count"]]
    all_formats = sorted(
        {fmt for run in runs for fmt in run.get("by_format", {})}
    )
    mean_by_format = {}
    for fmt in all_formats:
        vals = [run["by_format"][fmt] for run in runs if fmt in run["by_format"]]
        mean_by_format[fmt] = mean(vals)

    payload = {
        "condition": condition,
        "repeat_count": N_REPEATS,
        "item_count_per_run": runs[0]["item_count"] if runs else 0,
        "ordered_ids": runs[0]["ordered_ids"] if runs else [],
        "scores": scores,
        "mean_score": mean(scores) if scores else None,
        "median_score": median(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "mean_by_format": mean_by_format,
        "runs": runs,
    }
    write_json(path, payload)
    return payload


def compare_repeated(base_res, cand_res):
    if base_res["item_count_per_run"] == 0 or cand_res["item_count_per_run"] == 0:
        return {"valid": False, "reason": "no items"}

    if base_res["ordered_ids"] != cand_res["ordered_ids"]:
        return {"valid": False, "reason": "ordered IDs differ"}

    b = base_res["scores"]
    c = cand_res["scores"]
    if len(b) != len(c):
        return {"valid": False, "reason": "repeat count mismatch"}

    deltas = [cv - bv for bv, cv in zip(b, c)]
    return {
        "valid": True,
        "baseline_scores": b,
        "candidate_scores": c,
        "paired_deltas": deltas,
        "baseline_mean": base_res["mean_score"],
        "candidate_mean": cand_res["mean_score"],
        "mean_delta": cand_res["mean_score"] - base_res["mean_score"],
        "baseline_median": base_res["median_score"],
        "candidate_median": cand_res["median_score"],
        "median_delta": cand_res["median_score"] - base_res["median_score"],
        "non_regression_rate": sum(d >= -EPS for d in deltas) / len(deltas),
        "win_rate": sum(d > EPS for d in deltas) / len(deltas),
    }


def rule_gate(source_cmp, target_cmp):
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
        reasons.append("no positive same-format target mean gain")
    if target_cmp["non_regression_rate"] + EPS < TARGET_NONREG_RATE:
        reasons.append("target non-regression rate too low")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "source": source_cmp,
        "target": target_cmp,
    }


# ---------------------------------------------------------------------------
# Initial diagnostic cases
# ---------------------------------------------------------------------------

def diagnostic_dev(base, client):
    ms = manifests(base)
    a = base.evaluate(
        client, TASK, ms["fold_a"], "development",
        "diagnostic_baseline_fold_a", None
    )
    b = base.evaluate(
        client, TASK, ms["fold_b"], "development",
        "diagnostic_baseline_fold_b", None
    )
    rows = []
    for fold, result in [("A", a), ("B", b)]:
        for row in result["items"]:
            rows.append({**row, "_fold": fold})
    return a, b, rows


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------

def source_case_packet(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "fold": row["_fold"],
        "level": row["level"],
        "dataset": row["dataset"],
        "answer_format": row["answer_format"],
        "agent_answer": row["raw_output"],
        "score": row["score"],
        "gt_pass": row["gt_pass"],
        "surrogate_verdict": row["surrogate_verdict"],
        "surrogate_diagnosis": (row.get("surrogate") or {}).get("diagnosis", []),
        "surrogate_failed_checks": (row.get("surrogate") or {}).get("failed_checks", []),
        "development_evidence": row.get("development_evidence"),
    }


def validate_rule_text(text: str, row: dict[str, Any], holdout_ids: list[str]) -> list[str]:
    errors = []
    low = text.casefold()

    if not text.strip():
        errors.append("empty rule")

    if row["id"].casefold() in low:
        errors.append("development ID leak")

    if any(h.casefold() in low for h in holdout_ids):
        errors.append("holdout ID leak")

    evidence = row.get("development_evidence") or {}
    gold = str(evidence.get("reference_answer") or "")
    if len(gold) >= 20 and gold.casefold() in low:
        errors.append("copied gold answer")

    rendered = str(evidence.get("rendered_input") or "")
    numbers = set(
        re.findall(
            r"(?<![A-Za-z])(?:\d{4,}|-?\d+\.\d{3,})(?![A-Za-z])",
            rendered,
        )
    )
    if any(n in text for n in numbers):
        errors.append("case-specific signal value")

    if re.search(r"\b(option|answer)\s+[ABCD]\b", text, flags=re.I):
        errors.append("case-specific option label")

    return errors


def call_rule_generator(base, client, *, source_row):
    sid = source_row["id"]
    fmt = source_row["answer_format"]

    tdir = TRACE_DIR / TASK / "rule_generation"
    prefix = safe_name(f"{fmt}__{sid}")
    input_path = tdir / f"{prefix}_input.json"
    raw_path = tdir / f"{prefix}_raw_output.txt"
    parsed_path = tdir / f"{prefix}_parsed_output.json"

    packet = {
        "arm": ARM,
        "task_name": TASK,
        "source_case": source_case_packet(source_row),
        "maximum_rules": MAX_RULES_PER_SOURCE_CASE,
        "validation_design": {
            "same_answer_format_only": True,
            "leave_one_case_out": True,
            "repeated_evaluation": True,
            "holdout_access": False,
        },
    }
    write_json(input_path, packet)

    if raw_path.exists() or parsed_path.exists():
        raise RuntimeError(f"existing partial rule-generation trace: {sid}")

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=RULE_MODEL,
        messages=[
            {"role": "system", "content": RULE_GENERATOR_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(packet, indent=2, ensure_ascii=False, allow_nan=False),
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

    if parsed.get("source_case_id") != sid:
        errors.append("source_case_id mismatch")
    if parsed.get("answer_format") != fmt:
        errors.append("answer_format mismatch")

    rules = parsed.get("rules")
    if not isinstance(rules, list):
        errors.append("rules must be list")
        rules = []

    if len(rules) > MAX_RULES_PER_SOURCE_CASE:
        errors.append("too many rules")

    holdout_ids = [
        x["id"] for x in load_json(manifests(base)["holdout"])["items"]
    ]

    normalized = []
    for i, r in enumerate(rules, 1):
        if not isinstance(r, dict):
            errors.append(f"rule {i} not object")
            continue

        rid = str(r.get("rule_id") or f"R{i}")
        text = str(r.get("rule_text") or "").strip()
        for err in validate_rule_text(text, source_row, holdout_ids):
            errors.append(f"{rid}: {err}")

        normalized.append(
            {
                "rule_id": f"{safe_name(fmt)}__{sid[:8]}__{rid}",
                "source_case_id": sid,
                "source_fold": source_row["_fold"],
                "answer_format": fmt,
                "category": str(r.get("category") or "uncategorized"),
                "rule_text": text,
                "rationale": str(r.get("rationale") or ""),
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
        },
        "_normalized_rules": normalized,
    }
    write_json(parsed_path, envelope)

    if errors:
        raise RuntimeError(f"invalid rule-generator output for {sid}: {errors}")

    for r in normalized:
        write_json(
            rule_dir() / "candidates" / f"{safe_name(r['rule_id'])}.json",
            r,
        )

    return normalized


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_rules(base, client, accepted_rules):
    if not accepted_rules:
        return []
    if len(accepted_rules) == 1:
        r = accepted_rules[0]
        return [{
            "rule_id": "G1",
            "answer_format": r["answer_format"],
            "category": r["category"],
            "rule_text": r["rule_text"],
            "provenance_rule_ids": [r["rule_id"]],
        }]

    tdir = TRACE_DIR / TASK / "merge"
    input_path = tdir / "merge_input.json"
    raw_path = tdir / "merge_raw_output.txt"
    parsed_path = tdir / "merge_parsed_output.json"

    packet = {
        "accepted_rules": accepted_rules,
        "constraints": {
            "preserve_answer_format": True,
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
        prov = r.get("provenance_rule_ids")
        fmt = r.get("answer_format")
        if not isinstance(prov, list) or not prov:
            errors.append(f"merged rule {i}: missing provenance")
            continue
        source_formats = {
            accepted_by_id[p]["answer_format"]
            for p in prov
            if p in accepted_by_id
        }
        if len(source_formats) != 1 or fmt not in source_formats:
            errors.append(f"merged rule {i}: format broadened/changed")

    role_usage = base.usage(response)
    write_json(
        parsed_path,
        {
            **parsed,
            "_trace_validation": {
                "valid": not errors,
                "errors": errors,
                "model": RULE_MODEL,
                "usage": role_usage,
                "cost": base.safe_cost(RULE_MODEL, {**role_usage, "calls": 1}),
                "wall_time_seconds": wall,
            },
        },
    )

    if errors:
        raise RuntimeError(f"invalid merge output: {errors}")

    return rules


# ---------------------------------------------------------------------------
# Final gate
# ---------------------------------------------------------------------------

def final_gate(base_res, cand_res):
    cmp = compare_repeated(base_res, cand_res)
    reasons = []

    if not cmp.get("valid"):
        return {"eligible": False, "reasons": [cmp.get("reason", "invalid")]}

    if cmp["mean_delta"] < -EPS:
        reasons.append("overall mean regression")
    if cmp["non_regression_rate"] + EPS < FINAL_NONREG_RATE:
        reasons.append("overall non-regression rate too low")
    if cmp["mean_delta"] <= EPS:
        reasons.append("no strict overall mean gain")

    for fmt, base_fmt in base_res["mean_by_format"].items():
        cand_fmt = cand_res["mean_by_format"].get(fmt)
        if cand_fmt is not None and cand_fmt < base_fmt - EPS:
            reasons.append(f"mean format regression: {fmt}")

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "comparison": cmp,
    }


# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

def development(base, client):
    pool = load_dev_pool(base)
    fmt_pool = by_format_pool(pool)
    diag_a, diag_b, diagnostic_rows = diagnostic_dev(base, client)

    diagnostic_by_id = {r["id"]: r for r in diagnostic_rows}

    eligible_formats = {
        fmt: recs for fmt, recs in fmt_pool.items() if len(recs) >= 2
    }
    untestable_formats = {
        fmt: [r["item"].id for r in recs]
        for fmt, recs in fmt_pool.items()
        if len(recs) < 2
    }

    # Only GT-failed diagnostic cases become source cases.
    source_rows = [
        r for r in diagnostic_rows
        if r.get("gt_pass") is False
        and r["answer_format"] in eligible_formats
    ]

    generated_rules = []
    for row in source_rows:
        generated_rules.extend(
            call_rule_generator(base, client, source_row=row)
        )

    # Repeated baseline cache for exact record sets.
    baseline_cache = {}

    def get_baseline(records, key):
        if key not in baseline_cache:
            baseline_cache[key] = evaluate_repeated_records(
                base,
                client,
                records=records,
                rules=[],
                condition=f"baseline__{key}",
            )
        return baseline_cache[key]

    evaluations = []
    accepted = []

    # Map item id to pooled record.
    pooled_by_id = {r["item"].id: r for r in pool}

    for rule in generated_rules:
        sid = rule["source_case_id"]
        fmt = rule["answer_format"]

        source_rec = pooled_by_id[sid]
        target_recs = [
            r for r in eligible_formats[fmt]
            if r["item"].id != sid
        ]

        source_base = get_baseline([source_rec], f"{fmt}__source__{sid[:8]}")
        target_ids = "_".join(r["item"].id[:8] for r in target_recs)
        target_base = get_baseline(
            target_recs,
            f"{fmt}__targets__{target_ids}",
        )

        source_cand = evaluate_repeated_records(
            base,
            client,
            records=[source_rec],
            rules=[rule],
            condition=f"{rule['rule_id']}__source",
        )
        target_cand = evaluate_repeated_records(
            base,
            client,
            records=target_recs,
            rules=[rule],
            condition=f"{rule['rule_id']}__targets",
        )

        source_cmp = compare_repeated(source_base, source_cand)
        target_cmp = compare_repeated(target_base, target_cand)
        gate = rule_gate(source_cmp, target_cmp)

        record = {
            "rule_id": rule["rule_id"],
            "source_case_id": sid,
            "source_fold": rule["source_fold"],
            "answer_format": fmt,
            "target_case_ids": [r["item"].id for r in target_recs],
            "eligible": gate["eligible"],
            "reasons": gate["reasons"],
            "rule_text": rule["rule_text"],
            "source_comparison": gate.get("source"),
            "target_comparison": gate.get("target"),
        }
        evaluations.append(record)

        write_json(
            rule_dir() / "evaluations" / f"{safe_name(rule['rule_id'])}.json",
            record,
        )

        if gate["eligible"]:
            accepted.append(rule)

    merged = merge_rules(base, client, accepted)

    # Full development repeated gate.
    dev_base = evaluate_repeated_records(
        base,
        client,
        records=pool,
        rules=[],
        condition="final_dev_baseline",
    )

    dev_cand = None
    selection_gate = {
        "eligible": False,
        "reasons": ["no same-format cross-case accepted rules"],
    }

    if merged:
        write_json(
            skill_dir() / "candidate_final_skill.json",
            {
                "arm": ARM,
                "task_name": TASK,
                "rules": merged,
            },
        )

        dev_cand = evaluate_repeated_records(
            base,
            client,
            records=pool,
            rules=merged,
            condition="final_dev_candidate",
        )
        selection_gate = final_gate(dev_base, dev_cand)

    selection_path = skill_dir() / "selection.json"

    if selection_gate["eligible"]:
        selected_path = skill_dir() / "selected_adapter.json"
        write_json(
            selected_path,
            {
                "arm": ARM,
                "task_name": TASK,
                "rules": merged,
            },
        )
        selection = {
            "arm": ARM,
            "task_name": TASK,
            "decision": "ADAPTER",
            "selected_candidate": "format_stratified_loocv_skill",
            "selected_adapter_sha256": sha(selected_path),
            "accepted_rule_ids": [r["rule_id"] for r in accepted],
            "merged_rule_count": len(merged),
            "final_gate": selection_gate,
        }
    else:
        selection = {
            "arm": ARM,
            "task_name": TASK,
            "decision": "NO_ADAPTER",
            "selected_candidate": "baseline",
            "selected_adapter_sha256": None,
            "accepted_rule_ids": [r["rule_id"] for r in accepted],
            "merged_rule_count": len(merged),
            "final_gate": selection_gate,
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
            "format_stratified": True,
            "leave_one_case_out": True,
            "same_format_validation_only": True,
            "repeat_count": N_REPEATS,
            "source_nonreg_rate_threshold": SOURCE_NONREG_RATE,
            "target_nonreg_rate_threshold": TARGET_NONREG_RATE,
            "final_nonreg_rate_threshold": FINAL_NONREG_RATE,
            "surrogate_used_only_for_initial_diagnosis": True,
            "gt_visible_to_surrogate": False,
            "holdout_access": False,
        },
        "format_case_counts": {
            fmt: len(recs) for fmt, recs in sorted(fmt_pool.items())
        },
        "eligible_formats": sorted(eligible_formats),
        "untestable_formats": untestable_formats,
        "diagnostic_baseline_fold_a": base.compact(diag_a),
        "diagnostic_baseline_fold_b": base.compact(diag_b),
        "source_case_ids": [r["id"] for r in source_rows],
        "generated_rules": generated_rules,
        "rule_evaluations": evaluations,
        "accepted_rules": accepted,
        "merged_rules": merged,
        "final_dev_baseline": dev_base,
        "final_dev_candidate": dev_cand,
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

def load_holdout_pool(base):
    manifest, items = base.source_items(manifests(base)["holdout"])
    split_lookup = {x["id"]: x["split"] for x in manifest["items"]}
    return [
        {
            "fold": "H",
            "split": split_lookup[item.id],
            "item": item,
        }
        for item in items
    ]


def holdout(base, client):
    selection_path = skill_dir() / "selection.json"
    if not selection_path.exists():
        raise SystemExit("missing selection; run --phase development first")

    selection = load_json(selection_path)
    hpool = load_holdout_pool(base)

    baseline = evaluate_repeated_records(
        base,
        client,
        records=hpool,
        rules=[],
        condition="holdout_baseline",
    )

    selected = None
    comparison = None

    if selection["decision"] == "ADAPTER":
        payload = load_json(skill_dir() / "selected_adapter.json")
        selected = evaluate_repeated_records(
            base,
            client,
            records=hpool,
            rules=payload["rules"],
            condition="holdout_selected_adapter",
        )
        comparison = compare_repeated(baseline, selected)

    summary = {
        "arm": ARM,
        "task_name": TASK,
        "status": "HOLDOUT_COMPLETE",
        "selection_decision": selection["decision"],
        "selection_sha256": sha(selection_path),
        "repeat_count": N_REPEATS,
        "baseline": baseline,
        "selected_adapter": selected,
        "holdout_comparison": comparison,
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
    for p in manifests(base).values():
        if not p.exists():
            raise SystemExit(f"missing manifest: {p}")

    pool = load_dev_pool(base)
    fmt_pool = by_format_pool(pool)

    return {
        "arm": ARM,
        "task_name": TASK,
        "agent_model": base.AGENT_MODEL,
        "diagnostic_surrogate_model": base.SURROGATE_MODEL,
        "rule_model": RULE_MODEL,
        "repeat_count": N_REPEATS,
        "max_rules_per_source_case": MAX_RULES_PER_SOURCE_CASE,
        "source_nonreg_rate_threshold": SOURCE_NONREG_RATE,
        "target_nonreg_rate_threshold": TARGET_NONREG_RATE,
        "final_nonreg_rate_threshold": FINAL_NONREG_RATE,
        "format_stratified": True,
        "leave_one_case_out": True,
        "same_format_validation_only": True,
        "surrogate_used_only_for_initial_diagnosis": True,
        "gt_visible_to_surrogate": False,
        "holdout_feedback": False,
        "development_format_case_counts": {
            fmt: len(recs) for fmt, recs in sorted(fmt_pool.items())
        },
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
