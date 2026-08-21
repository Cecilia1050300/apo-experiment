#!/usr/bin/env python3
"""
Arm F.1 — Active-Format Gate Fix + Frozen Holdout Validation
=============================================================

Why this patch exists
---------------------
Arm F already found same-format cross-case transferable rules, but its final
selection gate rejected the candidate because single_letter_mcq regressed even
though NO selected Skill rule was active for single_letter_mcq.

That is a causal-attribution error:
an inactive answer format must not veto a Skill that never touches that format.

This patch DOES NOT regenerate or tune rules.
It reuses the completed Arm-F development evidence and performs:

  1) corrected development re-selection;
  2) active-format-only causal gate;
  3) frozen repeated Holdout evaluation if ADAPTER is selected.

Selection evidence remains DEVELOPMENT ONLY.
Holdout never revises rules or selection.

Primary generalization criterion on Holdout
-------------------------------------------
- active-format mean delta > 0

Deployment-safety criterion
---------------------------
- overall Holdout mean delta >= 0

Causal generalization success
-----------------------------
- combined active-format mean delta > 0;
- no testable active format regresses on mean score;
- at least one active format improves;
- no active-format parse regression.

Overall full-holdout mean is reported separately as a deployment-safety metric
because inactive formats do not receive the Skill and therefore cannot
causally veto the Skill.

The script writes to NEW Arm-F.1 directories and never overwrites the original
Arm-F experiment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/home/training/automatic_prompt_engineer/hermes_factorybench_skill_experiment")
TASK = "m1_factorybench_l123"

ARM_F_SCRIPT = ROOT / "scripts/meta_coevo/run_format_stratified_loocv.py"
ARM_F_SUMMARY = (
    ROOT
    / "results/meta_coevo/format_loocv/development"
    / f"{TASK}_summary.json"
)

F1_RESULT_DIR = ROOT / "results/meta_coevo/format_loocv_f1"
F1_SKILL_ROOT = ROOT / "prompts/adapters/meta_coevo_format_loocv_f1"
F1_TRACE_DIR = ROOT / "traces/meta_coevo/format_loocv_f1"

F1_SELECTION = F1_SKILL_ROOT / TASK / "selection.json"
F1_SELECTED_SKILL = F1_SKILL_ROOT / TASK / "selected_adapter.json"

EPS = 1e-12


def load_arm_f():
    if not ARM_F_SCRIPT.exists():
        raise SystemExit(f"missing Arm-F runner: {ARM_F_SCRIPT}")

    spec = importlib.util.spec_from_file_location("arm_f_original_for_f1", ARM_F_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit("failed to import Arm-F runner")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # All NEW repeated holdout outputs go to F.1.
    mod.ARM = "format_stratified_loocv_f1"
    mod.RESULT_DIR = F1_RESULT_DIR
    mod.TRACE_DIR = F1_TRACE_DIR
    mod.SKILL_ROOT = F1_SKILL_ROOT
    return mod


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def fixed_score_from_rows(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("cannot score empty row set")

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

    chance = sum(chances) / len(chances)
    return (sum(scores) / len(scores) - chance) / (1.0 - chance)


def subset_repeated(result: dict[str, Any], formats: set[str]) -> dict[str, Any]:
    """Re-score stored repeated runs using ONLY the specified answer formats."""
    scores = []
    parse_failures = []
    ordered_ids = None
    per_run = []

    for run in result["runs"]:
        rows = [
            row for row in run["items"]
            if row["answer_format"] in formats
        ]
        if not rows:
            continue

        ids = [r["id"] for r in rows]
        if ordered_ids is None:
            ordered_ids = ids
        elif ordered_ids != ids:
            raise RuntimeError("active subset ordered IDs changed across repeats")

        score = fixed_score_from_rows(rows)
        pf = sum(r.get("parse_error") is not None for r in rows)

        scores.append(score)
        parse_failures.append(pf)
        per_run.append({
            "run_label": run["run_label"],
            "score": score,
            "parse_failures": pf,
            "items": [
                {
                    "id": r["id"],
                    "answer_format": r["answer_format"],
                    "raw_output": r["raw_output"],
                    "score": r["score"],
                    "chance": r["chance"],
                    "parse_error": r["parse_error"],
                    "skill_active": r.get("skill_active"),
                }
                for r in rows
            ],
        })

    if not scores:
        return {
            "valid": False,
            "reason": "no active-format items",
            "formats": sorted(formats),
        }

    return {
        "valid": True,
        "formats": sorted(formats),
        "repeat_count": len(scores),
        "item_count_per_run": len(ordered_ids or []),
        "ordered_ids": ordered_ids or [],
        "scores": scores,
        "mean_score": sum(scores) / len(scores),
        "median_score": sorted(scores)[len(scores) // 2],
        "parse_failures_per_run": parse_failures,
        "parse_failures_total": sum(parse_failures),
        "runs": per_run,
    }


def compare_distributions(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare repeated stochastic runs without pretending run i is paired with run i.

    Baseline and candidate API calls are independent stochastic samples, so
    run-index deltas are not a valid causal pairing. We therefore use:
      - repeated mean / median difference as the primary effect estimate;
      - all-pairs probability of superiority as a descriptive stability metric.

    all_pairs_non_regression_probability = P(candidate >= baseline)
    all_pairs_win_probability            = P(candidate >  baseline)

    These all-pairs probabilities are descriptive only; they are NOT treated as
    independent statistical samples.
    """
    if not baseline.get("valid", True):
        return {"valid": False, "reason": baseline.get("reason", "baseline invalid")}
    if not candidate.get("valid", True):
        return {"valid": False, "reason": candidate.get("reason", "candidate invalid")}

    if baseline["ordered_ids"] != candidate["ordered_ids"]:
        return {"valid": False, "reason": "ordered IDs differ"}

    b = [float(x) for x in baseline["scores"]]
    c = [float(x) for x in candidate["scores"]]
    if not b or not c:
        return {"valid": False, "reason": "empty score distribution"}

    all_pair_deltas = [cv - bv for bv in b for cv in c]
    p_nonreg = sum(d >= -EPS for d in all_pair_deltas) / len(all_pair_deltas)
    p_win = sum(d > EPS for d in all_pair_deltas) / len(all_pair_deltas)

    return {
        "valid": True,
        "comparison_type": "unpaired_repeated_samples",
        "baseline_scores": b,
        "candidate_scores": c,
        "baseline_mean": baseline["mean_score"],
        "candidate_mean": candidate["mean_score"],
        "mean_delta": candidate["mean_score"] - baseline["mean_score"],
        "baseline_median": baseline["median_score"],
        "candidate_median": candidate["median_score"],
        "median_delta": candidate["median_score"] - baseline["median_score"],
        "all_pairs_non_regression_probability": p_nonreg,
        "all_pairs_win_probability": p_win,
        "all_pair_comparison_count": len(all_pair_deltas),
        "baseline_parse_failures_total": baseline.get("parse_failures_total"),
        "candidate_parse_failures_total": candidate.get("parse_failures_total"),
    }


def mean_format_regressions(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    active_formats: set[str],
) -> list[str]:
    reasons = []
    for fmt in sorted(active_formats):
        b = baseline["mean_by_format"].get(fmt)
        c = candidate["mean_by_format"].get(fmt)

        if b is None:
            reasons.append(f"active format missing from baseline: {fmt}")
            continue
        if c is None:
            reasons.append(f"active format missing from candidate: {fmt}")
            continue
        if c < b - EPS:
            reasons.append(
                f"active-format mean regression: {fmt} ({b:.6f} -> {c:.6f})"
            )
    return reasons


def audit_activation(candidate: dict[str, Any], active_formats: set[str]) -> dict[str, Any]:
    errors = []
    counts = {
        "active_format_items": 0,
        "inactive_format_items": 0,
        "active_format_skill_true": 0,
        "inactive_format_skill_false": 0,
    }

    for run in candidate["runs"]:
        for row in run["items"]:
            fmt = row["answer_format"]
            skill_active = bool(row.get("skill_active"))

            if fmt in active_formats:
                counts["active_format_items"] += 1
                if skill_active:
                    counts["active_format_skill_true"] += 1
                else:
                    errors.append(
                        f"expected active Skill but found false: {row['id']} {fmt}"
                    )
            else:
                counts["inactive_format_items"] += 1
                if not skill_active:
                    counts["inactive_format_skill_false"] += 1
                else:
                    errors.append(
                        f"unexpected Skill activation: {row['id']} {fmt}"
                    )

    return {
        "valid": not errors,
        "errors": errors,
        "counts": counts,
    }


def corrected_development_gate(
    arm_f,
    summary: dict[str, Any],
) -> dict[str, Any]:
    merged = summary.get("merged_rules") or []
    baseline = summary.get("final_dev_baseline")
    candidate = summary.get("final_dev_candidate")

    if not merged:
        return {
            "eligible": False,
            "reasons": ["no merged rules"],
        }
    if not baseline or not candidate:
        return {
            "eligible": False,
            "reasons": ["missing final development repeated results"],
        }

    active_formats = {r["answer_format"] for r in merged}

    activation = audit_activation(candidate, active_formats)

    active_baseline = subset_repeated(baseline, active_formats)
    active_candidate = subset_repeated(candidate, active_formats)
    active_cmp = compare_distributions(active_baseline, active_candidate)

    # Preserve all-development comparison as a secondary deployment metric.
    overall_cmp = arm_f.compare_repeated(baseline, candidate)

    reasons = []
    if not activation["valid"]:
        reasons.extend([f"activation audit: {e}" for e in activation["errors"]])

    if not active_cmp.get("valid"):
        reasons.append(f"active comparison invalid: {active_cmp.get('reason')}")
    else:
        if active_cmp["mean_delta"] <= EPS:
            reasons.append("no positive active-format mean gain")
        if (
            active_cmp["candidate_parse_failures_total"]
            > active_cmp["baseline_parse_failures_total"]
        ):
            reasons.append("active-format parse regression")

    # IMPORTANT FIX:
    # only formats where the selected Skill is actually active may veto the Skill.
    reasons.extend(
        mean_format_regressions(
            baseline,
            candidate,
            active_formats,
        )
    )

    inactive_formats = sorted(
        set(baseline["mean_by_format"]) - active_formats
    )
    inactive_audit = {
        fmt: {
            "baseline_mean": baseline["mean_by_format"].get(fmt),
            "candidate_mean": candidate["mean_by_format"].get(fmt),
            "used_for_selection": False,
            "reason": "Skill inactive for this format",
        }
        for fmt in inactive_formats
    }

    return {
        "eligible": not reasons,
        "reasons": reasons,
        "gate_version": "F1_active_formats_only",
        "active_formats": sorted(active_formats),
        "inactive_formats": inactive_formats,
        "activation_audit": activation,
        "active_format_comparison": active_cmp,
        "overall_development_comparison_secondary": overall_cmp,
        "inactive_format_audit": inactive_audit,
        "selection_statistics": {
            "primary": "unpaired repeated mean delta on active formats",
            "stability_descriptive": "all-pairs probability of superiority",
            "run_index_pairing_used": False,
        },
    }


def reselect(arm_f) -> dict[str, Any]:
    if not ARM_F_SUMMARY.exists():
        raise SystemExit(f"missing completed Arm-F summary: {ARM_F_SUMMARY}")

    summary = read_json(ARM_F_SUMMARY)
    gate = corrected_development_gate(arm_f, summary)
    merged = summary.get("merged_rules") or []
    accepted = summary.get("accepted_rules") or []

    if gate["eligible"]:
        write_json(
            F1_SELECTED_SKILL,
            {
                "arm": "format_stratified_loocv_f1",
                "task_name": TASK,
                "source_arm": "format_stratified_loocv",
                "selection_gate": "active_formats_only",
                "rules": merged,
            },
        )
        decision = "ADAPTER"
        selected_candidate = "format_stratified_loocv_skill"
        selected_sha = arm_f.sha(F1_SELECTED_SKILL)
    else:
        decision = "NO_ADAPTER"
        selected_candidate = "baseline"
        selected_sha = None

    selection = {
        "arm": "format_stratified_loocv_f1",
        "task_name": TASK,
        "status": "RESELECTION_COMPLETE",
        "source_arm_f_summary": str(ARM_F_SUMMARY),
        "source_arm_f_selection_decision": (
            summary.get("selection") or {}
        ).get("decision"),
        "decision": decision,
        "selected_candidate": selected_candidate,
        "selected_adapter_sha256": selected_sha,
        "accepted_rule_ids": [r["rule_id"] for r in accepted],
        "merged_rule_count": len(merged),
        "development_gate": gate,
        "holdout_access": False,
        "rules_regenerated": False,
        "development_model_calls_added": False,
    }
    write_json(F1_SELECTION, selection)

    out = F1_RESULT_DIR / "development" / f"{TASK}_reselection.json"
    write_json(out, selection)
    return selection


def holdout(arm_f, client) -> dict[str, Any]:
    if not F1_SELECTION.exists():
        raise SystemExit("missing F.1 selection; run --phase reselect first")

    selection = read_json(F1_SELECTION)
    if selection["decision"] != "ADAPTER":
        raise SystemExit(
            "F.1 development gate did not select ADAPTER; do not run holdout."
        )

    selected_payload = read_json(F1_SELECTED_SKILL)
    selected_rules = selected_payload["rules"]
    active_formats = {r["answer_format"] for r in selected_rules}

    # IMPORTANT:
    # Arm-F helper functions such as load_holdout_pool() and
    # evaluate_repeated_records() expect the imported Arm-B/static runner
    # object ("base"), because source_items(), call_chat(), _score_one(), etc.
    # live there. Passing the Arm-F module itself causes:
    #   AttributeError: module ... has no attribute 'source_items'
    base = arm_f.load_base()

    hpool = arm_f.load_holdout_pool(base)

    baseline = arm_f.evaluate_repeated_records(
        base,
        client,
        records=hpool,
        rules=[],
        condition="f1_holdout_baseline",
    )
    selected = arm_f.evaluate_repeated_records(
        base,
        client,
        records=hpool,
        rules=selected_rules,
        condition="f1_holdout_selected_adapter",
    )

    overall_cmp = arm_f.compare_repeated(baseline, selected)

    active_baseline = subset_repeated(baseline, active_formats)
    active_selected = subset_repeated(selected, active_formats)
    active_cmp = compare_distributions(active_baseline, active_selected)

    activation = audit_activation(selected, active_formats)

    per_active_format = {}
    for fmt in sorted(active_formats):
        b = baseline["mean_by_format"].get(fmt)
        c = selected["mean_by_format"].get(fmt)
        per_active_format[fmt] = {
            "baseline_mean": b,
            "candidate_mean": c,
            "mean_delta": None if b is None or c is None else c - b,
        }

    # Generalization is judged causally ONLY where the Skill is active.
    # Inactive formats are reported as deployment context, not as a causal veto.
    testable_active = {
        fmt: x for fmt, x in per_active_format.items()
        if x["mean_delta"] is not None
    }
    missing_active_formats = sorted(
        set(active_formats) - set(testable_active)
    )

    per_format_nonregression_pass = bool(testable_active) and all(
        x["mean_delta"] >= -EPS
        for x in testable_active.values()
    )
    at_least_one_active_format_gain = any(
        x["mean_delta"] > EPS
        for x in testable_active.values()
    )

    active_parse_safety_pass = (
        active_cmp.get("valid")
        and active_cmp["candidate_parse_failures_total"]
        <= active_cmp["baseline_parse_failures_total"]
    )

    causal_generalization_pass = bool(
        activation["valid"]
        and active_cmp.get("valid")
        and active_cmp["mean_delta"] > EPS
        and per_format_nonregression_pass
        and at_least_one_active_format_gain
        and active_parse_safety_pass
    )

    # Secondary deployment metric across the entire holdout. Because inactive
    # formats receive no Skill, stochastic movement there is NOT used to decide
    # whether the Skill generalized in its intended scope.
    deployment_safety_pass = bool(
        overall_cmp.get("valid")
        and overall_cmp["mean_delta"] >= -EPS
    )

    deployment_safe_generalization_pass = bool(
        causal_generalization_pass and deployment_safety_pass
    )

    summary = {
        "arm": "format_stratified_loocv_f1",
        "task_name": TASK,
        "status": "HOLDOUT_COMPLETE",
        "selection_decision": selection["decision"],
        "repeat_count": arm_f.N_REPEATS,
        "active_formats": sorted(active_formats),
        "baseline": baseline,
        "selected_adapter": selected,
        "overall_holdout_comparison": overall_cmp,
        "active_format_holdout_comparison": active_cmp,
        "per_active_format": per_active_format,
        "activation_audit": activation,
        "generalization_assessment": {
            "primary_causal_criteria": [
                "active-format combined mean delta > 0",
                "every testable active format mean delta >= 0",
                "at least one testable active format mean delta > 0",
                "no active-format parse regression",
                "activation audit valid"
            ],
            "causal_generalization_pass": causal_generalization_pass,
            "testable_active_formats": sorted(testable_active),
            "missing_active_formats_on_holdout": missing_active_formats,
            "per_format_nonregression_pass": per_format_nonregression_pass,
            "at_least_one_active_format_gain": at_least_one_active_format_gain,
            "active_parse_safety_pass": active_parse_safety_pass,
            "deployment_safety_criterion_secondary": "overall holdout mean delta >= 0",
            "deployment_safety_pass": deployment_safety_pass,
            "deployment_safe_generalization_pass": deployment_safe_generalization_pass,
            "run_index_pairing_used": False,
            "stability_metric": "all-pairs probability of superiority (descriptive)"
        },
        "no_holdout_feedback": True,
        "rule_generation_on_holdout": False,
        "rule_selection_on_holdout": False,
        "skill_revision_on_holdout": False,
        "surrogate_called_on_holdout": False,
    }

    out = F1_RESULT_DIR / "holdout" / f"{TASK}_summary.json"
    write_json(out, summary)
    return summary


def preflight(arm_f):
    info = {
        "arm": "format_stratified_loocv_f1",
        "task_name": TASK,
        "purpose": "fix Arm-F final gate causal attribution and validate frozen holdout",
        "source_arm_f_script": str(ARM_F_SCRIPT),
        "source_arm_f_summary": str(ARM_F_SUMMARY),
        "arm_f_summary_exists": ARM_F_SUMMARY.exists(),
        "repeat_count": arm_f.N_REPEATS,
        "final_nonreg_rate_threshold": arm_f.FINAL_NONREG_RATE,
        "selection_fix": (
            "format-regression veto applies ONLY to answer formats where "
            "the selected Skill is active"
        ),
        "holdout_primary_criterion": (
            "active-format combined mean delta > 0; every testable active format "
            "must be non-regressive; at least one must improve"
        ),
        "holdout_deployment_safety_secondary": "overall holdout mean delta >= 0",
        "stochastic_comparison": "unpaired repeated means + all-pairs superiority; no run-index pairing",
        "holdout_helper_base_runner_required": True,
        "holdout_feedback": False,
        "rules_regenerated_in_reselect": False,
        "development_model_calls_added_in_reselect": False,
        "f1_result_dir": str(F1_RESULT_DIR),
        "f1_skill_root": str(F1_SKILL_ROOT),
    }

    if ARM_F_SUMMARY.exists():
        s = read_json(ARM_F_SUMMARY)
        merged = s.get("merged_rules") or []
        info["arm_f_accepted_rule_count"] = len(s.get("accepted_rules") or [])
        info["arm_f_merged_rule_count"] = len(merged)
        info["active_formats"] = sorted(
            {r["answer_format"] for r in merged}
        )

    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["preflight", "reselect", "holdout"],
        required=True,
    )
    args = parser.parse_args()

    arm_f = load_arm_f()

    if args.phase == "preflight":
        print(json.dumps(preflight(arm_f), indent=2, ensure_ascii=False))
        return

    if args.phase == "reselect":
        result = reselect(arm_f)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return

    import os
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing in model process")

    result = holdout(arm_f, OpenAI())
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
