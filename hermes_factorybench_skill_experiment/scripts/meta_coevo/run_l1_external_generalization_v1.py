#!/usr/bin/env python3
"""
Arm F — Level-1 External Generalization Test v1
================================================

Frozen protocol:
- External manifest: 20 four_letter_tf + 30 scalar_range = 50 unseen L1 cases
- Baseline: GPT-4o-mini, no Skill, 5 repeated runs
- Candidate: exact frozen Arm-F Skill, 5 repeated runs
- NO rule generation / revision / selection
- NO surrogate verifier
- Deterministic FactoryBench GT scoring only
- Case-level paired bootstrap 95% CI (10,000 resamples)

Important statistical detail
----------------------------
The 5 baseline API runs and 5 candidate API runs are independent stochastic
samples, so run1-vs-run1 is NOT treated as a true pair.

For the bootstrap, pairing is instead done by CASE:
  1. average each case over its 5 baseline runs;
  2. average the same case over its 5 candidate runs;
  3. resample the 50 case IDs with replacement;
  4. recompute the exact chance-corrected FactoryBench fixed score;
  5. bootstrap the candidate - baseline delta.

This preserves the real experimental unit for external generalization: the case.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(
    "/home/training/automatic_prompt_engineer/"
    "hermes_factorybench_skill_experiment"
)

ARM_F_SCRIPT = ROOT / "scripts/meta_coevo/run_format_stratified_loocv.py"

EXTERNAL_MANIFEST = (
    ROOT / "data_manifests/meta_m1/factorybench_l1_external_test_v1.json"
)

FROZEN_SKILL = (
    ROOT
    / "prompts/adapters/meta_coevo_format_loocv_f1"
    / "m1_factorybench_l123"
    / "selected_adapter.json"
)

RESULT_DIR = ROOT / "results/meta_coevo/external_l1_v1"
SUMMARY_PATH = RESULT_DIR / "factorybench_l1_external_test_v1_summary.json"

EXPECTED_MANIFEST_SHA256 = (
    "34fcd83be643021f93a5331c9fd66e4e85dec7c8e1e3889230b2c3df0285d083"
)

EXPECTED_SKILL_SHA256 = (
    "253044ed73fa651337dcc86bbae797c227a65e37523b19195b74ddcf7efc2af1"
)

EXPECTED_FORMAT_COUNTS = {
    "four_letter_tf": 20,
    "scalar_range": 30,
}

N_REPEATS = 5
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260821
EPS = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def load_arm_f():
    if not ARM_F_SCRIPT.exists():
        raise SystemExit(f"missing Arm-F runner: {ARM_F_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "arm_f_for_external_l1_v1",
        ARM_F_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise SystemExit("failed to import Arm-F runner")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Reuse ONLY the frozen evaluation machinery.
    # Redirect all repeated outputs to an isolated external-test directory.
    mod.ARM = "external_l1_v1"
    mod.RESULT_DIR = RESULT_DIR
    mod.TRACE_DIR = ROOT / "traces/meta_coevo/external_l1_v1"
    mod.N_REPEATS = N_REPEATS

    return mod


def validate_frozen_artifacts() -> dict[str, Any]:
    if not EXTERNAL_MANIFEST.exists():
        raise SystemExit(f"missing external manifest: {EXTERNAL_MANIFEST}")

    if not FROZEN_SKILL.exists():
        raise SystemExit(f"missing frozen Skill: {FROZEN_SKILL}")

    manifest_sha = sha256(EXTERNAL_MANIFEST)
    skill_sha = sha256(FROZEN_SKILL)

    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "EXTERNAL MANIFEST SHA256 CHANGED.\n"
            f"expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"actual  : {manifest_sha}\n"
            "Do not continue with a modified external test set."
        )

    if skill_sha != EXPECTED_SKILL_SHA256:
        raise RuntimeError(
            "FROZEN SKILL SHA256 CHANGED.\n"
            f"expected: {EXPECTED_SKILL_SHA256}\n"
            f"actual  : {skill_sha}\n"
            "Do not continue with a modified Skill."
        )

    manifest = read_json(EXTERNAL_MANIFEST)
    rows = manifest["items"]

    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate IDs in external manifest")

    fmt_counts = Counter(r["answer_format"] for r in rows)

    if dict(fmt_counts) != EXPECTED_FORMAT_COUNTS:
        raise RuntimeError(
            "external format composition changed: "
            f"expected={EXPECTED_FORMAT_COUNTS}, actual={dict(fmt_counts)}"
        )

    if any(int(r["level"]) != 1 for r in rows):
        raise RuntimeError("external manifest contains non-Level-1 case")

    if any(r["split"] != "test" for r in rows):
        raise RuntimeError("external manifest contains non-test split")

    skill_payload = read_json(FROZEN_SKILL)
    rules = skill_payload.get("rules") or []
    active_formats = sorted({r["answer_format"] for r in rules})

    if set(active_formats) != set(EXPECTED_FORMAT_COUNTS):
        raise RuntimeError(
            "frozen Skill active formats changed: "
            f"{active_formats}"
        )

    return {
        "manifest_sha256": manifest_sha,
        "skill_sha256": skill_sha,
        "external_case_count": len(rows),
        "format_counts": dict(sorted(fmt_counts.items())),
        "active_formats": active_formats,
        "rule_count": len(rules),
        "rules": rules,
    }


def load_external_pool(base):
    manifest, items = base.source_items(EXTERNAL_MANIFEST)
    split_lookup = {
        row["id"]: row["split"]
        for row in manifest["items"]
    }

    return [
        {
            "fold": "EXTERNAL_L1_V1",
            "split": split_lookup[item.id],
            "item": item,
        }
        for item in items
    ]


def effective_score(row: dict[str, Any]) -> float:
    clean = (
        row.get("parse_error") is None
        and isinstance(row.get("score"), (int, float))
        and math.isfinite(float(row["score"]))
    )
    return float(row["score"]) if clean else 0.0


def build_case_table(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Collapse 5 stochastic runs into one record per case.

    Each case keeps:
    - average effective raw score across repeats
    - chance value
    - format
    - parse-failure count
    """
    table: dict[str, dict[str, Any]] = {}

    for run in result["runs"]:
        for row in run["items"]:
            case_id = row["id"]

            rec = table.setdefault(
                case_id,
                {
                    "id": case_id,
                    "answer_format": row["answer_format"],
                    "chance": float(row.get("chance", 0.0)),
                    "effective_scores": [],
                    "parse_failures": 0,
                },
            )

            if rec["answer_format"] != row["answer_format"]:
                raise RuntimeError(f"format changed across repeats: {case_id}")

            if abs(rec["chance"] - float(row.get("chance", 0.0))) > EPS:
                raise RuntimeError(f"chance changed across repeats: {case_id}")

            rec["effective_scores"].append(effective_score(row))
            rec["parse_failures"] += int(row.get("parse_error") is not None)

    for rec in table.values():
        if len(rec["effective_scores"]) != N_REPEATS:
            raise RuntimeError(
                f"case {rec['id']} has {len(rec['effective_scores'])} repeats, "
                f"expected {N_REPEATS}"
            )

        rec["mean_effective_score"] = (
            sum(rec["effective_scores"]) / len(rec["effective_scores"])
        )

    return table


def fixed_from_cases(
    case_ids: list[str],
    table: dict[str, dict[str, Any]],
) -> float:
    if not case_ids:
        raise ValueError("empty case sample")

    scores = [
        table[case_id]["mean_effective_score"]
        for case_id in case_ids
    ]
    chances = [
        table[case_id]["chance"]
        for case_id in case_ids
    ]

    mean_score = sum(scores) / len(scores)
    mean_chance = sum(chances) / len(chances)

    return (mean_score - mean_chance) / (1.0 - mean_chance)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")

    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return sorted_values[lo]

    weight = pos - lo
    return (
        sorted_values[lo] * (1.0 - weight)
        + sorted_values[hi] * weight
    )


def bootstrap_delta(
    baseline_table: dict[str, dict[str, Any]],
    candidate_table: dict[str, dict[str, Any]],
    case_ids: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    if set(case_ids) - set(baseline_table):
        raise RuntimeError("baseline missing requested case")
    if set(case_ids) - set(candidate_table):
        raise RuntimeError("candidate missing requested case")

    # Real pairing is by case identity.
    point_baseline = fixed_from_cases(case_ids, baseline_table)
    point_candidate = fixed_from_cases(case_ids, candidate_table)
    point_delta = point_candidate - point_baseline

    rng = random.Random(seed)
    deltas = []
    n = len(case_ids)

    for _ in range(BOOTSTRAP_SAMPLES):
        sampled = [
            case_ids[rng.randrange(n)]
            for _ in range(n)
        ]

        b = fixed_from_cases(sampled, baseline_table)
        c = fixed_from_cases(sampled, candidate_table)
        deltas.append(c - b)

    deltas.sort()

    ci_low = percentile(deltas, 0.025)
    ci_high = percentile(deltas, 0.975)

    return {
        "method": "paired_case_bootstrap_percentile",
        "pairing_unit": "case_id",
        "api_repeats_per_condition": N_REPEATS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": seed,
        "case_count": len(case_ids),
        "baseline_score": point_baseline,
        "candidate_score": point_candidate,
        "mean_delta": point_delta,
        "ci_95": [ci_low, ci_high],
        "ci_excludes_zero_positive": ci_low > 0.0,
        "bootstrap_probability_delta_gt_0": (
            sum(x > 0.0 for x in deltas) / len(deltas)
        ),
    }


def unpaired_repeat_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """
    Descriptive only. No fake run1-vs-run1 pairing.
    """
    b = [float(x) for x in baseline["scores"]]
    c = [float(x) for x in candidate["scores"]]

    all_deltas = [
        cv - bv
        for bv in b
        for cv in c
    ]

    return {
        "comparison_type": "independent_stochastic_repeat_distributions",
        "baseline_scores": b,
        "candidate_scores": c,
        "baseline_mean": baseline["mean_score"],
        "candidate_mean": candidate["mean_score"],
        "mean_delta": (
            candidate["mean_score"]
            - baseline["mean_score"]
        ),
        "baseline_median": baseline["median_score"],
        "candidate_median": candidate["median_score"],
        "median_delta": (
            candidate["median_score"]
            - baseline["median_score"]
        ),
        "all_pairs_non_regression_probability": (
            sum(d >= -EPS for d in all_deltas)
            / len(all_deltas)
        ),
        "all_pairs_win_probability": (
            sum(d > EPS for d in all_deltas)
            / len(all_deltas)
        ),
        "all_pair_comparison_count": len(all_deltas),
        "run_index_pairing_used": False,
    }


def parse_failure_total(result: dict[str, Any]) -> int:
    return sum(
        int(row.get("parse_error") is not None)
        for run in result["runs"]
        for row in run["items"]
    )


def run_external() -> dict[str, Any]:
    frozen = validate_frozen_artifacts()

    arm_f = load_arm_f()
    base = arm_f.load_base()

    # load_base() may change helper-internal directories, but all actual
    # repeated external outputs are controlled by arm_f.RESULT_DIR above.
    arm_f.RESULT_DIR = RESULT_DIR
    arm_f.N_REPEATS = N_REPEATS

    pool = load_external_pool(base)

    if len(pool) != 50:
        raise RuntimeError(f"expected 50 external cases, got {len(pool)}")

    from openai import OpenAI
    client = OpenAI()

    print("\n===== EXTERNAL GENERALIZATION TEST =====")
    print("cases:", len(pool))
    print("repeats per condition:", N_REPEATS)
    print("agent model:", base.AGENT_MODEL)
    print("manifest SHA256:", frozen["manifest_sha256"])
    print("Skill SHA256:", frozen["skill_sha256"])
    print("\nRunning baseline ×5 ...")

    baseline = arm_f.evaluate_repeated_records(
        base,
        client,
        records=pool,
        rules=[],
        condition="external_l1_v1_baseline",
    )

    print("Running frozen Skill ×5 ...")

    candidate = arm_f.evaluate_repeated_records(
        base,
        client,
        records=pool,
        rules=frozen["rules"],
        condition="external_l1_v1_frozen_skill",
    )

    if baseline["ordered_ids"] != candidate["ordered_ids"]:
        raise RuntimeError("baseline/candidate external IDs differ")

    baseline_table = build_case_table(baseline)
    candidate_table = build_case_table(candidate)

    all_ids = baseline["ordered_ids"]

    # Verify same external universe.
    if set(all_ids) != set(candidate_table):
        raise RuntimeError("candidate case universe differs")

    combined_bootstrap = bootstrap_delta(
        baseline_table,
        candidate_table,
        all_ids,
        seed=BOOTSTRAP_SEED,
    )

    per_format_bootstrap = {}
    for idx, fmt in enumerate(EXPECTED_FORMAT_COUNTS, start=1):
        fmt_ids = [
            case_id
            for case_id in all_ids
            if baseline_table[case_id]["answer_format"] == fmt
        ]

        per_format_bootstrap[fmt] = bootstrap_delta(
            baseline_table,
            candidate_table,
            fmt_ids,
            seed=BOOTSTRAP_SEED + idx,
        )

    baseline_parse = parse_failure_total(baseline)
    candidate_parse = parse_failure_total(candidate)

    repeat_summary = unpaired_repeat_summary(
        baseline,
        candidate,
    )

    # --------------------------------------------------------------
    # Frozen success criteria, defined before observing external result.
    # --------------------------------------------------------------
    combined_positive = combined_bootstrap["mean_delta"] > EPS

    all_formats_nonregressive = all(
        result["mean_delta"] >= -EPS
        for result in per_format_bootstrap.values()
    )

    at_least_one_format_improves = any(
        result["mean_delta"] > EPS
        for result in per_format_bootstrap.values()
    )

    parse_safety_pass = candidate_parse <= baseline_parse

    protocol_generalization_pass = bool(
        combined_positive
        and all_formats_nonregressive
        and at_least_one_format_improves
        and parse_safety_pass
    )

    # Stronger statistical-support flag. This is reported separately;
    # it does NOT retroactively change the frozen protocol threshold.
    bootstrap_ci_supports_positive_effect = bool(
        combined_bootstrap["ci_excludes_zero_positive"]
    )

    summary = {
        "experiment": "arm_f_external_l1_v1",
        "status": "EXTERNAL_TEST_COMPLETE",
        "purpose": "replicated_same_level_generalization",
        "agent_model": base.AGENT_MODEL,
        "level": 1,
        "external_case_count": len(all_ids),
        "repeat_count_per_condition": N_REPEATS,

        "frozen_artifacts": {
            "manifest_path": str(EXTERNAL_MANIFEST),
            "manifest_sha256": frozen["manifest_sha256"],
            "skill_path": str(FROZEN_SKILL),
            "skill_sha256": frozen["skill_sha256"],
            "skill_rule_count": frozen["rule_count"],
            "active_formats": frozen["active_formats"],
        },

        "protocol_integrity": {
            "rule_generation_on_external": False,
            "rule_selection_on_external": False,
            "skill_revision_on_external": False,
            "surrogate_called_on_external": False,
            "external_feedback_used": False,
            "run_index_pairing_used": False,
            "bootstrap_pairing_unit": "case_id",
        },

        "baseline": baseline,
        "frozen_skill": candidate,

        "repeat_distribution_summary": repeat_summary,

        "case_level_bootstrap": {
            "combined": combined_bootstrap,
            "by_format": per_format_bootstrap,
        },

        "parse_failures": {
            "baseline_total_across_all_runs": baseline_parse,
            "candidate_total_across_all_runs": candidate_parse,
            "non_regression_pass": parse_safety_pass,
        },

        "frozen_success_criteria": {
            "combined_active_format_mean_delta_gt_0": combined_positive,
            "every_active_format_mean_delta_ge_0": all_formats_nonregressive,
            "at_least_one_active_format_mean_delta_gt_0": (
                at_least_one_format_improves
            ),
            "parse_failures_do_not_increase": parse_safety_pass,
            "protocol_generalization_pass": protocol_generalization_pass,
        },

        "statistical_support": {
            "combined_bootstrap_95ci_excludes_zero_positive": (
                bootstrap_ci_supports_positive_effect
            ),
            "note": (
                "Bootstrap CI is additional evidence and was not used to "
                "retroactively redefine the frozen protocol pass threshold."
            ),
        },
    }

    write_json_new(SUMMARY_PATH, summary)

    print("\n========================================")
    print("EXTERNAL TEST COMPLETE")
    print("========================================")

    print("\nRepeated-score distribution:")
    print(
        "baseline mean:",
        repeat_summary["baseline_mean"],
    )
    print(
        "frozen Skill mean:",
        repeat_summary["candidate_mean"],
    )
    print(
        "mean delta:",
        repeat_summary["mean_delta"],
    )

    print("\nCase-level bootstrap — combined:")
    print(
        "delta:",
        combined_bootstrap["mean_delta"],
    )
    print(
        "95% CI:",
        combined_bootstrap["ci_95"],
    )
    print(
        "CI > 0:",
        combined_bootstrap["ci_excludes_zero_positive"],
    )

    print("\nPer format:")
    for fmt, result in per_format_bootstrap.items():
        print(
            f"  {fmt}: "
            f"{result['baseline_score']:.6f} -> "
            f"{result['candidate_score']:.6f} "
            f"(delta={result['mean_delta']:+.6f}, "
            f"95% CI={result['ci_95']})"
        )

    print("\nParse failures:")
    print("  baseline:", baseline_parse)
    print("  Skill   :", candidate_parse)

    print("\nGENERALIZATION:")
    print(
        "  protocol_generalization_pass =",
        protocol_generalization_pass,
    )
    print(
        "  bootstrap_95ci_positive =",
        bootstrap_ci_supports_positive_effect,
    )

    print("\nSummary:")
    print(SUMMARY_PATH)

    return summary


def preflight() -> dict[str, Any]:
    frozen = validate_frozen_artifacts()

    return {
        "status": "READY",
        "experiment": "arm_f_external_l1_v1",
        "external_case_count": frozen["external_case_count"],
        "format_counts": frozen["format_counts"],
        "manifest_sha256": frozen["manifest_sha256"],
        "skill_sha256": frozen["skill_sha256"],
        "skill_rule_count": frozen["rule_count"],
        "active_formats": frozen["active_formats"],
        "repeats_per_condition": N_REPEATS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_unit": "case_id",
        "run_index_pairing_used": False,
        "external_feedback_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["preflight", "run"],
        required=True,
    )
    args = parser.parse_args()

    if args.phase == "preflight":
        print(
            json.dumps(
                preflight(),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    run_external()


if __name__ == "__main__":
    main()
