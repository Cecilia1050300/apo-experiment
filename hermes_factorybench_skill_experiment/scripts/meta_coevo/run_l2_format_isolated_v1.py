#!/usr/bin/env python3
"""
L2 Format-Isolated Skill Selection v1
=====================================

This is a NEW development-stage iteration after L2 Skill Evolution v1 selected
NO_ADAPTER at the whole-adapter final gate.

It does NOT alter or inspect the frozen L2 holdout during selection.

Research question:
Should Skill selection be performed independently per answer format instead of
forcing all accepted format-specific rules into one joint adapter?

Phases
------
preflight
  - validate frozen L2 manifests
  - recover accepted rule IDs from prior L2 development result
  - verify candidate rule files exist

select
  - evaluate accepted rules independently for each answer format
  - baseline x5 vs format-specific Skill x5 on ONLY that format's 8 dev cases
  - require:
      mean delta > 0
      all-pairs non-regression probability >= 0.8
      parse failures do not increase
  - freeze only formats that pass
  - write a format-isolated selected adapter + SHA256 lock

holdout
  - only after select
  - exact frozen format-isolated adapter on untouched 50-case L2 holdout
  - baseline x5 vs Skill x5
  - case-level paired bootstrap 95% CI
  - inactive formats do not causally veto the Skill
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(
    "/home/training/automatic_prompt_engineer/"
    "hermes_factorybench_skill_experiment"
)

V3 = ROOT / "scripts/meta_coevo/run_l2_skill_evolution_v1_v3.py"

RESULT_DIR = ROOT / "results/meta_coevo/l2_format_isolated_v1"
SKILL_DIR = ROOT / "prompts/adapters/meta_coevo_l2_format_isolated_v1"
SELECTED_ADAPTER = SKILL_DIR / "selected_adapter.json"
SELECTION_LOCK = SKILL_DIR / "selection_lock.json"
SELECTION_SUMMARY = RESULT_DIR / "development_selection_summary.json"
HOLDOUT_SUMMARY = RESULT_DIR / "holdout_summary.json"

N_REPEATS = int(os.getenv("N_REPEATS", "5"))
BOOTSTRAP_SAMPLES = int(os.getenv("BOOTSTRAP_SAMPLES", "10000"))
BOOTSTRAP_SEED = 20260821
EPS = 1e-12
NONREG_THRESHOLD = 0.80

# These are the exact rules accepted by the prior L2 development run.
# The script also verifies them against the saved development result if found.
EXPECTED_ACCEPTED_RULE_IDS = [
    "four_letter_tf__28c9386b__R1",
    "scalar_range__9816a8be__R2",
    "scalar_range__f0b1ee65__R1",
    "scalar_range__f0b1ee65__R2",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing overwrite: {path}")
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_v3():
    if not V3.exists():
        raise RuntimeError(f"missing dependency: {V3}")

    spec = importlib.util.spec_from_file_location(
        "l2_skill_v3_for_format_isolated",
        V3,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import L2 v3 wrapper")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def effective_score(row: dict[str, Any]) -> float:
    ok = (
        row.get("parse_error") is None
        and isinstance(row.get("score"), (int, float))
        and math.isfinite(float(row["score"]))
    )
    return float(row["score"]) if ok else 0.0


def fixed_score(rows: list[dict[str, Any]]) -> float:
    scores = [effective_score(r) for r in rows]
    chances = [float(r.get("chance", 0.0)) for r in rows]
    mc = sum(chances) / len(chances)
    return (sum(scores) / len(scores) - mc) / (1.0 - mc)


def repeated_summary(result: dict[str, Any]) -> dict[str, Any]:
    scores = [float(x) for x in result["scores"]]

    parse_total = sum(
        int(row.get("parse_error") is not None)
        for run in result["runs"]
        for row in run["items"]
    )

    return {
        "scores": scores,
        "mean": sum(scores) / len(scores),
        "median": float(statistics.median(scores)),
        "parse_failures_total": parse_total,
    }


def compare_independent_repeats(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    b = repeated_summary(baseline)
    c = repeated_summary(candidate)

    deltas = [
        cv - bv
        for bv in b["scores"]
        for cv in c["scores"]
    ]

    return {
        "baseline_scores": b["scores"],
        "candidate_scores": c["scores"],
        "baseline_mean": b["mean"],
        "candidate_mean": c["mean"],
        "mean_delta": c["mean"] - b["mean"],
        "baseline_median": b["median"],
        "candidate_median": c["median"],
        "median_delta": c["median"] - b["median"],
        "all_pairs_non_regression_probability": (
            sum(d >= -EPS for d in deltas) / len(deltas)
        ),
        "all_pairs_win_probability": (
            sum(d > EPS for d in deltas) / len(deltas)
        ),
        "baseline_parse_failures": b["parse_failures_total"],
        "candidate_parse_failures": c["parse_failures_total"],
        "parse_non_regression": (
            c["parse_failures_total"] <= b["parse_failures_total"]
        ),
        "run_index_pairing_used": False,
    }


def find_prior_development_result(v3) -> Path | None:
    candidates = []

    for p in v3.RESULT_DIR.rglob("*.json"):
        # repeated files are not development summaries
        if "repeated" in p.parts:
            continue

        try:
            d = load_json(p)
        except Exception:
            continue

        if not isinstance(d, dict):
            continue

        selection = d.get("selection")
        if (
            isinstance(selection, dict)
            and isinstance(selection.get("accepted_rule_ids"), list)
        ):
            candidates.append(p)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def accepted_rule_ids(v3) -> tuple[list[str], str]:
    saved = find_prior_development_result(v3)

    if saved is None:
        return list(EXPECTED_ACCEPTED_RULE_IDS), "frozen_from_observed_development_output"

    d = load_json(saved)
    ids = d["selection"]["accepted_rule_ids"]

    if ids != EXPECTED_ACCEPTED_RULE_IDS:
        raise RuntimeError(
            "saved development accepted_rule_ids differ from the "
            "previously frozen observed result\n"
            f"saved={ids}\nexpected={EXPECTED_ACCEPTED_RULE_IDS}"
        )

    return ids, str(saved)


def load_rule_files(arm_f, rule_ids: list[str]) -> list[dict[str, Any]]:
    out = []

    for rid in rule_ids:
        p = (
            arm_f.rule_dir()
            / "candidates"
            / f"{arm_f.safe_name(rid)}.json"
        )

        if not p.exists():
            raise RuntimeError(f"missing accepted candidate rule: {p}")

        rule = load_json(p)

        if rule.get("rule_id") != rid:
            raise RuntimeError(
                f"rule_id mismatch in {p}: {rule.get('rule_id')} != {rid}"
            )

        out.append(rule)

    return out


def load_dev_records(arm_f, base):
    pool = arm_f.load_dev_pool(base)
    if len(pool) != 16:
        raise RuntimeError(f"expected 16 dev records, got {len(pool)}")
    return pool


def load_holdout_records(arm_f, base):
    pool = arm_f.load_holdout_pool(base)
    if len(pool) != 50:
        raise RuntimeError(f"expected 50 holdout records, got {len(pool)}")
    return pool


def format_of_record(rec) -> str:
    return rec["item"].answer_format.value


def evaluate_repeated(
    arm_f,
    base,
    client,
    records,
    rules,
    condition,
):
    # isolate this experiment's repeated caches
    old_result_dir = arm_f.RESULT_DIR
    try:
        arm_f.RESULT_DIR = RESULT_DIR
        arm_f.N_REPEATS = N_REPEATS
        return arm_f.evaluate_repeated_records(
            base,
            client,
            records=records,
            rules=rules,
            condition=condition,
        )
    finally:
        arm_f.RESULT_DIR = old_result_dir


def select_phase(v3, arm_f, base, client):
    if SELECTION_SUMMARY.exists() or SELECTED_ADAPTER.exists():
        raise RuntimeError(
            "format-isolated selection is already frozen; "
            "refusing to overwrite"
        )

    rule_ids, source = accepted_rule_ids(v3)
    rules = load_rule_files(arm_f, rule_ids)

    grouped_rules: dict[str, list[dict[str, Any]]] = {}
    for r in rules:
        grouped_rules.setdefault(r["answer_format"], []).append(r)

    dev = load_dev_records(arm_f, base)

    format_results = {}
    selected_rules = []
    selected_formats = []

    for fmt in sorted(grouped_rules):
        fmt_records = [
            rec for rec in dev
            if format_of_record(rec) == fmt
        ]

        fmt_rules = grouped_rules[fmt]

        if len(fmt_records) != 8:
            raise RuntimeError(
                f"{fmt}: expected 8 development cases, got {len(fmt_records)}"
            )

        print(
            f"\n===== FORMAT {fmt} =====\n"
            f"dev cases={len(fmt_records)} rules={len(fmt_rules)}"
        )

        baseline = evaluate_repeated(
            arm_f,
            base,
            client,
            fmt_records,
            [],
            f"format_isolated__{fmt}__baseline",
        )

        candidate = evaluate_repeated(
            arm_f,
            base,
            client,
            fmt_records,
            fmt_rules,
            f"format_isolated__{fmt}__candidate",
        )

        cmp = compare_independent_repeats(
            baseline,
            candidate,
        )

        mean_gain = cmp["mean_delta"] > EPS
        stable = (
            cmp["all_pairs_non_regression_probability"] + EPS
            >= NONREG_THRESHOLD
        )
        parse_safe = cmp["parse_non_regression"]

        passed = bool(
            mean_gain
            and stable
            and parse_safe
        )

        format_results[fmt] = {
            "rule_ids": [r["rule_id"] for r in fmt_rules],
            "comparison": cmp,
            "criteria": {
                "strict_mean_gain": mean_gain,
                "nonreg_probability_ge_0_8": stable,
                "parse_failures_do_not_increase": parse_safe,
                "format_selection_pass": passed,
            },
        }

        print("baseline mean:", cmp["baseline_mean"])
        print("candidate mean:", cmp["candidate_mean"])
        print("delta:", cmp["mean_delta"])
        print(
            "all-pairs non-reg:",
            cmp["all_pairs_non_regression_probability"],
        )
        print("parse safe:", parse_safe)
        print("PASS:", passed)

        if passed:
            selected_formats.append(fmt)
            selected_rules.extend(fmt_rules)

    decision = "ADAPTER" if selected_rules else "NO_ADAPTER"

    adapter_payload = {
        "experiment": "l2_format_isolated_v1",
        "selection_unit": "answer_format",
        "decision": decision,
        "rules": selected_rules,
        "active_formats": selected_formats,
        "source_accepted_rule_ids": rule_ids,
        "holdout_feedback_used": False,
    }

    SKILL_DIR.mkdir(parents=True, exist_ok=True)
    write_json_new(SELECTED_ADAPTER, adapter_payload)

    adapter_sha = sha256(SELECTED_ADAPTER)

    summary = {
        "experiment": "l2_format_isolated_v1",
        "stage": "development_only",
        "method_note": (
            "This protocol revision was designed after observing the prior "
            "L2 development NO_ADAPTER result. The frozen L2 holdout remained "
            "untouched during this selection."
        ),
        "prior_development_result_source": source,
        "prior_accepted_rule_ids": rule_ids,
        "format_results": format_results,
        "selection": {
            "decision": decision,
            "selected_formats": selected_formats,
            "selected_rule_ids": [
                r["rule_id"] for r in selected_rules
            ],
            "selected_adapter_path": str(SELECTED_ADAPTER),
            "selected_adapter_sha256": adapter_sha,
        },
        "holdout_used": False,
    }

    write_json_new(SELECTION_SUMMARY, summary)

    lock = {
        "experiment": "l2_format_isolated_v1",
        "selection_summary_sha256": sha256(SELECTION_SUMMARY),
        "selected_adapter_sha256": adapter_sha,
        "selected_formats": selected_formats,
        "holdout_sha256": sha256(v3.HOLDOUT),
        "holdout_feedback_used": False,
    }

    write_json_new(SELECTION_LOCK, lock)

    print("\n===== FORMAT-ISOLATED SELECTION =====")
    print("decision:", decision)
    print("selected formats:", selected_formats)
    print(
        "selected rule ids:",
        [r["rule_id"] for r in selected_rules],
    )
    print("adapter sha256:", adapter_sha)
    print("selection summary:", SELECTION_SUMMARY)

    return summary


def validate_frozen_selection(v3):
    if not SELECTION_SUMMARY.exists():
        raise RuntimeError("run --phase select first")
    if not SELECTED_ADAPTER.exists():
        raise RuntimeError("missing selected adapter")
    if not SELECTION_LOCK.exists():
        raise RuntimeError("missing selection lock")

    lock = load_json(SELECTION_LOCK)

    if sha256(SELECTION_SUMMARY) != lock["selection_summary_sha256"]:
        raise RuntimeError("selection summary changed after freeze")

    if sha256(SELECTED_ADAPTER) != lock["selected_adapter_sha256"]:
        raise RuntimeError("selected adapter changed after freeze")

    if sha256(v3.HOLDOUT) != lock["holdout_sha256"]:
        raise RuntimeError("frozen L2 holdout changed")

    return lock


def percentile(vals, q):
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)

    if lo == hi:
        return vals[lo]

    w = pos - lo
    return vals[lo] * (1 - w) + vals[hi] * w


def case_table(result):
    table = {}

    for run in result["runs"]:
        for row in run["items"]:
            rec = table.setdefault(
                row["id"],
                {
                    "answer_format": row["answer_format"],
                    "chance": float(row.get("chance", 0.0)),
                    "scores": [],
                },
            )
            rec["scores"].append(effective_score(row))

    for cid, rec in table.items():
        if len(rec["scores"]) != N_REPEATS:
            raise RuntimeError(
                f"{cid}: expected {N_REPEATS} repeats"
            )
        rec["mean_effective_score"] = (
            sum(rec["scores"]) / len(rec["scores"])
        )

    return table


def fixed_from_cases(ids, table):
    ms = sum(
        table[c]["mean_effective_score"]
        for c in ids
    ) / len(ids)

    mc = sum(
        table[c]["chance"]
        for c in ids
    ) / len(ids)

    return (ms - mc) / (1 - mc)


def bootstrap_delta(bt, ct, ids, seed):
    pb = fixed_from_cases(ids, bt)
    pc = fixed_from_cases(ids, ct)

    rng = random.Random(seed)
    ds = []
    n = len(ids)

    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            ids[rng.randrange(n)]
            for _ in range(n)
        ]

        ds.append(
            fixed_from_cases(sample, ct)
            - fixed_from_cases(sample, bt)
        )

    lo = percentile(ds, 0.025)
    hi = percentile(ds, 0.975)

    return {
        "baseline_score": pb,
        "candidate_score": pc,
        "mean_delta": pc - pb,
        "ci_95": [lo, hi],
        "ci_excludes_zero_positive": lo > 0,
        "bootstrap_probability_delta_gt_0": (
            sum(d > 0 for d in ds) / len(ds)
        ),
        "case_count": len(ids),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "pairing_unit": "case_id",
    }


def holdout_phase(v3, arm_f, base, client):
    if HOLDOUT_SUMMARY.exists():
        raise RuntimeError(
            "holdout summary already exists; refusing overwrite"
        )

    lock = validate_frozen_selection(v3)
    payload = load_json(SELECTED_ADAPTER)

    if payload["decision"] != "ADAPTER":
        raise RuntimeError(
            "format-isolated development selected NO_ADAPTER; "
            "do not evaluate holdout"
        )

    rules = payload["rules"]
    active_formats = payload["active_formats"]

    hold = load_holdout_records(arm_f, base)

    baseline = evaluate_repeated(
        arm_f,
        base,
        client,
        hold,
        [],
        "format_isolated_holdout__baseline",
    )

    candidate = evaluate_repeated(
        arm_f,
        base,
        client,
        hold,
        rules,
        "format_isolated_holdout__candidate",
    )

    bt = case_table(baseline)
    ct = case_table(candidate)

    all_ids = baseline["ordered_ids"]
    active_ids = [
        cid for cid in all_ids
        if bt[cid]["answer_format"] in active_formats
    ]

    combined = bootstrap_delta(
        bt,
        ct,
        active_ids,
        BOOTSTRAP_SEED,
    )

    by_format = {}
    for i, fmt in enumerate(active_formats, start=1):
        ids = [
            cid for cid in active_ids
            if bt[cid]["answer_format"] == fmt
        ]

        by_format[fmt] = bootstrap_delta(
            bt,
            ct,
            ids,
            BOOTSTRAP_SEED + i,
        )

    baseline_parse = sum(
        int(row.get("parse_error") is not None)
        for run in baseline["runs"]
        for row in run["items"]
        if row["answer_format"] in active_formats
    )

    candidate_parse = sum(
        int(row.get("parse_error") is not None)
        for run in candidate["runs"]
        for row in run["items"]
        if row["answer_format"] in active_formats
    )

    combined_positive = combined["mean_delta"] > EPS

    every_nonreg = all(
        x["mean_delta"] >= -EPS
        for x in by_format.values()
    )

    any_gain = any(
        x["mean_delta"] > EPS
        for x in by_format.values()
    )

    parse_safe = candidate_parse <= baseline_parse

    generalization_pass = bool(
        combined_positive
        and every_nonreg
        and any_gain
        and parse_safe
    )

    out = {
        "experiment": "l2_format_isolated_v1",
        "stage": "frozen_holdout",
        "frozen_selection": {
            "selected_formats": active_formats,
            "selected_adapter_sha256": (
                lock["selected_adapter_sha256"]
            ),
            "holdout_sha256": lock["holdout_sha256"],
        },
        "case_level_bootstrap": {
            "combined_active_formats": combined,
            "by_format": by_format,
        },
        "parse_failures": {
            "baseline_total": baseline_parse,
            "candidate_total": candidate_parse,
            "non_regression_pass": parse_safe,
        },
        "success_criteria": {
            "combined_active_format_mean_delta_gt_0": (
                combined_positive
            ),
            "every_active_format_mean_delta_ge_0": (
                every_nonreg
            ),
            "at_least_one_active_format_mean_delta_gt_0": (
                any_gain
            ),
            "parse_failures_do_not_increase": parse_safe,
            "l2_format_isolated_generalization_pass": (
                generalization_pass
            ),
        },
        "statistical_support": {
            "combined_bootstrap_95ci_excludes_zero_positive": (
                combined["ci_excludes_zero_positive"]
            )
        },
        "holdout_feedback_used": False,
    }

    write_json_new(HOLDOUT_SUMMARY, out)

    print("\n===== L2 FORMAT-ISOLATED HOLDOUT =====")
    print("active formats:", active_formats)
    print("baseline:", combined["baseline_score"])
    print("skill:", combined["candidate_score"])
    print("delta:", combined["mean_delta"])
    print("95% CI:", combined["ci_95"])
    print("by format:", json.dumps(by_format, indent=2))
    print("generalization pass:", generalization_pass)
    print(
        "bootstrap 95% CI positive:",
        combined["ci_excludes_zero_positive"],
    )
    print("summary:", HOLDOUT_SUMMARY)


def preflight(v3, arm_f, base):
    v3.validate_manifest_lock()

    ids, source = accepted_rule_ids(v3)
    rules = load_rule_files(arm_f, ids)
    dev = load_dev_records(arm_f, base)

    print(
        json.dumps(
            {
                "status": "READY",
                "experiment": "l2_format_isolated_v1",
                "method_revision_after_dev_result": True,
                "holdout_used_for_method_revision": False,
                "accepted_rule_source": source,
                "accepted_rule_ids": ids,
                "accepted_rule_formats": dict(
                    sorted(
                        Counter(
                            r["answer_format"]
                            for r in rules
                        ).items()
                    )
                ),
                "development_case_count": len(dev),
                "development_format_counts": dict(
                    sorted(
                        Counter(
                            format_of_record(rec)
                            for rec in dev
                        ).items()
                    )
                ),
                "repeats": N_REPEATS,
                "nonreg_threshold": NONREG_THRESHOLD,
                "holdout_sha256": sha256(v3.HOLDOUT),
                "holdout_feedback_allowed": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        choices=["preflight", "select", "holdout"],
        required=True,
    )
    args = ap.parse_args()

    v3 = load_v3()
    v3.validate_manifest_lock()

    arm_f = v3.import_arm_f()
    base = arm_f.load_base()

    if args.phase == "preflight":
        preflight(v3, arm_f, base)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    client = OpenAI()

    if args.phase == "select":
        select_phase(
            v3,
            arm_f,
            base,
            client,
        )
        return

    holdout_phase(
        v3,
        arm_f,
        base,
        client,
    )


if __name__ == "__main__":
    main()
