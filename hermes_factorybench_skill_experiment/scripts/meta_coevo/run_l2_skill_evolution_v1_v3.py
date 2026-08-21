#!/usr/bin/env python3
"""
L2-specific Skill Evolution v1
==============================

Wraps the existing successful Arm-F format-stratified LOOCV mechanism,
redirecting it to frozen Level-2 development/holdout manifests.

Key differences from original Arm F:
- L2-only manifests (8+8 development, 50 frozen holdout)
- manifest SHA256 lock validation before every phase
- corrected final gate: only formats where the candidate Skill is ACTIVE
  may causally veto/select the Skill
- repeated baseline/candidate API runs are treated as independent stochastic
  samples (no run-index causal pairing)
- holdout additionally reports paired-by-case bootstrap 95% CI
- no holdout feedback into rule generation/selection/revision
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
from pathlib import Path
from typing import Any

from openai import OpenAI

ROOT = Path('/home/training/automatic_prompt_engineer/hermes_factorybench_skill_experiment')
ARM_F_SCRIPT = ROOT / 'scripts/meta_coevo/run_format_stratified_loocv.py'
MANIFEST_DIR = ROOT / 'data_manifests/meta_m1'

FOLD_A = MANIFEST_DIR / 'factorybench_l2_skill_dev_fold_a.json'
FOLD_B = MANIFEST_DIR / 'factorybench_l2_skill_dev_fold_b.json'
HOLDOUT = MANIFEST_DIR / 'factorybench_l2_skill_holdout.json'
LOCK = MANIFEST_DIR / 'factorybench_l2_skill_v1_lock.json'

ARM = 'l2_skill_evolution_v1'
TASK = 'm1_factorybench_l123'
RESULT_DIR = ROOT / 'results/meta_coevo/l2_skill_evolution_v1'
TRACE_DIR = ROOT / 'traces/meta_coevo/l2_skill_evolution_v1'
SKILL_ROOT = ROOT / 'prompts/adapters/meta_coevo_l2_skill_v1'
RULE_ROOT = ROOT / 'prompts/meta_coevo/rules/l2_skill_v1'

N_REPEATS = int(os.getenv('N_REPEATS', '5'))
BOOTSTRAP_SAMPLES = int(os.getenv('BOOTSTRAP_SAMPLES', '10000'))
BOOTSTRAP_SEED = 20260821
EPS = 1e-12


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json_new(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f'refusing overwrite: {path}')
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest_lock() -> dict[str, Any]:
    if not LOCK.exists():
        raise RuntimeError(f'missing lock: {LOCK}')
    lock = load_json(LOCK)
    checks = [
        ('fold_a_sha256', FOLD_A),
        ('fold_b_sha256', FOLD_B),
        ('holdout_sha256', HOLDOUT),
    ]
    for key, path in checks:
        if not path.exists():
            raise RuntimeError(f'missing manifest: {path}')
        actual = sha256(path)
        expected = lock[key]
        if actual != expected:
            raise RuntimeError(
                f'frozen manifest changed: {path.name}\n'
                f'expected={expected}\nactual={actual}'
            )
    return lock


def import_arm_f():
    if not ARM_F_SCRIPT.exists():
        raise RuntimeError(f'missing Arm-F runner: {ARM_F_SCRIPT}')
    spec = importlib.util.spec_from_file_location('arm_f_for_l2_v1', ARM_F_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError('failed to import Arm F')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Redirect the exact Arm-F mechanism to new L2-only artifacts.
    mod.ARM = ARM
    mod.TASK = TASK
    mod.RESULT_DIR = RESULT_DIR
    mod.TRACE_DIR = TRACE_DIR
    mod.SKILL_ROOT = SKILL_ROOT
    mod.RULE_ROOT = RULE_ROOT
    mod.N_REPEATS = N_REPEATS

    # Freeze the L2 manifests used by Arm F.
    mod.manifests = lambda base: {
        'fold_a': FOLD_A,
        'fold_b': FOLD_B,
        'holdout': HOLDOUT,
    }

    # Fix original Arm-F causal gate: inactive formats cannot veto the Skill.
    mod.final_gate = corrected_active_format_final_gate_factory(mod)
    install_rule_generator_metadata_recovery(mod)
    return mod



def install_rule_generator_metadata_recovery(arm_f):
    """
    Resume or recover an existing Arm-F rule-generation trace safely.

    Supported cases
    ---------------
    1. parsed_output.json exists:
       - accept only if validation is already clean, OR
       - the only errors are metadata-copy errors:
         source_case_id mismatch / answer_format mismatch.

    2. raw_output.txt exists but parsed_output.json is missing:
       - parse and validate that exact existing raw output locally;
       - DO NOT call the rule model again;
       - materialize parsed_output.json and candidate rules only if:
         a) validation is clean, or
         b) errors are metadata-only.

    Any semantic/schema error beyond metadata-copy errors is still rejected.
    Raw LLM text is never modified.
    """
    original = arm_f.call_rule_generator

    allowed_metadata_errors = {
        "source_case_id mismatch",
        "answer_format mismatch",
    }

    def materialize_candidates(normalized):
        for r in normalized:
            arm_f.write_json(
                arm_f.rule_dir()
                / "candidates"
                / f"{arm_f.safe_name(r['rule_id'])}.json",
                r,
            )

    def validate_existing_raw(base, source_row, raw_path, parsed_path):
        sid = source_row["id"]
        fmt = source_row["answer_format"]

        raw = raw_path.read_text(encoding="utf-8")

        try:
            parsed = arm_f.json.loads(raw.strip())
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

        if len(rules) > arm_f.MAX_RULES_PER_SOURCE_CASE:
            errors.append("too many rules")

        holdout_ids = [
            x["id"]
            for x in arm_f.load_json(
                arm_f.manifests(base)["holdout"]
            )["items"]
        ]

        normalized = []
        for i, r in enumerate(rules, 1):
            if not isinstance(r, dict):
                errors.append(f"rule {i} not object")
                continue

            rid = str(r.get("rule_id") or f"R{i}")
            text = str(r.get("rule_text") or "").strip()

            for err in arm_f.validate_rule_text(
                text,
                source_row,
                holdout_ids,
            ):
                errors.append(f"{rid}: {err}")

            normalized.append(
                {
                    "rule_id": (
                        f"{arm_f.safe_name(fmt)}__"
                        f"{sid[:8]}__{rid}"
                    ),
                    "source_case_id": sid,
                    "source_fold": source_row["_fold"],
                    "answer_format": fmt,
                    "category": str(
                        r.get("category") or "uncategorized"
                    ),
                    "rule_text": text,
                    "rationale": str(r.get("rationale") or ""),
                }
            )

        recoverable = (
            not errors
            or set(errors).issubset(allowed_metadata_errors)
        )

        envelope = {
            **parsed,
            "_trace_validation": {
                "valid": not errors,
                "errors": errors,
                "model": arm_f.RULE_MODEL,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_input_tokens": 0,
                    "reasoning_tokens": 0,
                    "calls": 0,
                },
                "cost": 0.0,
                "wall_time_seconds": 0.0,
                "resumed_from_existing_raw_trace": True,
            },
            "_normalized_rules": normalized,
        }

        # parsed_output did not exist, so writing it does not overwrite
        # prior experiment evidence.
        arm_f.write_json(parsed_path, envelope)

        if not recoverable:
            raise RuntimeError(
                f"existing raw rule trace for {sid} is not safely "
                f"recoverable: {errors}"
            )

        materialize_candidates(normalized)

        return normalized, errors

    def recover_existing(base, source_row):
        sid = source_row["id"]
        fmt = source_row["answer_format"]

        tdir = (
            arm_f.TRACE_DIR
            / arm_f.TASK
            / "rule_generation"
        )
        prefix = arm_f.safe_name(f"{fmt}__{sid}")

        raw_path = tdir / f"{prefix}_raw_output.txt"
        parsed_path = tdir / f"{prefix}_parsed_output.json"

        # Case A: parsed trace already exists.
        if parsed_path.exists():
            envelope = arm_f.load_json(parsed_path)
            validation = envelope.get("_trace_validation") or {}
            errors = validation.get("errors") or []
            normalized = envelope.get("_normalized_rules")

            if not isinstance(normalized, list):
                raise RuntimeError(
                    f"existing parsed trace for {sid} has no "
                    "_normalized_rules"
                )

            if errors and not set(errors).issubset(
                allowed_metadata_errors
            ):
                raise RuntimeError(
                    f"existing parsed rule trace for {sid} has "
                    f"non-recoverable validation errors: {errors}"
                )

            for r in normalized:
                if r.get("source_case_id") != sid:
                    raise RuntimeError(
                        f"normalized source_case_id mismatch for {sid}"
                    )
                if r.get("answer_format") != fmt:
                    raise RuntimeError(
                        f"normalized answer_format mismatch for {sid}"
                    )

            materialize_candidates(normalized)

            if errors:
                audit = {
                    "recovered": True,
                    "reason": "metadata_only_schema_mismatch",
                    "original_validation_errors": errors,
                    "source_case_id": sid,
                    "answer_format": fmt,
                    "raw_output_modified": False,
                    "rule_text_modified": False,
                    "holdout_feedback_used": False,
                }
                arm_f.write_json(
                    tdir
                    / f"{prefix}_metadata_recovery_audit.json",
                    audit,
                )

                print(
                    f"[metadata-recovery] {sid}: "
                    f"reused parsed trace; errors={errors}"
                )
            else:
                print(
                    f"[trace-resume] {sid}: "
                    "reused valid parsed trace"
                )

            return normalized

        # Case B: API returned and raw trace exists, but run stopped
        # before parsed_output.json was written.
        if raw_path.exists():
            normalized, errors = validate_existing_raw(
                base,
                source_row,
                raw_path,
                parsed_path,
            )

            audit = {
                "recovered": True,
                "reason": (
                    "resume_existing_raw_trace"
                    if not errors
                    else "raw_trace_with_metadata_only_schema_mismatch"
                ),
                "original_validation_errors": errors,
                "source_case_id": sid,
                "answer_format": fmt,
                "raw_output_modified": False,
                "rule_text_modified": False,
                "rule_model_recalled": False,
                "holdout_feedback_used": False,
            }
            arm_f.write_json(
                tdir
                / f"{prefix}_raw_trace_resume_audit.json",
                audit,
            )

            print(
                f"[raw-trace-resume] {sid}: "
                f"validated existing raw output; errors={errors}"
            )

            return normalized

        return None

    def patched(base, client, *, source_row):
        recovered = recover_existing(base, source_row)
        if recovered is not None:
            return recovered

        try:
            return original(
                base,
                client,
                source_row=source_row,
            )
        except RuntimeError:
            # Original Arm-F writes raw/parsed traces before some
            # validation failures. Re-check once, without recalling
            # the model if a reusable trace now exists.
            recovered = recover_existing(base, source_row)
            if recovered is not None:
                return recovered
            raise

    arm_f.call_rule_generator = patched

def effective_score(row: dict[str, Any]) -> float:
    clean = (
        row.get('parse_error') is None
        and isinstance(row.get('score'), (int, float))
        and math.isfinite(float(row['score']))
    )
    return float(row['score']) if clean else 0.0


def fixed_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError('empty rows')
    scores = [effective_score(r) for r in rows]
    chances = [float(r.get('chance', 0.0)) for r in rows]
    mc = sum(chances) / len(chances)
    return (sum(scores) / len(scores) - mc) / (1.0 - mc)


def repeated_subset(result: dict[str, Any], formats: set[str]) -> dict[str, Any]:
    run_scores = []
    parse_total = 0
    by_format: dict[str, list[float]] = {}

    for run in result['runs']:
        rows = [r for r in run['items'] if r['answer_format'] in formats]
        if not rows:
            continue
        run_scores.append(fixed_score(rows))
        parse_total += sum(r.get('parse_error') is not None for r in rows)
        for fmt in sorted(formats):
            fr = [r for r in rows if r['answer_format'] == fmt]
            if fr:
                by_format.setdefault(fmt, []).append(fixed_score(fr))

    if not run_scores:
        return {'valid': False, 'reason': 'no active-format rows'}

    return {
        'valid': True,
        'scores': run_scores,
        'mean_score': sum(run_scores) / len(run_scores),
        'median_score': float(statistics.median(run_scores)),
        'parse_failures_total': parse_total,
        'mean_by_format': {
            fmt: sum(vals) / len(vals)
            for fmt, vals in sorted(by_format.items())
        },
    }


def compare_independent_repeats(b: dict[str, Any], c: dict[str, Any]) -> dict[str, Any]:
    if not b.get('valid') or not c.get('valid'):
        return {'valid': False, 'reason': 'invalid subset'}
    all_deltas = [cv - bv for bv in b['scores'] for cv in c['scores']]
    return {
        'valid': True,
        'baseline_scores': b['scores'],
        'candidate_scores': c['scores'],
        'baseline_mean': b['mean_score'],
        'candidate_mean': c['mean_score'],
        'mean_delta': c['mean_score'] - b['mean_score'],
        'baseline_median': b['median_score'],
        'candidate_median': c['median_score'],
        'median_delta': c['median_score'] - b['median_score'],
        'all_pairs_non_regression_probability': (
            sum(d >= -EPS for d in all_deltas) / len(all_deltas)
        ),
        'all_pairs_win_probability': (
            sum(d > EPS for d in all_deltas) / len(all_deltas)
        ),
        'run_index_pairing_used': False,
    }


def corrected_active_format_final_gate_factory(arm_f):
    def gate(base_res, cand_res):
        active_formats = {
            row['answer_format']
            for run in cand_res['runs']
            for row in run['items']
            if row.get('skill_active')
        }
        if not active_formats:
            return {
                'eligible': False,
                'reasons': ['candidate Skill inactive on all development formats'],
                'active_formats': [],
            }

        b = repeated_subset(base_res, active_formats)
        c = repeated_subset(cand_res, active_formats)
        cmp = compare_independent_repeats(b, c)
        reasons = []

        if not cmp.get('valid'):
            return {
                'eligible': False,
                'reasons': [cmp.get('reason', 'invalid')],
                'active_formats': sorted(active_formats),
            }

        if cmp['mean_delta'] <= EPS:
            reasons.append('no strict active-format mean gain')

        # Same 0.8 stability threshold as Arm F, but use all-pairs descriptively
        # instead of pretending independent API runs are paired by index.
        if (
            cmp['all_pairs_non_regression_probability'] + EPS
            < arm_f.FINAL_NONREG_RATE
        ):
            reasons.append('active-format non-regression probability too low')

        per_format = {}
        any_gain = False
        for fmt in sorted(active_formats):
            bv = b['mean_by_format'][fmt]
            cv = c['mean_by_format'][fmt]
            delta = cv - bv
            per_format[fmt] = {
                'baseline': bv,
                'candidate': cv,
                'delta': delta,
            }
            if delta < -EPS:
                reasons.append(f'active format regression: {fmt}')
            if delta > EPS:
                any_gain = True

        if not any_gain:
            reasons.append('no active format has positive mean gain')

        return {
            'eligible': not reasons,
            'reasons': reasons,
            'active_formats': sorted(active_formats),
            'comparison': cmp,
            'per_format': per_format,
            'causal_gate_active_formats_only': True,
        }
    return gate


def percentile(vals: list[float], q: float) -> float:
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def case_table(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for run in result['runs']:
        for row in run['items']:
            rec = table.setdefault(
                row['id'],
                {
                    'answer_format': row['answer_format'],
                    'chance': float(row.get('chance', 0.0)),
                    'scores': [],
                },
            )
            rec['scores'].append(effective_score(row))
    for cid, rec in table.items():
        if len(rec['scores']) != N_REPEATS:
            raise RuntimeError(f'{cid}: expected {N_REPEATS} repeats')
        rec['mean_effective_score'] = sum(rec['scores']) / len(rec['scores'])
    return table


def fixed_from_case_ids(ids: list[str], table: dict[str, dict[str, Any]]) -> float:
    scores = [table[c]['mean_effective_score'] for c in ids]
    chances = [table[c]['chance'] for c in ids]
    mc = sum(chances) / len(chances)
    return (sum(scores) / len(scores) - mc) / (1.0 - mc)


def bootstrap_delta(bt, ct, ids, seed):
    pb = fixed_from_case_ids(ids, bt)
    pc = fixed_from_case_ids(ids, ct)
    rng = random.Random(seed)
    deltas = []
    n = len(ids)
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [ids[rng.randrange(n)] for _ in range(n)]
        deltas.append(
            fixed_from_case_ids(sample, ct) - fixed_from_case_ids(sample, bt)
        )
    lo = percentile(deltas, 0.025)
    hi = percentile(deltas, 0.975)
    return {
        'baseline_score': pb,
        'candidate_score': pc,
        'mean_delta': pc - pb,
        'ci_95': [lo, hi],
        'ci_excludes_zero_positive': lo > 0,
        'bootstrap_probability_delta_gt_0': (
            sum(d > 0 for d in deltas) / len(deltas)
        ),
        'case_count': len(ids),
        'bootstrap_samples': BOOTSTRAP_SAMPLES,
        'pairing_unit': 'case_id',
    }


def holdout_bootstrap_report(arm_f, holdout_summary):
    if holdout_summary['selection_decision'] != 'ADAPTER':
        return {
            'available': False,
            'reason': 'development selected NO_ADAPTER',
        }

    b = holdout_summary['baseline']
    c = holdout_summary['selected_adapter']
    bt = case_table(b)
    ct = case_table(c)
    ids = b['ordered_ids']

    payload = load_json(arm_f.skill_dir() / 'selected_adapter.json')
    active_formats = sorted({r['answer_format'] for r in payload['rules']})
    active_ids = [cid for cid in ids if bt[cid]['answer_format'] in active_formats]

    combined_active = bootstrap_delta(bt, ct, active_ids, BOOTSTRAP_SEED)
    by_format = {}
    for i, fmt in enumerate(active_formats, 1):
        fmt_ids = [cid for cid in ids if bt[cid]['answer_format'] == fmt]
        by_format[fmt] = bootstrap_delta(bt, ct, fmt_ids, BOOTSTRAP_SEED + i)

    baseline_parse = sum(
        int(row.get('parse_error') is not None)
        for run in b['runs']
        for row in run['items']
        if row['answer_format'] in active_formats
    )
    candidate_parse = sum(
        int(row.get('parse_error') is not None)
        for run in c['runs']
        for row in run['items']
        if row['answer_format'] in active_formats
    )

    combined_positive = combined_active['mean_delta'] > EPS
    every_nonreg = all(x['mean_delta'] >= -EPS for x in by_format.values())
    any_gain = any(x['mean_delta'] > EPS for x in by_format.values())
    parse_safe = candidate_parse <= baseline_parse
    generalization_pass = bool(
        combined_positive and every_nonreg and any_gain and parse_safe
    )

    return {
        'available': True,
        'active_formats': active_formats,
        'combined_active_formats': combined_active,
        'by_format': by_format,
        'active_format_parse_failures': {
            'baseline_total': baseline_parse,
            'candidate_total': candidate_parse,
            'non_regression_pass': parse_safe,
        },
        'frozen_success_criteria': {
            'combined_active_format_mean_delta_gt_0': combined_positive,
            'every_active_format_mean_delta_ge_0': every_nonreg,
            'at_least_one_active_format_mean_delta_gt_0': any_gain,
            'parse_failures_do_not_increase': parse_safe,
            'l2_skill_generalization_pass': generalization_pass,
        },
        'statistical_support': {
            'combined_bootstrap_95ci_excludes_zero_positive': (
                combined_active['ci_excludes_zero_positive']
            )
        },
        'no_holdout_feedback': True,
    }


def preflight(arm_f):
    lock = validate_manifest_lock()
    base = arm_f.load_base()
    info = arm_f.preflight(base)
    info.update({
        'experiment': ARM,
        'level': 2,
        'manifest_lock_valid': True,
        'fold_a_sha256': lock['fold_a_sha256'],
        'fold_b_sha256': lock['fold_b_sha256'],
        'holdout_sha256': lock['holdout_sha256'],
        'development_case_count': lock['development_case_count'],
        'holdout_case_count': lock['holdout_case_count'],
        'corrected_active_format_final_gate': True,
        'run_index_pairing_used_for_final_gate': False,
    })
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--phase',
        choices=['preflight', 'development', 'holdout'],
        required=True,
    )
    args = parser.parse_args()

    validate_manifest_lock()
    arm_f = import_arm_f()
    base = arm_f.load_base()

    if args.phase == 'preflight':
        print(json.dumps(preflight(arm_f), indent=2, ensure_ascii=False))
        return

    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY missing')

    client = OpenAI()

    if args.phase == 'development':
        result = arm_f.development(base, client)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return

    # Frozen holdout: Arm F evaluates; wrapper adds case-level bootstrap only.
    result = arm_f.holdout(base, client)
    extra = holdout_bootstrap_report(arm_f, result)
    out = {
        'experiment': ARM,
        'arm_f_holdout': result,
        'case_level_bootstrap_analysis': extra,
        'holdout_feedback_used': False,
    }
    report_path = RESULT_DIR / 'holdout' / 'l2_skill_evolution_v1_bootstrap_summary.json'
    write_json_new(report_path, out)

    print('\n===== L2 SKILL EVOLUTION V1 HOLDOUT =====')
    print('selection:', result['selection_decision'])
    if extra.get('available'):
        x = extra['combined_active_formats']
        print('active formats:', extra['active_formats'])
        print('baseline:', x['baseline_score'])
        print('skill:', x['candidate_score'])
        print('delta:', x['mean_delta'])
        print('95% CI:', x['ci_95'])
        print(
            'generalization pass:',
            extra['frozen_success_criteria']['l2_skill_generalization_pass'],
        )
        print(
            'bootstrap 95% CI positive:',
            extra['statistical_support'][
                'combined_bootstrap_95ci_excludes_zero_positive'
            ],
        )
    else:
        print('bootstrap unavailable:', extra.get('reason'))
    print('summary:', report_path)


if __name__ == '__main__':
    main()
