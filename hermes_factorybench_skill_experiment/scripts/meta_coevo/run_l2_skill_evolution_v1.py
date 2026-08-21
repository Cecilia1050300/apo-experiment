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
    return mod


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
