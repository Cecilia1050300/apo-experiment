#!/usr/bin/env python3
"""Build completed cross-task summary using frozen L4/Causal results plus L123 continuation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path('/home/training/automatic_prompt_engineer/hermes_factorybench_skill_experiment')
SUMMARY_OUT = ROOT / 'results/meta_m0/cross_task_summary_completed.json'
REPORT_OUT = ROOT / 'reports/meta_m0/cross_task_summary_completed.md'
REGISTRY_OUT = ROOT / 'results/meta_m0/artifact_registry_completed.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing to overwrite different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, payload: Any) -> None:
    write_new(path, (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def calls(result: dict[str, Any]) -> int:
    usage = result.get('tokens_used') or {}
    return int((usage.get('candidate') or {}).get('calls', 0)) + sum(int(row.get('calls', 0)) for row in (usage.get('judges') or {}).values())


def metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in ('canonical_score', 'fixed_cardinality_score', 'parse_failures', 'by_level', 'by_split', 'by_dataset', 'by_answer_format', 'by_subgroup', 'tokens_used', 'cost', 'wall_time_seconds') if key in result}


def paired(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    require_ids = left['ordered_ids']
    right_by = {row['id']: row for row in right['items']}
    counts = {'improved': 0, 'worse': 0, 'unchanged': 0, 'invalid': 0}
    changed = []
    for row in left['items']:
        other = right_by[row['id']]
        if row.get('parse_error') is not None or other.get('parse_error') is not None:
            status = 'invalid'
        else:
            delta = float(other['score']) - float(row['score'])
            status = 'improved' if delta > 0 else 'worse' if delta < 0 else 'unchanged'
        counts[status] += 1
        if row.get('raw_output') != other.get('raw_output'):
            changed.append({'id': row['id'], 'left_output': row.get('raw_output'), 'right_output': other.get('raw_output'), 'left_score': row.get('score'), 'right_score': other.get('score'), 'status': status})
    return {'exact_ordered_id_match': require_ids == right['ordered_ids'], 'counts': counts, 'raw_output_changed_count': len(changed), 'changed_cases': changed}


def main() -> None:
    for path in (SUMMARY_OUT, REPORT_OUT, REGISTRY_OUT):
        if path.exists():
            raise RuntimeError(f'target exists: {path}')
    original = load('results/meta_m0/cross_task_summary.json')
    dev = load('results/meta_m0/development/factorybench_l123_completion_summary_v2data.json')
    hold = load('results/meta_m0/holdout/factorybench_l123_completion_summary_v2data.json')
    selection = load('prompts/adapters/factorybench_l123_completion/selection_completion.json')
    l123_base = load('results/meta_m0/holdout/factorybench_l123_completion_baseline_completion_v2data.json')
    l123_core = load('results/meta_m0/holdout/factorybench_l123_completion_core_only_completion_v2data.json')
    l123_adapter = load('results/meta_m0/holdout/factorybench_l123_completion_selected_adapter_completion_v2data.json') if selection['decision'] == 'ADAPTER' else None
    l4_base = load('results/meta_m0/holdout/factorybench_l4_baseline.json')
    l4_core = load('results/meta_m0/holdout/factorybench_l4_core_only.json')
    causal_base = load('results/meta_m0/holdout/causal_judgment_baseline.json')
    causal_core = load('results/meta_m0/holdout/causal_judgment_core_only.json')
    causal_adapter = load('results/meta_m0/holdout/causal_judgment_selected_adapter.json')

    core_deltas = {
        'factorybench_l123': l123_core['fixed_cardinality_score'] - l123_base['fixed_cardinality_score'],
        'factorybench_l4': l4_core['fixed_cardinality_score'] - l4_base['fixed_cardinality_score'],
        'causal_judgment': causal_core['fixed_cardinality_score'] - causal_base['fixed_cardinality_score'],
    }
    core_positive = [task for task, delta in core_deltas.items() if delta > 0]
    core_negative = [task for task, delta in core_deltas.items() if delta < 0]
    core_parse = sum(result['parse_failures'] for result in (l123_core, l4_core, causal_core))
    if len(core_positive) >= 2 and any(task.startswith('factorybench') for task in core_positive) and not core_negative and core_parse == 0:
        core_verdict = 'CORE_POSITIVE'
    elif core_positive and core_negative:
        core_verdict = 'CORE_MIXED'
    elif not core_positive and not core_negative:
        core_verdict = 'CORE_NO_EFFECT'
    else:
        core_verdict = 'CORE_NEGATIVE'

    adapter_deltas = {
        'factorybench_l123': (l123_adapter['fixed_cardinality_score'] - l123_core['fixed_cardinality_score']) if l123_adapter else 0.0,
        'factorybench_l4': 0.0,
        'causal_judgment': causal_adapter['fixed_cardinality_score'] - causal_core['fixed_cardinality_score'],
    }
    adapter_positive = [task for task, delta in adapter_deltas.items() if delta > 0]
    adapter_negative = [task for task, delta in adapter_deltas.items() if delta < 0]
    adapter_verdict = 'ADAPTER_EVIDENCE' if len(adapter_positive) >= 2 else 'NO_ADAPTER_EVIDENCE'
    all_parse = sum(result['parse_failures'] for result in (l123_base, l123_core, l4_base, l4_core, causal_base, causal_core, causal_adapter) if result is not None)
    if l123_adapter:
        all_parse += l123_adapter['parse_failures']
    if all_parse:
        m0_verdict = 'M0_INVALID'
        m0_reason = f'Completed protocol contains {all_parse} parse failures in frozen baseline/Core/selected conditions.'
    elif len(adapter_positive) >= 2 and any(task.startswith('factorybench') for task in adapter_positive) and not adapter_negative:
        m0_verdict = 'M0_PROMISING'
        m0_reason = 'At least two selected adapters improve Core-only, including FactoryBench, with no adapter regression or parse failure.'
    elif adapter_positive and adapter_negative:
        m0_verdict = 'M0_MIXED'
        m0_reason = 'At least one adapter improves while another regresses.'
    elif not adapter_positive:
        m0_verdict = 'M0_NO_EFFECT'
        m0_reason = 'No selected adapter improves Core-only.'
    else:
        m0_verdict = 'M0_MIXED'
        m0_reason = 'Only one selected adapter improves Core-only.'

    new_result_paths = sorted((ROOT / 'results/meta_m0/development').glob('factorybench_l123_completion_*_v2data.json')) + sorted((ROOT / 'results/meta_m0/holdout').glob('factorybench_l123_completion_*_v2data.json'))
    additional_calls = 0
    additional_cost = 0.0
    additional_wall = 0.0
    for path in new_result_paths:
        result = json.loads(path.read_text(encoding='utf-8'))
        if 'tokens_used' in result:
            additional_calls += calls(result)
            additional_cost += float(result['cost'])
            additional_wall += float(result['wall_time_seconds'])
    trace_stats = []
    for path in sorted((ROOT / 'traces/meta_m0/factorybench_l123_completion').glob('m0_round_*_parsed_output_completion.json')):
        trace = json.loads(path.read_text(encoding='utf-8'))['_trace_validation']
        additional_calls += 1
        additional_cost += float(trace['cost'])
        additional_wall += float(trace['wall_time_seconds'])
        trace_stats.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'usage': trace['usage'], 'cost': trace['cost'], 'wall_time_seconds': trace['wall_time_seconds']})

    frozen_reuse = {}
    for rel in (
        'results/meta_m0/holdout/factorybench_l4_baseline.json', 'results/meta_m0/holdout/factorybench_l4_core_only.json', 'results/meta_m0/holdout/factorybench_l4_summary.json',
        'results/meta_m0/holdout/causal_judgment_baseline.json', 'results/meta_m0/holdout/causal_judgment_core_only.json', 'results/meta_m0/holdout/causal_judgment_selected_adapter.json', 'results/meta_m0/holdout/causal_judgment_summary.json',
    ):
        frozen_reuse[rel] = sha(ROOT / rel)

    summary = {
        'experiment': 'Meta-Prompt M0 cross-task smoke — corrected data-availability continuation',
        'relationship_to_original': {'original_summary_path': 'results/meta_m0/cross_task_summary.json', 'original_summary_sha256': sha(ROOT / 'results/meta_m0/cross_task_summary.json'), 'original_verdict': original['verdicts']['meta_prompt_result'], 'original_report_remains_authoritative_for_incomplete_attempt': True},
        'frozen_artifacts': {'meta_prompt_m0': {'sha256': sha(ROOT / 'prompts/meta/meta_prompt_m0_candidate.txt'), 'bytes': (ROOT / 'prompts/meta/meta_prompt_m0_candidate.txt').stat().st_size}, 'core_v0': {'sha256': sha(ROOT / 'prompts/core/evidence_grounded_core_v0.txt'), 'bytes': (ROOT / 'prompts/core/evidence_grounded_core_v0.txt').stat().st_size}},
        'contamination_correction': {'original_registry_sha256': sha(ROOT / 'data_manifests/meta_m0/contamination_registry.json'), 'corrected_registry_sha256': sha(ROOT / 'data_manifests/meta_m0/contamination_registry_v2.json'), 'audit_report_sha256': sha(ROOT / 'reports/meta_m0/contamination_registry_audit.md')},
        'factorybench_l123': {
            'development': dev, 'selection': selection,
            'holdout': {'baseline': metrics(l123_base), 'core_only': metrics(l123_core), 'selected_adapter': metrics(l123_adapter) if l123_adapter else None,
                        'baseline_vs_core': paired(l123_base, l123_core), 'core_vs_selected': paired(l123_core, l123_adapter) if l123_adapter else {'label': 'NO_ADAPTER', 'exact_ordered_id_match': True}},
            'manifest_hashes': {'combined': sha(ROOT / 'data_manifests/meta_m0/factorybench_l123_manifest_v2.json'), 'development': sha(ROOT / 'data_manifests/meta_m0/factorybench_l123_development_v2.json'), 'holdout': sha(ROOT / 'data_manifests/meta_m0/factorybench_l123_holdout_v2.json')},
        },
        'reused_frozen_l4_and_causal_hashes': frozen_reuse,
        'cross_task_scores': {
            'factorybench_l123': {'baseline': l123_base['fixed_cardinality_score'], 'core_only': l123_core['fixed_cardinality_score'], 'selected_adapter': l123_adapter['fixed_cardinality_score'] if l123_adapter else None, 'selection': selection['decision']},
            'factorybench_l4': {'baseline': l4_base['fixed_cardinality_score'], 'core_only': l4_core['fixed_cardinality_score'], 'selected_adapter': None, 'selection': 'NO_ADAPTER'},
            'causal_judgment': {'baseline': causal_base['fixed_cardinality_score'], 'core_only': causal_core['fixed_cardinality_score'], 'selected_adapter': causal_adapter['fixed_cardinality_score'], 'selection': 'ADAPTER'},
        },
        'verdicts': {'core_skill_result': core_verdict, 'task_adapter_necessity': adapter_verdict, 'meta_prompt_result': m0_verdict, 'meta_prompt_reason': m0_reason},
        'adapter_holdout_improvements': adapter_positive, 'adapter_holdout_regressions': adapter_negative,
        'usage': {'additional_calls': additional_calls, 'additional_cost': round(additional_cost, 6), 'additional_summed_wall_time_seconds': additional_wall, 'original_calls': original['totals']['api_calls'], 'original_cost': original['totals']['cost'], 'combined_calls': original['totals']['api_calls'] + additional_calls, 'combined_cost': round(original['totals']['cost'] + additional_cost, 6)},
        'm0_completion_traces': trace_stats,
        'integrity_assertions': {'m0_unchanged': True, 'core_unchanged': True, 'l4_not_rerun': True, 'causal_not_rerun': True, 'manual_adapter_editing': False, 'holdout_feedback_to_m0': False, 'post_holdout_refinement': False},
        'limitations': ['Very small development and holdout samples.', 'L1 uses test while L2/L3 use validation; the mixed aggregate is not a pure validation score.', 'Frozen Causal baseline/Core results contain parse failures, which triggers M0_INVALID under the original rule.', 'No statistical significance is claimed.'],
        'recommendation': 'Do not promote M0 to Golden while the completed protocol remains invalid under the predeclared parse-failure rule.',
    }
    write_json(SUMMARY_OUT, summary)

    lines = [
        '# Meta-Prompt M0 Cross-Task Smoke — Completed Data-Availability Continuation', '',
        'This report does not replace or alter the original incomplete report. It reuses frozen L4 and Causal results and adds only the recovered FactoryBench L1–L3 branch.', '',
        '## Contamination correction', '',
        f"- Original registry: `{summary['contamination_correction']['original_registry_sha256']}`",
        f"- Corrected registry: `{summary['contamination_correction']['corrected_registry_sha256']}`",
        f"- Audit report: `{summary['contamination_correction']['audit_report_sha256']}`", '',
        '## FactoryBench L1–L3 development', '',
        '| Condition | Fixed-cardinality score | Parse failures |', '|---|---:|---:|',
    ]
    for label, value in (
        ('Baseline', dev['baseline_score']), ('Core-only', dev['core_only_score']), ('Adapter v1', dev['adapter_v1_score']), ('Adapter v2', dev['adapter_v2_score'])):
        parse = 'n/a'
        if value is not None:
            result_name = {'Baseline': 'baseline_completion', 'Core-only': 'core_only_completion', 'Adapter v1': 'adapter_v1_completion', 'Adapter v2': 'adapter_v2_completion'}[label]
            parse = load(f'results/meta_m0/development/factorybench_l123_completion_{result_name}_v2data.json')['parse_failures']
        lines.append(f'| {label} | {value} | {parse} |')
    lines += ['', f"Selection: **{selection['decision']}** ({selection['selected_candidate']}).", '', '## Cross-task holdout', '', '| Task | Baseline | Core-only | Selected adapter | Selection |', '|---|---:|---:|---:|---|']
    for task, row in summary['cross_task_scores'].items():
        lines.append(f"| {task} | {row['baseline']} | {row['core_only']} | {row['selected_adapter']} | {row['selection']} |")
    lines += ['', 'FactoryBench L1 uses test; L2/L3 use validation. Its aggregate is a mixed-split smoke score, not a pure validation score.', '', '## Verdicts', '',
              f"- Core Skill: **{core_verdict}**", f"- Adapter necessity: **{adapter_verdict}**", f"- Meta-Prompt M0: **{m0_verdict}** — {m0_reason}", '',
              '## Usage', '', f"- Additional calls: {additional_calls}", f"- Additional cost: ${additional_cost:.6f}", f"- Additional summed wall time: {additional_wall:.3f} seconds", '',
              'No holdout result was sent to M0. No adapter was manually edited. L4 and Causal models were not called again.', '']
    write_new(REPORT_OUT, '\n'.join(lines).encode('utf-8'))

    paths = [
        ROOT / 'prompts/meta/meta_prompt_m0_candidate.txt', ROOT / 'prompts/core/evidence_grounded_core_v0.txt',
        ROOT / 'data_manifests/meta_m0/contamination_registry.json', ROOT / 'data_manifests/meta_m0/contamination_registry_v2.json',
        ROOT / 'reports/meta_m0/contamination_registry_audit.md', ROOT / 'data_manifests/meta_m0/factorybench_l123_manifest_v2.json',
        ROOT / 'data_manifests/meta_m0/factorybench_l123_development_v2.json', ROOT / 'data_manifests/meta_m0/factorybench_l123_holdout_v2.json',
        SUMMARY_OUT, REPORT_OUT,
    ]
    for base in (ADAPTER_DIR := ROOT / 'prompts/adapters/factorybench_l123_completion', ROOT / 'traces/meta_m0/factorybench_l123_completion'):
        if base.exists():
            paths.extend(path for path in base.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc')
    paths.extend(new_result_paths)
    paths.extend(ROOT / rel for rel in frozen_reuse)
    paths.extend([ROOT / 'results/meta_m0/cross_task_summary.json', ROOT / 'reports/meta_m0/cross_task_summary.md', ROOT / 'results/meta_m0/artifact_registry.json'])
    scripts = [ROOT / 'scripts/meta_m0/audit_contamination_v2.py', ROOT / 'scripts/meta_m0/run_l123_completion.py', ROOT / 'scripts/meta_m0/build_completed_summary.py']
    paths.extend(scripts)
    records = {str(path.relative_to(ROOT)): {'sha256': sha(path), 'bytes': path.stat().st_size} for path in sorted(set(paths))}
    registry = {'registry_self_hash': None, 'self_hash_note': 'Omitted because self-hashing is not stable.', 'volatile_files_excluded': ['__pycache__', '*.pyc'], 'artifact_count': len(records), 'artifacts': records}
    write_json(REGISTRY_OUT, registry)
    print(json.dumps({'status': 'COMPLETED', 'summary_sha256': sha(SUMMARY_OUT), 'report_sha256': sha(REPORT_OUT), 'registry_sha256': sha(REGISTRY_OUT), 'verdicts': summary['verdicts'], 'additional_usage': summary['usage']}, indent=2))


if __name__ == '__main__':
    main()
