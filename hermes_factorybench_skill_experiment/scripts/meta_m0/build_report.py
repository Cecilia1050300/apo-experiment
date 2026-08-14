#!/usr/bin/env python3
"""Build final M0 cross-task summaries and artifact registry."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path('/home/training/automatic_prompt_engineer/hermes_factorybench_skill_experiment')
TASKS = ('factorybench_l123', 'factorybench_l4', 'causal_judgment')
CORE_PATH = ROOT / 'prompts/core/evidence_grounded_core_v0.txt'
M0_PATH = ROOT / 'prompts/meta/meta_prompt_m0_candidate.txt'
SUMMARY_JSON = ROOT / 'results/meta_m0/cross_task_summary.json'
SUMMARY_MD = ROOT / 'reports/meta_m0/cross_task_summary.md'
REGISTRY_PATH = ROOT / 'results/meta_m0/artifact_registry.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing to overwrite different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, payload: Any) -> None:
    write_new(path, (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def paired(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_by = {row['id']: row for row in left['items']}
    right_by = {row['id']: row for row in right['items']}
    expected = left['ordered_ids']
    exact = expected == right['ordered_ids']
    counts = {'improved': 0, 'worse': 0, 'unchanged': 0, 'invalid': 0}
    changed = []
    for item_id in expected:
        a, b = left_by[item_id], right_by[item_id]
        clean = a.get('parse_error') is None and b.get('parse_error') is None and math.isfinite(float(a['score'])) and math.isfinite(float(b['score']))
        if not clean:
            status = 'invalid'
        else:
            delta = float(b['score']) - float(a['score'])
            status = 'improved' if delta > 0 else 'worse' if delta < 0 else 'unchanged'
        counts[status] += 1
        if a.get('raw_output') != b.get('raw_output'):
            changed.append({'id': item_id, 'left_output': a.get('raw_output'), 'right_output': b.get('raw_output'), 'left_score': a.get('score'), 'right_score': b.get('score'), 'status': status})
    return {'exact_ordered_id_match': exact, 'counts': counts, 'raw_output_changed_count': len(changed), 'changed_cases': changed}


def result_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in ('canonical_score', 'fixed_cardinality_score', 'parse_failures', 'by_subgroup', 'by_answer_format', 'by_dataset', 'tokens_used', 'cost', 'wall_time_seconds') if key in result}


def calls(result: dict[str, Any]) -> int:
    tokens = result.get('tokens_used') or {}
    return int((tokens.get('candidate') or {}).get('calls', 0)) + sum(int(row.get('calls', 0)) for row in (tokens.get('judges') or {}).values())


def core_verdict(completed: dict[str, dict[str, Any]]) -> tuple[str, str]:
    deltas = {task: data['core_only']['fixed_cardinality_score'] - data['baseline']['fixed_cardinality_score'] for task, data in completed.items()}
    positives = [task for task, delta in deltas.items() if delta > 0]
    negatives = [task for task, delta in deltas.items() if delta < 0]
    parse_failures = sum(data['baseline']['parse_failures'] + data['core_only']['parse_failures'] for data in completed.values())
    if len(positives) >= 2 and any(task.startswith('factorybench') for task in positives) and not negatives and parse_failures == 0:
        return 'CORE_POSITIVE', 'Meets the predeclared rule on the completed task set.'
    if positives and negatives:
        return 'CORE_MIXED', 'Core improves at least one completed task and regresses another.'
    if not positives and not negatives:
        return 'CORE_NO_EFFECT', 'No completed task changes.'
    if negatives and not positives:
        return 'CORE_NEGATIVE', 'Core regresses without compensating evidence on completed tasks.'
    return 'CORE_UNCLASSIFIED_INCOMPLETE', 'The predeclared Core categories do not cover improvement on only one completed task with no regression, and FactoryBench L1-L3 is incomplete.'


def build() -> tuple[dict[str, Any], str]:
    manifests = {task: load(ROOT / f'data_manifests/meta_m0/{task}_manifest.json') for task in TASKS}
    selections = {task: load(ROOT / f'prompts/adapters/{task}/selection.json') for task in TASKS}
    completed = {}
    unavailable = {}
    for task in TASKS:
        if manifests[task]['status'] != 'AVAILABLE':
            unavailable[task] = {'status': manifests[task]['status'], 'reason': manifests[task].get('reason')}
            continue
        dev_summary_path = ROOT / f'results/meta_m0/development/{task}_summary.json'
        hold_summary_path = ROOT / f'results/meta_m0/holdout/{task}_summary.json'
        if not dev_summary_path.exists() or not hold_summary_path.exists():
            raise RuntimeError(f'missing completed summaries for {task}')
        dev = load(dev_summary_path)
        hold = load(hold_summary_path)
        baseline_result = load(ROOT / f'results/meta_m0/holdout/{task}_baseline.json')
        core_result = load(ROOT / f'results/meta_m0/holdout/{task}_core_only.json')
        selected_result = None
        if selections[task]['decision'] == 'ADAPTER':
            selected_result = load(ROOT / f'results/meta_m0/holdout/{task}_selected_adapter.json')
        completed[task] = {
            'development': dev,
            'selection': selections[task],
            'baseline': result_metrics(baseline_result),
            'core_only': result_metrics(core_result),
            'selected_adapter': result_metrics(selected_result) if selected_result else None,
            'baseline_vs_core': paired(baseline_result, core_result),
            'core_vs_selected_adapter': paired(core_result, selected_result) if selected_result else {'label': 'NO_ADAPTER', 'exact_ordered_id_match': True, 'counts': {'improved': 0, 'worse': 0, 'unchanged': core_result['item_count'], 'invalid': 0}, 'raw_output_changed_count': 0, 'changed_cases': []},
            'no_holdout_feedback_to_m0': hold['no_holdout_feedback_to_m0'],
        }

    core_name, core_note = core_verdict(completed)
    adapter_improvements = []
    adapter_regressions = []
    for task, data in completed.items():
        selected = data['selected_adapter']
        if selected is None:
            continue
        delta = selected['fixed_cardinality_score'] - data['core_only']['fixed_cardinality_score']
        if delta > 0:
            adapter_improvements.append(task)
        elif delta < 0:
            adapter_regressions.append(task)
    adapter_verdict = 'ADAPTER_EVIDENCE' if len(adapter_improvements) >= 2 else 'NO_ADAPTER_EVIDENCE'
    completed_parse_failures = sum(
        data['baseline']['parse_failures']
        + data['core_only']['parse_failures']
        + ((data['selected_adapter'] or {}).get('parse_failures', 0))
        for data in completed.values()
    )
    m0_verdict = 'M0_INVALID' if unavailable or completed_parse_failures else (
        'M0_PROMISING' if len(adapter_improvements) >= 2 and any(task.startswith('factorybench') for task in adapter_improvements) and not adapter_regressions else
        'M0_MIXED' if adapter_improvements and adapter_regressions else
        'M0_NO_EFFECT' if not adapter_improvements else 'M0_MIXED'
    )

    result_files = sorted((ROOT / 'results/meta_m0/development').glob('*.json')) + sorted((ROOT / 'results/meta_m0/holdout').glob('*.json'))
    total_calls = total_cost = total_wall = 0
    for path in result_files:
        data = load(path)
        if 'tokens_used' in data:
            total_calls += calls(data)
            total_cost += float(data.get('cost', 0.0))
            total_wall += float(data.get('wall_time_seconds', 0.0))
    m0_trace_stats = []
    for path in sorted((ROOT / 'traces/meta_m0').glob('*/m0_round_*_parsed_output.json')):
        data = load(path)
        stats = data['_trace_validation']
        total_calls += 1
        total_cost += float(stats['cost'])
        total_wall += float(stats['wall_time_seconds'])
        m0_trace_stats.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'usage': stats['usage'], 'cost': stats['cost'], 'wall_time_seconds': stats['wall_time_seconds']})

    summary = {
        'experiment': 'Golden Meta-Prompt Candidate M0 — Cross-Task Adapter Generation Smoke',
        'qualification': 'M0 is a candidate only and is not Golden.',
        'frozen_artifacts': {
            'meta_prompt_m0': {'path': str(M0_PATH.relative_to(ROOT)), 'sha256': sha(M0_PATH), 'bytes': M0_PATH.stat().st_size},
            'core_skill_v0': {'path': str(CORE_PATH.relative_to(ROOT)), 'sha256': sha(CORE_PATH), 'bytes': CORE_PATH.stat().st_size},
        },
        'contamination_registry': {'path': 'data_manifests/meta_m0/contamination_registry.json', 'sha256': sha(ROOT / 'data_manifests/meta_m0/contamination_registry.json'), 'counts': load(ROOT / 'data_manifests/meta_m0/contamination_registry.json')['excluded_id_counts']},
        'manifests': {task: {'status': manifests[task]['status'], 'sha256': sha(ROOT / f'data_manifests/meta_m0/{task}_manifest.json'), 'development_count': len(manifests[task]['development']), 'holdout_count': len(manifests[task]['holdout'])} for task in TASKS},
        'unavailable_tasks': unavailable, 'tasks': completed,
        'verdicts': {
            'core_skill_result': core_name, 'core_skill_note': core_note,
            'task_adapter_necessity': adapter_verdict,
            'meta_prompt_result': m0_verdict,
            'meta_prompt_invalid_reason': (
                'FactoryBench L1-L3 lacks sufficient defensibly unseen data under the frozen contamination registry; '
                f'completed holdouts also contain {completed_parse_failures} baseline/Core parse failures.'
                if unavailable else
                (f'Completed holdouts contain {completed_parse_failures} parse failures.' if completed_parse_failures else None)
            ),
        },
        'adapter_holdout_improvements': adapter_improvements,
        'adapter_holdout_regressions': adapter_regressions,
        'totals': {'api_calls': total_calls, 'cost': round(total_cost, 6), 'summed_condition_wall_time_seconds': total_wall},
        'm0_generation_traces': m0_trace_stats,
        'integrity_assertions': {'manual_adapter_editing': False, 'holdout_feedback_to_m0': False, 'post_holdout_refinement': False, 'skill_v3_and_task_a_unchanged': True},
        'limitations': [
            'FactoryBench L1-L3 was not run because the frozen registry left no eligible item in at least one required stratum.',
            'Completed development and holdout samples are very small.',
            'FactoryBench L4 free-form items use one gpt-5.5 judge rather than the three-provider paper-default ensemble.',
            'No statistical significance is claimed.',
        ],
        'next_stage_recommendation': 'Do not promote M0 to Golden. Resolve the L1-L3 contamination/data-availability boundary prospectively, then repeat a fully complete cross-task holdout without changing M0 or Core v0.',
    }

    lines = [
        '# Meta-Prompt M0 Cross-Task Smoke', '',
        'M0 remains a candidate and is not Golden.', '',
        '## Frozen artifacts', '',
        f"- M0: `{summary['frozen_artifacts']['meta_prompt_m0']['sha256']}` ({summary['frozen_artifacts']['meta_prompt_m0']['bytes']} bytes)",
        f"- Core v0: `{summary['frozen_artifacts']['core_skill_v0']['sha256']}` ({summary['frozen_artifacts']['core_skill_v0']['bytes']} bytes)",
        '', '## Data availability', '', '| Task | Status | Development | Holdout | Manifest SHA-256 |', '|---|---|---:|---:|---|'
    ]
    for task in TASKS:
        row = summary['manifests'][task]
        lines.append(f"| {task} | {row['status']} | {row['development_count']} | {row['holdout_count']} | `{row['sha256']}` |")
    lines += ['', '## Development and selection', '', '| Task | Baseline | Core-only | Adapter v1 | Adapter v2 | Selection |', '|---|---:|---:|---:|---:|---|']
    for task in TASKS:
        if task in unavailable:
            lines.append(f"| {task} | n/a | n/a | n/a | n/a | INSUFFICIENT_UNSEEN_DATA |")
            continue
        dev = completed[task]['development']
        lines.append(f"| {task} | {dev['baseline_score']:.6f} | {dev['core_only_score']:.6f} | {str(dev['adapter_v1_score'])} | {str(dev['adapter_v2_score'])} | {dev['selection']['decision']} ({dev['selection']['selected_candidate']}) |")
    lines += ['', '## Holdout', '', '| Task | Baseline | Core-only | Selected adapter | Selection |', '|---|---:|---:|---:|---|']
    for task in TASKS:
        if task in unavailable:
            lines.append(f"| {task} | n/a | n/a | n/a | not run |")
            continue
        data = completed[task]
        selected = data['selected_adapter']['fixed_cardinality_score'] if data['selected_adapter'] else None
        lines.append(f"| {task} | {data['baseline']['fixed_cardinality_score']:.6f} | {data['core_only']['fixed_cardinality_score']:.6f} | {str(selected)} | {data['selection']['decision']} |")
    lines += [
        '', '## Verdicts', '',
        f"- Core Skill: **{core_name}** — {core_note}",
        f"- Task Adapter necessity: **{adapter_verdict}**",
        f"- Meta-Prompt: **{m0_verdict}**",
        '', '## Integrity and limitations', '',
        '- No holdout result was sent to M0 or used for refinement.',
        '- No M0-generated adapter was manually edited.',
        '- Skill v3 and all Task A artifacts remain unchanged.',
        '- This smoke does not establish statistical significance.',
        '', f"Total recorded API calls: {total_calls}; cost: ${total_cost:.6f}; summed condition wall time: {total_wall:.3f} seconds.",
        '', 'Recommendation: do not promote M0 to Golden; resolve prospective L1-L3 data availability and repeat a complete cross-task holdout.', ''
    ]
    return summary, '\n'.join(lines)


def registry() -> dict[str, Any]:
    paths = []
    roots = [
        ROOT / 'prompts/meta', ROOT / 'prompts/core', ROOT / 'prompts/adapters',
        ROOT / 'data_manifests/meta_m0', ROOT / 'results/meta_m0', ROOT / 'logs/meta_m0',
        ROOT / 'reports/meta_m0', ROOT / 'traces/meta_m0', ROOT / 'scripts/meta_m0',
    ]
    for base in roots:
        if base.exists():
            paths.extend(
                path for path in base.rglob('*')
                if path.is_file()
                and path != REGISTRY_PATH
                and '__pycache__' not in path.parts
                and path.suffix != '.pyc'
            )
    records = {}
    for path in sorted(set(paths)):
        records[str(path.relative_to(ROOT))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return {'registry_self_hash': None, 'self_hash_note': 'Omitted because a file cannot contain its own stable hash.', 'artifact_count': len(records), 'artifacts': records}


def main() -> None:
    summary, report = build()
    write_json(SUMMARY_JSON, summary)
    write_new(SUMMARY_MD, report.encode('utf-8'))
    write_json(REGISTRY_PATH, registry())
    print(json.dumps({'status': 'COMPLETE', 'summary': str(SUMMARY_JSON), 'report': str(SUMMARY_MD), 'registry': str(REGISTRY_PATH), 'verdicts': summary['verdicts']}, indent=2))


if __name__ == '__main__':
    main()
