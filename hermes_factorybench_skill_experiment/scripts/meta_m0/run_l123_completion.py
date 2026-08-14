#!/usr/bin/env python3
"""Complete only the recovered FactoryBench L1-L3 M0 branch."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from factorybench.cost import compute_cost_from_usage
from factorybench.data import load_split
from factorybench.evaluate import _score_one
from factorybench.prompt import render_prompt

REPO = Path('/home/training/automatic_prompt_engineer')
ROOT = REPO / 'hermes_factorybench_skill_experiment'
MODEL = 'gpt-5.5'
REV = 'b3863519ccedbceab54dfa7600104eb42b985ed7'
M0_PATH = ROOT / 'prompts/meta/meta_prompt_m0_candidate.txt'
CORE_PATH = ROOT / 'prompts/core/evidence_grounded_core_v0.txt'
M0_SHA = 'e4bef66552518e4a1206aaa3b14b9d34335c046499291e70dfdc33a79f243257'
CORE_SHA = '463692cd0a201d916c3f0e39d10cda4c50d2a9a1ca3305cd9de804c295e482b3'
DEV_MANIFEST = ROOT / 'data_manifests/meta_m0/factorybench_l123_development_v2.json'
HOLD_MANIFEST = ROOT / 'data_manifests/meta_m0/factorybench_l123_holdout_v2.json'
ADAPTER_DIR = ROOT / 'prompts/adapters/factorybench_l123_completion'
TRACE_DIR = ROOT / 'traces/meta_m0/factorybench_l123_completion'
DEV_DIR = ROOT / 'results/meta_m0/development'
HOLD_DIR = ROOT / 'results/meta_m0/holdout'
M0_FIELDS = {
    'meta_prompt_version', 'task_name', 'mode', 'decision', 'task_scope',
    'supported_formats', 'applicability_conditions', 'failure_taxonomy',
    'adapter_text', 'fallback_policy', 'changes_from_previous',
    'predicted_regression_risks', 'evidence_limitations',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing to overwrite different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, payload: Any) -> None:
    write_new(path, (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def runtime_text(adapter: str | None) -> str:
    core = CORE_PATH.read_text(encoding='utf-8').rstrip('\n')
    return core + ('\n' if not adapter else '\n\nTASK ADAPTER:\n' + adapter.strip() + '\n')


def save_runtime(label: str, adapter: str | None) -> Path:
    path = ADAPTER_DIR / f'runtime_{label}.txt'
    write_new(path, runtime_text(adapter).encode('utf-8'))
    return path


def response_usage(response: Any) -> dict[str, int]:
    usage = response.usage
    return {'input_tokens': int(getattr(usage, 'prompt_tokens', 0) or 0), 'output_tokens': int(getattr(usage, 'completion_tokens', 0) or 0)}


class Caller:
    def __init__(self, client: OpenAI, system: str | None):
        self.client, self.system = client, system

    def __call__(self, prompt: str) -> tuple[str, dict[str, int], float, str | None]:
        messages = []
        if self.system:
            messages.append({'role': 'system', 'content': self.system})
        messages.append({'role': 'user', 'content': prompt})
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(model=MODEL, messages=messages, max_completion_tokens=8192)
            return response.choices[0].message.content or '', response_usage(response), time.perf_counter() - started, None
        except Exception as exc:
            return '', {'input_tokens': 0, 'output_tokens': 0}, time.perf_counter() - started, f'{type(exc).__name__}: {exc}'


def parallel(prompts: list[str], caller: Caller) -> list[tuple[str, dict[str, int], float, str | None]]:
    output: list[Any] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(caller, prompt): index for index, prompt in enumerate(prompts)}
        for future, index in futures.items():
            output[index] = future.result()
    return output


def source_items(manifest: dict[str, Any]) -> list[Any]:
    cache = {}
    for row in manifest['items']:
        key = (int(row['level']), row['split'])
        if key not in cache:
            cache[key] = {item.id: item for item in load_split(key[0], split=key[1], revision=REV, max_items=None)}
    items = [cache[(int(row['level']), row['split'])][row['id']] for row in manifest['items']]
    for row, item in zip(manifest['items'], items):
        if item.answer_format.value != row['answer_format'] or item.dataset != row['dataset'] or item.provenance.get('episode') != row['episode']:
            raise RuntimeError(f'manifest/source mismatch: {item.id}')
    return items


def fixed_score(rows: list[dict[str, Any]]) -> float:
    raw, chance = [], []
    for row in rows:
        clean = row.get('parse_error') is None and isinstance(row.get('score'), (int, float)) and math.isfinite(float(row['score']))
        raw.append(float(row['score']) if clean else 0.0)
        chance.append(float(row.get('chance', 0.0)))
    mean_raw, mean_chance = sum(raw) / len(raw), sum(chance) / len(chance)
    return (mean_raw - mean_chance) / (1.0 - mean_chance)


def canonical(rows: list[dict[str, Any]]) -> float | None:
    clean = [row for row in rows if row.get('parse_error') is None and isinstance(row.get('score'), (int, float)) and math.isfinite(float(row['score']))]
    if not clean:
        return None
    mean_raw = sum(float(row['score']) for row in clean) / len(clean)
    mean_chance = sum(float(row.get('chance', 0.0)) for row in clean) / len(clean)
    return (mean_raw - mean_chance) / (1.0 - mean_chance)


def groups(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    buckets = {}
    for row in rows:
        buckets.setdefault(str(row.get(field) or 'unknown'), []).append(row)
    return {key: fixed_score(value) for key, value in sorted(buckets.items())}


def evaluate(client: OpenAI, partition: str, condition: str, runtime: Path | None) -> dict[str, Any]:
    output_path = (DEV_DIR if partition == 'development' else HOLD_DIR) / f'factorybench_l123_completion_{condition}_v2data.json'
    if output_path.exists():
        return load(output_path)
    manifest_path = DEV_MANIFEST if partition == 'development' else HOLD_MANIFEST
    manifest = load(manifest_path)
    items = source_items(manifest)
    caller = Caller(client, runtime.read_text(encoding='utf-8') if runtime else None)
    started = time.perf_counter()
    calls = parallel([render_prompt(item) for item in items], caller)
    rows = []
    for item, (raw, usage, latency, transport_error) in zip(items, calls):
        result = _score_one(item, raw)
        rows.append({
            'id': item.id, 'level': item.level, 'split': next(row['split'] for row in manifest['items'] if row['id'] == item.id),
            'dataset': item.dataset, 'episode': item.provenance.get('episode'), 'answer_format': item.answer_format.value,
            'template_id': item.template_id, 'template_type': item.template_type,
            'raw_output': raw, 'parsed': result.parsed, 'score': result.score, 'chance': result.chance,
            'parse_error': result.parse_error, 'transport_error': transport_error,
            'candidate_usage': usage, 'latency_seconds': latency,
            'development_evidence': {'rendered_input': render_prompt(item), 'reference_answer': item.answer} if partition == 'development' else None,
        })
    usage = {
        'candidate': {'model': MODEL, 'input_tokens': sum(row[1]['input_tokens'] for row in calls), 'output_tokens': sum(row[1]['output_tokens'] for row in calls), 'calls': sum(bool(row[1]['input_tokens'] or row[1]['output_tokens']) for row in calls)},
        'judges': {},
    }
    payload = {
        'task_name': 'factorybench_l123', 'protocol': 'mixed_split_l1_test_l2_l3_validation',
        'partition': partition, 'condition': condition, 'model': MODEL, 'concurrency': 2,
        'manifest_path': str(manifest_path.relative_to(ROOT)), 'manifest_sha256': sha(manifest_path),
        'runtime_prompt_path': str(runtime.relative_to(ROOT)) if runtime else None,
        'runtime_prompt_sha256': sha(runtime) if runtime else None,
        'item_count': len(rows), 'ordered_ids': [row['id'] for row in rows],
        'canonical_score': canonical(rows), 'fixed_cardinality_score': fixed_score(rows),
        'parse_failures': sum(row['parse_error'] is not None for row in rows),
        'by_level': groups(rows, 'level'), 'by_split': groups(rows, 'split'),
        'by_dataset': groups(rows, 'dataset'), 'by_answer_format': groups(rows, 'answer_format'),
        'mixed_split_warning': manifest['mixed_split_warning'],
        'tokens_used': usage, 'cost': compute_cost_from_usage(usage),
        'wall_time_seconds': time.perf_counter() - started, 'items': rows,
    }
    write_json(output_path, payload)
    return payload


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        'condition': result['condition'], 'item_count': result['item_count'], 'ordered_ids': result['ordered_ids'],
        'canonical_score': result['canonical_score'], 'fixed_cardinality_score': result['fixed_cardinality_score'],
        'parse_failures': result['parse_failures'],
        'subgroups': {'by_level': result['by_level'], 'by_split': result['by_split'], 'by_dataset': result['by_dataset'], 'by_answer_format': result['by_answer_format']},
        'items': [{key: row.get(key) for key in ('id', 'level', 'split', 'dataset', 'answer_format', 'raw_output', 'parsed', 'score', 'chance', 'parse_error', 'development_evidence')} for row in result['items']],
    }


def optimization_input(mode: str, baseline: dict[str, Any], core: dict[str, Any], prior: dict[str, Any] | None, previous_adapter: dict[str, Any] | None) -> dict[str, Any]:
    return {
        'task_name': 'factorybench_l123', 'mode': mode,
        'task_description': 'FactoryBench L1-L3 machine-telemetry questions. L1 uses deterministic exact/range/MCQ formats; L2 and L3 include predictive, ranking, Boolean, scalar, and tensor outputs with strict item-specific contracts.',
        'execution_model': MODEL,
        'evaluator_specification': {'canonical': 'FactoryBench chance-corrected score', 'fixed_cardinality': 'failed/non-finite parses retained as zero raw score', 'parser': 'FactoryBench deterministic parser and scorer'},
        'output_contracts': ['Follow each item-specific exact output contract: one letter, four letters, scalar, or tensor as requested.'],
        'subgroups': ['level', 'split', 'dataset', 'answer_format'],
        'core_skill_sha256': CORE_SHA, 'development_manifest_sha256': sha(DEV_MANIFEST),
        'development_results': {'baseline': compact(baseline), 'core_only': compact(core), 'previous_adapter': compact(prior) if prior else None},
        'previous_adapter': previous_adapter,
        'constraints': {'maximum_rounds': 2, 'manual_adapter_editing': False, 'holdout_access': False, 'case_memorization': False},
    }


def validate_m0(parsed: Any, mode: str, dev_ids: list[str], hold_ids: list[str]) -> list[str]:
    errors = []
    if not isinstance(parsed, dict) or set(parsed) != M0_FIELDS:
        return ['schema field mismatch']
    if parsed.get('meta_prompt_version') != 'm0' or parsed.get('task_name') != 'factorybench_l123' or parsed.get('mode') != mode:
        errors.append('identity/mode mismatch')
    decision, adapter = parsed.get('decision'), parsed.get('adapter_text')
    if decision not in {'ADAPTER', 'NO_ADAPTER', 'INSUFFICIENT_DATA'} or not isinstance(adapter, str):
        errors.append('invalid decision or adapter_text')
        adapter = ''
    if decision == 'ADAPTER' and not adapter.strip():
        errors.append('empty ADAPTER')
    if decision != 'ADAPTER' and adapter != '':
        errors.append('non-empty null adapter')
    lower = adapter.casefold()
    for item_id in dev_ids + hold_ids:
        if item_id.casefold() in lower:
            errors.append(f'item ID leaked: {item_id}')
    if re.search(r'holdout|held[- ]?out|final[-_ ]?test', lower):
        errors.append('holdout reference')
    for key in ('task_scope', 'supported_formats', 'applicability_conditions', 'failure_taxonomy', 'changes_from_previous', 'predicted_regression_risks', 'evidence_limitations'):
        if not isinstance(parsed.get(key), list):
            errors.append(f'{key} not list')
    if not isinstance(parsed.get('fallback_policy'), str):
        errors.append('fallback_policy not string')
    return errors


def call_m0(client: OpenAI, round_number: int, data: dict[str, Any]) -> tuple[dict[str, Any], str | None, Path | None]:
    input_path = TRACE_DIR / f'm0_round_{round_number}_input_completion.json'
    raw_path = TRACE_DIR / f'm0_round_{round_number}_raw_output_completion.txt'
    parsed_path = TRACE_DIR / f'm0_round_{round_number}_parsed_output_completion.json'
    write_json(input_path, data)
    if raw_path.exists() or parsed_path.exists():
        raise RuntimeError('partial/existing M0 completion trace; refusing to rerun')
    user = '<OPTIMIZATION_INPUT>\n' + json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n</OPTIMIZATION_INPUT>'
    started = time.perf_counter()
    response = client.chat.completions.create(model=MODEL, messages=[
        {'role': 'system', 'content': M0_PATH.read_text(encoding='utf-8')}, {'role': 'user', 'content': user}
    ], max_completion_tokens=8192)
    wall = time.perf_counter() - started
    raw = response.choices[0].message.content or ''
    write_new(raw_path, raw.encode('utf-8'))
    try:
        parsed = json.loads(raw.strip())
        json_error = None
    except Exception as exc:
        parsed, json_error = {}, f'{type(exc).__name__}: {exc}'
    dev_ids = [row['id'] for row in load(DEV_MANIFEST)['items']]
    hold_ids = [row['id'] for row in load(HOLD_MANIFEST)['items']]
    errors = ([json_error] if json_error else []) + validate_m0(parsed, data['mode'], dev_ids, hold_ids)
    usage = response_usage(response)
    envelope = {**parsed, '_trace_validation': {
        'valid': not errors, 'errors': errors, 'input_sha256': sha(input_path), 'raw_output_sha256': sha(raw_path),
        'meta_prompt_sha256': sha(M0_PATH), 'core_skill_sha256': sha(CORE_PATH),
        'usage': usage, 'cost': compute_cost_from_usage({'candidate': {'model': MODEL, **usage, 'calls': 1}, 'judges': {}}),
        'wall_time_seconds': wall,
    }}
    write_json(parsed_path, envelope)
    if errors:
        raise RuntimeError(f'invalid M0 output: {errors}')
    adapter_path = None
    adapter = parsed['adapter_text'] if parsed['decision'] == 'ADAPTER' else None
    if adapter:
        adapter_path = ADAPTER_DIR / f'adapter_v{round_number}_completion.txt'
        write_new(adapter_path, adapter.encode('utf-8'))
    return envelope, adapter, adapter_path


def selection(core: dict[str, Any], candidates: list[tuple[str, dict[str, Any], Path | None]]) -> dict[str, Any]:
    path = ADAPTER_DIR / 'selection_completion.json'
    if path.exists():
        return load(path)
    expected_ids = [row['id'] for row in load(DEV_MANIFEST)['items']]
    records, qualified = [], []
    for label, result, adapter_path in candidates:
        reasons = []
        if result['ordered_ids'] != expected_ids or result['item_count'] != 3:
            reasons.append('item mismatch')
        if result['parse_failures']:
            reasons.append('parse failure')
        if not math.isfinite(float(result['fixed_cardinality_score'])):
            reasons.append('non-finite score')
        for level, score in core['by_level'].items():
            if result['by_level'].get(level, float('-inf')) < score:
                reasons.append(f'critical level regression L{level}')
        record = {
            'candidate': label, 'eligible': not reasons, 'ineligibility_reasons': reasons,
            'fixed_cardinality_score': result['fixed_cardinality_score'],
            'worst_level_score': min(result['by_level'].values()), 'parse_failures': result['parse_failures'],
            'adapter_path': str(adapter_path.relative_to(ROOT)) if adapter_path else None,
            'adapter_sha256': sha(adapter_path) if adapter_path else None,
            'adapter_bytes': adapter_path.stat().st_size if adapter_path else 0,
        }
        records.append(record)
        if label != 'core_only' and not reasons and result['fixed_cardinality_score'] > core['fixed_cardinality_score']:
            qualified.append(record)
    if qualified:
        order = {'adapter_v1': 1, 'adapter_v2': 2}
        chosen = sorted(qualified, key=lambda row: (-row['fixed_cardinality_score'], -row['worst_level_score'], row['parse_failures'], row['adapter_bytes'], order[row['candidate']]))[0]
        decision, selected = 'ADAPTER', chosen['candidate']
        source = ROOT / chosen['adapter_path']
        target = ADAPTER_DIR / 'selected_adapter_completion.txt'
        write_new(target, source.read_bytes())
        selected_hash = sha(target)
    else:
        decision, selected, selected_hash = 'NO_ADAPTER', 'core_only', None
    payload = {
        'task_name': 'factorybench_l123', 'continuation': 'v2data_completion',
        'decision': decision, 'selected_candidate': selected, 'selected_adapter_sha256': selected_hash,
        'development_manifest_sha256': sha(DEV_MANIFEST), 'core_skill_sha256': CORE_SHA, 'meta_prompt_sha256': M0_SHA,
        'core_only_score': core['fixed_cardinality_score'], 'candidates': records,
        'selection_rule': ['eligibility gates', 'strictly beat Core-only', 'highest fixed score', 'highest worst-level score', 'fewest parses', 'shortest bytes', 'v1 before v2'],
    }
    write_json(path, payload)
    return payload


def development(client: OpenAI) -> dict[str, Any]:
    core_runtime = save_runtime('core_only_completion', None)
    baseline = evaluate(client, 'development', 'baseline_completion', None)
    core = evaluate(client, 'development', 'core_only_completion', core_runtime)
    out1, adapter1, path1 = call_m0(client, 1, optimization_input('generate', baseline, core, None, None))
    candidates: list[tuple[str, dict[str, Any], Path | None]] = [('core_only', core, None)]
    result1 = None
    if adapter1 and path1:
        result1 = evaluate(client, 'development', 'adapter_v1_completion', save_runtime('adapter_v1_completion', adapter1))
        candidates.append(('adapter_v1', result1, path1))
    previous = {'sha256': sha(path1), 'text': adapter1} if path1 else {'sha256': None, 'text': ''}
    out2, adapter2, path2 = call_m0(client, 2, optimization_input('refine', baseline, core, result1, previous))
    result2 = None
    if adapter2 and path2:
        result2 = evaluate(client, 'development', 'adapter_v2_completion', save_runtime('adapter_v2_completion', adapter2))
        candidates.append(('adapter_v2', result2, path2))
    selected = selection(core, candidates)
    payload = {
        'task_name': 'factorybench_l123', 'status': 'DEVELOPMENT_COMPLETION_COMPLETE',
        'development_manifest_sha256': sha(DEV_MANIFEST),
        'baseline_score': baseline['fixed_cardinality_score'], 'core_only_score': core['fixed_cardinality_score'],
        'adapter_v1_decision': out1['decision'], 'adapter_v1_sha256': sha(path1) if path1 else None, 'adapter_v1_score': result1['fixed_cardinality_score'] if result1 else None,
        'adapter_v2_decision': out2['decision'], 'adapter_v2_sha256': sha(path2) if path2 else None, 'adapter_v2_score': result2['fixed_cardinality_score'] if result2 else None,
        'selection': selected,
    }
    write_json(DEV_DIR / 'factorybench_l123_completion_summary_v2data.json', payload)
    return payload


def holdout(client: OpenAI) -> dict[str, Any]:
    selected = load(ADAPTER_DIR / 'selection_completion.json')
    if (ADAPTER_DIR / 'selection_completion.json').stat().st_mtime_ns >= HOLD_MANIFEST.stat().st_mtime_ns:
        pass
    baseline = evaluate(client, 'holdout', 'baseline_completion', None)
    core_runtime = ADAPTER_DIR / 'runtime_core_only_completion.txt'
    core = evaluate(client, 'holdout', 'core_only_completion', core_runtime)
    adapter_result = None
    if selected['decision'] == 'ADAPTER':
        adapter = (ADAPTER_DIR / 'selected_adapter_completion.txt').read_text(encoding='utf-8')
        adapter_result = evaluate(client, 'holdout', 'selected_adapter_completion', save_runtime('selected_adapter_completion', adapter))
    payload = {
        'task_name': 'factorybench_l123', 'status': 'HOLDOUT_COMPLETION_COMPLETE',
        'holdout_manifest_sha256': sha(HOLD_MANIFEST), 'selection_sha256': sha(ADAPTER_DIR / 'selection_completion.json'),
        'selection_decision': selected['decision'], 'baseline': compact(baseline), 'core_only': compact(core),
        'selected_adapter': compact(adapter_result) if adapter_result else None,
        'selected_condition_label': 'selected_adapter' if adapter_result else 'NO_ADAPTER (identical to Core-only; no duplicate calls)',
        'no_holdout_feedback_to_m0': True, 'post_holdout_refinement': False,
    }
    write_json(HOLD_DIR / 'factorybench_l123_completion_summary_v2data.json', payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'holdout'], required=True)
    args = parser.parse_args()
    load_dotenv(REPO / '.env')
    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY unavailable')
    if sha(M0_PATH) != M0_SHA or sha(CORE_PATH) != CORE_SHA:
        raise SystemExit('frozen M0/Core hash mismatch')
    client = OpenAI()
    payload = development(client) if args.phase == 'development' else holdout(client)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
