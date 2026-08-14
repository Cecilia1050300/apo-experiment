#!/usr/bin/env python3
"""Run available tasks for the frozen Meta-Prompt M0 cross-task smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from factorybench.cost import compute_cost_from_usage
from factorybench.data import load_split
from factorybench.evaluate import _score_one
from factorybench.judges import JudgePanel
from factorybench.prompt import render_prompt

REPO = Path('/home/training/automatic_prompt_engineer')
ROOT = REPO / 'hermes_factorybench_skill_experiment'
MANIFEST_DIR = ROOT / 'data_manifests' / 'meta_m0'
DEVELOPMENT_DIR = ROOT / 'results' / 'meta_m0' / 'development'
HOLDOUT_DIR = ROOT / 'results' / 'meta_m0' / 'holdout'
TRACE_DIR = ROOT / 'traces' / 'meta_m0'
ADAPTER_DIR = ROOT / 'prompts' / 'adapters'
MODEL = 'gpt-5.5'
REVISION = 'b3863519ccedbceab54dfa7600104eb42b985ed7'
CORE_PATH = ROOT / 'prompts' / 'core' / 'evidence_grounded_core_v0.txt'
M0_PATH = ROOT / 'prompts' / 'meta' / 'meta_prompt_m0_candidate.txt'
CORE_SHA = '463692cd0a201d916c3f0e39d10cda4c50d2a9a1ca3305cd9de804c295e482b3'
M0_SHA = 'e4bef66552518e4a1206aaa3b14b9d34335c046499291e70dfdc33a79f243257'
M0_FIELDS = {
    'meta_prompt_version', 'task_name', 'mode', 'decision', 'task_scope',
    'supported_formats', 'applicability_conditions', 'failure_taxonomy',
    'adapter_text', 'fallback_policy', 'changes_from_previous',
    'predicted_regression_risks', 'evidence_limitations',
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def write_new_bytes(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing to overwrite different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_new_json(path: Path, payload: Any) -> None:
    write_new_bytes(path, json_bytes(payload))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def usage_from_response(response: Any) -> dict[str, int]:
    usage = getattr(response, 'usage', None)
    return {
        'input_tokens': int(getattr(usage, 'prompt_tokens', 0) or 0),
        'output_tokens': int(getattr(usage, 'completion_tokens', 0) or 0),
    }


def cost_for_usage(usage: dict[str, int], calls: int = 1) -> float:
    return compute_cost_from_usage({
        'candidate': {'model': MODEL, 'input_tokens': usage['input_tokens'], 'output_tokens': usage['output_tokens'], 'calls': calls},
        'judges': {},
    })


def runtime_text(adapter: str | None) -> str:
    core = CORE_PATH.read_text(encoding='utf-8').rstrip('\n')
    if not adapter:
        return core + '\n'
    return core + '\n\nTASK ADAPTER:\n' + adapter.strip() + '\n'


def save_runtime(task: str, label: str, adapter: str | None) -> Path:
    path = ADAPTER_DIR / task / f'runtime_{label}.txt'
    write_new_bytes(path, runtime_text(adapter).encode('utf-8'))
    return path


class PromptedOpenAI:
    def __init__(self, client: OpenAI, system_prompt: str | None):
        self.client = client
        self.system_prompt = system_prompt

    def call(self, user_prompt: str) -> tuple[str, dict[str, int], float, str | None]:
        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=MODEL, messages=messages, max_completion_tokens=8192,
            )
            wall = time.perf_counter() - started
            text = response.choices[0].message.content or ''
            return text, usage_from_response(response), wall, None
        except Exception as exc:
            return '', {'input_tokens': 0, 'output_tokens': 0}, time.perf_counter() - started, f'{type(exc).__name__}: {exc}'


def run_parallel(prompts: list[str], caller: PromptedOpenAI, concurrency: int = 2) -> list[tuple[str, dict[str, int], float, str | None]]:
    results: list[Any] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(caller.call, prompt): i for i, prompt in enumerate(prompts)}
        for future, index in futures.items():
            results[index] = future.result()
    return results


def aggregate_usage(rows: list[tuple[str, dict[str, int], float, str | None]]) -> dict[str, int]:
    return {
        'input_tokens': sum(row[1]['input_tokens'] for row in rows),
        'output_tokens': sum(row[1]['output_tokens'] for row in rows),
    }


def fixed_chance_score(items: list[dict[str, Any]]) -> float:
    raw = []
    chance = []
    for item in items:
        score = item.get('score')
        clean = item.get('parse_error') is None and isinstance(score, (int, float)) and math.isfinite(float(score))
        raw.append(float(score) if clean else 0.0)
        chance.append(float(item.get('chance', 0.0)))
    mean_raw = sum(raw) / len(raw)
    mean_chance = sum(chance) / len(chance)
    return (mean_raw - mean_chance) / (1.0 - mean_chance)


def chance_score_clean(items: list[dict[str, Any]]) -> float | None:
    clean = [item for item in items if item.get('parse_error') is None and isinstance(item.get('score'), (int, float)) and math.isfinite(float(item['score']))]
    if not clean:
        return None
    mean_raw = sum(float(item['score']) for item in clean) / len(clean)
    mean_chance = sum(float(item.get('chance', 0.0)) for item in clean) / len(clean)
    return (mean_raw - mean_chance) / (1.0 - mean_chance)


def group_scores(items: list[dict[str, Any]], field: str, score_fn: Callable[[list[dict[str, Any]]], float]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(str(item.get(field) or 'unknown'), []).append(item)
    return {key: score_fn(rows) for key, rows in sorted(groups.items())}


def fb_items(manifest: dict[str, Any], partition: str) -> list[Any]:
    rows = manifest[partition]
    by_split: dict[str, list[str]] = {}
    for row in rows:
        by_split.setdefault(row['split'], []).append(row['id'])
    source = []
    for split in by_split:
        source.extend(load_split(4, split=split, revision=REVISION, max_items=None))
    index = {item.id: item for item in source}
    return [index[row['id']] for row in rows]


def evaluate_factorybench_l4(client: OpenAI, manifest: dict[str, Any], partition: str, condition: str, runtime_path: Path | None, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        return load_json(output_path)
    items = fb_items(manifest, partition)
    system_prompt = runtime_path.read_text(encoding='utf-8') if runtime_path else None
    caller = PromptedOpenAI(client, system_prompt)
    batch_started = time.perf_counter()
    candidate_rows = run_parallel([render_prompt(item) for item in items], caller)
    panel = JudgePanel(['gpt-5.5'], concurrency=1)
    scored = []
    for item, (raw, usage, wall, transport_error) in zip(items, candidate_rows):
        result = _score_one(item, raw, panel=panel)
        row = {
            'id': result.id, 'level': result.level, 'template_id': result.template_id,
            'template_type': result.template_type, 'answer_format': result.answer_format,
            'dataset': result.dataset, 'episode': item.provenance.get('episode'),
            'raw_output': result.raw_output, 'parsed': result.parsed, 'score': result.score,
            'chance': result.chance, 'parse_error': result.parse_error,
            'judge_votes': result.judge_votes, 'transport_error': transport_error,
            'latency_seconds': wall, 'candidate_usage': usage,
            'development_evidence': {
                'rendered_input': render_prompt(item), 'reference_answer': item.answer,
                'reference_root_cause': item.root_cause,
            } if partition == 'development' else None,
        }
        scored.append(row)
    candidate_usage = aggregate_usage(candidate_rows)
    judge_usage = panel.usage_by_judge()
    tokens = {
        'candidate': {'model': MODEL, **candidate_usage, 'calls': sum(1 for row in candidate_rows if row[1]['input_tokens'] or row[1]['output_tokens'])},
        'judges': judge_usage,
    }
    payload = {
        'task_name': 'factorybench_l4', 'partition': partition, 'condition': condition,
        'model': MODEL, 'manifest_sha256': sha_file(MANIFEST_DIR / 'factorybench_l4_manifest.json'),
        'runtime_prompt_path': str(runtime_path.relative_to(ROOT)) if runtime_path else None,
        'runtime_prompt_sha256': sha_file(runtime_path) if runtime_path else None,
        'item_count': len(scored), 'ordered_ids': [row['id'] for row in scored],
        'canonical_score': chance_score_clean(scored), 'fixed_cardinality_score': fixed_chance_score(scored),
        'parse_failures': sum(row['parse_error'] is not None for row in scored),
        'by_answer_format': group_scores(scored, 'answer_format', fixed_chance_score),
        'by_dataset': group_scores(scored, 'dataset', fixed_chance_score),
        'tokens_used': tokens, 'cost': compute_cost_from_usage(tokens),
        'wall_time_seconds': time.perf_counter() - batch_started,
        'items': scored,
    }
    write_new_json(output_path, payload)
    return payload


def causal_source() -> dict[str, Any]:
    return load_json(REPO / 'data' / 'bigbench-ii' / 'causal_judgment' / 'task.json')


def causal_expected(example: dict[str, Any]) -> list[str]:
    scores = example.get('target_scores')
    if isinstance(scores, dict):
        best = max(float(value) for value in scores.values())
        return [str(key) for key, value in scores.items() if float(value) == best]
    target = example.get('target')
    return [str(value) for value in target] if isinstance(target, list) else [str(target)]


def causal_subgroup(text: str) -> str:
    question = text.strip().splitlines()[-1].casefold()
    return 'intentionality' if 'intentional' in question else 'causal_attribution'


def clean_label(text: str) -> str:
    value = text.strip()
    fenced = re.fullmatch(r'```(?:text|txt)?\s*(.*?)\s*```', value, re.S | re.I)
    if fenced:
        value = fenced.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()
    return value


def causal_manifest_examples(manifest: dict[str, Any], partition: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    task = causal_source()
    return [(row, task['examples'][int(row['source_index'])]) for row in manifest[partition]]


def accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(float(row['score']) if row.get('parse_error') is None else 0.0 for row in rows) / len(rows)


def evaluate_causal(client: OpenAI, manifest: dict[str, Any], partition: str, condition: str, runtime_path: Path | None, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        return load_json(output_path)
    examples = causal_manifest_examples(manifest, partition)
    system_prompt = runtime_path.read_text(encoding='utf-8') if runtime_path else None
    caller = PromptedOpenAI(client, system_prompt)
    batch_started = time.perf_counter()
    calls = run_parallel([example['input'] for _, example in examples], caller)
    items = []
    for (manifest_row, example), (raw, usage, wall, error) in zip(examples, calls):
        expected = causal_expected(example)
        parsed = clean_label(raw)
        valid = parsed.casefold() in {'yes', 'no'} and parsed == raw.strip()
        correct = any(parsed.casefold() == answer.casefold() for answer in expected)
        items.append({
            'id': manifest_row['id'], 'source_index': manifest_row['source_index'],
            'subgroup': causal_subgroup(example['input']), 'raw_output': raw,
            'parsed': parsed, 'score': 1.0 if correct else 0.0,
            'chance': 0.5, 'parse_error': error or (None if valid else 'expected exactly Yes or No'),
            'latency_seconds': wall, 'candidate_usage': usage,
            'development_evidence': {'input': example['input'], 'expected_outputs': expected} if partition == 'development' else None,
        })
    usage = aggregate_usage(calls)
    tokens = {'candidate': {'model': MODEL, **usage, 'calls': sum(1 for row in calls if row[1]['input_tokens'] or row[1]['output_tokens'])}, 'judges': {}}
    payload = {
        'task_name': 'causal_judgment', 'partition': partition, 'condition': condition,
        'model': MODEL, 'manifest_sha256': sha_file(MANIFEST_DIR / 'causal_judgment_manifest.json'),
        'runtime_prompt_path': str(runtime_path.relative_to(ROOT)) if runtime_path else None,
        'runtime_prompt_sha256': sha_file(runtime_path) if runtime_path else None,
        'item_count': len(items), 'ordered_ids': [row['id'] for row in items],
        'canonical_score': accuracy(items), 'fixed_cardinality_score': accuracy(items),
        'parse_failures': sum(row['parse_error'] is not None for row in items),
        'by_subgroup': group_scores(items, 'subgroup', accuracy),
        'tokens_used': tokens, 'cost': compute_cost_from_usage(tokens),
        'wall_time_seconds': time.perf_counter() - batch_started, 'items': items,
    }
    write_new_json(output_path, payload)
    return payload


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        'condition': result['condition'], 'item_count': result['item_count'],
        'ordered_ids': result['ordered_ids'], 'canonical_score': result['canonical_score'],
        'fixed_cardinality_score': result['fixed_cardinality_score'],
        'parse_failures': result['parse_failures'],
        'subgroups': result.get('by_subgroup') or {
            'answer_format': result.get('by_answer_format'), 'dataset': result.get('by_dataset')
        },
        'items': [{
            key: item.get(key) for key in (
                'id', 'answer_format', 'dataset', 'subgroup', 'raw_output', 'parsed',
                'score', 'chance', 'parse_error', 'development_evidence'
            ) if key in item
        } for item in result['items']],
    }


def task_contract(task: str, manifest_sha: str, mode: str, baseline: dict[str, Any], core: dict[str, Any], previous_result: dict[str, Any] | None, previous_adapter: dict[str, Any] | None) -> dict[str, Any]:
    if task == 'factorybench_l4':
        description = 'FactoryBench Level 4 industrial troubleshooting from machine telemetry. Items may require either one option letter or a concise diagnosis and remediation.'
        evaluator = {'canonical': 'FactoryBench chance-corrected scoring', 'mcq': 'deterministic single-letter parser and reference match', 'free_form': 'fixed FactoryBench L4 rubric with one gpt-5.5 judge', 'fixed_cardinality': 'failed parses retained as zero raw score'}
        contracts = ['single_letter_mcq: exactly one allowed option letter', 'free_form: answer the diagnosis/root-cause/remediation question directly']
        subgroups = ['answer_format', 'dataset']
    else:
        description = 'Binary causal attribution and intentionality judgment. Each scenario asks whether a specified factor caused or intentionally produced an outcome.'
        evaluator = {'canonical': 'case-insensitive exact-match accuracy', 'allowed_outputs': ['Yes', 'No'], 'format': 'exactly one label and no explanation'}
        contracts = ['exactly Yes or No']
        subgroups = ['causal_attribution', 'intentionality']
    return {
        'task_name': task, 'mode': mode, 'task_description': description,
        'execution_model': MODEL, 'evaluator_specification': evaluator,
        'output_contracts': contracts, 'subgroups': subgroups,
        'core_skill_sha256': CORE_SHA, 'development_manifest_sha256': manifest_sha,
        'development_results': {
            'baseline': compact_result(baseline), 'core_only': compact_result(core),
            'previous_adapter': compact_result(previous_result) if previous_result else None,
        },
        'previous_adapter': previous_adapter,
        'constraints': {'maximum_rounds': 2, 'manual_adapter_editing': False, 'holdout_access': False, 'case_memorization': False},
    }


def validate_m0_output(task: str, mode: str, parsed: Any, optimization_input: dict[str, Any], holdout_ids: list[str]) -> list[str]:
    errors = []
    if not isinstance(parsed, dict):
        return ['output is not an object']
    if set(parsed) != M0_FIELDS:
        errors.append(f'schema fields mismatch: {sorted(set(parsed) ^ M0_FIELDS)}')
    if parsed.get('meta_prompt_version') != 'm0' or parsed.get('task_name') != task or parsed.get('mode') != mode:
        errors.append('identity/version/mode mismatch')
    if parsed.get('decision') not in {'ADAPTER', 'NO_ADAPTER', 'INSUFFICIENT_DATA'}:
        errors.append('invalid decision')
    adapter = parsed.get('adapter_text')
    if not isinstance(adapter, str):
        errors.append('adapter_text must be a string')
        adapter = ''
    if parsed.get('decision') == 'ADAPTER' and not adapter.strip():
        errors.append('ADAPTER requires non-empty adapter_text')
    if parsed.get('decision') != 'ADAPTER' and adapter != '':
        errors.append('null decision requires empty adapter_text')
    for key in ('task_scope', 'supported_formats', 'applicability_conditions', 'failure_taxonomy', 'changes_from_previous', 'predicted_regression_risks', 'evidence_limitations'):
        if not isinstance(parsed.get(key), list):
            errors.append(f'{key} must be an array')
    if not isinstance(parsed.get('fallback_policy'), str):
        errors.append('fallback_policy must be a string')
    for item in parsed.get('failure_taxonomy') or []:
        if not isinstance(item, dict) or set(item) != {'category', 'evidence_count', 'description'} or not isinstance(item.get('evidence_count'), int):
            errors.append('invalid failure_taxonomy item')
            break
    lower = adapter.casefold()
    dev_items = optimization_input['development_results']['baseline']['items']
    for item in dev_items:
        if str(item['id']).casefold() in lower:
            errors.append(f'development ID leaked into adapter: {item["id"]}')
        evidence = item.get('development_evidence') or {}
        for key in ('reference_answer', 'reference_root_cause'):
            value = evidence.get(key)
            if isinstance(value, str) and len(value.strip()) >= 20 and value.strip().casefold() in lower:
                errors.append(f'copied development {key}')
    for item_id in holdout_ids:
        if item_id.casefold() in lower:
            errors.append(f'holdout ID referenced: {item_id}')
    if re.search(r'holdout|held[- ]?out|final[-_ ]?test', lower):
        errors.append('holdout/final-test reference in adapter')
    return errors


def call_m0(client: OpenAI, task: str, round_number: int, optimization_input: dict[str, Any], holdout_ids: list[str]) -> tuple[dict[str, Any], str | None, Path | None]:
    task_trace = TRACE_DIR / task
    input_path = task_trace / f'm0_round_{round_number}_input.json'
    raw_path = task_trace / f'm0_round_{round_number}_raw_output.txt'
    parsed_path = task_trace / f'm0_round_{round_number}_parsed_output.json'
    write_new_json(input_path, optimization_input)
    if raw_path.exists() or parsed_path.exists():
        parsed = load_json(parsed_path)
        adapter_path = ADAPTER_DIR / task / f'adapter_v{round_number}.txt'
        return parsed, parsed.get('adapter_text') or None, adapter_path if adapter_path.exists() else None
    user = '<OPTIMIZATION_INPUT>\n' + json.dumps(optimization_input, indent=2, ensure_ascii=False, allow_nan=False) + '\n</OPTIMIZATION_INPUT>'
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'system', 'content': M0_PATH.read_text(encoding='utf-8')}, {'role': 'user', 'content': user}],
        max_completion_tokens=8192,
    )
    wall = time.perf_counter() - started
    raw = response.choices[0].message.content or ''
    write_new_bytes(raw_path, raw.encode('utf-8'))
    try:
        parsed = json.loads(raw.strip())
        parse_error = None
    except json.JSONDecodeError as exc:
        parsed = {}
        parse_error = f'{type(exc).__name__}: {exc}'
    errors = ([parse_error] if parse_error else []) + validate_m0_output(task, optimization_input['mode'], parsed, optimization_input, holdout_ids)
    usage = usage_from_response(response)
    envelope = {
        **parsed,
        '_trace_validation': {
            'valid': not errors, 'errors': errors, 'input_sha256': sha_file(input_path),
            'raw_output_sha256': sha_file(raw_path), 'meta_prompt_sha256': M0_SHA,
            'usage': usage, 'cost': cost_for_usage(usage), 'wall_time_seconds': wall,
        },
    }
    write_new_json(parsed_path, envelope)
    if errors:
        raise RuntimeError(f'invalid M0 output for {task} round {round_number}: {errors}')
    adapter_path = None
    adapter = parsed['adapter_text'] if parsed['decision'] == 'ADAPTER' else None
    if adapter:
        adapter_path = ADAPTER_DIR / task / f'adapter_v{round_number}.txt'
        write_new_bytes(adapter_path, adapter.encode('utf-8'))
    return envelope, adapter, adapter_path


def result_path(task: str, partition: str, condition: str) -> Path:
    directory = DEVELOPMENT_DIR if partition == 'development' else HOLDOUT_DIR
    return directory / f'{task}_{condition}.json'


def evaluate(client: OpenAI, task: str, manifest: dict[str, Any], partition: str, condition: str, runtime_path: Path | None) -> dict[str, Any]:
    path = result_path(task, partition, condition)
    if task == 'factorybench_l4':
        return evaluate_factorybench_l4(client, manifest, partition, condition, runtime_path, path)
    return evaluate_causal(client, manifest, partition, condition, runtime_path, path)


def worst_subgroup(result: dict[str, Any]) -> float:
    groups = result.get('by_subgroup') or {}
    if result.get('by_answer_format'):
        groups = {**groups, **{f'format:{key}': value for key, value in result['by_answer_format'].items()}}
    if result.get('by_dataset'):
        groups = {**groups, **{f'dataset:{key}': value for key, value in result['by_dataset'].items()}}
    return min(groups.values()) if groups else float(result['fixed_cardinality_score'])


def candidate_eligibility(result: dict[str, Any], expected_ids: list[str], core: dict[str, Any], adapter_path: Path | None) -> tuple[bool, list[str]]:
    reasons = []
    if result['ordered_ids'] != expected_ids or result['item_count'] != len(expected_ids):
        reasons.append('item mismatch')
    if result['parse_failures'] != 0:
        reasons.append('parse failure')
    if not math.isfinite(float(result['fixed_cardinality_score'])):
        reasons.append('non-finite score')
    if any(item.get('parse_error') is not None for item in result['items']):
        reasons.append('invalid output contract')
    if adapter_path:
        text = adapter_path.read_text(encoding='utf-8')
        if any(item_id in text for item_id in expected_ids):
            reasons.append('case memorization ID')
    core_groups = core.get('by_subgroup') or {}
    result_groups = result.get('by_subgroup') or {}
    for field in ('by_answer_format', 'by_dataset'):
        for key, value in (core.get(field) or {}).items():
            if key in (result.get(field) or {}) and result[field][key] < value:
                reasons.append(f'critical subgroup regression {field}:{key}')
    for key, value in core_groups.items():
        if key in result_groups and result_groups[key] < value:
            reasons.append(f'critical subgroup regression:{key}')
    return not reasons, reasons


def select_adapter(task: str, manifest: dict[str, Any], core: dict[str, Any], candidates: list[tuple[str, dict[str, Any], Path | None]]) -> dict[str, Any]:
    selection_path = ADAPTER_DIR / task / 'selection.json'
    if selection_path.exists():
        return load_json(selection_path)
    expected_ids = [row['id'] for row in manifest['development']]
    records = []
    eligible_adapter_records = []
    for label, result, adapter_path in candidates:
        eligible, reasons = candidate_eligibility(result, expected_ids, core, adapter_path)
        record = {
            'candidate': label, 'eligible': eligible, 'ineligibility_reasons': reasons,
            'fixed_cardinality_score': result['fixed_cardinality_score'],
            'worst_subgroup_score': worst_subgroup(result), 'parse_failures': result['parse_failures'],
            'adapter_path': str(adapter_path.relative_to(ROOT)) if adapter_path else None,
            'adapter_sha256': sha_file(adapter_path) if adapter_path else None,
            'adapter_bytes': adapter_path.stat().st_size if adapter_path else 0,
        }
        records.append(record)
        if label != 'core_only' and eligible and result['fixed_cardinality_score'] > core['fixed_cardinality_score']:
            eligible_adapter_records.append(record)
    selected = None
    if eligible_adapter_records:
        round_order = {'adapter_v1': 1, 'adapter_v2': 2}
        selected = sorted(eligible_adapter_records, key=lambda row: (-row['fixed_cardinality_score'], -row['worst_subgroup_score'], row['parse_failures'], row['adapter_bytes'], round_order[row['candidate']]))[0]
    if selected is None:
        decision = 'NO_ADAPTER'
        selected_label = 'core_only'
        selected_hash = None
    else:
        decision = 'ADAPTER'
        selected_label = selected['candidate']
        source = ROOT / selected['adapter_path']
        destination = ADAPTER_DIR / task / 'selected_adapter.txt'
        write_new_bytes(destination, source.read_bytes())
        selected_hash = sha_file(destination)
    payload = {
        'task_name': task, 'decision': decision, 'selected_candidate': selected_label,
        'selected_adapter_sha256': selected_hash, 'development_manifest_sha256': sha_file(MANIFEST_DIR / f'{task}_manifest.json'),
        'core_skill_sha256': CORE_SHA, 'selection_rule': [
            'hard eligibility gates', 'adapter must strictly beat Core-only', 'highest fixed-cardinality score',
            'highest worst-subgroup score', 'lowest parse failures', 'shortest adapter bytes', 'v1 before v2'
        ],
        'core_only_score': core['fixed_cardinality_score'], 'candidates': records,
    }
    write_new_json(selection_path, payload)
    return payload


def run_development(client: OpenAI, task: str) -> dict[str, Any]:
    manifest_path = MANIFEST_DIR / f'{task}_manifest.json'
    manifest = load_json(manifest_path)
    if manifest['status'] != 'AVAILABLE':
        return {'task_name': task, 'status': manifest['status'], 'reason': manifest.get('reason')}
    core_runtime = save_runtime(task, 'core_only', None)
    baseline = evaluate(client, task, manifest, 'development', 'baseline', None)
    core = evaluate(client, task, manifest, 'development', 'core_only', core_runtime)
    holdout_ids = [row['id'] for row in manifest['holdout']]
    input_v1 = task_contract(task, sha_file(manifest_path), 'generate', baseline, core, None, None)
    output_v1, adapter_v1, adapter_path_v1 = call_m0(client, task, 1, input_v1, holdout_ids)
    candidates = [('core_only', core, None)]
    result_v1 = None
    if adapter_v1 and adapter_path_v1:
        runtime_v1 = save_runtime(task, 'adapter_v1', adapter_v1)
        result_v1 = evaluate(client, task, manifest, 'development', 'adapter_v1', runtime_v1)
        candidates.append(('adapter_v1', result_v1, adapter_path_v1))
    previous_record = {'sha256': sha_file(adapter_path_v1), 'text': adapter_v1} if adapter_path_v1 else {'sha256': None, 'text': ''}
    input_v2 = task_contract(task, sha_file(manifest_path), 'refine', baseline, core, result_v1, previous_record)
    output_v2, adapter_v2, adapter_path_v2 = call_m0(client, task, 2, input_v2, holdout_ids)
    result_v2 = None
    if adapter_v2 and adapter_path_v2:
        runtime_v2 = save_runtime(task, 'adapter_v2', adapter_v2)
        result_v2 = evaluate(client, task, manifest, 'development', 'adapter_v2', runtime_v2)
        candidates.append(('adapter_v2', result_v2, adapter_path_v2))
    selection = select_adapter(task, manifest, core, candidates)
    summary = {
        'task_name': task, 'status': 'DEVELOPMENT_COMPLETE',
        'manifest_sha256': sha_file(manifest_path), 'baseline_score': baseline['fixed_cardinality_score'],
        'core_only_score': core['fixed_cardinality_score'],
        'adapter_v1_decision': output_v1.get('decision'), 'adapter_v1_sha256': sha_file(adapter_path_v1) if adapter_path_v1 else None,
        'adapter_v1_score': result_v1['fixed_cardinality_score'] if result_v1 else None,
        'adapter_v2_decision': output_v2.get('decision'), 'adapter_v2_sha256': sha_file(adapter_path_v2) if adapter_path_v2 else None,
        'adapter_v2_score': result_v2['fixed_cardinality_score'] if result_v2 else None,
        'selection': selection,
    }
    write_new_json(DEVELOPMENT_DIR / f'{task}_summary.json', summary)
    return summary


def run_holdout(client: OpenAI, task: str) -> dict[str, Any]:
    manifest_path = MANIFEST_DIR / f'{task}_manifest.json'
    manifest = load_json(manifest_path)
    if manifest['status'] != 'AVAILABLE':
        return {'task_name': task, 'status': manifest['status'], 'reason': manifest.get('reason')}
    selection_path = ADAPTER_DIR / task / 'selection.json'
    if not selection_path.exists():
        raise RuntimeError(f'selection must be frozen before holdout: {selection_path}')
    selection = load_json(selection_path)
    core_runtime = ADAPTER_DIR / task / 'runtime_core_only.txt'
    baseline = evaluate(client, task, manifest, 'holdout', 'baseline', None)
    core = evaluate(client, task, manifest, 'holdout', 'core_only', core_runtime)
    selected = None
    if selection['decision'] == 'ADAPTER':
        adapter_text = (ADAPTER_DIR / task / 'selected_adapter.txt').read_text(encoding='utf-8')
        selected_runtime = save_runtime(task, 'selected_adapter', adapter_text)
        selected = evaluate(client, task, manifest, 'holdout', 'selected_adapter', selected_runtime)
        selected_label = 'selected_adapter'
    else:
        selected_label = 'NO_ADAPTER (identical to Core-only; no duplicate model calls)'
    summary = {
        'task_name': task, 'status': 'HOLDOUT_COMPLETE', 'manifest_sha256': sha_file(manifest_path),
        'selection_sha256': sha_file(selection_path), 'selection_decision': selection['decision'],
        'selected_condition_label': selected_label,
        'baseline': compact_result(baseline), 'core_only': compact_result(core),
        'selected_adapter': compact_result(selected) if selected else None,
        'no_holdout_feedback_to_m0': True,
    }
    write_new_json(HOLDOUT_DIR / f'{task}_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'holdout'], required=True)
    parser.add_argument('--task', choices=['factorybench_l4', 'causal_judgment'], required=True)
    args = parser.parse_args()
    load_dotenv(REPO / '.env')
    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('credential preflight failed: OPENAI_API_KEY is not set')
    if sha_file(CORE_PATH) != CORE_SHA or sha_file(M0_PATH) != M0_SHA:
        raise SystemExit('frozen Core or M0 hash mismatch')
    client = OpenAI()
    if args.phase == 'development':
        result = run_development(client, args.task)
    else:
        result = run_holdout(client, args.task)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
