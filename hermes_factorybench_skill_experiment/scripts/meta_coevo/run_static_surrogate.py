#!/usr/bin/env python3
"""Arm B: FactoryBench L1-L3 with a fixed surrogate verifier and role-separated models.

Research roles
--------------
- AGENT_MODEL: fixed weaker/edge-proxy model that answers FactoryBench items.
- SURROGATE_MODEL: stronger model that independently diagnoses answer quality.
- OPTIMIZER_MODEL: stronger model that revises the reusable Skill/Adapter.

All three roles use the same OPENAI_API_KEY. Only the model IDs differ.
The surrogate never receives reference answers, GT scores, acceptance bounds,
or holdout information.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from factorybench.cost import compute_cost_from_usage
from factorybench.data import load_split
from factorybench.evaluate import _score_one
from factorybench.prompt import render_prompt

# -----------------------------------------------------------------------------
# Paths / frozen protocol
# -----------------------------------------------------------------------------
REPO = Path('/home/training/automatic_prompt_engineer')
ROOT = REPO / 'hermes_factorybench_skill_experiment'

ARM = 'surrogate_static'
REV = 'b3863519ccedbceab54dfa7600104eb42b985ed7'

M1_PATH = ROOT / 'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt'
M1_SHA = '78187e3268294657d2398c9a79563a36f050c4189b2f6650cc569407512cb052'
SURROGATE_PATH = ROOT / 'prompts/meta_coevo/surrogate_verifier_v0.txt'

MANIFEST_DIR = ROOT / 'data_manifests/meta_m1'
RESULT_DIR = ROOT / 'results/meta_coevo/static'
TRACE_DIR = ROOT / 'traces/meta_coevo/static'
ADAPTER_ROOT = ROOT / 'prompts/adapters/meta_coevo_static'

# -----------------------------------------------------------------------------
# Model roles
# Same OPENAI_API_KEY is used for every role. Override IDs via env vars if needed.
# -----------------------------------------------------------------------------
AGENT_MODEL = os.getenv('AGENT_MODEL', 'gpt-4o-mini')
SURROGATE_MODEL = os.getenv('SURROGATE_MODEL', 'gpt-5.6-luna')
OPTIMIZER_MODEL = os.getenv('OPTIMIZER_MODEL', 'gpt-5.6-luna')

MAX_COMPLETION_TOKENS = int(os.getenv('MAX_COMPLETION_TOKENS', '8192'))
AGENT_CONCURRENCY = int(os.getenv('AGENT_CONCURRENCY', '2'))

M1_FIELDS = {
    'meta_prompt_version',
    'task_name',
    'mode',
    'decision',
    'task_scope',
    'supported_subtasks',
    'supported_formats',
    'applicability_conditions',
    'failure_taxonomy',
    'adapter_text',
    'fallback_policy',
    'changes_from_previous',
    'predicted_regression_risks',
    'evidence_limitations',
}


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    """Write once. Refuse to silently overwrite a different artifact."""
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, obj: Any) -> None:
    write_new(
        path,
        (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode(),
    )


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def usage(response: Any) -> dict[str, int]:
    u = getattr(response, 'usage', None)
    return {
        'input_tokens': int(
            getattr(u, 'input_tokens', getattr(u, 'prompt_tokens', 0)) or 0
        ),
        'output_tokens': int(
            getattr(u, 'output_tokens', getattr(u, 'completion_tokens', 0)) or 0
        ),
    }


def safe_cost(model: str, role_usage: dict[str, int]) -> float | None:
    """Return repo-calculated cost if the local price table knows the model."""
    try:
        return float(
            compute_cost_from_usage(
                {
                    'candidate': {
                        'model': model,
                        **role_usage,
                    },
                    'judges': {},
                }
            )
        )
    except Exception:
        # Cost accounting must never break the experiment itself.
        return None


def total_known_cost(*values: float | None) -> float | None:
    known = [v for v in values if v is not None]
    if len(known) != len(values):
        return None
    return float(sum(known))


# -----------------------------------------------------------------------------
# Dataset / FactoryBench scoring helpers
# -----------------------------------------------------------------------------
def manifests(task: str) -> dict[str, Path]:
    if task != 'm1_factorybench_l123':
        raise ValueError(f'unsupported task for Arm B: {task}')
    prefix = 'factorybench_l123'
    return {
        'fold_a': MANIFEST_DIR / f'{prefix}_dev_fold_a.json',
        'fold_b': MANIFEST_DIR / f'{prefix}_dev_fold_b.json',
        'holdout': MANIFEST_DIR / f'{prefix}_holdout.json',
    }


def adapter_dir(task: str) -> Path:
    return ADAPTER_ROOT / task


def result_path(task: str, part: str, condition: str) -> Path:
    return RESULT_DIR / part / f'{task}_{condition}.json'


def source_items(path: Path):
    manifest = load(path)
    cache: dict[tuple[int, str], dict[str, Any]] = {}

    for row in manifest['items']:
        key = (row['level'], row['split'])
        if key not in cache:
            cache[key] = {
                x.id: x
                for x in load_split(
                    key[0],
                    split=key[1],
                    revision=REV,
                    max_items=None,
                )
            }

    items = [
        cache[(row['level'], row['split'])][row['id']]
        for row in manifest['items']
    ]

    for row, item in zip(manifest['items'], items):
        if item.provenance.get('episode') != row['episode']:
            raise RuntimeError('manifest episode mismatch')
        if item.answer_format.value != row['answer_format']:
            raise RuntimeError('manifest answer-format mismatch')

    return manifest, items


def fixed(rows: list[dict[str, Any]]) -> float:
    scores = []
    chances = []
    for row in rows:
        clean = (
            row.get('parse_error') is None
            and isinstance(row.get('score'), (int, float))
            and math.isfinite(float(row['score']))
        )
        scores.append(float(row['score']) if clean else 0.0)
        chances.append(float(row.get('chance', 0)))

    mean_chance = sum(chances) / len(chances)
    return (sum(scores) / len(scores) - mean_chance) / (1 - mean_chance)


def canonical(rows: list[dict[str, Any]]) -> float | None:
    clean = [
        row
        for row in rows
        if row.get('parse_error') is None
        and isinstance(row.get('score'), (int, float))
        and math.isfinite(float(row['score']))
    ]
    if not clean:
        return None

    mean_chance = sum(float(row.get('chance', 0)) for row in clean) / len(clean)
    return (
        sum(float(row['score']) for row in clean) / len(clean) - mean_chance
    ) / (1 - mean_chance)


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(field) or 'unknown'), []).append(row)
    return {key: fixed(value) for key, value in sorted(buckets.items())}


def gt_pass_from_score(score: float | None, parse_error: str | None) -> bool:
    """Binary Arm-B smoke rule: only a fully correct item is GT PASS.

    FOUR_LETTER_TF and TENSOR_MARGIN may receive partial credit in FactoryBench.
    For this first static-surrogate experiment, partial credit is intentionally
    treated as GT FAIL so PASS means fully correct.
    """
    if parse_error is not None or score is None:
        return False
    try:
        value = float(score)
    except Exception:
        return False
    return math.isfinite(value) and math.isclose(
        value,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


# -----------------------------------------------------------------------------
# OpenAI role calls
# -----------------------------------------------------------------------------
def call_chat(
    client: OpenAI,
    *,
    model: str,
    system: str | None,
    prompt: str,
) -> tuple[str, dict[str, int], float, str | None]:
    started = time.perf_counter()
    try:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        return (
            response.choices[0].message.content or '',
            usage(response),
            time.perf_counter() - started,
            None,
        )
    except Exception as exc:
        return (
            '',
            {'input_tokens': 0, 'output_tokens': 0},
            time.perf_counter() - started,
            f'{type(exc).__name__}: {exc}',
        )


def parallel_agent_calls(
    client: OpenAI,
    system: str | None,
    prompts: list[str],
):
    """Only the target/agent model answers FactoryBench items."""
    output = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=AGENT_CONCURRENCY) as pool:
        futures = {
            pool.submit(
                call_chat,
                client,
                model=AGENT_MODEL,
                system=system,
                prompt=prompt,
            ): index
            for index, prompt in enumerate(prompts)
        }
        for future in as_completed(futures):
            output[futures[future]] = future.result()
    return output


def run_surrogate(client: OpenAI, item: Any, raw_output: str) -> dict[str, Any]:
    """Fixed verifier v0. No GT/reference/acceptance-bound fields are exposed."""
    system = SURROGATE_PATH.read_text(encoding='utf-8')

    # IMPORTANT: this payload intentionally contains no item.answer,
    # acceptance_bounds, score, chance, reference_answer, or holdout metadata.
    payload = {
        'question': render_prompt(item),
        'answer_format': item.answer_format.value,
        'agent_answer': raw_output,
    }

    raw, role_usage, latency, transport_error = call_chat(
        client,
        model=SURROGATE_MODEL,
        system=system,
        prompt=json.dumps(payload, ensure_ascii=False, indent=2),
    )

    parsed = None
    parse_error = None
    verdict = None

    if transport_error is None:
        try:
            parsed = json.loads(raw.strip())
        except Exception as exc:
            parse_error = f'surrogate_json_error: {type(exc).__name__}: {exc}'

    if parsed is not None:
        if not isinstance(parsed, dict):
            parse_error = 'surrogate_output_not_object'
        else:
            verdict = parsed.get('verdict')
            if verdict not in {'PASS', 'FAIL'}:
                verdict = None
                parse_error = 'invalid_surrogate_verdict'

    return {
        'model': SURROGATE_MODEL,
        'raw_output': raw,
        'parsed': parsed,
        'verdict': verdict,
        'diagnosis': parsed.get('diagnosis', []) if isinstance(parsed, dict) else [],
        'failed_checks': (
            parsed.get('failed_checks', []) if isinstance(parsed, dict) else []
        ),
        'confidence': parsed.get('confidence') if isinstance(parsed, dict) else None,
        'parse_error': parse_error,
        'transport_error': transport_error,
        'usage': role_usage,
        'latency_seconds': latency,
    }


# -----------------------------------------------------------------------------
# Arm-B evaluation
# -----------------------------------------------------------------------------
def surrogate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if row.get('surrogate_verdict') in {'PASS', 'FAIL'}
    ]
    gt_pass_rows = [row for row in valid if row.get('gt_pass') is True]
    gt_fail_rows = [row for row in valid if row.get('gt_pass') is False]

    false_pass = sum(bool(row.get('false_pass')) for row in valid)
    false_reject = sum(bool(row.get('false_reject')) for row in valid)
    agreement = sum(
        (row.get('surrogate_verdict') == 'PASS') == bool(row.get('gt_pass'))
        for row in valid
    )

    return {
        'valid_count': len(valid),
        'parse_or_transport_failures': len(rows) - len(valid),
        'agreement_rate': (agreement / len(valid)) if valid else None,
        'false_pass_count': false_pass,
        'false_reject_count': false_reject,
        'false_pass_rate_on_gt_fail': (
            false_pass / len(gt_fail_rows) if gt_fail_rows else None
        ),
        'false_reject_rate_on_gt_pass': (
            false_reject / len(gt_pass_rows) if gt_pass_rows else None
        ),
    }


def evaluate_l123(
    client: OpenAI,
    manifest_path: Path,
    part: str,
    condition: str,
    adapter: str | None,
):
    path = result_path('m1_factorybench_l123', part, condition)
    if path.exists():
        return load(path)

    if not SURROGATE_PATH.exists():
        raise RuntimeError(f'missing surrogate prompt: {SURROGATE_PATH}')

    manifest, items = source_items(manifest_path)
    system = adapter if adapter else None
    started = time.perf_counter()

    # Agent answers first, with the fixed weak/edge-proxy model.
    calls = parallel_agent_calls(
        client,
        system,
        [render_prompt(item) for item in items],
    )

    rows = []
    surrogate_usage = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}

    for item, (raw, agent_usage, latency, transport_error) in zip(items, calls):
        # 1) Independent surrogate runs BEFORE hidden GT scoring.
        surrogate = run_surrogate(client, item, raw)
        su = surrogate.get('usage') or {}
        surrogate_usage['input_tokens'] += int(su.get('input_tokens', 0) or 0)
        surrogate_usage['output_tokens'] += int(su.get('output_tokens', 0) or 0)
        if su.get('input_tokens') or su.get('output_tokens'):
            surrogate_usage['calls'] += 1

        # 2) Hidden FactoryBench deterministic GT oracle.
        scored = _score_one(item, raw)
        finite_score = (
            scored.score
            if isinstance(scored.score, (int, float))
            and math.isfinite(float(scored.score))
            else None
        )
        item_parse_error = scored.parse_error or (
            'non_finite_score' if finite_score is None else None
        )
        gt_pass = gt_pass_from_score(finite_score, item_parse_error)

        surrogate_verdict = surrogate.get('verdict')
        false_pass = surrogate_verdict == 'PASS' and not gt_pass
        false_reject = surrogate_verdict == 'FAIL' and gt_pass

        rows.append(
            {
                'id': item.id,
                'level': item.level,
                'split': next(
                    row['split']
                    for row in manifest['items']
                    if row['id'] == item.id
                ),
                'dataset': item.dataset,
                'episode': item.provenance.get('episode'),
                'answer_format': item.answer_format.value,
                'agent_model': AGENT_MODEL,
                'raw_output': raw,
                'parsed': scored.parsed,
                'score': finite_score,
                'chance': scored.chance,
                'parse_error': item_parse_error,
                'gt_pass': gt_pass,
                'surrogate': surrogate,
                'surrogate_verdict': surrogate_verdict,
                'false_pass': false_pass,
                'false_reject': false_reject,
                'transport_error': transport_error,
                'usage': agent_usage,
                'latency_seconds': latency,
                # GT is available only to the development-side optimizer packet.
                # It is never sent to the surrogate verifier.
                'development_evidence': {
                    'rendered_input': render_prompt(item),
                    'reference_answer': item.answer,
                }
                if part == 'development'
                else None,
            }
        )

    agent_usage_total = {
        'input_tokens': sum(call[1]['input_tokens'] for call in calls),
        'output_tokens': sum(call[1]['output_tokens'] for call in calls),
        'calls': sum(
            bool(call[1]['input_tokens'] or call[1]['output_tokens'])
            for call in calls
        ),
    }

    agent_cost = safe_cost(AGENT_MODEL, agent_usage_total)
    surrogate_cost = safe_cost(SURROGATE_MODEL, surrogate_usage)

    payload = {
        'arm': ARM,
        'task_name': 'm1_factorybench_l123',
        'partition': part,
        'condition': condition,
        'model_roles': {
            'agent_model': AGENT_MODEL,
            'surrogate_model': SURROGATE_MODEL,
            'optimizer_model': OPTIMIZER_MODEL,
        },
        'manifest_path': str(manifest_path.relative_to(ROOT)),
        'manifest_sha256': sha(manifest_path),
        'adapter_sha256': (
            hashlib.sha256(adapter.encode()).hexdigest() if adapter else None
        ),
        'surrogate_prompt_path': str(SURROGATE_PATH.relative_to(ROOT)),
        'surrogate_prompt_sha256': sha(SURROGATE_PATH),
        'surrogate_static': True,
        'core_prompt_used': False,
        'ordered_ids': [row['id'] for row in rows],
        'item_count': len(rows),
        'canonical_score': canonical(rows),
        'fixed_cardinality_score': fixed(rows),
        'parse_failures': sum(row['parse_error'] is not None for row in rows),
        'by_level': grouped(rows, 'level'),
        'by_format': grouped(rows, 'answer_format'),
        'by_dataset': grouped(rows, 'dataset'),
        'surrogate_summary': surrogate_summary(rows),
        'tokens_used': {
            'agent': {'model': AGENT_MODEL, **agent_usage_total},
            'surrogate': {'model': SURROGATE_MODEL, **surrogate_usage},
        },
        'cost': {
            'agent': agent_cost,
            'surrogate': surrogate_cost,
            'known_total': total_known_cost(agent_cost, surrogate_cost),
        },
        'wall_time_seconds': time.perf_counter() - started,
        'items': rows,
    }

    write_json(path, payload)
    return payload


def evaluate(
    client: OpenAI,
    task: str,
    manifest_path: Path,
    part: str,
    condition: str,
    adapter: str | None,
):
    if task != 'm1_factorybench_l123':
        raise ValueError('Arm B supports only m1_factorybench_l123')
    return evaluate_l123(client, manifest_path, part, condition, adapter)


# -----------------------------------------------------------------------------
# Optimizer packet / Skill revision (Luna)
# -----------------------------------------------------------------------------
def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        'condition': result['condition'],
        'item_count': result['item_count'],
        'ordered_ids': result['ordered_ids'],
        'canonical_score': result['canonical_score'],
        'fixed_cardinality_score': result['fixed_cardinality_score'],
        'parse_failures': result['parse_failures'],
        'by_format': result['by_format'],
        'by_dataset': result['by_dataset'],
        'by_level': result['by_level'],
        'surrogate_summary': result.get('surrogate_summary'),
        'items': [
            {
                **{
                    key: item.get(key)
                    for key in (
                        'id',
                        'level',
                        'dataset',
                        'answer_format',
                        'raw_output',
                        'parsed',
                        'score',
                        'chance',
                        'parse_error',
                        'gt_pass',
                        'surrogate_verdict',
                        'false_pass',
                        'false_reject',
                        'development_evidence',
                    )
                    if key in item
                },
                'surrogate_diagnosis': (
                    (item.get('surrogate') or {}).get('diagnosis', [])
                ),
                'surrogate_failed_checks': (
                    (item.get('surrogate') or {}).get('failed_checks', [])
                ),
            }
            for item in result['items']
        ],
    }


def m1_input(
    task: str,
    mode: str,
    basea: dict[str, Any],
    baseb: dict[str, Any],
    v1a: dict[str, Any] | None,
    v1b: dict[str, Any] | None,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    if task != 'm1_factorybench_l123':
        raise ValueError(f'unsupported task: {task}')

    ms = manifests(task)
    return {
        'task_name': task,
        'mode': mode,
        'task_description': (
            'FactoryBench L1-L3 manufacturing telemetry tasks with strict '
            'item-specific scalar, tensor, ranking, Boolean, and MCQ contracts.'
        ),
        'execution_model': AGENT_MODEL,
        'model_roles': {
            'agent_model': AGENT_MODEL,
            'surrogate_model': SURROGATE_MODEL,
            'optimizer_model': OPTIMIZER_MODEL,
        },
        'evaluator_specification': {
            'canonical': 'FactoryBench chance-corrected score',
            'fixed_cardinality': 'parse failures retained as zero',
            'parser': 'FactoryBench deterministic parser/scorer',
            'binary_gt_for_surrogate_comparison': 'PASS iff item score == 1.0',
        },
        'output_contracts': ['Follow each item-specific exact output contract.'],
        'supported_subtasks': [
            'L1 predictive/identification',
            'L2 predictive/identification',
            'L3 predictive',
        ],
        'surrogate_verifier': {
            'enabled': True,
            'mode': 'static_v0',
            'model': SURROGATE_MODEL,
            'prompt_sha256': sha(SURROGATE_PATH),
            'gt_visible_to_surrogate': False,
            'diagnosis_available_to_optimizer': True,
        },
        'development_manifests': {
            'fold_a_sha256': sha(ms['fold_a']),
            'fold_b_sha256': sha(ms['fold_b']),
        },
        'development_results': {
            'baseline_fold_a': compact(basea),
            'baseline_fold_b': compact(baseb),
            'adapter_v1_fold_a': compact(v1a) if v1a else None,
            'adapter_v1_fold_b': compact(v1b) if v1b else None,
        },
        'previous_adapter': previous,
        'constraints': {
            'maximum_adapter_rounds': 2,
            'manual_adapter_editing': False,
            'core_prompt': False,
            'holdout_access': False,
            'case_memorization': False,
            'surrogate_verifier_static': True,
        },
    }


def validate_m1(
    parsed: Any,
    task: str,
    mode: str,
    data: dict[str, Any],
    hold_ids: list[str],
) -> list[str]:
    errors: list[str] = []

    if not isinstance(parsed, dict) or set(parsed) != M1_FIELDS:
        return ['schema fields mismatch']

    if (
        parsed.get('meta_prompt_version') != 'm1'
        or parsed.get('task_name') != task
        or parsed.get('mode') != mode
    ):
        errors.append('identity mismatch')

    if (
        parsed.get('decision') not in {'ADAPTER', 'NO_ADAPTER', 'INSUFFICIENT_DATA'}
        or not isinstance(parsed.get('adapter_text'), str)
    ):
        errors.append('decision/text invalid')

    adapter = parsed.get('adapter_text', '')
    if parsed.get('decision') == 'ADAPTER' and not adapter.strip():
        errors.append('empty adapter')
    if parsed.get('decision') != 'ADAPTER' and adapter != '':
        errors.append('nonempty null adapter')

    dev_items = (
        data['development_results']['baseline_fold_a']['items']
        + data['development_results']['baseline_fold_b']['items']
    )
    low = adapter.casefold()

    for item in dev_items:
        if item['id'].casefold() in low:
            errors.append('development ID leak')

        evidence = item.get('development_evidence') or {}
        gold = str(evidence.get('reference_answer') or '')
        if len(gold) >= 20 and gold.casefold() in low:
            errors.append('copied gold answer')

        rendered = str(evidence.get('rendered_input') or '')
        numbers = set(
            re.findall(
                r'(?<![A-Za-z])(?:\d{4,}|-?\d+\.\d{3,})(?![A-Za-z])',
                rendered,
            )
        )
        if any(number in adapter for number in numbers):
            errors.append('case-specific signal value')
            break

    if any(item_id.casefold() in low for item_id in hold_ids):
        errors.append('holdout ID leak')
    if re.search(r'holdout|held[- ]?out|final[-_ ]?test', low):
        errors.append('holdout reference')

    for key in (
        'task_scope',
        'supported_subtasks',
        'supported_formats',
        'applicability_conditions',
        'failure_taxonomy',
        'changes_from_previous',
        'predicted_regression_risks',
        'evidence_limitations',
    ):
        if not isinstance(parsed.get(key), list):
            errors.append(f'{key} invalid')

    return errors


def call_m1(
    client: OpenAI,
    task: str,
    roundn: int,
    data: dict[str, Any],
):
    """Use the high-end optimizer model to revise the reusable Skill/Adapter."""
    trace_dir = TRACE_DIR / task
    input_path = trace_dir / f'm1_round_{roundn}_input.json'
    raw_path = trace_dir / f'm1_round_{roundn}_raw_output.txt'
    parsed_path = trace_dir / f'm1_round_{roundn}_parsed_output.json'

    write_json(input_path, data)
    if raw_path.exists() or parsed_path.exists():
        raise RuntimeError('existing partial M1 trace')

    user = (
        '<MANUFACTURING_PROMPT_OPTIMIZATION_INPUT>\n'
        + json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
        + '\n</MANUFACTURING_PROMPT_OPTIMIZATION_INPUT>'
    )

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=OPTIMIZER_MODEL,
        messages=[
            {'role': 'system', 'content': M1_PATH.read_text(encoding='utf-8')},
            {'role': 'user', 'content': user},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
    )
    wall = time.perf_counter() - started
    raw = response.choices[0].message.content or ''
    write_new(raw_path, raw.encode())

    try:
        parsed = json.loads(raw.strip())
        json_error = None
    except Exception as exc:
        parsed = {}
        json_error = f'{type(exc).__name__}: {exc}'

    hold_ids = [item['id'] for item in load(manifests(task)['holdout'])['items']]
    errors = ([json_error] if json_error else []) + validate_m1(
        parsed,
        task,
        data['mode'],
        data,
        hold_ids,
    )

    optimizer_usage = usage(response)
    optimizer_cost = safe_cost(OPTIMIZER_MODEL, {**optimizer_usage, 'calls': 1})

    envelope = {
        **parsed,
        '_trace_validation': {
            'valid': not errors,
            'errors': errors,
            'input_sha256': sha(input_path),
            'raw_output_sha256': sha(raw_path),
            'm1_sha256': sha(M1_PATH),
            'optimizer_model': OPTIMIZER_MODEL,
            'usage': optimizer_usage,
            'cost': optimizer_cost,
            'wall_time_seconds': wall,
        },
    }
    write_json(parsed_path, envelope)

    if errors:
        raise RuntimeError(f'invalid M1 output {errors}')

    adapter = None
    adapter_path = None
    if parsed['decision'] == 'ADAPTER':
        adapter = parsed['adapter_text']
        adapter_path = adapter_dir(task) / f'adapter_v{roundn}.txt'
        write_new(adapter_path, adapter.encode())

    return envelope, adapter, adapter_path


# -----------------------------------------------------------------------------
# Candidate selection / protocol
# -----------------------------------------------------------------------------
def select_candidate(
    task: str,
    basea: dict[str, Any],
    baseb: dict[str, Any],
    candidates: list[tuple[str, dict[str, Any], dict[str, Any], Path]],
):
    path = adapter_dir(task) / 'selection.json'
    if path.exists():
        return load(path)

    expected_a = basea['ordered_ids']
    expected_b = baseb['ordered_ids']
    records = []
    eligible = []

    base_formats: dict[str, list[dict[str, Any]]] = {}
    for row in basea['items'] + baseb['items']:
        base_formats.setdefault(row['answer_format'], []).append(row)
    base_format_scores = {key: fixed(value) for key, value in base_formats.items()}

    for label, fold_a, fold_b, adapter_path in candidates:
        reasons = []

        if fold_a['ordered_ids'] != expected_a or fold_b['ordered_ids'] != expected_b:
            reasons.append('ID mismatch')
        if fold_a['parse_failures'] + fold_b['parse_failures'] > 0:
            reasons.append('Adapter parse failure')
        if fold_a['fixed_cardinality_score'] < basea['fixed_cardinality_score']:
            reasons.append('fold A regression')
        if fold_b['fixed_cardinality_score'] < baseb['fixed_cardinality_score']:
            reasons.append('fold B regression')
        if not (
            fold_a['fixed_cardinality_score'] > basea['fixed_cardinality_score']
            or fold_b['fixed_cardinality_score'] > baseb['fixed_cardinality_score']
        ):
            reasons.append('no strict fold gain')

        combined = fold_a['items'] + fold_b['items']
        format_rows: dict[str, list[dict[str, Any]]] = {}
        for row in combined:
            format_rows.setdefault(row['answer_format'], []).append(row)
        format_scores = {key: fixed(value) for key, value in format_rows.items()}

        if any(
            format_scores.get(key, float('-inf')) < baseline
            for key, baseline in base_format_scores.items()
        ):
            reasons.append('critical format regression')

        record = {
            'candidate': label,
            'eligible': not reasons,
            'reasons': reasons,
            'fold_a_score': fold_a['fixed_cardinality_score'],
            'fold_b_score': fold_b['fixed_cardinality_score'],
            'macro_score': (
                fold_a['fixed_cardinality_score'] + fold_b['fixed_cardinality_score']
            )
            / 2,
            'minimum_fold_score': min(
                fold_a['fixed_cardinality_score'],
                fold_b['fixed_cardinality_score'],
            ),
            'worst_subgroup_score': min(format_scores.values()),
            'parse_failures': fold_a['parse_failures'] + fold_b['parse_failures'],
            'adapter_sha256': sha(adapter_path) if adapter_path else None,
            'adapter_bytes': adapter_path.stat().st_size if adapter_path else 0,
        }
        records.append(record)
        if not reasons:
            eligible.append(record)

    if eligible:
        order = {'adapter_v1': 1, 'adapter_v2': 2}
        chosen = sorted(
            eligible,
            key=lambda row: (
                -row['macro_score'],
                -row['minimum_fold_score'],
                -row['worst_subgroup_score'],
                row['adapter_bytes'],
                order[row['candidate']],
            ),
        )[0]
        decision = 'ADAPTER'
        selected = chosen['candidate']
        source = adapter_dir(task) / f'{selected}.txt'
        destination = adapter_dir(task) / 'selected_adapter.txt'
        write_new(destination, source.read_bytes())
        selected_hash = sha(destination)
    else:
        decision = 'NO_ADAPTER'
        selected = 'baseline'
        selected_hash = None

    payload = {
        'arm': ARM,
        'task_name': task,
        'decision': decision,
        'selected_candidate': selected,
        'selected_adapter_sha256': selected_hash,
        'm1_sha256': M1_SHA,
        'model_roles': {
            'agent_model': AGENT_MODEL,
            'surrogate_model': SURROGATE_MODEL,
            'optimizer_model': OPTIMIZER_MODEL,
        },
        'candidates': records,
    }
    write_json(path, payload)
    return payload


def development(client: OpenAI, task: str):
    ms = manifests(task)

    baseline_a = evaluate(
        client,
        task,
        ms['fold_a'],
        'development',
        'baseline_fold_a',
        None,
    )
    baseline_b = evaluate(
        client,
        task,
        ms['fold_b'],
        'development',
        'baseline_fold_b',
        None,
    )

    out1, adapter1, path1 = call_m1(
        client,
        task,
        1,
        m1_input(task, 'generate', baseline_a, baseline_b, None, None, None),
    )

    candidates = []
    round1_a = None
    round1_b = None

    if adapter1:
        round1_a = evaluate(
            client,
            task,
            ms['fold_a'],
            'development',
            'adapter_v1_fold_a',
            adapter1,
        )
        round1_b = evaluate(
            client,
            task,
            ms['fold_b'],
            'development',
            'adapter_v1_fold_b',
            adapter1,
        )
        candidates.append(('adapter_v1', round1_a, round1_b, path1))

    previous = (
        {'sha256': sha(path1), 'text': adapter1}
        if path1
        else {'sha256': None, 'text': ''}
    )

    out2, adapter2, path2 = call_m1(
        client,
        task,
        2,
        m1_input(
            task,
            'refine',
            baseline_a,
            baseline_b,
            round1_a,
            round1_b,
            previous,
        ),
    )

    round2_a = None
    round2_b = None
    if adapter2:
        round2_a = evaluate(
            client,
            task,
            ms['fold_a'],
            'development',
            'adapter_v2_fold_a',
            adapter2,
        )
        round2_b = evaluate(
            client,
            task,
            ms['fold_b'],
            'development',
            'adapter_v2_fold_b',
            adapter2,
        )
        candidates.append(('adapter_v2', round2_a, round2_b, path2))

    selection = select_candidate(task, baseline_a, baseline_b, candidates)

    summary = {
        'arm': ARM,
        'task_name': task,
        'status': 'DEVELOPMENT_COMPLETE',
        'model_roles': {
            'agent_model': AGENT_MODEL,
            'surrogate_model': SURROGATE_MODEL,
            'optimizer_model': OPTIMIZER_MODEL,
        },
        'baseline_fold_a': compact(baseline_a),
        'baseline_fold_b': compact(baseline_b),
        'adapter_v1_decision': out1['decision'],
        'adapter_v1_sha256': sha(path1) if path1 else None,
        'adapter_v1_fold_a': compact(round1_a) if round1_a else None,
        'adapter_v1_fold_b': compact(round1_b) if round1_b else None,
        'adapter_v2_decision': out2['decision'],
        'adapter_v2_sha256': sha(path2) if path2 else None,
        'adapter_v2_fold_a': compact(round2_a) if round2_a else None,
        'adapter_v2_fold_b': compact(round2_b) if round2_b else None,
        'selection': selection,
    }
    write_json(RESULT_DIR / 'development' / f'{task}_summary.json', summary)
    return summary


def holdout(client: OpenAI, task: str):
    """Frozen evaluation only. No optimizer/verifier revision is called here."""
    ms = manifests(task)
    selection = load(adapter_dir(task) / 'selection.json')

    baseline = evaluate(
        client,
        task,
        ms['holdout'],
        'holdout',
        'baseline',
        None,
    )

    selected_result = None
    if selection['decision'] == 'ADAPTER':
        adapter = (adapter_dir(task) / 'selected_adapter.txt').read_text(
            encoding='utf-8'
        )
        selected_result = evaluate(
            client,
            task,
            ms['holdout'],
            'holdout',
            'selected_adapter',
            adapter,
        )

    summary = {
        'arm': ARM,
        'task_name': task,
        'status': 'HOLDOUT_COMPLETE',
        'model_roles': {
            'agent_model': AGENT_MODEL,
            'surrogate_model': SURROGATE_MODEL,
            'optimizer_model': OPTIMIZER_MODEL,
        },
        'selection_sha256': sha(adapter_dir(task) / 'selection.json'),
        'selection_decision': selection['decision'],
        'baseline': compact(baseline),
        'selected_adapter': compact(selected_result) if selected_result else None,
        'selected_label': (
            'selected_adapter'
            if selected_result
            else 'NO_ADAPTER (baseline reused; no duplicate call)'
        ),
        'no_holdout_feedback': True,
        'optimizer_called_on_holdout': False,
        'verifier_revision_on_holdout': False,
        'core_prompt_used': False,
    }
    write_json(RESULT_DIR / 'holdout' / f'{task}_summary.json', summary)
    return summary


# -----------------------------------------------------------------------------
# Preflight / CLI
# -----------------------------------------------------------------------------
def preflight() -> dict[str, Any]:
    if not SURROGATE_PATH.exists():
        raise SystemExit(f'missing surrogate prompt: {SURROGATE_PATH}')
    if not M1_PATH.exists():
        raise SystemExit(f'missing optimizer prompt: {M1_PATH}')
    if sha(M1_PATH) != M1_SHA:
        raise SystemExit('M1 hash mismatch')

    for path in manifests('m1_factorybench_l123').values():
        if not path.exists():
            raise SystemExit(f'missing manifest: {path}')

    return {
        'arm': ARM,
        'agent_model': AGENT_MODEL,
        'surrogate_model': SURROGATE_MODEL,
        'optimizer_model': OPTIMIZER_MODEL,
        'same_api_key_env': 'OPENAI_API_KEY',
        'surrogate_prompt': str(SURROGATE_PATH),
        'surrogate_prompt_sha256': sha(SURROGATE_PATH),
        'm1_prompt': str(M1_PATH),
        'm1_sha256': sha(M1_PATH),
        'result_dir': str(RESULT_DIR),
        'trace_dir': str(TRACE_DIR),
        'adapter_dir': str(ADAPTER_ROOT),
        'gt_visible_to_surrogate': False,
        'surrogate_static': True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        choices=['m1_factorybench_l123'],
        default='m1_factorybench_l123',
    )
    parser.add_argument(
        '--phase',
        choices=['preflight', 'development', 'holdout'],
        required=True,
    )
    args = parser.parse_args()

    info = preflight()

    if args.phase == 'preflight':
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return

    if not os.getenv('OPENAI_API_KEY'):
        raise SystemExit('OPENAI_API_KEY missing in model process')

    client = OpenAI()
    if args.phase == 'development':
        result = development(client, args.task)
    else:
        result = holdout(client, args.task)

    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
