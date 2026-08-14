#!/usr/bin/env python3
"""Build the M0 contamination registry and deterministic task manifests."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from factorybench.data import load_split

REPO = Path('/home/training/automatic_prompt_engineer')
ROOT = REPO / 'hermes_factorybench_skill_experiment'
APO = REPO / 'apo_experiment'
REVISION = 'b3863519ccedbceab54dfa7600104eb42b985ed7'
CORE_SHA = '463692cd0a201d916c3f0e39d10cda4c50d2a9a1ca3305cd9de804c295e482b3'
M0_SHA = 'e4bef66552518e4a1206aaa3b14b9d34335c046499291e70dfdc33a79f243257'
UUID_RE = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I)
CAUSAL_RE = re.compile(r'\bcausal_judgment_\d{4}\b')
TEXT_SUFFIXES = {'.json', '.md', '.log', '.txt'}
MANIFEST_DIR = ROOT / 'data_manifests' / 'meta_m0'
REGISTRY_PATH = MANIFEST_DIR / 'contamination_registry.json'


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_new_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    if path.exists():
        if path.read_bytes() != encoded:
            raise SystemExit(f'refusing to overwrite different artifact: {path}')
        print(f'unchanged={path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    print(f'created={path} sha256={sha_bytes(encoded)} bytes={len(encoded)}')


def scan_files() -> list[Path]:
    roots = [
        ROOT / 'optimizer_inputs', ROOT / 'results', ROOT / 'data_manifests',
        ROOT / 'reports', ROOT / 'logs', ROOT / 'traces',
        APO / 'results', APO / 'prompts', APO / 'logs',
        APO / 'factorybench_experiment' / 'results',
        APO / 'factorybench_experiment' / 'logs',
        APO / 'factorybench_experiment' / 'data',
    ]
    paths: set[Path] = set()
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob('*'):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if MANIFEST_DIR in path.parents or (ROOT / 'results' / 'meta_m0') in path.parents:
                continue
            paths.add(path)
    return sorted(paths)


def build_registry() -> dict[str, Any]:
    sources: dict[str, set[str]] = defaultdict(set)
    scanned = []
    for path in scan_files():
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            continue
        rel = str(path.relative_to(REPO))
        ids = sorted(set(UUID_RE.findall(text)) | set(CAUSAL_RE.findall(text)))
        if ids:
            scanned.append({'path': rel, 'sha256': sha_file(path), 'matched_id_count': len(ids)})
        for item_id in ids:
            sources[item_id.lower() if UUID_RE.fullmatch(item_id) else item_id].add(rel)

    fb = {item_id: sorted(paths) for item_id, paths in sources.items() if UUID_RE.fullmatch(item_id)}
    causal = {item_id: sorted(paths) for item_id, paths in sources.items() if CAUSAL_RE.fullmatch(item_id)}
    registry = {
        'protocol': 'meta_m0_cross_task_contamination_registry',
        'scan_scope': [
            'existing Hermes FactoryBench optimizer inputs, results, manifests, traces, reports, and logs',
            'existing APO results, prompts, reports/logs, and FactoryBench L4 prepared subsets/results',
        ],
        'scan_rule': 'Extract FactoryBench UUIDs and causal_judgment_NNNN IDs from existing text/JSON artifacts; every matching source artifact is retained.',
        'excluded_id_counts': {'factorybench': len(fb), 'causal_judgment': len(causal), 'total': len(fb) + len(causal)},
        'excluded_ids': {'factorybench': fb, 'causal_judgment': causal},
        'scanned_artifacts_with_matches': scanned,
    }
    write_new_json(REGISTRY_PATH, registry)
    return registry


def sort_hash(seed: str, item_id: str) -> str:
    return hashlib.sha256((seed + item_id).encode('utf-8')).hexdigest()


def fb_row(item: Any, split: str, seed: str, rank: int) -> dict[str, Any]:
    return {
        'id': item.id,
        'level': item.level,
        'split': split,
        'dataset': item.dataset,
        'episode': item.provenance.get('episode'),
        'answer_format': item.answer_format.value,
        'template_id': item.template_id,
        'template_type': item.template_type,
        'selection_sha256': sort_hash(seed, item.id),
        'selection_rank_within_stratum': rank,
    }


def all_pinned_items() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in (1, 2, 3, 4):
        for split in ('validation', 'test'):
            try:
                items = load_split(level, split=split, revision=REVISION, max_items=None)
            except Exception:
                continue
            for item in items:
                result[item.id.lower()] = item
    return result


def contaminated_episodes(registry: dict[str, Any], pinned: dict[str, Any]) -> set[str]:
    out = set()
    for item_id in registry['excluded_ids']['factorybench']:
        item = pinned.get(item_id.lower())
        if item is not None and item.provenance.get('episode'):
            out.add(str(item.provenance['episode']))
    return out


def eligible_fb(items: list[Any], excluded_ids: set[str], excluded_episodes: set[str]) -> list[Any]:
    return [
        item for item in items
        if item.id.lower() not in excluded_ids
        and item.provenance.get('episode')
        and str(item.provenance.get('episode')) not in excluded_episodes
    ]


def choose(items: list[Any], seed: str, count: int) -> list[Any]:
    ranked = sorted(items, key=lambda item: (sort_hash(seed, item.id), item.id))
    if len(ranked) < count:
        raise ValueError(f'need {count} eligible items, found {len(ranked)}')
    return ranked[:count]


def build_l123(registry: dict[str, Any], pinned: dict[str, Any], used_episodes: set[str]) -> dict[str, Any]:
    excluded_ids = set(registry['excluded_ids']['factorybench'])
    dev_seed = 'meta-m0-factorybench-l123-dev'
    hold_seed = 'meta-m0-factorybench-l123-holdout'
    pools = {
        1: ('test', load_split(1, split='test', revision=REVISION, max_items=None)),
        2: ('validation', load_split(2, split='validation', revision=REVISION, max_items=None)),
        3: ('validation', load_split(3, split='validation', revision=REVISION, max_items=None)),
    }
    dev_items = []
    dev_rows = []
    dynamic_episodes = set(used_episodes)
    availability = {}
    for level in (1, 2, 3):
        split, source = pools[level]
        pool = eligible_fb(source, excluded_ids, dynamic_episodes)
        availability[f'L{level}_{split}_eligible_before_dev'] = len(pool)
        item = choose(pool, dev_seed, 1)[0]
        dev_items.append(item)
        dev_rows.append(fb_row(item, split, dev_seed, 1))
        dynamic_episodes.add(str(item.provenance['episode']))

    hold_rows = []
    for level in (1, 2, 3):
        split, source = pools[level]
        pool = eligible_fb(source, excluded_ids | {x.id.lower() for x in dev_items}, dynamic_episodes)
        availability[f'L{level}_{split}_eligible_before_holdout'] = len(pool)
        selected = choose(pool, hold_seed, 2)
        hold_rows.extend(fb_row(item, split, hold_seed, rank) for rank, item in enumerate(selected, 1))
        dynamic_episodes.update(str(item.provenance['episode']) for item in selected)

    return {
        'task_name': 'factorybench_l123', 'status': 'AVAILABLE', 'revision': REVISION,
        'selection_rule': 'Within each level stratum, SHA-256(UTF-8(seed + item_id)) ascending with item_id tie-break.',
        'seeds': {'development': dev_seed, 'holdout': hold_seed},
        'split_protocol': {'L1': 'test_episode_disjoint', 'L2': 'validation', 'L3': 'validation'},
        'contamination_registry_sha256': sha_file(REGISTRY_PATH),
        'core_skill_sha256': CORE_SHA, 'meta_prompt_sha256': M0_SHA,
        'availability': availability,
        'development': dev_rows, 'holdout': hold_rows,
        'counts': {'development': len(dev_rows), 'holdout': len(hold_rows)},
        'id_disjoint': not ({x['id'] for x in dev_rows} & {x['id'] for x in hold_rows}),
        'episode_disjoint': not ({x['episode'] for x in dev_rows} & {x['episode'] for x in hold_rows}),
    }


def build_l4(registry: dict[str, Any], used_episodes: set[str]) -> dict[str, Any]:
    excluded_ids = set(registry['excluded_ids']['factorybench'])
    dev_seed = 'meta-m0-factorybench-l4-dev'
    hold_seed = 'meta-m0-factorybench-l4-holdout'
    source = load_split(4, split='validation', revision=REVISION, max_items=None)
    pool = eligible_fb(source, excluded_ids, used_episodes)
    dev = choose(pool, dev_seed, 2)
    dev_eps = {str(x.provenance['episode']) for x in dev}
    hold_pool = eligible_fb(source, excluded_ids | {x.id.lower() for x in dev}, used_episodes | dev_eps)
    hold = choose(hold_pool, hold_seed, 3)
    dev_rows = [fb_row(item, 'validation', dev_seed, rank) for rank, item in enumerate(dev, 1)]
    hold_rows = [fb_row(item, 'validation', hold_seed, rank) for rank, item in enumerate(hold, 1)]
    return {
        'task_name': 'factorybench_l4', 'status': 'AVAILABLE', 'revision': REVISION,
        'selection_rule': 'SHA-256(UTF-8(seed + item_id)) ascending with item_id tie-break after ID and episode exclusions.',
        'seeds': {'development': dev_seed, 'holdout': hold_seed}, 'split_protocol': {'L4': 'validation'},
        'contamination_registry_sha256': sha_file(REGISTRY_PATH),
        'core_skill_sha256': CORE_SHA, 'meta_prompt_sha256': M0_SHA,
        'availability': {'validation_total': len(source), 'eligible_before_dev': len(pool), 'eligible_before_holdout': len(hold_pool)},
        'development': dev_rows, 'holdout': hold_rows,
        'counts': {'development': len(dev_rows), 'holdout': len(hold_rows)},
        'id_disjoint': not ({x['id'] for x in dev_rows} & {x['id'] for x in hold_rows}),
        'episode_disjoint': not ({x['episode'] for x in dev_rows} & {x['episode'] for x in hold_rows}),
        'evaluator': {'deterministic_validator': 'apo_experiment/factorybench_experiment/factorybench_evaluator_smoke_test.py:validate_deterministically', 'judge_pipeline': 'FactoryBench factorybench.judges.JudgePanel', 'judge_models': ['gpt-5.5']},
    }


def causal_examples() -> list[dict[str, Any]]:
    source = REPO / 'data' / 'bigbench-ii' / 'causal_judgment' / 'task.json'
    task = json.loads(source.read_text(encoding='utf-8'))
    rows = []
    for index, example in enumerate(task['examples']):
        rows.append({'id': f'causal_judgment_{index:04d}', 'source_index': index})
    return rows


def build_causal(registry: dict[str, Any]) -> dict[str, Any]:
    excluded = set(registry['excluded_ids']['causal_judgment'])
    source_rows = causal_examples()
    eligible = [row for row in source_rows if row['id'] not in excluded]
    dev_seed = 'meta-m0-causal-judgment-dev'
    hold_seed = 'meta-m0-causal-judgment-holdout'
    dev = sorted(eligible, key=lambda row: (sort_hash(dev_seed, row['id']), row['id']))[:3]
    if len(dev) != 3:
        raise ValueError('insufficient causal development items')
    dev_ids = {row['id'] for row in dev}
    hold_pool = [row for row in eligible if row['id'] not in dev_ids]
    hold = sorted(hold_pool, key=lambda row: (sort_hash(hold_seed, row['id']), row['id']))[:5]
    if len(hold) != 5:
        raise ValueError('insufficient causal holdout items')
    def rows(selected: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
        return [dict(row, selection_sha256=sort_hash(seed, row['id']), selection_rank=rank) for rank, row in enumerate(selected, 1)]
    dev_rows, hold_rows = rows(dev, dev_seed), rows(hold, hold_seed)
    return {
        'task_name': 'causal_judgment', 'status': 'AVAILABLE',
        'source': 'data/bigbench-ii/causal_judgment/task.json',
        'source_sha256': sha_file(REPO / 'data' / 'bigbench-ii' / 'causal_judgment' / 'task.json'),
        'selection_rule': 'SHA-256(UTF-8(seed + item_id)) ascending with item_id tie-break after contamination exclusions.',
        'seeds': {'development': dev_seed, 'holdout': hold_seed},
        'contamination_registry_sha256': sha_file(REGISTRY_PATH),
        'core_skill_sha256': CORE_SHA, 'meta_prompt_sha256': M0_SHA,
        'availability': {'source_total': len(source_rows), 'contaminated_ids': len(excluded), 'eligible_before_dev': len(eligible), 'eligible_before_holdout': len(hold_pool)},
        'development': dev_rows, 'holdout': hold_rows,
        'counts': {'development': len(dev_rows), 'holdout': len(hold_rows)},
        'id_disjoint': not ({x['id'] for x in dev_rows} & {x['id'] for x in hold_rows}),
        'evaluator': {'type': 'deterministic_exact_match', 'allowed_outputs': ['Yes', 'No'], 'case_insensitive': True, 'format_requires_single_label': True},
    }


def main() -> None:
    if REGISTRY_PATH.exists():
        registry = json.loads(REGISTRY_PATH.read_text(encoding='utf-8'))
        print(f'using_existing_frozen_registry={REGISTRY_PATH} sha256={sha_file(REGISTRY_PATH)}')
    else:
        registry = build_registry()
    pinned = all_pinned_items()
    episodes = contaminated_episodes(registry, pinned)
    builders = {
        'factorybench_l123_manifest.json': lambda: build_l123(registry, pinned, episodes),
        'factorybench_l4_manifest.json': lambda: build_l4(registry, episodes),
        'causal_judgment_manifest.json': lambda: build_causal(registry),
    }
    manifests = {}
    for name, builder in builders.items():
        try:
            manifests[name] = builder()
        except ValueError as exc:
            manifests[name] = {
                'task_name': name.removesuffix('_manifest.json'),
                'status': 'INSUFFICIENT_UNSEEN_DATA',
                'reason': str(exc),
                'revision': REVISION if name.startswith('factorybench_') else None,
                'contamination_registry_sha256': sha_file(REGISTRY_PATH),
                'core_skill_sha256': CORE_SHA,
                'meta_prompt_sha256': M0_SHA,
                'development': [],
                'holdout': [],
                'counts': {'development': 0, 'holdout': 0},
            }
    for name, payload in manifests.items():
        write_new_json(MANIFEST_DIR / name, payload)
    summary = {
        'status': 'PREFLIGHT_COMPLETE', 'revision': REVISION,
        'contamination_registry_sha256': sha_file(REGISTRY_PATH),
        'contaminated_factorybench_episode_count': len(episodes),
        'tasks': {
            name.removesuffix('_manifest.json'): {
                'status': payload['status'], 'development_count': payload['counts']['development'],
                'holdout_count': payload['counts']['holdout'],
                'manifest_path': f'data_manifests/meta_m0/{name}',
                'manifest_sha256': sha_file(MANIFEST_DIR / name),
            } for name, payload in manifests.items()
        },
    }
    write_new_json(MANIFEST_DIR / 'preflight_summary.json', summary)


if __name__ == '__main__':
    main()
