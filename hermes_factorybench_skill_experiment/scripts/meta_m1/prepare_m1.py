#!/usr/bin/env python3
"""Build M1 semantic contamination registry and frozen task manifests."""
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
REV = 'b3863519ccedbceab54dfa7600104eb42b985ed7'
M1_PATH = ROOT / 'prompts/meta/manufacturing_meta_prompt_m1_candidate.txt'
M1_SHA = '78187e3268294657d2398c9a79563a36f050c4189b2f6650cc569407512cb052'
BASE = ROOT / 'data_manifests/meta_m0/contamination_registry_v2.json'
OUT_DIR = ROOT / 'data_manifests/meta_m1'
REGISTRY = OUT_DIR / 'contamination_registry_m1.json'
AUDIT = ROOT / 'reports/meta_m1/contamination_registry_m1_audit.md'
UUID_RE = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.I)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(seed: str, item_id: str) -> str:
    return hashlib.sha256((seed + item_id).encode()).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing to overwrite: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, payload: Any) -> None:
    write_new(path, (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


def all_items() -> tuple[dict[str, Any], dict[tuple[int, str], list[Any]]]:
    index, pools = {}, {}
    for level in (1, 2, 3, 4):
        for split in ('validation', 'test'):
            try:
                items = load_split(level, split=split, revision=REV, max_items=None)
            except Exception:
                continue
            pools[(level, split)] = items
            index.update({item.id.lower(): item for item in items})
    return index, pools


def item_row(item: Any, split: str, seed: str, rank: int) -> dict[str, Any]:
    return {'id': item.id, 'level': item.level, 'split': split, 'dataset': item.dataset, 'episode': item.provenance.get('episode'), 'answer_format': item.answer_format.value, 'template_id': item.template_id, 'template_type': item.template_type, 'selection_sha256': digest(seed, item.id), 'selection_rank': rank}


def select(items: list[Any], excluded_ids: set[str], excluded_eps: set[str], seed: str, count: int, extra_ids: set[str], extra_eps: set[str]) -> list[Any]:
    eligible = [item for item in items if item.id.lower() not in excluded_ids | extra_ids and item.provenance.get('episode') and str(item.provenance['episode']) not in excluded_eps | extra_eps]
    ranked = sorted(eligible, key=lambda item: (digest(seed, item.id), item.id))
    if len(ranked) < count:
        raise RuntimeError(f'insufficient unseen data: need {count}, have {len(ranked)}')
    return ranked[:count]


def main() -> None:
    targets = [REGISTRY, AUDIT] + [OUT_DIR / name for name in (
        'factorybench_l123_dev_fold_a.json', 'factorybench_l123_dev_fold_b.json', 'factorybench_l123_holdout.json', 'factorybench_l123_combined_manifest.json',
        'factorybench_l4_dev_fold_a.json', 'factorybench_l4_dev_fold_b.json', 'factorybench_l4_holdout.json', 'factorybench_l4_combined_manifest.json')]
    if any(path.exists() for path in targets):
        raise RuntimeError('one or more M1 preflight targets already exist')
    if not M1_PATH.read_bytes() or sha(M1_PATH) != M1_SHA:
        raise RuntimeError('M1 identity mismatch')
    base = json.loads(BASE.read_text())
    index, pools = all_items()
    exposure_sources: dict[str, set[str]] = defaultdict(set)
    for token, sources in base['excluded_ids'].items():
        if token.lower() in index:
            for source in sources:
                exposure_sources[token.lower()].add(source)

    new_sources = []
    for area in ('development', 'holdout'):
        for path in sorted((ROOT / 'results/meta_m0' / area).glob('*.json')):
            data = path.read_text(encoding='utf-8')
            ids = sorted({token.lower() for token in UUID_RE.findall(data) if token.lower() in index})
            if not ids:
                continue
            rel = str(path.relative_to(ROOT))
            category = 'ACTUAL_MODEL_EVALUATION'
            new_sources.append({'source_artifact': rel, 'source_sha256': sha(path), 'classification': category, 'item_id_count': len(ids), 'remain_excluded': True, 'evidence': 'Frozen M0 result contains executed model outputs/scores for these item IDs.'})
            for item_id in ids:
                exposure_sources[item_id].add(rel)
    for path in sorted((ROOT / 'traces/meta_m0').glob('*/m0_round_*_input*.json')):
        data = path.read_text(encoding='utf-8')
        ids = sorted({token.lower() for token in UUID_RE.findall(data) if token.lower() in index})
        if not ids:
            continue
        rel = str(path.relative_to(ROOT))
        new_sources.append({'source_artifact': rel, 'source_sha256': sha(path), 'classification': 'ACTUAL_OPTIMIZER_EXPOSURE', 'item_id_count': len(ids), 'remain_excluded': True, 'evidence': 'Exact M0 optimization input exposed development item content/results.'})
        for item_id in ids:
            exposure_sources[item_id].add(rel)

    exposed_ids = set(exposure_sources)
    exposed_eps = {str(index[item_id].provenance['episode']) for item_id in exposed_ids if index[item_id].provenance.get('episode')}
    registry = {
        'protocol': 'manufacturing_meta_prompt_m1_semantic_contamination', 'revision': REV,
        'm1_sha256': M1_SHA, 'base_registry_path': str(BASE.relative_to(ROOT)), 'base_registry_sha256': sha(BASE),
        'semantic_rule': 'Only actual model, optimizer, judge, or executed-manifest exposure counts. Inventory/source/prospective UUIDs and episode UUIDs are not item contamination.',
        'excluded_item_ids': {item_id: sorted(sources) for item_id, sources in sorted(exposure_sources.items())},
        'exposed_episodes': sorted(exposed_eps),
        'counts': {'excluded_item_ids': len(exposed_ids), 'exposed_episodes': len(exposed_eps), 'm0_result_or_optimizer_sources_added': len(new_sources)},
        'm0_exposure_sources': new_sources,
    }
    write_json(REGISTRY, registry)

    used_ids, used_eps = set(), set()
    common = {'revision': REV, 'model': 'gpt-5.5', 'concurrency': 2, 'm1_sha256': M1_SHA, 'contamination_registry_m1_sha256': sha(REGISTRY), 'selection_rule': 'SHA-256(UTF-8(seed + item_id)) ascending with item_id tie-break after item and episode exclusions.'}
    l123_specs = [
        ('factorybench_l123_dev_fold_a.json', 'development_fold_a', 'm1-factorybench-l123-dev-fold-a', 1),
        ('factorybench_l123_dev_fold_b.json', 'development_fold_b', 'm1-factorybench-l123-dev-fold-b', 1),
        ('factorybench_l123_holdout.json', 'holdout', 'm1-factorybench-l123-holdout', 3),
    ]
    l123_manifests = []
    for filename, partition, seed, per_level in l123_specs:
        rows = []
        for level in (1, 2, 3):
            split = 'test' if level == 1 else 'validation'
            chosen = select(pools[(level, split)], exposed_ids, exposed_eps, seed, per_level, used_ids, used_eps)
            rows.extend(item_row(item, split, seed, rank) for rank, item in enumerate(chosen, 1))
            used_ids.update(item.id.lower() for item in chosen)
            used_eps.update(str(item.provenance['episode']) for item in chosen)
        payload = {'task_name': 'm1_factorybench_l123', 'partition': partition, **common, 'seed': seed, 'expected_count': len(rows), 'split_provenance': {'L1': 'test_episode_disjoint', 'L2': 'validation', 'L3': 'validation'}, 'mixed_split_warning': 'L1 test plus L2/L3 validation; not a pure validation score.', 'items': rows}
        path = OUT_DIR / filename
        write_json(path, payload)
        l123_manifests.append((path, payload))
    combined_l123 = {'task_name': 'm1_factorybench_l123', **common, 'manifests': {payload['partition']: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'count': payload['expected_count']} for path, payload in l123_manifests}, 'all_ids_disjoint': len(used_ids) == 15, 'all_episodes_disjoint': len(used_eps) == 15}
    write_json(OUT_DIR / 'factorybench_l123_combined_manifest.json', combined_l123)

    l4_used_ids, l4_used_eps = set(), set()
    l4_pool = [item for item in pools[(4, 'validation')] if item.answer_format.value == 'free_form']
    l4_specs = [
        ('factorybench_l4_dev_fold_a.json', 'development_fold_a', 'm1-factorybench-l4-dev-fold-a', 2),
        ('factorybench_l4_dev_fold_b.json', 'development_fold_b', 'm1-factorybench-l4-dev-fold-b', 2),
        ('factorybench_l4_holdout.json', 'holdout', 'm1-factorybench-l4-holdout', 6),
    ]
    l4_manifests = []
    for filename, partition, seed, count in l4_specs:
        chosen = select(l4_pool, exposed_ids, exposed_eps, seed, count, l4_used_ids, l4_used_eps)
        rows = [item_row(item, 'validation', seed, rank) for rank, item in enumerate(chosen, 1)]
        l4_used_ids.update(item.id.lower() for item in chosen)
        l4_used_eps.update(str(item.provenance['episode']) for item in chosen)
        payload = {'task_name': 'm1_factorybench_l4', 'partition': partition, **common, 'seed': seed, 'expected_count': len(rows), 'split': 'validation', 'answer_format_filter': 'free_form', 'output_contract': {'type': 'factorybench_diagnostic_json_v2', 'required_keys': ['root_cause', 'evidence', 'corrective_actions']}, 'evaluator': {'deterministic_validator': 'factorybench_evaluator_smoke_test.validate_deterministically', 'judge_pipeline': 'factorybench_evaluator_smoke_test.call_judge', 'judge_model': 'gpt-5.5', 'judge_count': 1}, 'items': rows}
        path = OUT_DIR / filename
        write_json(path, payload)
        l4_manifests.append((path, payload))
    combined_l4 = {'task_name': 'm1_factorybench_l4', **common, 'manifests': {payload['partition']: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'count': payload['expected_count']} for path, payload in l4_manifests}, 'all_ids_disjoint': len(l4_used_ids) == 10, 'all_episodes_disjoint': len(l4_used_eps) == 10, 'judge_model': 'gpt-5.5', 'judge_count': 1}
    write_json(OUT_DIR / 'factorybench_l4_combined_manifest.json', combined_l4)

    availability = {
        'L1_test': len([x for x in pools[(1, 'test')] if x.id.lower() not in exposed_ids and x.provenance.get('episode') and str(x.provenance['episode']) not in exposed_eps]),
        'L2_validation': len([x for x in pools[(2, 'validation')] if x.id.lower() not in exposed_ids and x.provenance.get('episode') and str(x.provenance['episode']) not in exposed_eps]),
        'L3_validation': len([x for x in pools[(3, 'validation')] if x.id.lower() not in exposed_ids and x.provenance.get('episode') and str(x.provenance['episode']) not in exposed_eps]),
        'L4_validation_free_form': len([x for x in l4_pool if x.id.lower() not in exposed_ids and x.provenance.get('episode') and str(x.provenance['episode']) not in exposed_eps]),
    }
    lines = ['# M1 Contamination Registry Audit', '', f'- M1 SHA-256: `{M1_SHA}`', f'- Base semantic registry: `{sha(BASE)}`', f'- M1 registry: `{sha(REGISTRY)}`', '', 'Only categories ACTUAL_MODEL_EVALUATION, ACTUAL_OPTIMIZER_EXPOSURE, ACTUAL_JUDGE_EXPOSURE, and EXECUTED_MANIFEST remain excluded. Item IDs and episode UUIDs are stored separately.', '', '## Counts', '', f"- Excluded item IDs: {len(exposed_ids)}", f"- Exposed episodes: {len(exposed_eps)}", f"- New M0 exposure sources incorporated: {len(new_sources)}", '', '## Eligible source counts', '']
    for key, value in availability.items():
        lines.append(f'- {key}: {value}')
    lines += ['', 'All M1 selected IDs and episodes are disjoint across development folds and holdout. L4 selection is restricted to free-form items so the existing deterministic diagnostic JSON validator and semantic judge contract remain applicable.', '']
    write_new(AUDIT, '\n'.join(lines).encode())
    print(json.dumps({'status': 'FROZEN', 'm1_sha256': M1_SHA, 'registry_sha256': sha(REGISTRY), 'availability': availability, 'l123_combined_sha256': sha(OUT_DIR / 'factorybench_l123_combined_manifest.json'), 'l4_combined_sha256': sha(OUT_DIR / 'factorybench_l4_combined_manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
