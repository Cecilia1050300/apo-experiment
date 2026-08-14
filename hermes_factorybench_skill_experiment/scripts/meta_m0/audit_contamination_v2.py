#!/usr/bin/env python3
"""Audit the original M0 FactoryBench contamination registry and recover L1-L3 data."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from factorybench.data import load_split

REPO = Path('/home/training/automatic_prompt_engineer')
ROOT = REPO / 'hermes_factorybench_skill_experiment'
REV = 'b3863519ccedbceab54dfa7600104eb42b985ed7'
M0_SHA = 'e4bef66552518e4a1206aaa3b14b9d34335c046499291e70dfdc33a79f243257'
CORE_SHA = '463692cd0a201d916c3f0e39d10cda4c50d2a9a1ca3305cd9de804c295e482b3'
ORIGINAL = ROOT / 'data_manifests/meta_m0/contamination_registry.json'
V2 = ROOT / 'data_manifests/meta_m0/contamination_registry_v2.json'
REPORT = ROOT / 'reports/meta_m0/contamination_registry_audit.md'
MANIFEST = ROOT / 'data_manifests/meta_m0/factorybench_l123_manifest_v2.json'
DEV_MANIFEST = ROOT / 'data_manifests/meta_m0/factorybench_l123_development_v2.json'
HOLD_MANIFEST = ROOT / 'data_manifests/meta_m0/factorybench_l123_holdout_v2.json'
QUALIFYING = {'ACTUAL_MODEL_EVALUATION', 'ACTUAL_OPTIMIZER_EXPOSURE', 'ACTUAL_JUDGE_EXPOSURE', 'EXECUTED_MANIFEST'}
UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(seed: str, item_id: str) -> str:
    return hashlib.sha256((seed + item_id).encode('utf-8')).hexdigest()


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'refusing to overwrite different artifact: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, payload: Any) -> None:
    write_new(path, (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def classify(path: str) -> tuple[str, str]:
    name = Path(path).name
    if '/data/' in path and 'factorybench_l4_subset' in name:
        return 'SOURCE_DATASET_ONLY', 'Prepared source/subset file; model and judge execution are established only by separate result artifacts.'
    if path.endswith('data_manifests/l123_all_unseen_generalization.json'):
        return 'INVENTORY_ONLY', 'Role is a broad all-unseen inventory; it lists thousands of available rows and has no associated result or execution log.'
    if path.endswith('data_manifests/l123_expanded_smoke_30.json'):
        return 'UNEXECUTED_PROSPECTIVE_MANIFEST', 'Prospective expanded-smoke manifest; repository search found no result or log executing these selected IDs.'
    if path.endswith('data_manifests/l123_sealed_generalization_30.json'):
        return 'UNEXECUTED_PROSPECTIVE_MANIFEST', 'Prospective sealed-generalization manifest; repository search found no result or log executing these selected IDs.'
    if '/data_manifests/task_a1_' in path or '/data_manifests/task_a2_' in path:
        return 'EXECUTED_MANIFEST', 'Matching Task A result JSONs and exit-status-zero logs establish execution of the manifest IDs.'
    if '/optimizer_inputs/' in path:
        return 'ACTUAL_OPTIMIZER_EXPOSURE', 'Optimizer packet contains development failures/results supplied during v2-v4 adapter refinement.'
    if '_judged' in name or 'optimization_judged' in name:
        return 'ACTUAL_JUDGE_EXPOSURE', 'Judged-result artifact contains judge records or scores for these IDs.'
    if name in {'gpt55_neutral_context_audit.json', 'mini_neutral_context_audit.json', 'contract_audit.json'}:
        return 'INVENTORY_ONLY', 'Deterministic context/contract audit only; no candidate-model or judge call is recorded in this artifact.'
    if name == 'oracle_fixture_corpus.json':
        return 'INVENTORY_ONLY', 'Fixture corpus used for plumbing; it does not record candidate-model, optimizer, or judge exposure.'
    if '/results/' in path and any(token in name for token in (
        'generation', 'control.json', 'optimization_results', 'evaluator_smoke_test',
        'final_test_skill_comparison_gpt-4o-mini.json', 'gpt55_l123_smoke',
        'baseline_gpt55', 'skill_v1_gpt55', 'skill_v2_gpt55', 'skill_v3_gpt55', 'skill_v4_gpt55',
    )):
        return 'ACTUAL_MODEL_EVALUATION', 'Result artifact contains candidate-model outputs, usage, scores, or a completed model-run status.'
    if '/results/' in path and any(token in name for token in ('failed_cases', 'comparison', 'failure_audit_source')):
        return 'EXECUTED_MANIFEST', 'Derivative result contains IDs and outputs/failures tied to a demonstrably executed model run.'
    if '/logs/' in path:
        return 'EXECUTED_MANIFEST', 'Execution log records selected IDs and/or a successful evaluator exit status.'
    if '/reports/' in path or name.endswith('_trace.json'):
        return 'EXECUTED_MANIFEST', 'Audit/report trace records IDs from demonstrably executed runs and optimizer rounds.'
    if '/results/smoke/' in path or '/results/generalization/' in path:
        return 'ACTUAL_MODEL_EVALUATION', 'FactoryBench result artifact contains model outputs or paired results from an executed run.'
    if '/results/' in path:
        return 'EXECUTED_MANIFEST', 'Result-side artifact is linked to a completed evaluation or judged comparison.'
    raise ValueError(f'unclassified FactoryBench contamination source: {path}')


def pinned_sources() -> tuple[dict[str, Any], dict[tuple[int, str], list[Any]]]:
    index = {}
    pools = {}
    for level, split in ((1, 'test'), (2, 'validation'), (3, 'validation'), (1, 'validation'), (2, 'test'), (3, 'test'), (4, 'validation'), (4, 'test')):
        try:
            items = load_split(level, split=split, revision=REV, max_items=None)
        except Exception:
            continue
        pools[(level, split)] = items
        for item in items:
            index[item.id.lower()] = item
    return index, pools


def eligible(items: list[Any], excluded_ids: set[str], excluded_episodes: set[str]) -> list[Any]:
    return [item for item in items if item.id.lower() not in excluded_ids and item.provenance.get('episode') and str(item.provenance['episode']) not in excluded_episodes]


def selected_row(item: Any, split: str, seed: str, rank: int) -> dict[str, Any]:
    return {
        'id': item.id, 'level': item.level, 'split': split, 'dataset': item.dataset,
        'episode': item.provenance.get('episode'), 'answer_format': item.answer_format.value,
        'template_id': item.template_id, 'template_type': item.template_type,
        'selection_sha256': digest(seed, item.id), 'selection_rank_within_level': rank,
    }


def main() -> None:
    for path in (V2, REPORT, MANIFEST, DEV_MANIFEST, HOLD_MANIFEST):
        if path.exists():
            raise RuntimeError(f'target already exists: {path}')
    original_bytes = ORIGINAL.read_bytes()
    original = json.loads(original_bytes)
    source_to_tokens: dict[str, set[str]] = defaultdict(set)
    for token, paths in original['excluded_ids']['factorybench'].items():
        for path in paths:
            source_to_tokens[path].add(token.lower())

    pinned, pools = pinned_sources()
    records = []
    corrected_sources: dict[str, set[str]] = defaultdict(set)
    nonqualifying_sources: dict[str, set[str]] = defaultdict(set)
    for path in sorted(source_to_tokens):
        category, evidence = classify(path)
        tokens = source_to_tokens[path]
        resolved_items = tokens & set(pinned)
        record = {
            'source_artifact': path, 'source_sha256': next((row['sha256'] for row in original['scanned_artifacts_with_matches'] if row['path'] == path), None),
            'classification': category, 'uuid_tokens_contributed': len(tokens),
            'resolved_pinned_item_ids_contributed': len(resolved_items),
            'remain_excluded': category in QUALIFYING, 'execution_evidence': evidence,
        }
        records.append(record)
        destination = corrected_sources if category in QUALIFYING else nonqualifying_sources
        for token in tokens:
            destination[token].add(path)

    availability_sources = [
        ('FactoryBench/FactoryBench@' + REV + ':factorybench_qa/level_1/test.jsonl', len(pools[(1, 'test')])),
        ('FactoryBench/FactoryBench@' + REV + ':factorybench_qa/level_2/validation.jsonl', len(pools[(2, 'validation')])),
        ('FactoryBench/FactoryBench@' + REV + ':factorybench_qa/level_3/validation.jsonl', len(pools[(3, 'validation')])),
        ('hermes_factorybench_skill_experiment/data_manifests/meta_m0/preflight_summary.json', 0),
    ]
    for path, count in availability_sources:
        records.append({
            'source_artifact': path, 'source_sha256': None, 'classification': 'SOURCE_DATASET_ONLY' if 'jsonl' in path else 'INVENTORY_ONLY',
            'uuid_tokens_contributed': 0, 'resolved_pinned_item_ids_contributed': 0,
            'source_dataset_item_count': count, 'remain_excluded': False,
            'execution_evidence': 'Availability/source record only; it contributed no IDs to the original registry and is not evidence of model, optimizer, or judge exposure.',
        })

    original_ids = set(original['excluded_ids']['factorybench'])
    corrected_tokens = set(corrected_sources)
    original_resolved = original_ids & set(pinned)
    corrected_resolved = corrected_tokens & set(pinned)
    original_episodes = {str(pinned[item_id].provenance['episode']) for item_id in original_resolved if pinned[item_id].provenance.get('episode')}
    corrected_episodes = {str(pinned[item_id].provenance['episode']) for item_id in corrected_resolved if pinned[item_id].provenance.get('episode')}

    before_after = {}
    for level, split in ((1, 'test'), (2, 'validation'), (3, 'validation')):
        before_after[f'L{level}_{split}'] = {
            'source_count': len(pools[(level, split)]),
            'eligible_under_original_registry': len(eligible(pools[(level, split)], original_ids, original_episodes)),
            'eligible_under_corrected_registry': len(eligible(pools[(level, split)], corrected_tokens, corrected_episodes)),
        }

    category_summary = {}
    for category in sorted({row['classification'] for row in records}):
        rows = [row for row in records if row['classification'] == category]
        category_summary[category] = {
            'source_count': len(rows), 'uuid_token_mentions': sum(row['uuid_tokens_contributed'] for row in rows),
            'resolved_pinned_item_mentions': sum(row['resolved_pinned_item_ids_contributed'] for row in rows),
            'counts_as_exposure': category in QUALIFYING,
        }

    registry_v2 = {
        'protocol': 'meta_m0_factorybench_contamination_registry_v2',
        'original_registry_path': str(ORIGINAL.relative_to(ROOT)), 'original_registry_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'classification_policy': {category: category in QUALIFYING for category in (
            'ACTUAL_MODEL_EVALUATION', 'ACTUAL_OPTIMIZER_EXPOSURE', 'ACTUAL_JUDGE_EXPOSURE', 'EXECUTED_MANIFEST',
            'INVENTORY_ONLY', 'SOURCE_DATASET_ONLY', 'UNEXECUTED_PROSPECTIVE_MANIFEST')},
        'root_cause': 'The original lexical UUID scan treated every UUID in broad inventories as an exposed item ID and did not distinguish provenance episode UUIDs from item UUIDs.',
        'source_classifications': records, 'category_summary': category_summary,
        'excluded_ids': {token: sorted(paths) for token, paths in sorted(corrected_sources.items())},
        'released_inventory_only_tokens': {token: sorted(paths) for token, paths in sorted(nonqualifying_sources.items()) if token not in corrected_tokens},
        'counts': {
            'original_uuid_tokens': len(original_ids), 'original_tokens_resolving_to_pinned_items': len(original_resolved),
            'corrected_excluded_uuid_tokens': len(corrected_tokens), 'corrected_tokens_resolving_to_pinned_items': len(corrected_resolved),
            'released_nonexposure_uuid_tokens': len(original_ids - corrected_tokens),
            'corrected_exposed_episode_count': len(corrected_episodes),
        },
        'eligible_counts_before_after': before_after,
    }
    write_json(V2, registry_v2)

    dev_seed = 'meta-m0-factorybench-l123-dev'
    hold_seed = 'meta-m0-factorybench-l123-holdout'
    pool_spec = {1: ('test', pools[(1, 'test')]), 2: ('validation', pools[(2, 'validation')]), 3: ('validation', pools[(3, 'validation')])}
    dev_rows = []
    dev_ids = set()
    blocked_episodes = set(corrected_episodes)
    for level in (1, 2, 3):
        split, source = pool_spec[level]
        candidates = sorted(eligible(source, corrected_tokens | dev_ids, blocked_episodes), key=lambda item: (digest(dev_seed, item.id), item.id))
        if not candidates:
            raise RuntimeError(f'no corrected eligible development item for L{level}')
        chosen = candidates[0]
        dev_rows.append(selected_row(chosen, split, dev_seed, 1))
        dev_ids.add(chosen.id.lower())
        blocked_episodes.add(str(chosen.provenance['episode']))

    hold_rows = []
    hold_ids = set()
    for level in (1, 2, 3):
        split, source = pool_spec[level]
        candidates = sorted(eligible(source, corrected_tokens | dev_ids | hold_ids, blocked_episodes), key=lambda item: (digest(hold_seed, item.id), item.id))
        if len(candidates) < 2:
            raise RuntimeError(f'fewer than two corrected eligible holdout items for L{level}')
        chosen = candidates[:2]
        for rank, item in enumerate(chosen, 1):
            hold_rows.append(selected_row(item, split, hold_seed, rank))
            hold_ids.add(item.id.lower())
            blocked_episodes.add(str(item.provenance['episode']))

    common = {
        'revision': REV, 'model': 'gpt-5.5', 'concurrency': 2,
        'meta_prompt_sha256': M0_SHA, 'core_skill_sha256': CORE_SHA,
        'contamination_registry_v2_sha256': sha(V2),
        'selection_rule': 'Within each level, SHA-256(UTF-8(seed + item_id)) ascending with item_id tie-break after corrected actual-exposure and episode exclusions.',
        'split_provenance': {'L1': 'test_episode_disjoint', 'L2': 'validation', 'L3': 'validation'},
        'mixed_split_warning': 'L1 uses test while L2/L3 use validation; no aggregate may be described as a pure validation score.',
    }
    dev_manifest = {'task_name': 'factorybench_l123', 'partition': 'development', **common, 'seed': dev_seed, 'expected_count': 3, 'items': dev_rows}
    hold_manifest = {'task_name': 'factorybench_l123', 'partition': 'holdout', **common, 'seed': hold_seed, 'expected_count': 6, 'items': hold_rows}
    write_json(DEV_MANIFEST, dev_manifest)
    write_json(HOLD_MANIFEST, hold_manifest)
    combined = {
        'task_name': 'factorybench_l123', 'status': 'AVAILABLE_AFTER_CONTAMINATION_AUDIT', **common,
        'development_manifest_path': str(DEV_MANIFEST.relative_to(ROOT)), 'development_manifest_sha256': sha(DEV_MANIFEST),
        'holdout_manifest_path': str(HOLD_MANIFEST.relative_to(ROOT)), 'holdout_manifest_sha256': sha(HOLD_MANIFEST),
        'development': dev_rows, 'holdout': hold_rows,
        'counts': {'development': len(dev_rows), 'holdout': len(hold_rows)},
        'id_disjoint': not ({row['id'] for row in dev_rows} & {row['id'] for row in hold_rows}),
        'episode_disjoint': not ({row['episode'] for row in dev_rows} & {row['episode'] for row in hold_rows}),
    }
    write_json(MANIFEST, combined)

    lines = [
        '# FactoryBench L1–L3 Contamination Registry Audit', '',
        f'- Original registry preserved: `{ORIGINAL.relative_to(ROOT)}`',
        f'- Original SHA-256: `{hashlib.sha256(original_bytes).hexdigest()}`',
        f'- Corrected registry: `{V2.relative_to(ROOT)}`',
        '', '## Root cause', '',
        registry_v2['root_cause'], '',
        'The broad `l123_all_unseen_generalization.json` file was an inventory, not an executed result. Its 7,572 UUID tokens included both prospective item IDs and UUID-valued provenance episodes. Treating all of them as actual exposure eliminated the required L1 test pool and substantially overexcluded the L2/L3 validation pools.', '',
        '## Source classifications', '',
        '| Source artifact | Classification | UUID tokens | Resolved pinned items | Remain excluded | Evidence |',
        '|---|---|---:|---:|---|---|',
    ]
    for row in records:
        evidence = row['execution_evidence'].replace('|', '\\|')
        lines.append(f"| `{row['source_artifact']}` | {row['classification']} | {row['uuid_tokens_contributed']} | {row['resolved_pinned_item_ids_contributed']} | {'yes' if row['remain_excluded'] else 'no'} | {evidence} |")
    lines += ['', '## Eligible counts before and after correction', '', '| Pool | Source | Original eligible | Corrected eligible |', '|---|---:|---:|---:|']
    for pool, row in before_after.items():
        lines.append(f"| {pool} | {row['source_count']} | {row['eligible_under_original_registry']} | {row['eligible_under_corrected_registry']} |")
    lines += [
        '', '## Recovery decision', '',
        f"Recovered {len(dev_rows)} development items and {len(hold_rows)} holdout items with exact ID and episode disjointness.",
        'L1 uses the pinned test split because L1 validation is unavailable. L2 and L3 use validation. The mixed-split aggregate is not a pure validation score.', '',
    ]
    write_new(REPORT, '\n'.join(lines).encode('utf-8'))
    print(json.dumps({
        'status': 'RECOVERED', 'registry_v2_sha256': sha(V2), 'audit_report_sha256': sha(REPORT),
        'development_manifest_sha256': sha(DEV_MANIFEST), 'holdout_manifest_sha256': sha(HOLD_MANIFEST),
        'combined_manifest_sha256': sha(MANIFEST), 'counts': registry_v2['counts'], 'eligible': before_after,
    }, indent=2))


if __name__ == '__main__':
    main()
