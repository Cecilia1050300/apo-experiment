#!/usr/bin/env python3
"""Run FactoryBench on the exact IDs frozen in a Task A manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from factorybench.cost import compute_cost_from_usage
from factorybench.data import load_split
from factorybench.evaluate import (
    _adapter_model_id,
    _run_predict,
    _score_one,
    _summarize_usage,
)
from factorybench.prompt import render_prompt
from factorybench.registry import get_model
from factorybench.result import Result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--expect-no-skill", action="store_true")
    parser.add_argument("--expect-skill-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("preflight_valid"):
        raise SystemExit("manifest preflight_valid is not true")
    if args.model != manifest["model"]:
        raise SystemExit("model does not match manifest")
    if args.concurrency != int(manifest["concurrency"]):
        raise SystemExit("concurrency does not match manifest")

    skill_env = os.environ.get("FACTORYBENCH_SKILL_FILE")
    if args.expect_no_skill:
        if skill_env:
            raise SystemExit("baseline requires FACTORYBENCH_SKILL_FILE to be unset")
        print("skill_environment=unset")
    elif args.expect_skill_sha256:
        if not skill_env:
            raise SystemExit("frozen-skill run requires FACTORYBENCH_SKILL_FILE")
        skill_path = Path(skill_env).expanduser().resolve()
        if not skill_path.is_file() or not skill_path.read_bytes():
            raise SystemExit("frozen skill is missing or empty")
        actual_hash = sha256(skill_path)
        if actual_hash != args.expect_skill_sha256:
            raise SystemExit("frozen skill hash mismatch")
        if actual_hash != manifest["frozen_skill_sha256"]:
            raise SystemExit("frozen skill does not match manifest")
        print(f"skill_environment={skill_path}")
        print(f"skill_sha256={actual_hash}")
    else:
        raise SystemExit("declare --expect-no-skill or --expect-skill-sha256")

    source_items = load_split(
        level=manifest["levels"],
        split=manifest["split"],
        revision=manifest["revision"],
        max_items=None,
    )
    source_by_id = {item.id: item for item in source_items}
    if len(source_by_id) != len(source_items):
        raise SystemExit("source split contains duplicate IDs")

    expected_rows = manifest["items"]
    expected_ids = [row["id"] for row in expected_rows]
    if len(expected_ids) != manifest["expected_total"]:
        raise SystemExit("manifest item count does not match expected_total")
    if len(set(expected_ids)) != len(expected_ids):
        raise SystemExit("manifest contains duplicate IDs")
    missing = [item_id for item_id in expected_ids if item_id not in source_by_id]
    if missing:
        raise SystemExit(f"manifest IDs missing from pinned source: {missing}")

    items = [source_by_id[item_id] for item_id in expected_ids]
    for row, item in zip(expected_rows, items):
        if int(row["level"]) != item.level:
            raise SystemExit(f"level mismatch for {item.id}")
        if row["answer_format"] != item.answer_format.value:
            raise SystemExit(f"answer-format mismatch for {item.id}")
        if row.get("dataset") != item.dataset:
            raise SystemExit(f"dataset mismatch for {item.id}")
        if row.get("episode") != item.provenance.get("episode"):
            raise SystemExit(f"episode mismatch for {item.id}")

    print(f"manifest={args.manifest.resolve()}")
    print(f"manifest_sha256={sha256(args.manifest)}")
    print(f"selected_item_count={len(items)}")
    print("selected_ids=" + ",".join(expected_ids))
    print("source_selection_preflight=passed")
    if args.preflight_only:
        print("model_calls=0")
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --preflight-only is used")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")

    instance = get_model(args.model)
    prompts = [render_prompt(item) for item in items]
    started = time.perf_counter()
    raw_outputs, usages = _run_predict(
        instance,
        prompts,
        progress=True,
        concurrency=args.concurrency,
    )
    wall = timedelta(seconds=time.perf_counter() - started)
    item_results = [_score_one(item, raw) for item, raw in zip(items, raw_outputs)]
    resolved_name = args.model
    model_id = _adapter_model_id(instance, fallback=resolved_name)
    tokens_used = _summarize_usage(
        candidate_model_id=model_id,
        candidate_usages=usages,
        panel=None,
    )
    cost = compute_cost_from_usage(tokens_used) if tokens_used else 0.0
    result = Result(
        model_name=resolved_name,
        items=item_results,
        wall_time=wall,
        cost=cost,
        tokens_used=tokens_used,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)

    score = result.score
    parse_failures = len(result.parse_failures())
    finite = sum(math.isfinite(item.score) for item in result.items)
    print(f"saved={args.output.resolve()}")
    print(f"result_sha256={sha256(args.output)}")
    print(f"reported_score={score}")
    print(f"parse_failures={parse_failures}")
    print(f"finite_item_scores={finite}")
    print(f"wall_time_seconds={result.wall_time.total_seconds()}")
    print(f"cost={result.cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
