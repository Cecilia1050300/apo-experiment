#!/usr/bin/env python3
"""Build paired Task A comparison artifacts from immutable result JSONs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

ROOT = Path("/home/training/automatic_prompt_engineer/hermes_factorybench_skill_experiment")
SKILL_HASH = "cc83006d7a2e24aaf86ee7de16900cad71cd668563fb0ac2d8ef62b03ccb527e"

CONFIGS = {
    "a1": {
        "name": "Task A1 — Split-Disjoint L2/L3 Validation",
        "manifest": "data_manifests/task_a1_l23_validation7_manifest.json",
        "baseline": "results/generalization/task_a1_l23_validation7_baseline_gpt55.json",
        "v3": "results/generalization/task_a1_l23_validation7_skill_v3_gpt55.json",
        "baseline_log": "logs/task_a1_l23_validation7_baseline_gpt55.log",
        "v3_log": "logs/task_a1_l23_validation7_skill_v3_gpt55.log",
        "expected": 14,
    },
    "a2": {
        "name": "Task A2 — L1 Episode-Disjoint Holdout",
        "manifest": "data_manifests/task_a2_l1_episode_disjoint7_manifest.json",
        "baseline": "results/generalization/task_a2_l1_episode_disjoint7_baseline_gpt55.json",
        "v3": "results/generalization/task_a2_l1_episode_disjoint7_skill_v3_gpt55.json",
        "baseline_log": "logs/task_a2_l1_episode_disjoint7_baseline_gpt55.log",
        "v3_log": "logs/task_a2_l1_episode_disjoint7_skill_v3_gpt55.log",
        "expected": 7,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_clean(row: dict[str, Any]) -> bool:
    score = row.get("score")
    return (
        row.get("parse_error") is None
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
    )


def fixed_score(rows: list[dict[str, Any]]) -> float:
    scores = [float(row["score"]) if finite_clean(row) else 0.0 for row in rows]
    chances = [float(row.get("chance", 0.0)) for row in rows]
    mean_score = sum(scores) / len(rows)
    mean_chance = sum(chances) / len(rows)
    return (mean_score - mean_chance) / (1.0 - mean_chance)


def grouped(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, float]:
    values = sorted({key(row) for row in rows})
    return {value: fixed_score([row for row in rows if key(row) == value]) for value in values}


def deltas(baseline: dict[str, float], v3: dict[str, float]) -> dict[str, dict[str, float]]:
    return {
        key: {"baseline": baseline[key], "v3": v3[key], "delta": v3[key] - baseline[key]}
        for key in sorted(set(baseline) | set(v3))
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    return value


def determine_verdict(task: str, valid: bool, delta: float, level_deltas: dict[str, dict[str, float]]) -> str:
    if not valid:
        return "INVALID"
    if task == "a1":
        any_regression = any(row["delta"] < 0 for row in level_deltas.values())
        if delta >= 0.05 and not any_regression:
            return "PASS_STRONG"
        if delta > 0 and not any_regression:
            return "PASS_WEAK"
        if delta > 0 and any_regression:
            return "MIXED"
        if delta == 0:
            return "NO_EFFECT"
        return "FAIL_REGRESSION"
    if delta > 0:
        return "POSITIVE_SIGNAL"
    if delta == 0:
        return "NO_EFFECT"
    return "NEGATIVE_SIGNAL"


def build(task: str) -> tuple[dict[str, Any], str]:
    cfg = CONFIGS[task]
    manifest = json.loads((ROOT / cfg["manifest"]).read_text())
    baseline = json.loads((ROOT / cfg["baseline"]).read_text())
    v3 = json.loads((ROOT / cfg["v3"]).read_text())
    expected_ids = [row["id"] for row in manifest["items"]]
    baseline_ids = [row["id"] for row in baseline["items"]]
    v3_ids = [row["id"] for row in v3["items"]]
    baseline_by_id = {row["id"]: row for row in baseline["items"]}
    v3_by_id = {row["id"]: row for row in v3["items"]}

    paired = []
    for item_id in expected_ids:
        left = baseline_by_id[item_id]
        right = v3_by_id[item_id]
        invalid = not finite_clean(left) or not finite_clean(right)
        if invalid:
            delta = None
            status = "invalid"
        else:
            delta = float(right["score"]) - float(left["score"])
            status = "improved" if delta > 0 else "worse" if delta < 0 else "unchanged"
        paired.append(
            sanitize(
                {
                    "id": item_id,
                    "level": left["level"],
                    "dataset": left.get("dataset"),
                    "answer_format": left["answer_format"],
                    "baseline_raw_output": left.get("raw_output"),
                    "v3_raw_output": right.get("raw_output"),
                    "baseline_parsed_output": left.get("parsed"),
                    "v3_parsed_output": right.get("parsed"),
                    "baseline_score": left.get("score"),
                    "v3_score": right.get("score"),
                    "score_delta": delta,
                    "baseline_parse_error": left.get("parse_error"),
                    "v3_parse_error": right.get("parse_error"),
                    "status": status,
                    "raw_output_changed": left.get("raw_output") != right.get("raw_output"),
                }
            )
        )

    baseline_fixed = fixed_score(baseline["items"])
    v3_fixed = fixed_score(v3["items"])
    delta = v3_fixed - baseline_fixed
    by_level = deltas(
        grouped(baseline["items"], lambda row: f"L{row['level']}"),
        grouped(v3["items"], lambda row: f"L{row['level']}"),
    )
    by_dataset = deltas(
        grouped(baseline["items"], lambda row: row.get("dataset") or "unknown"),
        grouped(v3["items"], lambda row: row.get("dataset") or "unknown"),
    )
    by_format = deltas(
        grouped(baseline["items"], lambda row: row["answer_format"]),
        grouped(v3["items"], lambda row: row["answer_format"]),
    )
    baseline_parse = sum(row.get("parse_error") is not None for row in baseline["items"])
    v3_parse = sum(row.get("parse_error") is not None for row in v3["items"])
    exact_match = baseline_ids == expected_ids == v3_ids
    valid = (
        manifest.get("preflight_valid") is True
        and exact_match
        and len(expected_ids) == cfg["expected"]
        and baseline_parse == 0
        and v3_parse == 0
        and all(finite_clean(row) for row in baseline["items"] + v3["items"])
    )
    verdict = determine_verdict(task, valid, delta, by_level)
    counts = {status: sum(row["status"] == status for row in paired) for status in ("improved", "worse", "unchanged", "invalid")}
    changed = [row for row in paired if row["raw_output_changed"]]
    immutable_hashes = {
        "frozen_skill_v3": {"path": "prompts/generated/factorybench_skill_v3.txt", "sha256": sha256(ROOT / "prompts/generated/factorybench_skill_v3.txt")},
        "original_invalid_preflight": {"path": "data_manifests/task_a_validation7_invalid_preflight.json", "sha256": sha256(ROOT / "data_manifests/task_a_validation7_invalid_preflight.json")},
        "original_invalid_preflight_report": {"path": "reports/task_a_validation7_invalid_preflight.md", "sha256": sha256(ROOT / "reports/task_a_validation7_invalid_preflight.md")},
        "optimization_id_source": {"path": "results/smoke/l123_baseline_fresh_gpt55.json", "sha256": sha256(ROOT / "results/smoke/l123_baseline_fresh_gpt55.json")},
        "manifest": {"path": cfg["manifest"], "sha256": sha256(ROOT / cfg["manifest"])},
        "baseline_result": {"path": cfg["baseline"], "sha256": sha256(ROOT / cfg["baseline"])},
        "v3_result": {"path": cfg["v3"], "sha256": sha256(ROOT / cfg["v3"])},
        "baseline_log": {"path": cfg["baseline_log"], "sha256": sha256(ROOT / cfg["baseline_log"])},
        "v3_log": {"path": cfg["v3_log"], "sha256": sha256(ROOT / cfg["v3_log"])},
        "exact_id_runner": {"path": "scripts/task_a_exact_id_runner.py", "sha256": sha256(ROOT / "scripts/task_a_exact_id_runner.py")},
        "comparison_builder": {"path": "scripts/task_a_build_comparison.py", "sha256": sha256(ROOT / "scripts/task_a_build_comparison.py")},
    }
    if task == "a2":
        immutable_hashes["refinement_trace"] = {
            "path": "results/smoke/hermes_self_evolution_trace.json",
            "sha256": sha256(ROOT / "results/smoke/hermes_self_evolution_trace.json"),
        }
    comparison = {
        "task": cfg["name"],
        "protocol": manifest["protocol"],
        "configuration": {
            "dataset": manifest["dataset"],
            "revision": manifest["revision"],
            "split": manifest["split"],
            "levels": manifest["levels"],
            "expected_total": manifest["expected_total"],
            "model": manifest["model"],
            "concurrency": manifest["concurrency"],
            "frozen_skill_sha256": manifest["frozen_skill_sha256"],
        },
        "pairing": {
            "item_count": len(expected_ids),
            "exact_ordered_id_match": exact_match,
            "manifest_preflight_valid": manifest.get("preflight_valid"),
        },
        "selection_preflight": (
            {
                "seed": manifest["selection_seed"],
                "rule": manifest["selection_rule"],
                "source_l1_test_count": manifest["source_l1_test_count"],
                "exclusion_counts": manifest["exclusion_counts"],
                "eligible_item_count": manifest["eligible_item_count"],
                "selected_ids": expected_ids,
                "selected_episodes": [row["episode"] for row in manifest["items"]],
            }
            if task == "a2" else None
        ),
        "scores": {
            "baseline_factorybench_reported": baseline["score"],
            "v3_factorybench_reported": v3["score"],
            "baseline_fixed_cardinality": baseline_fixed,
            "v3_fixed_cardinality": v3_fixed,
            "absolute_delta": delta,
            "absolute_percentage_point_delta": 100.0 * delta,
            "relative_percentage_change": None if baseline_fixed == 0 else 100.0 * delta / abs(baseline_fixed),
        },
        "by_level": by_level,
        "by_dataset": by_dataset,
        "by_answer_format": by_format,
        "counts": counts,
        "parse_failures": {"baseline": baseline_parse, "v3": v3_parse},
        "cost": {"baseline": baseline["cost"], "v3": v3["cost"], "delta": v3["cost"] - baseline["cost"]},
        "token_usage": {"baseline": baseline["tokens_used"], "v3": v3["tokens_used"]},
        "wall_time_seconds": {"baseline": baseline["wall_time_seconds"], "v3": v3["wall_time_seconds"], "delta": v3["wall_time_seconds"] - baseline["wall_time_seconds"]},
        "changed_case_count": len(changed),
        "changed_cases": changed,
        "items": paired,
        "valid": valid,
        "verdict": verdict,
        "artifact_hashes": immutable_hashes,
        "final_hash_registry": "results/generalization/task_a_stratified_integrity_hashes.json",
        "notes": [
            "No parse failure occurred, so reported and fixed-cardinality scores are identical.",
            "Final hashes of this comparison JSON and its Markdown report are recorded in the external final hash registry to avoid self-referential hashing."
        ],
    }

    def fmt(value: float) -> str:
        return f"{value:.10f}"

    lines = [
        f"# {cfg['name']}",
        "",
        "This is a frozen Skill v3 smoke evaluation. No refinement occurred and no result was fed back into Hermes.",
        "",
        "## Configuration",
        "",
        f"- Revision: `{manifest['revision']}`",
        f"- Split: `{manifest['split']}`",
        f"- Levels: `{manifest['levels']}`",
        f"- Items: {len(expected_ids)}",
        f"- Model: `{manifest['model']}`",
        f"- Concurrency: {manifest['concurrency']}",
        f"- Frozen Skill v3 SHA-256: `{manifest['frozen_skill_sha256']}`",
        "",
    ]
    if task == "a2":
        lines += [
            "## Frozen selection preflight",
            "",
            f"- Seed: `{manifest['selection_seed']}`",
            f"- Rule: {manifest['selection_rule']}",
            f"- Pinned L1 test items: {manifest['source_l1_test_count']}",
            f"- Union excluded: {manifest['exclusion_counts']['union_excluded_l1_items']}",
            f"- Eligible: {manifest['eligible_item_count']}",
            f"- Selected: {manifest['selected_item_count']}",
            "- Selected IDs and provenance episodes are frozen in the manifest.",
            "",
        ]
    lines += [
        "## Overall scores",
        "",
        "| Run | FactoryBench score | Fixed-cardinality score | Parse failures | Cost | Wall time |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Baseline | {fmt(baseline['score'])} | {fmt(baseline_fixed)} | {baseline_parse} | ${baseline['cost']:.6f} | {baseline['wall_time_seconds']:.6f} s |",
        f"| Frozen Skill v3 | {fmt(v3['score'])} | {fmt(v3_fixed)} | {v3_parse} | ${v3['cost']:.6f} | {v3['wall_time_seconds']:.6f} s |",
        "",
        f"Absolute delta: `{delta}` ({100.0 * delta:+.6f} percentage points; {comparison['scores']['relative_percentage_change']:+.6f}% relative).",
        "",
        "## Group scores",
        "",
    ]
    for title, table in (("By level", by_level), ("By dataset", by_dataset), ("By answer format", by_format)):
        lines += [f"### {title}", "", "| Group | Baseline | v3 | Delta |", "|---|---:|---:|---:|"]
        for key, row in table.items():
            lines.append(f"| {key} | {fmt(row['baseline'])} | {fmt(row['v3'])} | {row['delta']:+.10f} |")
        lines.append("")
    lines += [
        "## Paired outcomes",
        "",
        f"- Improved: {counts['improved']}",
        f"- Worse: {counts['worse']}",
        f"- Unchanged: {counts['unchanged']}",
        f"- Invalid: {counts['invalid']}",
        f"- Raw-output changed cases: {len(changed)}",
        "",
        "### Changed cases",
        "",
    ]
    if changed:
        lines += ["| ID | Level | Format | Baseline output | v3 output | Baseline score | v3 score | Status |", "|---|---:|---|---|---|---:|---:|---|"]
        for row in changed:
            left = str(row["baseline_raw_output"]).replace("|", "\\|")
            right = str(row["v3_raw_output"]).replace("|", "\\|")
            lines.append(f"| `{row['id']}` | L{row['level']} | {row['answer_format']} | `{left}` | `{right}` | {row['baseline_score']} | {row['v3_score']} | {row['status']} |")
    else:
        lines.append("None.")
    lines += ["", "## Token usage", "", "```json", json.dumps(comparison["token_usage"], indent=2), "```", "", "## Artifact hashes", ""]
    for name, record in immutable_hashes.items():
        lines.append(f"- {name}: `{record['sha256']}` — `{record['path']}`")
    lines += [
        "- Final comparison/report hashes: recorded in `results/generalization/task_a_stratified_integrity_hashes.json`.",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "The earlier `INVALID_PREFLIGHT` record remains preserved as a separate artifact and was not overwritten or relabeled.",
        "",
    ]
    if task == "a1":
        lines.append("The overall delta is negative, so this frozen L2/L3 validation smoke is a regression under the predeclared rule.")
    else:
        lines.append("The frozen Skill v3 score is below baseline on the episode-disjoint L1 holdout, so this is a negative signal.")
    lines += [
        "",
        "This small smoke result is not a test of a Golden Meta-Prompt and is not proof of broad generalization.",
        "",
    ]
    return comparison, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    comparison, report = build(args.task)
    args.json_out.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.report_out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
