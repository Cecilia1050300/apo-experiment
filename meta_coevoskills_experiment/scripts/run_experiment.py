#!/usr/bin/env python3
"""Controlled CoEvoSkills-style shared-Skill experiment for FactoryBench L1-L3."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
FOLD_SCHEDULE = [
    {"round": 1, "optimization": "A", "validation": "B"},
    {"round": 2, "optimization": "B", "validation": "A"},
    {"round": 3, "optimization": "A", "validation": "B"},
]
REQUIRED_LEVELS = {1, 2, 3}
FORBIDDEN_SKILL_TERMS = ("holdout", "held-out", "fold a", "fold b", "surrogate verifier", "control arm")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a mapping")
    config["_config_path"] = str(path.resolve())
    return config


def assert_no_sol(model: str) -> None:
    normalized = re.sub(r"[^a-z0-9]", "", str(model).casefold())
    if (
        "gpt56sol" in normalized
        or (normalized.startswith("gpt") and normalized.endswith("sol"))
        or "gpt56lunapro" in normalized
    ):
        raise ValueError(f"Sol models are forbidden in this experiment: {model}")


def assert_reasoning_config(model_config: dict[str, Any], role: str) -> None:
    required = {"model", "reasoning_mode", "reasoning_effort"}
    missing = required - set(model_config)
    if missing:
        raise ValueError(f"{role} model configuration missing explicit fields: {sorted(missing)}")
    mode = model_config["reasoning_mode"]
    effort = model_config["reasoning_effort"]
    if mode not in {None, "standard", "pro"}:
        raise ValueError(f"{role} reasoning mode is invalid: {mode}")
    if effort not in {None, "none", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError(f"{role} reasoning effort is invalid: {effort}")
    if role == "target" and (mode is not None or effort is not None):
        raise ValueError("target model must not use Responses reasoning settings")
    if role != "target" and (mode is None or effort is None):
        raise ValueError(f"{role} model requires reasoning mode and effort")


def responses_reasoning(model_config: dict[str, Any], role: str) -> dict[str, str]:
    assert_reasoning_config(model_config, role)
    if role == "target":
        raise ValueError("target model does not use the Responses API reasoning parameter")
    return {"mode": model_config["reasoning_mode"], "effort": model_config["reasoning_effort"]}


def model_identity(model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model_config["model"],
        "reasoning_mode": model_config["reasoning_mode"],
        "reasoning_effort": model_config["reasoning_effort"],
    }


def assert_model_roles(config: dict[str, Any]) -> None:
    models = config["models"]
    expected = {
        "target": ("gpt-4o-mini", None, None),
        "verifier": ("gpt-5.6-luna", "standard", "high"),
        "rewriter": ("gpt-5.6-luna", "pro", "xhigh"),
    }
    for role, (model_id, mode, effort) in expected.items():
        actual = models[role]
        assert_reasoning_config(actual, role)
        assert_no_sol(actual["model"])
        if actual["model"] != model_id:
            raise ValueError(f"{role} model must be {model_id}, got {actual['model']}")
        if actual.get("reasoning_mode") != mode:
            raise ValueError(f"{role} reasoning mode must be {mode}")
        if actual.get("reasoning_effort") != effort:
            raise ValueError(f"{role} reasoning effort must be {effort}")
    if models["target"]["model"] in {models["verifier"]["model"], models["rewriter"]["model"]}:
        raise ValueError("target model must remain distinct from verifier/rewriter")


def manifest_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config["manifests"][key]["path"])


def validate_manifest(config: dict[str, Any], key: str) -> dict[str, Any]:
    spec = config["manifests"][key]
    path = manifest_path(config, key)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = sha256_file(path)
    if actual_sha != spec["sha256"]:
        raise ValueError(f"manifest {key} SHA256 mismatch: {actual_sha}")
    payload = load_json(path)
    rows = payload.get("items")
    if not isinstance(rows, list) or len(rows) != int(spec["expected_count"]):
        raise ValueError(f"manifest {key} expected {spec['expected_count']} items")
    if payload.get("expected_count") != len(rows):
        raise ValueError(f"manifest {key} embedded expected_count mismatch")
    levels = [int(row.get("level", -1)) for row in rows]
    if set(levels) - REQUIRED_LEVELS or 4 in levels:
        raise ValueError(f"manifest {key} must contain only FactoryBench Levels 1, 2, and 3")
    expected_per_level = 3 if key == "holdout" else 1
    counts = Counter(levels)
    if counts != Counter({1: expected_per_level, 2: expected_per_level, 3: expected_per_level}):
        raise ValueError(f"manifest {key} level balance mismatch: {dict(counts)}")
    for row in rows:
        expected_split = "test" if row["level"] == 1 else "validation"
        if row.get("split") != expected_split:
            raise ValueError(f"manifest {key} split provenance mismatch for {row['id']}")
        if not row.get("episode"):
            raise ValueError(f"manifest {key} missing episode for {row['id']}")
    warning = str(payload.get("mixed_split_warning", ""))
    if "not a pure validation score" not in warning:
        raise ValueError(f"manifest {key} lost the mixed-split warning")
    return payload


def preflight(config: dict[str, Any], *, verify_sources: bool = False) -> dict[str, Any]:
    assert_model_roles(config)
    m1_root = resolve_path(config["frozen_m1"]["root"])
    frozen_paths = {
        "runner_sha256": m1_root / "scripts/meta_m1/run_m1.py",
        "preparation_sha256": m1_root / "scripts/meta_m1/prepare_m1.py",
        "combined_manifest_sha256": m1_root / "data_manifests/meta_m1/factorybench_l123_combined_manifest.json",
        "contamination_registry_sha256": m1_root / "data_manifests/meta_m1/contamination_registry_m1.json",
        "selection_sha256": m1_root / "prompts/adapters/m1_factorybench_l123/selection.json",
        "factorybench_evaluate_sha256": REPO / "FactoryBench/FactoryBench/factorybench/evaluate.py",
        "factorybench_prompt_sha256": REPO / "FactoryBench/FactoryBench/factorybench/prompt.py",
        "factorybench_parse_sha256": REPO / "FactoryBench/FactoryBench/factorybench/parse.py",
        "factorybench_score_sha256": REPO / "FactoryBench/FactoryBench/factorybench/score.py",
    }
    for key, path in frozen_paths.items():
        if not path.exists() or sha256_file(path) != config["frozen_m1"][key]:
            raise ValueError(f"frozen M1 reference hash mismatch: {path}")
    manifests = {key: validate_manifest(config, key) for key in ("A", "B", "holdout")}
    ids = {key: [row["id"] for row in value["items"]] for key, value in manifests.items()}
    episodes = {key: [str(row["episode"]) for row in value["items"]] for key, value in manifests.items()}
    if set(ids["A"]) & set(ids["B"]) or (set(ids["A"]) | set(ids["B"])) & set(ids["holdout"]):
        raise ValueError("development/holdout IDs are not disjoint")
    all_episodes = episodes["A"] + episodes["B"] + episodes["holdout"]
    if len(set(all_episodes)) != len(all_episodes):
        raise ValueError("M1 contamination policy requires disjoint episodes across folds and holdout")

    control = resolve_path(config["skills"]["control_v0"])
    surrogate = resolve_path(config["skills"]["surrogate_v0"])
    if not control.exists() or not surrogate.exists():
        raise FileNotFoundError("both v0 Skill files must exist")
    if control.read_bytes() != surrogate.read_bytes():
        raise ValueError("CONTROL and SURROGATE Skill v0 files are not byte-identical")
    skill_sha = sha256_file(control)
    expected_sha = config["skills"]["expected_v0_sha256"]
    if skill_sha != expected_sha or skill_sha != EMPTY_SHA256:
        raise ValueError(f"Skill v0 SHA256 mismatch: {skill_sha}")

    verifier_schema = load_json(resolve_path(config["verifier"]["schema"]))
    rewriter_schema = load_json(resolve_path(config["rewriter"]["schema"]))
    if verifier_schema.get("type") != "object" or rewriter_schema.get("type") != "object":
        raise ValueError("structured-output schemas must be JSON object schemas")

    source_count = None
    if verify_sources:
        source_items = load_source_items(config, manifests)
        source_count = len(source_items)
        expected_rows = [row for key in ("A", "B", "holdout") for row in manifests[key]["items"]]
        for row in expected_rows:
            item = source_items[row["id"]]
            if item.level != row["level"] or item.answer_format.value != row["answer_format"]:
                raise ValueError(f"manifest/source mismatch for {row['id']}")
            if str(item.provenance.get("episode")) != str(row["episode"]):
                raise ValueError(f"manifest/source episode mismatch for {row['id']}")

    return {
        "status": "PASS",
        "api_model_validation": "local exact-config validation; no API call",
        "api_models": {
            role: {
                "model": config["models"][role]["model"],
                "reasoning_mode": config["models"][role]["reasoning_mode"],
                "reasoning_effort": config["models"][role]["reasoning_effort"],
            }
            for role in ("target", "verifier", "rewriter")
        },
        "levels": [1, 2, 3],
        "counts": {key: len(value["items"]) for key, value in manifests.items()},
        "all_ids_disjoint": True,
        "all_episodes_disjoint": True,
        "skill_v0_byte_identical": True,
        "skill_v0_sha256": skill_sha,
        "frozen_m1_hashes_verified": len(frozen_paths),
        "source_items_verified": source_count,
        "mixed_split_warning": "FactoryBench L1-L3 mixed-split evaluation: L1 test plus L2/L3 validation; not a pure validation score.",
    }


def selected_arms(arm: str) -> list[str]:
    return ["control", "surrogate"] if arm == "both" else [arm]


def build_run_plan(config: dict[str, Any], *, arm: str, rounds: int, smoke: bool) -> dict[str, Any]:
    if arm not in {"control", "surrogate", "both"}:
        raise ValueError("arm must be control, surrogate, or both")
    if rounds < 1 or rounds > 3:
        raise ValueError("rounds must be between 1 and 3")
    arms = selected_arms(arm)
    treatment_count = int("surrogate" in arms)
    if smoke:
        target_calls = 3 + len(arms) * rounds * 3
        holdout_calls = 0
    else:
        target_calls = 15 + len(arms) * rounds * 12
        holdout_calls = 9 + len(arms) * rounds * 9
    verifier_calls = treatment_count * rounds * 3
    rewriter_calls = len(arms) * rounds
    calls = {
        "target": target_calls,
        "verifier": verifier_calls,
        "rewriter": rewriter_calls,
        "maximum_total": target_calls + verifier_calls + rewriter_calls,
    }
    mode = "smoke" if smoke else "full"
    return {
        "experiment": config["experiment"]["name"],
        "evaluation_label": "FactoryBench L1-L3 mixed-split evaluation",
        "mode": mode,
        "arms": arms,
        "rounds": rounds,
        "models": {role: config["models"][role]["model"] for role in ("target", "verifier", "rewriter")},
        "inference_settings": {
            role: {key: value for key, value in config["models"][role].items() if key != "model"}
            for role in ("target", "verifier", "rewriter")
        },
        "reasoning": {
            role: {
                "mode": config["models"][role]["reasoning_mode"],
                "effort": config["models"][role]["reasoning_effort"],
            }
            for role in ("verifier", "rewriter")
        },
        "concurrency": config["execution"]["concurrency"],
        "fold_schedule": FOLD_SCHEDULE[:rounds],
        "sample_levels": [1, 2, 3],
        "manifests": {key: {"path": str(manifest_path(config, key)), "sha256": config["manifests"][key]["sha256"]} for key in ("A", "B", "holdout")},
        "frozen_artifacts": {
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "runner_sha256": sha256_file(Path(__file__)),
            "skill_v0_sha256": config["skills"]["expected_v0_sha256"],
            "verifier_prompt_sha256": sha256_file(resolve_path(config["verifier"]["prompt"])),
            "verifier_schema_sha256": sha256_file(resolve_path(config["verifier"]["schema"])),
            "rewriter_prompt_sha256": sha256_file(resolve_path(config["rewriter"]["prompt"])),
            "rewriter_schema_sha256": sha256_file(resolve_path(config["rewriter"]["schema"])),
            "dataset_revision": config["execution"]["dataset_revision"],
            "repository_git_commit_at_design": config["execution"]["repository_git_commit_at_design"],
        },
        "output_directory": str(ROOT / "runs" / mode),
        "http_policy": {
            "max_retries": config["execution"]["max_retries"],
            "timeout_seconds": config["execution"]["timeout_seconds"],
        },
        "calls": calls,
        "holdout_calls": holdout_calls,
        "checkpoint_policy": "atomic per-call checkpoints; --resume reuses completed calls",
        "holdout_policy": "measurement only; never enters verifier, aggregation, or rewriter input",
    }


def plan_sha256(plan: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(plan))


def print_run_plan(plan: dict[str, Any]) -> None:
    digest = plan_sha256(plan)
    print("=" * 72)
    print("RUN PLAN — NO API CALL HAS BEEN MADE")
    print("=" * 72)
    print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"PLAN_SHA256={digest}")
    print("Execution requires both --execute-paid and this exact --plan-sha256 value.")


def authorize_paid_execution(plan: dict[str, Any], *, execute_paid: bool, supplied_plan_sha256: str | None) -> None:
    expected = plan_sha256(plan)
    if not execute_paid:
        raise SystemExit("RUN PLAN only: stopped before paid API calls")
    if supplied_plan_sha256 != expected:
        raise ValueError(f"plan SHA256 authorization mismatch; expected {expected}")


def archive_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    attempt = 1
    while True:
        archive = path.with_name(f"{path.stem}.attempt_{attempt}_{payload.get('status', 'unknown')}{path.suffix}")
        if not archive.exists():
            atomic_write_json(archive, payload)
            return archive
        attempt += 1


def checkpointed_call(
    path: Path,
    call: Callable[[], dict[str, Any]],
    *,
    resume: bool,
    request_sha256: str,
) -> dict[str, Any]:
    if path.exists():
        existing = load_json(path)
        if resume and existing.get("_request_sha256") != request_sha256:
            raise ValueError(f"checkpoint request identity mismatch: {path}")
        if resume and existing.get("status") == "complete":
            return existing
        if resume:
            archive_checkpoint(path, existing)
        if not resume:
            raise FileExistsError(f"checkpoint exists; pass --resume to reuse: {path}")
    try:
        result = call()
    except Exception as exc:
        atomic_write_json(path, {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "_request_sha256": request_sha256})
        raise
    result.setdefault("status", "complete")
    result["_request_sha256"] = request_sha256
    atomic_write_json(path, result)
    return result


def mark_checkpoint_invalid(path: Path, exc: Exception) -> None:
    payload = load_json(path)
    payload["status"] = "invalid"
    payload["validation_error_type"] = type(exc).__name__
    payload["validation_error"] = str(exc)
    atomic_write_json(path, payload)


def write_or_verify_json(path: Path, payload: Any, *, resume: bool) -> None:
    encoded = canonical_json_bytes(payload)
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        if path.read_bytes() != encoded:
            raise ValueError(f"resume input identity mismatch: {path}")
        return
    atomic_write_bytes(path, encoded)


def request_identity(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def validate_returned_model(expected: str, actual: str) -> None:
    assert_no_sol(actual)
    if actual != expected:
        raise ValueError(f"returned model mismatch: expected {expected}, got {actual}")


def require_complete_call(call: dict[str, Any]) -> None:
    if call.get("status") != "complete":
        raise ValueError(call.get("validation_error") or call.get("error") or "paid call checkpoint is not valid")


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0),
    }


def estimated_cost(config: dict[str, Any], model: str, usage: dict[str, int]) -> float | None:
    rates = config.get("pricing_per_million_usd", {}).get(model)
    if not rates:
        return None
    return (usage["input_tokens"] * float(rates["input"]) + usage["output_tokens"] * float(rates["output"])) / 1_000_000


def load_source_items(config: dict[str, Any], manifests: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    from factorybench.data import load_split

    manifests = manifests or {key: load_json(manifest_path(config, key)) for key in ("A", "B", "holdout")}
    keys = sorted({(int(row["level"]), str(row["split"])) for manifest in manifests.values() for row in manifest["items"]})
    index: dict[str, Any] = {}
    revision = config["execution"]["dataset_revision"]
    for level, split in keys:
        for item in load_split(level, split=split, revision=revision, max_items=None):
            index[item.id] = item
    required = {row["id"] for manifest in manifests.values() for row in manifest["items"]}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"manifest IDs unavailable from pinned FactoryBench revision: {missing}")
    return {item_id: index[item_id] for item_id in required}


def fixed_score(rows: list[dict[str, Any]]) -> float:
    raw_scores, chances = [], []
    for row in rows:
        valid = row.get("parse_error") is None and isinstance(row.get("score"), (int, float)) and math.isfinite(float(row["score"]))
        raw_scores.append(float(row["score"]) if valid else 0.0)
        chances.append(float(row.get("chance", 0.0)))
    mean_score = sum(raw_scores) / len(raw_scores)
    mean_chance = sum(chances) / len(chances)
    return (mean_score - mean_chance) / (1.0 - mean_chance) if mean_chance < 1.0 else 0.0


def canonical_score(rows: list[dict[str, Any]]) -> float | None:
    clean = [row for row in rows if row.get("parse_error") is None and isinstance(row.get("score"), (int, float)) and math.isfinite(float(row["score"]))]
    return fixed_score(clean) if clean else None


def grouped_fixed(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: fixed_score(value) for key, value in sorted(groups.items())}


def target_call(client: Any, config: dict[str, Any], skill: str, prompt: str) -> dict[str, Any]:
    model = config["models"]["target"]
    assert_no_sol(model["model"])
    messages = []
    if skill:
        messages.append({"role": "system", "content": skill})
    messages.append({"role": "user", "content": prompt})
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model["model"],
        messages=messages,
        temperature=model["temperature"],
        seed=model["seed"],
        max_completion_tokens=model["max_completion_tokens"],
    )
    wall = time.perf_counter() - started
    usage = usage_dict(response)
    actual_model = str(response.model)
    status = "complete"
    validation_error = None
    try:
        validate_returned_model(model["model"], actual_model)
    except ValueError as exc:
        status = "invalid"
        validation_error = str(exc)
    return {
        "status": status,
        "model": actual_model,
        "response_id": response.id,
        "raw_output": response.choices[0].message.content or "",
        "finish_reason": response.choices[0].finish_reason,
        "usage": usage,
        "estimated_cost_usd": estimated_cost(config, model["model"], usage),
        "wall_time_seconds": wall,
        "validation_error": validation_error,
    }


def validate_cached_evaluation(
    payload: dict[str, Any],
    skill_path: Path,
    expected_ids: list[str],
    model: str,
    evaluation_request_sha256: str,
) -> None:
    if (
        payload.get("skill_sha256") != sha256_file(skill_path)
        or payload.get("model") != model
        or payload.get("ordered_ids") != expected_ids
        or payload.get("evaluation_request_sha256") != evaluation_request_sha256
    ):
        raise ValueError("stale evaluation checkpoint does not match the complete evaluation request identity")


def evaluation_request_identity(config: dict[str, Any], skill_path: Path, items: list[Any]) -> str:
    from factorybench.prompt import render_prompt

    return request_identity({
        "role": "evaluation_aggregate",
        "model_configs": {
            role: model_identity(config["models"][role])
            for role in ("target", "verifier", "rewriter")
        },
        "skill_sha256": sha256_file(skill_path),
        "ordered_ids": [item.id for item in items],
        "rendered_prompt_sha256": [sha256_bytes(render_prompt(item).encode("utf-8")) for item in items],
        "dataset_revision": config["execution"]["dataset_revision"],
        "config_sha256": sha256_file(Path(config["_config_path"])),
        "runner_sha256": sha256_file(Path(__file__)),
        "factorybench_hashes": {
            key: config["frozen_m1"][key]
            for key in (
                "factorybench_evaluate_sha256",
                "factorybench_prompt_sha256",
                "factorybench_parse_sha256",
                "factorybench_score_sha256",
            )
        },
    })


def evaluate_items(client: Any, config: dict[str, Any], items: list[Any], skill_path: Path, result_path: Path, trace_dir: Path, *, resume: bool) -> dict[str, Any]:
    from factorybench.evaluate import _score_one
    from factorybench.prompt import render_prompt

    evaluation_request_sha = evaluation_request_identity(config, skill_path, items)
    if result_path.exists() and resume:
        cached = load_json(result_path)
        validate_cached_evaluation(
            cached,
            skill_path,
            [item.id for item in items],
            config["models"]["target"]["model"],
            evaluation_request_sha,
        )
        return cached
    if result_path.exists() and not resume:
        raise FileExistsError(f"result exists; pass --resume: {result_path}")
    skill = skill_path.read_text(encoding="utf-8")
    concurrency = int(config["execution"]["concurrency"])

    def run_one(index_item: tuple[int, Any]) -> tuple[int, Any, str, dict[str, Any]]:
        index, item = index_item
        prompt = render_prompt(item)
        checkpoint = trace_dir / f"{index:02d}_{item.id}_target.json"
        request_sha = request_identity({
            "role": "target",
            "model": config["models"]["target"],
            "skill_sha256": sha256_bytes(skill.encode("utf-8")),
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "config_sha256": sha256_file(Path(config["_config_path"])),
            "runner_sha256": sha256_file(Path(__file__)),
        })
        call = checkpointed_call(
            checkpoint,
            lambda: target_call(client, config, skill, prompt),
            resume=resume,
            request_sha256=request_sha,
        )
        require_complete_call(call)
        return index, item, prompt, call

    indexed = list(enumerate(items))
    if concurrency <= 1:
        calls = [run_one(pair) for pair in indexed]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            calls = list(pool.map(run_one, indexed))
    calls.sort(key=lambda row: row[0])

    rows = []
    for _, item, _, call in calls:
        result = _score_one(item, call["raw_output"])
        score = float(result.score) if isinstance(result.score, (int, float)) and math.isfinite(float(result.score)) else None
        rows.append({
            "id": item.id,
            "level": item.level,
            "dataset": item.dataset,
            "episode": item.provenance.get("episode"),
            "answer_format": item.answer_format.value,
            "template_id": item.template_id,
            "template_type": item.template_type,
            "raw_output": call["raw_output"],
            "parsed": result.parsed,
            "score": score,
            "chance": result.chance,
            "parse_error": result.parse_error or ("non_finite_score" if score is None else None),
            "target_usage": call["usage"],
            "target_estimated_cost_usd": call["estimated_cost_usd"],
            "target_wall_time_seconds": call["wall_time_seconds"],
        })
    usage = {
        "input_tokens": sum(row[3]["usage"]["input_tokens"] for row in calls),
        "output_tokens": sum(row[3]["usage"]["output_tokens"] for row in calls),
    }
    payload = {
        "evaluation_label": "FactoryBench L1-L3 mixed-split evaluation",
        "model": config["models"]["target"]["model"],
        "evaluation_request_sha256": evaluation_request_sha,
        "skill_path": str(skill_path),
        "skill_sha256": sha256_file(skill_path),
        "ordered_ids": [row["id"] for row in rows],
        "item_count": len(rows),
        "canonical_score": canonical_score(rows),
        "fixed_cardinality_score": fixed_score(rows),
        "parse_failures": sum(row["parse_error"] is not None for row in rows),
        "format_validity": sum(row["parse_error"] is None for row in rows) / len(rows),
        "by_level": grouped_fixed(rows, "level"),
        "by_answer_format": grouped_fixed(rows, "answer_format"),
        "target_usage": usage,
        "target_estimated_cost_usd": estimated_cost(config, config["models"]["target"]["model"], usage),
        "target_wall_time_seconds": sum(float(row[3]["wall_time_seconds"]) for row in calls),
        "items": rows,
    }
    atomic_write_json(result_path, payload)
    return payload


def build_coarse_feedback(evaluation: dict[str, Any]) -> dict[str, Any]:
    failed = []
    for row in evaluation["items"]:
        score = row.get("score")
        if row.get("parse_error") is not None or score is None or float(score) < 1.0:
            failed.append({"sample_id": row["id"], "level": row["level"], "answer_format": row["answer_format"]})
    return {
        "canonical_score": evaluation["canonical_score"],
        "fixed_cardinality_score": evaluation["fixed_cardinality_score"],
        "parse_failures": evaluation["parse_failures"],
        "failed_samples": failed,
    }


def build_verifier_payload(sample_id: str, level: int, rendered_input: str, answer_format: str, target_answer: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "level": level,
        "public_rendered_task_input": rendered_input,
        "output_format_requirement": answer_format,
        "target_model_answer": target_answer,
    }


def validate_diagnosis(value: dict[str, Any], expected_id: str, expected_level: int, allowed: set[str]) -> None:
    required = {"sample_id", "level", "surrogate_pass", "failed_assertions", "failure_types", "root_cause", "skill_gap", "actionable_revision", "confidence"}
    if set(value) != required:
        raise ValueError(f"diagnosis schema fields mismatch: {set(value) ^ required}")
    if value["sample_id"] != expected_id or value["level"] != expected_level:
        raise ValueError("diagnosis sample identity mismatch")
    if not isinstance(value["surrogate_pass"], bool) or not isinstance(value["failed_assertions"], list):
        raise ValueError("diagnosis pass/assertion types invalid")
    if not isinstance(value["failure_types"], list) or not set(value["failure_types"]) <= allowed:
        raise ValueError("diagnosis contains unsupported failure type")
    if not 0.0 <= float(value["confidence"]) <= 1.0:
        raise ValueError("diagnosis confidence out of range")
    for assertion in value["failed_assertions"]:
        if set(assertion) != {"assertion", "observed", "expected_behavior"}:
            raise ValueError("failed_assertions item schema mismatch")
    if value["surrogate_pass"] and (value["failed_assertions"] or value["failure_types"]):
        raise ValueError("passing diagnosis cannot contain failures")


def verifier_call(client: Any, config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    model = config["models"]["verifier"]
    assert_no_sol(model["model"])
    schema = load_json(resolve_path(config["verifier"]["schema"]))
    prompt = resolve_path(config["verifier"]["prompt"]).read_text(encoding="utf-8")
    started = time.perf_counter()
    response = client.responses.create(
        model=model["model"],
        reasoning=responses_reasoning(model, "verifier"),
        instructions=prompt,
        input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        max_output_tokens=model["max_output_tokens"],
        text={"format": {"type": "json_schema", "name": "factorybench_surrogate_diagnosis", "strict": True, "schema": schema}},
    )
    wall = time.perf_counter() - started
    usage = usage_dict(response)
    actual_model = str(response.model)
    status = "complete"
    validation_error = None
    try:
        validate_returned_model(model["model"], actual_model)
    except ValueError as exc:
        status = "invalid"
        validation_error = str(exc)
    return {
        "status": status,
        "model": actual_model,
        "reasoning_mode": model["reasoning_mode"],
        "reasoning_effort": model["reasoning_effort"],
        "response_id": response.id,
        "raw_output": response.output_text,
        "usage": usage,
        "estimated_cost_usd": estimated_cost(config, model["model"], usage),
        "wall_time_seconds": wall,
        "validation_error": validation_error,
    }


def run_verifier(client: Any, config: dict[str, Any], evaluation: dict[str, Any], source_items: dict[str, Any], output_dir: Path, *, resume: bool) -> list[dict[str, Any]]:
    from factorybench.prompt import render_prompt

    diagnoses = []
    allowed = set(config["verifier"]["allowed_failure_types"])
    for row in evaluation["items"]:
        item = source_items[row["id"]]
        payload = build_verifier_payload(row["id"], row["level"], render_prompt(item), row["answer_format"], row["raw_output"])
        input_path = output_dir / f"{row['id']}_verifier_input.json"
        write_or_verify_json(input_path, payload, resume=resume)
        checkpoint = output_dir / f"{row['id']}_verifier_response.json"
        request_sha = request_identity({
            "role": "verifier",
            "model": config["models"]["verifier"]["model"],
            "reasoning_mode": config["models"]["verifier"]["reasoning_mode"],
            "reasoning_effort": config["models"]["verifier"]["reasoning_effort"],
            "input_sha256": sha256_file(input_path),
            "prompt_sha256": sha256_file(resolve_path(config["verifier"]["prompt"])),
            "schema_sha256": sha256_file(resolve_path(config["verifier"]["schema"])),
            "runner_sha256": sha256_file(Path(__file__)),
        })
        call = checkpointed_call(
            checkpoint,
            lambda payload=payload: verifier_call(client, config, payload),
            resume=resume,
            request_sha256=request_sha,
        )
        require_complete_call(call)
        try:
            diagnosis = json.loads(call["raw_output"])
            validate_diagnosis(diagnosis, row["id"], row["level"], allowed)
        except Exception as exc:
            mark_checkpoint_invalid(checkpoint, exc)
            raise
        diagnosis["_usage"] = call["usage"]
        diagnosis["_estimated_cost_usd"] = call["estimated_cost_usd"]
        diagnosis["_wall_time_seconds"] = call["wall_time_seconds"]
        parsed_path = output_dir / f"{row['id']}_diagnosis.json"
        write_or_verify_json(parsed_path, diagnosis, resume=resume)
        diagnoses.append(diagnosis)
    return diagnoses


def summarize_verifier_usage(diagnoses: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "usage": {
            "input_tokens": sum(int(row["_usage"]["input_tokens"]) for row in diagnoses),
            "output_tokens": sum(int(row["_usage"]["output_tokens"]) for row in diagnoses),
        },
        "estimated_cost_usd": sum(float(row.get("_estimated_cost_usd") or 0.0) for row in diagnoses),
        "wall_time_seconds": sum(float(row.get("_wall_time_seconds") or 0.0) for row in diagnoses),
        "calls": len(diagnoses),
    }


def aggregate_diagnoses(diagnoses: list[dict[str, Any]], *, round_number: int, optimization_fold: str) -> dict[str, Any]:
    type_counts = Counter(value for diagnosis in diagnoses for value in diagnosis["failure_types"])
    type_levels: dict[str, set[int]] = defaultdict(set)
    level_patterns: dict[str, list[str]] = {"1": [], "2": [], "3": []}
    uncertain = []
    for diagnosis in diagnoses:
        level = int(diagnosis["level"])
        for failure_type in diagnosis["failure_types"]:
            type_levels[failure_type].add(level)
            level_patterns[str(level)].append(failure_type)
        if "UNKNOWN" in diagnosis["failure_types"] or float(diagnosis["confidence"]) < 0.5:
            uncertain.append(f"L{level}: diagnosis withheld because observable evidence was insufficient")
    recurring = [f"{failure_type} observed {count} times" for failure_type, count in sorted(type_counts.items()) if count >= 2]
    cross_level = [
        f"{failure_type} occurred across levels {sorted(levels)}"
        for failure_type, levels in sorted(type_levels.items())
        if len(levels) >= 2
    ]
    generic_gap_map = {
        "FORMAT_ERROR": "Strengthen bounded finalization and exact output-contract checks.",
        "INSTRUCTION_MISREAD": "Add a concise objective-and-contract identification step.",
        "TEMPORAL_REASONING_ERROR": "Add general temporal alignment, phase, and boundary checks.",
        "NUMERIC_REASONING_ERROR": "Add unit-aware numeric verification and operation checks.",
        "EVIDENCE_MISUSE": "Separate observed telemetry from inference before answering.",
        "UNSUPPORTED_INFERENCE": "Require every conclusion to be supported by supplied evidence.",
        "MISSING_CONSTRAINT": "Inventory explicit constraints before solving and finalizing.",
        "OVER_GENERALIZATION": "Limit rules to conditions supported by the current task input.",
        "TASK_STRATEGY_ERROR": "Select a task-appropriate procedure before computation.",
    }
    candidate_gaps = [
        generic_gap_map[failure_type]
        for failure_type, count in sorted(type_counts.items())
        if count >= 2 and failure_type in generic_gap_map
    ]
    return {
        "round": round_number,
        "optimization_fold": optimization_fold,
        "failure_type_counts": dict(sorted(type_counts.items())),
        "recurring_patterns": recurring,
        "cross_level_patterns": cross_level,
        "level_specific_patterns": {key: sorted(set(values)) for key, values in level_patterns.items()},
        "candidate_skill_gaps": candidate_gaps,
        "uncertain_diagnoses": uncertain,
    }


def rewriter_call(client: Any, config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    model = config["models"]["rewriter"]
    assert_no_sol(model["model"])
    schema = load_json(resolve_path(config["rewriter"]["schema"]))
    prompt = resolve_path(config["rewriter"]["prompt"]).read_text(encoding="utf-8")
    started = time.perf_counter()
    response = client.responses.create(
        model=model["model"],
        reasoning=responses_reasoning(model, "rewriter"),
        instructions=prompt,
        input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
        max_output_tokens=model["max_output_tokens"],
        text={"format": {"type": "json_schema", "name": "factorybench_shared_skill_rewrite", "strict": True, "schema": schema}},
    )
    wall = time.perf_counter() - started
    usage = usage_dict(response)
    actual_model = str(response.model)
    status = "complete"
    validation_error = None
    try:
        validate_returned_model(model["model"], actual_model)
    except ValueError as exc:
        status = "invalid"
        validation_error = str(exc)
    return {
        "status": status,
        "model": actual_model,
        "reasoning_mode": model["reasoning_mode"],
        "reasoning_effort": model["reasoning_effort"],
        "response_id": response.id,
        "raw_output": response.output_text,
        "usage": usage,
        "estimated_cost_usd": estimated_cost(config, model["model"], usage),
        "wall_time_seconds": wall,
        "validation_error": validation_error,
    }


def validate_new_skill(skill: str, config: dict[str, Any]) -> None:
    if not skill.strip():
        raise ValueError("rewriter produced an empty Skill")
    assert_no_sol(skill)
    lowered = skill.casefold()
    for term in FORBIDDEN_SKILL_TERMS:
        if term in lowered:
            raise ValueError(f"rewritten Skill contains forbidden term: {term}")
    for key in ("A", "B", "holdout"):
        for row in load_json(manifest_path(config, key))["items"]:
            if row["id"].casefold() in lowered:
                raise ValueError("rewritten Skill contains a sample ID")


def rewrite_skill(
    client: Any,
    config: dict[str, Any],
    arm: str,
    current_skill_path: Path,
    feedback: dict[str, Any],
    output_dir: Path,
    round_number: int,
    *,
    resume: bool,
    skill_dir: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    current_skill = current_skill_path.read_text(encoding="utf-8")
    payload = {
        "task": "Shared FactoryBench Levels 1-3 Skill for GPT-4o-mini",
        "target_model": config["models"]["target"]["model"],
        "round": round_number,
        "feedback_mode": "coarse_evaluator_feedback" if arm == "control" else "aggregate_surrogate_diagnosis",
        "current_skill": current_skill,
        "feedback": feedback,
        "constraints": {
            "shared_skill": True,
            "levels": [1, 2, 3],
            "generalizable_only": True,
            "no_sample_memorization": True,
            "minimum_change": True,
        },
    }
    input_path = output_dir / f"round_{round_number}_rewriter_input.json"
    write_or_verify_json(input_path, payload, resume=resume)
    checkpoint = output_dir / f"round_{round_number}_rewriter_response.json"
    request_sha = request_identity({
        "role": "rewriter",
        "model": config["models"]["rewriter"]["model"],
        "reasoning_mode": config["models"]["rewriter"]["reasoning_mode"],
        "reasoning_effort": config["models"]["rewriter"]["reasoning_effort"],
        "input_sha256": sha256_file(input_path),
        "prompt_sha256": sha256_file(resolve_path(config["rewriter"]["prompt"])),
        "schema_sha256": sha256_file(resolve_path(config["rewriter"]["schema"])),
        "runner_sha256": sha256_file(Path(__file__)),
    })
    call = checkpointed_call(
        checkpoint,
        lambda: rewriter_call(client, config, payload),
        resume=resume,
        request_sha256=request_sha,
    )
    require_complete_call(call)
    try:
        parsed = json.loads(call["raw_output"])
        required = set(load_json(resolve_path(config["rewriter"]["schema"]))["required"])
        if set(parsed) != required:
            raise ValueError(f"rewriter output fields mismatch: {set(parsed) ^ required}")
        new_skill = parsed["new_skill"]
        validate_new_skill(new_skill, config)
    except Exception as exc:
        mark_checkpoint_invalid(checkpoint, exc)
        raise
    skill_path = (skill_dir or (output_dir / "skills")) / f"skill_v{round_number}.txt"
    if skill_path.exists() and skill_path.read_text(encoding="utf-8") != new_skill and resume:
        raise ValueError(f"resume Skill differs from checkpoint: {skill_path}")
    atomic_write_bytes(skill_path, new_skill.encode("utf-8"))
    parsed["_trace"] = {
        "model": call["model"],
        "reasoning_mode": call["reasoning_mode"],
        "reasoning_effort": config["models"]["rewriter"]["reasoning_effort"],
        "usage": call["usage"],
        "estimated_cost_usd": call["estimated_cost_usd"],
        "wall_time_seconds": call["wall_time_seconds"],
        "skill_sha256": sha256_file(skill_path),
    }
    write_or_verify_json(output_dir / f"round_{round_number}_rewriter_parsed.json", parsed, resume=resume)
    return skill_path, parsed


def item_changes(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, int]:
    base = {row["id"]: row for row in baseline["items"]}
    counts = {"improved": 0, "regressed": 0, "unchanged_correct": 0, "unchanged_wrong": 0}
    for row in current["items"]:
        prior = base[row["id"]]
        now_score = float(row["score"] or 0.0) if row.get("parse_error") is None else 0.0
        old_score = float(prior["score"] or 0.0) if prior.get("parse_error") is None else 0.0
        if now_score > old_score:
            counts["improved"] += 1
        elif now_score < old_score:
            counts["regressed"] += 1
        elif now_score == 1.0:
            counts["unchanged_correct"] += 1
        else:
            counts["unchanged_wrong"] += 1
    return counts


def skill_token_length(text: str, model: str) -> int:
    try:
        from factorybench.tokens import count_tokens
        return int(count_tokens(text, model=model))
    except Exception:
        return max(0, math.ceil(len(text) / 4))


def generation_metrics(
    arm: str,
    generation: int,
    skill_path: Path,
    optimization: dict[str, Any],
    validation: dict[str, Any],
    holdout: dict[str, Any] | None,
    v0_holdout: dict[str, Any] | None,
    rewrite: dict[str, Any] | None,
    *,
    verifier_usage: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    skill = skill_path.read_text(encoding="utf-8")
    changes = item_changes(holdout, v0_holdout) if holdout is not None and v0_holdout is not None else None
    target_phases = {"optimization": optimization, "validation": validation, "holdout": holdout}
    target_usage = {
        phase: value.get("target_usage") if value is not None else None
        for phase, value in target_phases.items()
    }
    target_cost = sum(float(value.get("target_estimated_cost_usd") or 0.0) for value in target_phases.values() if value is not None)
    target_wall = sum(float(value.get("target_wall_time_seconds") or 0.0) for value in target_phases.values() if value is not None)
    rewrite_trace = rewrite.get("_trace", {}) if rewrite else {}
    model_configs = (config or {}).get("models", {})
    rewrite_wall = float(rewrite_trace.get("wall_time_seconds") or 0.0)
    verifier_wall = float((verifier_usage or {}).get("wall_time_seconds") or 0.0)
    return {
        "arm": arm,
        "generation": f"v{generation}",
        "evaluation_label": "FactoryBench L1-L3 mixed-split evaluation",
        "development_optimization": {key: optimization[key] for key in ("canonical_score", "fixed_cardinality_score", "parse_failures", "format_validity", "by_level", "by_answer_format")},
        "development_validation": {key: validation[key] for key in ("canonical_score", "fixed_cardinality_score", "parse_failures", "format_validity", "by_level", "by_answer_format")},
        "holdout": ({key: holdout[key] for key in ("canonical_score", "fixed_cardinality_score", "parse_failures", "format_validity", "by_level", "by_answer_format")} if holdout else None),
        "changes_vs_v0_holdout": changes,
        "skill_sha256": sha256_file(skill_path),
        "skill_characters": len(skill),
        "skill_tokens": skill_token_length(skill, "gpt-4o-mini"),
        "models": {
            "target": model_configs.get("target", {}).get("model", "gpt-4o-mini"),
            "verifier": model_configs.get("verifier", {}).get("model") if arm == "surrogate" else None,
            "rewriter": model_configs.get("rewriter", {}).get("model", "gpt-5.6-luna") if generation else None,
        },
        "reasoning": {
            "verifier": ({
                "mode": model_configs["verifier"]["reasoning_mode"],
                "effort": model_configs["verifier"]["reasoning_effort"],
            } if arm == "surrogate" and "verifier" in model_configs else None),
            "rewriter": ({
                "mode": model_configs["rewriter"]["reasoning_mode"],
                "effort": model_configs["rewriter"]["reasoning_effort"],
            } if generation and "rewriter" in model_configs else {"mode": "pro", "effort": "xhigh"} if generation else None),
        },
        "target_usage": target_usage,
        "target_estimated_cost_usd": target_cost,
        "verifier_usage": verifier_usage,
        "rewriter_usage": rewrite_trace.get("usage") if rewrite else None,
        "rewriter_estimated_cost_usd": rewrite_trace.get("estimated_cost_usd") if rewrite else None,
        "wall_time_seconds": target_wall + verifier_wall + rewrite_wall,
    }


def copy_v0_for_run(config: dict[str, Any], arm: str, run_dir: Path) -> Path:
    source = resolve_path(config["skills"][f"{arm}_v0"])
    destination = run_dir / "skills" / "skill_v0.txt"
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise ValueError(f"run v0 mismatch: {destination}")
    atomic_write_bytes(destination, source.read_bytes())
    return destination


def manifest_items(manifest: dict[str, Any], source_items: dict[str, Any]) -> list[Any]:
    return [source_items[row["id"]] for row in manifest["items"]]


def run_experiment(config: dict[str, Any], client: Any, *, arm: str, rounds: int, smoke: bool, resume: bool) -> dict[str, Any]:
    mode = "smoke" if smoke else "full"
    run_root = ROOT / "runs" / mode
    trace_root = ROOT / "traces" / mode
    manifests = {key: load_json(manifest_path(config, key)) for key in ("A", "B", "holdout")}
    source = load_source_items(config, manifests)
    arms = selected_arms(arm)

    common_dir = run_root / "common"
    common_skill = copy_v0_for_run(config, arms[0], common_dir)
    common_a = evaluate_items(client, config, manifest_items(manifests["A"], source), common_skill, common_dir / "v0_fold_A.json", trace_root / "common" / "v0_fold_A", resume=resume)
    if smoke:
        common_b = None
        common_holdout = None
    else:
        common_b = evaluate_items(client, config, manifest_items(manifests["B"], source), common_skill, common_dir / "v0_fold_B.json", trace_root / "common" / "v0_fold_B", resume=resume)
        common_holdout = evaluate_items(client, config, manifest_items(manifests["holdout"], source), common_skill, common_dir / "v0_holdout.json", trace_root / "common" / "v0_holdout", resume=resume)

    arm_summaries = {}
    for arm_name in arms:
        arm_dir = run_root / arm_name
        arm_trace = trace_root / arm_name
        current_skill = copy_v0_for_run(config, arm_name, arm_dir)
        if not smoke:
            if common_b is None or common_holdout is None:
                raise RuntimeError("full run requires frozen v0 validation and holdout results")
            v0_metrics = generation_metrics(arm_name, 0, current_skill, common_a, common_b, common_holdout, common_holdout, None, config=config)
            atomic_write_json(arm_dir / "generation_v0_metrics.json", v0_metrics)
        prior_validation: dict[str, Any] | None = None
        generation_records = []
        for schedule in FOLD_SCHEDULE[:rounds]:
            round_number = schedule["round"]
            optimization_fold = schedule["optimization"]
            validation_fold = schedule["validation"]
            if round_number == 1:
                optimization = common_a
            else:
                if prior_validation is None:
                    raise RuntimeError("prior validation result missing for checkpoint reuse")
                optimization = prior_validation
                if optimization["ordered_ids"] != [row["id"] for row in manifests[optimization_fold]["items"]]:
                    raise RuntimeError("prior validation IDs do not match next optimization fold")

            coarse = build_coarse_feedback(optimization)
            verifier_usage = None
            if arm_name == "control":
                feedback = coarse
                atomic_write_json(arm_dir / f"round_{round_number}_coarse_feedback.json", feedback)
            else:
                diagnosis_dir = arm_trace / f"round_{round_number}_verifier"
                diagnoses = run_verifier(client, config, optimization, source, diagnosis_dir, resume=resume)
                verifier_usage = summarize_verifier_usage(diagnoses)
                atomic_write_json(arm_dir / f"round_{round_number}_verifier_usage.json", verifier_usage)
                feedback = aggregate_diagnoses(diagnoses, round_number=round_number, optimization_fold=optimization_fold)
                atomic_write_json(arm_dir / f"round_{round_number}_surrogate_aggregate.json", feedback)

            new_skill, rewrite = rewrite_skill(
                client,
                config,
                arm_name,
                current_skill,
                feedback,
                arm_trace,
                round_number,
                resume=resume,
                skill_dir=arm_dir / "skills",
            )
            validation = evaluate_items(
                client,
                config,
                manifest_items(manifests[validation_fold], source),
                new_skill,
                arm_dir / f"v{round_number}_validation_fold_{validation_fold}.json",
                arm_trace / f"v{round_number}_validation_fold_{validation_fold}",
                resume=resume,
            )
            if smoke:
                holdout = None
            else:
                holdout = evaluate_items(
                    client,
                    config,
                    manifest_items(manifests["holdout"], source),
                    new_skill,
                    arm_dir / f"v{round_number}_holdout.json",
                    arm_trace / f"v{round_number}_holdout",
                    resume=resume,
                )
            metrics = generation_metrics(
                arm_name,
                round_number,
                new_skill,
                optimization,
                validation,
                holdout,
                common_holdout,
                rewrite,
                verifier_usage=verifier_usage,
                config=config,
            )
            atomic_write_json(arm_dir / f"generation_v{round_number}_metrics.json", metrics)
            generation_records.append(metrics)
            current_skill = new_skill
            prior_validation = validation
        arm_summaries[arm_name] = generation_records

    summary = {
        "status": "SMOKE_COMPLETE" if smoke else "EXPERIMENT_COMPLETE",
        "evaluation_label": "FactoryBench L1-L3 mixed-split evaluation",
        "arms": arm_summaries,
        "primary_endpoint": "final v3 holdout performance" if not smoke and rounds == 3 else None,
        "holdout_used_for_selection": False,
    }
    atomic_write_json(ROOT / "results" / f"{mode}_summary.json", summary)
    if not smoke and set(arms) == {"control", "surrogate"}:
        write_direct_comparison(run_root, rounds)
    return summary


def write_direct_comparison(run_root: Path, rounds: int) -> None:
    rows = []
    for generation in range(rounds + 1):
        control = load_json(run_root / "control" / f"generation_v{generation}_metrics.json")
        surrogate = load_json(run_root / "surrogate" / f"generation_v{generation}_metrics.json")
        if generation == 0:
            control_items = surrogate_items = load_json(run_root / "common" / "v0_holdout.json")
        else:
            control_items = load_json(run_root / "control" / f"v{generation}_holdout.json")
            surrogate_items = load_json(run_root / "surrogate" / f"v{generation}_holdout.json")
        rows.append({
            "generation": f"v{generation}",
            "control_holdout": control["holdout"],
            "surrogate_holdout": surrogate["holdout"],
            "fixed_cardinality_delta_surrogate_minus_control": surrogate["holdout"]["fixed_cardinality_score"] - control["holdout"]["fixed_cardinality_score"],
            "control_changes_vs_v0": control["changes_vs_v0_holdout"],
            "surrogate_changes_vs_v0": surrogate["changes_vs_v0_holdout"],
            "paired_items": paired_item_comparison(control_items, surrogate_items),
        })
    payload = {
        "evaluation_label": "FactoryBench L1-L3 mixed-split evaluation",
        "primary_endpoint_generation": f"v{rounds}",
        "selection_used_holdout": False,
        "rows": rows,
    }
    atomic_write_json(ROOT / "results" / "control_vs_surrogate.json", payload)


def paired_item_comparison(control: dict[str, Any], surrogate: dict[str, Any]) -> list[dict[str, Any]]:
    control_ids = [row["id"] for row in control["items"]]
    surrogate_ids = [row["id"] for row in surrogate["items"]]
    if control_ids != surrogate_ids:
        raise ValueError("direct arm comparison requires identical ordered IDs")
    rows = []
    for control_row, surrogate_row in zip(control["items"], surrogate["items"]):
        control_score = float(control_row.get("score") or 0.0) if control_row.get("parse_error") is None else 0.0
        surrogate_score = float(surrogate_row.get("score") or 0.0) if surrogate_row.get("parse_error") is None else 0.0
        rows.append({
            "id": control_row["id"],
            "level": control_row["level"],
            "answer_format": control_row["answer_format"],
            "control_score": control_score,
            "surrogate_score": surrogate_score,
            "surrogate_minus_control": surrogate_score - control_score,
            "control_parse_failure": control_row.get("parse_error") is not None,
            "surrogate_parse_failure": surrogate_row.get("parse_error") is not None,
        })
    return rows


def make_client(config: dict[str, Any]) -> Any:
    from dotenv import load_dotenv
    from openai import OpenAI

    dotenv = resolve_path(config["execution"]["dotenv_path"])
    load_dotenv(dotenv)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable; no API call started")
    return OpenAI(
        max_retries=int(config["execution"]["max_retries"]),
        timeout=float(config["execution"]["timeout_seconds"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="validate and print the plan, then stop before API calls")
    parser.add_argument("--smoke", action="store_true", help="run one L1/L2/L3 smoke path; no holdout calls")
    parser.add_argument("--arm", choices=["control", "surrogate", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute-paid", action="store_true", help="explicitly authorize paid calls after reviewing a dry-run")
    parser.add_argument("--plan-sha256", help="exact digest printed by the reviewed dry-run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = preflight(config, verify_sources=True)
    atomic_write_json(ROOT / "logs" / "latest_preflight.json", report)
    plan = build_run_plan(config, arm=args.arm, rounds=args.rounds, smoke=args.smoke)
    atomic_write_json(ROOT / "logs" / "latest_run_plan.json", plan)
    plan_name = f"dry_run_{plan['mode']}_{args.arm}_{args.rounds}_rounds.json"
    atomic_write_json(ROOT / "logs" / plan_name, plan)
    print_run_plan(plan)
    if args.dry_run and args.execute_paid:
        raise SystemExit("--dry-run and --execute-paid are mutually exclusive")
    if not args.execute_paid:
        print("RUN PLAN only: stopped before paid API calls")
        return
    authorize_paid_execution(plan, execute_paid=args.execute_paid, supplied_plan_sha256=args.plan_sha256)
    client = make_client(config)
    result = run_experiment(config, client, arm=args.arm, rounds=args.rounds, smoke=args.smoke, resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
