from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# ---------- Common helpers ----------

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S | re.I)
    if m:
        return json.loads(m.group(1))

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("模型輸出不是有效 JSON")


def client() -> OpenAI:
    load_dotenv()
    return OpenAI()


def model_name(cli_model: str | None, env_name: str) -> str:
    return cli_model or os.getenv(env_name) or os.getenv("TARGET_MODEL") or "gpt-5.5"


def prompt_candidates(data: Any) -> list[dict[str, str]]:
    if isinstance(data, dict):
        raw = data.get("prompts") or data.get("results")
    elif isinstance(data, list):
        raw = data
    else:
        raw = None

    if not isinstance(raw, list) or not raw:
        raise ValueError("Prompt JSON 必須包含非空 prompts/results 陣列")

    out = []
    for i, item in enumerate(raw, 1):
        if isinstance(item, str):
            out.append({
                "prompt_id": f"candidate_{i}",
                "strategy": "",
                "prompt": item.strip(),
            })
        elif isinstance(item, dict):
            p = str(item.get("prompt") or item.get("instruction") or "").strip()
            if not p:
                raise ValueError(f"第 {i} 個 Prompt 為空")
            out.append({
                "prompt_id": str(item.get("prompt_id") or item.get("id") or f"candidate_{i}"),
                "strategy": str(item.get("strategy") or ""),
                "prompt": p,
            })
        else:
            raise ValueError(f"第 {i} 個 Prompt 格式不支援")
    return out


def get_split(dataset: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = dataset.get("splits", {}).get(split) if isinstance(dataset.get("splits"), dict) else dataset.get(split)
    if not isinstance(rows, list):
        raise ValueError(f"找不到 split: {split}")
    return rows


def clean_answer(text: str) -> str:
    t = text.strip()
    m = re.fullmatch(r"```(?:text|txt)?\s*(.*?)\s*```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in {'"', "'"}:
        t = t[1:-1].strip()
    return t


def exact_match(prediction: str, expected: list[str]) -> bool:
    p = clean_answer(prediction).casefold()
    return any(p == clean_answer(x).casefold() for x in expected)


def format_ok(prediction: str) -> bool:
    return clean_answer(prediction).casefold() in {"yes", "no"}


def usage_dict(response: Any) -> dict[str, int]:
    u = getattr(response, "usage", None)
    if u is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    def val(name: str) -> int:
        if isinstance(u, dict):
            return int(u.get(name, 0) or 0)
        return int(getattr(u, name, 0) or 0)

    return {
        "input_tokens": val("input_tokens"),
        "output_tokens": val("output_tokens"),
        "total_tokens": val("total_tokens"),
    }


# ---------- 0. Inspect BIG-bench task ----------

def cmd_inspect(args: argparse.Namespace) -> None:
    data = load_json(args.source)
    examples = data.get("examples", [])
    print("name            :", data.get("name"))
    print("description     :", data.get("description"))
    print("preferred_score :", data.get("preferred_score"))
    print("metrics         :", data.get("metrics"))
    print("example_count   :", len(examples))
    for i, ex in enumerate(examples[:args.show], 1):
        print("\n" + "=" * 80)
        print(f"Example {i}")
        print(ex.get("input"))
        print("target_scores:", ex.get("target_scores"))
        print("target:", ex.get("target"))


# ---------- 1. Prepare causal dataset ----------

def expected_outputs(example: dict[str, Any]) -> list[str]:
    if isinstance(example.get("target_scores"), dict):
        scores = example["target_scores"]
        best = max(float(v) for v in scores.values())
        return [str(k) for k, v in scores.items() if float(v) == best]

    target = example.get("target")
    if isinstance(target, list):
        return [str(x) for x in target]
    if target is not None:
        return [str(target)]
    raise ValueError("Example 無 target_scores / target")


def cmd_prepare(args: argparse.Namespace) -> None:
    task = load_json(args.source)
    examples = task.get("examples", [])
    if not isinstance(examples, list) or not examples:
        raise ValueError("task.json 找不到 examples")

    records = []
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict) or "input" not in ex:
            continue
        records.append({
            "id": f"causal_judgment_{i:04d}",
            "input": str(ex["input"]),
            "expected_outputs": expected_outputs(ex),
            "source_index": i,
        })

    need = args.optimization_size + args.final_test_size
    if len(records) < need:
        raise ValueError(f"資料只有 {len(records)} 筆，但要求 {need} 筆")

    rng = random.Random(args.seed)
    rng.shuffle(records)

    output = {
        "task": task.get("name", "causal_judgment"),
        "task_description": task.get("description", "Answer questions about causal attribution"),
        "source_task_json": str(args.source),
        "source_example_count": len(records),
        "seed": args.seed,
        "splits": {
            "optimization": records[:args.optimization_size],
            "final_test": records[args.optimization_size:need],
        },
    }
    save_json(args.output, output)
    print(f"[DONE] {args.output}")
    print(f"optimization={args.optimization_size}, final_test={args.final_test_size}, unused={len(records)-need}")


# ---------- 2. Round 1 zero-shot candidate generation ----------

def cmd_gen_r1(args: argparse.Namespace) -> None:
    data = load_json(args.dataset)
    model = model_name(args.model, "PROMPT_OPTIMIZER_MODEL")

    meta = f"""
You are generating initial candidate system prompts.

Task: {data.get("task")}
Task description: {data.get("task_description")}

The target model receives one causal-attribution scenario at a time.
The gold output is exactly one label: Yes or No.

Generate exactly {args.count} meaningfully different candidate system prompts.

Requirements:
- Do not use QA demonstrations.
- Do not copy any dataset examples.
- Require causal reasoning, not keyword matching.
- Require exactly "Yes" or "No" and no explanation.
- Keep prompts concise.
- Vary reasoning emphasis rather than only paraphrasing.

Return only JSON:
{{
  "prompts": [
    {{
      "prompt_id": "candidate_1",
      "strategy": "short strategy",
      "prompt": "..."
    }}
  ]
}}
""".strip()

    resp = client().responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "Return valid JSON only."}]},
            {"role": "user", "content": [{"type": "input_text", "text": meta}]},
        ],
    )
    parsed = extract_json(resp.output_text)
    prompts = parsed.get("prompts", [])
    if len(prompts) != args.count:
        raise ValueError(f"預期 {args.count} 個 Prompt，實際 {len(prompts)}")

    save_json(args.output, {
        "task": data.get("task"),
        "task_description": data.get("task_description"),
        "generation_method": "zero_shot_candidate_generation",
        "optimizer_model": model,
        "candidate_count": len(prompts),
        "prompts": prompts,
    })
    print(f"[DONE] {args.output}")


# ---------- 3. Evaluate ----------

def call_target(c: OpenAI, model: str, prompt: str, user_input: str, retries: int):
    last = None
    for attempt in range(retries + 1):
        started = time.perf_counter()
        try:
            resp = c.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                    {"role": "user", "content": [{"type": "input_text", "text": user_input}]},
                ],
            )
            latency = time.perf_counter() - started
            text = str(resp.output_text or "").strip()
            if not text:
                raise RuntimeError("empty output")
            return text, usage_dict(resp), latency
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(last)


def cmd_eval(args: argparse.Namespace) -> None:
    p_data = load_json(args.prompts)
    dataset = load_json(args.dataset)
    prompts = prompt_candidates(p_data)
    rows = get_split(dataset, args.split)
    if args.max_cases:
        rows = rows[:args.max_cases]

    model = model_name(args.model, "TARGET_MODEL")
    c = client()
    results = []

    print(f"task={dataset.get('task')}, split={args.split}, model={model}")
    print(f"{len(prompts)} prompts x {len(rows)} cases = {len(prompts)*len(rows)} requests")

    for pi, p in enumerate(prompts, 1):
        records = []
        correct_n = fmt_n = 0
        latency_total = 0.0
        token_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        print(f"\n[PROMPT {pi}/{len(prompts)}] {p['prompt_id']}")
        for ri, row in enumerate(rows, 1):
            expected = [str(x) for x in row.get("expected_outputs", [])]
            error = None
            try:
                pred, usage, latency = call_target(c, model, p["prompt"], row["input"], args.retries)
            except Exception as exc:
                pred, latency = "", 0.0
                usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                error = str(exc)

            ok = exact_match(pred, expected)
            fmt = format_ok(pred)
            correct_n += int(ok)
            fmt_n += int(fmt)
            latency_total += latency
            for k in token_total:
                token_total[k] += usage[k]

            records.append({
                "id": row.get("id"),
                "input": row["input"],
                "expected_outputs": expected,
                "prediction": pred,
                "correct": ok,
                "format_compliant": fmt,
                "error": error,
                "latency_seconds": round(latency, 4),
                "usage": usage,
            })
            print(f"  [{ri:>3}/{len(rows)}] {'PASS' if ok else 'FAIL'} expected={expected} pred={pred!r}")

        total = len(records)
        results.append({
            "prompt_id": p["prompt_id"],
            "strategy": p.get("strategy", ""),
            "prompt": p["prompt"],
            "accuracy": correct_n / total,
            "correct_count": correct_n,
            "total": total,
            "failed_count": total - correct_n,
            "format_compliance": fmt_n / total,
            "format_compliant_count": fmt_n,
            "average_latency_seconds": round(latency_total / total, 4),
            "total_latency_seconds": round(latency_total, 4),
            "usage": token_total,
            "records": records,
        })

    ranked = sorted(
        results,
        key=lambda x: (
            -x["accuracy"],
            -x["format_compliance"],
            x["usage"]["total_tokens"],
            x["average_latency_seconds"],
        ),
    )
    best = ranked[0]

    save_json(args.output, {
        "task": dataset.get("task"),
        "target_model": model,
        "dataset_split": args.split,
        "evaluation": {
            "primary_metric": "exact_match_accuracy",
            "secondary_metrics": ["format_compliance", "total_tokens", "average_latency_seconds"],
        },
        "results": ranked,
        "best_prompt_id": best["prompt_id"],
        "best_prompt": best["prompt"],
        "best_accuracy": best["accuracy"],
    })

    failed_path = args.failed_output
    if failed_path is None:
        stem = args.output.stem
        if stem.endswith("_results"):
            stem = stem[:-8]
        failed_path = args.output.with_name(stem + "_failed_cases.json")

    failed = [
        {
            "id": r["id"],
            "input": r["input"],
            "expected_outputs": r["expected_outputs"],
            "prediction": r["prediction"],
            "format_compliant": r["format_compliant"],
            "error": r["error"],
        }
        for r in best["records"] if not r["correct"]
    ]
    save_json(failed_path, {
        "task": dataset.get("task"),
        "dataset_split": args.split,
        "best_prompt_id": best["prompt_id"],
        "best_prompt": best["prompt"],
        "best_accuracy": best["accuracy"],
        "failed_case_count": len(failed),
        "failed_cases": failed,
    })
    print(f"\n[BEST] {best['prompt_id']} {best['accuracy']:.2%}")
    print(f"[SAVE] {args.output}")
    print(f"[SAVE] {failed_path}")


# ---------- 4. Round 2 error attribution + rewriting ----------

def cmd_gen_r2(args: argparse.Namespace) -> None:
    r1 = load_json(args.round_1_results)
    failed = load_json(args.failed_cases)
    model = model_name(args.model, "PROMPT_OPTIMIZER_MODEL")

    history = [{
        "prompt_id": x["prompt_id"],
        "accuracy": x["accuracy"],
        "format_compliance": x.get("format_compliance"),
        "prompt": x["prompt"],
    } for x in r1["results"]]

    failure_sample = failed.get("failed_cases", [])[:args.max_failures]

    meta = f"""
You are optimizing a system prompt for causal_judgment.

Goal:
Given a scenario and a causal/intentionality question, return exactly Yes or No.

Round 1 prompt and score history:
{json.dumps(history, ensure_ascii=False, indent=2)}

Failed cases of the current best prompt:
{json.dumps(failure_sample, ensure_ascii=False, indent=2)}

First identify general failure patterns. Then generate exactly {args.count} new candidates.

Rules:
- Never copy exact failed QA pairs into a prompt.
- Do not memorize names/entities.
- Prefer generalizable reasoning principles.
- Candidate 1 must be a minimal-edit revision of the Round 1 best prompt.
- Candidate 2 should focus on causal contribution, necessity/sufficiency, and counterfactual reasoning.
- Candidate 3 should focus on intentionality, norms, side effects, and causal attribution when relevant.
- Every candidate must require exactly "Yes" or "No", no explanation.
- Keep prompts concise.

Return only JSON:
{{
  "error_attribution": [
    {{
      "error_type": "...",
      "evidence": "...",
      "revision_direction": "..."
    }}
  ],
  "prompts": [
    {{
      "prompt_id": "round_2_candidate_1",
      "strategy": "...",
      "change_type": "minimal_edit",
      "prompt": "..."
    }}
  ]
}}
""".strip()

    resp = client().responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "Return valid JSON only."}]},
            {"role": "user", "content": [{"type": "input_text", "text": meta}]},
        ],
    )
    parsed = extract_json(resp.output_text)
    prompts = parsed.get("prompts", [])
    if len(prompts) != args.count:
        raise ValueError(f"預期 {args.count} 個 Prompt，實際 {len(prompts)}")

    save_json(args.output, {
        "task": r1.get("task"),
        "generation_method": "failure_attribution_plus_history_guided_rewriting",
        "method_note": "ProTeGi/OPRO-inspired hybrid; not an exact reproduction.",
        "optimizer_model": model,
        "source_best_prompt_id": r1.get("best_prompt_id"),
        "source_best_accuracy": r1.get("best_accuracy"),
        "error_attribution": parsed.get("error_attribution", []),
        "prompts": prompts,
    })
    print(f"[DONE] {args.output}")


# ---------- 5. Regression comparison ----------

def find_best(data: dict[str, Any]) -> dict[str, Any]:
    best_id = data["best_prompt_id"]
    for r in data["results"]:
        if r["prompt_id"] == best_id:
            return r
    return data["results"][0]


def cmd_compare(args: argparse.Namespace) -> None:
    d1, d2 = load_json(args.round_1), load_json(args.round_2)
    r1, r2 = find_best(d1), find_best(d2)
    a = {str(x["id"]): x for x in r1["records"]}
    b = {str(x["id"]): x for x in r2["records"]}

    fixed, regressed, still_failed, still_correct = [], [], [], []
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        item = {
            "id": key,
            "input": y["input"],
            "expected_outputs": y["expected_outputs"],
            "round_1_prediction": x["prediction"],
            "round_2_prediction": y["prediction"],
            "round_1_correct": bool(x["correct"]),
            "round_2_correct": bool(y["correct"]),
        }
        if not x["correct"] and y["correct"]:
            fixed.append(item)
        elif x["correct"] and not y["correct"]:
            regressed.append(item)
        elif not x["correct"] and not y["correct"]:
            still_failed.append(item)
        else:
            still_correct.append(item)

    out = {
        "task": d2.get("task"),
        "dataset_split": d2.get("dataset_split"),
        "round_1": {"best_prompt_id": r1["prompt_id"], "accuracy": r1["accuracy"]},
        "round_2": {"best_prompt_id": r2["prompt_id"], "accuracy": r2["accuracy"]},
        "comparison": {
            "accuracy_delta": round(r2["accuracy"] - r1["accuracy"], 6),
            "fixed_case_count": len(fixed),
            "newly_failed_case_count": len(regressed),
            "still_failed_case_count": len(still_failed),
            "still_correct_case_count": len(still_correct),
        },
        "fixed_cases": fixed,
        "newly_failed_cases": regressed,
        "still_failed_cases": still_failed,
    }
    save_json(args.output, out)
    print(json.dumps(out["comparison"], ensure_ascii=False, indent=2))
    print(f"[SAVE] {args.output}")


# ---------- 6. Freeze final comparison prompts ----------

def cmd_build_final(args: argparse.Namespace) -> None:
    r1, r2 = load_json(args.round_1_results), load_json(args.round_2_results)
    out = {
        "task": "causal_judgment",
        "note": "Freeze this file before final_test. Do not rewrite using final-test failures.",
        "prompts": [
            {
                "prompt_id": "manual_seed",
                "strategy": "manual baseline",
                "prompt": (
                    "Determine whether the event or action asked about caused or intentionally "
                    "brought about the stated outcome in the scenario. Output exactly Yes or No."
                ),
            },
            {
                "prompt_id": "round_1_best",
                "strategy": "best Round 1 prompt",
                "prompt": r1["best_prompt"],
            },
            {
                "prompt_id": "round_2_best",
                "strategy": "best Round 2 prompt",
                "prompt": r2["best_prompt"],
            },
        ],
    }
    save_json(args.output, out)
    print(f"[DONE] {args.output}")


# ---------- CLI ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal Judgment APO experiment")
    sub = p.add_subparsers(dest="command", required=True)

    x = sub.add_parser("inspect")
    x.add_argument("--source", type=Path, required=True)
    x.add_argument("--show", type=int, default=3)
    x.set_defaults(func=cmd_inspect)

    x = sub.add_parser("prepare")
    x.add_argument("--source", type=Path, required=True)
    x.add_argument("--output", type=Path, default=Path("results/causal_judgment_dataset_split.json"))
    x.add_argument("--optimization-size", type=int, default=50)
    x.add_argument("--final-test-size", type=int, default=50)
    x.add_argument("--seed", type=int, default=42)
    x.set_defaults(func=cmd_prepare)

    x = sub.add_parser("gen-r1")
    x.add_argument("--dataset", type=Path, default=Path("results/causal_judgment_dataset_split.json"))
    x.add_argument("--output", type=Path, default=Path("prompts/causal_judgment_round_1_prompts.json"))
    x.add_argument("--count", type=int, default=5)
    x.add_argument("--model", default=None)
    x.set_defaults(func=cmd_gen_r1)

    x = sub.add_parser("eval")
    x.add_argument("--prompts", type=Path, required=True)
    x.add_argument("--dataset", type=Path, required=True)
    x.add_argument("--split", default="optimization")
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--failed-output", type=Path, default=None)
    x.add_argument("--model", default=None)
    x.add_argument("--max-cases", type=int, default=None)
    x.add_argument("--retries", type=int, default=2)
    x.set_defaults(func=cmd_eval)

    x = sub.add_parser("gen-r2")
    x.add_argument("--round-1-results", type=Path, default=Path("results/causal_judgment_round_1_results.json"))
    x.add_argument("--failed-cases", type=Path, default=Path("results/causal_judgment_round_1_failed_cases.json"))
    x.add_argument("--output", type=Path, default=Path("prompts/causal_judgment_round_2_prompts.json"))
    x.add_argument("--count", type=int, default=3)
    x.add_argument("--max-failures", type=int, default=20)
    x.add_argument("--model", default=None)
    x.set_defaults(func=cmd_gen_r2)

    x = sub.add_parser("compare")
    x.add_argument("--round-1", type=Path, default=Path("results/causal_judgment_round_1_results.json"))
    x.add_argument("--round-2", type=Path, default=Path("results/causal_judgment_round_2_results.json"))
    x.add_argument("--output", type=Path, default=Path("results/causal_judgment_round_1_vs_round_2_regression.json"))
    x.set_defaults(func=cmd_compare)

    x = sub.add_parser("build-final")
    x.add_argument("--round-1-results", type=Path, default=Path("results/causal_judgment_round_1_results.json"))
    x.add_argument("--round-2-results", type=Path, default=Path("results/causal_judgment_round_2_results.json"))
    x.add_argument("--output", type=Path, default=Path("prompts/causal_judgment_final_comparison_prompts.json"))
    x.set_defaults(func=cmd_build_final)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
