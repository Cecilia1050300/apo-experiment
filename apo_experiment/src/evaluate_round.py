from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def extract_prompt_candidates(data: Any) -> list[dict[str, str]]:
    """
    支援下列 Prompt JSON 格式：

    1. Round 2 格式：
       {
         "prompts": [
           {"prompt_id": "...", "prompt": "...", "strategy": "..."}
         ]
       }

    2. Round 1 / result-like 格式：
       {
         "results": [
           {"prompt_id": "...", "prompt": "..."}
         ]
       }

    3. 純陣列：
       [
         {"prompt_id": "...", "prompt": "..."},
         "Prompt text..."
       ]
    """
    if isinstance(data, dict):
        if isinstance(data.get("prompts"), list):
            raw_prompts = data["prompts"]
        elif isinstance(data.get("results"), list):
            raw_prompts = data["results"]
        else:
            raise ValueError(
                "Prompt 檔案找不到 prompts 或 results 陣列。"
            )
    elif isinstance(data, list):
        raw_prompts = data
    else:
        raise ValueError("Prompt 檔案最外層必須是 JSON object 或 array。")

    candidates: list[dict[str, str]] = []

    for index, item in enumerate(raw_prompts, start=1):
        if isinstance(item, str):
            prompt_text = item.strip()
            prompt_id = f"candidate_{index}"
            strategy = ""
        elif isinstance(item, dict):
            prompt_text = str(
                item.get("prompt")
                or item.get("instruction")
                or item.get("text")
                or ""
            ).strip()

            prompt_id = str(
                item.get("prompt_id")
                or item.get("id")
                or f"candidate_{index}"
            ).strip()

            strategy = str(item.get("strategy") or "").strip()
        else:
            raise ValueError(f"第 {index} 個 Prompt 格式不正確。")

        if not prompt_text:
            raise ValueError(f"第 {index} 個 Prompt 內容為空。")

        candidates.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "strategy": strategy,
            }
        )

    if not candidates:
        raise ValueError("候選 Prompt 不可為空。")

    return candidates


def extract_dataset_split(data: Any, split: str) -> list[dict[str, Any]]:
    """
    支援常見資料集格式：

    1.
       {
         "optimization": [...],
         "validation": [...],
         "final_test": [...]
       }

    2.
       {
         "splits": {
           "optimization": [...]
         }
       }

    3.
       {
         "data": {
           "optimization": [...]
         }
       }

    4. 直接是一個 records array。
    """
    records: Any = None

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        if isinstance(data.get(split), list):
            records = data[split]

        elif isinstance(data.get("splits"), dict):
            records = data["splits"].get(split)

        elif isinstance(data.get("data"), dict):
            records = data["data"].get(split)

        elif (
            data.get("dataset_split") == split
            and isinstance(data.get("records"), list)
        ):
            records = data["records"]

    if not isinstance(records, list):
        available: list[str] = []

        if isinstance(data, dict):
            available.extend(
                key for key, value in data.items() if isinstance(value, list)
            )

            for container_key in ("splits", "data"):
                container = data.get(container_key)
                if isinstance(container, dict):
                    available.extend(
                        f"{container_key}.{key}"
                        for key, value in container.items()
                        if isinstance(value, list)
                    )

        raise ValueError(
            f"找不到資料集 split：{split}。"
            f"可見的 list 欄位：{available or '無'}"
        )

    normalized_records: list[dict[str, Any]] = []

    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"資料集第 {index} 筆不是 JSON object。")

        input_text = (
            item.get("input")
            or item.get("input_text")
            or item.get("question")
            or item.get("source")
        )

        expected = (
            item.get("expected_outputs")
            or item.get("expected_output")
            or item.get("output")
            or item.get("answer")
            or item.get("target")
        )

        if input_text is None:
            raise ValueError(f"資料集第 {index} 筆缺少 input。")

        if expected is None:
            raise ValueError(f"資料集第 {index} 筆缺少 expected output。")

        if isinstance(expected, list):
            expected_outputs = [str(value) for value in expected]
        else:
            expected_outputs = [str(expected)]

        normalized_records.append(
            {
                "input": str(input_text),
                "expected_outputs": expected_outputs,
                "source_record": item,
            }
        )

    return normalized_records


def clean_model_output(text: str) -> str:
    cleaned = text.strip()

    # 移除完整 code fence。
    fenced = re.fullmatch(
        r"```(?:text|txt)?\s*(.*?)\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        cleaned = fenced.group(1).strip()

    # 移除包住整個答案的單引號或雙引號。
    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {"'", '"'}
    ):
        cleaned = cleaned[1:-1].strip()

    return cleaned


def normalize_for_match(text: str, strict_exact: bool) -> str:
    cleaned = clean_model_output(text)

    if strict_exact:
        return cleaned

    # 預設忽略大小寫與首尾空白，但不任意移除標點或改寫內容。
    return cleaned.casefold()


def is_correct_prediction(
    prediction: str,
    expected_outputs: list[str],
    strict_exact: bool,
) -> bool:
    normalized_prediction = normalize_for_match(
        prediction,
        strict_exact=strict_exact,
    )

    return any(
        normalized_prediction
        == normalize_for_match(expected, strict_exact=strict_exact)
        for expected in expected_outputs
    )


def get_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def read_value(name: str) -> int:
        if isinstance(usage, dict):
            value = usage.get(name, 0)
        else:
            value = getattr(usage, name, 0)

        return int(value or 0)

    return {
        "input_tokens": read_value("input_tokens"),
        "output_tokens": read_value("output_tokens"),
        "total_tokens": read_value("total_tokens"),
    }


def run_prediction(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    input_text: str,
    max_retries: int,
    retry_wait_seconds: float,
    reasoning_effort: str | None,
) -> tuple[str, dict[str, int], float]:
    request: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": input_text,
                    }
                ],
            },
        ],
    }

    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        started_at = time.perf_counter()

        try:
            response = client.responses.create(**request)
            elapsed_seconds = time.perf_counter() - started_at
            output_text = str(response.output_text or "").strip()

            if not output_text:
                raise RuntimeError("模型回傳空字串。")

            return output_text, get_usage(response), elapsed_seconds

        except Exception as exc:
            last_error = exc

            if attempt >= max_retries:
                break

            wait_seconds = retry_wait_seconds * (attempt + 1)
            print(
                f"[WARN] API 呼叫失敗，"
                f"{wait_seconds:.1f} 秒後重試 "
                f"({attempt + 1}/{max_retries})：{exc}"
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"API 呼叫在 {max_retries + 1} 次嘗試後仍失敗：{last_error}"
    )


def infer_task_name(
    prompt_data: Any,
    dataset_data: Any,
    fallback: str,
) -> str:
    for data in (prompt_data, dataset_data):
        if isinstance(data, dict) and data.get("task"):
            return str(data["task"])

    return fallback


def infer_model(
    args_model: str | None,
    prompt_data: Any,
) -> str:
    if args_model:
        return args_model

    if isinstance(prompt_data, dict):
        for key in ("target_model", "model"):
            if prompt_data.get(key):
                return str(prompt_data[key])

    env_model = os.getenv("TARGET_MODEL")
    if env_model:
        return env_model

    return "gpt-5.5"


def derive_failed_output_path(output_path: Path) -> Path:
    stem = output_path.stem

    if stem.endswith("_results"):
        stem = stem[: -len("_results")]

    return output_path.with_name(f"{stem}_failed_cases.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "通用 APO Prompt 評估程式："
            "接受任意數量候選 Prompt，使用指定資料集 split 評估。"
        )
    )

    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="候選 Prompt JSON 路徑。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/antonyms_dataset_split.json"),
        help="資料集切分 JSON 路徑。",
    )
    parser.add_argument(
        "--split",
        default="optimization",
        help="要使用的資料集 split，例如 optimization、validation、final_test。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="完整評估結果輸出 JSON 路徑。",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=None,
        help="最佳 Prompt 失敗案例輸出路徑；未指定時自動產生。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Target Model。優先順序："
            "--model > Prompt JSON target_model > TARGET_MODEL 環境變數 > gpt-5.5。"
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="只評估每個 Prompt 的前 N 筆；預設全部。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="每次 API 呼叫失敗後的最大重試次數。",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=2.0,
        help="重試等待基準秒數。",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=0.0,
        help="每次成功 API 呼叫後等待秒數。",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help="選填 reasoning effort；不指定時不傳 reasoning 參數。",
    )
    parser.add_argument(
        "--strict-exact",
        action="store_true",
        help="嚴格 Exact Match：區分大小寫。預設忽略大小寫。",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="選填：驗證候選 Prompt 數量；不指定則接受任意非空數量。",
    )

    args = parser.parse_args()

    if args.max_cases is not None and args.max_cases <= 0:
        raise ValueError("--max-cases 必須大於 0。")

    if args.max_retries < 0:
        raise ValueError("--max-retries 不可小於 0。")

    load_dotenv()

    prompt_data = load_json(args.prompts)
    dataset_data = load_json(args.dataset)

    candidates = extract_prompt_candidates(prompt_data)
    records = extract_dataset_split(dataset_data, args.split)

    if args.expected_count is not None and len(candidates) != args.expected_count:
        raise ValueError(
            f"預期 {args.expected_count} 個候選 Prompt，"
            f"實際為 {len(candidates)} 個。"
        )

    if args.max_cases is not None:
        records = records[: args.max_cases]

    if not records:
        raise ValueError("評估資料不可為空。")

    model = infer_model(args.model, prompt_data)
    task = infer_task_name(prompt_data, dataset_data, fallback="unknown")
    failed_output_path = (
        args.failed_output
        if args.failed_output is not None
        else derive_failed_output_path(args.output)
    )

    client = OpenAI()

    print("=" * 72)
    print(f"[INFO] Task            : {task}")
    print(f"[INFO] Target model    : {model}")
    print(f"[INFO] Dataset split   : {args.split}")
    print(f"[INFO] Candidate count : {len(candidates)}")
    print(f"[INFO] Cases / prompt  : {len(records)}")
    print(f"[INFO] Total requests  : {len(candidates) * len(records)}")
    print("=" * 72)

    all_results: list[dict[str, Any]] = []

    for prompt_index, candidate in enumerate(candidates, start=1):
        prompt_id = candidate["prompt_id"]
        prompt_text = candidate["prompt"]
        strategy = candidate["strategy"]

        print(
            f"\n[PROMPT {prompt_index}/{len(candidates)}] "
            f"{prompt_id}"
        )

        prompt_records: list[dict[str, Any]] = []
        correct_count = 0
        usage_total = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        latency_total_seconds = 0.0

        for case_index, record in enumerate(records, start=1):
            input_text = record["input"]
            expected_outputs = record["expected_outputs"]

            try:
                prediction, usage, elapsed_seconds = run_prediction(
                    client,
                    model=model,
                    prompt=prompt_text,
                    input_text=input_text,
                    max_retries=args.max_retries,
                    retry_wait_seconds=args.retry_wait,
                    reasoning_effort=args.reasoning_effort,
                )

                correct = is_correct_prediction(
                    prediction,
                    expected_outputs,
                    strict_exact=args.strict_exact,
                )
                error_message = None

            except Exception as exc:
                prediction = ""
                usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
                elapsed_seconds = 0.0
                correct = False
                error_message = str(exc)

            if correct:
                correct_count += 1

            for key in usage_total:
                usage_total[key] += usage[key]

            latency_total_seconds += elapsed_seconds

            prompt_records.append(
                {
                    "input": input_text,
                    "expected_outputs": expected_outputs,
                    "prediction": prediction,
                    "correct": correct,
                    "error": error_message,
                    "latency_seconds": round(elapsed_seconds, 4),
                    "usage": usage,
                }
            )

            status = "PASS" if correct else "FAIL"
            print(
                f"  [{case_index:>3}/{len(records)}] "
                f"{status} | "
                f"input={input_text!r} | "
                f"prediction={prediction!r}"
            )

            if args.request_interval > 0:
                time.sleep(args.request_interval)

        total = len(prompt_records)
        accuracy = correct_count / total if total else 0.0

        all_results.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "strategy": strategy,
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total": total,
                "failed_count": total - correct_count,
                "average_latency_seconds": (
                    round(latency_total_seconds / total, 4)
                    if total
                    else 0.0
                ),
                "total_latency_seconds": round(
                    latency_total_seconds,
                    4,
                ),
                "usage": usage_total,
                "records": prompt_records,
            }
        )

        print(
            f"[RESULT] {prompt_id}: "
            f"{correct_count}/{total} = {accuracy:.2%}"
        )

    # 依 Accuracy、正確數、較少 Token、較低延遲排序。
    ranked_results = sorted(
        all_results,
        key=lambda item: (
            -item["accuracy"],
            -item["correct_count"],
            item["usage"]["total_tokens"],
            item["average_latency_seconds"],
            item["prompt_id"],
        ),
    )

    best = ranked_results[0]
    best_accuracy = best["accuracy"]

    tied_best_prompt_ids = [
        item["prompt_id"]
        for item in ranked_results
        if item["accuracy"] == best_accuracy
    ]

    output_data = {
        "task": task,
        "target_model": model,
        "dataset_split": args.split,
        "prompt_source": str(args.prompts),
        "dataset_source": str(args.dataset),
        "max_cases": args.max_cases,
        "candidate_count": len(candidates),
        "evaluation": {
            "primary_metric": "exact_match",
            "strict_exact": args.strict_exact,
            "normalization": (
                "strip_only"
                if args.strict_exact
                else "strip_and_casefold"
            ),
        },
        "results": ranked_results,
        "best_prompt_id": best["prompt_id"],
        "best_prompt": best["prompt"],
        "best_accuracy": best["accuracy"],
        "best_correct_count": best["correct_count"],
        "best_total": best["total"],
        "tied_best_prompt_ids": tied_best_prompt_ids,
    }

    failed_cases = [
        {
            "input": record["input"],
            "expected_outputs": record["expected_outputs"],
            "prediction": record["prediction"],
            "correct": record["correct"],
            "error": record["error"],
        }
        for record in best["records"]
        if not record["correct"]
    ]

    failed_output_data = {
        "task": task,
        "target_model": model,
        "dataset_split": args.split,
        "best_prompt_id": best["prompt_id"],
        "best_prompt": best["prompt"],
        "best_accuracy": best["accuracy"],
        "failed_case_count": len(failed_cases),
        "failed_cases": failed_cases,
    }

    save_json(args.output, output_data)
    save_json(failed_output_path, failed_output_data)

    print("\n" + "=" * 72)
    print("[DONE] 評估完成")
    print(f"[BEST] {best['prompt_id']}: {best_accuracy:.2%}")

    if len(tied_best_prompt_ids) > 1:
        print(
            "[TIE] 並列最佳 Prompt："
            + ", ".join(tied_best_prompt_ids)
        )

    print(f"[SAVE] Results      : {args.output}")
    print(f"[SAVE] Failed cases : {failed_output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()