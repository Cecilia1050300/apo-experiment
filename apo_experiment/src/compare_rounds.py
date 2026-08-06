from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON 最外層必須是 object：{path}")

    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_best_result(data: dict[str, Any]) -> dict[str, Any]:
    best_prompt_id = data.get("best_prompt_id")
    results = data.get("results")

    if not isinstance(results, list):
        raise ValueError("結果檔缺少 results 陣列。")

    for item in results:
        if (
            isinstance(item, dict)
            and item.get("prompt_id") == best_prompt_id
        ):
            return item

    if results and isinstance(results[0], dict):
        return results[0]

    raise ValueError("找不到最佳 Prompt 的完整結果。")


def records_by_input(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = result.get("records")

    if not isinstance(records, list):
        raise ValueError("最佳 Prompt 結果缺少 records 陣列。")

    output: dict[str, dict[str, Any]] = {}

    for item in records:
        if not isinstance(item, dict):
            continue

        input_text = str(item.get("input", ""))
        if not input_text:
            continue

        if input_text in output:
            raise ValueError(f"發現重複 input，無法唯一比對：{input_text}")

        output[input_text] = item

    return output


def compact_case(
    input_text: str,
    old_record: dict[str, Any],
    new_record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input": input_text,
        "expected_outputs": new_record.get(
            "expected_outputs",
            old_record.get("expected_outputs", []),
        ),
        "round_1_prediction": old_record.get("prediction"),
        "round_2_prediction": new_record.get("prediction"),
        "round_1_correct": bool(old_record.get("correct")),
        "round_2_correct": bool(new_record.get("correct")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比較 Round 1 與 Round 2 最佳 Prompt 的逐筆結果。"
    )
    parser.add_argument(
        "--round-1",
        type=Path,
        default=Path("results/round_1_results.json"),
    )
    parser.add_argument(
        "--round-2",
        type=Path,
        default=Path("results/round_2_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/round_1_vs_round_2_regression.json"
        ),
    )
    args = parser.parse_args()

    round_1_data = load_json(args.round_1)
    round_2_data = load_json(args.round_2)

    round_1_best = get_best_result(round_1_data)
    round_2_best = get_best_result(round_2_data)

    round_1_records = records_by_input(round_1_best)
    round_2_records = records_by_input(round_2_best)

    common_inputs = sorted(
        set(round_1_records) & set(round_2_records)
    )
    only_round_1 = sorted(
        set(round_1_records) - set(round_2_records)
    )
    only_round_2 = sorted(
        set(round_2_records) - set(round_1_records)
    )

    fixed_cases: list[dict[str, Any]] = []
    newly_failed_cases: list[dict[str, Any]] = []
    still_failed_cases: list[dict[str, Any]] = []
    still_correct_cases: list[dict[str, Any]] = []

    for input_text in common_inputs:
        old_record = round_1_records[input_text]
        new_record = round_2_records[input_text]

        old_correct = bool(old_record.get("correct"))
        new_correct = bool(new_record.get("correct"))

        item = compact_case(
            input_text,
            old_record,
            new_record,
        )

        if not old_correct and new_correct:
            fixed_cases.append(item)
        elif old_correct and not new_correct:
            newly_failed_cases.append(item)
        elif not old_correct and not new_correct:
            still_failed_cases.append(item)
        else:
            still_correct_cases.append(item)

    round_1_accuracy = float(
        round_1_best.get(
            "accuracy",
            round_1_data.get("best_accuracy", 0.0),
        )
    )
    round_2_accuracy = float(
        round_2_best.get(
            "accuracy",
            round_2_data.get("best_accuracy", 0.0),
        )
    )

    output_data = {
        "task": round_2_data.get(
            "task",
            round_1_data.get("task"),
        ),
        "dataset_split": round_2_data.get(
            "dataset_split",
            round_1_data.get("dataset_split"),
        ),
        "round_1": {
            "best_prompt_id": round_1_best.get("prompt_id"),
            "best_prompt": round_1_best.get("prompt"),
            "accuracy": round_1_accuracy,
            "correct_count": round_1_best.get("correct_count"),
            "total": round_1_best.get("total"),
        },
        "round_2": {
            "best_prompt_id": round_2_best.get("prompt_id"),
            "best_prompt": round_2_best.get("prompt"),
            "accuracy": round_2_accuracy,
            "correct_count": round_2_best.get("correct_count"),
            "total": round_2_best.get("total"),
        },
        "comparison": {
            "accuracy_delta": round(
                round_2_accuracy - round_1_accuracy,
                6,
            ),
            "common_case_count": len(common_inputs),
            "fixed_case_count": len(fixed_cases),
            "newly_failed_case_count": len(newly_failed_cases),
            "still_failed_case_count": len(still_failed_cases),
            "still_correct_case_count": len(still_correct_cases),
            "only_in_round_1": only_round_1,
            "only_in_round_2": only_round_2,
        },
        "fixed_cases": fixed_cases,
        "newly_failed_cases": newly_failed_cases,
        "still_failed_cases": still_failed_cases,
    }

    save_json(args.output, output_data)

    print("=" * 72)
    print("[DONE] Regression comparison completed")
    print(
        f"Round 1: {round_1_accuracy:.2%} "
        f"({round_1_best.get('prompt_id')})"
    )
    print(
        f"Round 2: {round_2_accuracy:.2%} "
        f"({round_2_best.get('prompt_id')})"
    )
    print(
        f"Delta  : "
        f"{round_2_accuracy - round_1_accuracy:+.2%}"
    )
    print(f"Fixed cases        : {len(fixed_cases)}")
    print(
        f"Newly failed cases : {len(newly_failed_cases)}"
    )
    print(f"Still failed cases : {len(still_failed_cases)}")
    print(f"Saved to           : {args.output}")
    print("=" * 72)


if __name__ == "__main__":
    main()