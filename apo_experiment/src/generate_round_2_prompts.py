from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    """
    嘗試從模型輸出中取出單一 JSON 物件。
    """
    cleaned = text.strip()

    # 移除 Markdown code fence
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"模型輸出不包含有效 JSON：\n{cleaned}")

        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("模型輸出的最外層必須是 JSON object。")

    return parsed


def summarize_history(round_results: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []

    for item in round_results.get("results", []):
        history.append(
            {
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "accuracy": item["accuracy"],
                "correct_count": item["correct_count"],
                "total": item["total"],
            }
        )

    return history


def build_meta_prompt(
    task: str,
    best_prompt: str,
    best_accuracy: float,
    history: list[dict[str, Any]],
    failed_cases: list[dict[str, Any]],
    candidate_count: int,
) -> str:
    payload = {
        "task": task,
        "task_description": (
            "Given a single English word, output its opposite or antonym, "
            "typically by adding, removing, or changing a prefix or suffix "
            "when appropriate."
        ),
        "current_best_prompt": best_prompt,
        "current_best_accuracy": best_accuracy,
        "prompt_history": history,
        "failed_cases": failed_cases,
    }

    return f"""
You are an automatic prompt optimizer.

Your job is to improve an instruction used by a target language model.
The target model receives the instruction and one English input word,
then must output the expected antonym.

Experiment information:
{json.dumps(payload, ensure_ascii=False, indent=2)}

First, analyze the failure patterns at an abstract level. Consider:
- ambiguity caused by words with multiple senses,
- preserving part of speech and derivational morphology,
- preferring direct affix reversal when the dataset expects it,
- choosing conventional lexical antonym pairs,
- avoiding broad but non-matching semantic alternatives.

Do not copy individual failed input-output pairs into the rewritten prompts.
Do not create lookup rules for specific test words.
Do not mention the evaluation dataset or accuracy inside the prompts.

Generate exactly {candidate_count} meaningfully different revised prompts:

1. One candidate emphasizing morphological and part-of-speech preservation.
2. One candidate emphasizing conventional lexical antonym pairs and ambiguity handling.
3. One candidate using an explicit decision priority between affix reversal,
   morphological preservation, lexical pairs, and broader semantic opposites.

Each candidate must:
- be usable as a standalone instruction;
- require exactly one English word as output;
- forbid explanations and additional text;
- remain concise;
- not contain test-case answers.

Return only one JSON object with this exact structure:
{{
  "analysis": {{
    "failure_patterns": ["..."],
    "revision_principles": ["..."]
  }},
  "prompts": [
    {{
      "prompt_id": "round_2_candidate_1",
      "strategy": "...",
      "prompt": "..."
    }},
    {{
      "prompt_id": "round_2_candidate_2",
      "strategy": "...",
      "prompt": "..."
    }},
    {{
      "prompt_id": "round_2_candidate_3",
      "strategy": "...",
      "prompt": "..."
    }}
  ]
}}
""".strip()


def validate_output(data: dict[str, Any], expected_count: int) -> None:
    prompts = data.get("prompts")

    if not isinstance(prompts, list):
        raise ValueError("輸出缺少 prompts list。")

    if len(prompts) != expected_count:
        raise ValueError(
            f"預期 {expected_count} 個 Prompt，實際得到 {len(prompts)} 個。"
        )

    required_keys = {"prompt_id", "strategy", "prompt"}

    for index, item in enumerate(prompts, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 個 Prompt 不是 JSON object。")

        missing = required_keys - item.keys()
        if missing:
            raise ValueError(f"第 {index} 個 Prompt 缺少欄位：{missing}")

        prompt = str(item["prompt"]).strip()
        if not prompt:
            raise ValueError(f"第 {index} 個 Prompt 為空。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根據 Round 1 結果及錯誤案例產生 Round 2 Prompt。"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/round_1_results.json"),
    )
    parser.add_argument(
        "--failed-cases",
        type=Path,
        default=Path("results/round_1_failed_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prompts/round_2_prompts.json"),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PROMPT_OPTIMIZER_MODEL", "gpt-5.5"),
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    if args.candidate_count != 3:
        raise ValueError(
            "目前 Meta Prompt 固定定義三種策略，請先使用 --candidate-count 3。"
        )

    load_dotenv()

    round_results = load_json(args.results)
    failed_data = load_json(args.failed_cases)

    task = str(round_results["task"])
    best_prompt = str(round_results["best_prompt"])
    best_accuracy = float(round_results["best_accuracy"])
    history = summarize_history(round_results)
    failed_cases = failed_data.get("failed_cases", [])

    meta_prompt = build_meta_prompt(
        task=task,
        best_prompt=best_prompt,
        best_accuracy=best_accuracy,
        history=history,
        failed_cases=failed_cases,
        candidate_count=args.candidate_count,
    )

    client = OpenAI()

    print(f"[INFO] Prompt optimizer model: {args.model}")
    print(f"[INFO] Failed cases: {len(failed_cases)}")
    print(f"[INFO] Generating {args.candidate_count} Round 2 prompts...")

    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "system",
                "content": (
                    "You optimize task instructions using evaluation history "
                    "and failure analysis. Output valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": meta_prompt,
            },
        ],
    )

    raw_text = response.output_text.strip()
    generated = extract_json_object(raw_text)
    validate_output(generated, args.candidate_count)

    output_data = {
        "task": task,
        "round": 2,
        "method": "protegi_opro_inspired_rewriting",
        "optimizer_model": args.model,
        "parent_prompt_id": round_results["best_prompt_id"],
        "parent_prompt": best_prompt,
        "parent_accuracy": best_accuracy,
        "source_failed_case_count": len(failed_cases),
        "analysis": generated.get("analysis", {}),
        "prompts": generated["prompts"],
    }

    save_json(args.output, output_data)

    print(f"[DONE] Saved to: {args.output}")
    for item in output_data["prompts"]:
        print(f"\n{item['prompt_id']} — {item['strategy']}")
        print(item["prompt"])


if __name__ == "__main__":
    main()