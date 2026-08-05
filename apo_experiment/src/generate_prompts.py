import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.data.instruction_induction.load_data import load_data


TASK = "antonyms"
NUM_EXAMPLES = 15
NUM_CANDIDATES = 5
MODEL_NAME = "gpt-5.5"

OUTPUT_PATH = (
    REPO_ROOT
    / "apo_experiment"
    / "prompts"
    / "round_1_prompts.json"
)


def build_generation_prompt(inputs, outputs) -> str:
    examples = []

    for input_text, expected_outputs in list(
        zip(inputs, outputs)
    )[:NUM_EXAMPLES]:
        examples.append(
            f"Input: {input_text}\n"
            f"Output: {expected_outputs[0]}"
        )

    demonstrations = "\n\n".join(examples)

    return f"""
You are an automatic prompt optimizer.

Infer the underlying task from the following input-output examples.

{demonstrations}

Generate exactly {NUM_CANDIDATES} different candidate system prompts
for another language model to perform this task.

Requirements:
- Each prompt must clearly describe the inferred task.
- The target model must return only the final answer.
- The target model must not explain its reasoning.
- Keep each candidate concise.
- Do not include specific answers from the examples.
- Return valid JSON only.

Required JSON schema:
{{
  "task_description": "short description of the inferred task",
  "prompts": [
    "candidate system prompt 1",
    "candidate system prompt 2",
    "candidate system prompt 3",
    "candidate system prompt 4",
    "candidate system prompt 5"
  ]
}}
""".strip()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 OPENAI_API_KEY，請確認 Repo 根目錄的 .env"
        )

    client = OpenAI(api_key=api_key)

    induce_inputs, induce_outputs = load_data(
        "induce",
        TASK,
    )

    request_text = build_generation_prompt(
        induce_inputs,
        induce_outputs,
    )

    response = client.responses.create(
        model=MODEL_NAME,
        input=request_text,
        reasoning={"effort": "none"},
    )

    raw_text = response.output_text.strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError:
        print("GPT 回傳內容不是有效 JSON：")
        print(raw_text)
        raise

    prompts = result.get("prompts", [])

    if len(prompts) != NUM_CANDIDATES:
        raise ValueError(
            f"預期 {NUM_CANDIDATES} 個 Prompt，"
            f"實際取得 {len(prompts)} 個"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "task": TASK,
        "optimizer_model": MODEL_NAME,
        "task_description": result.get(
            "task_description",
            "",
        ),
        "prompts": prompts,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Task: {TASK}")
    print(
        "Task description:",
        output_data["task_description"],
    )
    print()

    for index, prompt in enumerate(
        prompts,
        start=1,
    ):
        print(f"Prompt {index}:")
        print(prompt)
        print("-" * 60)

    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()