import json
import os
import time
from pathlib import Path

from openai import OpenAI

DATASET = Path(
    "factorybench_experiment/data/factorybench_l4_subset_with_temporal_features.json"
)

SKILL = Path(
    "factorybench_experiment/skills/skill_v2.md"
)

OUTPUT = Path(
    "factorybench_experiment/results/skill_v2_temporal_optimization_results.json"
)

FAILED_OUTPUT = Path(
    "factorybench_experiment/results/skill_v2_temporal_optimization_failed_cases.json"
)

MODEL = os.getenv("TARGET_MODEL", "gpt-5.5")

client = OpenAI()


def normalize_root_cause(text: str) -> str:
    text = str(text).strip().lower()

    if "root_cause:" in text:
        text = text.split("root_cause:", 1)[1]

    text = text.splitlines()[0]
    text = text.strip(" `\"'.,:;")

    return text


def main():
    with DATASET.open(encoding="utf-8") as f:
        dataset = json.load(f)

    with SKILL.open(encoding="utf-8") as f:
        skill_text = f.read()

    cases = dataset["splits"]["optimization"]

    records = []
    correct_count = 0

    system_prompt = f"""
You are solving FactoryBench Level 4 industrial troubleshooting tasks.

Use the following reusable troubleshooting skill.

--- BEGIN SKILL ---
{skill_text}
--- END SKILL ---

Analyze the FactoryBench Level 4 troubleshooting case using the supplied skill.

Answer concisely using exactly this structure:

Root cause:
<the most likely underlying machine fault or condition>

Corrective action:
<the appropriate corrective procedure>

Base the diagnosis only on the supplied machine telemetry and context.
Do not guess dataset label names.
Do not return option letters such as A, B, C, or D.
"""

    for i, item in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] {item['id']}")

        start = time.time()

        response = client.responses.create(
            model=MODEL,
            instructions=system_prompt,
            input=item["input"],
        )

        latency = time.time() - start

        prediction_raw = response.output_text.strip()
        prediction = normalize_root_cause(prediction_raw)
        gold = normalize_root_cause(item["root_cause"])

        correct = prediction == gold

        if correct:
            correct_count += 1

        print("prediction:", prediction)
        print("gold      :", gold)
        print("result    :", "PASS" if correct else "FAIL")

        usage = getattr(response, "usage", None)

        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", 0) if usage else 0

        records.append({
            "id": item["id"],
            "prediction_raw": prediction_raw,
            "prediction": prediction,
            "expected_root_cause": gold,
            "correct": correct,
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input": item["input"],
            "reference_answer": item["reference_answer"],
            "metadata": item["metadata"],
        })

    total = len(records)
    accuracy = correct_count / total if total else 0.0

    result = {
        "task": "factorybench_l4_troubleshooting",
        "skill_version": "v2_temporal_tool",
        "split": "optimization",
        "model": MODEL,
        "correct_count": correct_count,
        "total": total,
        "accuracy": accuracy,
        "records": records,
    }

    failed = [
        r for r in records
        if not r["correct"]
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with FAILED_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            failed,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n==============================")
    print(
        f"Skill v2 optimization accuracy: "
        f"{correct_count}/{total} = {accuracy:.2%}"
    )
    print("Results:", OUTPUT)
    print("Failed :", FAILED_OUTPUT)


if __name__ == "__main__":
    main()
