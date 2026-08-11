import json
import os
import time
from pathlib import Path

from openai import OpenAI

INPUT = Path(
    "factorybench_experiment/results/"
    "skill_v1_optimization_results.json"
)

OUTPUT = Path(
    "factorybench_experiment/results/"
    "skill_v1_optimization_judged.json"
)

MODEL = os.getenv("JUDGE_MODEL", "gpt-5.5")

client = OpenAI()

JUDGE_INSTRUCTIONS = """
You are evaluating answers for FactoryBench Level 4 industrial troubleshooting.

Use ONLY the provided:
- model prediction
- gold root cause
- reference corrective answer

Evaluate semantic correctness, not exact wording.

Scoring rubric:

0:
The predicted diagnosis does not identify the correct underlying root cause.

0.5:
The predicted diagnosis identifies the correct root cause or a clearly equivalent
description, but does not provide the correct corrective protocol.

1.0:
The answer identifies the correct root cause and also provides the appropriate
corrective protocol.

Important:
- Do not reward a symptom when the gold answer specifies an underlying cause.
- Do not require exact snake_case label matching.
- Semantically equivalent descriptions count as correct.
- Do not infer extra content that the model did not actually provide.
- If the model only names a root cause and gives no corrective action, the
  maximum score is 0.5.

Return JSON only:

{
  "score": 0,
  "root_cause_correct": false,
  "protocol_correct": false,
  "reason": "brief explanation"
}
"""


def judge(record):
    payload = {
        "prediction": record["prediction_raw"],
        "gold_root_cause": record["expected_root_cause"],
        "reference_corrective_answer": record["reference_answer"],
    }

    response = client.responses.create(
        model=MODEL,
        instructions=JUDGE_INSTRUCTIONS,
        input=json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
    )

    text = response.output_text.strip()

    # Remove possible markdown fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


def main():
    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    judged_records = []
    score_sum = 0.0

    records = data["records"]

    for i, record in enumerate(records, 1):
        print(f"\n[{i}/{len(records)}] {record['id']}")

        result = judge(record)

        score = float(result["score"])
        score_sum += score

        judged = {
            **record,
            "judge": result,
        }

        judged_records.append(judged)

        print("prediction :", record["prediction_raw"])
        print("gold       :", record["expected_root_cause"])
        print("score      :", score)
        print("reason     :", result["reason"])

        time.sleep(0.2)

    total = len(judged_records)
    mean_score = score_sum / total if total else 0.0

    output = {
        "task": "factorybench_l4_troubleshooting",
        "skill_version": "v1",
        "split": "optimization",
        "target_model": data.get("model"),
        "judge_model": MODEL,
        "rubric": {
            "0": "incorrect root cause",
            "0.5": "correct root cause, incomplete/no corrective protocol",
            "1": "correct root cause and corrective protocol",
        },
        "total": total,
        "score_sum": score_sum,
        "mean_score": mean_score,
        "records": judged_records,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n==============================")
    print(f"Total score : {score_sum}/{total}")
    print(f"Mean score  : {mean_score:.3f}")
    print("Saved       :", OUTPUT)


if __name__ == "__main__":
    main()
