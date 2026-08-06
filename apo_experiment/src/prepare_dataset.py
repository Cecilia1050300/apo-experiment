import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.data.instruction_induction.load_data import load_data


TASK = "antonyms"
RANDOM_SEED = 42
NUM_PROMPT_GENERATION_EXAMPLES = 15

OUTPUT_PATH = (
    REPO_ROOT
    / "apo_experiment"
    / "results"
    / "antonyms_dataset_split.json"
)


def to_records(inputs, outputs):
    return [
        {
            "input": input_text,
            "expected_outputs": expected_outputs,
        }
        for input_text, expected_outputs in zip(inputs, outputs)
    ]


def main():
    random.seed(RANDOM_SEED)

    induce_inputs, induce_outputs = load_data("induce", TASK)
    execute_inputs, execute_outputs = load_data("execute", TASK)

    induce_records = to_records(induce_inputs, induce_outputs)
    execute_records = to_records(execute_inputs, execute_outputs)

    random.shuffle(induce_records)
    random.shuffle(execute_records)

    prompt_generation_records = induce_records[
        :NUM_PROMPT_GENERATION_EXAMPLES
    ]

    optimization_records = execute_records[:50]
    final_test_records = execute_records[50:100]

    dataset = {
        "task": TASK,
        "random_seed": RANDOM_SEED,
        "prompt_generation": prompt_generation_records,
        "optimization": optimization_records,
        "final_test": final_test_records,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Task: {TASK}")
    print(
        "Prompt generation examples:",
        len(prompt_generation_records),
    )
    print(
        "Optimization examples:",
        len(optimization_records),
    )
    print(
        "Final test examples:",
        len(final_test_records),
    )
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
