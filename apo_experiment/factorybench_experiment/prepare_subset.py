import json
import random
from pathlib import Path

SOURCE = Path("factorybench_experiment/data/level_4_train.jsonl")
OUTPUT = Path("factorybench_experiment/data/factorybench_l4_subset.json")

SEED = 42
OPTIMIZATION_SIZE = 5
FINAL_TEST_SIZE = 5

def build_input(item):
    context = item.get("context", {})
    provenance = item.get("provenance", {})

    dataset_name = provenance.get("dataset", "").lower()

    machine_sentences = {
        "aursad": (
            "The following sensor data comes from a Universal Robots UR3e "
            "collaborative robot performing a screwdriving task."
        ),
        "voraus-ad": (
            "The following sensor data comes from a Yu-Cobot collaborative "
            "robot performing an industrial pick-and-place task."
        ),
        "factorywave": (
            "The following sensor data comes from an industrial robotic "
            "platform executing a manufacturing task."
        ),
    }

    machine_sentence = machine_sentences.get(
        dataset_name,
        "The following sensor data comes from an industrial robot."
    )

    ts_format = context.get("time_series_format", {})
    acronym_mapping = ts_format.get("acronym_mapping", {})
    time_series = context.get("time_series", [])

    mapping_text = "\n".join(
        f"- {k}: {v}"
        for k, v in acronym_mapping.items()
    )

    ts_text = "\n".join(time_series)

    return (
        f"{machine_sentence}\n\n"
        f"Signal mapping:\n{mapping_text}\n\n"
        f"Time series:\n{ts_text}\n\n"
        f"Question:\n{item['question']}"
    )

def convert(item):
    return {
        "id": item["id"],
        "input": build_input(item),

        # 給 evaluator / harness 用
        "expected_outputs": [item["root_cause"]],

        # 保留 FactoryBench 原始 gold
        "root_cause": item["root_cause"],
        "reference_answer": item["answer"],

        # audit / trace
        "metadata": {
            "level": item["level"],
            "template_id": item["template_id"],
            "template_type": item["template_type"],
            "provenance": item.get("provenance"),
        },
    }

with SOURCE.open(encoding="utf-8") as f:
    items = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

troubleshooting = [
    x for x in items
    if x.get("level") == 4
    and x.get("template_type") == "troubleshooting"
]

print("Total L4 troubleshooting:", len(troubleshooting))

random.seed(SEED)
random.shuffle(troubleshooting)

needed = OPTIMIZATION_SIZE + FINAL_TEST_SIZE

if len(troubleshooting) < needed:
    raise ValueError(
        f"Need {needed} troubleshooting items, "
        f"but only found {len(troubleshooting)}"
    )

selected = troubleshooting[:needed]

optimization = [
    convert(x)
    for x in selected[:OPTIMIZATION_SIZE]
]

final_test = [
    convert(x)
    for x in selected[
        OPTIMIZATION_SIZE:
        OPTIMIZATION_SIZE + FINAL_TEST_SIZE
    ]
]

output = {
    "task": "factorybench_l4_troubleshooting",
    "task_description": (
        "FactoryBench Level 4 industrial troubleshooting: "
        "identify the most likely root cause from machine telemetry "
        "and recommend corrective action."
    ),
    "seed": SEED,
    "source": str(SOURCE),
    "splits": {
        "optimization": optimization,
        "final_test": final_test,
    },
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("Saved:", OUTPUT)
print("Optimization:", len(optimization))
print("Final test:", len(final_test))

print("\nOptimization root causes:")
for x in optimization:
    print("-", x["id"], x["root_cause"])

print("\nFinal-test root causes:")
for x in final_test:
    print("-", x["id"], x["root_cause"])
