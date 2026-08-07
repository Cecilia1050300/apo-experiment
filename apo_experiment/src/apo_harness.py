import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Basic I/O
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[SAVE] {path}")


# ============================================================
# BIG-bench Dataset
# ============================================================

def get_expected_outputs(
    target_scores: Dict[str, float],
) -> List[str]:
    """
    Example:

    {
        "entailment": 0,
        "non-entailment": 1
    }

    -> ["non-entailment"]
    """

    if not target_scores:
        raise ValueError("target_scores is empty")

    best_score = max(target_scores.values())

    return [
        label
        for label, score in target_scores.items()
        if score == best_score
    ]


def convert_examples(
    examples: List[Dict[str, Any]],
    task_name: str,
) -> List[Dict[str, Any]]:

    converted = []

    for idx, example in enumerate(examples):

        if "input" not in example:
            raise ValueError(
                f"Example {idx} missing 'input'"
            )

        if "target_scores" not in example:
            raise ValueError(
                f"Example {idx} missing 'target_scores'"
            )

        expected_outputs = get_expected_outputs(
            example["target_scores"]
        )

        item = {
            "id": f"{task_name}_{idx:04d}",
            "input": example["input"],
            "expected_outputs": expected_outputs,
            "target_scores": example["target_scores"],
        }

        # 保留 BIG-bench 額外資訊
        if "comment" in example:
            item["comment"] = example["comment"]

        converted.append(item)

    return converted


# ============================================================
# Inspect
# ============================================================

def inspect_task(source: Path) -> None:

    data = load_json(source)

    print("=" * 80)
    print("BIG-bench Task Inspect")
    print("=" * 80)

    print(f"name            : {data.get('name')}")
    print(f"description     : {data.get('description')}")
    print(f"preferred_score : {data.get('preferred_score')}")
    print(f"metrics         : {data.get('metrics')}")
    print(f"task_prefix     : {repr(data.get('task_prefix'))}")

    examples = data.get("examples", [])

    print(f"example_count   : {len(examples)}")

    labels = set()

    for example in examples:
        target_scores = example.get(
            "target_scores",
            {},
        )
        labels.update(target_scores.keys())

    print(f"labels          : {sorted(labels)}")

    print()
    print("First 3 examples")
    print("-" * 80)

    for idx, example in enumerate(
        examples[:3]
    ):

        print(f"\nExample {idx}")

        print("input:")
        print(example.get("input"))

        print("\ntarget_scores:")

        print(
            json.dumps(
                example.get("target_scores"),
                ensure_ascii=False,
                indent=2,
            )
        )

        if "comment" in example:
            print("\ncomment:")
            print(example["comment"])


# ============================================================
# Prepare Dataset
# ============================================================

def prepare_dataset(
    source: Path,
    task_name: Optional[str],
    optimization_size: int,
    final_test_size: int,
    seed: int,
    output: Optional[Path],
) -> None:

    data = load_json(source)

    source_task_name = data.get("name")

    if task_name is None:
        task_name = source_task_name

    if not task_name:
        raise ValueError(
            "Task name cannot be determined. "
            "Please provide --task."
        )

    examples = data.get("examples")

    if not examples:
        raise ValueError(
            "No examples found in task.json"
        )

    converted = convert_examples(
        examples=examples,
        task_name=task_name,
    )

    total_needed = (
        optimization_size
        + final_test_size
    )

    if len(converted) < total_needed:
        raise ValueError(
            f"Dataset only contains "
            f"{len(converted)} examples, "
            f"but requested {total_needed}."
        )

    rng = random.Random(seed)

    shuffled = converted.copy()

    rng.shuffle(shuffled)

    optimization = shuffled[
        :optimization_size
    ]

    final_test = shuffled[
        optimization_size:
        optimization_size + final_test_size
    ]

    labels = sorted(
        {
            label
            for item in converted
            for label
            in item["target_scores"].keys()
        }
    )

    result = {
        "task": task_name,
        "task_description": data.get(
            "description"
        ),
        "source": str(source),
        "source_format": (
            "bigbench_task_json"
        ),
        "seed": seed,
        "labels": labels,
        "split_method": (
            "random_fixed_seed"
        ),
        "split_sizes": {
            "optimization": len(
                optimization
            ),
            "final_test": len(
                final_test
            ),
        },
        "optimization": optimization,
        "final_test": final_test,
    }

    if output is None:
        output = Path(
            f"results/"
            f"{task_name}_dataset_split.json"
        )

    save_json(
        result,
        output,
    )

    print()
    print("=" * 80)
    print("Dataset Prepared")
    print("=" * 80)

    print(f"task          : {task_name}")
    print(
        f"description   : "
        f"{data.get('description')}"
    )
    print(f"labels        : {labels}")
    print(
        f"total source  : "
        f"{len(converted)}"
    )
    print(
        f"optimization  : "
        f"{len(optimization)}"
    )
    print(
        f"final_test    : "
        f"{len(final_test)}"
    )
    print(f"seed          : {seed}")
    print(f"output        : {output}")


# ============================================================
# JSON Parser for Optimizer
# ============================================================

def extract_json_object(
    text: str,
) -> Dict[str, Any]:

    text = text.strip()

    # 1. 直接解析 JSON
    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # 2. ```json ... ```
    match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        re.DOTALL,
    )

    if match:

        result = json.loads(
            match.group(1)
        )

        if isinstance(result, dict):
            return result

    # 3. 找第一個 { 到最後一個 }
    start = text.find("{")
    end = text.rfind("}")

    if (
        start != -1
        and end != -1
        and end > start
    ):

        result = json.loads(
            text[start:end + 1]
        )

        if isinstance(result, dict):
            return result

    raise ValueError(
        "Could not parse JSON "
        "from model output:\n"
        + text
    )


# ============================================================
# Round 1 Prompt Generation
# ============================================================

def generate_round_1_prompts(
    dataset_path: Path,
    count: int,
    model: str,
    output: Optional[Path],
) -> None:

    load_dotenv()

    client = OpenAI()

    dataset = load_json(
        dataset_path
    )

    task = dataset["task"]

    task_description = dataset.get(
        "task_description",
        "",
    )

    labels = dataset["labels"]

    meta_prompt = f"""
You are an automatic prompt optimizer.

Generate candidate SYSTEM PROMPTS for a target language model.

Task name:
{task}

Task description:
{task_description}

Allowed output labels:
{json.dumps(labels, ensure_ascii=False)}

Generate exactly {count} distinct candidate system prompts.

Requirements:

1. Each candidate prompt must help the target model solve the task accurately.

2. Use meaningfully different reasoning strategies across candidates.

3. Do NOT use or quote any dataset examples.

4. Do NOT include few-shot demonstrations.

5. The target model must output exactly one of the allowed labels and nothing else.

6. The system prompt may instruct the target model to reason internally before producing the final label.

7. Keep each prompt concise enough for practical deployment.

8. When the task contains nested beliefs, knowledge, assumptions, suspicions, perceptions, memories, or other mental-state operators, preserve who holds each mental state and its scope.

9. Do not automatically treat "Person A believes that Person B knows X" as evidence that "Person B knows X" unless the semantics of the outer mental-state operator license that inference.

10. After resolving perspective and mental-state scope, evaluate ordinary semantic entailment between the premise and hypothesis.

11. Avoid reversing subject relations, swapping belief holders, or adding information that is not supported by the premise.

Return JSON only:

{{
  "prompts": [
    {{
      "strategy": "short strategy name",
      "prompt": "system prompt text"
    }}
  ]
}}
""".strip()

    print(
        f"[GEN-R1] task={task} "
        f"model={model} "
        f"count={count}"
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": meta_prompt,
                    }
                ],
            }
        ],
        reasoning={
            "effort": "high"
        },
    )

    raw_text = (
        response.output_text.strip()
    )

    parsed = extract_json_object(
        raw_text
    )

    prompts = parsed.get(
        "prompts",
        [],
    )

    if not isinstance(
        prompts,
        list,
    ):
        raise ValueError(
            "'prompts' must be a list"
        )

    if len(prompts) != count:

        raise ValueError(
            f"Expected {count} prompts, "
            f"but received "
            f"{len(prompts)}"
        )

    output_prompts = []

    for idx, item in enumerate(
        prompts,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Candidate {idx} "
                f"is not an object"
            )

        if "prompt" not in item:
            raise ValueError(
                f"Candidate {idx} "
                f"missing 'prompt'"
            )

        output_prompts.append(
            {
                "prompt_id": (
                    f"candidate_{idx}"
                ),
                "strategy": item.get(
                    "strategy",
                    f"strategy_{idx}",
                ),
                "prompt": str(
                    item["prompt"]
                ).strip(),
            }
        )

    result = {
        "task": task,
        "task_description": (
            task_description
        ),
        "labels": labels,
        "generation_method": (
            "zero_shot_candidate_generation"
        ),
        "optimizer_model": model,
        "candidate_count": count,
        "prompts": output_prompts,
    }

    if output is None:

        output = Path(
            f"prompts/"
            f"{task}_round_1_prompts.json"
        )

    save_json(
        result,
        output,
    )

    print()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# Evaluation Helpers
# ============================================================

def normalize_prediction(
    text: str,
) -> str:

    return text.strip().lower()


def run_target_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_input: str,
    labels: List[str],
    max_retries: int = 2,
) -> Dict[str, Any]:

    last_error = None

    normalized_label_map = {
        normalize_prediction(label): label
        for label in labels
    }

    for attempt in range(
        max_retries + 1
    ):

        try:

            time1 = time.time()

            response = (
                client.responses.create(
                    model=model,
                    input=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        system_prompt
                                    ),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        user_input
                                    ),
                                }
                            ],
                        },
                    ],
                    reasoning={
                        "effort": "high"
                    },
                )
            )

            time2 = time.time()

            latency = (
                time2 - time1
            )

            raw_prediction = (
                response.output_text
                .strip()
            )

            normalized = (
                normalize_prediction(
                    raw_prediction
                )
            )

            format_compliant = (
                normalized
                in normalized_label_map
            )

            if format_compliant:

                prediction = (
                    normalized_label_map[
                        normalized
                    ]
                )

            else:

                prediction = (
                    raw_prediction
                )

            usage = getattr(
                response,
                "usage",
                None,
            )

            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            if usage is not None:

                input_tokens = (
                    getattr(
                        usage,
                        "input_tokens",
                        0,
                    )
                    or 0
                )

                output_tokens = (
                    getattr(
                        usage,
                        "output_tokens",
                        0,
                    )
                    or 0
                )

                total_tokens = (
                    getattr(
                        usage,
                        "total_tokens",
                        0,
                    )
                    or 0
                )

            return {
                "prediction": (
                    prediction
                ),
                "raw_prediction": (
                    raw_prediction
                ),
                "format_compliant": (
                    format_compliant
                ),
                "error": None,
                "latency_seconds": round(
                    latency,
                    4,
                ),
                "usage": {
                    "input_tokens": (
                        input_tokens
                    ),
                    "output_tokens": (
                        output_tokens
                    ),
                    "total_tokens": (
                        total_tokens
                    ),
                },
            }

        except Exception as e:

            last_error = str(e)

            print(
                f"[Attempt "
                f"{attempt + 1}] "
                f"error: "
                f"{last_error}"
            )

            if attempt < max_retries:
                time.sleep(1)

    return {
        "prediction": "",
        "raw_prediction": "",
        "format_compliant": False,
        "error": last_error,
        "latency_seconds": 0.0,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


# ============================================================
# Evaluate
# ============================================================

def evaluate_prompts(
    prompts_path: Path,
    dataset_path: Path,
    split: str,
    model: str,
    output: Path,
    max_cases: Optional[int],
) -> None:

    load_dotenv()

    client = OpenAI()

    prompt_data = load_json(
        prompts_path
    )

    dataset = load_json(
        dataset_path
    )

    task = dataset["task"]

    labels = dataset["labels"]

    if split not in dataset:

        raise ValueError(
            f"Split '{split}' "
            f"not found"
        )

    cases = dataset[split]

    if max_cases is not None:

        if max_cases <= 0:
            raise ValueError(
                "--max-cases must "
                "be greater than 0"
            )

        cases = cases[
            :max_cases
        ]

    if not cases:
        raise ValueError(
            "No evaluation cases"
        )

    prompts = prompt_data.get(
        "prompts",
        [],
    )

    if not prompts:
        raise ValueError(
            "No prompts found"
        )

    all_results = []

    # --------------------------------------------------------
    # Evaluate each prompt
    # --------------------------------------------------------

    for p_idx, prompt_item in enumerate(
        prompts,
        start=1,
    ):

        prompt_id = (
            prompt_item[
                "prompt_id"
            ]
        )

        system_prompt = (
            prompt_item[
                "prompt"
            ]
        )

        print()
        print("=" * 80)

        print(
            f"[Prompt "
            f"{p_idx}/"
            f"{len(prompts)}] "
            f"{prompt_id}"
        )

        print("=" * 80)

        records = []

        correct_count = 0

        format_count = 0

        total_latency = 0.0

        usage_sum = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        # ----------------------------------------------------
        # Evaluate each case
        # ----------------------------------------------------

        for c_idx, case in enumerate(
            cases,
            start=1,
        ):

            result = (
                run_target_model(
                    client=client,
                    model=model,
                    system_prompt=(
                        system_prompt
                    ),
                    user_input=(
                        case["input"]
                    ),
                    labels=labels,
                )
            )

            expected_outputs = [
                normalize_prediction(x)
                for x
                in case[
                    "expected_outputs"
                ]
            ]

            prediction_norm = (
                normalize_prediction(
                    result[
                        "prediction"
                    ]
                )
            )

            correct = (
                prediction_norm
                in expected_outputs
            )

            if correct:
                correct_count += 1

            if result[
                "format_compliant"
            ]:
                format_count += 1

            total_latency += (
                result[
                    "latency_seconds"
                ]
            )

            for key in usage_sum:

                usage_sum[key] += (
                    result[
                        "usage"
                    ][key]
                )

            record = {
                "id": case["id"],
                "input": (
                    case["input"]
                ),
                "expected_outputs": (
                    case[
                        "expected_outputs"
                    ]
                ),
                "prediction": (
                    result[
                        "prediction"
                    ]
                ),
                "raw_prediction": (
                    result[
                        "raw_prediction"
                    ]
                ),
                "correct": correct,
                "format_compliant": (
                    result[
                        "format_compliant"
                    ]
                ),
                "error": (
                    result["error"]
                ),
                "latency_seconds": (
                    result[
                        "latency_seconds"
                    ]
                ),
                "usage": (
                    result[
                        "usage"
                    ]
                ),
            }

            records.append(
                record
            )

            print(
                f"[{c_idx:02d}/"
                f"{len(cases)}] "
                f"expected="
                f"{case['expected_outputs']} "
                f"pred="
                f"{result['prediction']} "
                f"correct={correct}"
            )

        total = len(cases)

        accuracy = (
            correct_count
            / total
        )

        format_compliance = (
            format_count
            / total
        )

        avg_latency = (
            total_latency
            / total
        )

        result_item = {
            "prompt_id": (
                prompt_id
            ),
            "strategy": (
                prompt_item.get(
                    "strategy"
                )
            ),
            "prompt": (
                system_prompt
            ),
            "accuracy": round(
                accuracy,
                4,
            ),
            "correct_count": (
                correct_count
            ),
            "total": total,
            "failed_count": (
                total
                - correct_count
            ),
            "format_compliance": (
                round(
                    format_compliance,
                    4,
                )
            ),
            "format_compliant_count": (
                format_count
            ),
            "average_latency_seconds": (
                round(
                    avg_latency,
                    4,
                )
            ),
            "total_latency_seconds": (
                round(
                    total_latency,
                    4,
                )
            ),
            "usage": (
                usage_sum
            ),
            "records": (
                records
            ),
        }

        all_results.append(
            result_item
        )

    # --------------------------------------------------------
    # Rank
    #
    # 1. Accuracy higher is better
    # 2. Format compliance higher
    # 3. Tokens lower
    # 4. Latency lower
    # --------------------------------------------------------

    all_results.sort(
        key=lambda x: (
            -x["accuracy"],
            -x[
                "format_compliance"
            ],
            x["usage"][
                "total_tokens"
            ],
            x[
                "average_latency_seconds"
            ],
        )
    )

    best = all_results[0]

    # --------------------------------------------------------
    # Save full results
    # --------------------------------------------------------

    result = {
        "task": task,
        "target_model": model,
        "dataset_split": split,
        "labels": labels,
        "evaluation": {
            "primary_metric": (
                "exact_match_accuracy"
            ),
            "secondary_metrics": [
                "format_compliance",
                "total_tokens",
                "average_latency_seconds",
            ],
        },
        "results": (
            all_results
        ),
    }

    save_json(
        result,
        output,
    )

    # --------------------------------------------------------
    # Save failed cases of BEST prompt
    # --------------------------------------------------------

    failed_cases = [
        record
        for record
        in best["records"]
        if not record[
            "correct"
        ]
    ]

    failed_result = {
        "task": task,
        "dataset_split": split,
        "best_prompt_id": (
            best[
                "prompt_id"
            ]
        ),
        "best_prompt": (
            best[
                "prompt"
            ]
        ),
        "best_accuracy": (
            best[
                "accuracy"
            ]
        ),
        "failed_case_count": (
            len(
                failed_cases
            )
        ),
        "failed_cases": (
            failed_cases
        ),
    }

    stem = output.stem

    if stem.endswith(
        "_results"
    ):

        failed_stem = (
            stem[:-8]
            + "_failed_cases"
        )

    else:

        failed_stem = (
            stem
            + "_failed_cases"
        )

    failed_output = (
        output.parent
        / f"{failed_stem}.json"
    )

    save_json(
        failed_result,
        failed_output,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Best Prompt")
    print("=" * 80)

    print(
        f"id       : "
        f"{best['prompt_id']}"
    )

    print(
        f"accuracy : "
        f"{best['accuracy']:.2%}"
    )

    print(
        f"correct  : "
        f"{best['correct_count']}"
        f"/{best['total']}"
    )

    print(
        f"format   : "
        f"{best['format_compliance']:.2%}"
    )

    print(
        f"tokens   : "
        f"{best['usage']['total_tokens']}"
    )

    print(
        f"latency  : "
        f"{best['average_latency_seconds']:.4f}s"
    )


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Generic APO Evaluation Harness "
            "for BIG-bench tasks"
        )
    )

    subparsers = (
        parser.add_subparsers(
            dest="command",
            required=True,
        )
    )

    # --------------------------------------------------------
    # inspect
    # --------------------------------------------------------

    inspect_parser = (
        subparsers.add_parser(
            "inspect",
            help=(
                "Inspect BIG-bench task"
            ),
        )
    )

    inspect_parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )

    # --------------------------------------------------------
    # prepare
    # --------------------------------------------------------

    prepare_parser = (
        subparsers.add_parser(
            "prepare",
            help=(
                "Prepare dataset split"
            ),
        )
    )

    prepare_parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )

    prepare_parser.add_argument(
        "--task",
        type=str,
        default=None,
    )

    prepare_parser.add_argument(
        "--optimization-size",
        type=int,
        default=50,
    )

    prepare_parser.add_argument(
        "--final-test-size",
        type=int,
        default=50,
    )

    prepare_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    prepare_parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    # --------------------------------------------------------
    # gen-r1
    # --------------------------------------------------------

    gen_r1_parser = (
        subparsers.add_parser(
            "gen-r1",
            help=(
                "Generate Round 1 "
                "prompt candidates"
            ),
        )
    )

    gen_r1_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    gen_r1_parser.add_argument(
        "--count",
        type=int,
        default=5,
    )

    gen_r1_parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.5",
    )

    gen_r1_parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    # --------------------------------------------------------
    # eval
    # --------------------------------------------------------

    eval_parser = (
        subparsers.add_parser(
            "eval",
            help=(
                "Evaluate prompt candidates"
            ),
        )
    )

    eval_parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
    )

    eval_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    eval_parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=[
            "optimization",
            "final_test",
        ],
    )

    eval_parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.5",
    )

    eval_parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )

    eval_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = build_parser()

    args = parser.parse_args()

    if args.command == "inspect":

        inspect_task(
            source=args.source,
        )

    elif args.command == "prepare":

        prepare_dataset(
            source=args.source,
            task_name=args.task,
            optimization_size=(
                args.optimization_size
            ),
            final_test_size=(
                args.final_test_size
            ),
            seed=args.seed,
            output=args.output,
        )

    elif args.command == "gen-r1":

        generate_round_1_prompts(
            dataset_path=args.dataset,
            count=args.count,
            model=args.model,
            output=args.output,
        )

    elif args.command == "eval":

        evaluate_prompts(
            prompts_path=args.prompts,
            dataset_path=args.dataset,
            split=args.split,
            model=args.model,
            output=args.output,
            max_cases=args.max_cases,
        )

    else:

        raise ValueError(
            f"Unknown command: "
            f"{args.command}"
        )


if __name__ == "__main__":
    main()