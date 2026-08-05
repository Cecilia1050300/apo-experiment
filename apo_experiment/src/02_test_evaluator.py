from evaluator import evaluate_predictions


def main():
    records = [
        {
            "input": "fortunate",
            "expected_outputs": ["unfortunate"],
            "prediction": "unfortunate",
        },
        {
            "input": "urban",
            "expected_outputs": ["rural"],
            "prediction": "Rural.",
        },
        {
            "input": "strength",
            "expected_outputs": ["weakness"],
            "prediction": "The answer is weakness.",
        },
    ]

    result = evaluate_predictions(records)

    print(f"Accuracy: {result['accuracy']:.3f}")
    print(
        f"Correct: {result['correct_count']}"
        f"/{result['total']}"
    )

    for record in result["records"]:
        print(record)


if __name__ == "__main__":
    main()