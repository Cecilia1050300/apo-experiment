import re


def normalize_text(text):
    text = str(text).strip().lower()

    text = re.sub(
        r"^[\"'`]+|[\"'`]+$",
        "",
        text,
    )

    text = re.sub(
        r"[.,!?;:]+$",
        "",
        text,
    )

    return text.strip()


def is_correct(prediction, expected_outputs):
    normalized_prediction = normalize_text(prediction)

    normalized_expected = [
        normalize_text(output)
        for output in expected_outputs
    ]

    return normalized_prediction in normalized_expected


def evaluate_predictions(records):
    evaluated_records = []
    correct_count = 0

    for record in records:
        prediction = record["prediction"]
        expected_outputs = record["expected_outputs"]

        correct = is_correct(
            prediction,
            expected_outputs,
        )

        if correct:
            correct_count += 1

        evaluated_records.append({
            **record,
            "correct": correct,
        })

    total = len(evaluated_records)

    accuracy = (
        correct_count / total
        if total > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total": total,
        "records": evaluated_records,
    }