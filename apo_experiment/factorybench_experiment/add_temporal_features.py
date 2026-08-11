import json
import re
from pathlib import Path
from statistics import mean, pstdev

INPUT = Path(
    "factorybench_experiment/data/factorybench_l4_subset.json"
)

OUTPUT = Path(
    "factorybench_experiment/data/factorybench_l4_subset_with_temporal_features.json"
)


def parse_time_series(lines):
    timestamps = []
    signals = {}

    for line in lines:
        if ":" not in line:
            continue

        time_part, value_part = line.split(":", 1)

        match = re.search(r"t=(\d+)", time_part)
        if match:
            timestamps.append(int(match.group(1)))

        for token in value_part.split(","):
            token = token.strip()

            if "=" not in token:
                continue

            key, value = token.split("=", 1)

            try:
                value = float(value.strip())
            except ValueError:
                continue

            signals.setdefault(key.strip(), []).append(value)

    return timestamps, signals


def safe_mean(values):
    return mean(values) if values else 0.0


def first_half_second_half(values):
    if len(values) < 4:
        return None, None

    mid = len(values) // 2

    return (
        safe_mean(values[:mid]),
        safe_mean(values[mid:])
    )


def relative_change(a, b):
    denominator = max(abs(a), abs(b), 1e-6)
    return abs(b - a) / denominator


def classify_variation(values):
    if len(values) < 2:
        return "insufficient-data"

    value_range = max(values) - min(values)
    magnitude = max(abs(safe_mean(values)), 1e-6)

    ratio = abs(value_range) / magnitude

    if value_range < 1e-6:
        return "nearly-constant"
    elif ratio < 0.05:
        return "low-variation"
    elif ratio < 0.25:
        return "moderate-variation"
    else:
        return "high-variation"


def detect_change(values):
    first, second = first_half_second_half(values)

    if first is None:
        return None

    change = second - first
    ratio = relative_change(first, second)

    return {
        "first_half_mean": first,
        "second_half_mean": second,
        "change": change,
        "relative_change": ratio,
    }


def detect_spike(values):
    if len(values) < 4:
        return {
            "detected": False,
            "peak_deviation": 0.0,
        }

    mu = mean(values)
    sigma = pstdev(values)

    if sigma < 1e-9:
        return {
            "detected": False,
            "peak_deviation": 0.0,
        }

    max_z = max(
        abs(v - mu) / sigma
        for v in values
    )

    return {
        "detected": max_z >= 2.5,
        "peak_deviation": max_z,
    }


def build_temporal_summary(context):
    ts_format = context.get("time_series_format", {})
    mapping = ts_format.get("acronym_mapping", {})
    lines = context.get("time_series", [])

    timestamps, signals = parse_time_series(lines)

    out = []

    out.append("=== Deterministic Temporal Feature Analysis ===")
    out.append(
        "These observations are computed directly from telemetry. "
        "They describe temporal behavior only and do not provide a diagnosis."
    )

    if timestamps:
        duration = timestamps[-1] - timestamps[0]

        out.append(
            f"- timesteps={len(timestamps)}, "
            f"start_timestamp={timestamps[0]}, "
            f"end_timestamp={timestamps[-1]}, "
            f"duration={duration}"
        )

    # --------------------------------------------------
    # Position tracking
    # --------------------------------------------------

    out.append("")
    out.append("=== Position Tracking ===")

    found_pair = False

    for joint in range(6):
        spo = f"spo{joint}"
        fpo = f"fpo{joint}"

        if spo not in signals or fpo not in signals:
            continue

        n = min(
            len(signals[spo]),
            len(signals[fpo])
        )

        if n == 0:
            continue

        found_pair = True

        errors = [
            signals[fpo][i] - signals[spo][i]
            for i in range(n)
        ]

        abs_errors = [abs(x) for x in errors]

        command_delta = (
            signals[spo][n - 1] -
            signals[spo][0]
        )

        feedback_delta = (
            signals[fpo][n - 1] -
            signals[fpo][0]
        )

        out.append(
            f"- joint_{joint}: "
            f"mean_abs_tracking_error={safe_mean(abs_errors):.6f}, "
            f"max_abs_tracking_error={max(abs_errors):.6f}, "
            f"setpoint_delta={command_delta:.6f}, "
            f"feedback_delta={feedback_delta:.6f}"
        )

    if not found_pair:
        out.append("- no setpoint/feedback position pairs available")

    # --------------------------------------------------
    # Motion activity
    # --------------------------------------------------

    out.append("")
    out.append("=== Motion Activity ===")

    for joint in range(6):
        speed = f"fsp{joint}"

        if speed not in signals:
            continue

        values = signals[speed]

        mean_abs_speed = safe_mean(
            [abs(v) for v in values]
        )

        max_abs_speed = max(
            [abs(v) for v in values],
            default=0.0
        )

        out.append(
            f"- joint_{joint}: "
            f"mean_abs_speed={mean_abs_speed:.6f}, "
            f"max_abs_speed={max_abs_speed:.6f}, "
            f"variation={classify_variation(values)}"
        )

    # --------------------------------------------------
    # Temporal changes
    # --------------------------------------------------

    out.append("")
    out.append("=== Temporal Changes ===")

    for name in sorted(signals):
        change = detect_change(signals[name])

        if change is None:
            continue

        long_name = mapping.get(name, name)

        direction = (
            "increase"
            if change["change"] > 0
            else "decrease"
            if change["change"] < 0
            else "stable"
        )

        out.append(
            f"- {name} ({long_name}): "
            f"first_half_mean={change['first_half_mean']:.4f}, "
            f"second_half_mean={change['second_half_mean']:.4f}, "
            f"direction={direction}, "
            f"relative_change={change['relative_change']:.4f}"
        )

    # --------------------------------------------------
    # Spike detection
    # --------------------------------------------------

    out.append("")
    out.append("=== Abrupt Spike Check ===")

    any_spike = False

    for name in sorted(signals):
        result = detect_spike(signals[name])

        if result["detected"]:
            any_spike = True

            out.append(
                f"- {name}: abrupt deviation detected "
                f"(max_z={result['peak_deviation']:.2f})"
            )

    if not any_spike:
        out.append(
            "- no strong isolated spike detected using the current heuristic"
        )

    # --------------------------------------------------
    # Cross-signal factual observations
    # --------------------------------------------------

    out.append("")
    out.append("=== Cross-Signal Observations ===")

    close_tracking = []
    large_tracking = []

    for joint in range(6):
        spo = f"spo{joint}"
        fpo = f"fpo{joint}"

        if spo not in signals or fpo not in signals:
            continue

        n = min(
            len(signals[spo]),
            len(signals[fpo])
        )

        if n == 0:
            continue

        mae = safe_mean([
            abs(signals[fpo][i] - signals[spo][i])
            for i in range(n)
        ])

        if mae < 0.02:
            close_tracking.append(str(joint))
        elif mae > 0.10:
            large_tracking.append(str(joint))

    if close_tracking:
        out.append(
            "- joints with close setpoint/feedback tracking: "
            + ", ".join(close_tracking)
        )

    if large_tracking:
        out.append(
            "- joints with comparatively large tracking error: "
            + ", ".join(large_tracking)
        )

    if not close_tracking and not large_tracking:
        out.append(
            "- no strong setpoint/feedback tracking pattern detected "
            "by the current heuristic"
        )

    out.append("")
    out.append(
        "Interpretation constraint: Do not treat negative torque, "
        "large force, a spike, or a tracking difference as a root cause "
        "by itself. Use the troubleshooting skill to infer the physical "
        "or task-level cause."
    )

    return "\n".join(out)


def extract_context_from_input(text):
    mapping = {}

    if "Signal mapping:\n" in text:
        mapping_text = text.split(
            "Signal mapping:\n",
            1
        )[1]

        if "\n\nTime series:" in mapping_text:
            mapping_text = mapping_text.split(
                "\n\nTime series:",
                1
            )[0]

        for line in mapping_text.splitlines():
            line = line.strip()

            match = re.match(
                r"-\s+([^:]+):\s+(.+)",
                line
            )

            if match:
                mapping[match.group(1)] = match.group(2)

    if "Time series:\n" not in text:
        raise ValueError(
            "Time series block not found"
        )

    ts_text = text.split(
        "Time series:\n",
        1
    )[1]

    if "\n\nQuestion:" in ts_text:
        ts_text = ts_text.split(
            "\n\nQuestion:",
            1
        )[0]

    lines = [
        line.strip()
        for line in ts_text.splitlines()
        if line.strip().startswith("t=")
    ]

    return {
        "time_series_format": {
            "acronym_mapping": mapping,
        },
        "time_series": lines,
    }


def augment_item(item):
    new_item = dict(item)

    context = extract_context_from_input(
        item["input"]
    )

    temporal_summary = build_temporal_summary(
        context
    )

    new_item["temporal_features"] = temporal_summary

    new_item["input"] = (
        item["input"]
        + "\n\n"
        + temporal_summary
        + "\n\n"
        + "Use the deterministic temporal observations together "
          "with the original telemetry and troubleshooting skill "
          "to identify the root cause and corrective action."
    )

    return new_item


def main():
    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    new_data = dict(data)

    new_data["experiment_variant"] = (
        "temporal_feature_tool_augmented"
    )

    new_splits = {}

    for split_name, items in data["splits"].items():
        new_splits[split_name] = [
            augment_item(item)
            for item in items
        ]

    new_data["splits"] = new_splits

    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(
            new_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved:", OUTPUT)

    for split_name, items in new_splits.items():
        print(split_name, len(items))

    print("\n===== FIRST OPTIMIZATION TEMPORAL FEATURES =====")
    print(
        new_splits["optimization"][0][
            "temporal_features"
        ]
    )


if __name__ == "__main__":
    main()
