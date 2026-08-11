import json
import re
from pathlib import Path
from statistics import mean

INPUT = Path(
    "factorybench_experiment/data/factorybench_l4_subset.json"
)

OUTPUT = Path(
    "factorybench_experiment/data/factorybench_l4_subset_with_summary.json"
)


def parse_time_series(lines):
    """
    Input:
      t=123: etto0=1.2, etto1=-3.4, ...

    Output:
      {
        "etto0": [1.2, ...],
        ...
      }
    """
    signals = {}

    for line in lines:
        if ":" not in line:
            continue

        _, values = line.split(":", 1)

        for item in values.split(","):
            item = item.strip()

            if "=" not in item:
                continue

            key, value = item.split("=", 1)
            key = key.strip()

            try:
                value = float(value.strip())
            except ValueError:
                continue

            signals.setdefault(key, []).append(value)

    return signals


def signal_stats(values):
    if not values:
        return None

    start = values[0]
    end = values[-1]

    return {
        "start": start,
        "end": end,
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "range": max(values) - min(values),
        "delta": end - start,
    }


def build_summary(context):
    ts_format = context.get("time_series_format", {})
    mapping = ts_format.get("acronym_mapping", {})
    lines = context.get("time_series", [])

    signals = parse_time_series(lines)

    stats = {
        name: signal_stats(values)
        for name, values in signals.items()
    }

    output = []

    output.append("=== Deterministic Signal Analysis ===")
    output.append(
        "The following statistics were computed directly from the supplied "
        "telemetry. They are measurements, not diagnoses."
    )

    # --------------------------------------------------
    # Individual signal statistics
    # --------------------------------------------------
    for name in sorted(stats):
        s = stats[name]

        if s is None:
            continue

        long_name = mapping.get(name, name)

        output.append(
            f"- {name} ({long_name}): "
            f"start={s['start']:.4f}, "
            f"end={s['end']:.4f}, "
            f"mean={s['mean']:.4f}, "
            f"min={s['min']:.4f}, "
            f"max={s['max']:.4f}, "
            f"range={s['range']:.4f}, "
            f"delta={s['delta']:.4f}"
        )

    # --------------------------------------------------
    # Setpoint vs feedback mismatch
    # spo0 <-> fpo0 etc.
    # --------------------------------------------------
    output.append("")
    output.append("=== Setpoint / Feedback Position Comparison ===")

    found_position_pair = False

    for joint in range(6):
        spo = f"spo{joint}"
        fpo = f"fpo{joint}"

        if spo not in signals or fpo not in signals:
            continue

        n = min(len(signals[spo]), len(signals[fpo]))

        if n == 0:
            continue

        errors = [
            abs(signals[spo][i] - signals[fpo][i])
            for i in range(n)
        ]

        found_position_pair = True

        output.append(
            f"- joint_{joint}: "
            f"mean_abs_position_error={mean(errors):.6f}, "
            f"max_abs_position_error={max(errors):.6f}"
        )

    if not found_position_pair:
        output.append("- no comparable setpoint/feedback position pairs")

    # --------------------------------------------------
    # Motion / stability overview
    # --------------------------------------------------
    output.append("")
    output.append("=== Signal Variation Overview ===")

    variable = []
    stable = []

    for name, s in stats.items():
        if s is None:
            continue

        # Relative heuristic only for summarisation.
        magnitude = max(abs(s["mean"]), 1e-6)
        relative_range = abs(s["range"]) / magnitude

        if s["range"] < 1e-6:
            stable.append(name)
        elif relative_range >= 0.25:
            variable.append(name)

    output.append(
        "- comparatively stable signals: "
        + (", ".join(stable) if stable else "none")
    )

    output.append(
        "- signals with relatively large variation: "
        + (", ".join(variable) if variable else "none")
    )

    output.append("")
    output.append(
        "Important: Large, negative, or changing values are not automatically "
        "faults. Interpret these statistics using machine state, task phase, "
        "cross-signal consistency, and the troubleshooting skill."
    )

    return "\n".join(output)


def augment_item(item):
    new_item = dict(item)

    # The original context isn't directly stored in our converted subset,
    # so recover telemetry from the input string is undesirable.
    # We instead use metadata's original_id later if needed.
    #
    # prepare_subset.py already embedded the telemetry into "input".
    # Extract the Time series block and rebuild a minimal context.

    text = item["input"]

    marker = "Time series:\n"

    if marker not in text:
        raise ValueError(
            f"Time series block not found for case {item['id']}"
        )

    ts_text = text.split(marker, 1)[1]

    if "\n\nQuestion:" in ts_text:
        ts_text = ts_text.split("\n\nQuestion:", 1)[0]

    lines = [
        line.strip()
        for line in ts_text.splitlines()
        if line.strip().startswith("t=")
    ]

    # Extract signal mapping from input.
    mapping = {}

    if "Signal mapping:\n" in text:
        mapping_text = text.split("Signal mapping:\n", 1)[1]

        if "\n\nTime series:" in mapping_text:
            mapping_text = mapping_text.split(
                "\n\nTime series:", 1
            )[0]

        for line in mapping_text.splitlines():
            line = line.strip()

            match = re.match(r"-\s+([^:]+):\s+(.+)", line)

            if match:
                mapping[match.group(1)] = match.group(2)

    context = {
        "time_series_format": {
            "acronym_mapping": mapping
        },
        "time_series": lines,
    }

    summary = build_summary(context)

    # Keep original raw input + add tool output.
    new_item["signal_summary"] = summary

    new_item["input"] = (
        item["input"]
        + "\n\n"
        + summary
        + "\n\n"
        + "Use both the original telemetry and the deterministic signal "
          "analysis above to answer the troubleshooting question."
    )

    return new_item


def main():
    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    new_data = dict(data)
    new_data["experiment_variant"] = "signal_tool_augmented"

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

    print("\n===== FIRST OPTIMIZATION SUMMARY =====")
    print(
        new_splits["optimization"][0]["signal_summary"]
    )


if __name__ == "__main__":
    main()
