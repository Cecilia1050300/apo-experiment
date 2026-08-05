from experiments.data.instruction_induction.load_data import load_data


def main() -> None:
    task = "antonyms"

    induce_inputs, induce_outputs = load_data("induce", task)
    execute_inputs, execute_outputs = load_data("execute", task)

    print(f"Task: {task}")
    print(f"Induce count: {len(induce_inputs)}")
    print(f"Execute count: {len(execute_inputs)}")

    print("\nFirst 5 induce examples:")
    for input_text, expected_outputs in list(
        zip(induce_inputs, induce_outputs)
    )[:5]:
        print({
            "input": input_text,
            "outputs": expected_outputs,
        })


if __name__ == "__main__":
    main()