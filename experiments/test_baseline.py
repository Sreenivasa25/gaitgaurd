import json

from src.baseline.baseline_state import (
    PersonalGaitState,
)


INPUT_PATH = "data/features/gait_features.json"
OUTPUT_PATH = (
    "data/models/personal_gait_baseline.json"
)


def main():
    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    cycles = data[
        "phase_normalized_cycles"
    ]

    print(
        f"Loaded {len(cycles)} gait cycles."
    )

    if not cycles:
        raise RuntimeError(
            "No normalized gait cycles found."
        )

    baseline = PersonalGaitState()

    for index, cycle in enumerate(
        cycles,
        start=1,
    ):
        baseline.update(cycle)

        print(
            f"Cycle {index}: "
            f"baseline updated"
        )

    print("\nBaseline summary:")

    for key, value in baseline.summary().items():
        print(f"  {key}: {value}")

    baseline.save(OUTPUT_PATH)

    print(
        f"\n✓ Baseline saved to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()