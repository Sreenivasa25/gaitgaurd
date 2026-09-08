import json
import numpy as np

from src.baseline.baseline_state import PersonalGaitState
from src.drift.deviation import calculate_cycle_deviation


FEATURE_FILE = "data/features/gait_features.json"


def build_baseline(cycles):
    baseline = PersonalGaitState()

    for cycle in cycles:
        baseline.update(cycle)

    return baseline


def evaluate_leave_one_out(cycles):
    print("\n=== Leave-One-Cycle-Out Evaluation ===")

    scores = []

    for held_out_index in range(len(cycles)):
        training_cycles = [
            cycle
            for i, cycle in enumerate(cycles)
            if i != held_out_index
        ]

        test_cycle = cycles[held_out_index]

        baseline = build_baseline(training_cycles)

        result = calculate_cycle_deviation(
            baseline,
            test_cycle,
        )

        score = result["overall_score"]

        scores.append(score)

        print(
            f"Cycle {held_out_index + 1}: "
            f"{score:.4f}"
        )

    print("\nLOO summary")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std:  {np.std(scores):.4f}")
    print(f"Min:  {np.min(scores):.4f}")
    print(f"Max:  {np.max(scores):.4f}")


def evaluate_temporal_split(cycles):
    print("\n=== Temporal Split Evaluation ===")

    split = max(2, len(cycles) // 2)

    training_cycles = cycles[:split]
    test_cycles = cycles[split:]

    baseline = build_baseline(training_cycles)

    print(
        f"Training cycles: {len(training_cycles)}"
    )
    print(
        f"Test cycles: {len(test_cycles)}"
    )

    scores = []

    for index, cycle in enumerate(
        test_cycles,
        start=split + 1,
    ):
        result = calculate_cycle_deviation(
            baseline,
            cycle,
        )

        score = result["overall_score"]

        scores.append(score)

        print(
            f"Cycle {index}: "
            f"{score:.4f}"
        )

    if scores:
        print("\nTemporal summary")
        print(f"Mean: {np.mean(scores):.4f}")
        print(f"Std:  {np.std(scores):.4f}")
        print(f"Min:  {np.min(scores):.4f}")
        print(f"Max:  {np.max(scores):.4f}")


def main():
    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    cycles = data["phase_normalized_cycles"]

    if len(cycles) < 3:
        raise RuntimeError(
            "At least 3 gait cycles are required."
        )

    print(
        f"Loaded {len(cycles)} cycles."
    )

    evaluate_leave_one_out(cycles)
    evaluate_temporal_split(cycles)


if __name__ == "__main__":
    main()