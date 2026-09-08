import json
import numpy as np

from src.baseline.baseline_state import PersonalGaitState
from src.drift.deviation import calculate_cycle_deviation
from src.drift.temporal_analysis import analyze_deviation_sequence


TRAINING_FILES = [
    "data/features/gait_session_20260831_144149_features.json",
    "data/features/gait_session_20260831_152349_features.json",
    "data/features/gait_session_20260831_154635_features.json",
]

TEST_FILE = (
    "data/features/"
    "gait_session_20260831_160334_features.json"
)

# Diagnostic threshold only.
# This is NOT a fatigue-risk threshold.
THRESHOLD_PERCENTILE = 95


def load_cycles(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["phase_normalized_cycles"]


def build_baseline(training_groups):
    baseline = PersonalGaitState()

    for cycles in training_groups:
        for cycle in cycles:
            baseline.update(cycle)

    return baseline


def get_scores(baseline, cycles):
    scores = []

    for cycle in cycles:
        result = calculate_cycle_deviation(
            baseline,
            cycle,
        )
        scores.append(result["overall_score"])

    return np.asarray(scores, dtype=float)


def main():
    training_groups = [
        load_cycles(path)
        for path in TRAINING_FILES
    ]

    test_cycles = load_cycles(TEST_FILE)

    baseline = build_baseline(training_groups)

    training_scores = np.concatenate(
        [
            get_scores(baseline, cycles)
            for cycles in training_groups
        ]
    )

    test_scores = get_scores(
        baseline,
        test_cycles,
    )

    threshold = float(
        np.percentile(
            training_scores,
            THRESHOLD_PERCENTILE,
        )
    )

    analysis = analyze_deviation_sequence(
        test_scores,
        threshold=threshold,
        smoothing_window=3,
        trend_window=4,
    )

    print("========================================")
    print("GaitGuard S4 Temporal Analysis")
    print("========================================")

    print(
        f"Training cycles: {baseline.cycle_count}"
    )

    print(
        f"S4 test cycles: {len(test_scores)}"
    )

    print(
        f"Training {THRESHOLD_PERCENTILE}th percentile: "
        f"{threshold:.4f}"
    )

    print("\nCycle-by-cycle analysis:")
    print(
        "Cycle | Deviation | Smoothed | Slope | "
        "Persistence | Recovery"
    )
    print("-" * 68)

    for i in range(len(test_scores)):
        print(
            f"{i + 1:5d} | "
            f"{test_scores[i]:9.4f} | "
            f"{analysis['smoothed'][i]:8.4f} | "
            f"{analysis['local_slope'][i]:5.4f} | "
            f"{analysis['persistence'][i]:11.2f} | "
            f"{analysis['recovery'][i]:8.4f}"
        )

    print("\nOverall S4:")
    print(
        f"Mean deviation: "
        f"{np.mean(test_scores):.4f}"
    )
    print(
        f"Maximum deviation: "
        f"{np.max(test_scores):.4f}"
    )
    print(
        f"First deviation: "
        f"{test_scores[0]:.4f}"
    )
    print(
        f"Last deviation: "
        f"{test_scores[-1]:.4f}"
    )

    print("\nTemporal diagnostics:")
    print(
        f"Final local slope: "
        f"{analysis['local_slope'][-1]:.4f}"
    )
    print(
        f"Final persistence: "
        f"{analysis['persistence'][-1]:.2f}"
    )
    print(
        f"Final recovery: "
        f"{analysis['recovery'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()