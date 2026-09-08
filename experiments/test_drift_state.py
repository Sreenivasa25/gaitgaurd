import json

import numpy as np

from src.baseline.baseline_state import PersonalGaitState
from src.drift.deviation import calculate_cycle_deviation
from src.drift.drift_state import DriftStateMachine
from src.drift.temporal_analysis import (
    analyze_deviation_sequence,
)


TRAINING_FILES = [
    "data/features/"
    "gait_session_20260831_144149_features.json",
    "data/features/"
    "gait_session_20260831_152349_features.json",
    "data/features/"
    "gait_session_20260831_154635_features.json",
]

TEST_FILE = (
    "data/features/"
    "gait_session_20260831_160334_features.json"
)


def load_cycles(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["phase_normalized_cycles"]


def build_baseline(training_groups):
    baseline = PersonalGaitState()

    for cycles in training_groups:
        for cycle in cycles:
            baseline.update(cycle)

    return baseline


def calculate_scores(baseline, cycles):
    return np.asarray(
        [
            calculate_cycle_deviation(
                baseline,
                cycle,
            )["overall_score"]
            for cycle in cycles
        ],
        dtype=float,
    )


def main():
    training_groups = [
        load_cycles(path)
        for path in TRAINING_FILES
    ]

    test_cycles = load_cycles(TEST_FILE)

    baseline = build_baseline(
        training_groups
    )

    training_scores = np.concatenate(
        [
            calculate_scores(
                baseline,
                cycles,
            )
            for cycles in training_groups
        ]
    )

    test_scores = calculate_scores(
        baseline,
        test_cycles,
    )

    threshold = float(
        np.percentile(
            training_scores,
            95,
        )
    )

    temporal = analyze_deviation_sequence(
        test_scores,
        threshold=threshold,
        smoothing_window=3,
        trend_window=4,
    )

    machine = DriftStateMachine()

    print("========================================")
    print("GaitGuard Drift State Experiment")
    print("========================================")
    print(
        f"Training cycles: "
        f"{baseline.cycle_count}"
    )
    print(
        f"S4 test cycles: "
        f"{len(test_scores)}"
    )
    print(
        f"Deviation threshold: "
        f"{threshold:.4f}"
    )

    print(
        "\nCycle | Dev | Smooth | Slope | "
        "Persist | Recovery | State"
    )
    print("-" * 78)

    for i in range(len(test_scores)):

        state = machine.update(
            deviation=float(
                temporal["smoothed"][i]
            ),
            slope=float(
                temporal["local_slope"][i]
            ),
            persistence=float(
                temporal["persistence"][i]
            ),
            recovery=float(
                temporal["recovery"][i]
            ),
        )

        print(
            f"{i + 1:5d} | "
            f"{test_scores[i]:.3f} | "
            f"{temporal['smoothed'][i]:.3f} | "
            f"{temporal['local_slope'][i]:.3f} | "
            f"{temporal['persistence'][i]:.2f} | "
            f"{temporal['recovery'][i]:.3f} | "
            f"{state}"
        )


if __name__ == "__main__":
    main()