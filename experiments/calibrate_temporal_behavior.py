import json

import numpy as np

from src.baseline.baseline_state import PersonalGaitState
from src.drift.deviation import calculate_cycle_deviation
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


def build_baseline(groups):
    baseline = PersonalGaitState()

    for cycles in groups:
        for cycle in cycles:
            baseline.update(cycle)

    return baseline


def get_scores(baseline, cycles):
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


def temporal_stats(scores, threshold):
    analysis = analyze_deviation_sequence(
        scores,
        threshold=threshold,
        smoothing_window=3,
        trend_window=4,
    )

    return analysis


def percentile_summary(name, values):
    values = np.asarray(
        values,
        dtype=float,
    )

    print(f"\n{name}")

    print(
        f"  Mean: {np.mean(values):.4f}"
    )
    print(
        f"  Std:  {np.std(values):.4f}"
    )
    print(
        f"  50th: {np.percentile(values, 50):.4f}"
    )
    print(
        f"  75th: {np.percentile(values, 75):.4f}"
    )
    print(
        f"  90th: {np.percentile(values, 90):.4f}"
    )
    print(
        f"  95th: {np.percentile(values, 95):.4f}"
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
            get_scores(
                baseline,
                cycles,
            )
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
            95,
        )
    )

    training_temporal = []

    for cycles in training_groups:
        scores = get_scores(
            baseline,
            cycles,
        )

        analysis = temporal_stats(
            scores,
            threshold,
        )

        training_temporal.append(
            analysis
        )

    # Pool temporal behavior from normal training sessions.
    slopes = np.concatenate(
        [
            analysis["local_slope"]
            for analysis in training_temporal
        ]
    )

    persistence = np.concatenate(
        [
            analysis["persistence"]
            for analysis in training_temporal
        ]
    )

    recovery = np.concatenate(
        [
            analysis["recovery"]
            for analysis in training_temporal
        ]
    )

    test_analysis = temporal_stats(
        test_scores,
        threshold,
    )

    print("========================================")
    print(
        "GaitGuard Personal Temporal Calibration"
    )
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

    percentile_summary(
        "Normal training slope",
        slopes,
    )

    percentile_summary(
        "Normal training persistence",
        persistence,
    )

    percentile_summary(
        "Normal training recovery",
        recovery,
    )

    percentile_summary(
        "S4 slope",
        test_analysis["local_slope"],
    )

    percentile_summary(
        "S4 persistence",
        test_analysis["persistence"],
    )

    percentile_summary(
        "S4 recovery",
        test_analysis["recovery"],
    )


if __name__ == "__main__":
    main()