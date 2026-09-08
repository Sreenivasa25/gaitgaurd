import json

import numpy as np

from src.baseline.baseline_state import PersonalGaitState
from src.drift.deviation import calculate_cycle_deviation


SESSION_FILES = [
    (
        "Session 1",
        "data/features/"
        "gait_session_20260831_144149_features.json",
    ),
    (
        "Session 2",
        "data/features/"
        "gait_session_20260831_152349_features.json",
    ),
    (
        "Session 3",
        "data/features/"
        "gait_session_20260831_154635_features.json",
    ),
    (
        "Session 4",
        "data/features/"
        "gait_session_20260831_160334_features.json",
    ),
]


def load_cycles(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["phase_normalized_cycles"]


def build_baseline(cycle_groups):
    baseline = PersonalGaitState()

    for cycles in cycle_groups:
        for cycle in cycles:
            baseline.update(cycle)

    return baseline


def calculate_scores(baseline, cycles):
    scores = []

    for cycle in cycles:
        result = calculate_cycle_deviation(
            baseline,
            cycle,
        )
        scores.append(result["overall_score"])

    return np.asarray(scores, dtype=float)


def print_summary(name, scores):
    print(f"\n{name}")

    for index, score in enumerate(scores, start=1):
        print(
            f"  Cycle {index:02d}: "
            f"{score:.4f}"
        )

    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")

    print("\n  Cumulative mean:")
    cumulative = np.cumsum(scores) / np.arange(
        1,
        len(scores) + 1,
    )

    for index, value in enumerate(
        cumulative,
        start=1,
    ):
        print(
            f"    Cycle {index:02d}: "
            f"{value:.4f}"
        )


def main():
    sessions = []

    for name, path in SESSION_FILES:
        cycles = load_cycles(path)

        print(
            f"{name}: "
            f"{len(cycles)} cycles"
        )

        sessions.append(
            (name, cycles)
        )

    # -------------------------------------------------
    # TRAINING: S1 + S2 + S3
    # TEST: S4
    # -------------------------------------------------

    training_groups = [
        sessions[0][1],
        sessions[1][1],
        sessions[2][1],
    ]

    test_cycles = sessions[3][1]

    baseline = build_baseline(
        training_groups
    )

    print(
        "\n========================================"
    )
    print(
        "GaitGuard Candidate A "
        "Temporal Evaluation"
    )
    print(
        "========================================"
    )

    print(
        f"Training cycles: "
        f"{baseline.cycle_count}"
    )

    print(
        f"Unseen test cycles: "
        f"{len(test_cycles)}"
    )

    # -------------------------------------------------
    # Training-session sanity check
    # -------------------------------------------------

    print(
        "\n--- Training Session Statistics ---"
    )

    for name, cycles in sessions[:3]:
        scores = calculate_scores(
            baseline,
            cycles,
        )
        print_summary(
            name,
            scores,
        )

    # -------------------------------------------------
    # Unseen Session 4
    # -------------------------------------------------

    print(
        "\n--- UNSEEN SESSION 4 ---"
    )

    test_scores = calculate_scores(
        baseline,
        test_cycles,
    )

    print_summary(
        "Session 4",
        test_scores,
    )

    # -------------------------------------------------
    # Trend statistics
    # -------------------------------------------------

    if len(test_scores) >= 3:
        x = np.arange(
            len(test_scores),
            dtype=float,
        )

        slope = np.polyfit(
            x,
            test_scores,
            1,
        )[0]

        print(
            "\n--- S4 Trend ---"
        )

        print(
            f"Linear trend slope: "
            f"{slope:.6f}"
        )

        print(
            f"First-cycle deviation: "
            f"{test_scores[0]:.4f}"
        )

        print(
            f"Last-cycle deviation: "
            f"{test_scores[-1]:.4f}"
        )

        print(
            f"Change from first to last: "
            f"{test_scores[-1] - test_scores[0]:.4f}"
        )


if __name__ == "__main__":
    main()