import json
import os
import sys

import numpy as np

from src.baseline.baseline_state import PersonalGaitState


DEFAULT_EPSILON = 1e-3
DEFAULT_MAX_Z = 6.0


def calculate_cycle_deviation(
    baseline,
    cycle,
    epsilon=DEFAULT_EPSILON,
    max_z=DEFAULT_MAX_Z,
):
    """
    Compare one normalized gait cycle against a personal baseline.

    Returns:
        overall_score
        per-feature scores
        phase-wise deviation arrays
    """

    feature_names = baseline.feature_names

    feature_scores = {}
    phase_deviations = {}

    for feature_name in feature_names:
        if feature_name not in cycle:
            continue

        values = np.asarray(
            cycle[feature_name],
            dtype=float,
        )

        mean = baseline.means[feature_name]
        std = baseline.standard_deviation(
            feature_name
        )

        if len(values) != baseline.phase_samples:
            continue

        # Prevent unstable z-scores from tiny variance.
        safe_std = np.maximum(
            std,
            epsilon,
        )

        z = np.abs(
            (values - mean) / safe_std
        )

        # Bound the influence of extreme values.
        z = np.minimum(z, max_z)

        phase_deviations[feature_name] = z.tolist()

        feature_scores[feature_name] = float(
            np.mean(z)
        )

    if not feature_scores:
        raise ValueError(
            "No compatible features were found."
        )

    overall_score = float(
        np.mean(
            list(feature_scores.values())
        )
    )

    return {
        "overall_score": overall_score,
        "feature_scores": feature_scores,
        "phase_deviations": phase_deviations,
    }


def load_baseline(path):
    """Load a saved PersonalGaitState."""
    return PersonalGaitState.load(path)


def load_cycles(path):
    """Load phase-normalized gait cycles."""
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data


def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python -m src.drift.deviation "
            "<baseline_json> <features_json>"
        )
        sys.exit(1)

    baseline_path = sys.argv[1]
    features_path = sys.argv[2]

    if not os.path.isfile(baseline_path):
        print(
            f"ERROR: Baseline not found: "
            f"{baseline_path}"
        )
        sys.exit(1)

    if not os.path.isfile(features_path):
        print(
            f"ERROR: Feature file not found: "
            f"{features_path}"
        )
        sys.exit(1)

    baseline = load_baseline(
        baseline_path
    )

    data = load_cycles(
        features_path
    )

    cycles = data[
        "phase_normalized_cycles"
    ]

    print(
        f"Loaded {len(cycles)} cycles."
    )

    if not cycles:
        print(
            "ERROR: No normalized gait cycles."
        )
        sys.exit(1)

    print(
        "\nCandidate A deviation scores:"
    )

    for index, cycle in enumerate(
        cycles,
        start=1,
    ):
        result = calculate_cycle_deviation(
            baseline,
            cycle,
        )

        print(
            f"Cycle {index}: "
            f"{result['overall_score']:.4f}"
        )

        for (
            feature,
            score,
        ) in result[
            "feature_scores"
        ].items():
            print(
                f"    {feature}: "
                f"{score:.4f}"
            )


if __name__ == "__main__":
    main()
