import json

import numpy as np

from src.baseline.baseline_state import PersonalGaitState
from src.drift.deviation import calculate_cycle_deviation
from src.drift.drift_evidence import (
    DriftEvidence,
    EvidenceStateMachine,
)
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


def calculate_scores(
    baseline,
    cycles,
):
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

    evidence = DriftEvidence()
    states = EvidenceStateMachine()

    print("========================================")
    print("GaitGuard Candidate C - Real S4 Test")
    print("========================================")

    print(
        f"Training cycles: {baseline.cycle_count}"
    )

    print(
        f"S4 cycles: {len(test_scores)}"
    )

    print(
        f"Training 95th percentile: "
        f"{threshold:.4f}"
    )

    print(
        "\nCycle | Dev | Trend | Persist | "
        "Recovery | Evidence | State"
    )
    print("-" * 78)

    evidence_values = []

    for i in range(len(test_scores)):

        current_evidence = evidence.update(
            deviation=float(test_scores[i]),
            slope=float(
                temporal["local_slope"][i]
            ),
            persistence=float(
                temporal["persistence"][i]
            ),
            recovery=float(
                temporal["recovery"][i]
            ),
            threshold=threshold,
        )

        state = states.update(
            evidence=current_evidence,
            recovery=float(
                temporal["recovery"][i]
            ),
        )

        evidence_values.append(
            current_evidence
        )

        print(
            f"{i + 1:5d} | "
            f"{test_scores[i]:.3f} | "
            f"{temporal['local_slope'][i]:+.3f} | "
            f"{temporal['persistence'][i]:.2f} | "
            f"{temporal['recovery'][i]:+.3f} | "
            f"{current_evidence:8.3f} | "
            f"{state}"
        )

    print("\nSummary")
    print(
        f"Maximum evidence: "
        f"{max(evidence_values):.4f}"
    )

    print(
        f"Final evidence: "
        f"{evidence_values[-1]:.4f}"
    )

    print(
        f"Maximum deviation: "
        f"{max(test_scores):.4f}"
    )

    print(
        f"Final deviation: "
        f"{test_scores[-1]:.4f}"
    )

    print(
        f"Final state: "
        f"{states.state}"
    )


if __name__ == "__main__":
    main()