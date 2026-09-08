import numpy as np

from src.drift.drift_evidence_v2 import (
    DriftEvidenceV2,
    EvidenceStateMachineV2,
)
from src.drift.temporal_analysis import (
    analyze_deviation_sequence,
)


THRESHOLD = 1.0


SCENARIOS = {
    "A - Isolated spike": np.array(
        [0.60, 0.60, 0.60, 1.70, 0.60, 0.60]
    ),

    "B - Persistent increase": np.array(
        [0.60, 0.70, 0.80, 1.00, 1.20, 1.40, 1.50]
    ),

    "C - Recovery": np.array(
        [1.40, 1.50, 1.50, 1.20, 0.90, 0.70]
    ),

    "D - Noisy normal variation": np.array(
        [0.70, 1.10, 0.60, 1.20, 0.70, 1.00, 0.60]
    ),
}


def run_scenario(name, scores):

    temporal = analyze_deviation_sequence(
        scores,
        threshold=THRESHOLD,
        smoothing_window=3,
        trend_window=4,
    )

    evidence_model = DriftEvidenceV2()
    state_machine = EvidenceStateMachineV2()

    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)

    print(
        "Cycle | Dev | Slope | Persist | Recovery | "
        "Evidence | State"
    )
    print("-" * 78)

    states = []

    for i in range(len(scores)):

        evidence = evidence_model.update(
            deviation=float(scores[i]),
            slope=float(
                temporal["local_slope"][i]
            ),
            persistence=float(
                temporal["persistence"][i]
            ),
            recovery=float(
                temporal["recovery"][i]
            ),
            threshold=THRESHOLD,
        )

        state = state_machine.update(
            evidence=evidence,
            recovery=float(
                temporal["recovery"][i]
            ),
        )

        states.append(state)

        print(
            f"{i + 1:5d} | "
            f"{scores[i]:.2f} | "
            f"{temporal['local_slope'][i]:+.3f} | "
            f"{temporal['persistence'][i]:.2f} | "
            f"{temporal['recovery'][i]:+.3f} | "
            f"{evidence:8.3f} | "
            f"{state}"
        )

    print("\nTransitions:")

    previous = None

    for i, state in enumerate(
        states,
        start=1,
    ):
        if state != previous:
            print(
                f"  Cycle {i}: {state}"
            )
            previous = state

    print(
        f"\nFinal evidence: "
        f"{evidence_model.evidence:.3f}"
    )

    print(
        f"Final state: "
        f"{state_machine.state}"
    )


def main():

    print(
        "GaitGuard Candidate D "
        "Synthetic Test"
    )

    print(
        f"Threshold: {THRESHOLD:.2f}"
    )

    for name, scores in SCENARIOS.items():
        run_scenario(name, scores)


if __name__ == "__main__":
    main()