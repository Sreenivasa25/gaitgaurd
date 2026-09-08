import numpy as np

from src.drift.drift_state import DriftState
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
    analysis = analyze_deviation_sequence(
        scores,
        threshold=THRESHOLD,
        smoothing_window=3,
        trend_window=4,
    )

    machine = DriftState()

    states = []

    for i in range(len(scores)):
        state = machine.update(
            deviation=float(scores[i]),
            slope=float(
                analysis["local_slope"][i]
            ),
            persistence=float(
                analysis["persistence"][i]
            ),
            recovery=float(
                analysis["recovery"][i]
            ),
            threshold=THRESHOLD,
        )

        states.append(state)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        "Cycle | Dev | Smooth | Slope | "
        "Persist | Recovery | State"
    )
    print("-" * 70)

    for i in range(len(scores)):
        print(
            f"{i + 1:5d} | "
            f"{scores[i]:.2f} | "
            f"{analysis['smoothed'][i]:.2f} | "
            f"{analysis['local_slope'][i]:+.3f} | "
            f"{analysis['persistence'][i]:.2f} | "
            f"{analysis['recovery'][i]:+.3f} | "
            f"{states[i]}"
        )

    print("\nTransitions:")

    previous = None

    for i, state in enumerate(states, start=1):
        if state != previous:
            print(
                f"  Cycle {i}: {state}"
            )
            previous = state

    print(
        f"\nFinal state: {machine.state}"
    )


def main():
    print(
        "GaitGuard Candidate B "
        "Synthetic Temporal Test"
    )

    print(
        f"Threshold: {THRESHOLD:.2f}"
    )

    for name, scores in SCENARIOS.items():
        run_scenario(name, scores)


if __name__ == "__main__":
    main()