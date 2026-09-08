from dataclasses import dataclass


@dataclass
class DriftEvidence:
    """
    Continuous evidence accumulator for longitudinal gait drift.

    Evidence increases when deviation, trend, and persistence
    indicate sustained movement away from baseline.

    Evidence decreases when recovery is detected.

    This is an experimental research candidate, not a final
    fatigue-risk model.
    """

    evidence: float = 0.0

    def update(
        self,
        deviation: float,
        slope: float,
        persistence: float,
        recovery: float,
        threshold: float,
    ):
        """
        Update accumulated drift evidence.

        Returns:
            current evidence value
        """

        # --------------------------------------------------
        # Normalize the current deviation above threshold.
        # --------------------------------------------------

        excess = max(
            0.0,
            deviation - threshold,
        )

        # Normalize by the threshold so the quantity
        # is approximately scale-independent.
        magnitude_signal = (
            excess / max(threshold, 1e-6)
        )

        # --------------------------------------------------
        # Positive trend contributes evidence.
        # Negative trend does not create evidence.
        # --------------------------------------------------

        trend_signal = max(
            0.0,
            slope,
        )

        # --------------------------------------------------
        # Persistence contributes only when there is
        # evidence of repeated elevation.
        # --------------------------------------------------

        persistence_signal = max(
            0.0,
            persistence,
        )

        # --------------------------------------------------
        # Recovery removes accumulated evidence.
        # --------------------------------------------------

        recovery_signal = max(
            0.0,
            recovery,
        )

        # --------------------------------------------------
        # Candidate C update equation.
        #
        # These weights are experimental.
        # --------------------------------------------------

        increment = (
            1.0 * magnitude_signal
            + 1.5 * trend_signal
            + 0.5 * persistence_signal
            - 1.5 * recovery_signal
        )

        # --------------------------------------------------
        # Evidence decay.
        #
        # A small amount of forgetting prevents old
        # events from dominating forever.
        # --------------------------------------------------

        self.evidence *= 0.90

        self.evidence += increment

        # Evidence cannot become negative.
        self.evidence = max(
            0.0,
            self.evidence,
        )

        return self.evidence

    def reset(self):
        """Reset accumulated evidence."""
        self.evidence = 0.0


class EvidenceStateMachine:
    """
    Convert continuous evidence into interpretable states.

    Thresholds are intentionally configurable.
    """

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    PERSISTENT_DRIFT = "PERSISTENT_DRIFT"
    RECOVERY = "RECOVERY"

    def __init__(
        self,
        watch_threshold=0.75,
        drift_threshold=1.50,
        recovery_threshold=0.25,
    ):
        self.watch_threshold = watch_threshold
        self.drift_threshold = drift_threshold
        self.recovery_threshold = recovery_threshold

        self.state = self.NORMAL
        self.state_duration = 0

    def update(
        self,
        evidence,
        recovery,
    ):
        """Update state using evidence and recovery."""

        previous_state = self.state

        # Strong recovery while evidence remains elevated.
        if (
            recovery > self.recovery_threshold
            and evidence > self.watch_threshold
        ):
            self.state = self.RECOVERY

        elif evidence >= self.drift_threshold:
            self.state = self.PERSISTENT_DRIFT

        elif evidence >= self.watch_threshold:
            self.state = self.WATCH

        else:
            self.state = self.NORMAL

        if self.state == previous_state:
            self.state_duration += 1
        else:
            self.state_duration = 1

        return self.state

    def reset(self):
        """Reset state."""
        self.state = self.NORMAL
        self.state_duration = 0
