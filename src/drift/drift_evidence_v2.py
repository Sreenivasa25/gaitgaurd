from dataclasses import dataclass


@dataclass
class DriftEvidenceV2:
    """
    Candidate D: bounded, directionally gated drift evidence.

    Evidence accumulates primarily while the current observation
    is above the personalized deviation threshold.

    When the current observation returns below the threshold,
    accumulated evidence decays more aggressively.

    This is an experimental research candidate.
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
        threshold = max(threshold, 1e-6)

        # --------------------------------------------------
        # Current normalized excess above the threshold.
        # --------------------------------------------------

        excess = max(
            0.0,
            (deviation - threshold) / threshold,
        )

        # --------------------------------------------------
        # Positive trend contributes only while the current
        # deviation is actually elevated.
        # --------------------------------------------------

        trend_signal = 0.0

        if deviation > threshold:
            trend_signal = max(0.0, slope)

        # --------------------------------------------------
        # Persistence only contributes when the current
        # observation is elevated.
        # --------------------------------------------------

        persistence_signal = 0.0

        if deviation > threshold:
            persistence_signal = max(
                0.0,
                persistence,
            )

        # --------------------------------------------------
        # CASE 1: Current observation is elevated.
        # Accumulate evidence.
        # --------------------------------------------------

        if deviation > threshold:

            increment = (
                0.9 * excess
                + 1.2 * trend_signal
                + 0.4 * persistence_signal
            )

            # Mild forgetting prevents indefinite accumulation.
            self.evidence *= 0.90

            self.evidence += increment

        # --------------------------------------------------
        # CASE 2: Current observation has returned below
        # the threshold.
        #
        # Recovery should actively remove accumulated
        # evidence rather than merely waiting for decay.
        # --------------------------------------------------

        else:

            recovery_strength = max(
                0.0,
                recovery,
            )

            # Base recovery decay.
            decay = 0.30

            # Stronger recovery accelerates decay.
            decay += min(
                0.40,
                0.60 * recovery_strength,
            )

            # Negative local slope is additional evidence
            # that the deviation is moving downward.
            if slope < 0:
                decay += min(
                    0.20,
                    abs(slope),
                )

            decay = min(
                decay,
                0.85,
            )

            self.evidence *= max(
                0.0,
                1.0 - decay,
            )

        # Keep evidence bounded.
        self.evidence = min(
            max(self.evidence, 0.0),
            5.0,
        )

        return self.evidence

    def reset(self):
        self.evidence = 0.0


class EvidenceStateMachineV2:
    """
    Convert Candidate D evidence into a temporal state.
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
        previous_state = self.state

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
        self.state = self.NORMAL
        self.state_duration = 0