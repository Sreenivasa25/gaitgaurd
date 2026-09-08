from dataclasses import dataclass


@dataclass
class DriftStateConfig:
    """
    Configuration for the experimental temporal drift state machine.

    These values are development parameters, not validated
    fatigue thresholds.
    """

    deviation_threshold: float = 1.0

    watch_persistence: float = 0.50
    drift_persistence: float = 0.75

    positive_slope_threshold: float = 0.02
    recovery_slope_threshold: float = -0.02

    recovery_threshold: float = 0.05

    # Number of consecutive observations needed before
    # changing state.
    watch_confirmation: int = 2
    drift_confirmation: int = 2
    recovery_confirmation: int = 2


class DriftStateMachine:
    """
    Experimental temporal state machine for GaitGuard.

    States:
        NORMAL
        WATCH
        PERSISTENT_DRIFT
        RECOVERY

    This module does not produce a fatigue score.
    It classifies the temporal behavior of deviation.
    """

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    PERSISTENT_DRIFT = "PERSISTENT_DRIFT"
    RECOVERY = "RECOVERY"

    def __init__(self, config=None):
        self.config = (
            config
            if config is not None
            else DriftStateConfig()
        )

        self.state = self.NORMAL

        self.watch_count = 0
        self.drift_count = 0
        self.recovery_count = 0

    def reset(self):
        """Reset the state machine."""
        self.state = self.NORMAL

        self.watch_count = 0
        self.drift_count = 0
        self.recovery_count = 0

    def update(
        self,
        deviation,
        slope,
        persistence,
        recovery,
    ):
        """
        Update temporal state from one observation.

        Parameters:
            deviation:
                Current deviation from personal baseline.

            slope:
                Local trend of smoothed deviation.

            persistence:
                Fraction of recent observations above
                the deviation threshold.

            recovery:
                Positive values indicate movement back
                toward the baseline region.

        Returns:
            Current state.
        """

        cfg = self.config

        # -------------------------------------------------
        # Determine local conditions
        # -------------------------------------------------

        elevated = (
            deviation >= cfg.deviation_threshold
        )

        watch_condition = (
            elevated
            and persistence >= cfg.watch_persistence
        )

        drift_condition = (
            elevated
            and persistence >= cfg.drift_persistence
            and slope >= cfg.positive_slope_threshold
        )

        recovery_condition = (
            recovery >= cfg.recovery_threshold
            or slope <= cfg.recovery_slope_threshold
        )

        # -------------------------------------------------
        # NORMAL
        # -------------------------------------------------

        if self.state == self.NORMAL:

            self.recovery_count = 0
            self.drift_count = 0

            if drift_condition:
                self.drift_count += 1
                self.watch_count = 0

                if (
                    self.drift_count
                    >= cfg.drift_confirmation
                ):
                    self.state = (
                        self.PERSISTENT_DRIFT
                    )

            elif watch_condition:
                self.watch_count += 1

                if (
                    self.watch_count
                    >= cfg.watch_confirmation
                ):
                    self.state = self.WATCH

            else:
                self.watch_count = 0

            return self.state

        # -------------------------------------------------
        # WATCH
        # -------------------------------------------------

        if self.state == self.WATCH:

            self.recovery_count = 0

            if drift_condition:
                self.drift_count += 1

                if (
                    self.drift_count
                    >= cfg.drift_confirmation
                ):
                    self.state = (
                        self.PERSISTENT_DRIFT
                    )
                    return self.state

            else:
                self.drift_count = 0

            # If deviation drops sufficiently,
            # return to normal rather than keeping WATCH.
            if (
                not elevated
                and not watch_condition
            ):
                self.watch_count = 0
                self.state = self.NORMAL

            return self.state

        # -------------------------------------------------
        # PERSISTENT DRIFT
        # -------------------------------------------------

        if self.state == self.PERSISTENT_DRIFT:

            self.drift_count = 0

            if recovery_condition:
                self.recovery_count += 1

                if (
                    self.recovery_count
                    >= cfg.recovery_confirmation
                ):
                    self.state = self.RECOVERY
                    self.recovery_count = 0

            else:
                self.recovery_count = 0

            return self.state

        # -------------------------------------------------
        # RECOVERY
        # -------------------------------------------------

        if self.state == self.RECOVERY:

            # Strong continuing recovery eventually returns
            # the system to NORMAL.

            if (
                recovery_condition
                and not elevated
            ):
                self.recovery_count += 1

                if (
                    self.recovery_count
                    >= cfg.recovery_confirmation
                ):
                    self.state = self.NORMAL
                    self.recovery_count = 0

            elif drift_condition:
                # If deviation starts increasing again,
                # go back to persistent drift monitoring.
                self.recovery_count = 0
                self.drift_count += 1

                if (
                    self.drift_count
                    >= cfg.drift_confirmation
                ):
                    self.state = (
                        self.PERSISTENT_DRIFT
                    )
                    self.drift_count = 0

            else:
                self.recovery_count = 0

            return self.state

        # Defensive fallback.
        self.state = self.NORMAL

        return self.state

    def get_state(self):
        """Return the current state."""
        return self.state