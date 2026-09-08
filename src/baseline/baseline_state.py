import json
from datetime import datetime, timezone

import numpy as np


class PersonalGaitState:
    """
    Represents a worker-specific gait baseline.

    The state stores phase-conditioned statistics for each
    gait feature. It is intentionally independent of the
    future drift-scoring algorithm.
    """

    def __init__(
        self,
        feature_names=None,
        phase_samples=50,
    ):
        if feature_names is None:
            feature_names = [
                "left_ankle_x",
                "left_ankle_y",
                "right_ankle_x",
                "right_ankle_y",
                "left_knee_angle",
                "right_knee_angle",
                "knee_angle_difference",
                "ankle_x_difference",
                "ankle_y_difference",
                "ankle_midpoint_x",
                "ankle_midpoint_y",
            ]

        self.feature_names = feature_names
        self.phase_samples = phase_samples

        # Number of gait cycles incorporated into the state.
        self.cycle_count = 0

        # Per-feature, per-phase statistics.
        self.means = {
            name: np.zeros(phase_samples, dtype=float)
            for name in self.feature_names
        }

        self.m2 = {
            name: np.zeros(phase_samples, dtype=float)
            for name in self.feature_names
        }

        # Data quality / confidence information.
        self.confidence = {
            name: np.zeros(phase_samples, dtype=float)
            for name in self.feature_names
        }

        self.created_at = datetime.now(
            timezone.utc
        ).isoformat()

        self.updated_at = self.created_at

    def update(self, cycle):
        """
        Add one normalized gait cycle to the baseline.

        Uses an online mean/variance update so that the full
        historical dataset does not need to remain in memory.
        """

        if "phase" not in cycle:
            raise ValueError(
                "Cycle must contain phase information."
            )

        for name in self.feature_names:
            if name not in cycle:
                continue

            values = np.asarray(
                cycle[name],
                dtype=float,
            )

            if len(values) != self.phase_samples:
                raise ValueError(
                    f"Feature '{name}' must contain "
                    f"{self.phase_samples} phase samples."
                )

            if not np.all(np.isfinite(values)):
                continue

            if self.cycle_count == 0:
                self.means[name] = values.copy()
                self.m2[name] = np.zeros(
                    self.phase_samples,
                    dtype=float,
                )
                self.confidence[name] = np.ones(
                    self.phase_samples,
                    dtype=float,
                )
            else:
                delta = (
                    values
                    - self.means[name]
                )

                new_count = self.cycle_count + 1

                self.means[name] += (
                    delta / new_count
                )

                delta2 = (
                    values
                    - self.means[name]
                )

                self.m2[name] += (
                    delta * delta2
                )

                # More observations increase confidence,
                # with diminishing returns.
                self.confidence[name] = (
                    1.0
                    - 1.0 / np.sqrt(new_count)
                )

        self.cycle_count += 1

        self.updated_at = datetime.now(
            timezone.utc
        ).isoformat()

    def variance(self, feature_name):
        """Return the online population variance."""

        if feature_name not in self.means:
            raise KeyError(
                f"Unknown feature: {feature_name}"
            )

        if self.cycle_count <= 1:
            return np.zeros(
                self.phase_samples,
                dtype=float,
            )

        return (
            self.m2[feature_name]
            / self.cycle_count
        )

    def standard_deviation(self, feature_name):
        """Return the standard deviation."""

        return np.sqrt(
            np.maximum(
                self.variance(feature_name),
                0.0,
            )
        )

    def get_feature_statistics(
        self,
        feature_name,
    ):
        """Return mean/std/confidence for a feature."""

        if feature_name not in self.means:
            raise KeyError(
                f"Unknown feature: {feature_name}"
            )

        return {
            "mean": self.means[
                feature_name
            ].copy(),
            "std": self.standard_deviation(
                feature_name
            ),
            "confidence": self.confidence[
                feature_name
            ].copy(),
        }

    def summary(self):
        """Return a compact summary of the baseline."""

        return {
            "cycle_count": self.cycle_count,
            "feature_count": len(
                self.feature_names
            ),
            "phase_samples": self.phase_samples,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def save(self, output_path):
        """Save the baseline state as JSON."""

        output = {
            "project": "GaitGuard",
            "model": "PersonalGaitState",
            "summary": self.summary(),
            "feature_names": self.feature_names,
            "means": {
                name: values.tolist()
                for name, values in self.means.items()
            },
            "variances": {
                name: self.variance(name).tolist()
                for name in self.feature_names
            },
            "confidence": {
                name: values.tolist()
                for name, values in self.confidence.items()
            },
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=2,
                allow_nan=False,
            )

    @classmethod
    def load(cls, input_path):
        """Load a previously saved baseline."""

        with open(
            input_path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        state = cls(
            feature_names=data[
                "feature_names"
            ],
            phase_samples=data[
                "summary"
            ]["phase_samples"],
        )

        state.cycle_count = data[
            "summary"
        ]["cycle_count"]

        for name in state.feature_names:
            state.means[name] = np.asarray(
                data["means"][name],
                dtype=float,
            )

            variance_values = np.asarray(
                data["variances"][name],
                dtype=float,
            )

            # Reconstruct M2 from population variance.
            state.m2[name] = (
                variance_values
                * state.cycle_count
            )

            state.confidence[name] = (
                np.asarray(
                    data["confidence"][name],
                    dtype=float,
                )
            )

        state.created_at = data[
            "summary"
        ]["created_at"]

        state.updated_at = data[
            "summary"
        ]["updated_at"]

        return state