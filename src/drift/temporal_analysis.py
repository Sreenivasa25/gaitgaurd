import numpy as np


def moving_average(values, window=3):
    """Calculate a causal moving average."""
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    window = max(1, min(window, len(values)))

    result = np.empty_like(values)

    for i in range(len(values)):
        start = max(0, i - window + 1)
        result[i] = np.mean(values[start:i + 1])

    return result


def rolling_slope(values, window=4):
    """
    Calculate the local linear slope over a rolling window.
    """
    values = np.asarray(values, dtype=float)

    slopes = np.zeros(len(values))

    for i in range(len(values)):
        start = max(0, i - window + 1)
        segment = values[start:i + 1]

        if len(segment) < 2:
            slopes[i] = 0.0
            continue

        x = np.arange(len(segment), dtype=float)

        slopes[i] = np.polyfit(
            x,
            segment,
            1,
        )[0]

    return slopes


def persistence_ratio(
    values,
    threshold,
    window=4,
):
    """
    Fraction of recent observations above threshold.
    """
    values = np.asarray(values, dtype=float)

    result = np.zeros(len(values))

    for i in range(len(values)):
        start = max(0, i - window + 1)

        segment = values[start:i + 1]

        result[i] = np.mean(
            segment > threshold
        )

    return result


def recovery_ratio(
    values,
    threshold,
    window=4,
):
    """
    Estimate whether elevated deviation is moving
    back toward the baseline region.

    Positive values indicate recovery.
    """
    values = np.asarray(values, dtype=float)

    result = np.zeros(len(values))

    for i in range(len(values)):
        start = max(0, i - window + 1)

        segment = values[start:i + 1]

        if len(segment) < 2:
            continue

        previous = segment[:-1]
        current = segment[-1]

        previous_excess = np.mean(
            np.maximum(
                previous - threshold,
                0.0,
            )
        )

        current_excess = max(
            current - threshold,
            0.0,
        )

        result[i] = (
            previous_excess
            - current_excess
        )

    return result


def analyze_deviation_sequence(
    scores,
    threshold,
    smoothing_window=3,
    trend_window=4,
):
    """
    Produce diagnostic temporal quantities.

    This is intentionally NOT a fatigue-risk score.
    """
    scores = np.asarray(
        scores,
        dtype=float,
    )

    if len(scores) == 0:
        raise ValueError(
            "Deviation sequence is empty."
        )

    smoothed = moving_average(
        scores,
        smoothing_window,
    )

    slopes = rolling_slope(
        smoothed,
        trend_window,
    )

    persistence = persistence_ratio(
        smoothed,
        threshold,
        trend_window,
    )

    recovery = recovery_ratio(
        smoothed,
        threshold,
        trend_window,
    )

    return {
        "scores": scores,
        "smoothed": smoothed,
        "local_slope": slopes,
        "persistence": persistence,
        "recovery": recovery,
    }