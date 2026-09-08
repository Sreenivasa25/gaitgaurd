import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks, savgol_filter


# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def to_array(landmark):
    """Convert a MediaPipe landmark dictionary to [x, y, z]."""
    return np.array(
        [
            landmark["x"],
            landmark["y"],
            landmark["z"],
        ],
        dtype=float,
    )


def calculate_angle(a, b, c):
    """Calculate angle ABC in degrees."""
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return np.nan

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def extract_landmark_frame(frame):
    """
    Convert one recorded frame into a dictionary keyed by landmark ID.
    """
    return {
        landmark["id"]: {
            "point": to_array(landmark),
            "visibility": float(landmark["visibility"]),
        }
        for landmark in frame["landmarks"]
    }


def calculate_scale(points):
    """
    Estimate body scale using shoulder and hip geometry.

    This makes gait features less dependent on camera distance
    and the worker's absolute body size.
    """
    shoulder_center = (
        points[LEFT_SHOULDER]["point"]
        + points[RIGHT_SHOULDER]["point"]
    ) / 2.0

    hip_center = (
        points[LEFT_HIP]["point"]
        + points[RIGHT_HIP]["point"]
    ) / 2.0

    torso_length = np.linalg.norm(
        shoulder_center - hip_center
    )

    if torso_length < 1e-8:
        return np.nan

    return torso_length


def extract_frame_features(frame, min_visibility=0.5):
    """
    Extract normalized gait features from one frame.

    Coordinates are:
    1. centered at the hip midpoint
    2. normalized by torso length
    """
    points = extract_landmark_frame(frame)

    required_ids = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE,
    ]

    if not all(idx in points for idx in required_ids):
        return None

    required_visibility = [
        points[idx]["visibility"]
        for idx in required_ids
    ]

    if min(required_visibility) < min_visibility:
        return None

    left_hip = points[LEFT_HIP]["point"]
    right_hip = points[RIGHT_HIP]["point"]

    left_knee = points[LEFT_KNEE]["point"]
    right_knee = points[RIGHT_KNEE]["point"]

    left_ankle = points[LEFT_ANKLE]["point"]
    right_ankle = points[RIGHT_ANKLE]["point"]

    hip_center = (left_hip + right_hip) / 2.0

    scale = calculate_scale(points)

    if not np.isfinite(scale):
        return None

    # Normalize all coordinates relative to the worker's hip center.
    left_hip_n = (left_hip - hip_center) / scale
    right_hip_n = (right_hip - hip_center) / scale

    left_knee_n = (left_knee - hip_center) / scale
    right_knee_n = (right_knee - hip_center) / scale

    left_ankle_n = (left_ankle - hip_center) / scale
    right_ankle_n = (right_ankle - hip_center) / scale

    # Joint angles
    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle,
    )

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle,
    )

    # Useful normalized gait descriptors
    ankle_x_difference = (
        left_ankle_n[0] - right_ankle_n[0]
    )

    ankle_y_difference = (
        left_ankle_n[1] - right_ankle_n[1]
    )

    knee_angle_difference = (
        left_knee_angle - right_knee_angle
        if np.isfinite(left_knee_angle)
        and np.isfinite(right_knee_angle)
        else np.nan
    )

    ankle_midpoint = (
        left_ankle_n + right_ankle_n
    ) / 2.0

    return {
        "frame_number": int(frame["frame_number"]),
        "timestamp": float(frame["timestamp"]),

        # Body-normalized positions
        "left_knee_x": float(left_knee_n[0]),
        "left_knee_y": float(left_knee_n[1]),
        "right_knee_x": float(right_knee_n[0]),
        "right_knee_y": float(right_knee_n[1]),

        "left_ankle_x": float(left_ankle_n[0]),
        "left_ankle_y": float(left_ankle_n[1]),
        "right_ankle_x": float(right_ankle_n[0]),
        "right_ankle_y": float(right_ankle_n[1]),

        "ankle_midpoint_x": float(ankle_midpoint[0]),
        "ankle_midpoint_y": float(ankle_midpoint[1]),

        # Joint dynamics
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "knee_angle_difference": knee_angle_difference,

        # Symmetry indicators
        "ankle_x_difference": float(
            ankle_x_difference
        ),
        "ankle_y_difference": float(
            ankle_y_difference
        ),

        # Data quality
        "min_visibility": float(
            min(required_visibility)
        ),
        "body_scale": float(scale),
    }


def interpolate_nan(values):
    """Interpolate missing numeric values."""
    values = np.asarray(values, dtype=float)

    if not np.isnan(values).any():
        return values

    valid = ~np.isnan(values)

    if valid.sum() < 2:
        return None

    indices = np.arange(len(values))

    values[~valid] = np.interp(
        indices[~valid],
        indices[valid],
        values[valid],
    )

    return values


def smooth_signal(values):
    """Apply a light Savitzky-Golay smoothing filter."""
    values = interpolate_nan(values)

    if values is None or len(values) < 7:
        return values

    # Window must be odd and smaller than the sequence.
    window = min(21, len(values))

    if window % 2 == 0:
        window -= 1

    if window < 5:
        return values

    return savgol_filter(
        values,
        window_length=window,
        polyorder=2,
    )


def detect_gait_cycles(features):
    """
    Detect candidate gait cycles using the normalized
    left-right ankle vertical separation signal.

    This is intentionally a prototype segmentation method.
    We will validate and improve it experimentally.
    """
    if len(features) < 20:
        return []

    timestamps = np.array(
        [f["timestamp"] for f in features],
        dtype=float,
    )

    ankle_difference = np.array(
        [
            f["ankle_y_difference"]
            for f in features
        ],
        dtype=float,
    )

    signal = smooth_signal(ankle_difference)

    if signal is None:
        return []

    duration = timestamps[-1] - timestamps[0]

    if duration <= 0:
        return []

    frame_rate = len(features) / duration

    # Don't accept unrealistically close cycle markers.
    min_distance = max(
        5,
        int(frame_rate * 0.35),
    )

    prominence = max(
        np.std(signal) * 0.25,
        0.01,
    )

    peaks, _ = find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence,
    )

    cycles = []

    for start_idx, end_idx in zip(
        peaks[:-1],
        peaks[1:],
    ):
        cycle_duration = (
            timestamps[end_idx]
            - timestamps[start_idx]
        )

        if 0.4 <= cycle_duration <= 3.0:
            cycles.append(
                {
                    "start_index": int(start_idx),
                    "end_index": int(end_idx),
                    "start_timestamp": float(
                        timestamps[start_idx]
                    ),
                    "end_timestamp": float(
                        timestamps[end_idx]
                    ),
                    "duration_seconds": float(
                        cycle_duration
                    ),
                }
            )

    return cycles


def normalize_cycle(
    features,
    start_idx,
    end_idx,
    samples=50,
):
    """
    Resample one gait cycle to a fixed number
    of phase-normalized samples.
    """
    cycle = features[start_idx:end_idx + 1]

    if len(cycle) < 4:
        return None

    source_t = np.linspace(
        0.0,
        1.0,
        len(cycle),
    )

    target_t = np.linspace(
        0.0,
        1.0,
        samples,
    )

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

    normalized = {
        "phase": target_t.tolist(),
    }

    for name in feature_names:
        values = np.array(
            [
                f[name]
                for f in cycle
            ],
            dtype=float,
        )

        values = interpolate_nan(values)

        if values is None:
            return None

        normalized[name] = np.interp(
            target_t,
            source_t,
            values,
        ).tolist()

    normalized["duration_seconds"] = float(
        cycle[-1]["timestamp"]
        - cycle[0]["timestamp"]
    )

    return normalized


def build_feature_dataset(input_path):
    """Load raw pose JSON and produce normalized gait features."""

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    frame_features = []

    for frame in raw["frames"]:
        extracted = extract_frame_features(frame)

        if extracted is not None:
            frame_features.append(extracted)

    cycles = detect_gait_cycles(
        frame_features
    )

    normalized_cycles = []

    for cycle in cycles:
        normalized = normalize_cycle(
            frame_features,
            cycle["start_index"],
            cycle["end_index"],
        )

        if normalized is not None:
            normalized_cycles.append(
                normalized
            )

    return raw, frame_features, cycles, normalized_cycles


def save_dataset(
    raw,
    frame_features,
    cycles,
    normalized_cycles,
    output_path,
):
    """Save the processed gait representation."""
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    result = {
        "project": "GaitGuard",
        "subject_id": raw.get(
            "subject_id",
            "unknown",
        ),
        "experiment_id": raw.get(
            "experiment_id",
            "unknown",
        ),
        "source_frame_count": raw.get(
            "frame_count",
            0,
        ),
        "valid_feature_frames": len(
            frame_features
        ),
        "detected_cycles": len(cycles),
        "normalized_cycles": len(
            normalized_cycles
        ),
        "frame_features": frame_features,
        "cycles": cycles,
        "phase_normalized_cycles": normalized_cycles,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            allow_nan=False,
        )


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python -m src.pose_extraction."
            "gait_features <input_json>"
        )
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.isfile(input_path):
        print(
            f"ERROR: Input file not found: "
            f"{input_path}"
        )
        sys.exit(1)

    raw, frame_features, cycles, normalized = (
        build_feature_dataset(input_path)
    )

    session_name = os.path.splitext(
        os.path.basename(input_path)
    )[0]

    output_path = os.path.join(
        "data",
        "features",
        f"{session_name}_features.json",
    )

    save_dataset(
        raw,
        frame_features,
        cycles,
        normalized,
        output_path,
    )

    print(
        "✓ Gait feature extraction complete"
    )
    print(
        f"✓ Raw pose frames: "
        f"{raw['frame_count']}"
    )
    print(
        f"✓ Valid feature frames: "
        f"{len(frame_features)}"
    )
    print(
        f"✓ Candidate gait cycles: "
        f"{len(cycles)}"
    )
    print(
        f"✓ Phase-normalized cycles: "
        f"{len(normalized)}"
    )
    print(
        f"✓ Output: {output_path}"
    )


if __name__ == "__main__":
    main()
