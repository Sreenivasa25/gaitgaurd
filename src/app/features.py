"""Feature extraction utilities: normalize keypoints and produce embeddings.

The embedding here is a flattened vector of selected normalized keypoints.
"""

from typing import List
import numpy as np

# We select a subset of landmarks that are informative for gait/posture.
# MediaPipe Pose has 33 landmarks; we'll use a standard subset.
USE_INDICES = [11, 12, 23, 24, 25, 26, 27, 28]  # shoulders, hips, knees, ankles (approx)


def normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """Normalize keypoints to be root-relative and scale-invariant.

    kps: (N,3) array in pixel coords.
    Returns flattened (2*len(USE_INDICES),) vector (x,y) normalized.
    """
    if kps is None or len(kps) == 0:
        return np.zeros(len(USE_INDICES) * 2, dtype=np.float32)
    pts = kps[:, :2].copy()
    # root = midpoint of hips if available
    try:
        left_hip = kps[23, :2]
        right_hip = kps[24, :2]
        root = (left_hip + right_hip) / 2.0
    except Exception:
        # fallback to mean
        root = np.nanmean(pts, axis=0)
    pts -= root
    # scale by torso length (distance between shoulders and hips)
    try:
        left_sh = kps[11, :2]
        right_sh = kps[12, :2]
        shoulders = (left_sh + right_sh) / 2.0
        torso_len = np.linalg.norm(shoulders - root)
        if torso_len < 1e-3:
            torso_len = 1.0
    except Exception:
        torso_len = np.std(pts) + 1e-3
    pts /= torso_len
    # select subset and flatten
    sel = []
    for idx in USE_INDICES:
        if idx < pts.shape[0]:
            sel.append(pts[idx, 0])
            sel.append(pts[idx, 1])
        else:
            sel.append(0.0)
            sel.append(0.0)
    return np.array(sel, dtype=np.float32)


def embedding_from_window(kps_seq: List[np.ndarray]) -> np.ndarray:
    """Aggregate a short sequence of keypoint arrays into a fixed-dim embedding.

    kps_seq: list of normalized flattened vectors (from normalize_keypoints)
    Returns mean and std concatenated vector.
    """
    if len(kps_seq) == 0:
        return np.zeros(len(USE_INDICES) * 4, dtype=np.float32)
    arr = np.stack(kps_seq, axis=0)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return np.concatenate([mean, std], axis=0)
