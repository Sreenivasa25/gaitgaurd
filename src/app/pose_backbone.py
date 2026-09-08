"""MediaPipe Pose wrapper with a consistent detection output.

Returns detections as a list of dicts:
  {"keypoints": np.ndarray of shape (N,3) [x,y,score], "bbox": (x,y,w,h), "score": float}
"""

from typing import List, Dict, Tuple
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None


class PoseModel:
    def __init__(self, model_complexity: int = 0, min_detection_confidence: float = 0.5):
        if mp is None:
            raise RuntimeError("mediapipe is required for PoseModel")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      model_complexity=model_complexity,
                                      min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=0.3)

    def detect(self, frame: np.ndarray) -> List[Dict]:
        # frame: BGR image (as read by OpenCV)
        img_rgb = frame[:, :, ::-1]
        res = self.pose.process(img_rgb)
        detections = []
        if not res.pose_landmarks:
            return detections
        lm = res.pose_landmarks.landmark
        h, w, _ = frame.shape
        kps = np.zeros((len(lm), 3), dtype=np.float32)
        for i, l in enumerate(lm):
            kps[i, 0] = l.x * w
            kps[i, 1] = l.y * h
            kps[i, 2] = l.visibility
        # bounding box from keypoints
        xs = kps[:, 0]
        ys = kps[:, 1]
        minx, maxx = np.min(xs), np.max(xs)
        miny, maxy = np.min(ys), np.max(ys)
        bw = maxx - minx
        bh = maxy - miny
        bbox = (float(minx), float(miny), float(bw), float(bh))
        score = float(np.nanmean(kps[:, 2]))
        detections.append({"keypoints": kps, "bbox": bbox, "score": score})
        return detections

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass
