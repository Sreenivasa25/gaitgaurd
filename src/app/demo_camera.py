"""Demo that captures from the MacBook camera, runs MediaPipe Pose, tracks people,
computes embeddings, updates baseline, and prints risk scores.

Run: python src/demos/demo_camera.py
"""
import time
import argparse
import cv2
import numpy as np
from collections import defaultdict, deque

from app.pose_backbone import PoseModel
from app.tracker import Tracker
from app.features import normalize_keypoints, embedding_from_window
from app.baseline import BaselineStore
from app.trend import TrendScorer
from app.storage import Storage


def main(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    model = PoseModel(model_complexity=0, min_detection_confidence=0.5)
    tracker = Tracker(max_distance=150.0, max_misses=15)
    # embedding dim: features.embedding -> mean+std ->  (len(USE_INDICES)*2)*2
    # import here to get USE_INDICES
    from app.features import USE_INDICES
    emb_dim = len(USE_INDICES) * 4
    baseline = BaselineStore(dim=emb_dim, alpha=0.005, gate_threshold=4.0)
    trend = TrendScorer(window_size=240, ewma_alpha=0.2)
    storage = Storage(path=args.sqlite)

    # small per-track buffer of recent normalized keypoints to create short-window embeddings
    buffers = defaultdict(lambda: deque(maxlen=5))

    try:
        last_ts = time.time()
        frame_interval = 1.0 / float(args.fps)
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            detections = model.detect(frame)
            tracked = tracker.update(detections)
            ts = time.time()
            for det in tracked:
                tid = det.get('track_id')
                kps = det.get('keypoints')
                norm = normalize_keypoints(kps)
                buffers[tid].append(norm)
                emb = embedding_from_window(list(buffers[tid]))
                score = baseline.score(tid, emb)
                # if baseline uninitialized, initialize with first few embeddings
                if score == float('inf'):
                    baseline.update(tid, emb)
                    score = baseline.score(tid, emb)
                else:
                    # update baseline (gated inside)
                    baseline.update(tid, emb)
                # add to trend
                trend.add_sample(tid, ts, score if score != float('inf') else 0.0)
                r = trend.risk_score(tid)
                # persist baseline periodically
                storage.save_baseline(tid, baseline.serialize().get(str(tid), {}))
                # persist series
                storage.append_series(tid, ts, float(score) if score != float('inf') else 0.0)
                # annotate frame
                x, y, w, h = det['bbox']
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid} Risk:{r:.3f}", (int(x), int(max(y - 10, 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('gaitguard-demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # throttle to target fps
            elapsed = time.time() - t0
            to_sleep = frame_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    finally:
        model.close()
        storage.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=5, help='target processing FPS')
    parser.add_argument('--sqlite', type=str, default='gaitguard.db', help='sqlite path')
    args = parser.parse_args()
    main(args)
