#!/usr/bin/env python3
"""
Demo that captures from the MacBook camera, runs MediaPipe Pose, tracks people,
computes embeddings, updates baseline, and prints risk scores.

Run: python src/demos/demo_camera.py
"""
import os
import sys
# make sure `src` (one level up) is on sys.path so `from app...` imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    tracker = Tracker(max_distance=0.15, max_misses=15)  # max_distance will be scaled by frame diag below
    # embedding dim: features.embedding -> mean+std ->  (len(USE_INDICES)*2)*2
    from app.features import USE_INDICES
    emb_dim = len(USE_INDICES) * 4
    baseline = BaselineStore(dim=emb_dim, alpha=0.005, gate_threshold=4.0, warmup_n=10)
    trend = TrendScorer(window_size=240, ewma_alpha=0.2)
    storage = Storage(path=args.sqlite)

    # per-track buffer of recent normalized keypoints to create short-window embeddings
    buffers = defaultdict(lambda: deque(maxlen=5))

    # DB batching buffers
    pending_series = []
    last_flush = time.time()
    flush_interval = 10.0  # seconds
    last_baseline_save = time.time()
    baseline_save_interval = 30.0  # seconds

    try:
        frame_interval = 1.0 / float(args.fps)
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # scale tracker max_distance relative to frame diagonal
            h, w = frame.shape[:2]
            diag = (w*w + h*h) ** 0.5
            # convert tracker.max_distance from fraction to pixels if it is small (<1)
            if tracker.max_distance <= 1.0:
                tracker_px_threshold = tracker.max_distance * diag
                tracker.max_distance = tracker_px_threshold

            detections = model.detect(frame)
            tracked = tracker.update(detections)
            ts = time.time()
            for det in tracked:
                tid = det.get('track_id')
                kps = det.get('keypoints')
                norm = normalize_keypoints(kps)
                buffers[tid].append(norm)
                emb = embedding_from_window(list(buffers[tid]))

                # Check baseline score (None means warm-up/uninitialized)
                score = baseline.score(tid)
                if score is None or score == float('inf'):
                    # baseline not ready: feed the warmup buffer; BaselineStore will initialize when ready
                    baseline.update(tid, emb)
                    status_text = "warmup"
                    display_score = 0.0
                else:
                    # baseline exists: update (gated inside) and then use score
                    baseline.update(tid, emb)
                    score = baseline.score(tid)
                    display_score = float(score) if score != float('inf') else 0.0
                    trend.add_sample(tid, ts, display_score)
                    status_text = f"{display_score:.3f}"
                    pending_series.append((tid, ts, display_score))

                    # buffer series row for batched flush
                    pending_series.append((tid, ts, display_score))

                # annotate frame
                x, y, wbox, hbox = det['bbox']
                cv2.rectangle(frame, (int(x), int(y)), (int(x + wbox), int(y + hbox)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid} Score:{status_text}", (int(x), int(max(y - 10, 10))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('gaitguard-demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # periodically flush pending series and save baselines
            now = time.time()
            if now - last_flush > flush_interval and pending_series:
                for (wid, tstamp, val) in pending_series:
                    storage.append_series(wid, tstamp, val)
                pending_series.clear()
                last_flush = now

            if now - last_baseline_save > baseline_save_interval:
                # save all baselines at once
                serial = baseline.serialize()
                for k, v in serial.items():
                    storage.save_baseline(int(k), v)
                last_baseline_save = now

            # throttle to target fps
            elapsed = time.time() - t0
            to_sleep = frame_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
    finally:
        # flush any pending series and save baselines on shutdown
        for (wid, tstamp, val) in pending_series:
            storage.append_series(wid, tstamp, val)
        serial = baseline.serialize()
        for k, v in serial.items():
            storage.save_baseline(int(k), v)

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