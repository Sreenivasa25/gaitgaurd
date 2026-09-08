#!/usr/bin/env python3
"""
Demo that captures from the MacBook camera, runs MediaPipe Pose, tracks people,
computes embeddings, updates baseline, and prints risk scores.

Run: python src/demos/demo_camera.py
"""
import os
import sys
import time
import argparse
import math
import signal
import cv2
import numpy as np
from collections import defaultdict, deque

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pose_backbone import PoseModel
from app.tracker import Tracker
from app.features import normalize_keypoints, embedding_from_window
from app.baseline import BaselineStore
from app.trend import TrendScorer
from app.storage import Storage

# graceful shutdown flag
STOP_REQUESTED = False
def _on_signal(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def main(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    model = PoseModel(model_complexity=0, min_detection_confidence=0.5)
    tracker = Tracker(max_distance=0.15, max_misses=15)  # fraction of diag or pixels if >1.0
    # embedding dim: features.embedding -> mean+std ->  (len(USE_INDICES)*2)*2
    from app.features import USE_INDICES
    emb_dim = len(USE_INDICES) * 4
    # warmup_n=10 collects 10 embeddings before baseline initialization
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
    baseline_save_interval = 60.0  # seconds

    try:
        frame_interval = 1.0 / float(args.fps)
        while True:
            if STOP_REQUESTED:
                print("Shutdown requested, exiting main loop...")
                break

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # scale tracker max_distance relative to frame diagonal (compute pixel threshold per-frame)
            frame_h, frame_w = frame.shape[:2]
            diag = math.hypot(frame_w, frame_h)
            if tracker.max_distance <= 1.0:
                tracker_px_threshold = tracker.max_distance * diag
            else:
                tracker_px_threshold = tracker.max_distance

            # Temporarily set tracker.max_distance to pixel threshold for this update and restore after
            tracker_orig = tracker.max_distance
            tracker.max_distance = tracker_px_threshold
            detections = model.detect(frame)
            tracked = tracker.update(detections)
            tracker.max_distance = tracker_orig  # restore

            ts = time.time()
            for det in tracked:
                tid = det.get('track_id')
                kps = det.get('keypoints')
                norm = normalize_keypoints(kps)
                buffers[tid].append(norm)
                emb = embedding_from_window(list(buffers[tid]))

                # Check baseline score (float('inf') sentinel means not ready)
                score = baseline.score(tid)
                if math.isinf(score):
                    # baseline not ready: feed the warmup buffer; BaselineStore will initialize when ready
                    baseline.update(tid, emb)
                    status_text = "warmup"
                    display_score = 0.0
                else:
                    # baseline exists: update (gated inside) and then use score
                    baseline.update(tid, emb)
                    score = baseline.score(tid)
                    display_score = float(score) if not math.isinf(score) else 0.0
                    trend.add_sample(tid, ts, display_score)
                    status_text = f"{display_score:.3f}"
                    # buffer series row for batched flush (single append)
                    pending_series.append((tid, ts, display_score))

                # annotate frame
                x, y, wbox, hbox = det['bbox']
                cv2.rectangle(frame, (int(x), int(y)), (int(x + wbox), int(y + hbox)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{tid} Score:{status_text}", (int(x), int(max(y - 10, 10))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('gaitguard-demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed.")
                break

            # periodically flush pending series and save baselines
            now = time.time()
            if now - last_flush > flush_interval and pending_series:
                try:
                    storage.append_series_batch(pending_series)
                except AttributeError:
                    # fallback: append one-by-one if batch API not available
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
        if pending_series:
            try:
                storage.append_series_batch(pending_series)
            except AttributeError:
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
