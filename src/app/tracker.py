"""A minimal centroid-based tracker to maintain consistent IDs while a person is visible.

This is intentionally simple and not robust to occlusion; it's suitable for the demo.
"""
from typing import List, Tuple, Dict
import numpy as np


class Track:
    def __init__(self, tid: int, bbox: Tuple[float, float, float, float]):
        self.id = tid
        self.bbox = bbox  # x,y,w,h
        self.misses = 0


class Tracker:
    def __init__(self, max_distance: float = 150.0, max_misses: int = 10):
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.max_distance = max_distance
        self.max_misses = max_misses

    def _centroid(self, bbox):
        x, y, w, h = bbox
        return np.array([x + w / 2.0, y + h / 2.0])

    def update(self, detections: List[Dict]) -> List[Dict]:
        # detections: list with 'bbox' and 'keypoints' etc.
        det_centroids = [self._centroid(d['bbox']) for d in detections]
        assigned = {}
        used_tracks = set()
        results = []

        # match existing tracks greedily by centroid distance
        for i, c in enumerate(det_centroids):
            best_tid = None
            best_dist = None
            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                tc = self._centroid(t.bbox)
                dist = np.linalg.norm(c - tc)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_tid = tid
            if best_dist is not None and best_dist < self.max_distance:
                # assign
                assigned[i] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid].bbox = detections[i]['bbox']
                self.tracks[best_tid].misses = 0
            else:
                # create new track
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, detections[i]['bbox'])
                assigned[i] = tid
        # increment misses and remove old tracks
        for tid, t in list(self.tracks.items()):
            if tid not in used_tracks:
                t.misses += 1
                if t.misses > self.max_misses:
                    del self.tracks[tid]
        # build results
        for i, det in enumerate(detections):
            tid = assigned.get(i)
            out = det.copy()
            out['track_id'] = tid
            results.append(out)
        return results
