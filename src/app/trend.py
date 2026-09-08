"""Trend scoring over a sliding window per worker.

Keeps a bounded history of deviation scores and computes an EWMA and linear slope to produce a risk score.
"""
from collections import deque
from typing import Dict, Deque, Tuple
import numpy as np
import time


class TrendScorer:
    def __init__(self, window_size: int = 240, ewma_alpha: float = 0.2):
        # window_size = number of samples to keep (e.g., 240 ~ 20 minutes at 5s sampling)
        self.window_size = window_size
        self.ewma_alpha = ewma_alpha
        self.history: Dict[int, Deque[Tuple[float, float]]] = {}  # worker_id -> deque of (t, deviation)

    def add_sample(self, worker_id: int, timestamp: float, deviation: float):
        if worker_id not in self.history:
            self.history[worker_id] = deque(maxlen=self.window_size)
        self.history[worker_id].append((timestamp, float(deviation)))

    def _compute_slope(self, arr):
        # arr: list of (t, val)
        if len(arr) < 3:
            return 0.0
        times = np.array([t for t, v in arr])
        vals = np.array([v for t, v in arr])
        # normalize times to seconds from start
        times = times - times[0]
        # fit linear slope
        try:
            A = np.vstack([times, np.ones_like(times)]).T
            m, c = np.linalg.lstsq(A, vals, rcond=None)[0]
            return float(m)
        except Exception:
            return 0.0

    def _compute_ewma(self, arr):
        if len(arr) == 0:
            return 0.0
        v = arr[0][1]
        for _, val in arr[1:]:
            v = (1 - self.ewma_alpha) * v + self.ewma_alpha * val
        return float(v)

    def risk_score(self, worker_id: int) -> float:
        arr = list(self.history.get(worker_id, []))
        if len(arr) == 0:
            return 0.0
        ewma = self._compute_ewma(arr)
        slope = self._compute_slope(arr)
        # combine: positive slope increases risk; normalize by simple heuristic
        score = max(0.0, ewma) + max(0.0, slope) * 10.0
        # clamp
        return float(min(1.0, score / 10.0))
