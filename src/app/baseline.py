"""Baseline v1: Running mean and covariance with exponential decay and gating.

Provides an online per-worker baseline that is updated only when the sample is within a gating
threshold to avoid contaminating the baseline with already-anomalous frames.
"""
from typing import Dict
import numpy as np
import math
from collections import deque

class RunningMeanCov:
    def __init__(self, dim: int, alpha: float = 0.001, gate_threshold: float = 3.0):
        self.dim = dim
        self.alpha = alpha  # decay for exponential moving statistics
        self.gate_threshold = gate_threshold
        self.mean = np.zeros(dim, dtype=np.float64)
        self.S = np.zeros((dim, dim), dtype=np.float64)  # scaled covariance accumulator
        self.initialized = False

    def score(self, x: np.ndarray) -> float:
        """Mahalanobis-like distance. Returns large value if not initialized."""
        if not self.initialized:
            return float('inf')
        delta = x - self.mean
        cov = self.S + np.eye(self.dim) * 1e-6
        try:
            inv = np.linalg.inv(cov)
            d2 = float(np.dot(delta, inv.dot(delta)))
            return math.sqrt(d2)
        except np.linalg.LinAlgError:
            return float('inf')

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        if not self.initialized:
            # initialize with first sample if needed (but we will prefer warm-up initialization)
            self.mean = x.copy()
            self.S = np.eye(self.dim) * 1e-6
            self.initialized = True
            return
        d = x - self.mean
        try:
            score = self.score(x)
        except Exception:
            score = float('inf')
        if score > self.gate_threshold:
            # do not update
            return
        # exponential moving updates
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        outer = np.outer(d, d)
        self.S = (1 - self.alpha) * self.S + self.alpha * outer

    def to_dict(self):
        return {"dim": self.dim, "alpha": float(self.alpha), "gate": float(self.gate_threshold),
                "mean": self.mean.tolist(), "S": self.S.tolist(), "initialized": bool(self.initialized)}

    @classmethod
    def from_dict(cls, data: Dict):
        inst = cls(int(data['dim']), alpha=float(data.get('alpha', 0.001)), gate_threshold=float(data.get('gate', 3.0)))
        inst.mean = np.array(data['mean'], dtype=np.float64)
        inst.S = np.array(data['S'], dtype=np.float64)
        inst.initialized = bool(data.get('initialized', False))
        return inst


class BaselineStore:
    def __init__(self, dim: int, alpha: float = 0.001, gate_threshold: float = 3.0, warmup_n: int = 10):
        self.dim = dim
        self.alpha = alpha
        self.gate_threshold = gate_threshold
        self.store = {}  # worker_id -> RunningMeanCov
        # warm-up buffers: collect N embeddings before initializing
        self.warmup_n = warmup_n
        self.warmups: Dict[int, deque] = {}

    def score(self, worker_id: int, x=None):
        """Return deviation score or None if baseline not yet initialized."""
        if worker_id not in self.store:
            # if warmup buffer exists but not yet full, indicate not ready
            return None
        return self.store[worker_id].score(np.asarray(x, dtype=np.float64) if x is not None else np.zeros(self.dim))

    def update(self, worker_id: int, x):
        x = np.asarray(x, dtype=np.float64)
        if worker_id not in self.store:
            buf = self.warmups.setdefault(worker_id, deque(maxlen=self.warmup_n))
            buf.append(x)
            if len(buf) >= self.warmup_n:
                arr = np.stack(list(buf), axis=0)
                rm = RunningMeanCov(self.dim, alpha=self.alpha, gate_threshold=self.gate_threshold)
                rm.mean = np.mean(arr, axis=0)
                # initialize S as diagonal from sample variance (regularized)
                var = np.var(arr, axis=0) + 1e-6
                rm.S = np.diag(var)
                rm.initialized = True
                self.store[worker_id] = rm
                # clean warmup buffer
                del self.warmups[worker_id]
            return
        # existing baseline update
        self.store[worker_id].update(x)

    def serialize(self):
        return {str(k): v.to_dict() for k, v in self.store.items()}

    def load_from_dict(self, dct):
        for k, v in dct.items():
            self.store[int(k)] = RunningMeanCov.from_dict(v)