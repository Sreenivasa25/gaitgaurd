"""Baseline v1: Running mean and covariance with exponential decay and gating.

Provides an online per-worker baseline that is updated only when the sample is within a gating
threshold to avoid contaminating the baseline with already-anomalous frames.
"""
from typing import Dict
import numpy as np
import math


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
        # regularize S
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
            # initialize with first sample
            self.mean = x.copy()
            self.S = np.eye(self.dim) * 1e-6
            self.initialized = True
            return
        d = x - self.mean
        # gating: update only if within gate
        try:
            score = self.score(x)
        except Exception:
            score = float('inf')
        if score > self.gate_threshold:
            # do not update
            return
        # exponential moving updates
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x
        # update scaled covariance accumulator (approx EMA of outer products)
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
    def __init__(self, dim: int, alpha: float = 0.001, gate_threshold: float = 3.0):
        self.dim = dim
        self.alpha = alpha
        self.gate_threshold = gate_threshold
        self.store = {}  # worker_id -> RunningMeanCov

    def score(self, worker_id: int, x) -> float:
        x = np.asarray(x, dtype=np.float64)
        if worker_id not in self.store:
            # create but not initialized
            self.store[worker_id] = RunningMeanCov(self.dim, self.alpha, self.gate_threshold)
            return float('inf')
        return self.store[worker_id].score(x)

    def update(self, worker_id: int, x):
        x = np.asarray(x, dtype=np.float64)
        if worker_id not in self.store:
            self.store[worker_id] = RunningMeanCov(self.dim, self.alpha, self.gate_threshold)
        self.store[worker_id].update(x)

    def serialize(self):
        return {str(k): v.to_dict() for k, v in self.store.items()}

    def load_from_dict(self, dct):
        for k, v in dct.items():
            self.store[int(k)] = RunningMeanCov.from_dict(v)
